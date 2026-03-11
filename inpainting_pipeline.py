"""
RePaint-Style Inpainting Pipeline — Stable Diffusion 2.

Mask convention:
    PIL Image : WHITE = inpaint, BLACK = keep
    Numpy arr : 1 = inpaint, 0 = keep
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from pathlib import Path


class AttentionStore:
    """
    Stores self-attention keys and values from a reference UNet forward pass,
    then injects them into the masked region during the inpainting forward pass.

    Only self-attention layers are targeted (cross-attention uses text embeddings
    as K/V so injection there would corrupt the prompt conditioning).

    Usage:
        store = AttentionStore(unet)
        store.enable_reference_mode()   # next forward pass stores K/V
        unet(reference_latent, ...)
        store.enable_injection_mode(mask_latent)  # next forward pass injects
        unet(x_t, ...)
        store.disable()
    """

    def __init__(self, unet: UNet2DConditionModel):
        self.unet   = unet
        self._hooks = []
        self._store: dict[str, dict] = {}
        self._mode  = "off"
        self._mask_latent: torch.Tensor = None

    def _is_self_attn(self, module) -> bool:
        return (
                hasattr(module, "to_q") and
                hasattr(module, "to_k") and
                hasattr(module, "to_v") and
                not hasattr(module, "add_k_proj")
        )

    def _make_hook(self, name: str):
        def hook(module, args, kwargs, output):
            if self._mode == "off":
                return output

            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return output

            B, seq, dim = hidden.shape

            q = module.to_q(hidden)
            k = module.to_k(hidden)
            v = module.to_v(hidden)

            heads     = module.heads
            head_dim  = dim // heads

            def split_heads(x):
                return x.reshape(B, seq, heads, head_dim).permute(0, 2, 1, 3)

            if self._mode == "store":
                self._store[name] = {
                    "k": split_heads(k).detach(),
                    "v": split_heads(v).detach(),
                }
                return output

            if self._mode == "inject" and name in self._store:
                ref_k = self._store[name]["k"]
                ref_v = self._store[name]["v"]

                if ref_k.shape[0] != B:
                    ref_k = ref_k.expand(B, -1, -1, -1)
                    ref_v = ref_v.expand(B, -1, -1, -1)

                q_heads = split_heads(q)
                k_heads = split_heads(k)
                v_heads = split_heads(v)

                mask_down = self._get_spatial_mask(seq, hidden.device, hidden.dtype)
                mask_flat = mask_down.reshape(1, 1, seq, 1)

                k_blended = k_heads * (1 - mask_flat) + ref_k * mask_flat
                v_blended = v_heads * (1 - mask_flat) + ref_v * mask_flat

                k_combined = torch.cat([k_heads, k_blended], dim=2)
                v_combined = torch.cat([v_heads, v_blended], dim=2)

                scale  = head_dim ** -0.5
                scores = torch.einsum("bhqd,bhkd->bhqk", q_heads * scale, k_combined)
                attn   = scores.softmax(dim=-1)
                out    = torch.einsum("bhqk,bhkd->bhqd", attn, v_combined)

                out = out.permute(0, 2, 1, 3).reshape(B, seq, dim)
                out = module.to_out[0](out)
                out = module.to_out[1](out)
                return out

            return output

        return hook

    def _get_spatial_mask(self, seq_len: int, device, dtype) -> torch.Tensor:
        if self._mask_latent is None:
            return torch.ones(seq_len, device=device, dtype=dtype)
        H = W = int(seq_len ** 0.5)
        m = F.interpolate(
            self._mask_latent.float(),
            size=(H, W),
            mode="nearest"
        ).squeeze().reshape(-1).to(device=device, dtype=dtype)
        return m

    def enable_reference_mode(self):
        self._store.clear()
        self._mode = "store"
        self._register_hooks()

    def enable_injection_mode(self, mask_latent: torch.Tensor):
        self._mask_latent = mask_latent
        self._mode = "inject"
        self._register_hooks()

    def disable(self):
        self._mode = "off"
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _register_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for name, module in self.unet.named_modules():
            if self._is_self_attn(module):
                h = module.register_forward_hook(
                    self._make_hook(name), with_kwargs=True
                )
                self._hooks.append(h)


def load_pipeline_components(
        model_id: str = "sd2-community/stable-diffusion-2-base",
        device: str = "cuda"
):
    print(f"Loading model components from '{model_id}' ...")
    # Use fp16 on CUDA for ~4x speedup with minimal quality loss
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
    vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype).to(device)
    unet         = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype).to(device)
    scheduler    = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    for model in [text_encoder, vae, unet]:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    print("All components loaded successfully.")
    return tokenizer, text_encoder, vae, unet, scheduler


def preprocess_image(image: Image.Image, resolution: int = 512) -> torch.Tensor:
    """Returns (1, 3, H, W) tensor in [-1, 1]."""
    image = image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    arr   = np.array(image).astype(np.float32) / 255.0
    arr   = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def preprocess_mask(mask, resolution: int = 512) -> torch.Tensor:
    """
    Returns (1, 1, H//8, W//8) binary tensor: 1=inpaint, 0=keep.
    Downscaled to latent resolution (VAE compresses by 8x).
    """
    latent_res = resolution // 8

    if isinstance(mask, np.ndarray):
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        pil_mask = pil_mask.resize((latent_res, latent_res), Image.NEAREST)
        arr = np.array(pil_mask).astype(np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32)
    else:
        # PIL: white(255) -> 1=inpaint, black(0) -> 0=keep
        pil_mask = mask.convert("L").resize((latent_res, latent_res), Image.NEAREST)
        arr = np.array(pil_mask).astype(np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32) # CHANGED: < 0.5 to > 0.5

    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def preprocess_mask_pixel(mask, resolution: int = 512) -> torch.Tensor:
    """
    Same as preprocess_mask but at full pixel resolution (1, 1, H, W).
    Used for postprocessing in pixel space.
    """
    if isinstance(mask, np.ndarray):
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        pil_mask = pil_mask.resize((resolution, resolution), Image.NEAREST)
        arr = np.array(pil_mask).astype(np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32)
    else:
        pil_mask = mask.convert("L").resize((resolution, resolution), Image.NEAREST)
        arr = np.array(pil_mask).astype(np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32) # CHANGED: < 0.5 to > 0.5

    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def encode_image_to_latent(image_tensor: torch.Tensor, vae: AutoencoderKL, device: str) -> torch.Tensor:
    """Encodes pixel image to VAE latent space. Returns (1, 4, H//8, W//8)."""
    # Cast to same dtype as VAE (fp16 on CUDA)
    image_tensor = image_tensor.to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    return latent


def decode_latent_to_image(latent: torch.Tensor, vae: AutoencoderKL, device: str) -> Image.Image:
    """Decodes VAE latent back to a PIL image."""
    latent = latent.to(device=device, dtype=vae.dtype)  # ensure fp16
    with torch.no_grad():
        image_tensor = vae.decode(latent / vae.config.scaling_factor).sample
    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
    image_tensor = (image_tensor.clamp(-1, 1) + 1.0) / 2.0
    return Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))


def encode_text_prompt(prompt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: str) -> torch.Tensor:
    """
    Returns stacked CLIP embeddings (2, 77, 1024):
        [0] = conditional (real prompt), [1] = unconditional (empty string)
    Both are needed for Classifier-Free Guidance (CFG).
    """
    tokens = tokenizer(
        [prompt, ""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids.to(device))[0]
    return embeddings


def postprocess(result_image: Image.Image, original_image: Image.Image, mask, resolution: int = 512) -> Image.Image:
    """
    Hard pixel swap: for every pixel where mask is NOT white, replace the
    result pixel with the exact original pixel, regardless of result value.
    White mask pixels (inpaint region) are left untouched.
    """
    # Resize both images to pipeline resolution
    orig_np   = np.array(original_image.convert("RGB").resize((resolution, resolution), Image.LANCZOS))
    result_np = np.array(result_image)

    # Build binary bitmap from mask
    if isinstance(mask, np.ndarray):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    else:
        mask_pil = mask.convert("RGB")
    mask_np = np.array(mask_pil.resize((resolution, resolution), Image.NEAREST))

    # CHANGED: A pixel is "white" (inpaint) if all channels are above threshold 250
    is_white = np.all(mask_np > 250, axis=2)  # (H, W) bool, True=inpaint

    # Hard swap: wherever is_white is False (keep region), use original pixel exactly
    output_np = result_np.copy()
    output_np[~is_white] = orig_np[~is_white]

    return Image.fromarray(output_np)


def repaint_inpainting(
        image: Image.Image,
        mask,
        prompt: str,
        tokenizer,
        text_encoder,
        vae,
        unet,
        scheduler,
        num_inference_steps: int  = 50,
        guidance_scale: float     = 7.5,
        seed: int                 = 42,
        resolution: int           = 512,
        device: str               = "cuda",
        inject_attention: bool    = True,
        injection_end: float      = 0.7
) -> Image.Image:
    """
    Args:
        image               : Original PIL image
        mask                : PIL image (white=inpaint) or numpy array (1=inpaint)
        prompt              : Text describing what to generate in the masked region
        tokenizer/text_encoder/vae/unet/scheduler: pre-loaded model components
        num_inference_steps : Reverse diffusion steps (more = better quality, slower)
        guidance_scale      : CFG strength — higher follows prompt more strictly
        seed                : Random seed for reproducibility
        inject_attention    : Whether to use self-attention feature injection
        injection_end       : Fraction of timesteps after which injection stops.
                              Early/mid steps benefit most; late steps can be left free
                              so fine details aren't over-constrained by the reference.
    Returns:
        Inpainted PIL Image with postprocessing applied
    """
    scheduler.set_timesteps(num_inference_steps)
    generator = torch.Generator(device=device).manual_seed(seed)

    image_tensor    = preprocess_image(image, resolution)
    mask_latent     = preprocess_mask(mask, resolution).to(device)
    original_latent = encode_image_to_latent(image_tensor, vae, device)
    text_embeddings = encode_text_prompt(prompt, tokenizer, text_encoder, device)

    x_t = torch.randn(original_latent.shape, generator=generator, device=device, dtype=unet.dtype)
    x_t = x_t * scheduler.init_noise_sigma

    attn_store = AttentionStore(unet) if inject_attention else None
    total_steps = len(scheduler.timesteps)

    for step_idx, t in enumerate(tqdm(scheduler.timesteps, desc="Inpainting")):

        progress = step_idx / total_steps
        should_inject = inject_attention and progress < injection_end

        if should_inject:
            t_int = t.item() if hasattr(t, "item") else int(t)
            alpha_bar = scheduler.alphas_cumprod[t_int]
            noise = torch.randn_like(original_latent, dtype=unet.dtype)
            original_at_t = (alpha_bar ** 0.5) * original_latent + ((1 - alpha_bar) ** 0.5) * noise

            attn_store.enable_reference_mode()
            with torch.no_grad():
                unet(
                    scheduler.scale_model_input(original_at_t, t).to(dtype=unet.dtype),
                    t,
                    encoder_hidden_states=text_embeddings[0:1]
                )
            attn_store.disable()

            attn_store.enable_injection_mode(mask_latent)

        unet_input = scheduler.scale_model_input(torch.cat([x_t, x_t], dim=0), t)
        unet_input = unet_input.to(dtype=unet.dtype)
        with torch.no_grad():
            noise_pred = unet(unet_input, t, encoder_hidden_states=text_embeddings).sample

        if should_inject:
            attn_store.disable()

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        x_t_minus_1 = scheduler.step(noise_pred, t, x_t).prev_sample

        current_idx = (scheduler.timesteps == t).nonzero().item()
        t_prev = scheduler.timesteps[current_idx + 1].item() if current_idx + 1 < len(scheduler.timesteps) else 0
        alpha_bar_prev = scheduler.alphas_cumprod[t_prev]
        noise = torch.randn_like(original_latent, dtype=unet.dtype)

        if t_prev > 0:
            original_at_t_minus_1 = (alpha_bar_prev ** 0.5) * original_latent + ((1 - alpha_bar_prev) ** 0.5) * noise
        else:
            original_at_t_minus_1 = original_latent

        x_t = (mask_latent * x_t_minus_1) + ((1 - mask_latent) * original_at_t_minus_1)

    result_image = decode_latent_to_image(x_t, vae, device)
    result_image = postprocess(result_image, image, mask, resolution)

    return result_image


def load_triplets(images_dir: str, masks_dir: str, prompts_dir: str):
    """
    Loads matching (image, mask, prompt) triplets from three directories.
    Files are matched by filename stem — e.g. desk.jpg / desk.png / desk.txt.
    Skips stems that don't have all three files.
    """
    images_dir  = Path(images_dir)
    masks_dir   = Path(masks_dir)
    prompts_dir = Path(prompts_dir)

    image_stems  = {f.stem: f for ext in [".jpg", ".jpeg", ".png"] for f in images_dir.glob(f"*{ext}")}
    mask_stems   = {f.stem: f for ext in [".jpg", ".jpeg", ".png"] for f in masks_dir.glob(f"*{ext}")}
    prompt_stems = {f.stem: f for f in prompts_dir.glob("*.txt")}

    valid_stems = set(image_stems) & set(mask_stems) & set(prompt_stems)
    skipped     = (set(image_stems) | set(mask_stems) | set(prompt_stems)) - valid_stems

    if skipped:
        print(f"Warning: skipping {len(skipped)} incomplete triplet(s): {sorted(skipped)}")
    if not valid_stems:
        raise ValueError(
            f"No complete triplets found!\n"
            f"  Images: {len(image_stems)}, Masks: {len(mask_stems)}, Prompts: {len(prompt_stems)}\n"
            f"Make sure image, mask and prompt share the same filename stem."
        )

    triplets = []
    for stem in sorted(valid_stems):
        image  = Image.open(image_stems[stem]).convert("RGB")
        mask   = Image.open(mask_stems[stem])
        prompt = prompt_stems[stem].read_text(encoding="utf-8").strip()
        triplets.append({"name": stem, "image": image, "mask": mask, "prompt": prompt})
        print(f"  Loaded '{stem}': prompt = \"{prompt}\"")

    print(f"\nFound {len(triplets)} valid triplet(s).\n")
    return triplets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RePaint inpainting with SD2-base — batch directory mode")
    parser.add_argument("--images",  type=str, required=True,     help="Directory of input images (.jpg/.png)")
    parser.add_argument("--masks",   type=str, required=True,     help="Directory of mask images (white=inpaint, black=keep)")
    parser.add_argument("--prompts", type=str, required=True,     help="Directory of prompt .txt files")
    parser.add_argument("--output",  type=str, default="results", help="Directory to save results")
    parser.add_argument("--steps",            type=int,   default=50,   help="Diffusion steps (default: 50)")
    parser.add_argument("--guidance",          type=float, default=7.5,  help="CFG guidance scale (default: 7.5)")
    parser.add_argument("--seed",              type=int,   default=42,   help="Base random seed (default: 42)")
    parser.add_argument("--device",            type=str,   default="cuda", help="cuda or cpu (default: cuda)")
    parser.add_argument("--no-inject",         action="store_true",      help="Disable self-attention injection")
    parser.add_argument("--injection-end",     type=float, default=0.7,  help="Fraction of steps to inject attention for (default: 0.7)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning directories for matching triplets...")
    triplets = load_triplets(args.images, args.masks, args.prompts)

    print("Loading model (this happens once for all images)...")
    tokenizer, text_encoder, vae, unet, scheduler = load_pipeline_components(device=args.device)

    for i, triplet in enumerate(triplets):
        name, image, mask, prompt = triplet["name"], triplet["image"], triplet["mask"], triplet["prompt"]
        print(f"\n[{i+1}/{len(triplets)}] Processing '{name}' ...")
        print(f"  Prompt: \"{prompt}\"")

        per_image_seed = args.seed + i
        result = repaint_inpainting(
            image               = image,
            mask                = mask,
            prompt              = prompt,
            tokenizer           = tokenizer,
            text_encoder        = text_encoder,
            vae                 = vae,
            unet                = unet,
            scheduler           = scheduler,
            num_inference_steps = args.steps,
            guidance_scale      = args.guidance,
            seed                = per_image_seed,
            device              = args.device,
            inject_attention    = not args.no_inject,
            injection_end       = args.injection_end
        )
        print(f"  Seed: {per_image_seed}")

        out_path = output_dir / f"{name}_result.png"
        result.save(out_path)
        print(f"  Saved -> {out_path}")

    print(f"\nDone! All results saved to: {output_dir}/")