"""
RePaint-Style Inpainting Pipeline — Stable Diffusion 2.

Mask convention:
    PIL Image : WHITE = inpaint, BLACK = keep
    Numpy arr : 1 = inpaint, 0 = keep
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from pathlib import Path


class AttentionStore:

    def __init__(self, unet: UNet2DConditionModel):
        self.unet          = unet
        self._mode         = False
        self._strength     = 1.0
        self._mask_latent  = None
        self._mask_cache   = {}
        self._hooks        = []
        self._register_hooks()

    def _is_self_attn(self, module) -> bool:
        return (
                hasattr(module, "to_q") and
                hasattr(module, "to_k") and
                hasattr(module, "to_v") and
                module.to_q.in_features == module.to_k.in_features
        )

    def _make_hook(self, heads: int, head_dim: int):
        def hook(module, args, kwargs, output):
            if not self._mode:
                return output
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is None:
                return output
            B, seq, dim = hidden.shape

            def split(x):
                return x.reshape(B, seq, heads, head_dim).permute(0, 2, 1, 3)

            q = split(module.to_q(hidden))
            k = split(module.to_k(hidden))
            v = split(module.to_v(hidden))

            ref_k = k[2:3]
            ref_v = v[2:3]

            mask_flat = self._get_spatial_mask(seq, hidden.device, hidden.dtype)
            w = self._strength * mask_flat.reshape(1, 1, seq, 1)

            q_inpaint = q[:2]
            k_inpaint = k[:2]
            v_inpaint = v[:2]

            ref_k_exp = ref_k.expand(2, -1, -1, -1)
            ref_v_exp = ref_v.expand(2, -1, -1, -1)

            k_blended = k_inpaint * (1 - w) + ref_k_exp * w
            v_blended = v_inpaint * (1 - w) + ref_v_exp * w

            k_combined = torch.cat([k_inpaint, k_blended], dim=2)
            v_combined = torch.cat([v_inpaint, v_blended], dim=2)

            out_inpaint = F.scaled_dot_product_attention(q_inpaint, k_combined, v_combined)
            out_ref     = F.scaled_dot_product_attention(q[2:3], k[2:3], v[2:3])

            out = torch.cat([out_inpaint, out_ref], dim=0)
            out = out.permute(0, 2, 1, 3).reshape(B, seq, dim)
            out = module.to_out[0](out)
            out = module.to_out[1](out)
            return out
        return hook

    def _get_spatial_mask(self, seq_len: int, device, dtype) -> torch.Tensor:
        if seq_len in self._mask_cache:
            return self._mask_cache[seq_len].to(device=device, dtype=dtype)
        if self._mask_latent is None:
            m = torch.ones(seq_len, device=device, dtype=dtype)
        else:
            H = W = int(seq_len ** 0.5)
            m = F.interpolate(
                self._mask_latent.float(), size=(H, W), mode="nearest"
            ).squeeze().reshape(-1).to(device=device, dtype=dtype)
        self._mask_cache[seq_len] = m
        return m

    def enable(self, mask_latent: torch.Tensor, strength: float = 1.0):
        if mask_latent is not self._mask_latent:
            self._mask_latent = mask_latent
            self._mask_cache.clear()
        self._strength = strength
        self._mode     = True

    def disable(self):
        self._mode = False

    def _register_hooks(self):
        for name, module in self.unet.named_modules():
            if self._is_self_attn(module):
                heads    = module.heads
                head_dim = module.to_q.out_features // heads
                h = module.register_forward_hook(
                    self._make_hook(heads, head_dim), with_kwargs=True
                )
                self._hooks.append(h)


def compute_injection_strength(
        step_idx: int,
        total_steps: int,
        injection_end: float,
        schedule: str,
        exp_decay: float = 5.0
) -> float:
    progress = step_idx / max(total_steps - 1, 1)
    if progress >= injection_end:
        return 0.0
    p = progress / injection_end
    if schedule == "linear":
        return 1.0 - p
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * p))
    if schedule == "exp":
        return math.exp(-exp_decay * p)
    return 1.0


def load_pipeline_components(
        model_id: str = "sd2-community/stable-diffusion-2-base",
        device: str = "cuda"
):
    print(f"Loading model components from '{model_id}' ...")
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
    image = image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    arr   = np.array(image).astype(np.float32) / 255.0
    arr   = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def preprocess_mask(mask, resolution: int = 512) -> torch.Tensor:
    latent_res = resolution // 8
    if isinstance(mask, np.ndarray):
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        pil_mask = pil_mask.resize((latent_res, latent_res), Image.NEAREST)
        arr = np.array(pil_mask).astype(np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32)
    else:
        pil_mask = mask.convert("L").resize((latent_res, latent_res), Image.NEAREST)
        arr = np.array(pil_mask).astype(np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def encode_image_to_latent(image_tensor: torch.Tensor, vae: AutoencoderKL, device: str) -> torch.Tensor:
    image_tensor = image_tensor.to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    return latent


def decode_latent_to_image(latent: torch.Tensor, vae: AutoencoderKL, device: str) -> Image.Image:
    latent = latent.to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        image_tensor = vae.decode(latent / vae.config.scaling_factor).sample
    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)
    image_tensor = (image_tensor.clamp(-1, 1) + 1.0) / 2.0
    return Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))


def encode_text_prompt(prompt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: str) -> torch.Tensor:
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
    orig_np   = np.array(original_image.convert("RGB").resize((resolution, resolution), Image.LANCZOS))
    result_np = np.array(result_image)
    if isinstance(mask, np.ndarray):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    else:
        mask_pil = mask.convert("RGB")
    mask_np  = np.array(mask_pil.resize((resolution, resolution), Image.NEAREST))
    is_white = np.all(mask_np > 250, axis=2)
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
        injection_end: float      = 0.7,
        injection_schedule: str   = "step",
        exp_decay: float          = 5.0
) -> Image.Image:
    scheduler.set_timesteps(num_inference_steps)
    generator = torch.Generator(device=device).manual_seed(seed)

    image_tensor    = preprocess_image(image, resolution)
    mask_latent     = preprocess_mask(mask, resolution).to(device)
    original_latent = encode_image_to_latent(image_tensor, vae, device)
    text_embeddings = encode_text_prompt(prompt, tokenizer, text_encoder, device)

    x_t = torch.randn(original_latent.shape, generator=generator, device=device, dtype=unet.dtype)
    x_t = x_t * scheduler.init_noise_sigma

    attn_store     = AttentionStore(unet) if inject_attention else None
    total_steps    = len(scheduler.timesteps)
    timesteps_list = list(scheduler.timesteps)

    for step_idx, t in enumerate(tqdm(scheduler.timesteps, desc="Inpainting")):

        strength = 0.0
        if inject_attention:
            strength = compute_injection_strength(
                step_idx, total_steps, injection_end, injection_schedule, exp_decay
            )

        should_inject = strength > 0.0

        if should_inject:
            t_int     = t.item() if hasattr(t, "item") else int(t)
            alpha_bar = scheduler.alphas_cumprod[t_int]
            noise     = torch.randn_like(original_latent, dtype=unet.dtype)
            original_at_t = (alpha_bar ** 0.5) * original_latent + ((1 - alpha_bar) ** 0.5) * noise

            attn_store.enable(mask_latent, strength=strength)
            unet_input = scheduler.scale_model_input(
                torch.cat([x_t, x_t, original_at_t], dim=0), t
            ).to(dtype=unet.dtype)
            embeddings = torch.cat([text_embeddings, text_embeddings[0:1]], dim=0)
        else:
            if inject_attention:
                attn_store.disable()
            unet_input = scheduler.scale_model_input(
                torch.cat([x_t, x_t], dim=0), t
            ).to(dtype=unet.dtype)
            embeddings = text_embeddings

        with torch.no_grad():
            noise_pred = unet(unet_input, t, encoder_hidden_states=embeddings).sample

        if should_inject:
            attn_store.disable()
            noise_pred = noise_pred[:2]

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        x_t_minus_1    = scheduler.step(noise_pred, t, x_t).prev_sample
        t_prev         = timesteps_list[step_idx + 1].item() if step_idx + 1 < len(timesteps_list) else 0
        alpha_bar_prev = scheduler.alphas_cumprod[t_prev]
        noise          = torch.randn_like(original_latent, dtype=unet.dtype)

        if t_prev > 0:
            original_at_t_minus_1 = (alpha_bar_prev ** 0.5) * original_latent + ((1 - alpha_bar_prev) ** 0.5) * noise
        else:
            original_at_t_minus_1 = original_latent

        x_t = (mask_latent * x_t_minus_1) + ((1 - mask_latent) * original_at_t_minus_1)

    result_image = decode_latent_to_image(x_t, vae, device)
    result_image = postprocess(result_image, image, mask, resolution)
    return result_image


def load_triplets(images_dir: str, masks_dir: str, prompts_dir: str):
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


VARIANTS = [
    {"name": "end70", "injection_end": 0.7},
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RePaint inpainting with SD2-base — injection_end sweep mode")
    parser.add_argument("--images",   type=str, required=True,      help="Directory of input images (.jpg/.png)")
    parser.add_argument("--masks",    type=str, required=True,      help="Directory of mask images (white=inpaint, black=keep)")
    parser.add_argument("--prompts",  type=str, required=True,      help="Directory of prompt .txt files")
    parser.add_argument("--output",   type=str, default="results",  help="Root output directory (default: results)")
    parser.add_argument("--steps",    type=int,   default=50,       help="Diffusion steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=7.5,      help="CFG guidance scale (default: 7.5)")
    parser.add_argument("--seed",     type=int,   default=42,       help="Base random seed (default: 42)")
    parser.add_argument("--device",   type=str,   default="cuda",   help="cuda or cpu (default: cuda)")
    args = parser.parse_args()

    root_dir = Path(args.output)
    for v in VARIANTS:
        (root_dir / f"Results-{v['name']}").mkdir(parents=True, exist_ok=True)

    print("Scanning directories for matching triplets...")
    triplets = load_triplets(args.images, args.masks, args.prompts)

    print("Loading model (this happens once for all images)...")
    tokenizer, text_encoder, vae, unet, scheduler = load_pipeline_components(device=args.device)

    for i, triplet in enumerate(triplets):
        name, image, mask, prompt = triplet["name"], triplet["image"], triplet["mask"], triplet["prompt"]
        per_image_seed = args.seed + i
        print(f"\n[{i+1}/{len(triplets)}] Processing '{name}' — seed {per_image_seed}")
        print(f"  Prompt: \"{prompt}\"")

        for v in VARIANTS:
            print(f"  -> injection_end={v['injection_end']}")
            result = repaint_inpainting(
                image                = image,
                mask                 = mask,
                prompt               = prompt,
                tokenizer            = tokenizer,
                text_encoder         = text_encoder,
                vae                  = vae,
                unet                 = unet,
                scheduler            = scheduler,
                num_inference_steps  = args.steps,
                guidance_scale       = args.guidance,
                seed                 = per_image_seed,
                device               = args.device,
                inject_attention     = True,
                injection_end        = v["injection_end"],
                injection_schedule   = "step",
                exp_decay            = 5.0
            )
            out_path = root_dir / f"Results-{v['name']}" / f"{name}_result.png"
            result.save(out_path)
            print(f"     Saved -> {out_path}")

    print(f"\nDone! All results saved under: {root_dir}/")