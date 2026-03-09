"""
RePaint-Style Inpainting Pipeline — Stable Diffusion 2.

Mask convention:
    PIL Image : WHITE = inpaint, BLACK = keep
    Numpy arr : 1 = inpaint, 0 = keep
"""

import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from pathlib import Path


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
        arr = (arr > 0.5).astype(np.float32)

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


def poisson_blend(result_np: np.ndarray, original_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    mask_u8 = np.where(mask_np > 127, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return result_np

    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    cx = x + w // 2
    cy = y + h // 2

    try:
        blended = cv2.seamlessClone(result_np, original_np, mask_u8, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        return result_np

    return blended


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
        use_poisson_blend: bool   = True
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
    Returns:
        Inpainted PIL Image with postprocessing applied
    """
    scheduler.set_timesteps(num_inference_steps)
    generator = torch.Generator(device=device).manual_seed(seed)

    image_tensor    = preprocess_image(image, resolution)
    mask_latent     = preprocess_mask(mask, resolution).to(device)
    original_latent = encode_image_to_latent(image_tensor, vae, device)
    text_embeddings = encode_text_prompt(prompt, tokenizer, text_encoder, device)

    # Start from pure Gaussian noise (x_T), cast to model dtype (fp16 on CUDA)
    x_t = torch.randn(original_latent.shape, generator=generator, device=device, dtype=unet.dtype)
    x_t = x_t * scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps, desc="Inpainting"):

        # CFG: run UNet with both conditional and unconditional embeddings in one pass
        unet_input = scheduler.scale_model_input(torch.cat([x_t, x_t], dim=0), t)
        unet_input = unet_input.to(dtype=unet.dtype)  # ensure fp16
        with torch.no_grad():
            noise_pred = unet(unet_input, t, encoder_hidden_states=text_embeddings).sample

        # Combine conditional and unconditional predictions
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Scheduler step: remove predicted noise to get x_{t-1}
        x_t_minus_1 = scheduler.step(noise_pred, t, x_t).prev_sample

        # Re-noise original to level t-1: x_{t-1} = sqrt(a)*x0 + sqrt(1-a)*eps
        current_idx = (scheduler.timesteps == t).nonzero().item()
        t_prev = scheduler.timesteps[current_idx + 1].item() if current_idx + 1 < len(scheduler.timesteps) else 0
        alpha_bar_prev = scheduler.alphas_cumprod[t_prev]
        noise = torch.randn_like(original_latent, dtype=unet.dtype)

        if t_prev > 0:
            original_at_t_minus_1 = (alpha_bar_prev ** 0.5) * original_latent + ((1 - alpha_bar_prev) ** 0.5) * noise
        else:
            original_at_t_minus_1 = original_latent

        # RePaint blend: masked region from UNet, unmasked from re-noised original
        x_t = (mask_latent * x_t_minus_1) + ((1 - mask_latent) * original_at_t_minus_1)

    result_image = decode_latent_to_image(x_t, vae, device)

    result_image = postprocess(result_image, image, mask, resolution)

    if use_poisson_blend:
        orig_np   = np.array(image.convert("RGB").resize((resolution, resolution), Image.LANCZOS))
        result_np = np.array(result_image)
        if isinstance(mask, np.ndarray):
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        else:
            mask_pil = mask.convert("L")
        mask_np   = np.array(mask_pil.resize((resolution, resolution), Image.NEAREST))
        result_np = poisson_blend(result_np, orig_np, mask_np)
        result_image = Image.fromarray(result_np)

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
    parser.add_argument("--steps",       type=int,   default=50,      help="Diffusion steps (default: 50)")
    parser.add_argument("--guidance",    type=float, default=7.5,     help="CFG guidance scale (default: 7.5)")
    parser.add_argument("--seed",        type=int,   default=42,      help="Random seed (default: 42)")
    parser.add_argument("--device",      type=str,   default="cuda",  help="cuda or cpu (default: cuda)")
    parser.add_argument("--no-poisson",  action="store_true",         help="Disable Poisson blending")
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
            seed                = args.seed,
            device              = args.device,
            use_poisson_blend   = not args.no_poisson
        )

        out_path = output_dir / f"{name}_result.png"
        result.save(out_path)
        print(f"  Saved -> {out_path}")

    print(f"\nDone! All results saved to: {output_dir}/")