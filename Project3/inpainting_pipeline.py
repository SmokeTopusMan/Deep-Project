"""
RePaint-Style Inpainting Pipeline — Stable Diffusion 2.

Mask convention:
    PIL Image : WHITE = inpaint, BLACK = keep
    Numpy arr : 1 = inpaint, 0 = keep
"""

import math
import torch
import numpy as np
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
        pil_mask = mask.convert("L").resize((latent_res, latent_res), Image.NEAREST)
        arr = np.array(pil_mask).astype(np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32)

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
        arr = (arr > 0.5).astype(np.float32)

    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def encode_image_to_latent(image_tensor: torch.Tensor, vae: AutoencoderKL, device: str) -> torch.Tensor:
    """Encodes pixel image to VAE latent space. Returns (1, 4, H//8, W//8)."""
    image_tensor = image_tensor.to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
    return latent


def decode_latent_to_image(latent: torch.Tensor, vae: AutoencoderKL, device: str) -> Image.Image:
    """Decodes VAE latent back to a PIL image."""
    latent = latent.to(device=device, dtype=vae.dtype)
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
    orig_np   = np.array(original_image.convert("RGB").resize((resolution, resolution), Image.LANCZOS))
    result_np = np.array(result_image)

    if isinstance(mask, np.ndarray):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    else:
        mask_pil = mask.convert("RGB")
    mask_np = np.array(mask_pil.resize((resolution, resolution), Image.NEAREST))

    is_white = np.all(mask_np > 250, axis=2)

    output_np = result_np.copy()
    output_np[~is_white] = orig_np[~is_white]

    return Image.fromarray(output_np)


def build_resample_schedule(
        num_steps: int,
        jump_max: int,
        jump_min: int,
        rep_max: int
) -> list:
    """
    Builds a list of segments (start, jump, rep) that together cover all num_steps.

    Each segment starts at index `start`, walks forward `jump` steps, and repeats `rep` times.
    Segments are built greedily: start=0, then start+=jump, until we cover num_steps.

    Cosine schedule per segment:
        progress = 0 (start, high noise):  rep -> rep_max,  jump -> jump_min
        progress = 1 (end,   low  noise):  rep -> 1,         jump -> jump_max
    """
    segments = []
    i = 0
    while i < num_steps:
        progress = i / max(num_steps - 1, 1)
        jump = max(1, round(jump_min + (jump_max - jump_min) * 0.5 * (1.0 - math.cos(math.pi * progress))))
        jump = min(jump, num_steps - i)
        rep  = max(1, round(rep_max * 0.5 * (1.0 + math.cos(math.pi * progress))))
        segments.append((i, jump, rep))
        i += jump
    return segments





def denoise_blend_step(
        x_t: torch.Tensor,
        step_idx: int,
        timesteps_list: list,
        original_latent: torch.Tensor,
        mask_latent: torch.Tensor,
        text_embeddings: torch.Tensor,
        unet,
        scheduler,
        guidance_scale: float,
        device: str
) -> torch.Tensor:
    """
    Runs one full denoising + RePaint blend step.
    Takes x_t at noise level timesteps_list[step_idx],
    returns x_t at noise level timesteps_list[step_idx + 1] (or 0 at the last step).
    """
    t = timesteps_list[step_idx]

    unet_input = scheduler.scale_model_input(torch.cat([x_t, x_t], dim=0), t)
    unet_input = unet_input.to(dtype=unet.dtype)
    with torch.no_grad():
        noise_pred = unet(unet_input, t, encoder_hidden_states=text_embeddings).sample

    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    x_t_minus_1 = scheduler.step(noise_pred, t, x_t).prev_sample

    t_prev         = timesteps_list[step_idx + 1].item() if step_idx + 1 < len(timesteps_list) else 0
    alpha_bar_prev = scheduler.alphas_cumprod[t_prev]
    noise          = torch.randn_like(original_latent, dtype=unet.dtype)

    if t_prev > 0:
        original_at_t_minus_1 = (alpha_bar_prev ** 0.5) * original_latent + ((1 - alpha_bar_prev) ** 0.5) * noise
    else:
        original_at_t_minus_1 = original_latent

    return (mask_latent * x_t_minus_1) + ((1 - mask_latent) * original_at_t_minus_1)


def renoise(
        x_t: torch.Tensor,
        from_idx: int,
        to_idx: int,
        timesteps_list: list,
        scheduler,
        device: str
) -> torch.Tensor:
    """
    Adds noise to x_t to jump from a cleaner noise level back to a noisier one.

    from_idx > to_idx  (to_idx is earlier in the schedule = higher noise level)

    Uses the DDPM conditional forward process:
        x_{t_to} = sqrt(alpha_to / alpha_from) * x_{t_from}
                 + sqrt(1 - alpha_to / alpha_from) * eps
    """
    t_from = timesteps_list[from_idx].item() if from_idx < len(timesteps_list) else 0
    t_to   = timesteps_list[to_idx].item()

    alpha_from = scheduler.alphas_cumprod[t_from].to(device) if t_from > 0 else torch.ones(1, device=device)
    alpha_to   = scheduler.alphas_cumprod[t_to].to(device)

    ratio       = (alpha_to / alpha_from).sqrt()
    noise_scale = (1.0 - alpha_to / alpha_from).clamp(min=0.0).sqrt()

    noise = torch.randn_like(x_t)
    return (ratio * x_t + noise_scale * noise).to(dtype=x_t.dtype)


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
        jump_max: int             = 10,
        jump_min: int             = 2,
        rep_max: int              = 5
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
        jump_max            : Max jump size (used at end of schedule, low noise)
        jump_min            : Min jump size (used at start of schedule, high noise)
        rep_max             : Max repetitions per step (used at start of schedule)
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

    segments       = build_resample_schedule(num_inference_steps, jump_max, jump_min, rep_max)
    timesteps_list = list(scheduler.timesteps)

    total_denoise_ops = sum(jump * rep for _, jump, rep in segments)
    print(f"  Resampling schedule: {len(segments)} segments, ~{total_denoise_ops} total denoising ops (base: {num_inference_steps})")

    pbar = tqdm(total=num_inference_steps, desc="Inpainting")

    for start, jump, rep in segments:
        end = start + jump

        for r in range(rep):
            for j in range(start, end):
                x_t = denoise_blend_step(
                    x_t, j, timesteps_list,
                    original_latent, mask_latent, text_embeddings,
                    unet, scheduler, guidance_scale, device
                )

            if r < rep - 1:
                x_t = renoise(x_t, end, start, timesteps_list, scheduler, device)

        pbar.update(jump)

    pbar.close()

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
    parser.add_argument("--images",   type=str, required=True,     help="Directory of input images (.jpg/.png)")
    parser.add_argument("--masks",    type=str, required=True,     help="Directory of mask images (white=inpaint, black=keep)")
    parser.add_argument("--prompts",  type=str, required=True,     help="Directory of prompt .txt files")
    parser.add_argument("--output",   type=str, default="results", help="Directory to save results")
    parser.add_argument("--steps",    type=int,   default=50,      help="Diffusion steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=7.5,     help="CFG guidance scale (default: 7.5)")
    parser.add_argument("--seed",     type=int,   default=42,      help="Random seed (default: 42)")
    parser.add_argument("--device",   type=str,   default="cuda",  help="cuda or cpu (default: cuda)")
    parser.add_argument("--jump-max", type=int,   default=10,      help="Max "
                                                                        "jump size at end of schedule (default: 10)")
    parser.add_argument("--jump-min", type=int,   default=3,       help="Min "
                                                                        "jump size at start of schedule (default: 2)")
    parser.add_argument("--rep-max",  type=int,   default=2,       help="Max "
                                                                        "repetitions at start of schedule (default: 5)")
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
            jump_max            = args.jump_max,
            jump_min            = args.jump_min,
            rep_max             = args.rep_max
        )

        out_path = output_dir / f"{name}_result.png"
        result.save(out_path)
        print(f"  Saved -> {out_path}")

    print(f"\nDone! All results saved to: {output_dir}/")