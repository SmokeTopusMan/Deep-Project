"""
Base RePaint-Style Inpainting Pipeline using Stable Diffusion 2.

Flow per iteration:
    1. UNet predicts noise in x_t (using text prompt via CFG)
    2. Scheduler removes predicted noise → x_{t-1}
    3. Keep x_{t-1} where mask=1 (inpaint region  → use UNet output)
    4. Re-noise original image to level t-1
    5. Paste re-noised original where mask=0 (keep region → use original)
    6. Blended result becomes new x_t for next iteration

Mask convention (both formats supported):
    PIL Image : BLACK (0)   = inpaint (change this region)
                WHITE (255) = keep    (preserve this region)
    Numpy arr : 1 = inpaint, 0 = keep
"""

import torch
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm


# =============================================================================
# STAGE 1 — Load all SD2 components
# =============================================================================

def load_pipeline_components(
        model_id: str = "sd2-community/stable-diffusion-2-base",
        device: str = "cuda"
):
    """
    Load all SD2 components individually so we have full manual control
    over the denoising loop.

    Components:
        tokenizer    : converts text prompt to token ids
        text_encoder : converts token ids to CLIP embeddings
        vae          : encodes images to latent space / decodes back
        unet         : predicts noise at each timestep
        scheduler    : manages the noise schedule (DDPM)
    """
    print(f"Loading model components from '{model_id}' ...")

    tokenizer    = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae          = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet         = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    scheduler    = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze all weights — we are at inference only, no training
    for model in [text_encoder, vae, unet]:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    print("All components loaded successfully.")
    return tokenizer, text_encoder, vae, unet, scheduler


# =============================================================================
# STAGE 2 — Preprocessing helpers
# =============================================================================

def preprocess_image(image: Image.Image, resolution: int = 512) -> torch.Tensor:
    """
    Prepare a PIL image for VAE encoding.

    Steps:
        - Resize to (resolution x resolution)
        - Convert pixel values from [0, 255] to [-1, 1]
          (SD2 VAE expects this range)

    Returns:
        Tensor of shape (1, 3, H, W) with values in [-1, 1]
    """
    image = image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    arr   = np.array(image).astype(np.float32) / 255.0  # [0, 1]
    arr   = arr * 2.0 - 1.0                              # [-1, 1]
    # (H, W, 3) → (3, H, W) → (1, 3, H, W)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def preprocess_mask(mask, resolution: int = 512) -> torch.Tensor:
    """
    Prepare a mask for use in the blending step.

    Accepts two formats:
        PIL Image : BLACK (0)   → inpaint here (will become 1)
                    WHITE (255) → keep here    (will become 0)
        Numpy arr : 1 = inpaint, 0 = keep (used as-is)

    The mask is downscaled to latent resolution (resolution // 8)
    because everything in the pipeline runs in latent space.

    Returns:
        Tensor of shape (1, 1, H//8, W//8) with binary values {0, 1}
        where 1 = inpaint, 0 = keep
    """
    latent_res = resolution // 8  # VAE compresses by factor of 8

    if isinstance(mask, np.ndarray):
        # Numpy bitmap: 1=inpaint, 0=keep — use directly
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
        pil_mask = pil_mask.resize((latent_res, latent_res), Image.NEAREST)
        arr = np.array(pil_mask).astype(np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32)   # 1=inpaint, 0=keep

    else:
        # PIL image: BLACK=inpaint, WHITE=keep — flip so black becomes 1
        pil_mask = mask.convert("L").resize((latent_res, latent_res), Image.NEAREST)
        arr = np.array(pil_mask).astype(np.float32) / 255.0
        arr = (arr < 0.5).astype(np.float32)   # black(0) → 1=inpaint

    # (H, W) → (1, 1, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


# =============================================================================
# STAGE 3 — VAE encode / decode
# =============================================================================

def encode_image_to_latent(
        image_tensor: torch.Tensor,
        vae: AutoencoderKL,
        device: str
) -> torch.Tensor:
    """
    Compress a pixel-space image into VAE latent space.

    The VAE encodes 512×512×3 → 64×64×4 (8x spatial compression).
    The latent is then scaled by vae.config.scaling_factor (0.18215)
    to normalize the latent distribution for the UNet.

    Returns:
        Latent tensor of shape (1, 4, H//8, W//8)
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        latent = latent * vae.config.scaling_factor   # scale: × 0.18215
    return latent


def decode_latent_to_image(
        latent: torch.Tensor,
        vae: AutoencoderKL,
        device: str
) -> Image.Image:
    """
    Decompress a VAE latent back to pixel-space image.

    Reverses the encoding: undo the scaling factor, then decode.

    Returns:
        PIL Image of size (512, 512)
    """
    latent = latent.to(device)
    with torch.no_grad():
        # Undo the scaling factor before decoding
        image_tensor = vae.decode(latent / vae.config.scaling_factor).sample

    # [-1, 1] → [0, 1] → [0, 255] uint8
    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0)        # (H, W, 3)
    image_tensor = (image_tensor.clamp(-1, 1) + 1.0) / 2.0        # [0, 1]
    image_np     = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


# =============================================================================
# STAGE 4 — Text encoding
# =============================================================================

def encode_text_prompt(
        prompt: str,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        device: str
) -> torch.Tensor:
    """
    Encode text prompt into CLIP embeddings for Classifier-Free Guidance (CFG).

    CFG requires TWO embeddings per step:
        [0] conditional   : the real prompt  ("a dog sitting on a desk")
        [1] unconditional : empty prompt     ("")

    At each denoising step, both are passed through the UNet in one batched
    forward pass, then combined:
        noise_pred = uncond + guidance_scale × (cond - uncond)

    Returns:
        Stacked embeddings of shape (2, seq_len, hidden_dim)
        Index 0 = conditional, Index 1 = unconditional
    """
    # Tokenize both prompts together in one call
    tokens = tokenizer(
        [prompt, ""],                        # [conditional, unconditional]
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids.to(device))[0]
    # Shape: (2, 77, 1024) — both conditional and unconditional stacked
    return embeddings


# =============================================================================
# STAGE 5 — The main RePaint inpainting pipeline
# =============================================================================

def repaint_inpainting(
        image: Image.Image,
        mask,
        prompt: str,
        model_id: str        = "sd2-community/stable-diffusion-2-base",
        num_inference_steps: int   = 50,
        guidance_scale: float      = 7.5,
        seed: int                  = 42,
        resolution: int            = 512,
        device: str                = "cuda"
) -> Image.Image:
    """
    Perform RePaint-style inpainting using Stable Diffusion 2.

    Args:
        image               : Original PIL image to inpaint
        mask                : PIL image (black=inpaint) or numpy array (1=inpaint)
        prompt              : Text describing what to generate in the masked region
        model_id            : HuggingFace model ID for SD2
        num_inference_steps : Number of reverse diffusion steps (more = slower but better)
        guidance_scale      : CFG strength. Higher = more prompt-adherent (7.5 is standard)
        seed                : Random seed for reproducibility
        resolution          : Image resolution (SD2-base uses 512)
        device              : "cuda" or "cpu"

    Returns:
        Inpainted PIL Image
    """

    # ------------------------------------------------------------------
    # 1. Load model components
    # ------------------------------------------------------------------
    tokenizer, text_encoder, vae, unet, scheduler = load_pipeline_components(
        model_id, device
    )

    # Tell the scheduler how many steps we want
    # This sets up the timestep sequence: [T, T-step, T-2*step, ..., 0]
    scheduler.set_timesteps(num_inference_steps)

    generator = torch.Generator(device=device).manual_seed(seed)

    # ------------------------------------------------------------------
    # 2. Preprocess inputs
    # ------------------------------------------------------------------
    # Image: (1, 3, 512, 512) in [-1, 1]
    image_tensor = preprocess_image(image, resolution)

    # Mask: (1, 1, 64, 64) with values {0, 1}
    #   1 = inpaint (generate new content here)
    #   0 = keep    (preserve original here)
    mask_latent = preprocess_mask(mask, resolution).to(device)

    # ------------------------------------------------------------------
    # 3. Encode original image to latent space
    #    x_0: (1, 4, 64, 64)
    #    This is the "ground truth" latent we will anchor the unmasked
    #    region to throughout the denoising loop.
    # ------------------------------------------------------------------
    original_latent = encode_image_to_latent(image_tensor, vae, device)

    # ------------------------------------------------------------------
    # 4. Encode text prompt for CFG
    #    text_embeddings: (2, 77, 1024)
    #    [0] = conditional (real prompt)
    #    [1] = unconditional (empty prompt "")
    # ------------------------------------------------------------------
    text_embeddings = encode_text_prompt(prompt, tokenizer, text_encoder, device)

    # ------------------------------------------------------------------
    # 5. Initialize x_T — pure Gaussian noise
    #    This is the starting point for the masked region.
    #    The unmasked region will be immediately replaced in step E,
    #    so its initial values don't matter.
    # ------------------------------------------------------------------
    x_t = torch.randn(
        original_latent.shape,   # (1, 4, 64, 64)
        generator=generator,
        device=device
    )
    # Scale by the initial noise sigma that the scheduler expects
    x_t = x_t * scheduler.init_noise_sigma

    # ------------------------------------------------------------------
    # 6. Reverse diffusion loop: T → 0
    #    Each iteration = one timestep, going from noisy to clean
    # ------------------------------------------------------------------
    for t in tqdm(scheduler.timesteps, desc="Inpainting"):

        # --------------------------------------------------------------
        # Step A: Prepare UNet input for CFG
        # Duplicate x_t so we can run conditional and unconditional
        # passes in a single batched forward call through the UNet.
        # Shape: (2, 4, 64, 64)
        # --------------------------------------------------------------
        unet_input = torch.cat([x_t, x_t], dim=0)
        unet_input = scheduler.scale_model_input(unet_input, t)

        # --------------------------------------------------------------
        # Step B: UNet predicts the noise inside x_t
        # The UNet uses text_embeddings as cross-attention context —
        # this is how the prompt influences what gets generated.
        # noise_pred shape: (2, 4, 64, 64)
        #   [0] = conditional prediction   (guided by prompt)
        #   [1] = unconditional prediction (no prompt guidance)
        # --------------------------------------------------------------
        with torch.no_grad():
            noise_pred = unet(
                unet_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample

        # --------------------------------------------------------------
        # Step C: Apply Classifier-Free Guidance (CFG)
        # Push the prediction further in the direction the prompt pulls.
        # guidance_scale=7.5 means "go 7.5x in the prompt direction"
        # --------------------------------------------------------------
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # --------------------------------------------------------------
        # Step D: Scheduler removes the predicted noise
        # Takes x_t and the predicted noise, outputs x_{t-1}
        # x_{t-1} is one step less noisy than x_t
        # This is the UNet's best guess for the ENTIRE image at t-1
        # (both masked and unmasked regions — we will discard the
        # unmasked region in Step F below)
        # --------------------------------------------------------------
        x_t_minus_1 = scheduler.step(noise_pred, t, x_t).prev_sample

        # --------------------------------------------------------------
        # Step E: Re-noise the original image to level t-1
        # We cannot paste the clean original (noise level 0) directly
        # because the masked region is at noise level t-1.
        # Both regions must be at the same noise level.
        #
        # Forward process formula:
        #   x_{t-1} = sqrt(α_bar_{t-1}) * x_0 + sqrt(1 - α_bar_{t-1}) * ε
        #
        # As t → 0: α_bar_{t-1} → 1, so less and less noise is added,
        # and at t=0 the unmasked region becomes exactly x_0.
        # --------------------------------------------------------------

        # Get the previous timestep index
        # (the timestep value one step ahead in the schedule)
        current_idx = (scheduler.timesteps == t).nonzero().item()
        if current_idx + 1 < len(scheduler.timesteps):
            t_prev = scheduler.timesteps[current_idx + 1].item()
        else:
            t_prev = 0

        # Get α_bar at t-1 from the scheduler's precomputed values
        alpha_bar_prev = scheduler.alphas_cumprod[t_prev]

        # Sample fresh noise for the forward process
        noise = torch.randn_like(original_latent)

        if t_prev > 0:
            # Re-noise the original to level t-1:
            # more noise early (t large), less noise late (t small)
            original_at_t_minus_1 = (
                    (alpha_bar_prev ** 0.5)       * original_latent +
                    ((1 - alpha_bar_prev) ** 0.5) * noise
            )
        else:
            # At the very last step (t=0), use the clean original directly
            original_at_t_minus_1 = original_latent

        # --------------------------------------------------------------
        # Step F: RePaint blending
        # For each pixel in the latent:
        #   mask=1 (inpaint region) → keep UNet's output   → generate dog
        #   mask=0 (keep region)    → use re-noised original → keep desk
        #
        # x_{t-1}_final = mask × x_{t-1}_UNet + (1 - mask) × original_at_{t-1}
        # --------------------------------------------------------------
        x_t = (mask_latent * x_t_minus_1) + ((1 - mask_latent) * original_at_t_minus_1)

    # ------------------------------------------------------------------
    # 7. Decode the final clean latent (x_0) back to pixel space
    # ------------------------------------------------------------------
    result_image = decode_latent_to_image(x_t, vae, device)
    return result_image


# =============================================================================
# Directory loading helper
# =============================================================================

def load_triplets(images_dir: str, masks_dir: str, prompts_dir: str):
    """
    Load matching (image, mask, prompt) triplets from three directories.

    Matching is done by filename stem (the part before the extension).
    For example, these three files form one triplet:
        Images/  dog_on_desk.jpg
        Masks/   dog_on_desk.png
        Prompts/ dog_on_desk.txt

    Rules:
        - Images  : .jpg or .png files
        - Masks   : .jpg or .png files
        - Prompts : .txt files
        - A triplet is only valid if ALL THREE files exist for a given stem
        - Files with no matching counterparts are skipped with a warning

    Returns:
        List of dicts, each with keys:
            "name"   : the shared filename stem (e.g. "dog_on_desk")
            "image"  : PIL Image (RGB)
            "mask"   : PIL Image (L, black=inpaint, white=keep)
            "prompt" : str
    """
    from pathlib import Path

    images_dir  = Path(images_dir)
    masks_dir   = Path(masks_dir)
    prompts_dir = Path(prompts_dir)

    # Collect all stems that have an image file
    image_stems = {}
    for ext in [".jpg", ".jpeg", ".png"]:
        for f in images_dir.glob(f"*{ext}"):
            image_stems[f.stem] = f

    # Collect all stems that have a mask file
    mask_stems = {}
    for ext in [".jpg", ".jpeg", ".png"]:
        for f in masks_dir.glob(f"*{ext}"):
            mask_stems[f.stem] = f

    # Collect all stems that have a prompt file
    prompt_stems = {}
    for f in prompts_dir.glob("*.txt"):
        prompt_stems[f.stem] = f

    # Find stems present in ALL THREE directories
    all_stems    = set(image_stems) | set(mask_stems) | set(prompt_stems)
    valid_stems  = set(image_stems) & set(mask_stems) & set(prompt_stems)
    skipped      = all_stems - valid_stems

    if skipped:
        print(f"\nWarning: skipping {len(skipped)} incomplete triplet(s): {sorted(skipped)}")

    if not valid_stems:
        raise ValueError(
            f"No complete triplets found!\n"
            f"  Images dir  : {images_dir}  ({len(image_stems)} files)\n"
            f"  Masks dir   : {masks_dir}   ({len(mask_stems)} files)\n"
            f"  Prompts dir : {prompts_dir} ({len(prompt_stems)} files)\n"
            f"Make sure image, mask and prompt share the same filename stem."
        )

    # Load and return all valid triplets, sorted by name for reproducibility
    triplets = []
    for stem in sorted(valid_stems):
        image_path  = image_stems[stem]
        mask_path   = mask_stems[stem]
        prompt_path = prompt_stems[stem]

        image  = Image.open(image_path).convert("RGB")
        mask   = Image.open(mask_path)                   # preprocess_mask handles conversion
        prompt = prompt_path.read_text(encoding="utf-8").strip()

        triplets.append({
            "name"   : stem,
            "image"  : image,
            "mask"   : mask,
            "prompt" : prompt,
        })

        print(f"  Loaded '{stem}': prompt = \"{prompt}\"")

    print(f"\nFound {len(triplets)} valid triplet(s).\n")
    return triplets


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="RePaint inpainting with SD2-base — batch directory mode",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--images_dir",  type=str, required=True,
        help="Directory containing input images (.jpg / .png)"
    )
    parser.add_argument(
        "--masks_dir",   type=str, required=True,
        help="Directory containing mask images (.jpg / .png)\n"
             "  BLACK pixels = inpaint (change here)\n"
             "  WHITE pixels = keep    (preserve here)"
    )
    parser.add_argument(
        "--prompts_dir", type=str, required=True,
        help="Directory containing prompt text files (.txt)\n"
             "Each .txt file must share the same filename stem as its image/mask."
    )
    parser.add_argument(
        "--output_dir",  type=str, default="results",
        help="Directory to save inpainted results (default: ./results)"
    )
    parser.add_argument("--steps",    type=int,   default=50,  help="Number of diffusion steps (default: 50)")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG guidance scale (default: 7.5)")
    parser.add_argument("--seed",     type=int,   default=42,  help="Random seed (default: 42)")
    parser.add_argument("--device",   type=str,   default="cuda", help="cuda or cpu (default: cuda)")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all triplets from the directories
    print("Scanning directories for matching triplets...")
    triplets = load_triplets(args.images_dir, args.masks_dir, args.prompts_dir)

    # Load model components once, reuse for all images
    print("Loading model (this happens once for all images)...")
    tokenizer, text_encoder, vae, unet, scheduler = load_pipeline_components(
        device=args.device
    )

    # Process each triplet
    for i, triplet in enumerate(triplets):
        name   = triplet["name"]
        image  = triplet["image"]
        mask   = triplet["mask"]
        prompt = triplet["prompt"]

        print(f"\n[{i+1}/{len(triplets)}] Processing '{name}' ...")
        print(f"  Prompt: \"{prompt}\"")

        result = repaint_inpainting(
            image               = image,
            mask                = mask,
            prompt              = prompt,
            num_inference_steps = args.steps,
            guidance_scale      = args.guidance,
            seed                = args.seed,
            device              = args.device
        )

        # Save result with same name as input, always as PNG
        out_path = output_dir / f"{name}_result.png"
        result.save(out_path)
        print(f"  Saved → {out_path}")

    print(f"\nDone! All results saved to: {output_dir}/")