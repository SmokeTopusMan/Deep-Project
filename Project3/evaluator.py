'''
evaluator.py — Unified evaluator for inpainting:
- Ground-truth metrics (PSNR, SSIM, LPIPS + boundary versions + boundary gradient L1)
- No-ground-truth metrics (CLIP prompt-image similarity, NIQE, BRISQUE, ImageReward)
- No-ground-truth border metrics (leakage + gradient jump across seam)
- CSV logging

Conventions:
- mask: white=inpaint, black=keep
- Internally mask_01: 1=inpaint, 0=keep
'''

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Any
import csv
import os

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import pyiqa
import piq
import lpips

# CLIP for prompt-image similarity
from transformers import CLIPModel, CLIPProcessor

# Optional deps
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import ImageReward as RM
except ImportError:
    RM = None


# ----------------------------
# Helpers: conversions
# ----------------------------

def pil_to_torch01(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> torch float in [0,1], shape (1,3,H,W)."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.clamp(0.0, 1.0)

def pil_mask_to_01(mask: Image.Image, size_hw: tuple[int, int]) -> torch.Tensor:
    """
    PIL mask -> (1,1,H,W) float {0,1}, where 1=inpaint (white), 0=keep (black).
    """
    H, W = size_hw
    m = mask.convert("L").resize((W, H), Image.NEAREST)
    arr = (np.array(m).astype(np.float32) / 255.0)  # [0,1]
    arr = (arr > 0.5).astype(np.float32)           # white -> 1
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return t

def _check_img_shapes(pred: torch.Tensor, other: torch.Tensor) -> None:
    if pred.ndim != 4 or other.ndim != 4:
        raise ValueError("Expected tensors of shape (N,3,H,W)")
    if pred.shape != other.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, other={other.shape}")
    if pred.shape[1] != 3:
        raise ValueError(f"Expected 3 channels, got {pred.shape[1]}")

def make_boundary_band(mask_01: torch.Tensor, band_px: int = 10) -> torch.Tensor:
    """
    mask_01: (N,1,H,W) where 1=inpaint, 0=keep
    returns: (N,1,H,W) boundary ring around mask edges
    """
    if band_px <= 0:
        raise ValueError("band_px must be > 0")

    k = band_px * 2 + 1
    dil = F.max_pool2d(mask_01, kernel_size=k, stride=1, padding=band_px)
    ero = 1.0 - F.max_pool2d(1.0 - mask_01, kernel_size=k, stride=1, padding=band_px)
    band = (dil - ero).clamp(0.0, 1.0)
    return (band > 0).float()

def apply_band_only(pred: torch.Tensor, ref: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """Outside band, replace pred with ref so full-image metrics act only on band."""
    band3 = band.repeat(1, 3, 1, 1)
    return pred * band3 + ref * (1.0 - band3)

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)

def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """x: (N,3,H,W) in [0,1] -> (N,1,H,W) gradient magnitude."""
    xg = rgb_to_gray(x)
    device, dtype = x.device, x.dtype
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    gx = F.conv2d(xg, kx, padding=1)
    gy = F.conv2d(xg, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


# ----------------------------
# Evaluator
# ----------------------------

@dataclass
class EvaluatorConfig:
    device: str = "cuda"
    band_px: int = 10
    enable_brisque: bool = True
    enable_imagereward: bool = True

class InpaintingEvaluator:
    """
    Compute both GT and no-GT metrics.
    - If gt_image is None -> compute only no-GT metrics
    - Else -> compute both sets
    """

    def __init__(self, cfg: EvaluatorConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.niqe_metric = pyiqa.create_metric('niqe', device=torch.device(self.device))
        self.niqe_metric.eval()
        # LPIPS (GT)
        self.lpips_fn = lpips.LPIPS(net="vgg").to(self.device).eval()

        # CLIP (no-GT prompt alignment)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # ImageReward (optional)
        self.reward_model = None
        if cfg.enable_imagereward:
            if RM is not None:
                self.reward_model = RM.load("ImageReward-v1.0")
            else:
                # Not fatal; user may not have installed it
                self.reward_model = None

    @torch.no_grad()
    def compute(
        self,
        name: str,
        pred_img: Image.Image,
        input_img: Image.Image,
        mask_img: Image.Image,
        prompt: str,
        gt_img: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:

        # to tensors
        pred = pil_to_torch01(pred_img).to(self.device)
        inp  = pil_to_torch01(input_img).to(self.device)
        _check_img_shapes(pred, inp)

        H, W = pred.shape[2], pred.shape[3]
        mask_01 = pil_mask_to_01(mask_img, (H, W)).to(self.device)  # (1,1,H,W)
        band = make_boundary_band(mask_01, band_px=self.cfg.band_px)

        out: Dict[str, Any] = {"name": name}

        # ------------------
        # No-GT metrics
        # ------------------

        # Prompt-image CLIP similarity (cosine)
        out["CLIP_sim"] = float(self._clip_similarity(pred_img, prompt))


        pred_for_niqe = pred.float()  # make sure float32
        out["NIQE"] = float(self.niqe_metric(pred_for_niqe).item())

        # BRISQUE (optional, needs opencv-contrib-python)
        if self.cfg.enable_brisque and cv2 is not None:
            out["BRISQUE"] = float(self._brisque(pred_img))
        else:
            out["BRISQUE"] = None

        # ImageReward (optional, needs image-reward)
        if self.cfg.enable_imagereward and self.reward_model is not None:
            out["ImageReward"] = float(self.reward_model.score(prompt, pred_img))
        else:
            out["ImageReward"] = None

        # No-GT border leakage: outside-mask band should stay close to input
        # (detects bleeding / unintended edits outside mask)
        band_out = band * (1.0 - mask_01)             # ring on KEEP side
        band_out3 = band_out.repeat(1, 3, 1, 1)
        out["BorderLeak_MAE"] = float(masked_mean((pred - inp).abs(), band_out3).item())

        # No-GT border gradient jump: difference between inside-band gradient and outside-band gradient
        gp = sobel_magnitude(pred)                    # (1,1,H,W)
        band_in  = band * mask_01                     # ring on INPAINT side
        mean_in  = masked_mean(gp, band_in)
        mean_out = masked_mean(gp, band_out)
        out["BorderGradJump"] = float((mean_in - mean_out).abs().item())

        # ------------------
        # GT metrics (if gt exists)
        # ------------------
        if gt_img is not None:
            gt = pil_to_torch01(gt_img).to(self.device)
            _check_img_shapes(pred, gt)

            # Full image metrics
            out["PSNR_GT"] = float(piq.psnr(pred, gt, data_range=1.0, reduction="mean").item())
            out["SSIM_GT"] = float(piq.ssim(pred, gt, data_range=1.0).item())
            out["LPIPS_GT"] = float(self._lpips(pred, gt))

            # Boundary metrics: evaluate only band area
            pred_b = apply_band_only(pred, gt, band)
            out["Boundary_PSNR_GT"] = float(piq.psnr(pred_b, gt, data_range=1.0, reduction="mean").item())
            out["Boundary_SSIM_GT"] = float(piq.ssim(pred_b, gt, data_range=1.0).item())
            out["Boundary_LPIPS_GT"] = float(self._lpips(pred_b, gt))

            # Boundary gradient L1 vs GT
            gg = sobel_magnitude(gt)
            out["Boundary_GradL1_GT"] = float(masked_mean((gp - gg).abs(), band).item())
        else:
            out["PSNR_GT"] = None
            out["SSIM_GT"] = None
            out["LPIPS_GT"] = None
            out["Boundary_PSNR_GT"] = None
            out["Boundary_SSIM_GT"] = None
            out["Boundary_LPIPS_GT"] = None
            out["Boundary_GradL1_GT"] = None

        return out

    @torch.no_grad()
    def _lpips(self, pred01: torch.Tensor, gt01: torch.Tensor) -> float:
        # LPIPS expects [-1,1]
        p = pred01 * 2.0 - 1.0
        g = gt01 * 2.0 - 1.0
        return float(self.lpips_fn(p, g).mean().item())

    @torch.no_grad()
    def _clip_similarity(self, image: Image.Image, prompt: str) -> float:
        inputs = self.clip_proc(text=[prompt], images=[image.convert("RGB")], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.clip_model(**inputs)

        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        sim = (img_emb * txt_emb).sum(dim=-1)  # cosine similarity
        return float(sim.item())

    def _brisque(self, image: Image.Image) -> float:
        from brisque import BRISQUE
        import numpy as np

        img = np.array(image.convert("RGB"))
        scorer = BRISQUE()          # uses built-in model assets from the package
        return float(scorer.score(img))


# ----------------------------
# CSV logger
# ----------------------------

class CSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._fieldnames = None
        self._file_exists = os.path.exists(csv_path)

    def log(self, row: Dict[str, Any]) -> None:
        # Define columns on first row
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())

        write_header = not self._file_exists
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                w.writeheader()
                self._file_exists = True
            w.writerow(row)