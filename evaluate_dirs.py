"""
evaluate_dirs.py — Batch inpainting evaluator across multiple prediction directories.

Expected directory structure:
    Images/          GT images:    sample_NNNNN.jpg
    Masks/           Mask images:  sample_NNNNN.png
    Prompts/         Text prompts: sample_NNNNN.txt
    Results-foo/     Predictions:  sample_NNNNN_result.png
    Results-bar/     Predictions:  sample_NNNNN_result.png
    ...

Usage:
    python evaluate_dirs.py Results-foo Results-bar Results-baz
        --GT      <Images dir>
        --masks   <Masks dir>
        --prompts <Prompts dir>
        --output  <output dir for histogram PNGs>
        [--input  <corrupted input dir>]
        [--device cuda]
        [--band_px 10]
        [--no_brisque]
        [--no_imagereward]

Notes:
  - Matching is done on the base stem (e.g. "sample_00000"), stripping the
    "_result" suffix from prediction filenames automatically.
  - min_items = number of files in the smallest prediction directory.
    Only that many images (first min_items from the sorted common stems) are evaluated.
  - One histogram PNG is saved per metric to --output.
    Each bar is labeled with the prediction directory basename.
  - FID is computed once per directory across the full batch (not per-image).
  - If --input is omitted, GT is used as the corrupted input
    (BorderLeak_MAE becomes uninformative but will not crash).
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from evaluator import EvaluatorConfig, FIDAccumulator, InpaintingEvaluator

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESULT_SUFFIX = "_result"

LOWER_IS_BETTER = {
    "LPIPS_GT", "Boundary_LPIPS_GT",
    "FID",
    "NIQE", "BRISQUE",
    "BorderLeak_MAE", "BorderGradJump", "Boundary_GradL1_GT",
}

def score_to_color(goodness: float) -> tuple[float, float, float]:
    """
    goodness in [0, 1] where 1 = best.
    0   -> deep red   (0.85, 0.10, 0.10)
    0.5 -> neutral grey-purple
    1   -> deep blue  (0.10, 0.25, 0.85)
    """
    r = 0.85 * (1.0 - goodness) + 0.10 * goodness
    g = 0.10 * (1.0 - goodness) + 0.25 * goodness
    b = 0.10 * (1.0 - goodness) + 0.85 * goodness
    return (r, g, b)


def compute_bar_colors(metric: str, values: list[float | None]) -> list[tuple[float, float, float]]:
    valid = [v for v in values if v is not None]
    if len(valid) <= 1:
        return [score_to_color(0.5) if v is None else score_to_color(1.0) for v in values]

    lo, hi = min(valid), max(valid)
    colors = []
    for v in values:
        if v is None:
            colors.append(score_to_color(0.5))
            continue
        if hi == lo:
            norm = 0.5
        else:
            norm = (v - lo) / (hi - lo)
        goodness = (1.0 - norm) if metric in LOWER_IS_BETTER else norm
        colors.append(score_to_color(goodness))
    return colors


def get_stem_to_file(directory: str, strip_suffix: str = "") -> dict[str, str]:
    mapping = {}
    for fname in os.listdir(directory):
        if Path(fname).suffix.lower() not in IMAGE_EXTS:
            continue
        stem = Path(fname).stem
        if strip_suffix and stem.endswith(strip_suffix):
            stem = stem[: -len(strip_suffix)]
        mapping[stem] = fname
    return mapping


def get_prompt(prompts_dir: str, stem: str) -> str:
    txt_path = os.path.join(prompts_dir, stem + ".txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Prompt file not found: {txt_path}")
    with open(txt_path, encoding="utf-8") as f:
        return f.read().strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inpainting evaluator")
    p.add_argument("dirs",      nargs="+",     help="One or more prediction directories")
    p.add_argument("--GT",      required=True, help="Ground-truth image directory (Images/)")
    p.add_argument("--masks",   required=True, help="Mask image directory (Masks/)")
    p.add_argument("--prompts", required=True, help="Per-image .txt prompt directory (Prompts/)")
    p.add_argument("--output",  required=True, help="Directory to save histogram PNGs")
    p.add_argument("--input",   default=None,  help="Corrupted input image directory (optional)")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--band_px", type=int, default=10)
    p.add_argument("--no_brisque",     action="store_true")
    p.add_argument("--no_imagereward", action="store_true")
    return p.parse_args()


def build_common_stems(pred_dirs: list[str], gt_dir: str) -> tuple[list[str], int]:
    gt_stems = set(get_stem_to_file(gt_dir).keys())
    pred_stem_maps = [get_stem_to_file(d, strip_suffix=RESULT_SUFFIX) for d in pred_dirs]

    common = gt_stems.copy()
    for sm in pred_stem_maps:
        common &= set(sm.keys())
    common = sorted(common)

    min_items = min(len(sm) for sm in pred_stem_maps)
    return common[:min_items], min_items


def plot_metric(
    metric: str,
    dir_names: list[str],
    values: list[float | None],
    min_items: int,
    output_dir: str,
) -> None:
    bar_values = [v if v is not None else 0.0 for v in values]
    x = np.arange(len(dir_names))
    colors = compute_bar_colors(metric, values)
    fig, ax = plt.subplots(figsize=(max(6, len(dir_names) * 1.8), 5))
    bars = ax.bar(x, bar_values, color=colors, edgecolor="black", width=0.6)

    for bar, v in zip(bars, values):
        if v is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(dir_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{metric}  (avg over {min_items} images)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    safe_name = re.sub(r'[\\/*?:"<>|&]', "_", metric)
    plt.savefig(os.path.join(output_dir, f"{safe_name}.png"), dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    gt_stem_map   = get_stem_to_file(args.GT)
    mask_stem_map = get_stem_to_file(args.masks)
    pred_stem_maps = {d: get_stem_to_file(d, strip_suffix=RESULT_SUFFIX) for d in args.dirs}
    input_stem_map = get_stem_to_file(args.input) if args.input else None

    common_stems, min_items = build_common_stems(args.dirs, args.GT)

    if not common_stems:
        raise RuntimeError(
            "No common image stems found across all directories and GT. "
            "Check that prediction filenames follow the pattern sample_NNNNN_result.png "
            "and match the GT stems (sample_NNNNN)."
        )

    print(f"Found {len(common_stems)} common stems. Evaluating {min_items} per directory.")

    cfg = EvaluatorConfig(
        device=args.device,
        band_px=args.band_px,
        enable_brisque=not args.no_brisque,
        enable_imagereward=not args.no_imagereward,
    )
    evaluator = InpaintingEvaluator(cfg)

    dir_scores: dict[str, dict[str, list[float]]] = {d: defaultdict(list) for d in args.dirs}
    fid_accums: dict[str, FIDAccumulator] = {d: FIDAccumulator(device=args.device) for d in args.dirs}

    for stem in common_stems:
        gt_img = Image.open(os.path.join(args.GT, gt_stem_map[stem])).convert("RGB")

        mask_fname = mask_stem_map.get(stem)
        if mask_fname is None:
            raise FileNotFoundError(f"Mask not found for stem '{stem}' in {args.masks}")
        mask_img = Image.open(os.path.join(args.masks, mask_fname)).convert("L")

        prompt = get_prompt(args.prompts, stem)

        if input_stem_map is not None:
            inp_fname = input_stem_map.get(stem)
            if inp_fname is None:
                raise FileNotFoundError(f"Input image not found for stem '{stem}' in {args.input}")
            input_img = Image.open(os.path.join(args.input, inp_fname)).convert("RGB")
        else:
            input_img = gt_img

        for pred_dir in args.dirs:
            pred_fname = pred_stem_maps[pred_dir].get(stem)
            if pred_fname is None:
                raise FileNotFoundError(f"Prediction not found for stem '{stem}' in {pred_dir}")
            pred_img = Image.open(os.path.join(pred_dir, pred_fname)).convert("RGB")

            result = evaluator.compute(
                name=stem,
                pred_img=pred_img,
                input_img=input_img,
                mask_img=mask_img,
                prompt=prompt,
                gt_img=gt_img,
            )

            fid_accums[pred_dir].update(gt_img, pred_img)

            for metric, value in result.items():
                if metric == "name" or value is None:
                    continue
                dir_scores[pred_dir][metric].append(float(value))

        print(f"  processed {stem}")

    dir_names = [Path(d).name for d in args.dirs]

    all_metrics: set[str] = set()
    for scores in dir_scores.values():
        all_metrics.update(scores.keys())
    all_metrics.add("FID")
    all_metrics = sorted(all_metrics)

    avg_scores: dict[str, dict[str, float | None]] = {}
    for pred_dir, dir_name in zip(args.dirs, dir_names):
        avg_scores[dir_name] = {}
        for metric in all_metrics:
            if metric == "FID":
                try:
                    avg_scores[dir_name]["FID"] = fid_accums[pred_dir].compute()
                except Exception as e:
                    print(f"  FID failed for {dir_name}: {e}")
                    avg_scores[dir_name]["FID"] = None
            else:
                vals = dir_scores[pred_dir].get(metric, [])
                avg_scores[dir_name][metric] = float(np.mean(vals)) if vals else None

    saved = 0
    for metric in all_metrics:
        values = [avg_scores[dn].get(metric) for dn in dir_names]
        if all(v is None for v in values):
            continue
        plot_metric(metric, dir_names, values, min_items, args.output)
        saved += 1

    print(f"\nDone. Saved {saved} histogram(s) to: {args.output}")

    print("\n=== Average Scores ===")
    col_w = max(len(dn) for dn in dir_names) + 2
    for metric in all_metrics:
        row = f"{metric:<30}"
        for dn in dir_names:
            v = avg_scores[dn].get(metric)
            row += f"  {dn:<{col_w}} {v:.4f}" if v is not None else f"  {dn:<{col_w}} {'N/A':>8}"
        print(row)


if __name__ == "__main__":
    main()
