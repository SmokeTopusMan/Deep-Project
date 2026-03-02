import os
import argparse
from datasets import load_dataset
from tqdm import tqdm

def main():
    # 1. Set up Command Line Arguments
    parser = argparse.ArgumentParser(description="Download InpaintCOCO and organize into folders.")
    parser.add_argument("--images", type=str, required=True, help="Folder for original COCO images")
    parser.add_argument("--masks", type=str, required=True, help="Folder for inpainting masks")
    parser.add_argument("--prompts", type=str, required=True, help="Folder for text prompts/captions")
    args = parser.parse_args()

    # 2. Create directories
    for folder in [args.images, args.masks, args.prompts]:
        os.makedirs(folder, exist_ok=True)

    # 3. Load the dataset (Full dataset, split='test' by default for this repo)
    print("Loading dataset from Hugging Face...")
    ds = load_dataset("phiyodr/InpaintCOCO", split="test")

    print(f"Downloading and saving {len(ds)} items...")

    # 4. Iterate and Save
    for i, item in enumerate(tqdm(ds)):
        file_basename = f"sample_{i:05d}"

        # Save Image (PIL format)
        img_path = os.path.join(args.images, f"{file_basename}.jpg")
        item["coco_image"].convert("RGB").save(img_path)

        # Save Mask (PIL format)
        mask_path = os.path.join(args.masks, f"{file_basename}.png")
        item["mask"].convert("L").save(mask_path)

        # Save Prompt (Text file)
        prompt_path = os.path.join(args.prompts, f"{file_basename}.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(item["inpaint_caption"])

    print(f"\nDone! Data saved to:\n- {args.images}\n- {args.masks}\n- {args.prompts}")

if __name__ == "__main__":
    main()