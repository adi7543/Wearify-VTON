import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = r"../my_custom_segformer_v4"  # Your trained model
GT_IMAGES_DIR = r"../dataset_manual/images"  # Your 400 manual photos
GT_MASKS_DIR = r"../dataset_manual/masks"  # Your 400 manual masks (The "Answer Key")
IMG_SIZE = (512, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_iou(pred_mask, gt_mask):
    # Flatten to 1D arrays
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()

    # Intersection: Where BOTH are 1 (Shirt)
    intersection = np.logical_and(pred, gt).sum()

    # Union: Where EITHER is 1
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0  # Both empty = Perfect match

    return intersection / union


def main():
    print(f"--- EVALUATING MODEL: {MODEL_PATH} ---")

    try:
        processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error: {e}")
        return

    files = [f for f in os.listdir(GT_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    iou_scores = []

    print(f"Testing on {len(files)} manual images...")

    for filename in tqdm(files):
        # 1. Load Image
        img_path = os.path.join(GT_IMAGES_DIR, filename)
        image = Image.open(img_path).convert("RGB")

        # 2. Load Ground Truth Mask (The Answer)
        name_base = os.path.splitext(filename)[0]
        mask_path = os.path.join(GT_MASKS_DIR, f"{name_base}.png")

        # Handle jpg masks if png missing
        if not os.path.exists(mask_path):
            mask_path = os.path.join(GT_MASKS_DIR, f"{name_base}.jpg")

        if not os.path.exists(mask_path):
            print(f"Skipping {filename}: GT Mask missing")
            continue

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        # Binarize GT (0 or 1)
        gt_mask = (gt_mask > 127).astype(np.uint8)

        # 3. Model Prediction
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Resize logits to match image size
        logits = torch.nn.functional.interpolate(logits, size=IMG_SIZE[::-1], mode="bilinear", align_corners=False)
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Get Prediction (Threshold 0.5)
        pred_mask = (probs[0, 1].cpu().numpy() > 0.5).astype(np.uint8)

        # 4. Calculate IoU
        score = calculate_iou(pred_mask, gt_mask)
        iou_scores.append(score)

    # --- REPORT CARD ---
    if len(iou_scores) > 0:
        mean_iou = np.mean(iou_scores)
        print("\n" + "=" * 30)
        print(f"FINAL REPORT CARD (mIoU)")
        print("=" * 30)
        print(f"Mean IoU:      {mean_iou:.4f}  (Target: > 0.85)")
        print(f"Min IoU:       {np.min(iou_scores):.4f}")
        print(f"Max IoU:       {np.max(iou_scores):.4f}")
        print("-" * 30)

        if mean_iou > 0.90:
            print("VERDICT: EXCELLENT. Your model is a Surgeon.")
        elif mean_iou > 0.80:
            print("VERDICT: GOOD. Solid performance for VTON.")
        else:
            print("VERDICT: POOR. The model is struggling to learn.")
    else:
        print("No metrics calculated.")


if __name__ == "__main__":
    main()