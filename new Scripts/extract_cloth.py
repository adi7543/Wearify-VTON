import os
import cv2
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_DIR = r"../dataset_final"
EROSION_AMOUNT = 3


def main():
    # 1. Setup Folders
    dirs = {
        "images": os.path.join(DATASET_DIR, "images"),
        "masks": os.path.join(DATASET_DIR, "masks"),
        "ref_cloth": os.path.join(DATASET_DIR, "ref_cloth"),
        "ref_cloth_mask": os.path.join(DATASET_DIR, "ref_cloth_mask")
    }

    os.makedirs(dirs["ref_cloth"], exist_ok=True)
    os.makedirs(dirs["ref_cloth_mask"], exist_ok=True)

    files = [f for f in os.listdir(dirs["images"]) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for filename in tqdm(files):
        name_base = os.path.splitext(filename)[0]

        # Paths
        img_path = os.path.join(dirs["images"], filename)
        # Handle png/jpg mask mismatch
        mask_path = os.path.join(dirs["masks"], f"{name_base}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(dirs["masks"], f"{name_base}.jpg")

        if not os.path.exists(mask_path): continue

        # Load
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None: continue

        # Resize mask to match image
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # --- THE FIX: EROSION (Trimming) ---
        # We shrink the white area of the mask.
        # This disconnects the sleeve from the hand.
        kernel = np.ones((EROSION_AMOUNT, EROSION_AMOUNT), np.uint8)
        clean_mask = cv2.erode(mask, kernel, iterations=1)

        # --- CREATE REF CLOTH ---
        # 1. White Background Canvas
        white_bg = np.ones_like(image) * 255

        # 2. Cut out cloth using CLEAN mask
        ref_cloth = np.where(clean_mask[:, :, None] == 255, image, white_bg)

        # --- SAVE ---
        # Save Cloth
        cv2.imwrite(os.path.join(dirs["ref_cloth"], filename), ref_cloth)

        # Save Cloth Mask (The eroded one)
        cv2.imwrite(os.path.join(dirs["ref_cloth_mask"], f"{name_base}.png"), clean_mask)

    print("Done! Check 'dataset_mfp_final/ref_cloth'. The hands should be gone.")


if __name__ == "__main__":
    main()