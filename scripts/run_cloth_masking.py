import os
import cv2
import numpy as np
from tqdm import tqdm

# --- 1. DEFINE PATHS ---
input_dir = r"../raw images/cloth-resized"  # RGBA kurta-only images
output_dir = r"../raw images/cloth-mask"     # binary masks

os.makedirs(output_dir, exist_ok=True)

print(f"Input Cloth (RGBA): {input_dir}")
print(f"Output Masks: {output_dir}")

# Get list of images
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png'))]
print(f"Found {len(image_files)} images to process...")

# --- 2. PROCESS IMAGES ---
for file_name in tqdm(image_files):
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    try:
        # Read image WITH alpha channel
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[2] < 4:
            print(f"Skipping (no alpha): {file_name}")
            continue

        # --- 3. EXTRACT ALPHA CHANNEL ---
        alpha_channel = image[:, :, 3]

        # --- 4. CONVERT TO BINARY MASK ---
        _, binary_mask = cv2.threshold(
            alpha_channel, 127, 255, cv2.THRESH_BINARY
        )

        # Save as single-channel mask
        cv2.imwrite(output_path, binary_mask)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print("--- Cloth alpha-to-mask conversion finished! ---")
