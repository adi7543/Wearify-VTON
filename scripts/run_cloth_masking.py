# import os
# import cv2
# import numpy as np
# from rembg import remove
# from tqdm import tqdm
#
# # --- 1. DEFINE PATHS ---
# # Update these if your folder names are different
# input_dir = r"our dataset\New folder\fixed_garments"
# output_dir = r"our dataset\cloth-mask-2"
#
# # Create output folder
# os.makedirs(output_dir, exist_ok=True)
#
# print(f"Input Cloth: {input_dir}")
# print(f"Output Masks: {output_dir}")
#
# # Get list of images
# image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
# print(f"Found {len(image_files)} images to process...")
#
# # --- 2. PROCESS IMAGES ---
# for file_name in tqdm(image_files):
#     input_path = os.path.join(input_dir, file_name)
#     output_path = os.path.join(output_dir, file_name) # Keep same filename
#
#     try:
#         # Read the image
#         image = cv2.imread(input_path)
#         if image is None:
#             continue
#
#         # Run rembg to remove background
#         # This returns the image with an alpha channel (transparency)
#         result_rgba = remove(image)
#
#         # --- 3. CONVERT TO BINARY MASK ---
#         # We don't want the color image; we want a black/white mask.
#         # Extract the alpha channel (the 4th channel: 0=transparent, 255=opaque)
#         alpha_channel = result_rgba[:, :, 3]
#
#         # Threshold it to be sure it's binary (0 or 255)
#         _, binary_mask = cv2.threshold(alpha_channel, 127, 255, cv2.THRESH_BINARY)
#
#         # Save the mask
#         cv2.imwrite(output_path, binary_mask)
#
#     except Exception as e:
#         print(f"Error processing {file_name}: {e}")
#
# print("--- Cloth masking finished! ---")


import os
import cv2
import numpy as np
from tqdm import tqdm

# --- 1. DEFINE PATHS ---
input_dir = r"../our dataset/New folder/fixed_garments"  # RGBA kurta-only images
output_dir = r"our dataset\cloth-mask-2"     # binary masks

os.makedirs(output_dir, exist_ok=True)

print(f"Input Cloth (RGBA): {input_dir}")
print(f"Output Masks: {output_dir}")

# Get list of images
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
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
