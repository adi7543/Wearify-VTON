import os
import torch
import numpy as np
import cv2
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image

# --- 1. PATH CONFIGURATION ---
input_folder = r"our dataset/images-resized"  # Folder containing your original JPGs
output_folder = r"our dataset/images-parse"  # Folder to save the new Maps
os.makedirs(output_folder, exist_ok=True)

# --- 2. LOAD SEGFORMER MODEL ---
# We use a model fine-tuned on the "ATR" dataset (Human Parsing)
model_name = "mattmdjaga/segformer_b2_clothes"

processor = SegformerImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

# Auto-detect GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Model loaded on {device}")
# --- 3. PROCESSING LOOP ---
valid_extensions = ('.jpg', '.jpeg', '.png')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)

    try:
        # A. Load Image
        image = Image.open(img_path).convert("RGB")

        # B. Prepare for Model
        inputs = processor(images=image, return_tensors="pt").to(device)

        # C. Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu()

            # Upscale the output mask to match original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],  # PIL is (W, H), torch needs (H, W)
                mode="bilinear",
                align_corners=False,
            )
            pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

            # Get the Class ID (0, 1, 2...) for every pixel
            pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

            # --- E. Visualization palette (ATR / clothes-style) ---
            # palette = {
            #     0: (0, 0, 0),  # background
            #     1: (128, 0, 0),  # hat
            #     2: (255, 0, 0),  # hair
            #     3: (0, 0, 255),  # sunglasses
            #     4: (255, 255, 0),  # upper clothes
            #     5: (0, 255, 255),  # skirt
            #     6: (255, 0, 255),  # pants
            #     7: (128, 128, 128),  # dress
            #     8: (0, 128, 0),  # belt
            #     9: (128, 0, 128),  # left shoe
            #     10: (0, 128, 128),  # right shoe
            #     11: (128, 128, 0),  # face
            #     12: (255, 165, 0),  # left leg
            #     13: (0, 165, 255),  # right leg
            #     14: (255, 192, 203),  # left arm
            #     15: (255, 105, 180),  # right arm
            #     16: (139, 69, 19),  # bag
            #     17: (75, 0, 130),  # scarf
            # }

            palette = {
                0: (0, 0, 0),  # background

                # Head / upper body
                2: (255, 0, 0),  # hair
                11: (128, 128, 0),  # face / neck

                # Upper garment (KURTA)
                4: (255, 255, 0),  # kurta torso (reused from upper clothes)

                # Arms & sleeves interaction
                14: (255, 192, 203),  # left arm
                15: (255, 105, 180),  # right arm

                # Optional / contextual garments
                17: (75, 0, 130),  # dupatta (reused from scarf, female)
                6: (255, 0, 255),  # lower garment (shalwar / trousers) – optional

                # Everything below should be IGNORED or MASKED OUT in conditioning
                1: (0, 0, 0),  # hat (unused)
                3: (0, 0, 0),  # sunglasses (unused)
                5: (0, 0, 0),  # skirt (unused)
                7: (0, 0, 0),  # dress (unused)
                8: (0, 0, 0),  # belt (unused)
                9: (0, 0, 0),  # left shoe (unused)
                10: (0, 0, 0),  # right shoe (unused)
                12: (0, 0, 0),  # left leg (unused)
                13: (0, 0, 0),  # right leg (unused)
                16: (0, 0, 0),  # bag (unused)
            }


        # D. Save as Grayscale PNG
        # The values (0-17) are saved directly as pixel values.
        output_filename = os.path.splitext(img_name)[0] + ".png"
        output_path = os.path.join(output_folder, output_filename)

        # Use OpenCV to save (ensures correct grayscale formatting)
        cv2.imwrite(output_path, pred_seg.astype(np.uint8))

    except Exception as e:
        print(f"❌ Error processing {img_name}: {e}")

print("✅ All Done!")