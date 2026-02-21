import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Your New V2 Model (The one that produced the red/yellow heatmap)
MODEL_PATH = r"../my_custom_segformer_v3"

# 2. Your Raw Images (The folder with 1000+ photos)
INPUT_DIR = r"../data/images"

# 3. Output Folder (This will be your MFP-VTON Dataset)
OUTPUT_DIR = r"../dataset_mfp_final_2"

# 4. Resolution
IMG_SIZE = (512, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_dirs():
    # Create the 5 necessary folders
    for d in ["images", "masks", "agnostic", "ref_person", "vis"]:
        os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)


def main():
    setup_dirs()
    print(f"--- Generating Full Dataset with V3 Model on {DEVICE} ---")

    # 1. Load Model & Processor
    # CRITICAL FIX: Load Processor from NVIDIA, Model from Local
    try:
        processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you run train_segmentation_v2.py fully?")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(files)} images. Starting processing...")

    for filename in tqdm(files):
        try:
            # A. Load Image
            src_path = os.path.join(INPUT_DIR, filename)
            image_pil = Image.open(src_path).convert("RGB")
            image_pil = image_pil.resize(IMG_SIZE, Image.BILINEAR)
            image_np = np.array(image_pil)

            # B. Predict (Inference)
            inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Upsample logic
            logits = torch.nn.functional.interpolate(logits, size=IMG_SIZE[::-1], mode="bilinear", align_corners=False)
            probs = torch.nn.functional.softmax(logits, dim=1)

            # C. Create Mask (Thresholding)
            # We use a safe threshold of 0.5.
            # Since V2 uses class weights, the model should be very confident (probs > 0.9).
            kurta_prob = probs[0, 1].cpu().numpy()
            mask = (kurta_prob > 0.5).astype(np.uint8) * 255

            # D. Clean Up (Morphology)
            # Remove small white noise specs
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # E. Generate Outputs
            # 1. Agnostic (Grey Hole)
            agnostic = image_np.copy()
            agnostic[mask == 255] = 128

            # 2. Visualization (Green Overlay)
            vis = image_np.copy()
            vis[mask == 255] = vis[mask == 255] * 0.5 + np.array([0, 255, 0]) * 0.5

            # F. Save Files (BGR for OpenCV)
            name_base = os.path.splitext(filename)[0]

            cv2.imwrite(os.path.join(OUTPUT_DIR, "images", filename), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUTPUT_DIR, "ref_person", filename),
                        cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))  # Source = Target for training
            cv2.imwrite(os.path.join(OUTPUT_DIR, "agnostic", filename), cv2.cvtColor(agnostic, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUTPUT_DIR, "vis", filename), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(OUTPUT_DIR, "masks", f"{name_base}.png"), mask)

        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    print(f"Done! Check the '{OUTPUT_DIR}/vis' folder.")
    print("If the green masks look correct, you are ready to train MFP-VTON!")


if __name__ == "__main__":
    main()