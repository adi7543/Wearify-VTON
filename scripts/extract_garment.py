import sys
import os
import types

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUNDING_DINO_PATH = os.path.join(PROJECT_ROOT, "GroundingDINO")

sys.path.insert(0, GROUNDING_DINO_PATH)

import os
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- GroundingDINO imports ---
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util.box_ops import box_cxcywh_to_xyxy

# --- SAM imports ---
from segment_anything import sam_model_registry, SamPredictor

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# Paths
GROUNDINGDINO_CONFIG = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDINGDINO_WEIGHTS = "../GroundingDINO/groundingdino_swint_ogc.pth"
SAM_WEIGHTS = "../GroundingDINO/sam_vit_h_4b8939.pth"

INPUT_FOLDER = "../raw images/images-2"
OUTPUT_FOLDER = "../raw images/New Folder"
TEXT_PROMPT = "tunic top, shirt, blouse, kurta"  # Change if you want different garments

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Load Models ---
print("Loading GroundingDINO...")
grounding_dino_model = load_model(GROUNDINGDINO_CONFIG, GROUNDINGDINO_WEIGHTS)
grounding_dino_model = grounding_dino_model.to(DEVICE)

print("Loading SAM...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHTS).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# --- Helper function for extraction ---
# def extract_garment(image_path, text_prompt=TEXT_PROMPT):
#     image_source, image_transformed = load_image(image_path)
#     h_img, w_img, _ = image_source.shape
#
#     # Predict boxes
#     boxes, logits, phrases = predict(
#         model=grounding_dino_model,
#         image=image_transformed,
#         caption=text_prompt,
#         box_threshold=0.35,
#         text_threshold=0.25
#     )
#
#     if len(boxes) == 0:
#         print(f"No garment detected in {image_path}")
#         return None
#
#     # Convert box to xyxy format
#     boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([w_img, h_img, w_img, h_img])
#     target_box = boxes_xyxy[0].cpu().numpy()
#     x1, y1, x2, y2 = target_box
#
#     # --- REFINEMENT LOGIC ---
#     # 1. Define specific points to guide SAM
#     center_x = (x1 + x2) / 2
#
#     # Foreground points (Top area)
#     chest_y = y1 + (y2 - y1) * 0.3
#     upper_chest_y = y1 + (y2 - y1) * 0.15
#
#     # Background points (Lower area/Trousers)
#     # We place points at the bottom corners and bottom center to exclude pants
#     waist_y = y1 + (y2 - y1) * 0.85  # Point near the bottom of the box
#     left_hip_x = x1 + (x2 - x1) * 0.2
#     right_hip_x = x1 + (x2 - x1) * 0.8
#
#     input_points = np.array([
#         [center_x, chest_y],  # Foreground
#         [center_x, upper_chest_y],  # Foreground
#         [center_x, waist_y],  # Background (exclude pants)
#         [left_hip_x, waist_y],  # Background (exclude pants)
#         [right_hip_x, waist_y]  # Background (exclude pants)
#     ])
#
#     input_labels = np.array([1, 1, 0, 0, 0])  # 1 for foreground, 0 for background
#
#     # 2. Run SAM predictor
#     sam_predictor.set_image(image_source)
#     masks, scores, _ = sam_predictor.predict(
#         point_coords=input_points,
#         point_labels=input_labels,
#         box=target_box,
#         multimask_output=True  # We take the highest score mask
#     )
#
#     # Select the mask with the highest stability score
#     mask = masks[np.argmax(scores)]
#
#     image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
#     image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
#     image_rgba[:, :, 3] = (mask * 255).astype(np.uint8)
#
#     return Image.fromarray(image_rgba)
# def extract_garment(image_path, text_prompt=TEXT_PROMPT):
#     image_source, image_transformed = load_image(image_path)
#     h_img, w_img, _ = image_source.shape
#
#     # 1. Detect both the Top and the Trousers to create a 'no-go' zone
#     # We use a combined prompt to get separate boxes
#     detection_prompt = f"{text_prompt} . trousers"
#     boxes, logits, phrases = predict(
#         model=grounding_dino_model,
#         image=image_transformed,
#         caption=detection_prompt,
#         box_threshold=0.30,
#         text_threshold=0.25
#     )
#
#     if len(boxes) == 0:
#         return None
#
#     # Convert boxes to pixel coordinates
#     boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([w_img, h_img, w_img, h_img])
#
#     # Identify which box is the top and which is trousers
#     top_box = None
#     trouser_box = None
#
#     for i, phrase in enumerate(phrases):
#         if "trouser" in phrase.lower():
#             trouser_box = boxes_xyxy[i].cpu().numpy()
#         else:
#             top_box = boxes_xyxy[i].cpu().numpy()
#
#     # Fallback if only one thing was detected
#     if top_box is None:
#         top_box = boxes_xyxy[0].cpu().numpy()
#
#     tx1, ty1, tx2, ty2 = top_box
#     bw, bh = tx2 - tx1, ty2 - ty1
#
#     # 2. CREATE SURGICAL POINTS
#     input_points = [
#         [tx1 + bw * 0.5, ty1 + bh * 0.2],  # Foreground: Neck
#         [tx1 + bw * 0.5, ty1 + bh * 0.5],  # Foreground: Mid-torso
#         [tx1 + bw * 0.5, ty2 - 10],  # Foreground: Bottom Edge (Pulls hem down)
#     ]
#     input_labels = [1, 1, 1]
#
#     # Add background points on hands to avoid skin inclusion
#     input_points.append([tx1 + bw * 0.1, ty1 + bh * 0.6])  # Left hand area
#     input_points.append([tx2 - bw * 0.1, ty1 + bh * 0.6])  # Right hand area
#     input_labels.extend([0, 0])
#
#     # 3. USE TROUSER BOX AS EXCLUSION
#     # If we found trousers, put negative points inside the top of the trouser box
#     if trouser_box is not None:
#         trx1, try1, trx2, try2 = trouser_box
#         # Place a 'barrier' of negative points at the top of the trousers
#         input_points.append([tx1 + bw * 0.5, try1 + 20])
#         input_points.append([tx1 + bw * 0.2, try1 + 20])
#         input_points.append([tx1 + bw * 0.8, try1 + 20])
#         input_labels.extend([0, 0, 0])
#
#     # 4. SAM Prediction
#     sam_predictor.set_image(image_source)
#     masks, scores, _ = sam_predictor.predict(
#         point_coords=np.array(input_points),
#         point_labels=np.array(input_labels),
#         box=top_box,
#         multimask_output=True
#     )
#
#     mask = masks[np.argmax(scores)]
#
#     # Final Cleanup: Remove small noise
#     kernel = np.ones((3, 3), np.uint8)
#     mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
#
#     # Convert to RGBA
#     image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
#     image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
#     image_rgba[:, :, 3] = (mask * 255).astype(np.uint8)
#
#     return Image.fromarray(image_rgba)

# def extract_garment(image_path, text_prompt=TEXT_PROMPT):
#     image_source, image_transformed = load_image(image_path)
#     h_img, w_img, _ = image_source.shape
#
#     # 1. Detect both the Top and the Trousers
#     detection_prompt = f"{text_prompt} . trousers"
#     boxes, logits, phrases = predict(
#         model=grounding_dino_model,
#         image=image_transformed,
#         caption=detection_prompt,
#         box_threshold=0.30,
#         text_threshold=0.25
#     )
#
#     if len(boxes) == 0:
#         return None
#
#     boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([w_img, h_img, w_img, h_img])
#     top_box, trouser_box = None, None
#
#     for i, phrase in enumerate(phrases):
#         if "trouser" in phrase.lower() or "pants" in phrase.lower():
#             trouser_box = boxes_xyxy[i].cpu().numpy()
#         else:
#             top_box = boxes_xyxy[i].cpu().numpy()
#
#     if top_box is None:
#         top_box = boxes_xyxy[0].cpu().numpy()
#
#     # --- IMPROVEMENT: BOX OVERLAP TRUNCATION ---
#     # If the trousers are detected, ensure the top box doesn't dive too deep into them
#     if trouser_box is not None:
#         # If the bottom of the top box (ty2) is below the top of the trousers (try1)
#         if top_box[3] > trouser_box[1]:
#             # Pull the top box up slightly to create a hard boundary
#             top_box[3] = trouser_box[1] + 5
#
#     tx1, ty1, tx2, ty2 = top_box
#     bw, bh = tx2 - tx1, ty2 - ty1
#
#     # 2. CREATE SURGICAL POINTS
#     input_points = [
#         [tx1 + bw * 0.5, ty1 + bh * 0.2],  # Neck
#         [tx1 + bw * 0.5, ty1 + bh * 0.5],  # Center
#         [tx1 + bw * 0.2, ty1 + bh * 0.5],  # Left Chest
#         [tx1 + bw * 0.8, ty1 + bh * 0.5],  # Right Chest
#     ]
#     input_labels = [1, 1, 1, 1]
#
#     # Background points: Arms/Side gaps (helps when close to camera)
#     input_points.extend([
#         [tx1 - 10, ty1 + bh * 0.5],       # Left outside
#         [tx2 + 10, ty1 + bh * 0.5],       # Right outside
#         [tx1 + bw * 0.1, ty1 + bh * 0.8], # Lower left arm area
#         [tx2 - bw * 0.1, ty1 + bh * 0.8], # Lower right arm area
#     ])
#     input_labels.extend([0, 0, 0, 0])
#
#     # 3. AGGRESSIVE TROUSER EXCLUSION
#     if trouser_box is not None:
#         trx1, try1, trx2, try2 = trouser_box
#         # Create a horizontal "barrier" of negative points across the waistline
#         # This tells SAM: "Anything from this line down is NOT the top"
#         waist_y = try1 + 10
#         for x_factor in [0.2, 0.4, 0.5, 0.6, 0.8]:
#             input_points.append([tx1 + bw * x_factor, waist_y])
#             input_labels.append(0)
#
#     # 4. SAM Prediction
#     sam_predictor.set_image(image_source)
#     masks, scores, _ = sam_predictor.predict(
#         point_coords=np.array(input_points),
#         point_labels=np.array(input_labels),
#         box=top_box, # SAM uses this as a hard constraint
#         multimask_output=True
#     )
#
#     mask = masks[np.argmax(scores)]
#
#     # Final Cleanup: Stronger Opening to detach small overlaps
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
#
#     # Convert to RGBA
#     image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
#     image_rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
#     image_rgba[:, :, 3] = (mask * 255).astype(np.uint8)
#
#     return Image.fromarray(image_rgba)

def extract_garment(image_path, text_prompt=TEXT_PROMPT):
    image_source, image_transformed = load_image(image_path)
    h_img, w_img, _ = image_source.shape

    # --------------------------------------------------
    # 1. Detect KURTA / TOP
    # --------------------------------------------------
    boxes_top, _, phrases_top = predict(
        model=grounding_dino_model,
        image=image_transformed,
        caption=text_prompt,
        box_threshold=0.35,
        text_threshold=0.25
    )

    if len(boxes_top) == 0:
        return None

    boxes_top = box_cxcywh_to_xyxy(boxes_top) * torch.tensor(
        [w_img, h_img, w_img, h_img]
    )

    top_box = max(
        boxes_top,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
    ).cpu().numpy()

    # --------------------------------------------------
    # 2. Detect TROUSERS (SEPARATE PASS)
    # --------------------------------------------------
    boxes_tr, _, phrases_tr = predict(
        model=grounding_dino_model,
        image=image_transformed,
        caption="trousers, pants",
        box_threshold=0.30,
        text_threshold=0.25
    )

    trouser_box = None
    if len(boxes_tr) > 0:
        boxes_tr = box_cxcywh_to_xyxy(boxes_tr) * torch.tensor(
            [w_img, h_img, w_img, h_img]
        )
        trouser_box = max(
            boxes_tr,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
        ).cpu().numpy()

    # --------------------------------------------------
    # 3. HARD CLAMP KURTA BOX (THIS IS THE FIX)
    # --------------------------------------------------
    x1, y1, x2, y2 = map(int, top_box)

    if trouser_box is not None:
        _, try1, _, _ = map(int, trouser_box)

        # 🔥 FORCE kurta to end BEFORE trousers
        if y2 > try1:
            y2 = try1 - 5

    # widen slightly (sleeves)
    pad = int(0.05 * (x2 - x1))
    x1 = max(0, x1 - pad)
    x2 = min(w_img, x2 + pad)

    target_box = np.array([x1, y1, x2, y2])

    # --------------------------------------------------
    # 4. SAM POINTS (NOW THEY ACTUALLY WORK)
    # --------------------------------------------------
    cx = (x1 + x2) // 2

    pos_points = np.array([
        [cx, y1 + 0.2 * (y2 - y1)],
        [cx, y1 + 0.5 * (y2 - y1)],
        [x1 + 0.2 * (x2 - x1), y1 + 0.5 * (y2 - y1)],
        [x1 + 0.8 * (x2 - x1), y1 + 0.5 * (y2 - y1)],
    ])

    neg_points = np.array([
        [cx, y2 + 10],
        [5, 5],
        [w_img - 5, 5],
    ])

    input_points = np.vstack([pos_points, neg_points])
    input_labels = np.array([1]*len(pos_points) + [0]*len(neg_points))

    # --------------------------------------------------
    # 5. SAM
    # --------------------------------------------------
    sam_predictor.set_image(image_source)
    masks, scores, _ = sam_predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=target_box,
        multimask_output=True
    )

    mask = masks[np.argmax(scores)].astype(np.uint8)

    # --------------------------------------------------
    # 6. Cleanup
    # --------------------------------------------------
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    image_rgba = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGBA)
    image_rgba[:, :, 3] = mask * 255

    return Image.fromarray(image_rgba)

# --- Process Folder ---
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png",".JPG"))]

print(f"Processing {len(image_files)} images...")

for img_name in tqdm(image_files):
    path = os.path.join(INPUT_FOLDER, img_name)
    try:
        result = extract_garment(path)
        if result:
            save_name = os.path.splitext(img_name)[0] + ".1.png"
            result.save(os.path.join(OUTPUT_FOLDER, save_name))
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print(f"✅ Extraction complete. Check the folder '{OUTPUT_FOLDER}'")