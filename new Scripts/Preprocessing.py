import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import Model as DinoModel

# --- CONFIGURATIONS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Segformer Config
SEGFORMER_MODEL_PATH = r"../my_custom_segformer_v4"

# 2. DINO/SAM Config
DINO_CONFIG_PATH = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS_PATH = "../GroundingDINO/groundingdino_swint_ogc.pth"
SAM_ENCODER_VERSION = "vit_h"
SAM_WEIGHTS_PATH = "../GroundingDINO/sam_vit_h_4b8939.pth"

# 3. Human Parsing Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARSER_PATH = os.path.join(SCRIPT_DIR, "..", "OOTDiffusion", "preprocess", "humanparsing")
sys.path.append(os.path.abspath(PARSER_PATH))
try:
    from run_parsing import Parsing  # type: ignore
except ImportError:
    pass


# ==========================================
# MODEL LOADERS
# ==========================================
def load_preprocessing_models():
    print("Loading Preprocessing Models (Segformer, DINO, SAM, Parser)...")
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    seg_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_PATH).to(DEVICE)
    seg_model.eval()

    dino = DinoModel(model_config_path=DINO_CONFIG_PATH, model_checkpoint_path=DINO_WEIGHTS_PATH, device=DEVICE)
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_WEIGHTS_PATH).to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    parser = Parsing(gpu_id=0)

    return {"seg_proc": processor, "seg_model": seg_model, "dino": dino, "sam": sam_predictor, "parser": parser}


# ==========================================
# IN-MEMORY PROCESSORS
# ==========================================
def get_cloth_mask(image_pil, processor, model, img_size=(512, 512), threshold=0.90, dilation=1):
    """Generates the mask using Custom Segformer for either the person OR the garment."""
    image_resized = image_pil.resize(img_size, Image.BILINEAR)
    inputs = processor(images=image_resized, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits

    logits = torch.nn.functional.interpolate(logits, size=img_size[::-1], mode="bilinear", align_corners=False)
    probs = torch.nn.functional.softmax(logits, dim=1)
    mask = (probs[0, 1].cpu().numpy() > threshold).astype(np.uint8) * 255

    if dilation > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=dilation)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100: mask[labels == i] = 0

    orig_w, orig_h = image_pil.size
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def extract_pure_garment(cloth_pil, garment_mask_np, erosion_amount=3):
    """Cuts the garment out from the background and places it on a pure white canvas."""
    cloth_bgr = cv2.cvtColor(np.array(cloth_pil), cv2.COLOR_RGB2BGR)

    kernel = np.ones((erosion_amount, erosion_amount), np.uint8)
    clean_mask = cv2.erode(garment_mask_np, kernel, iterations=1)

    white_bg = np.ones_like(cloth_bgr) * 255
    extracted_bgr = np.where(clean_mask[:, :, None] == 255, cloth_bgr, white_bg)

    return Image.fromarray(cv2.cvtColor(extracted_bgr, cv2.COLOR_BGR2RGB))


def get_parse_map(image_pil, parser):
    parsed_data, _ = parser(image_pil)
    mask_np = parsed_data.squeeze().cpu().numpy() if hasattr(parsed_data, 'cpu') else np.array(parsed_data).squeeze()
    return mask_np.astype(np.uint8)


def get_identity_mask(image_bgr, grounding_dino, sam_predictor, max_area_fraction=0.18):
    h, w = image_bgr.shape[:2]
    total_px = h * w
    final_mask = np.zeros((h, w), dtype=np.uint8)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detections = grounding_dino.predict_with_classes(image=image_bgr, classes=["face", "hand", "neck"],
                                                     box_threshold=0.30, text_threshold=0.25)
    if len(detections.xyxy) == 0: return final_mask

    filtered_boxes = [box for box in detections.xyxy if
                      ((box[2] - box[0]) * (box[3] - box[1])) / total_px <= max_area_fraction]
    if not filtered_boxes: return final_mask

    sam_predictor.set_image(image_rgb)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
        torch.tensor(np.array(filtered_boxes), device=sam_predictor.device), image_rgb.shape[:2])
    masks, _, _ = sam_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes,
                                              multimask_output=False)

    for mask_tensor in masks:
        mask_np = mask_tensor.squeeze().cpu().numpy()
        if mask_np.sum() / total_px <= max_area_fraction: final_mask[mask_np > 0] = 255

    return final_mask


def build_agnostic(image_bgr, cloth_mask, parse_map, identity_mask):
    h, w = image_bgr.shape[:2]
    close_kernel = np.ones((15, 15), np.uint8)
    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, close_kernel)
    contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cloth_mask, contours, -1, 255, thickness=cv2.FILLED)

    arm_mask = np.zeros((h, w), dtype=np.uint8)
    arm_mask[parse_map == 14] = 255
    arm_mask[parse_map == 15] = 255

    dilate_kernel = np.ones((11, 11), np.uint8)
    canvas_mask = cv2.bitwise_or(cloth_mask, arm_mask)
    canvas_mask = cv2.morphologyEx(canvas_mask, cv2.MORPH_CLOSE, close_kernel)
    _, canvas_mask = cv2.threshold(canvas_mask, 127, 255, cv2.THRESH_BINARY)
    canvas_mask = cv2.dilate(canvas_mask, dilate_kernel, iterations=1)
    canvas_mask[identity_mask == 255] = 0

    tight_mask = cv2.dilate(cloth_mask.copy(), dilate_kernel, iterations=1)
    tight_mask[identity_mask == 255] = 0

    agnostic_img = image_bgr.copy()
    agnostic_img[canvas_mask == 255] = 128

    return Image.fromarray(cv2.cvtColor(agnostic_img, cv2.COLOR_BGR2RGB)), Image.fromarray(tight_mask)