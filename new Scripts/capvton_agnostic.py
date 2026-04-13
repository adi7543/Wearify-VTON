# """
# CaP-VTON: Agnostic Person Generator (Upper Body)
# =================================================
# Assumes Step 1 (Human Parsing) is already done via OOTDiffusion ONNX parser.
# This script handles:
#   Step 2 - Build Agnostic Mask   (from your parsed label PNG)
#   Step 3 - Skin Inpainting       (Stable Diffusion inpainting)
#
# ATR Label Map (what your parser outputs):
#   0  = Background       9  = Left-shoe
#   1  = Hat              10 = Right-shoe
#   2  = Hair             11 = Face
#   3  = Sunglasses       12 = Left-leg
#   4  = Upper-clothes    13 = Right-leg
#   5  = Skirt            14 = Left-arm
#   6  = Pants            15 = Right-arm
#   7  = Dress            16 = Bag
#   8  = Belt             17 = Scarf
#
# Install:
#   pip install torch diffusers accelerate pillow numpy opencv-python transformers
# """
#
# import os
# import numpy as np
# import cv2
# from PIL import Image
# import torch
# from diffusers import StableDiffusionInpaintPipeline
#
# # ── Config ──────────────────────────────────────────────────────────────────
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# PERSON_IMAGE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "images"))           # original person photo
# PARSE_LABEL_PATH  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "parsed_labels"))     # your ONNX parser output (raw label IDs)
#
# OUTPUT_MASK_PATH      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "agnostic_mask"))
# OUTPUT_AGNOSTIC_PATH  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "agnostic"))
# OUTPUT_PARSE_VIZ_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "Viz"))   # colorized parse for debugging
#
# DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
# SD_MODEL = "runwayml/stable-diffusion-inpainting"
#
# # Labels to MASK (replace with skin)
# UPPER_CLOTHING_LABELS = [4]       # upper-clothes
# ARM_LABELS            = [14, 15]  # left-arm, right-arm
#
# # Labels to NEVER mask (always preserve)
# PRESERVE_LABELS = [0, 2, 11]     # background, hair, face
#
#
# # ── Colormap for visualizing parse output ────────────────────────────────────
#
# ATR_COLORMAP = {
#     0:  (0,   0,   0),    # Background   - black
#     1:  (128, 0,   0),    # Hat          - dark red
#     2:  (255, 165, 0),    # Hair         - orange
#     3:  (128, 128, 0),    # Sunglasses   - olive
#     4:  (0,   128, 0),    # Upper-clothes- green   <- this is what we mask
#     5:  (128, 0,  128),   # Skirt        - purple
#     6:  (0,   128, 128),  # Pants        - teal
#     7:  (64,  0,  128),   # Dress        - dark purple
#     8:  (192, 0,   0),    # Belt         - crimson
#     9:  (64, 128,   0),   # Left-shoe    - yellow-green
#     10: (192, 128,  0),   # Right-shoe   - dark yellow
#     11: (255, 220, 177),  # Face         - skin tone
#     12: (0,   64, 128),   # Left-leg     - steel blue
#     13: (128, 64,   0),   # Right-leg    - brown
#     14: (0,  192,   0),   # Left-arm     - bright green  <- masked if mask_arms=True
#     15: (0,  64,    0),   # Right-arm    - dark green    <- masked if mask_arms=True
#     16: (0,   0,  192),   # Bag          - blue
#     17: (64,  64,  64),   # Scarf        - gray
# }
#
#
# # ── Utilities ────────────────────────────────────────────────────────────────
#
# def load_parse_label(path: str) -> np.ndarray:
#     """Load the raw label map (uint8, values 0-17)."""
#     label_map = np.array(Image.open(path).convert("L"))
#     unique = np.unique(label_map).tolist()
#     print(f"[Parse] Loaded label map: shape={label_map.shape}, unique labels={unique}")
#
#     # Auto-detect if labels were scaled (e.g. multiplied by 10 or 15)
#     if max(unique) > 17:
#         print(f"  ⚠️  Max label value is {max(unique)} — labels may be scaled.")
#         print(f"  Attempting to detect scale factor...")
#         for scale in [10, 15, 16, 255]:
#             scaled = label_map // scale
#             if scaled.max() <= 17:
#                 print(f"  Dividing by {scale} to recover class IDs.")
#                 label_map = scaled
#                 break
#
#     return label_map
#
#
# def visualize_parse(label_map: np.ndarray, save_path: str):
#     """Save a colorized version of the label map for debugging."""
#     h, w = label_map.shape
#     vis = np.zeros((h, w, 3), dtype=np.uint8)
#     for label_id, color in ATR_COLORMAP.items():
#         vis[label_map == label_id] = color
#     Image.fromarray(vis).save(save_path)
#     print(f"[Parse] Visualization saved -> {save_path}")
#
#
# # ── Step 2: Build Agnostic Mask ──────────────────────────────────────────────
#
# def build_agnostic_mask(
#     label_map: np.ndarray,
#     mask_arms: bool = True,
#     dilate_px: int = 15,
# ) -> np.ndarray:
#     """
#     Returns a binary mask (uint8): 255 = clothing region, 0 = keep as-is.
#
#     Args:
#         label_map  : H x W array of ATR label IDs from your ONNX parser
#         mask_arms  : also mask arm pixels (important for short-sleeve try-on)
#         dilate_px  : dilation in pixels to close gaps at clothing edges
#     """
#     print("\n[Step 2] Building agnostic mask...")
#
#     labels_to_mask = UPPER_CLOTHING_LABELS.copy()
#     if mask_arms:
#         labels_to_mask += ARM_LABELS
#         print(f"  Masking labels: {labels_to_mask} (upper-clothes + arms)")
#     else:
#         print(f"  Masking labels: {labels_to_mask} (upper-clothes only)")
#
#     # Build raw mask
#     mask = np.zeros(label_map.shape, dtype=np.uint8)
#     for label in labels_to_mask:
#         pixels = (label_map == label).sum()
#         print(f"  Label {label:2d} -> {pixels} pixels")
#         mask[label_map == label] = 255
#
#     if mask.sum() == 0:
#         print("\n  WARNING: No clothing pixels found!")
#         print("  Check parse_visualization.png to see what was detected.")
#         return mask
#
#     # Dilate to capture clothing boundary fringe pixels
#     if dilate_px > 0:
#         kernel = cv2.getStructuringElement(
#             cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)
#         )
#         mask = cv2.dilate(mask, kernel, iterations=1)
#
#     # Always preserve face, hair, background
#     for label in PRESERVE_LABELS:
#         mask[label_map == label] = 0
#
#     print(f"  Masked pixels: {(mask == 255).sum():,} / {mask.size:,} "
#           f"({100*(mask==255).sum()/mask.size:.1f}%)")
#     return mask
#
#
# # ── Step 3: Skin Inpainting ───────────────────────────────────────────────────
#
# def inpaint_skin(
#     person_image: Image.Image,
#     mask: np.ndarray,
#     prompt: str = (
#         "photorealistic bare human skin, natural arms, shoulders, "
#         "seamless body, no clothing, studio lighting"
#     ),
#     negative_prompt: str = (
#         "clothing, shirt, fabric, garment, wrinkles, pattern, "
#         "blurry, deformed, extra limbs, artifacts"
#     ),
#     num_inference_steps: int = 30,
#     guidance_scale: float = 7.5,
#     seed: int = 42,
# ) -> Image.Image:
#     """
#     CaP-VTON 'Generate Skin' step:
#     Inpaints natural bare skin over the masked clothing region using SD.
#     """
#     print("\n[Step 3] Running skin inpainting (Generate Skin)...")
#     print(f"  Model  : {SD_MODEL}")
#     print(f"  Device : {DEVICE}")
#
#     pipe = StableDiffusionInpaintPipeline.from_pretrained(
#         SD_MODEL,
#         torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
#     ).to(DEVICE)
#     pipe.set_progress_bar_config(desc="  Inpainting")
#
#     mask_pil = Image.fromarray(mask)
#     orig_w, orig_h = person_image.size
#
#     # SD works best at 512x512 -- composite back at full res
#     person_512 = person_image.resize((512, 512), Image.LANCZOS)
#     mask_512   = mask_pil.resize((512, 512), Image.NEAREST)
#
#     generator = torch.Generator(device=DEVICE).manual_seed(seed)
#     result_512 = pipe(
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         image=person_512,
#         mask_image=mask_512,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#         generator=generator,
#     ).images[0]
#
#     result = result_512.resize((orig_w, orig_h), Image.LANCZOS)
#     print("  Done.")
#     return result
#
#
# # ── Composite ────────────────────────────────────────────────────────────────
#
# def composite(
#     original: Image.Image,
#     inpainted: Image.Image,
#     mask: np.ndarray,
# ) -> Image.Image:
#     """
#     Paste inpainted skin ONLY in the masked region.
#     Everything outside the mask stays pixel-perfect from the original.
#     """
#     orig_np   = np.array(original).astype(np.float32)
#     inp_np    = np.array(inpainted).astype(np.float32)
#
#     # Feather mask edges to avoid hard seams
#     mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)
#     alpha     = (mask_blur / 255.0)[..., np.newaxis]
#
#     blended = (inp_np * alpha + orig_np * (1 - alpha)).clip(0, 255).astype(np.uint8)
#     return Image.fromarray(blended)
#
#
# # ── Main ─────────────────────────────────────────────────────────────────────
#
# def generate_agnostic_person(
#     person_image_path: str = PERSON_IMAGE_PATH,
#     parse_label_path:  str = PARSE_LABEL_PATH,
#     mask_arms: bool = True,
#     dilate_px: int = 15,
# ):
#     print("=" * 50)
#     print("  CaP-VTON Agnostic Person Generator")
#     print("=" * 50)
#
#     person_image = Image.open(person_image_path).convert("RGB")
#     label_map    = load_parse_label(parse_label_path)
#
#     # Resize label map to match image if needed
#     img_w, img_h = person_image.size
#     map_h, map_w = label_map.shape
#     if (img_h, img_w) != (map_h, map_w):
#         print(f"\n  Size mismatch: image={img_w}x{img_h}, label_map={map_w}x{map_h}")
#         print("  Resizing label map to match image...")
#         label_map = np.array(
#             Image.fromarray(label_map).resize((img_w, img_h), Image.NEAREST)
#         )
#
#     # Always save colorized parse for debugging
#     visualize_parse(label_map, OUTPUT_PARSE_VIZ_PATH)
#
#     # Step 2
#     mask = build_agnostic_mask(label_map, mask_arms=mask_arms, dilate_px=dilate_px)
#     Image.fromarray(mask).save(OUTPUT_MASK_PATH)
#     print(f"  Mask saved -> {OUTPUT_MASK_PATH}")
#
#     if mask.sum() == 0:
#         print("\nStopping: mask is empty. Fix label IDs and re-run.")
#         return
#
#     # Step 3
#     inpainted = inpaint_skin(person_image, mask)
#
#     agnostic = composite(person_image, inpainted, mask)
#     agnostic.save(OUTPUT_AGNOSTIC_PATH)
#     print(f"\nDone! Agnostic person saved -> {OUTPUT_AGNOSTIC_PATH}")
#
#     return agnostic, mask
#
#
# if __name__ == "__main__":
#     generate_agnostic_person(
#         person_image_path = PERSON_IMAGE_PATH,
#         parse_label_path  = PARSE_LABEL_PATH,
#         mask_arms         = True,   # False = only mask shirt body, keep arms
#         dilate_px         = 15,
#     )





"""
CaP-VTON: Agnostic Person Generator (Batch Version)
"""

import os
import numpy as np
import cv2
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PERSON_IMAGE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "images"))
PARSE_LABEL_PATH  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "parsed_labels"))

OUTPUT_MASK_PATH      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "agnostic_mask"))
OUTPUT_AGNOSTIC_PATH  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "agnostic"))
OUTPUT_PARSE_VIZ_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "Viz"))

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SD_MODEL = "runwayml/stable-diffusion-inpainting"

UPPER_CLOTHING_LABELS = [4]
ARM_LABELS            = [14, 15]
PRESERVE_LABELS       = [0, 2, 11]

ATR_COLORMAP = {
    0:(0,0,0),1:(128,0,0),2:(255,165,0),3:(128,128,0),4:(0,128,0),
    5:(128,0,128),6:(0,128,128),7:(64,0,128),8:(192,0,0),9:(64,128,0),
    10:(192,128,0),11:(255,220,177),12:(0,64,128),13:(128,64,0),
    14:(0,192,0),15:(0,64,0),16:(0,0,192),17:(64,64,64),
}

# ── Utilities ────────────────────────────────────────────────────────────────

def load_parse_label(path):
    label_map = np.array(Image.open(path).convert("L"))
    unique = np.unique(label_map).tolist()

    if max(unique) > 17:
        for scale in [10, 15, 16, 255]:
            scaled = label_map // scale
            if scaled.max() <= 17:
                label_map = scaled
                break

    return label_map


def visualize_parse(label_map, save_path):
    h, w = label_map.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in ATR_COLORMAP.items():
        vis[label_map == label_id] = color
    Image.fromarray(vis).save(save_path)


# ── Step 2 ──────────────────────────────────────────────────────────────────

def build_agnostic_mask(label_map, mask_arms=True, dilate_px=15):
    labels_to_mask = UPPER_CLOTHING_LABELS.copy()
    if mask_arms:
        labels_to_mask += ARM_LABELS

    mask = np.zeros(label_map.shape, dtype=np.uint8)
    for label in labels_to_mask:
        mask[label_map == label] = 255

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, kernel, iterations=1)

    for label in PRESERVE_LABELS:
        mask[label_map == label] = 0

    return mask


# ── Step 3 ──────────────────────────────────────────────────────────────────

def inpaint_skin(pipe, person_image, mask):
    mask_pil = Image.fromarray(mask)

    orig_w, orig_h = person_image.size
    person_512 = person_image.resize((512, 512), Image.LANCZOS)
    mask_512   = mask_pil.resize((512, 512), Image.NEAREST)

    generator = torch.Generator(device=DEVICE).manual_seed(42)

    result_512 = pipe(
        prompt="photorealistic bare human skin, natural arms, shoulders",
        negative_prompt="clothing, shirt, fabric, blurry, artifacts",
        image=person_512,
        mask_image=mask_512,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

    return result_512.resize((orig_w, orig_h), Image.LANCZOS)


# ── Composite ────────────────────────────────────────────────────────────────

def composite(original, inpainted, mask):
    orig_np = np.array(original).astype(np.float32)
    inp_np  = np.array(inpainted).astype(np.float32)

    mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)
    alpha = (mask_blur / 255.0)[..., np.newaxis]

    blended = (inp_np * alpha + orig_np * (1 - alpha)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ── MAIN (BATCH) ─────────────────────────────────────────────────────────────

def generate_agnostic_person_batch():
    print("=" * 50)
    print("CaP-VTON Batch Processing")
    print("=" * 50)

    os.makedirs(OUTPUT_MASK_PATH, exist_ok=True)
    os.makedirs(OUTPUT_AGNOSTIC_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PARSE_VIZ_PATH, exist_ok=True)

    # 🔥 Load model ONCE (important)
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    image_files = [f for f in os.listdir(PERSON_IMAGE_PATH)
                   if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    print(f"Found {len(image_files)} images")

    for file in image_files:
        try:
            print(f"\nProcessing: {file}")

            img_path = os.path.join(PERSON_IMAGE_PATH, file)

            # assume labels are PNG
            label_name = os.path.splitext(file)[0] + ".png"
            label_path = os.path.join(PARSE_LABEL_PATH, label_name)

            if not os.path.exists(label_path):
                print("  Missing label, skipping")
                continue

            person_image = Image.open(img_path).convert("RGB")
            label_map = load_parse_label(label_path)

            # resize label map if needed
            if person_image.size[::-1] != label_map.shape:
                label_map = np.array(
                    Image.fromarray(label_map).resize(person_image.size, Image.NEAREST)
                )

            # save visualization
            visualize_parse(label_map, os.path.join(OUTPUT_PARSE_VIZ_PATH, file))

            mask = build_agnostic_mask(label_map)
            Image.fromarray(mask).save(os.path.join(OUTPUT_MASK_PATH, file))

            if mask.sum() == 0:
                print("  Empty mask, skipping")
                continue

            inpainted = inpaint_skin(pipe, person_image, mask)
            agnostic = composite(person_image, inpainted, mask)

            out_path = os.path.join(OUTPUT_AGNOSTIC_PATH, file)
            agnostic.save(out_path)

            print("  Done")

        except Exception as e:
            print(f"  Error: {e}")


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_agnostic_person_batch()