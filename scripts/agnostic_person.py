# import cv2
# import numpy as np
# import os
# from tqdm import tqdm
#
# images = r"../raw images/images-resized"
# seg = r"../raw images/images-parse"
# clothpath = r"../raw images/cloth-mask"
# output_dir = r"../raw images/agnostic"
# os.makedirs(output_dir, exist_ok=True)
#
# source = [f for f in os.listdir(images) if f.lower() .endswith(".jpg")]
#
# for img in tqdm(source):
#     image = os.path.join(images,img)
#     parse = os.path.join(seg, img)
#     cloth = os.path.join(clothpath,img)
#     parse = parse.replace(".jpg", ".png")
#     cloth = cloth.replace(".jpg", ".1.png")
#     person = cv2.imread(image)
#     segment = cv2.imread(parse,0)
#     cloth_mask = cv2.imread(cloth, 0)
#
#     KURTA = 4
#     DUPATTA = 17
#     LEFT_ARM = 14
#     RIGHT_ARM = 15
#
#
#     _, cloth_mask = cv2.threshold(cloth_mask, 127, 255, cv2.THRESH_BINARY)
#
#     seg_mask = np.zeros_like(segment, dtype=np.uint8)
#     seg_mask[segment == KURTA] = 255
#     seg_mask[segment == DUPATTA] = 255
#     seg_mask[segment == LEFT_ARM] = 0
#     seg_mask[segment == RIGHT_ARM] = 0
#
#     final_mask = cv2.bitwise_or(seg_mask, cloth_mask)
#
#     kernel = np.ones((3, 3), np.uint8)
#     final_mask = cv2.dilate(final_mask, kernel, iterations=1)
#
#     final_mask[segment == LEFT_ARM] = 0
#     final_mask[segment == RIGHT_ARM] = 0
#
#     inv_mask = cv2.bitwise_not(final_mask)
#     hollow_person = cv2.bitwise_and(person, person, mask=inv_mask)
#
#     neutral = np.zeros_like(person)
#     neutral[:] = (128, 128, 128)
#     fill = cv2.bitwise_and(neutral, neutral, mask=final_mask)
#
#     agnostic = cv2.add(hollow_person, fill)
#
#     out_path = os.path.join(output_dir, img)
#     cv2.imwrite(out_path, agnostic)
#
#
# # import cv2
# # import numpy as np
# # import os
# # import torch
# # from tqdm import tqdm
# # from segment_anything import sam_model_registry, SamPredictor
# #
# # # --- CONFIGURATION ---
# # images_dir = r"../raw images/images-resized"
# # seg_dir = r"../raw images/images-parse"
# # output_dir = r"../raw images/agnostic"
# #
# # SAM_WTS = r"../GroundingDINO/sam_vit_h_4b8939.pth"
# # sam_model = sam_model_registry["vit_h"](checkpoint=SAM_WTS).to("cuda")
# #
# # os.makedirs(output_dir, exist_ok=True)
# #
# # # --- INITIALIZE SAM MODEL ---
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # print(f"Loading SAM model ({sam_model}) to {device}...")
# #
# # sam = sam_model_registry[sam_model](checkpoint=SAM_WTS)
# # sam.to(device=device)
# # predictor = SamPredictor(sam)
# #
# # # --- LABELS ---
# # KURTA = 4
# # DUPATTA = 17
# # LEFT_ARM = 14
# # RIGHT_ARM = 15
# #
# # # Get list of images
# # source = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg"))]
# #
# # print("Starting generation...")
# #
# # for img in tqdm(source):
# #     # 1. Setup Paths
# #     image_path = os.path.join(images_dir, img)
# #     # Parse maps .jpg -> .png
# #     parse_path = os.path.join(seg_dir, img.replace(".jpg", ".png").replace(".jpeg", ".png"))
# #
# #     # 2. Load Images
# #     person = cv2.imread(image_path)
# #     segment = cv2.imread(parse_path, 0)
# #
# #     if person is None or segment is None:
# #         print(f"Warning: Data missing for {img}, skipping.")
# #         continue
# #
# #     # 3. Preprocess for SAM (Requires RGB)
# #     person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
# #
# #     # 4. Create Initial Prompt Mask (SegFormer)
# #     # We combine Kurta and Dupatta, but explicitly EXCLUDE arms so the box doesn't cover them if possible
# #     cloth_mask = np.zeros_like(segment, dtype=np.uint8)
# #     cloth_mask[(segment == KURTA) | (segment == DUPATTA)] = 255
# #     cloth_mask[(segment == LEFT_ARM) | (segment == RIGHT_ARM)] = 0
# #
# #     # Check if clothing exists in this image
# #     if cv2.countNonZero(cloth_mask) == 0:
# #         # Save original if no clothing found
# #         cv2.imwrite(os.path.join(output_dir, img), person)
# #         continue
# #
# #     # 5. Get Bounding Box from SegFormer Mask
# #     contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #
# #     if not contours:
# #         cv2.imwrite(os.path.join(output_dir, img), person)
# #         continue
# #
# #     # Combine all contours to get one big box around the upper body/clothing
# #     all_points = np.concatenate(contours)
# #     x, y, w, h = cv2.boundingRect(all_points)
# #     input_box = np.array([x, y, x + w, y + h])  # Format: [x_min, y_min, x_max, y_max]
# #
# #     # 6. Run SAM Prediction
# #     predictor.set_image(person_rgb)
# #
# #     masks, _, _ = predictor.predict(
# #         point_coords=None,
# #         point_labels=None,
# #         box=input_box[None, :],  # Add batch dim
# #         multimask_output=False
# #     )
# #
# #     # SAM returns the refined mask
# #     refined_mask = (masks[0].astype(np.uint8) * 255)
# #
# #     # 7. Safety: Ensure arms are masked out
# #     # (SAM might expand into arms, so we force them back to 0 using the parse)
# #     refined_mask[(segment == LEFT_ARM) | (segment == RIGHT_ARM)] = 0
# #
# #     # 8. Create Agnostic Image
# #     inv_mask = cv2.bitwise_not(refined_mask)
# #     hollow_person = cv2.bitwise_and(person, person, mask=inv_mask)  # Keep background/skin
# #
# #     # Create gray fill
# #     neutral = np.full_like(person, 128)
# #     fill = cv2.bitwise_and(neutral, neutral, mask=refined_mask)
# #
# #     agnostic = cv2.add(hollow_person, fill)
# #
# #     # 9. Save
# #     cv2.imwrite(os.path.join(output_dir, img), agnostic)
# #
# # print("Done.")
#
#
# # import cv2
# # import numpy as np
# # import os
# # import torch
# # from tqdm import tqdm
# # from segment_anything import sam_model_registry, SamPredictor
# #
# # # --- CONFIGURATION ---
# # images_dir = r"../raw images/images-resized"
# # seg_dir = r"../raw images/images-parse"
# # output_dir = r"../raw images/agnostic"
# #
# # # SAM Model Setup
# # SAM_WTS = r"../GroundingDINO/sam_vit_h_4b8939.pth"
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model_type = "vit_h"
# #
# # print(f"Loading SAM model ({device})...")
# # sam_model = sam_model_registry[model_type](checkpoint=SAM_WTS).to(device)
# # predictor = SamPredictor(sam_model)
# #
# # os.makedirs(output_dir, exist_ok=True)
# #
# # # --- YOUR SPECIFIC PALETTE INDICES ---
# # # Items to REPLACE with Grey (The prompting mask)
# # # We include 6, 7, 5, 9 just in case the model used them for the bottom of the dress
# # LABELS_TO_MASK = [
# #     4,  # Kurta / Upper Garment
# #     17,  # Dupatta
# #     6,  # Lower Garment (often covers the bottom of the kurta)
# #     7,  # Dress (common misclassification for long kurtas)
# #     5,  # Skirt (common misclassification for bottom of kurta)
# # ]
# #
# # # Items to PROTECT (Keep Original Pixels)
# # # We must strictly exclude these from the gray mask
# # PROTECTED_LABELS = [
# #     14,  # Left Arm
# #     15,  # Right Arm
# #     11,  # Face / Neck (CRITICAL FIX: used to be 13)
# #     2,  # Hair
# #     9,  # Left Shoe (Optional, keep if visible)
# #     10,  # Right Shoe (Optional, keep if visible)
# # ]
# #
# # source_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
# # print(f"Found {len(source_files)} images. Starting processing...")
# #
# # for img_name in tqdm(source_files):
# #     # 1. Robust Path Finding
# #     name_root, _ = os.path.splitext(img_name)
# #     img_path = os.path.join(images_dir, img_name)
# #
# #     # Try finding the parse file (checking .png, .jpg, etc)
# #     parse_path = None
# #     for ext in ['.png', '.PNG', '.jpg', '.jpeg']:
# #         candidate = os.path.join(seg_dir, name_root + ext)
# #         if os.path.exists(candidate):
# #             parse_path = candidate
# #             break
# #
# #     if parse_path is None:
# #         # print(f"Skipping {img_name}: Parse file missing.")
# #         continue
# #
# #     # 2. Load Data
# #     person = cv2.imread(img_path)
# #     seg = cv2.imread(parse_path, 0)  # Loads indices (0-17)
# #
# #     if person is None or seg is None:
# #         continue
# #
# #     # Resize seg if dimensions mismatch
# #     if seg.shape != person.shape[:2]:
# #         seg = cv2.resize(seg, (person.shape[1], person.shape[0]), interpolation=cv2.INTER_NEAREST)
# #
# #     # 3. Create "Prompt Mask" for SAM
# #     # Combine ALL clothing parts (Kurta + Dupatta + Lower + Dress misclassifications)
# #     prompt_mask = np.zeros_like(seg, dtype=np.uint8)
# #     for label_id in LABELS_TO_MASK:
# #         prompt_mask[seg == label_id] = 255
# #
# #     # Strictly remove arms/face from the prompt mask
# #     # This ensures the bounding box doesn't accidentally include the hand
# #     for label_id in PROTECTED_LABELS:
# #         prompt_mask[seg == label_id] = 0
# #
# #     # 4. Get Bounding Box
# #     contours, _ = cv2.findContours(prompt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #
# #     if not contours:
# #         # If no clothing detected, save original
# #         cv2.imwrite(os.path.join(output_dir, img_name), person)
# #         continue
# #
# #     # Combine all contours to get the full bounding box
# #     all_points = np.concatenate(contours)
# #     x, y, w, h = cv2.boundingRect(all_points)
# #
# #     # Add Padding (Important for full coverage)
# #     pad = 20
# #     h_img, w_img = seg.shape
# #     x1 = max(0, x - pad)
# #     y1 = max(0, y - pad)
# #     x2 = min(w_img, x + w + pad)
# #     y2 = min(h_img, y + h + pad)
# #     input_box = np.array([x1, y1, x2, y2])
# #
# #     # 5. Run SAM
# #     person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
# #     predictor.set_image(person_rgb)
# #
# #     masks, _, _ = predictor.predict(
# #         point_coords=None,
# #         point_labels=None,
# #         box=input_box[None, :],
# #         multimask_output=False
# #     )
# #
# #     refined_mask = (masks[0].astype(np.uint8) * 255)
# #
# #     # 6. Post-Processing (The Protection Step)
# #     # SAM might overflow onto the arms or face. We use the SegFormer parse to cut it back.
# #     for label_id in PROTECTED_LABELS:
# #         refined_mask[seg == label_id] = 0
# #
# #     # Dilate slightly to ensure seams are covered
# #     kernel = np.ones((5, 5), np.uint8)
# #     refined_mask = cv2.dilate(refined_mask, kernel, iterations=1)
# #
# #     # Re-apply protection after dilation (Safety first!)
# #     for label_id in PROTECTED_LABELS:
# #         refined_mask[seg == label_id] = 0
# #
# #     # 7. Generate Agnostic Image
# #     inv_mask = cv2.bitwise_not(refined_mask)
# #     hollow_person = cv2.bitwise_and(person, person, mask=inv_mask)
# #
# #     # Neutral Grey Fill
# #     neutral = np.full_like(person, 128)
# #     fill = cv2.bitwise_and(neutral, neutral, mask=refined_mask)
# #
# #     agnostic = cv2.add(hollow_person, fill)
# #
# #     # Save
# #     cv2.imwrite(os.path.join(output_dir, img_name), agnostic)
# #
# # print("Batch processing complete.")