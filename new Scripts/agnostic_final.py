# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# from ultralytics import YOLO
#
# # ==========================================
# # 1. PATH SETUP
# # ==========================================
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#
# IMG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "images"))
# CLOTH_MASK_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "ref_cloth_mask"))
# PARSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "parsed_labels"))
# IDENTITY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "identity_masks"))
#
# MASK_OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "agnostic_mask"))
# AGNOSTIC_OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "agnostic"))
#
# os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
# os.makedirs(AGNOSTIC_OUTPUT_DIR, exist_ok=True)
#
# # ==========================================
# # 2. CONFIGURATION
# # ==========================================
# ARM_IDS = [14, 15]
# TROUSER_IDS = [9]
#
# # Massive Brute-Force Arms to obliterate baggy sleeves
# BICEP_THICKNESS = 40
# FOREARM_THICKNESS = 40
#
# # YOLO Drape & Hand Zone Settings
# HAND_ZONE_RADIUS = 75
# KURTA_EXPAND_PADDING = 20
# KURTA_DROP_BELOW_KNEE = 55
# FEATHER_RADIUS = 11
#
# # Gravity Kernel (Breathing Room)
# SHIRT_EXPAND_KERNEL = np.array([
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1]
# ], np.uint8)
# SHIRT_EXPAND_ITER = 4
#
# print("Loading lightweight YOLO for Geometry...")
# pose_model = YOLO("yolov8n-pose.pt", verbose=False)
#
#
# def build_final_agnostic():
#     img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     print(f"Starting Blazing-Fast Agnostic Builder for {len(img_files)} images...")
#
#     for img_name in tqdm(img_files):
#         base_name = os.path.splitext(img_name)[0]
#
#         img_path = os.path.join(IMG_DIR, img_name)
#         cloth_path = os.path.join(CLOTH_MASK_DIR, f"{base_name}.png")
#         parse_path = os.path.join(PARSED_DIR, f"{base_name}.png")
#         identity_path = os.path.join(IDENTITY_DIR, f"{base_name}.png")
#
#         # Skip if Phase 1 hasn't processed this image yet
#         if not all(os.path.exists(p) for p in [cloth_path, parse_path, identity_path]):
#             continue
#
#         # ==========================================
#         # 1. BUILD THE GREY CANVAS (Clothes + Arms)
#         # ==========================================
#         img = cv2.imread(img_path)
#         h, w = img.shape[:2]
#
#         # A. Base Shirt
#         cloth_mask = cv2.imread(cloth_path, cv2.IMREAD_GRAYSCALE)
#         _, cloth_mask = cv2.threshold(cloth_mask, 127, 255, cv2.THRESH_BINARY)
#         cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
#         contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(cloth_mask, contours, -1, (255), thickness=cv2.FILLED)
#
#         cloth_mask_raw = cloth_mask.copy()
#
#         cloth_mask = cv2.dilate(cloth_mask, SHIRT_EXPAND_KERNEL, iterations=SHIRT_EXPAND_ITER)
#
#         # B. Parser Arms
#         parse_map = cv2.imread(parse_path, cv2.IMREAD_GRAYSCALE)
#         if len(parse_map.shape) > 2:
#             parse_map = parse_map[:, :, 0]
#         arm_mask = np.zeros((h, w), dtype=np.uint8)
#         for arm_id in ARM_IDS:
#             arm_mask[parse_map == arm_id] = 255
#
#         master_mask = cv2.bitwise_or(cloth_mask, arm_mask)
#
#         # ==========================================
#         # 2. YOLO GEOMETRY (Kurta Drape & Thick Arms)
#         # ==========================================
#         results = pose_model(img, verbose=False)
#         valid_id_zone = np.zeros((h, w), dtype=np.uint8)  # The Cookie Cutter
#
#         try:
#             kpts = results[0].keypoints.xy[0].cpu().numpy()
#
#             # --- SKELETAL ARMS ---
#             # Guarantees the arms are painted grey, obliterating baggy sleeves
#             if len(kpts) > 9:
#                 s_l, e_l, w_l = kpts[5], kpts[7], kpts[9]
#                 if s_l[0] != 0 and e_l[0] != 0:
#                     cv2.line(master_mask, (int(s_l[0]), int(s_l[1])), (int(e_l[0]), int(e_l[1])), 255, BICEP_THICKNESS)
#                 if e_l[0] != 0 and w_l[0] != 0:
#                     cv2.line(master_mask, (int(e_l[0]), int(e_l[1])), (int(w_l[0]), int(w_l[1])), 255,
#                              FOREARM_THICKNESS)
#
#             if len(kpts) > 10:
#                 s_r, e_r, w_r = kpts[6], kpts[8], kpts[10]
#                 if s_r[0] != 0 and e_r[0] != 0:
#                     cv2.line(master_mask, (int(s_r[0]), int(s_r[1])), (int(e_r[0]), int(e_r[1])), 255, BICEP_THICKNESS)
#                 if e_r[0] != 0 and w_r[0] != 0:
#                     cv2.line(master_mask, (int(e_r[0]), int(e_r[1])), (int(w_r[0]), int(w_r[1])), 255,
#                              FOREARM_THICKNESS)
#
#             # --- KURTA DRAPE ---
#             hip_xs, hip_ys, knee_ys = [], [], []
#             for idx in [11, 12]:
#                 if len(kpts) > idx and kpts[idx][1] != 0:
#                     hip_xs.append(int(kpts[idx][0]))
#                     hip_ys.append(int(kpts[idx][1]))
#             for idx in [13, 14]:
#                 if len(kpts) > idx and kpts[idx][1] != 0:
#                     knee_ys.append(int(kpts[idx][1]))
#
#             if len(hip_xs) == 2 and knee_ys:
#                 target_bottom_y = min(h, max(knee_ys) + KURTA_DROP_BELOW_KNEE)
#                 hip_y = int(np.mean(hip_ys))
#                 body_left, body_right = min(hip_xs), max(hip_xs)
#
#                 if (body_right - body_left) < 80:
#                     center = (body_left + body_right) // 2
#                     body_left, body_right = center - 40, center + 40
#
#                 pts = np.array([
#                     [max(0, body_left - KURTA_EXPAND_PADDING), hip_y - 30],
#                     [min(w, body_right + KURTA_EXPAND_PADDING), hip_y - 30],
#                     [min(w, body_right + (KURTA_EXPAND_PADDING + 20)), target_bottom_y],
#                     [max(0, body_left - (KURTA_EXPAND_PADDING + 20)), target_bottom_y]
#                 ], np.int32)
#                 cv2.fillPoly(master_mask, [pts], 255)
#
#             # --- THE COOKIE CUTTER (Protect Hands & Face) ---
#             # Keep everything above the shoulders
#             shoulder_ys = [kpts[5][1] if len(kpts) > 5 else 0, kpts[6][1] if len(kpts) > 6 else 0]
#             valid_shoulder_ys = [y for y in shoulder_ys if y != 0]
#
#             if valid_shoulder_ys:
#                 neck_line = int(min(valid_shoulder_ys)) + 25
#                 cv2.rectangle(valid_id_zone, (0, 0), (w, neck_line), 255, -1)
#             else:
#                 cv2.rectangle(valid_id_zone, (0, 0), (w, int(h * 0.30)), 255, -1)  # Fallback
#
#             # Keep 75px circles exactly at the wrists
#             for idx in [9, 10]:
#                 if len(kpts) > idx:
#                     wx, wy = int(kpts[idx][0]), int(kpts[idx][1])
#                     if wx != 0 and wy != 0:
#                         cv2.circle(valid_id_zone, (wx, wy), HAND_ZONE_RADIUS, 255, -1)
#
#         except Exception:
#             pass
#
#         # Smooth Master Canvas
#         master_mask = cv2.morphologyEx(master_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
#         _, binary_mask = cv2.threshold(master_mask, 127, 255, cv2.THRESH_BINARY)
#         soft_mask = cv2.GaussianBlur(binary_mask, (FEATHER_RADIUS, FEATHER_RADIUS), 0)
#
#         # ==========================================
#         # 3. THE GOLDEN SAM SUBTRACTION
#         # ==========================================
#         identity_mask = cv2.imread(identity_path, cv2.IMREAD_GRAYSCALE)
#         _, identity_mask = cv2.threshold(identity_mask, 127, 255, cv2.THRESH_BINARY)
#
#         # Apply the Cookie Cutter! Deletes stray SAM forearms instantly.
#         filtered_identity = cv2.bitwise_and(identity_mask, valid_id_zone)
#
#         # Tight inpaint mask — cloth + arms only, no kurta drape polygon
#         tight_inpaint_mask = cv2.bitwise_or(cloth_mask_raw, arm_mask)
#         tight_inpaint_mask = cv2.bitwise_or(tight_inpaint_mask, skeletal_arms)
#         tight_inpaint_mask = cv2.morphologyEx(tight_inpaint_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
#         tight_inpaint_mask[filtered_identity == 255] = 0  # still subtract face/hands
#
#         trouser_mask = np.zeros((h, w), dtype=np.uint8)
#         for tid in TROUSER_IDS:
#             trouser_mask[parse_map == tid] = 255
#         tight_inpaint_mask[trouser_mask == 255] = 0
#
#         # Subtract the perfect hands, face, and neck from the grey canvas!
#         soft_mask[filtered_identity == 255] = 0
#         binary_mask[filtered_identity == 255] = 0
#
#         # ==========================================
#         # 4. ALPHA BLEND & FINAL OUTPUT
#         # ==========================================
#         alpha = soft_mask.astype(np.float32) / 255.0
#         alpha = np.expand_dims(alpha, axis=2)
#         agnostic_img = (img.astype(np.float32) * (1.0 - alpha) + 128.0 * alpha).astype(np.uint8)
#
#         cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, f"{base_name}_mask.png"), binary_mask)  # large — for agnostic image only
#         cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, f"{base_name}_inpaint_mask.png"), tight_inpaint_mask)  # tight — for CatVTON
#         cv2.imwrite(os.path.join(AGNOSTIC_OUTPUT_DIR, f"{base_name}_agnostic.jpg"), agnostic_img)
#
#
# if __name__ == "__main__":
#     build_final_agnostic()
#     print(f"\nDone! Flawless Agnostic Pipeline Complete.")



import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ==========================================
# 1. PATH SETUP
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_DIR         = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "images"))
CLOTH_MASK_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "ref_cloth_mask"))
PARSED_DIR      = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "parsed_labels"))
IDENTITY_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "identity_masks"))

MASK_OUTPUT_DIR     = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "agnostic_mask"))
AGNOSTIC_OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_final", "agnostic"))

os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
os.makedirs(AGNOSTIC_OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. CONFIGURATION
# ==========================================
ARM_IDS     = [14, 15]

# FIX 3: Both LIP and ATR parser trouser class IDs — adjust after
# running the debug print below if results look wrong.
# LIP: trousers=9   ATR: trousers=9 (usually same, but verify)
TROUSER_IDS = [9]

# Skeletal arm thickness
BICEP_THICKNESS    = 40
FOREARM_THICKNESS  = 40

# YOLO Drape & Hand Zone Settings
HAND_ZONE_RADIUS     = 75
KURTA_EXPAND_PADDING = 20
KURTA_DROP_BELOW_KNEE = 55
FEATHER_RADIUS       = 11

# Gravity Kernel (Breathing Room) — downward-only dilation for kurta length
SHIRT_EXPAND_KERNEL = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
], np.uint8)
SHIRT_EXPAND_ITER = 4

print("Loading lightweight YOLO for Geometry...")
pose_model = YOLO("yolov8n-pose.pt", verbose=False)


def build_final_agnostic():
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Starting Agnostic Builder for {len(img_files)} images...")

    for img_name in tqdm(img_files):
        base_name = os.path.splitext(img_name)[0]

        img_path      = os.path.join(IMG_DIR, img_name)
        cloth_path    = os.path.join(CLOTH_MASK_DIR, f"{base_name}.png")
        parse_path    = os.path.join(PARSED_DIR,     f"{base_name}.png")
        identity_path = os.path.join(IDENTITY_DIR,   f"{base_name}.png")

        if not all(os.path.exists(p) for p in [cloth_path, parse_path, identity_path]):
            continue

        # ==========================================
        # 1. BUILD THE GREY CANVAS (Clothes + Arms)
        # ==========================================
        img    = cv2.imread(img_path)
        h, w   = img.shape[:2]

        # A. Base Shirt — save raw BEFORE dilation for tight mask
        cloth_mask = cv2.imread(cloth_path, cv2.IMREAD_GRAYSCALE)
        _, cloth_mask = cv2.threshold(cloth_mask, 127, 255, cv2.THRESH_BINARY)
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
        contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cloth_mask, contours, -1, 255, thickness=cv2.FILLED)

        cloth_mask_raw = cloth_mask.copy()  # snapshot before expansion

        cloth_mask = cv2.dilate(cloth_mask, SHIRT_EXPAND_KERNEL, iterations=SHIRT_EXPAND_ITER)

        # B. Parser Arms
        parse_map = cv2.imread(parse_path, cv2.IMREAD_GRAYSCALE)
        if len(parse_map.shape) > 2:
            parse_map = parse_map[:, :, 0]

        # FIX 3 DEBUG: uncomment temporarily to verify class IDs in your parser output
        # print(f"  [{base_name}] unique parse classes: {np.unique(parse_map)}")

        arm_mask = np.zeros((h, w), dtype=np.uint8)
        for arm_id in ARM_IDS:
            arm_mask[parse_map == arm_id] = 255

        master_mask = cv2.bitwise_or(cloth_mask, arm_mask)

        # ==========================================
        # 2. YOLO GEOMETRY (Kurta Drape & Thick Arms)
        # ==========================================
        results      = pose_model(img, verbose=False)
        valid_id_zone = np.zeros((h, w), dtype=np.uint8)   # Cookie Cutter

        # FIX 1: Track skeletal arm strokes separately so they can be
        # included in the tight inpaint mask later.
        skeletal_arms = np.zeros((h, w), dtype=np.uint8)

        try:
            kpts = results[0].keypoints.xy[0].cpu().numpy()

            # --- SKELETAL ARMS ---
            # Draw on BOTH master_mask and skeletal_arms simultaneously
            if len(kpts) > 9:
                s_l, e_l, w_l = kpts[5], kpts[7], kpts[9]
                if s_l[0] != 0 and e_l[0] != 0:
                    cv2.line(master_mask,  (int(s_l[0]), int(s_l[1])), (int(e_l[0]), int(e_l[1])), 255, BICEP_THICKNESS)
                    cv2.line(skeletal_arms,(int(s_l[0]), int(s_l[1])), (int(e_l[0]), int(e_l[1])), 255, BICEP_THICKNESS)
                if e_l[0] != 0 and w_l[0] != 0:
                    cv2.line(master_mask,  (int(e_l[0]), int(e_l[1])), (int(w_l[0]), int(w_l[1])), 255, FOREARM_THICKNESS)
                    cv2.line(skeletal_arms,(int(e_l[0]), int(e_l[1])), (int(w_l[0]), int(w_l[1])), 255, FOREARM_THICKNESS)

            if len(kpts) > 10:
                s_r, e_r, w_r = kpts[6], kpts[8], kpts[10]
                if s_r[0] != 0 and e_r[0] != 0:
                    cv2.line(master_mask,  (int(s_r[0]), int(s_r[1])), (int(e_r[0]), int(e_r[1])), 255, BICEP_THICKNESS)
                    cv2.line(skeletal_arms,(int(s_r[0]), int(s_r[1])), (int(e_r[0]), int(e_r[1])), 255, BICEP_THICKNESS)
                if e_r[0] != 0 and w_r[0] != 0:
                    cv2.line(master_mask,  (int(e_r[0]), int(e_r[1])), (int(w_r[0]), int(w_r[1])), 255, FOREARM_THICKNESS)
                    cv2.line(skeletal_arms,(int(e_r[0]), int(e_r[1])), (int(w_r[0]), int(w_r[1])), 255, FOREARM_THICKNESS)

            # --- KURTA DRAPE (large mask only — NOT in tight mask) ---
            hip_xs, hip_ys, knee_ys = [], [], []
            for idx in [11, 12]:
                if len(kpts) > idx and kpts[idx][1] != 0:
                    hip_xs.append(int(kpts[idx][0]))
                    hip_ys.append(int(kpts[idx][1]))
            for idx in [13, 14]:
                if len(kpts) > idx and kpts[idx][1] != 0:
                    knee_ys.append(int(kpts[idx][1]))

            if len(hip_xs) == 2 and knee_ys:
                target_bottom_y = min(h, max(knee_ys) + KURTA_DROP_BELOW_KNEE)
                hip_y           = int(np.mean(hip_ys))
                body_left       = min(hip_xs)
                body_right      = max(hip_xs)

                if (body_right - body_left) < 80:
                    center     = (body_left + body_right) // 2
                    body_left  = center - 40
                    body_right = center + 40

                pts = np.array([
                    [max(0, body_left  - KURTA_EXPAND_PADDING),        hip_y - 30],
                    [min(w, body_right + KURTA_EXPAND_PADDING),         hip_y - 30],
                    [min(w, body_right + KURTA_EXPAND_PADDING + 20),    target_bottom_y],
                    [max(0, body_left  - KURTA_EXPAND_PADDING - 20),    target_bottom_y],
                ], np.int32)
                cv2.fillPoly(master_mask, [pts], 255)
                # Note: kurta drape polygon is intentionally NOT added to skeletal_arms

            # --- COOKIE CUTTER (protect face & hands) ---
            shoulder_ys       = [kpts[5][1] if len(kpts) > 5 else 0,
                                  kpts[6][1] if len(kpts) > 6 else 0]
            valid_shoulder_ys = [y for y in shoulder_ys if y != 0]

            if valid_shoulder_ys:
                neck_line = int(min(valid_shoulder_ys)) + 25
                cv2.rectangle(valid_id_zone, (0, 0), (w, neck_line), 255, -1)
            else:
                cv2.rectangle(valid_id_zone, (0, 0), (w, int(h * 0.30)), 255, -1)

            for idx in [9, 10]:
                if len(kpts) > idx:
                    wx, wy = int(kpts[idx][0]), int(kpts[idx][1])
                    if wx != 0 and wy != 0:
                        cv2.circle(valid_id_zone, (wx, wy), HAND_ZONE_RADIUS, 255, -1)

        except Exception:
            pass

        # Smooth large canvas
        master_mask = cv2.morphologyEx(master_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        _, binary_mask = cv2.threshold(master_mask, 127, 255, cv2.THRESH_BINARY)
        soft_mask      = cv2.GaussianBlur(binary_mask, (FEATHER_RADIUS, FEATHER_RADIUS), 0)

        # ==========================================
        # 3. GOLDEN SAM SUBTRACTION
        # ==========================================
        identity_mask = cv2.imread(identity_path, cv2.IMREAD_GRAYSCALE)
        _, identity_mask = cv2.threshold(identity_mask, 127, 255, cv2.THRESH_BINARY)

        # Apply Cookie Cutter — removes stray SAM forearms below shoulder line
        filtered_identity = cv2.bitwise_and(identity_mask, valid_id_zone)

        # ==========================================
        # 4. BUILD TIGHT INPAINT MASK
        # ==========================================
        # FIX 1: Include skeletal_arms so baggy sleeves are always covered
        tight_inpaint_mask = cv2.bitwise_or(cloth_mask_raw, arm_mask)
        tight_inpaint_mask = cv2.bitwise_or(tight_inpaint_mask, skeletal_arms)
        tight_inpaint_mask = cv2.morphologyEx(tight_inpaint_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        # Subtract face/hands from tight mask
        tight_inpaint_mask[filtered_identity == 255] = 0

        # FIX 2: Build trouser mask and erode it slightly before subtracting
        # so the inpaint mask stops cleanly just above the trouser boundary
        # (avoids a 1-2px grey seam at the hemline)
        trouser_mask = np.zeros((h, w), dtype=np.uint8)
        for tid in TROUSER_IDS:
            trouser_mask[parse_map == tid] = 255
        trouser_mask_eroded = cv2.erode(trouser_mask, np.ones((3, 3), np.uint8), iterations=2)
        tight_inpaint_mask[trouser_mask_eroded == 255] = 0

        # Subtract face/hands from large canvas too
        soft_mask[filtered_identity == 255] = 0
        binary_mask[filtered_identity == 255] = 0

        # ==========================================
        # 5. ALPHA BLEND & FINAL OUTPUT
        # ==========================================
        alpha        = soft_mask.astype(np.float32) / 255.0
        alpha        = np.expand_dims(alpha, axis=2)
        agnostic_img = (img.astype(np.float32) * (1.0 - alpha) + 128.0 * alpha).astype(np.uint8)

        # Large mask  — used only to paint the grey agnostic canvas
        cv2.imwrite(os.path.join(MASK_OUTPUT_DIR,     f"{base_name}_mask.png"),         binary_mask)
        # Tight mask  — passed to CatVTON UNet + used for compositing
        cv2.imwrite(os.path.join(MASK_OUTPUT_DIR,     f"{base_name}_inpaint_mask.png"), tight_inpaint_mask)
        # Agnostic image — large grey region gives model room for long garments
        cv2.imwrite(os.path.join(AGNOSTIC_OUTPUT_DIR, f"{base_name}_agnostic.jpg"),     agnostic_img)


if __name__ == "__main__":
    build_final_agnostic()
    print(f"\nDone!")