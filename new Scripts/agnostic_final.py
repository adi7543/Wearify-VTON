# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# from ultralytics import YOLO
#
# # ==========================================
#
# # 1. PATH SETUP
#
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
#
# # 2. CONFIGURATION
#
# # ==========================================
# ARM_IDS = [14, 15]
# # 9=pants/trousers, 12=skirt (shalwar often parsed as skirt)
# TROUSER_IDS = [9, 12, 16, 17]
#
# leg_pairs = [(13, 15), (14, 16)]
# LEG_THICKNESS = 60
#
# # Arm thickness — only used on the large (agnostic grey) canvas.
# BICEP_THICKNESS = 40
# FOREARM_THICKNESS_LARGE = 40
#
# # YOLO Drape & Hand Zone Settings
# HAND_ZONE_RADIUS = 75
# KURTA_EXPAND_PADDING = 20
# KURTA_DROP_BELOW_KNEE = 55
# FEATHER_RADIUS = 7
#
# # Trouser erosion iterations
# TROUSER_EROSION_ITER = 3
#
# # Gravity Kernel — downward-only dilation for kurta length
# SHIRT_EXPAND_KERNEL = np.array([
# [0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1]
# ], np.uint8)
# SHIRT_EXPAND_ITER = 4
#
# print("Loading lightweight YOLO for Geometry...")
# pose_model = YOLO("yolov8n-pose.pt", verbose=False)
#
# def build_final_agnostic():
#     img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     print(f"Starting Agnostic Builder for {len(img_files)} images...")
#
#     for img_name in tqdm(img_files):
#         base_name = os.path.splitext(img_name)[0]
#         img_path = os.path.join(IMG_DIR, img_name)
#         cloth_path = os.path.join(CLOTH_MASK_DIR, f"{base_name}.png")
#         parse_path = os.path.join(PARSED_DIR, f"{base_name}.png")
#         identity_path = os.path.join(IDENTITY_DIR, f"{base_name}.png")
#
#         if not all(os.path.exists(p) for p in [cloth_path, parse_path, identity_path]):
#             continue
#         img = cv2.imread(img_path)
#         h, w = img.shape[:2]
#
#         # ==========================================
#
#         # 1. BASE CLOTH MASK
#
#         # ==========================================
#         cloth_mask = cv2.imread(cloth_path, cv2.IMREAD_GRAYSCALE)
#         _, cloth_mask = cv2.threshold(cloth_mask, 127, 255, cv2.THRESH_BINARY)
#         cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
#         contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(cloth_mask, contours, -1, 255, thickness=cv2.FILLED)
#         cloth_mask_raw = cloth_mask.copy()
#
#         # Bottom of existing garment — trouser subtraction only fires below this
#         cloth_rows = np.where(np.any(cloth_mask_raw > 0, axis=1))[0]
#         cloth_bottom_y = int(cloth_rows[-1]) if len(cloth_rows) > 0 else h
#         cloth_mask = cv2.dilate(cloth_mask, SHIRT_EXPAND_KERNEL, iterations=SHIRT_EXPAND_ITER)
#
#
#         # ==========================================
#
#         # 2. PARSER ARMS & TROUSERS
#
#         # ==========================================
#         parse_map = cv2.imread(parse_path, cv2.IMREAD_GRAYSCALE)
#         if len(parse_map.shape) > 2:
#             parse_map = parse_map[:, :, 0]
#
#         arm_mask = np.zeros((h, w), dtype=np.uint8)
#         for arm_id in ARM_IDS:
#             arm_mask[parse_map == arm_id] = 255
#
#
#
#         trouser_mask = np.zeros((h, w), dtype=np.uint8)
#         for tid in TROUSER_IDS:
#             trouser_mask[parse_map == tid] = 255
#
#
#
#         # ==========================================
#
#         # 3. GREY CANVAS MASK (agnostic image only)
#
#         # KEY: drape polygon is NOT added here.
#
#         # The agnostic image must show real trouser pixels so the model
#
#         # sees "legs are here" as context and renders the kurta above them.
#
#         # ==========================================
#         canvas_mask = cv2.bitwise_or(cloth_mask, arm_mask)
#
#
#
#         # ==========================================
#
#         # 4. YOLO GEOMETRY
#
#         # ==========================================
#         results = pose_model(img, verbose=False)
#         valid_id_zone = np.zeros((h, w), dtype=np.uint8)
#         kurta_drape_polygon = np.zeros((h, w), dtype=np.uint8)
#         yolo_leg_mask = np.zeros((h, w), dtype=np.uint8)
#
#         try:
#             kpts = results[0].keypoints.xy[0].cpu().numpy()
#
#             # Skeletal arms → canvas only (NOT tight mask)
#             if len(kpts) > 9:
#                 s_l, e_l, w_l = kpts[5], kpts[7], kpts[9]
#                 if s_l[0] != 0 and e_l[0] != 0:
#                     cv2.line(canvas_mask, (int(s_l[0]), int(s_l[1])), (int(e_l[0]), int(e_l[1])), 255, BICEP_THICKNESS)
#                 if e_l[0] != 0 and w_l[0] != 0:
#                     cv2.line(canvas_mask, (int(e_l[0]), int(e_l[1])), (int(w_l[0]), int(w_l[1])), 255, FOREARM_THICKNESS_LARGE)
#
#             if len(kpts) > 10:
#                 s_r, e_r, w_r = kpts[6], kpts[8], kpts[10]
#                 if s_r[0] != 0 and e_r[0] != 0:
#                     cv2.line(canvas_mask, (int(s_r[0]), int(s_r[1])), (int(e_r[0]), int(e_r[1])), 255, BICEP_THICKNESS)
#                 if e_r[0] != 0 and w_r[0] != 0:
#                     cv2.line(canvas_mask, (int(e_r[0]), int(e_r[1])), (int(w_r[0]), int(w_r[1])), 255, FOREARM_THICKNESS_LARGE)
#
#             # Drape polygon → tight inpaint mask ONLY
#
#             # NOT added to canvas_mask so trousers remain visible in agnostic
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
#                 body_left = min(hip_xs)
#                 body_right = max(hip_xs)
#
#                 if (body_right - body_left) < 80:
#                     center = (body_left + body_right) // 2
#                     body_left = center - 40
#                     body_right = center + 40
#
#                 pts = np.array([
#                     [max(0, body_left - KURTA_EXPAND_PADDING), hip_y - 30],
#                     [min(w, body_right + KURTA_EXPAND_PADDING), hip_y - 30],
#                     [min(w, body_right + KURTA_EXPAND_PADDING + 20), target_bottom_y],
#                     [max(0, body_left - KURTA_EXPAND_PADDING - 20), target_bottom_y],
#                 ], np.int32)
#                 cv2.fillPoly(kurta_drape_polygon, [pts], 255)  # inpaint mask only
#
#             # Cookie cutter — protect face & hands
#             shoulder_ys = [kpts[5][1] if len(kpts) > 5 else 0,
#                            kpts[6][1] if len(kpts) > 6 else 0]
#             valid_shoulder_ys = [y for y in shoulder_ys if y != 0]
#
#             if valid_shoulder_ys:
#                 neck_line = int(min(valid_shoulder_ys)) + 25
#                 cv2.rectangle(valid_id_zone, (0, 0), (w, neck_line), 255, -1)
#             else:
#                 cv2.rectangle(valid_id_zone, (0, 0), (w, int(h * 0.30)), 255, -1)
#
#             # ---- FIX: protect hands using wrist keypoints 9 (left) and 10 (right) ----
#             # HAND_ZONE_RADIUS was defined but previously unused — now applied here.
#             for wrist_idx in [9, 10]:
#                 if len(kpts) > wrist_idx and kpts[wrist_idx][0] != 0:
#                     wx, wy = int(kpts[wrist_idx][0]), int(kpts[wrist_idx][1])
#                     cv2.circle(valid_id_zone, (wx, wy), HAND_ZONE_RADIUS, 255, -1)
#             # --------------------------------------------------------------------------
#
#             for top_idx, bot_idx in leg_pairs:
#                 if len(kpts) > bot_idx:
#                     top = kpts[top_idx]
#                     bot = kpts[bot_idx]
#                 if top[0] != 0 and bot[0] != 0:
#                     cv2.line(yolo_leg_mask,
#                              (int(top[0]), int(top[1])),
#                              (int(bot[0]), int(bot[1])),
#                              255, LEG_THICKNESS)
#
#         except Exception:
#             pass
#
#         trouser_mask = cv2.bitwise_or(trouser_mask, yolo_leg_mask)
#
#         # ==========================================
#
#         # 5. SAM SUBTRACTION
#
#         # ==========================================
#         identity_mask = cv2.imread(identity_path, cv2.IMREAD_GRAYSCALE)
#         _, identity_mask = cv2.threshold(identity_mask, 127, 255, cv2.THRESH_BINARY)
#         filtered_identity = cv2.bitwise_and(identity_mask, valid_id_zone)
#
#         # ==========================================
#
#         # 6. FINALIZE GREY CANVAS MASK
#
#         # ==========================================
#         canvas_mask = cv2.morphologyEx(canvas_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
#         _, binary_mask = cv2.threshold(canvas_mask, 127, 255, cv2.THRESH_BINARY)
#
#         # Subtract face/hands from canvas
#         binary_mask[filtered_identity == 255] = 0
#
#         # Also subtract trousers from canvas so original trouser pixels
#         # are visible in the agnostic — model uses this as context
#         binary_mask[trouser_mask == 255] = 0
#
#         soft_mask = cv2.GaussianBlur(binary_mask, (FEATHER_RADIUS, FEATHER_RADIUS), 0)
#
#         # ==========================================
#
#         # 7. BUILD TIGHT INPAINT MASK
#
#         # ==========================================
#         tight_inpaint_mask = cv2.bitwise_or(cloth_mask_raw, arm_mask)
#         tight_inpaint_mask = cv2.bitwise_or(tight_inpaint_mask, kurta_drape_polygon)
#         tight_inpaint_mask = cv2.morphologyEx(tight_inpaint_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
#
#         # Subtract face/hands (including wrist zones punched out above)
#         tight_inpaint_mask[filtered_identity == 255] = 0
#
#         # Subtract trousers only below original hemline
#         trouser_mask_eroded = cv2.erode(
#             trouser_mask, np.ones((3, 3), np.uint8), iterations=TROUSER_EROSION_ITER
#         )
#         trouser_below_hemline = trouser_mask_eroded.copy()
#         trouser_below_hemline[:cloth_bottom_y, :] = 0
#         tight_inpaint_mask[trouser_below_hemline == 255] = 0
#
#         # ==========================================
#
#         # 8. ALPHA BLEND & SAVE
#
#         # ==========================================
#         alpha = soft_mask.astype(np.float32) / 255.0
#         alpha = np.expand_dims(alpha, axis=2)
#         agnostic_img = (img.astype(np.float32) * (1.0 - alpha) + 128.0 * alpha).astype(np.uint8)
#
#         # Large canvas mask — for reference/debug only
#         cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, f"{base_name}_mask.png"), binary_mask)
#         # Tight inpaint mask — passed to CatVTON UNet + compositing
#         cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, f"{base_name}_inpaint_mask.png"), tight_inpaint_mask)
#         # Trouser restore mask — used in inference to force-paste trouser pixels
#         cv2.imwrite(os.path.join(MASK_OUTPUT_DIR, f"{base_name}_trouser_mask.png"), trouser_mask)
#         # Agnostic image — grey only over garment, trousers visible as context
#         cv2.imwrite(os.path.join(AGNOSTIC_OUTPUT_DIR, f"{base_name}_agnostic.jpg"), agnostic_img)
#
# if __name__ == "__main__":
#     build_final_agnostic()
# print(f"\nDone! Agnostic Pipeline Complete.")












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
TROUSER_IDS = [9, 12, 16, 17]   # 9=pants, 12=skirt, shalwar often hits 12/16/17

leg_pairs = [(13, 15), (14, 16)]
LEG_THICKNESS = 60

BICEP_THICKNESS      = 40
FOREARM_THICKNESS    = 40

HAND_ZONE_RADIUS  = 75
FEATHER_RADIUS    = 7

TROUSER_EROSION_ITER = 3

# Downward-only gravity kernel — extends kurta hem downward, not sideways
SHIRT_EXPAND_KERNEL = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
], np.uint8)
SHIRT_EXPAND_ITER = 4

print("Loading YOLOv8-pose...")
pose_model = YOLO("yolov8n-pose.pt", verbose=False)


def build_agnostic():
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing {len(img_files)} images...")

    for img_name in tqdm(img_files):
        base_name = os.path.splitext(img_name)[0]

        img_path      = os.path.join(IMG_DIR,        img_name)
        cloth_path    = os.path.join(CLOTH_MASK_DIR,  f"{base_name}.png")
        parse_path    = os.path.join(PARSED_DIR,      f"{base_name}.png")
        identity_path = os.path.join(IDENTITY_DIR,    f"{base_name}.png")

        if not all(os.path.exists(p) for p in [cloth_path, parse_path, identity_path]):
            continue

        img  = cv2.imread(img_path)
        h, w = img.shape[:2]

        # ==========================================
        # STEP 1 — BASE CLOTH MASK
        # ==========================================
        cloth_mask = cv2.imread(cloth_path, cv2.IMREAD_GRAYSCALE)
        _, cloth_mask = cv2.threshold(cloth_mask, 127, 255, cv2.THRESH_BINARY)
        cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
        contours, _ = cv2.findContours(cloth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cloth_mask, contours, -1, 255, thickness=cv2.FILLED)

        cloth_mask_raw = cloth_mask.copy()

        # Record hemline so trouser subtraction only fires below it
        cloth_rows    = np.where(np.any(cloth_mask_raw > 0, axis=1))[0]
        cloth_bottom_y = int(cloth_rows[-1]) if len(cloth_rows) > 0 else h

        # Extend hem downward only (gravity kernel)
        cloth_mask = cv2.dilate(cloth_mask, SHIRT_EXPAND_KERNEL, iterations=SHIRT_EXPAND_ITER)

        # ==========================================
        # STEP 2 — PARSER: ARMS & TROUSERS
        # ==========================================
        parse_map = cv2.imread(parse_path, cv2.IMREAD_GRAYSCALE)
        if len(parse_map.shape) > 2:
            parse_map = parse_map[:, :, 0]

        arm_mask = np.zeros((h, w), dtype=np.uint8)
        for arm_id in ARM_IDS:
            arm_mask[parse_map == arm_id] = 255

        trouser_mask = np.zeros((h, w), dtype=np.uint8)
        for tid in TROUSER_IDS:
            trouser_mask[parse_map == tid] = 255

        # ==========================================
        # STEP 3 — GREY CANVAS MASK
        # Trousers intentionally NOT masked here so the model
        # can see the legs as layout context in the agnostic image.
        # ==========================================
        canvas_mask = cv2.bitwise_or(cloth_mask, arm_mask)

        # ==========================================
        # STEP 4 — YOLO GEOMETRY
        # ==========================================
        results       = pose_model(img, verbose=False)
        valid_id_zone = np.zeros((h, w), dtype=np.uint8)
        yolo_leg_mask = np.zeros((h, w), dtype=np.uint8)

        try:
            kpts = results[0].keypoints.xy[0].cpu().numpy()

            # Paint skeletal arms onto canvas (handles baggy sleeves)
            if len(kpts) > 9:
                s_l, e_l, w_l = kpts[5], kpts[7], kpts[9]
                if s_l[0] != 0 and e_l[0] != 0:
                    cv2.line(canvas_mask, tuple(map(int, s_l)), tuple(map(int, e_l)), 255, BICEP_THICKNESS)
                if e_l[0] != 0 and w_l[0] != 0:
                    cv2.line(canvas_mask, tuple(map(int, e_l)), tuple(map(int, w_l)), 255, FOREARM_THICKNESS)

            if len(kpts) > 10:
                s_r, e_r, w_r = kpts[6], kpts[8], kpts[10]
                if s_r[0] != 0 and e_r[0] != 0:
                    cv2.line(canvas_mask, tuple(map(int, s_r)), tuple(map(int, e_r)), 255, BICEP_THICKNESS)
                if e_r[0] != 0 and w_r[0] != 0:
                    cv2.line(canvas_mask, tuple(map(int, e_r)), tuple(map(int, w_r)), 255, FOREARM_THICKNESS)

            # Cookie cutter zone: face + wrist circles (identity to protect)
            shoulder_ys = [kpts[5][1] if len(kpts) > 5 else 0,
                           kpts[6][1] if len(kpts) > 6 else 0]
            valid_ys = [y for y in shoulder_ys if y != 0]

            if valid_ys:
                neck_line = int(min(valid_ys)) + 25
                cv2.rectangle(valid_id_zone, (0, 0), (w, neck_line), 255, -1)
            else:
                cv2.rectangle(valid_id_zone, (0, 0), (w, int(h * 0.30)), 255, -1)

            for wrist_idx in [9, 10]:
                if len(kpts) > wrist_idx and kpts[wrist_idx][0] != 0:
                    wx, wy = int(kpts[wrist_idx][0]), int(kpts[wrist_idx][1])
                    cv2.circle(valid_id_zone, (wx, wy), HAND_ZONE_RADIUS, 255, -1)

            # YOLO leg skeleton (catches what the parser misses on shalwar)
            for top_idx, bot_idx in leg_pairs:
                if len(kpts) > bot_idx:
                    top = kpts[top_idx]
                    bot = kpts[bot_idx]
                    if top[0] != 0 and bot[0] != 0:
                        cv2.line(yolo_leg_mask,
                                 (int(top[0]), int(top[1])),
                                 (int(bot[0]), int(bot[1])),
                                 255, LEG_THICKNESS)

        except Exception:
            pass

        trouser_mask = cv2.bitwise_or(trouser_mask, yolo_leg_mask)

        # ==========================================
        # STEP 5 — SAM / IDENTITY SUBTRACTION
        # ==========================================
        identity_mask = cv2.imread(identity_path, cv2.IMREAD_GRAYSCALE)
        _, identity_mask = cv2.threshold(identity_mask, 127, 255, cv2.THRESH_BINARY)
        # Only trust identity pixels inside the face+wrist protection zone
        filtered_identity = cv2.bitwise_and(identity_mask, valid_id_zone)

        # ==========================================
        # STEP 6 — FINALIZE CANVAS MASK
        # ==========================================
        canvas_mask = cv2.morphologyEx(canvas_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        _, binary_mask = cv2.threshold(canvas_mask, 127, 255, cv2.THRESH_BINARY)

        binary_mask[filtered_identity == 255] = 0   # protect face & hands
        binary_mask[trouser_mask == 255]      = 0   # keep trousers visible

        soft_mask = cv2.GaussianBlur(binary_mask, (FEATHER_RADIUS, FEATHER_RADIUS), 0)

        # ==========================================
        # STEP 7 — TIGHT INPAINT MASK
        # No kurta drape polygon. The cloth_mask_raw + arms + hemline
        # erosion is all that's needed. The model inpaints what it sees.
        # ==========================================
        tight_inpaint_mask = cv2.bitwise_or(cloth_mask_raw, arm_mask)
        tight_inpaint_mask = cv2.morphologyEx(tight_inpaint_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        # Protect face & hands
        tight_inpaint_mask[filtered_identity == 255] = 0

        # Subtract trousers ONLY below the original hemline (don't eat the kurta body)
        trouser_eroded = cv2.erode(trouser_mask, np.ones((3, 3), np.uint8), iterations=TROUSER_EROSION_ITER)
        trouser_below  = trouser_eroded.copy()
        trouser_below[:cloth_bottom_y, :] = 0
        tight_inpaint_mask[trouser_below == 255] = 0

        # ==========================================
        # STEP 8 — ALPHA BLEND & SAVE
        # ==========================================
        alpha       = soft_mask.astype(np.float32) / 255.0
        alpha_3ch   = np.expand_dims(alpha, axis=2)
        agnostic_img = (img.astype(np.float32) * (1.0 - alpha_3ch) + 128.0 * alpha_3ch).astype(np.uint8)

        cv2.imwrite(os.path.join(MASK_OUTPUT_DIR,     f"{base_name}_mask.png"),          binary_mask)
        cv2.imwrite(os.path.join(MASK_OUTPUT_DIR,     f"{base_name}_inpaint_mask.png"),  tight_inpaint_mask)
        cv2.imwrite(os.path.join(MASK_OUTPUT_DIR,     f"{base_name}_trouser_mask.png"),  trouser_mask)
        cv2.imwrite(os.path.join(AGNOSTIC_OUTPUT_DIR, f"{base_name}_agnostic.jpg"),      agnostic_img)


if __name__ == "__main__":
    build_agnostic()
    print("\nDone!")