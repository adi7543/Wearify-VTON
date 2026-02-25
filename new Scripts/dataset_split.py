import os
import random
import cv2
import numpy as np

# --- CONFIGURATION ---
DATA_ROOT = r"../dataset_mfp_final"
OUTPUT_TRAIN = r"../dataset_mfp_final/train_pairs.txt"
OUTPUT_TEST = r"../dataset_mfp_final/test_pairs.txt"
SPLIT_RATIO = 0.80


def main():
    print(f"--- CHECKING DATASET AT: {DATA_ROOT} ---")

    # 1. Verify Root Exists
    if not os.path.exists(DATA_ROOT):
        print(f"CRITICAL ERROR: Folder '{DATA_ROOT}' does not exist!")
        print("Please check the folder name in your file explorer.")
        return

    # 2. Check Subfolders
    required_folders = ["images", "masks", "agnostic", "ref_cloth", "pose_img"]
    for folder in required_folders:
        path = os.path.join(DATA_ROOT, folder)
        if not os.path.exists(path):
            print(f"WARNING: Missing subfolder '{folder}'")
            if folder == "pose_img":
                print("  -> Did you run 'render_pose.py'? The training needs skeleton IMAGES, not JSONs.")

    # 3. Scan Files
    images_dir = os.path.join(DATA_ROOT, "images")
    if not os.path.exists(images_dir): return

    all_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Found {len(all_files)} images in 'images' folder.")

    valid_names = []
    missing_log = []  # Track why files fail

    for filename in all_files:
        name_base = os.path.splitext(filename)[0]

        # Paths
        p_img = os.path.join(DATA_ROOT, "images", filename)
        p_mask = os.path.join(DATA_ROOT, "masks", f"{name_base}.png")
        p_agnostic = os.path.join(DATA_ROOT, "agnostic", filename)
        p_cloth = os.path.join(DATA_ROOT, "ref_cloth", filename)

        # Check for Pose Image (Try both naming conventions)
        p_pose_1 = os.path.join(DATA_ROOT, "pose_img", f"{name_base}.png")
        p_pose_2 = os.path.join(DATA_ROOT, "pose_img", f"{name_base}_keypoints.png")

        has_pose = os.path.exists(p_pose_1) or os.path.exists(p_pose_2)

        # DEBUG CHECK
        missing = []
        if not os.path.exists(p_mask): missing.append("Mask")
        if not os.path.exists(p_agnostic): missing.append("Agnostic")
        if not os.path.exists(p_cloth): missing.append("Cloth")
        if not has_pose: missing.append("Pose_Img")

        if len(missing) == 0:
            valid_names.append(filename)

            # Create Cloth Mask (Only for valid pairs)
            cloth_mask_dir = os.path.join(DATA_ROOT, "cloth_mask")
            os.makedirs(cloth_mask_dir, exist_ok=True)
            cm_path = os.path.join(cloth_mask_dir, filename)

            if not os.path.exists(cm_path):
                cloth_img = cv2.imread(p_cloth)
                if cloth_img is not None:
                    diff = cv2.absdiff(cloth_img, (128, 128, 128))
                    mask = np.sum(diff, axis=2)
                    _, cloth_mask = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
                    cv2.imwrite(cm_path, cloth_mask)
        else:
            missing_log.append(f"{filename}: Missing {missing}")

    # 4. Report Errors
    if len(valid_names) == 0:
        print("\n--- NO VALID PAIRS FOUND ---")
        print("Here is why the first 5 failed:")
        for log in missing_log[:5]:
            print(log)
        return

    # 5. Save Split
    random.seed(42)
    random.shuffle(valid_names)
    split_idx = int(len(valid_names) * SPLIT_RATIO)
    train_list = valid_names[:split_idx]
    test_list = valid_names[split_idx:]

    with open(os.path.join(DATA_ROOT, OUTPUT_TRAIN), "w") as f:
        for name in train_list: f.write(f"{name} {name}\n")
    with open(os.path.join(DATA_ROOT, OUTPUT_TEST), "w") as f:
        for name in test_list: f.write(f"{name} {name}\n")

    print(f"\nSUCCESS!")
    print(f"Total Valid: {len(valid_names)}")
    print(f"Training:    {len(train_list)}")
    print(f"Testing:     {len(test_list)}")


if __name__ == "__main__":
    main()