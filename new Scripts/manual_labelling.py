import cv2
import numpy as np
import torch
import os
from segment_anything import sam_model_registry, SamPredictor

# --- CONFIGURATION ---
INPUT_DIR = r"../dataset_manual/images_to_fix"  # Your images
OUTPUT_DIR = r"../dataset_manual/fixed"  # Where to save results
IMG_SIZE = (512, 512)  # Standard VTON resolution

# Path to SAM Checkpoint
SAM_CHECKPOINT = "../GroundingDINO/sam_vit_h_4b8939.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================

# Global variables for mouse callback
click_points = []
click_labels = []
predictor = None
current_image = None
current_mask = None


def setup_dirs():
    for d in ["GT", "masks", "agnostic", "vis"]:
        os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)


def mouse_callback(event, x, y, flags, param):
    global click_points, click_labels, current_mask

    # Left Click = Add Foreground Point (Green)
    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append([x, y])
        click_labels.append(1)
        update_mask()

    # Right Click = Add Background Point (Red)
    elif event == cv2.EVENT_RBUTTONDOWN:
        click_points.append([x, y])
        click_labels.append(0)
        update_mask()


def update_mask():
    global current_mask, predictor, click_points, click_labels
    if not click_points:
        return

    # Run SAM with prompts
    masks, scores, _ = predictor.predict(
        point_coords=np.array(click_points),
        point_labels=np.array(click_labels),
        multimask_output=False  # We want the single best mask
    )
    current_mask = masks[0].astype(np.uint8) * 255


def apply_overlay(image, mask):
    if mask is None: return image

    # Create Green Overlay
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 1] = 255  # Green channel

    # Blend: 70% Image + 30% Mask
    alpha = 0.4
    binary_mask = mask > 0

    output = image.copy()
    output[binary_mask] = cv2.addWeighted(image[binary_mask], 1 - alpha, colored_mask[binary_mask], alpha, 0)

    # Draw contours for sharpness
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    return output


def main():
    global predictor, current_image, current_mask, click_points, click_labels

    setup_dirs()
    print(f"Loading SAM on {DEVICE}...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(DEVICE)
    predictor = SamPredictor(sam)

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))])
    print(f"Found {len(files)} images. Controls: Left Click=Select, Right Click=Exclude, SPACE=Save, R=Reset, Q=Quit")

    cv2.namedWindow("Labeling Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Labeling Tool", mouse_callback)

    for i, filename in enumerate(files):
        print(f"[{i + 1}/{len(files)}] Processing {filename}...")

        # Load & Resize
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)

        current_image = img
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Reset State
        click_points = []
        click_labels = []
        current_mask = None

        while True:
            # Visualize
            display = current_image.copy()
            if current_mask is not None:
                display = apply_overlay(display, current_mask)

            # Draw Points
            for pt, label in zip(click_points, click_labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(display, tuple(pt), 5, color, -1)

            cv2.imshow("Labeling Tool", display)
            key = cv2.waitKey(1) & 0xFF

            # R = Reset
            if key == ord('r'):
                click_points = []
                click_labels = []
                current_mask = None
                print("Reset.")

            # SPACE = Save & Next
            elif key == 32:  # Space bar
                if current_mask is None:
                    print("No mask created! Click on the image first.")
                    continue

                # Save Ground Truth
                cv2.imwrite(os.path.join(OUTPUT_DIR, "GT", filename), current_image)

                # Save Mask
                cv2.imwrite(os.path.join(OUTPUT_DIR, "masks", filename.replace('.jpg', '.png')), current_mask)

                # Save Agnostic (Grey Hole)
                agnostic = current_image.copy()
                agnostic[current_mask == 255] = 128
                cv2.imwrite(os.path.join(OUTPUT_DIR, "agnostic", filename), agnostic)

                # Save Vis
                cv2.imwrite(os.path.join(OUTPUT_DIR, "vis", filename), display)

                print(f"Saved {filename}")
                break

            # Q = Quit
            elif key == ord('q'):
                print("Exiting...")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("All images processed!")


if __name__ == "__main__":
    main()