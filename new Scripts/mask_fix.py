import cv2
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_DIR = r"../dataset_manual/images"

MASKS_DIR = r"../dataset_manual/masks"

OUTPUT_DIR = r"../dataset_manual/fixed_masks"

# Visual Settings
BRUSH_SIZE = 1
ALPHA = 0.5


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Folder '{INPUT_DIR}' not found.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all images
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    total_files = len(files)

    print(f"--- RESUMABLE ERASER: Found {total_files} images ---")
    print("LEFT CLICK: Draw Mask (Restore)")
    print("RIGHT CLICK: Erase (Cut Webbing)")
    print("[S]: Save & Next")
    print("[Q]: Quit")

    cv2.namedWindow("Eraser", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eraser", 1000, 800)

    current_idx = 0

    while current_idx < total_files:
        filename = files[current_idx]
        name_base = os.path.splitext(filename)[0]

        # --- AUTO-RESUME CHECK ---
        # If this mask already exists in the Output folder, skip it!
        save_path = os.path.join(OUTPUT_DIR, f"{name_base}.png")
        if os.path.exists(save_path):
            print(f"Skipping {filename} (Already Fixed)")
            current_idx += 1
            continue

        # Load Image
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)

        # Load Mask (Try png then jpg)
        mask_path = os.path.join(MASKS_DIR, f"{name_base}.png")
        if not os.path.exists(mask_path): mask_path = os.path.join(MASKS_DIR, f"{name_base}.jpg")

        if not os.path.exists(mask_path):
            print(f"Mask missing for {filename}, skipping.")
            current_idx += 1
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize mask to match image
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Mouse Callback (Defined inside loop to capture current 'mask' variable)
        drawing = False
        erasing = False

        def paint(event, x, y, flags, param):
            nonlocal drawing, erasing
            # Global brush size check
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                cv2.circle(mask, (x, y), BRUSH_SIZE, 255, -1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                erasing = True
                cv2.circle(mask, (x, y), BRUSH_SIZE, 0, -1)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing: cv2.circle(mask, (x, y), BRUSH_SIZE, 255, -1)
                if erasing: cv2.circle(mask, (x, y), BRUSH_SIZE, 0, -1)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
            elif event == cv2.EVENT_RBUTTONUP:
                erasing = False

        cv2.setMouseCallback("Eraser", paint)

        print(f"Processing: {filename} ({current_idx + 1}/{total_files})")

        while True:
            # Create Green Overlay Live
            vis = img.copy()
            vis[mask == 255] = vis[mask == 255] * 0.5 + np.array([0, 255, 0]) * 0.5

            cv2.imshow("Eraser", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # Save & Next
                cv2.imwrite(save_path, mask)
                print(f"Saved: {filename}")
                current_idx += 1
                break  # Break inner loop, go to next image

            elif key == ord('q'):  # Quit
                print("Quitting... You can resume later!")
                cv2.destroyAllWindows()
                return  # Exit script completely

    cv2.destroyAllWindows()
    print("All images processed! You are ready for V4 Training.")


if __name__ == "__main__":
    main()