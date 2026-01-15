import os
from PIL import Image

# ---- CONFIG ----
IMAGE_DIR = r"F:\University\fyp\dataset\our dataset\New folder"  # change this
MIN_WIDTH = 768
MIN_HEIGHT = 1024
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# ---- SCRIPT ----
def delete_low_resolution_images(folder):
    deleted_count = 0

    for file_name in os.listdir(folder):
        if not file_name.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        file_path = os.path.join(folder, file_name)

        try:
            with Image.open(file_path) as img:
                width, height = img.size

            if width < MIN_WIDTH and height < MIN_HEIGHT:
                os.remove(file_path)
                deleted_count += 1
                print(f"Deleted: {file_name} ({width}x{height})")

        except Exception as e:
            print(f"Skipped (error): {file_name} -> {e}")

    print(f"\nDone. Total images deleted: {deleted_count}")

# ---- RUN ----
if __name__ == "__main__":
    delete_low_resolution_images(IMAGE_DIR)
