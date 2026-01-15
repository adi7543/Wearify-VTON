import os
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# Input: Where your current (bluish) images are
input_folder = r"our dataset\New folder\extracted_garments6"

# Output: Where the fixed images will go (Using F: to be safe)
output_folder = r"our dataset\New folder\fixed_garments"
os.makedirs(output_folder, exist_ok=True)

# --- BATCH PROCESSING ---
print(f"🚀 Starting Batch Color Fix...")
print(f"Input: {input_folder}")
print(f"Output: {output_folder}")

# Get list of images
files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for filename in tqdm(files):
    try:
        input_path = os.path.join(input_folder, filename)
        save_path = os.path.join(output_folder, filename)

        # 1. Open Image
        img = Image.open(input_path)

        # 2. Force Channel Swap (The Nuclear Fix)
        # We assume the image is technically RGB/RGBA but the data is swapped
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            # Swap Red and Blue channels
            fixed_img = Image.merge("RGBA", (b, g, r, a))
        elif img.mode == 'RGB':
            r, g, b = img.split()
            fixed_img = Image.merge("RGB", (b, g, r))
        else:
            # Fallback for grayscale/other modes
            fixed_img = img

        # 3. Save to the safe drive
        fixed_img.save(save_path)

    except Exception as e:
        print(f"❌ Skipped {filename}: {e}")

print(f"\n✅ DONE! All fixed images are in: {output_folder}")