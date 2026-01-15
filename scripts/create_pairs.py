import os

# --- 1. DEFINE PATHS ---
# Assuming this script is inside your main project folder
# and your data is in 'data/train/...'
base_data_path = '../data'

image_dir = os.path.join(base_data_path, 'train', 'image')
cloth_dir = os.path.join(base_data_path, 'train', 'cloth')
output_file = os.path.join(base_data_path, 'train_pairs.txt')

print(f"Scanning images in: {image_dir}")
print(f"Scanning cloth in:  {cloth_dir}")

# --- 2. VALIDATION CHECK ---
if not os.path.exists(image_dir) or not os.path.exists(cloth_dir):
    print("❌ ERROR: Could not find your data folders!")
    print(f"Please check if '{image_dir}' exists.")
    exit()

# --- 3. GET FILE LISTS ---
# Get all files, but filter for valid images only
valid_exts = ('.jpg', '.png', '.jpeg')
images = set(f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts))
clothes = set(f for f in os.listdir(cloth_dir) if f.lower().endswith(valid_exts))

# Find the intersection (files that exist in BOTH folders)
# This prevents errors if you have an image but are missing the cloth (or vice versa)
common_files = sorted(list(images.intersection(clothes)))

print(f"Found {len(images)} person images.")
print(f"Found {len(clothes)} cloth images.")
print(f"Found {len(common_files)} VALID PAIRS to write.")

# --- 4. WRITE THE PAIRS FILE ---
with open(output_file, 'w') as f:
    for filename in common_files:
        # Format: image_filename [SPACE] cloth_filename
        line = f"{filename} {filename}\n"
        f.write(line)

print("-" * 30)
print("✅ DONE! 'train_pairs.txt' has been created.")
print(f"Location: {os.path.abspath(output_file)}")
