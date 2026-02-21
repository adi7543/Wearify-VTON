import os

# -------- CONFIGURATION --------
FOLDER_PATH = r"../our dataset/agnostic/"
START_NUMBER = 1
ZERO_PADDING = 1   # 001, 002, 003
# --------------------------------

files = sorted(os.listdir(FOLDER_PATH))

counter = START_NUMBER

for filename in files:
    old_path = os.path.join(FOLDER_PATH, filename)

    # Skip directories
    if not os.path.isfile(old_path):
        continue

    name, ext = os.path.splitext(filename)

    new_name = f"{str(counter).zfill(ZERO_PADDING)}{ext}"
    new_path = os.path.join(FOLDER_PATH, new_name)

    os.rename(old_path, new_path)
    counter += 1

print("Renaming completed.")