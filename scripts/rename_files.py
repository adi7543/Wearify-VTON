# import os
#
# # -------- CONFIGURATION --------
# FOLDER_PATH = r"F:\University\fyp\dataset\our dataset\New folder"
# START_NUMBER = 401
# ZERO_PADDING = 1   # 001, 002, 003
# # --------------------------------
#
# files = sorted(os.listdir(FOLDER_PATH))
#
# counter = START_NUMBER
#
# for filename in files:
#     old_path = os.path.join(FOLDER_PATH, filename)
#
#     # Skip directories
#     if not os.path.isfile(old_path):
#         continue
#
#     name, ext = os.path.splitext(filename)
#
#     new_name = f"{str(counter).zfill(ZERO_PADDING)}{ext}"
#     new_path = os.path.join(FOLDER_PATH, new_name)
#
#     os.rename(old_path, new_path)
#     counter += 1
#
# print("Renaming completed.")

import os

# -------- CONFIGURATION --------
FOLDER_PATH = r"F:\University\fyp\dataset\our dataset\New folder\fixed_garments"
START_NUMBER = 401
SUFFIX = ".1"   # constant decimal part
# --------------------------------

files = sorted(os.listdir(FOLDER_PATH))
counter = START_NUMBER

for filename in files:
    old_path = os.path.join(FOLDER_PATH, filename)

    # Skip directories
    if not os.path.isfile(old_path):
        continue

    _, ext = os.path.splitext(filename)

    new_name = f"{counter}{SUFFIX}{ext}"
    new_path = os.path.join(FOLDER_PATH, new_name)

    os.rename(old_path, new_path)
    counter += 1

print("Renaming completed.")