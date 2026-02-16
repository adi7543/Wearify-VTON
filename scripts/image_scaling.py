import os , cv2
import numpy as np
from tqdm import tqdm

images = r"..\raw images\images"
output = r"..\raw images\images-resized"
os.makedirs(output, exist_ok=True)

for img in tqdm(os.listdir(images)):
    if img .endswith((".jpg","jpeg",".JPG",".png")):
        path = os.path.join(images, img)
        img_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        h, w  = img_data.shape[:2]
        maxLength = max(h,w)
        start = (max(h,w) - min(h,w))/2
        end = (start + (min(h,w)))
        if img .endswith((".jpg",".JPG",".jpeg")):
            black_image = np.zeros((maxLength, maxLength, 3), np.uint8)
            black_image[0:int(maxLength), int(start):int(end)] = img_data
        else :
            black_image = np.zeros((maxLength, maxLength, 4), np.uint8)
            black_image[0:int(maxLength), int(start):int(end)] = img_data
        resized_image = cv2.resize(black_image, (512,512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output,img),resized_image)
#

# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
#
# images = r"..\raw images\cloth"
# output = r"..\raw images\cloth-resized"
# os.makedirs(output, exist_ok=True)
#
# for img in tqdm(os.listdir(images)):
#     if img.lower().endswith((".jpg", ".jpeg", ".png")):
#         path = os.path.join(images, img)
#
#         img_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#         if img_data is None:
#             continue
#
#         h, w = img_data.shape[:2]
#         max_len = max(h, w)
#
#         # detect channels safely
#         channels = img_data.shape[2] if len(img_data.shape) == 3 else 1
#
#         # create square black canvas
#         black_image = np.zeros((max_len, max_len, channels), dtype=np.uint8)
#
#         # center image
#         y_offset = (max_len - h) // 2
#         x_offset = (max_len - w) // 2
#
#         black_image[y_offset:y_offset+h, x_offset:x_offset+w] = img_data
#
#         resized_image = cv2.resize(
#             black_image, (512, 512), interpolation=cv2.INTER_AREA
#         )
#
#         cv2.imwrite(os.path.join(output, img), resized_image)