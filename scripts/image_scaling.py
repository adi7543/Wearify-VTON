import os , cv2
import numpy as np
from tqdm import tqdm

images = r"our dataset\images"
output = r"our dataset\images-resized"
os.makedirs(output, exist_ok=True)

for img in tqdm(os.listdir(images)):
    if img .endswith((".jpg","jpeg",".JPG",".png")):
        path = os.path.join(images, img)
        img_data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        h, w  = img_data.shape[:2]
        maxLength = max(h,w)
        start = (max(h,w) - min(h,w))/2
        end = (start + (min(h,w)))
        # if img .endswith(".jpg"):
        black_image = np.zeros((maxLength, maxLength, 3), np.uint8)
        # else :
        #     black_image = np.zeros((maxLength, maxLength, 4), np.uint8)
        black_image[0:int(maxLength), int(start):int(end)] = img_data
        resized_image = cv2.resize(black_image, (512,512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output,img),resized_image)

