import cv2
import numpy as np
import os
from tqdm import tqdm

images = r"our dataset\images-resized"
seg = r"our dataset\images-parse"
clothpath = r"our dataset\cloth-mask"
output_dir = r"../our dataset/agnostic"
os.makedirs(output_dir, exist_ok=True)

for img in tqdm(os.listdir(images)):
    if img .endswith((".JPG",".jpg")):
        image = os.path.join(images,img)
        parse = os.path.join(seg, img)
        cloth = os.path.join(clothpath,img)
        if img .endswith("JPG"):
            parse = parse.replace(".JPG", ".png")
            cloth = cloth.replace(".JPG", ".1.png")
        else:
            parse = parse.replace(".jpg", ".png")
            cloth = cloth.replace(".jpg", ".1.png")
        person = cv2.imread(image)
        segment = cv2.imread(parse,0)
        cloth_mask = cv2.imread(cloth, 0)

        KURTA = 4
        DUPATTA = 17
        LEFT_ARM = 14
        RIGHT_ARM = 15


        _, cloth_mask = cv2.threshold(cloth_mask, 127, 255, cv2.THRESH_BINARY)

        seg_mask = np.zeros_like(segment, dtype=np.uint8)
        seg_mask[segment == KURTA] = 255
        seg_mask[segment == DUPATTA] = 255
        seg_mask[segment == LEFT_ARM] = 0
        seg_mask[segment == RIGHT_ARM] = 0

        final_mask = cv2.bitwise_or(seg_mask, cloth_mask)

        kernel = np.ones((3, 3), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)

        final_mask[segment == LEFT_ARM] = 0
        final_mask[segment == RIGHT_ARM] = 0

        inv_mask = cv2.bitwise_not(final_mask)
        hollow_person = cv2.bitwise_and(person, person, mask=inv_mask)

        neutral = np.zeros_like(person)
        neutral[:] = (128, 128, 128)
        fill = cv2.bitwise_and(neutral, neutral, mask=final_mask)

        agnostic = cv2.add(hollow_person, fill)

        out_path = os.path.join(output_dir, img)
        cv2.imwrite(out_path, agnostic)