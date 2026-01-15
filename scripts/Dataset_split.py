import os
import random
import shutil

train_folders = ["our dataset\\train\\cloth-resized\\","our dataset\\train\\images-resized\\",
           "our dataset\\train\\images-parse\\","our dataset\\train\\openpose-skeleton\\",
           "our dataset\\train\\agnostic\\","our dataset\\train\\cloth-mask\\"]
test_folders = ["our dataset\\test\\cloth-resized\\","our dataset\\test\\images-resized\\",
           "our dataset\\test\\images-parse\\","our dataset\\test\\openpose-skeleton\\",
           "our dataset\\test\\agnostic\\","our dataset\\test\\cloth-mask\\"]
train = "our dataset\\train\\"
test = "our dataset\\test\\"
os.makedirs(train, exist_ok=True)
os.makedirs(test, exist_ok=True)

for folder in train_folders:
    os.makedirs(folder, exist_ok=True)

for folder in test_folders:
    os.makedirs(folder, exist_ok=True)

images_folder = "our dataset\\images-resized\\"
pose_folder = "our dataset\\openpose-skeleton\\"
parse_folder = "our dataset\\images-parse\\"
cloth_folder = "our dataset\\cloth-resized\\"
cloth_mask_folder = "our dataset\\cloth-mask\\"
agnostic_folder = "our dataset\\agnostic\\"

all_images = [f for f in os.listdir(images_folder) if f .endswith((".jpg","JPG"))]

random.shuffle(all_images)
# print(len(all_images))
split = int(len(all_images)*0.8)
# print(split)
train_images = all_images[:split]
test_images = all_images[split:]
# print(len(train_images))
# print(len(test_images))


for img in train_images:
    if img .endswith((".jpg",".JPG")):
        image = os.path.join(images_folder,img)
        pose = os.path.join(pose_folder,img)
        parse = os.path.join(parse_folder, img)
        cloth = os.path.join(cloth_folder,img)
        cloth_mask = os.path.join(cloth_mask_folder, img)
        agnostic = os.path.join(agnostic_folder, img)
        caption = os.path.join(images_folder, img)
        if img .endswith(".JPG"):
            pose = pose.replace(".JPG","_keypoints.png")
            parse = parse.replace(".JPG", ".png")
            caption = caption.replace(".JPG",".txt")
            cloth = cloth.replace(".JPG", ".1.png")
            cloth_mask = cloth_mask.replace(".JPG", ".1.png")
        else:
            pose = pose.replace(".jpg", "_keypoints.png")
            parse = parse.replace(".jpg", ".png")
            caption = caption.replace(".jpg", ".txt")
            cloth = cloth.replace(".jpg", ".1.png")
            cloth_mask = cloth_mask.replace(".jpg", ".1.png")

        shutil.copy(image,train_folders[1])
        shutil.copy(caption,train_folders[1])
        shutil.copy(pose, train_folders[3])
        shutil.copy(cloth, train_folders[0])
        shutil.copy(cloth_mask, train_folders[5])
        shutil.copy(parse, train_folders[2])
        shutil.copy(agnostic, train_folders[4])

for img in test_images:
    if img .endswith((".jpg",".JPG")):
        image = os.path.join(images_folder,img)
        pose = os.path.join(pose_folder,img)
        parse = os.path.join(parse_folder, img)
        cloth = os.path.join(cloth_folder,img)
        cloth_mask = os.path.join(cloth_mask_folder, img)
        agnostic = os.path.join(agnostic_folder, img)
        caption = os.path.join(images_folder, img)
        if img .endswith(".JPG"):
            pose = pose.replace(".JPG","_keypoints.png")
            parse = parse.replace(".JPG", ".png")
            caption = caption.replace(".JPG",".txt")
            cloth = cloth.replace(".JPG", ".1.png")
            cloth_mask = cloth_mask.replace(".JPG", ".1.png")
        else:
            pose = pose.replace(".jpg", "_keypoints.png")
            parse = parse.replace(".jpg", ".png")
            caption = caption.replace(".jpg", ".txt")
            cloth = cloth.replace(".jpg", ".1.png")
            cloth_mask = cloth_mask.replace(".jpg", ".1.png")

        shutil.copy(image,test_folders[1])
        shutil.copy(caption,test_folders[1])
        shutil.copy(pose, test_folders[3])
        shutil.copy(cloth, test_folders[0])
        shutil.copy(cloth_mask, test_folders[5])
        shutil.copy(parse, test_folders[2])
        shutil.copy(agnostic, test_folders[4])