import os
import random
import shutil

train_folders = [r"../our dataset\\train\\cloth-resized\\",r"../our dataset\\train\\images-resized\\",
           r"../our dataset\\train\\images-parse\\",r"../our dataset\\train\\openpose-json\\",
           r"../our dataset\\train\\agnostic\\",r"../our dataset\\train\\cloth-mask\\",
                 r"../our dataset\\train\\openpose-skeleton\\"]
test_folders = [r"../our dataset\\test\\cloth-resized\\",r"../our dataset\\test\\images-resized\\",
           r"../our dataset\\test\\images-parse\\",r"../our dataset\\test\\openpose-json\\",
           r"../our dataset\\test\\agnostic\\",r"../our dataset\\test\\cloth-mask\\",
                r"../our dataset\\test\\openpose-skeleton\\"]
train = r"../our dataset/train/"
test = r"../our dataset/test/"
os.makedirs(train, exist_ok=True)
os.makedirs(test, exist_ok=True)

for folder in train_folders:
    os.makedirs(folder, exist_ok=True)

for folder in test_folders:
    os.makedirs(folder, exist_ok=True)

images_folder = r"../our dataset/images-resized/"
pose_folder = r"../our dataset/openpose-json/"
parse_folder = r"../our dataset/images-parse/"
cloth_folder = r"../our dataset/cloth-resized/"
cloth_mask_folder = r"../our dataset/cloth-mask/"
agnostic_folder = r"../our dataset/agnostic/"
skeleton_folder = r"../our dataset/openpose-skeleton/"

all_images = [f for f in os.listdir(images_folder) if f.lower() .endswith((".jpg"))]

random.shuffle(all_images)
# print(len(all_images))
split = int(len(all_images)*0.8)
# print(split)
train_images = all_images[:split]
test_images = all_images[split:]
# print(len(train_images))
# print(len(test_images))

for img in train_images:
    if img.lower() .endswith((".jpg")):
        image = os.path.join(images_folder,img)
        pose = os.path.join(pose_folder,img)
        skeleton = os.path.join(skeleton_folder,img)
        parse = os.path.join(parse_folder, img)
        cloth = os.path.join(cloth_folder,img)
        cloth_mask = os.path.join(cloth_mask_folder, img)
        agnostic = os.path.join(agnostic_folder, img)
        if img .endswith(".JPG"):
            pose = pose.replace(".JPG","_keypoints.json")
            skeleton = skeleton.replace(".JPG", "_keypoints.png")
            parse = parse.replace(".JPG", ".png")
            cloth = cloth.replace(".JPG", ".1.png")
            cloth_mask = cloth_mask.replace(".JPG", ".1.png")
        else:
            pose = pose.replace(".jpg", "_keypoints.json")
            skeleton = skeleton.replace(".jpg", "_keypoints.png")
            parse = parse.replace(".jpg", ".png")
            cloth = cloth.replace(".jpg", ".1.png")
            cloth_mask = cloth_mask.replace(".jpg", ".1.png")

        shutil.copy(image,train_folders[1])
        shutil.copy(pose, train_folders[3])
        shutil.copy(cloth, train_folders[0])
        shutil.copy(cloth_mask, train_folders[5])
        shutil.copy(parse, train_folders[2])
        shutil.copy(agnostic, train_folders[4])
        shutil.copy(skeleton, train_folders[6])

for img in test_images:
    if img.lower() .endswith((".jpg")):
        image = os.path.join(images_folder,img)
        pose = os.path.join(pose_folder,img)
        skeleton = os.path.join(skeleton_folder, img)
        parse = os.path.join(parse_folder, img)
        cloth = os.path.join(cloth_folder,img)
        cloth_mask = os.path.join(cloth_mask_folder, img)
        agnostic = os.path.join(agnostic_folder, img)
        if img .endswith(".JPG"):
            pose = pose.replace(".JPG","_keypoints.json")
            skeleton = skeleton.replace(".JPG", "_keypoints.png")
            parse = parse.replace(".JPG", ".png")
            cloth = cloth.replace(".JPG", ".1.png")
            cloth_mask = cloth_mask.replace(".JPG", ".1.png")
        else:
            pose = pose.replace(".jpg", "_keypoints.json")
            skeleton = skeleton.replace(".jpg", "_keypoints.png")
            parse = parse.replace(".jpg", ".png")
            cloth = cloth.replace(".jpg", ".1.png")
            cloth_mask = cloth_mask.replace(".jpg", ".1.png")

        shutil.copy(image,test_folders[1])
        shutil.copy(pose, test_folders[3])
        shutil.copy(cloth, test_folders[0])
        shutil.copy(cloth_mask, test_folders[5])
        shutil.copy(parse, test_folders[2])
        shutil.copy(agnostic, test_folders[4])
        shutil.copy(skeleton, test_folders[6])