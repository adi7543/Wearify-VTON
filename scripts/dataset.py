import os, json

# images_folder = "our dataset\\train\\images-resized\\"
# pose_folder = "our dataset\\train\\openpose-skeleton\\"
# parse_folder = "our dataset\\train\\images-parse\\"
# cloth_folder = "our dataset\\train\\cloth-resized\\"
# cloth_mask_folder = "our dataset\\train\\cloth-mask\\"
# agnostic_folder = "our dataset\\train\\agnostic\\"

images_folder = "our dataset\\test\\images-resized\\"
pose_folder = "our dataset\\test\\openpose-skeleton\\"
parse_folder = "our dataset\\test\\images-parse\\"
cloth_folder = "our dataset\\test\\cloth-resized\\"
cloth_mask_folder = "our dataset\\test\\cloth-mask\\"
agnostic_folder = "our dataset\\test\\agnostic\\"

images = [f for f in os.listdir(images_folder) if f .endswith((".jpg",".JPG"))]
# print(images)
with open("../our dataset/test/test.json", 'w') as j:
    for img in images:
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
        # print(image, pose, parse, caption)
        if os.path.exists(caption) and os.path.exists(pose) and os.path.exists(parse):
            with open(caption, 'r') as c:
                raw_caption_line = c.read().strip()
                caption_line = raw_caption_line.replace(" ' s", "'s").replace("t - shirt", "t-shirt")
                content = {
                    "target": image.replace("\\", "/"),
                    "pose": pose.replace("\\", "/"),
                    "parse": parse.replace("\\", "/"),
                    "cloth":cloth.replace("\\", "/"),
                    "cloth_mask":cloth_mask.replace("\\", "/"),
                    "agnostic": agnostic.replace("\\", "/"),
                    "prompt":caption_line
                }
                line = json.dumps(content)
                j.write(line + "\n")