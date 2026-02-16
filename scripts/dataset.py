import os, json

# images_folder = r"../our dataset/train/images-resized/"
# pose_folder = r"../our dataset/train/openpose-skeleton/"
# openpose_json_folder = r"../our dataset/train/openpose-json/"
# parse_folder = r"../our dataset/train/images-parse/"
# cloth_folder = r"../our dataset/train/cloth-resized/"
# cloth_mask_folder = r"../our dataset/train/cloth-mask/"
# agnostic_folder = r"../our dataset/train/agnostic/"

images_folder = r"../our dataset/test/images-resized/"
pose_folder = r"../our dataset/test/openpose-skeleton/"
openpose_json_folder = r"../our dataset/test/openpose-json/"
parse_folder = r"../our dataset/test/images-parse/"
cloth_folder = r"../our dataset/test/cloth-resized/"
cloth_mask_folder = r"../our dataset/test/cloth-mask/"
agnostic_folder = r"../our dataset/test/agnostic/"

images = [f for f in os.listdir(images_folder) if f.lower().endswith(".jpg")]
# output_file = r"../our dataset/test/test.json"
output_file = r"../our dataset/test/test.json"
# print(images)
with open(output_file, 'w') as j:
    for img in images:
        image = os.path.join(images_folder,img)
        pose = os.path.join(pose_folder,img)
        openpose_json = os.path.join(openpose_json_folder,img)
        parse = os.path.join(parse_folder, img)
        cloth = os.path.join(cloth_folder,img)
        cloth_mask = os.path.join(cloth_mask_folder, img)
        agnostic = os.path.join(agnostic_folder, img)
        if img .endswith(".JPG"):
            pose = pose.replace(".JPG","_keypoints.png")
            openpose_json = openpose_json.replace(".JPG", "_keypoints.json")
            parse = parse.replace(".JPG", ".png")
            # caption = caption.replace(".JPG",".txt")
            cloth = cloth.replace(".JPG", ".1.png")
            cloth_mask = cloth_mask.replace(".JPG", ".1.png")
        else:
            pose = pose.replace(".jpg", "_keypoints.png")
            openpose_json = openpose_json.replace(".jpg", "_keypoints.json")
            parse = parse.replace(".jpg", ".png")
            cloth = cloth.replace(".jpg", ".1.png")
            cloth_mask = cloth_mask.replace(".jpg", ".1.png")
        # print(image, pose, parse, caption)
        if os.path.exists(pose) and os.path.exists(parse):
            # with open(caption, 'r') as c:
                # raw_caption_line = c.read().strip()
                # caption_line = raw_caption_line.replace(" ' s", "'s").replace("t - shirt", "t-shirt")
                content = {
                    "target": image.replace("\\", "/"),
                    "skeleton": pose.replace("\\", "/"),
                    "keypoints": openpose_json.replace("\\", "/"),
                    "parse": parse.replace("\\", "/"),
                    "cloth":cloth.replace("\\", "/"),
                    "cloth_mask":cloth_mask.replace("\\", "/"),
                    "agnostic": agnostic.replace("\\", "/")
                }
                line = json.dumps(content)
                j.write(line + "\n")



# import os, json, random
#
# # ------------------
# # PATHS (TRAIN ONLY)
# # ------------------
# root = "../our dataset/train"
#
# images_folder = os.path.join(root, "images-resized")
# pose_folder = os.path.join(root, "openpose-skeleton")
# openpose_json_folder = os.path.join(root, "openpose-json")
# agnostic_folder = os.path.join(root, "agnostic")
# cloth_folder = os.path.join(root, "cloth-resized")
# cloth_mask_folder = os.path.join(root, "cloth-mask")
#
# output_file = os.path.join(root, "train.json")
#
# # ------------------
# # COLLECT FILES
# # ------------------
# person_images = sorted([f for f in os.listdir(images_folder) if f.lower().endswith(".jpg")])
# cloth_images = sorted([f for f in os.listdir(cloth_folder) if f.lower().endswith(".png")])
#
# assert len(person_images) > 1, "Need at least 2 images to avoid identity pairing"
#
# # ------------------
# # BUILD JSON
# # ------------------
# with open(output_file, "w") as f:
#     for person_img in person_images:
#
#         person_id = os.path.splitext(person_img)[0]
#
#         # ---- PERSON FILES ----
#         target = os.path.join(images_folder, person_img)
#         skeleton = os.path.join(pose_folder, f"{person_id}_keypoints.png")
#         keypoints = os.path.join(openpose_json_folder, f"{person_id}_keypoints.json")
#         agnostic = os.path.join(agnostic_folder, person_img)
#
#         if not (os.path.exists(skeleton) and os.path.exists(keypoints)):
#             continue
#
#         # ---- RANDOM DIFFERENT CLOTH ----
#         cloth_img = random.choice(cloth_images)
#
#         # Ensure cloth does NOT come from same identity
#         while cloth_img.startswith(person_id):
#             cloth_img = random.choice(cloth_images)
#
#         cloth = os.path.join(cloth_folder, cloth_img)
#         cloth_mask = os.path.join(cloth_mask_folder, cloth_img)
#
#         if not os.path.exists(cloth_mask):
#             continue
#
#         sample = {
#             "target": target.replace("\\", "/"),
#             "skeleton": skeleton.replace("\\", "/"),
#             "keypoints": keypoints.replace("\\", "/"),
#             "agnostic": agnostic.replace("\\", "/"),
#             "cloth": cloth.replace("\\", "/"),
#             "cloth_mask": cloth_mask.replace("\\", "/")
#         }
#
#         f.write(json.dumps(sample) + "\n")
#
# print("✅ train.json rebuilt with cross-identity cloth pairing")
