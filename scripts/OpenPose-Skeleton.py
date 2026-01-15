import json
import os
import cv2
import numpy as np


coordts = os.listdir("../our dataset/openpose-json\\")
input_folder = "our dataset\\openpose-json\\"
output_folder = "our dataset\\openpose-skeleton\\"
os.makedirs(output_folder, exist_ok=True)

skeleton = [
    (1, 2), (2, 3), (3, 4),        # 3 Right arm
    (1, 5), (5, 6), (6, 7),        # 3 Left arm
    (1, 8),                        # 1 Torso
    (8, 9), (9, 10), (10, 11),     # 3 Right leg
    (8, 12), (12, 13), (13, 14),   # 3 Left leg
    (1, 0),                        # 1 Neck → Nose
    (0, 15), (15, 17),             # 2 Face right
    (0, 16), (16, 18)              # 2 Face left
]

pose_colors = [
    # Right arm (orange → yellow)
    (0, 128, 255),
    (0, 165, 255),
    (0, 200, 255),

    # Left arm (green → lime)
    (0, 255, 128),
    (0, 255, 80),
    (0, 255, 40),

    # Torso (red)
    (0, 0, 255),

    # Left leg (yellow-green)
    (100, 255, 0),
    (80, 220, 0),
    (60, 200, 0),

    # Right leg (cyan → blue)
    (255, 255, 0),
    (255, 200, 0),
    (255, 150, 0),

    # Neck
    (100, 0, 255),

    # Face (purple / magenta)
    (180, 0, 255),(200, 0, 200),
    (255, 0, 255),(255, 0, 180)
]

joint_colors = [
    (255, 0, 255),     # 0: Nose (Purple)
    (0, 0, 255),       # 1: Neck (Red)
    (0, 100, 255),     # 2: R-Shoulder (Orange)
    (0, 165, 255),     # 3: R-Elbow (Orange)
    (0, 255, 255),     # 4: R-Wrist (Yellow)
    (0, 255, 0),       # 5: L-Shoulder (Green)
    (0, 255, 80),      # 6: L-Elbow (Green)
    (0, 255, 128),     # 7: L-Wrist (Lime)
    (0, 0, 255),       # 8: MidHip (Red)
    (100, 255, 0),     # 9: R-Hip (Cyan)
    (80, 220, 0),     # 10: R-Knee (Cyan-Blue)
    (60, 200, 0),     # 11: R-Ankle (Blue)
    (255, 255, 0),     # 12: L-Hip (Yellow-Green)
    (255, 200, 0),      # 13: L-Knee (Yellow-Green)
    (255, 150, 0),      # 14: L-Ankle (Green)
    (255, 0, 255),     # 15: R-Eye (Purple)
    (255, 0, 255),     # 16: L-Eye (Purple)
    (255, 0, 255),     # 17: R-Ear (Purple)
    (255, 0, 255)      # 18: L-Ear (Purple)
]

for img in coordts:
    if img .endswith(".json"):
        json_file = os.path.join(input_folder, img)
        with open(json_file, 'r') as file:
            data = json.load(file)
            person = data["people"][0]
            keypoints = person["pose_keypoints_2d"]
            xy_list = []
            for i in range(0, len(keypoints),3):
                x, y = keypoints[i], keypoints[i+1]
                xy_list.append((x,y))

            height = 512
            width = 512
            black_image = np.zeros((height, width , 3), np.uint8)
            for i, (start, end) in enumerate(skeleton):
                x1, y1 = xy_list[start]
                x2, y2 = xy_list[end]

                if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                        continue
                color = pose_colors[i]
                cv2.line(black_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

            for i in range(len(xy_list)):
                x, y = xy_list[i]
                if (x == 0 and y == 0) or i >= len(joint_colors):
                    continue
                color = joint_colors[i]
                cv2.circle(black_image, (int(x), int(y)), 5, color, -1)

            output_path = os.path.join(output_folder, img.replace(".json", ".png"))
            cv2.imwrite(output_path,black_image)