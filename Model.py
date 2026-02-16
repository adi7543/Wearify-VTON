# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from diffusers import UNet2DModel, DDPMScheduler
# from tqdm import tqdm
#
# # =====================
# # CONFIGURATION
# # =====================
# CHECKPOINT_EPOCH = 6
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IMAGE_SIZE = 256
#
# # FOLDERS
# INPUT_ROOT = "raw images"
# AGNOSTIC_DIR = os.path.join(INPUT_ROOT, "agnostic")
# CLOTH_DIR = os.path.join(INPUT_ROOT, "cloth-resized")
# MASK_DIR = os.path.join(INPUT_ROOT, "cloth-mask")
# SKELETON_DIR = os.path.join(INPUT_ROOT, "openpose-skeleton")
#
# OUTPUT_DIR = "final_predictions"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#
# # =====================
# # MODEL ARCHITECTURE
# # =====================
# class FlowWarpingModule(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(7, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
#             nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
#             nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
#             nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
#             nn.Conv2d(64, 2, 3, padding=1)
#         )
#         nn.init.constant_(self.decoder[-1].weight, 0)
#         nn.init.constant_(self.decoder[-1].bias, 0)
#
#     def forward(self, cloth, mask, skeleton):
#         B, _, H, W = cloth.shape
#         x = torch.cat([cloth, mask, skeleton], dim=1)
#         feat = self.encoder(x)
#         flow = self.decoder(feat)
#         flow = torch.clamp(flow, -0.9, 0.9)
#
#         grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=cloth.device),
#                                         torch.linspace(-1, 1, W, device=cloth.device), indexing='ij')
#         base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
#         final_grid = base_grid + flow.permute(0, 2, 3, 1)
#
#         warped_cloth = F.grid_sample(cloth, final_grid, align_corners=True, padding_mode="border")
#         warped_mask = F.grid_sample(mask, final_grid, align_corners=True, padding_mode="zeros")
#
#         return warped_cloth, warped_mask, flow
#
#     # =====================
#
#
# # PREDICTOR CLASS
# # =====================
# class WearifyPredictor:
#     def __init__(self, checkpoint_dir="checkpoints"):
#         print(f"Loading Models (Epoch {CHECKPOINT_EPOCH})...")
#
#         self.warp = FlowWarpingModule().to(DEVICE)
#         self.warp.load_state_dict(torch.load(f"{checkpoint_dir}/flow_{CHECKPOINT_EPOCH}.pt", map_location=DEVICE))
#         self.warp.eval()
#
#         self.unet = UNet2DModel(
#             sample_size=IMAGE_SIZE,
#             in_channels=10,
#             out_channels=3,
#             block_out_channels=(32, 64, 128, 128)
#         ).to(DEVICE)
#         self.unet.load_state_dict(torch.load(f"{checkpoint_dir}/unet_{CHECKPOINT_EPOCH}.pt", map_location=DEVICE))
#         self.unet.eval()
#
#         self.scheduler = DDPMScheduler(num_train_timesteps=1000)
#         print("Ready for Inference.")
#
#     def preprocess(self, img_path, is_mask=False):
#         if not os.path.exists(img_path): return None
#         img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#         if img is None: return None
#         img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#
#         if is_mask:
#             if len(img.shape) == 3: img = img[:, :, 0]
#             t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
#         else:
#             if len(img.shape) == 3 and img.shape[2] == 4:
#                 b, g, r, a = cv2.split(img)
#                 img = cv2.merge((r, g, b))
#             else:
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
#
#         return t.unsqueeze(0).to(DEVICE)
#
#     def postprocess(self, tensor):
#         img = (tensor.detach().cpu().numpy()[0] + 1) / 2
#         img = np.clip(img, 0, 1)
#         img = np.transpose(img, (1, 2, 0))
#         img = (img * 255).astype(np.uint8)
#         return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#     def predict(self, agnostic_path, cloth_path, mask_path, skeleton_path, save_path):
#         with torch.no_grad():
#             agnostic = self.preprocess(agnostic_path)
#             cloth = self.preprocess(cloth_path)
#             mask = self.preprocess(mask_path, is_mask=True)
#             skeleton = self.preprocess(skeleton_path)
#
#             if any(x is None for x in [agnostic, cloth, mask, skeleton]):
#                 print(f"[ERROR] Could not load images. Check paths.")
#                 return
#
#             # Warp
#             warped_cloth, warped_mask, _ = self.warp(cloth, mask, skeleton)
#
#             # Generate
#             generated_image = torch.randn_like(agnostic).to(DEVICE)
#             for t in tqdm(self.scheduler.timesteps, desc="Processing", leave=False):
#                 inp = torch.cat([generated_image, warped_cloth, warped_mask, agnostic], dim=1)
#                 model_output = self.unet(inp, t).sample
#                 generated_image = self.scheduler.step(model_output, t, generated_image).prev_sample
#
#             # Save
#             final_img = self.postprocess(generated_image)
#             cv2.imwrite(save_path, final_img)
#             print(f"Result saved: {save_path}")
#
#
# # =====================
# # SINGLE FILE AUTO-DETECTION
# # =====================
# def get_first_file(folder):
#     """Returns the full path of the first valid image file in a folder."""
#     if not os.path.exists(folder):
#         return None
#     files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#     if len(files) > 0:
#         return os.path.join(folder, files[0])
#     return None
#
#
# if __name__ == "__main__":
#     predictor = WearifyPredictor()
#
#     print("\nScanning folders for input...")
#
#     # 1. Grab whatever file is in each folder
#     agnostic_p = get_first_file(AGNOSTIC_DIR)
#     cloth_p = get_first_file(CLOTH_DIR)
#     mask_p = get_first_file(MASK_DIR)
#     skeleton_p = get_first_file(SKELETON_DIR)
#
#     # 2. Check if we found everything
#     missing = []
#     if not agnostic_p: missing.append("Agnostic (Person)")
#     if not cloth_p:    missing.append("Cloth")
#     if not mask_p:     missing.append("Cloth Mask")
#     if not skeleton_p: missing.append("Skeleton")
#
#     if missing:
#         print("\n[ERROR] Missing input files in the following folders:")
#         for m in missing: print(f" - raw images/{m.lower().replace(' ', '-')}")
#         print("Please ensure there is at least 1 image in each 'raw images' subfolder.")
#     else:
#         print(f"Found inputs:")
#         print(f" - Person:   {os.path.basename(agnostic_p)}")
#         print(f" - Cloth:    {os.path.basename(cloth_p)}")
#         print(f" - Mask:     {os.path.basename(mask_p)}")
#         print(f" - Skeleton: {os.path.basename(skeleton_p)}")
#
#         # 3. Generate Output Name
#         # Result name combines person and cloth names for clarity
#         p_name = os.path.splitext(os.path.basename(agnostic_p))[0]
#         c_name = os.path.splitext(os.path.basename(cloth_p))[0]
#         output_path = os.path.join(OUTPUT_DIR, f"result_{p_name}_{c_name}.png")
#
#         # 4. Run Prediction
#         predictor.predict(agnostic_p, cloth_p, mask_p, skeleton_p, output_path)



import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm

# =====================
# CONFIGURATION
# =====================
CHECKPOINT_EPOCH = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256

# FOLDERS
INPUT_ROOT = "raw images"
AGNOSTIC_DIR = os.path.join(INPUT_ROOT, "agnostic")
CLOTH_DIR = os.path.join(INPUT_ROOT, "cloth-resized")
MASK_DIR = os.path.join(INPUT_ROOT, "cloth-mask")
SKELETON_DIR = os.path.join(INPUT_ROOT, "openpose-skeleton")

OUTPUT_DIR = "final_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# FLOW WARPING MODULE
# =====================
class FlowWarpingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1)
        )
        nn.init.constant_(self.decoder[-1].weight, 0)
        nn.init.constant_(self.decoder[-1].bias, 0)

    def forward(self, cloth, mask, skeleton):
        B, _, H, W = cloth.shape
        x = torch.cat([cloth, mask, skeleton], dim=1)

        feat = self.encoder(x)
        flow = self.decoder(feat)

        # === MATCH TRAINING ===
        flow = torch.clamp(flow, -1.5, 1.5)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=cloth.device),
            torch.linspace(-1, 1, W, device=cloth.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        final_grid = base_grid + flow.permute(0, 2, 3, 1)

        warped_cloth = F.grid_sample(cloth, final_grid, align_corners=True, padding_mode="border")
        warped_mask = F.grid_sample(mask, final_grid, align_corners=True, padding_mode="zeros")

        return warped_cloth, warped_mask, flow

# =====================
# PREDICTOR
# =====================
class WearifyPredictor:
    def __init__(self, checkpoint_dir="checkpoints"):
        print(f"Loading Models (Epoch {CHECKPOINT_EPOCH})...")

        self.warp = FlowWarpingModule().to(DEVICE)
        self.warp.load_state_dict(
            torch.load(f"{checkpoint_dir}/flow_{CHECKPOINT_EPOCH}.pt", map_location=DEVICE)
        )
        self.warp.eval()

        self.unet = UNet2DModel(
            sample_size=IMAGE_SIZE,
            in_channels=10,
            out_channels=3,
            block_out_channels=(32, 64, 128, 128)
        ).to(DEVICE)
        self.unet.load_state_dict(
            torch.load(f"{checkpoint_dir}/unet_{CHECKPOINT_EPOCH}.pt", map_location=DEVICE)
        )
        self.unet.eval()

        # === SCHEDULER FIX ===
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.scheduler.set_timesteps(1000)

        print("Ready for Inference.")

    def preprocess(self, img_path, is_mask=False):
        if not os.path.exists(img_path):
            return None

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        if is_mask:
            if len(img.shape) == 3:
                img = img[:, :, 0]
            t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            if len(img.shape) == 3 and img.shape[2] == 4:
                b, g, r, a = cv2.split(img)
                img = cv2.merge((r, g, b))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)

        return t.unsqueeze(0).to(DEVICE)

    def postprocess(self, tensor):
        img = (tensor.detach().cpu().numpy()[0] + 1) / 2
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def predict(self, agnostic_path, cloth_path, mask_path, skeleton_path, save_path):
        with torch.no_grad():
            agnostic = self.preprocess(agnostic_path)
            cloth = self.preprocess(cloth_path)
            mask = self.preprocess(mask_path, is_mask=True)
            skeleton = self.preprocess(skeleton_path)

            if any(x is None for x in [agnostic, cloth, mask, skeleton]):
                print("[ERROR] Could not load images.")
                return

            # === WARP ===
            warped_cloth, warped_mask, _ = self.warp(cloth, mask, skeleton)

            # === DDPM SAMPLING ===
            generated_image = torch.randn(
                (1, 3, IMAGE_SIZE, IMAGE_SIZE),
                device=DEVICE
            )

            for t in tqdm(self.scheduler.timesteps, desc="Denoising", leave=False):
                inp = torch.cat([generated_image, warped_cloth, warped_mask, agnostic], dim=1)
                noise_pred = self.unet(inp, t).sample
                generated_image = self.scheduler.step(
                    noise_pred, t, generated_image
                ).prev_sample

            final_img = self.postprocess(generated_image)
            cv2.imwrite(save_path, final_img)
            print(f"Result saved: {save_path}")

# =====================
# AUTO INPUT DETECTION
# =====================
def get_first_file(folder):
    if not os.path.exists(folder):
        return None
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    return os.path.join(folder, files[0]) if files else None

if __name__ == "__main__":
    predictor = WearifyPredictor()

    agnostic_p = get_first_file(AGNOSTIC_DIR)
    cloth_p = get_first_file(CLOTH_DIR)
    mask_p = get_first_file(MASK_DIR)
    skeleton_p = get_first_file(SKELETON_DIR)

    missing = []
    if not agnostic_p: missing.append("agnostic")
    if not cloth_p: missing.append("cloth")
    if not mask_p: missing.append("mask")
    if not skeleton_p: missing.append("skeleton")

    if missing:
        print("[ERROR] Missing files:", missing)
    else:
        p_name = os.path.splitext(os.path.basename(agnostic_p))[0]
        c_name = os.path.splitext(os.path.basename(cloth_p))[0]
        out_path = os.path.join(OUTPUT_DIR, f"result_{p_name}_{c_name}.png")

        predictor.predict(
            agnostic_p,
            cloth_p,
            mask_p,
            skeleton_p,
            out_path
        )

