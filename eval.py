import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ------------------ SETTINGS ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoints/wearify_epoch_9.pt"  # your latest checkpoint
output_dir = "test_generated"
os.makedirs(output_dir, exist_ok=True)

json_file = r"our dataset/test/test.json"
dataset_root = r"our dataset/test/"

# ------------------ DATASET ------------------
class WearifyDataset(Dataset):
    def __init__(self, json_file, dataset_root):
        self.dataset_root = dataset_root
        self.data_list = []
        with open(json_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data_list.append(json.loads(line))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]

        def load_img(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = (img.astype(np.float32) / 127.5) - 1.0
            return torch.from_numpy(img).permute(2, 0, 1)

        def load_mask(path):
            mask = cv2.imread(path, 0)
            mask = cv2.resize(mask, (256, 256))
            mask = mask.astype(np.float32) / 255.0
            return torch.from_numpy(mask).unsqueeze(0)

        return {
            "target": load_img(item['target']),
            "agnostic": load_img(item['agnostic']),
            "cloth": load_img(item['cloth']),
            "cloth_mask": load_mask(item['cloth_mask']),
            "pose": load_mask(item['pose']),
            "parse": load_mask(item['parse']),
            "prompt": item['prompt']
        }

dataset = WearifyDataset(json_file, dataset_root)

# ------------------ TOKENIZER & TEXT ENCODER ------------------
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_encoder.eval()
for p in text_encoder.parameters():
    p.requires_grad = False

# ------------------ UNET ------------------
unet = UNet2DConditionModel(
    sample_size=256,
    in_channels=11,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=512
).to(device)

unet.load_state_dict(torch.load(checkpoint_path, map_location=device))
unet.eval()

# ------------------ NOISE SCHEDULER ------------------
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")

# ------------------ GENERATION & METRICS ------------------
psnr_list = []
ssim_list = []

for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
    # Prepare conditioning
    agnostic = sample["agnostic"].unsqueeze(0).to(device)
    pose = sample["pose"].unsqueeze(0).to(device)
    cloth = sample["cloth"].unsqueeze(0).to(device)
    mask = sample["cloth_mask"].unsqueeze(0).to(device)
    cond = torch.cat([agnostic, pose, cloth, mask], dim=1)

    # Text conditioning
    tokens = tokenizer(sample["prompt"], padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = text_encoder(**tokens).last_hidden_state

    # Initialize pure noise
    latents = torch.randn((1, 3, 256, 256), device=device)

    # Multi-step DDPM sampling
    timesteps = noise_scheduler.timesteps[::20]
    for t in tqdm(timesteps, desc=f"Sampling image {idx}", leave=False):
        model_input = torch.cat([latents, cond], dim=1)
        with torch.no_grad():
            noise_pred = unet(model_input, torch.tensor([t], device=device), encoder_hidden_states=text_embeds).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # Convert to image
    out_img = (latents.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)

    # Ground truth
    target_img = (sample["target"].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
    target_img = np.clip(target_img, 0, 255).astype(np.uint8)

    # Save image
    save_path = os.path.join(output_dir, f"generated_{idx}.png")
    cv2.imwrite(save_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    # Compute metrics
    psnr_val = psnr(target_img, out_img, data_range=255)
    ssim_val = ssim(target_img, out_img, data_range=255, channel_axis=2)

    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

# ------------------ PRINT AVERAGE METRICS ------------------
print(f"Average PSNR: {np.mean(psnr_list):.4f}")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")