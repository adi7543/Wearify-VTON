import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ================== SETTINGS ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoints/wearify_epoch_9.pt"
output_dir = "eval_results"
os.makedirs(output_dir, exist_ok=True)
img_size = 256
json_file = r"our dataset/test/test.json"


# ================== DATASET (Same as yours) ==================
class WearifyDataset(Dataset):
    def __init__(self, json_file):
        self.data = []
        with open(json_file, "r") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img_raw = img.copy()  # For metric calculation
        img = (img.astype(np.float32) / 127.5) - 1.0
        return torch.from_numpy(img).permute(2, 0, 1), img_raw

    def load_mask(self, path):
        mask = cv2.imread(path, 0)
        mask = cv2.resize(mask, (img_size, img_size))
        mask = mask.astype(np.float32) / 255.0
        return torch.from_numpy(mask).unsqueeze(0)

    def __getitem__(self, idx):
        item = self.data[idx]
        target_tensor, target_raw = self.load_img(item["target"])
        agnostic_tensor, _ = self.load_img(item["agnostic"])
        cloth_tensor, _ = self.load_img(item["cloth"])

        return {
            "target": target_tensor,
            "target_raw": target_raw,
            "agnostic": agnostic_tensor,
            "cloth": cloth_tensor,
            "cloth_mask": self.load_mask(item["cloth_mask"]),
            "pose": self.load_mask(item["pose"]),
            "prompt": item["prompt"],
            "filename": os.path.basename(item["target"])
        }


# ================== LOAD MODELS ==================
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_encoder.eval()

unet = UNet2DConditionModel(
    sample_size=img_size,
    in_channels=11,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=512,
).to(device)

unet.load_state_dict(torch.load(checkpoint_path, map_location=device))
unet.eval()

scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")
scheduler.set_timesteps(1000)

# ================== EVALUATION LOOP ==================
dataset = WearifyDataset(json_file)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

all_ssim = []
all_psnr = []

for batch in tqdm(dataloader, desc="Evaluating Test Set"):
    # Conditionings
    agnostic = batch["agnostic"].to(device)
    pose = batch["pose"].to(device)
    cloth = batch["cloth"].to(device)
    mask = batch["cloth_mask"].to(device)
    prompt = batch["prompt"]
    target_raw = batch["target_raw"].squeeze(0).numpy()

    cond = torch.cat([agnostic, pose, cloth, mask], dim=1)

    # Text Embeddings
    tokens = tokenizer(prompt, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = text_encoder(**tokens).last_hidden_state

    # Generation
    latents = torch.randn((1, 3, img_size, img_size), device=device)
    for t in scheduler.timesteps:
        model_input = torch.cat([latents, cond], dim=1)
        with torch.no_grad():
            noise_pred = unet(model_input, t, encoder_hidden_states=text_embeds).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Post-process for Image Saving & Metrics
    out = latents.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = (out + 1.0) * 127.5
    generated_img = np.clip(out, 0, 255).astype(np.uint8)

    # Calculate Metrics against Target Raw
    batch_ssim = ssim(target_raw, generated_img, channel_axis=2, data_range=255)
    batch_psnr = psnr(target_raw, generated_img, data_range=255)

    all_ssim.append(batch_ssim)
    all_psnr.append(batch_psnr)

    # Save Side-by-Side (Agnostic | Cloth | Target | Generated)
    agnostic_img = ((batch["agnostic"].squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
    cloth_img = ((batch["cloth"].squeeze(0).permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)

    # Optional: Convert to BGR for OpenCV saving
    comparison = np.hstack([
        cv2.cvtColor(agnostic_img, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(cloth_img, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(target_raw, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(generated_img, cv2.COLOR_RGB2BGR)
    ])
    cv2.imwrite(os.path.join(output_dir, batch["filename"][0]), comparison)

# ================== FINAL RESULTS ==================
print("\n--- Evaluation Summary ---")
print(f"Total Images Evaluated: {len(all_ssim)}")
print(f"Average SSIM: {np.mean(all_ssim):.4f}")
print(f"Average PSNR: {np.mean(all_psnr):.2f} dB")
print(f"Results saved to: {output_dir}")