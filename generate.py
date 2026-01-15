import os, json, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm

# ================== SETTINGS ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoints/wearify_epoch_9.pt"   # change if needed
output_path = "generated_test1.png"
sample_index = 0   # which test image to generate
img_size = 256
# ==============================================


# ================== DATASET ==================
json_file = r"our dataset/test/test.json"

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
        img = (img.astype(np.float32) / 127.5) - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def load_mask(self, path):
        mask = cv2.imread(path, 0)
        mask = cv2.resize(mask, (img_size, img_size))
        mask = mask.astype(np.float32) / 255.0
        return torch.from_numpy(mask).unsqueeze(0)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "target": self.load_img(item["target"]),
            "agnostic": self.load_img(item["agnostic"]),
            "cloth": self.load_img(item["cloth"]),
            "cloth_mask": self.load_mask(item["cloth_mask"]),
            "pose": self.load_mask(item["pose"]),
            "parse": self.load_mask(item["parse"]),
            "prompt": item["prompt"]
        }


dataset = WearifyDataset(json_file)
sample = dataset[sample_index]

# ================== TEXT ENCODER ==================
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

text_encoder.eval()
for p in text_encoder.parameters():
    p.requires_grad = False

tokens = tokenizer(
    sample["prompt"],
    padding="max_length",
    truncation=True,
    max_length=77,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    text_embeds = text_encoder(**tokens).last_hidden_state

# ================== LOAD UNET ==================
unet = UNet2DConditionModel(
    sample_size=img_size,
    in_channels=11,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    ),
    up_block_types=(
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    cross_attention_dim=512,
).to(device)

unet.load_state_dict(torch.load(checkpoint_path, map_location=device))
unet.eval()

# ================== SCHEDULER ==================
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear"
)
scheduler.set_timesteps(1000)

# ================== CONDITIONING ==================
agnostic = sample["agnostic"].unsqueeze(0).to(device)
pose = sample["pose"].unsqueeze(0).to(device)
cloth = sample["cloth"].unsqueeze(0).to(device)
mask = sample["cloth_mask"].unsqueeze(0).to(device)

cond = torch.cat([agnostic, pose, cloth, mask], dim=1)

# ================== DIFFUSION SAMPLING ==================
latents = torch.randn((1, 3, img_size, img_size), device=device)

for t in tqdm(scheduler.timesteps, desc="Generating"):
    model_input = torch.cat([latents, cond], dim=1)

    with torch.no_grad():
        noise_pred = unet(
            model_input,
            t,
            encoder_hidden_states=text_embeds
        ).sample

    latents = scheduler.step(noise_pred, t, latents).prev_sample

# ================== SAVE IMAGE ==================
out = latents.squeeze(0).permute(1, 2, 0).cpu().numpy()
out = (out + 1.0) * 127.5
out = np.clip(out, 0, 255).astype(np.uint8)

cv2.imwrite(output_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
print(f"✅ Image generated and saved to {output_path}")