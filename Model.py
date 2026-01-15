import os, json, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

json_file = r"our dataset/train/train.json"
dataset_root = r"our dataset/train/"

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

        img_path = item['target']
        agnostic_path = item['agnostic']
        cloth_path = item['cloth']
        mask_path = item['cloth_mask']
        pose_path = item['pose']
        parse_path = item['parse']
        prompt = item['prompt']

        def load_img(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = (img.astype(np.float32) / 127.5) - 1.0
            return torch.from_numpy(img).permute(2, 0, 1)

        def load_mask(path):
            mask = cv2.imread(path, 0)  # Grayscale
            mask = cv2.resize(mask, (256, 256))
            mask = mask.astype(np.float32) / 255.0
            return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)


        return {
            "target": load_img(img_path),
            "agnostic": load_img(agnostic_path),
            "cloth": load_img(cloth_path),
            "cloth_mask": load_mask(mask_path),
            "pose": load_mask(pose_path),
            "parse": load_mask(parse_path),
            "prompt": prompt
        }

dataset = WearifyDataset(json_file, dataset_root)
train_loader = DataLoader(
    dataset,
    batch_size=1,      # Process one pair at a time for 6GB VRAM
    shuffle=True,      # Shuffle to prevent the model from memorizing order
    num_workers=0,     # Use CPU to pre-fetch data while GPU trains
    pin_memory=True    # Speeds up data transfer from RAM to GPU
)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)
text_encoder.eval()

for p in text_encoder.parameters():
    p.requires_grad = False


unet = UNet2DConditionModel(
    sample_size=256,            # or 768 if VRAM allows
    in_channels=11,
    out_channels=3,             # predict noise
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

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear"
)

resume_epoch = 9
checkpoint_path = f"checkpoints/wearify_epoch_{resume_epoch}.pt"

if os.path.exists(checkpoint_path):
    print(f"Resuming training from epoch {resume_epoch}")
    unet.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print("No checkpoint found, starting from scratch")
    resume_epoch = -1

optimizer = AdamW(unet.parameters(), lr=1e-4)

total_epochs = 20
grad_accum = 2   # helps 6GB VRAM

unet.train()

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(resume_epoch + 1, total_epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        target = batch["target"].to(device)
        agnostic = batch["agnostic"].to(device)
        pose = batch["pose"].to(device)
        cloth = batch["cloth"].to(device)
        mask = batch["cloth_mask"].to(device)

        # ---- conditioning ----
        cond = torch.cat([agnostic, pose, cloth, mask], dim=1)

        # ---- text ----
        tokens = tokenizer(
            batch["prompt"],
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            text_embeds = text_encoder(**tokens).last_hidden_state

        # ---- diffusion ----
        noise = torch.randn_like(target)
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps,
            (target.size(0),),
            device=device
        ).long()

        noisy_target = noise_scheduler.add_noise(
            target, noise, timesteps
        )

        model_input = torch.cat([noisy_target, cond], dim=1)

        noise_pred = unet(
            model_input,
            timesteps,
            encoder_hidden_states=text_embeds,
        ).sample

        loss = F.mse_loss(noise_pred, noise)
        loss.backward()

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=loss.item())

    torch.save(unet.state_dict(), f"checkpoints/wearify_epoch_{epoch}.pt")
