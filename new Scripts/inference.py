import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.utils.torch_utils import randn_tensor
from peft import PeftModel
from ultralytics import YOLO

os.environ["HF_HUB_CACHE"] = r"D:\.cache\huggingface\hub"
os.environ["HF_HOME"] = "D:/.cache/huggingface"

# --- CONFIG ---
BASE_MODEL = "booksforcharlie/stable-diffusion-inpainting"
VAE_MODEL = "stabilityai/sd-vae-ft-mse"
OUTPUT_DIR = "../CatVTON/catvton_finetuned_v3"
EPOCH = 19
IMG_HEIGHT, IMG_WIDTH = 512, 320
DDIM_STEPS = 50
GUIDANCE = 2.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16


# ==============================
# MODEL LOADERS
# ==============================
def init_skip_cross_attn(unet):
    import torch.nn as nn
    class SkipAttnProcessor(nn.Module):
        def __call__(self, attn, hidden_states, **kwargs): return hidden_states

    unet.set_attn_processor(
        {name: (SkipAttnProcessor() if "attn2" in name else proc) for name, proc in unet.attn_processors.items()})
    return unet


def load_inference_models():
    print("Loading CatVTON and SD Inpainting Pipelines...")
    scheduler = DDIMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL).to(DEVICE, dtype=DTYPE).eval()
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
    unet = init_skip_cross_attn(unet)

    lora_path = os.path.join(OUTPUT_DIR, f"lora_epoch_{EPOCH}")
    if os.path.exists(lora_path):
        unet = PeftModel.from_pretrained(unet, lora_path, is_trainable=False).to(DEVICE).eval()
    unet.requires_grad_(False)

    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=DTYPE, variant="fp16",
        cache_dir="D:/.cache/huggingface/hub", local_files_only=True, safety_checker=None
    )
    sd_pipe.enable_model_cpu_offload()

    pose_model = YOLO("yolov8n-pose.pt", verbose=False).to(DEVICE)

    return {"vae": vae, "unet": unet, "scheduler": scheduler, "sd_pipe": sd_pipe, "yolo": pose_model}


# ==============================
# HELPERS
# ==============================
def preprocess_image(pil_img, h, w): return transforms.Normalize([0.5] * 3, [0.5] * 3)(
    transforms.ToTensor()(pil_img.resize((w, h), Image.LANCZOS))).unsqueeze(0)


def preprocess_mask(pil_mask, h, w): return transforms.ToTensor()(
    pil_mask.resize((w, h), Image.NEAREST).convert("L")).unsqueeze(0)


def tensor_to_pil(t): return transforms.ToPILImage()((t * 0.5 + 0.5).clamp(0, 1).cpu())


def preprocess_cloth_image(pil_img, height, width):
    arr = np.array(pil_img)
    is_cloth = ~((arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235))
    rows, cols = np.any(is_cloth, axis=1), np.any(is_cloth, axis=0)
    cloth_crop = pil_img.crop((int(np.argmax(cols)), int(np.argmax(rows)), len(cols) - int(np.argmax(cols[::-1])),
                               len(rows) - int(np.argmax(rows[::-1])))) if rows.any() else pil_img
    cw, ch = cloth_crop.size
    scale = min((height * 0.90) / ch, (width * 0.95) / cw)
    resized = cloth_crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return transforms.Normalize([0.5] * 3, [0.5] * 3)(transforms.ToTensor()(canvas)).unsqueeze(0)


def composite_hands(result_img_rgb, person_img_rgb, identity_mask_np):
    if identity_mask_np.sum() == 0: return result_img_rgb
    mask_blur = cv2.GaussianBlur(identity_mask_np, (11, 11), 0).astype(np.float32) / 255.0
    alpha = np.stack([mask_blur] * 3, axis=-1)
    return (person_img_rgb.astype(np.float32) * alpha + result_img_rgb.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8)


# ==============================
# MAIN API INFERENCE LOGIC
# ==============================
@torch.no_grad()
def run_vton(person_pil, cloth_pil, mask_pil, agnostic_pil, identity_mask_np, models):
    vae, unet, scheduler, sd_pipe = models["vae"], models["unet"], models["scheduler"], models["sd_pipe"]
    person_np = np.array(person_pil)

    # Standard VTON Prep
    person_t = preprocess_image(person_pil, IMG_HEIGHT, IMG_WIDTH).to(DEVICE, dtype=DTYPE)
    cloth_t = preprocess_cloth_image(cloth_pil, IMG_HEIGHT, IMG_WIDTH).to(DEVICE, dtype=DTYPE)
    agnostic_t = preprocess_image(agnostic_pil, IMG_HEIGHT, IMG_WIDTH).to(DEVICE, dtype=DTYPE)
    mask_t = preprocess_mask(mask_pil, IMG_HEIGHT, IMG_WIDTH).to(DEVICE, dtype=DTYPE)

    with torch.amp.autocast(DEVICE):
        person_latent = vae.encode(person_t).latent_dist.sample() * vae.config.scaling_factor
        cloth_latent = vae.encode(cloth_t).latent_dist.sample() * vae.config.scaling_factor
        agnostic_latent = vae.encode(agnostic_t).latent_dist.sample() * vae.config.scaling_factor

        mask_latent = F.interpolate(mask_t, size=person_latent.shape[-2:], mode="nearest").clamp(0, 1)
        masked_latent = agnostic_latent * (mask_latent < 0.5)

        masked_concat = torch.cat([masked_latent, cloth_latent], dim=-2)
        mask_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=-2)
        uncond_concat = torch.cat([masked_latent, torch.zeros_like(cloth_latent)], dim=-2)

        masked_concat_cfg = torch.cat([uncond_concat, masked_concat])
        mask_concat_cfg = torch.cat([mask_concat] * 2)

        latents = randn_tensor(
            masked_concat.shape,
            generator=torch.Generator(device=torch.device(DEVICE)),
            device=torch.device(DEVICE),
            dtype=DTYPE
        ) * scheduler.init_noise_sigma

        scheduler.set_timesteps(DDIM_STEPS)

        for t in scheduler.timesteps:
            lmi = torch.cat([latents] * 2)
            lmi = scheduler.scale_model_input(lmi, t)

            # This perfectly builds the 9-channel input (4 + 1 + 4 = 9)
            ui = torch.cat([lmi, mask_concat_cfg, masked_concat_cfg], dim=1)

            pred = unet(ui, t, encoder_hidden_states=None, return_dict=False)[0]
            u, c = pred.chunk(2)
            pred = u + GUIDANCE * (c - u)
            latents = scheduler.step(pred, t, latents).prev_sample

        result_latent = latents.split(latents.shape[-2] // 2, dim=-2)[0]
        scaled_latent = (result_latent / vae.config.scaling_factor).to(DEVICE, dtype=DTYPE)
        result = vae.decode(scaled_latent).sample.clamp(-1, 1)

    result_np = np.array(tensor_to_pil(result[0].float()).resize(person_pil.size, Image.LANCZOS))
    final_np = composite_hands(result_np, person_np, identity_mask_np)

    return ImageEnhance.Color(Image.fromarray(final_np)).enhance(1.4)