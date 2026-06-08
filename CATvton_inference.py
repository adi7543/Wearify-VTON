import os
import sys
import cv2 as _cv2

# Set HuggingFace cache path
os.environ["HF_HUB_CACHE"] = r"D:\.cache\huggingface\hub"
os.environ["HF_HOME"] = "D:/.cache/huggingface"

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from peft import PeftModel

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==============================
# CONFIG
# ==============================

DATASET_DIR = "../dataset_final"
BASE_MODEL = "runwayml/stable-diffusion-inpainting"
VAE_MODEL = "stabilityai/sd-vae-ft-mse"
OUTPUT_DIR = "catvton_finetuned_v4"
EPOCH = 20

IMG_HEIGHT = 512
IMG_WIDTH = 512
DDIM_STEPS = 30
GUIDANCE = 2.5
DEVICE = "cuda"

RESULTS_DIR = "catvton_finetuned_v4/results"
BATCH_MODE = False
TEST_PAIRS = "test_pairs.txt"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# CATVTON HELPER
# ==============================

def init_skip_cross_attn(unet):
    """Replace cross-attention with skip processors — no text conditioning."""
    import torch.nn as nn

    class SkipAttnProcessor(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def __call__(self, attn, hidden_states,
                     encoder_hidden_states=None,
                     attention_mask=None, temb=None, **kwargs):
            return hidden_states

    attn_procs = {}
    for name, proc in unet.attn_processors.items():
        if "attn2" in name:
            attn_procs[name] = SkipAttnProcessor()
        else:
            attn_procs[name] = proc
    unet.set_attn_processor(attn_procs)
    return unet


# ==============================
# LOAD MODELS
# ==============================

def load_catvton_attn_weights(unet, ckpt_path, version="mix"):
    sub_folder = {
        "mix": "mix-48k-1024",
        "vitonhd": "vitonhd-16k-512",
        "dresscode": "dresscode-16k-512",
    }[version]

    attn_path = os.path.join(ckpt_path, sub_folder, "attention", "model.safetensors")
    if not os.path.exists(attn_path):
        print(f"WARNING: CatVTON attention weights not found at {attn_path}")
        return unet

    try:
        from safetensors.torch import load_file
        from collections import defaultdict

        state_dict = load_file(attn_path)

        attn1_modules = []
        for name, module in unet.named_modules():
            if (name.endswith("attn1") and
                    hasattr(module, "to_q") and
                    hasattr(module, "to_k")):
                attn1_modules.append((name, module))

        ckpt_indices = sorted(set(int(k.split(".")[0]) for k in state_dict.keys()))
        idx_map = {i: ckpt_idx for i, ckpt_idx in enumerate(ckpt_indices)}

        loaded = skipped = 0
        for our_idx, (name, module) in enumerate(attn1_modules):
            ckpt_idx = idx_map.get(our_idx)
            if ckpt_idx is None:
                continue
            module_weights = {
                k.split(".", 1)[1]: v
                for k, v in state_dict.items()
                if k.startswith(f"{ckpt_idx}.")
            }
            for attr, weight in module_weights.items():
                parts = attr.split(".")
                obj = module
                try:
                    for p in parts[:-1]:
                        obj = getattr(obj, p)
                    param = getattr(obj, parts[-1])
                    if isinstance(param, torch.nn.Parameter):
                        if param.shape == weight.shape:
                            with torch.no_grad():
                                param.copy_(weight.to(param.device))
                            loaded += 1
                        else:
                            skipped += 1
                except AttributeError:
                    skipped += 1

        print(f"  CatVTON attention weights loaded: {loaded}/{loaded + skipped}")

    except Exception as e:
        print(f"WARNING: Could not load attention weights: {e}")

    return unet


def load_models(epoch=EPOCH):
    print(f"Loading models (epoch {epoch}) …")

    scheduler = DDIMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(
        VAE_MODEL
    ).to(DEVICE, dtype=torch.float16).eval()
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
    unet = init_skip_cross_attn(unet)

    # load pretrained CatVTON attention weights
    CATVTON_CKPT = r"D:\.cache\huggingface\hub\models--zhengchong--CatVTON\snapshots\2969fcf85fe62f2036605716f0b56f0b81d01d79"
    unet = load_catvton_attn_weights(unet, CATVTON_CKPT, version="mix")

    lora_path = os.path.join(OUTPUT_DIR, f"lora_epoch_{epoch}")
    unet = PeftModel.from_pretrained(unet, lora_path, is_trainable=False).to(DEVICE).eval()
    unet.requires_grad_(False)
    print(f"  LoRA loaded : {lora_path}")

    print("Models loaded.\n")
    return vae, unet, scheduler


# ==============================
# PREPROCESSING
# ==============================
def preprocess_image(pil_img, height, width):
    """Resize and normalize image to [-1, 1] tensor."""
    pil_img = pil_img.resize((width, height), Image.LANCZOS)
    t = transforms.ToTensor()(pil_img)
    t = transforms.Normalize([0.5] * 3, [0.5] * 3)(t)
    return t.unsqueeze(0)


def preprocess_mask(pil_mask, height, width):
    """Resize mask to [0,1] tensor."""
    pil_mask = pil_mask.resize((width, height), Image.NEAREST).convert("L")
    t = transforms.ToTensor()(pil_mask)
    return t.unsqueeze(0)


def tensor_to_pil(t):
    return transforms.ToPILImage()((t * 0.5 + 0.5).clamp(0, 1).cpu())


def preprocess_cloth_image(pil_img, height, width):
    """
    For cloth images — resize to fill more of the frame
    so the model sees a larger garment and generates
    correct length and sleeves.
    """
    arr = np.array(pil_img)
    is_cloth = ~((arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235))
    rows = np.any(is_cloth, axis=1)
    cols = np.any(is_cloth, axis=0)
    if rows.any():
        r0 = int(np.argmax(rows));
        r1 = len(rows) - int(np.argmax(rows[::-1]))
        c0 = int(np.argmax(cols));
        c1 = len(cols) - int(np.argmax(cols[::-1]))
        cloth_crop = pil_img.crop((c0, r0, c1, r1))
    else:
        cloth_crop = pil_img

    cw, ch = cloth_crop.size
    scale = (height * 0.90) / ch
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    if new_w > width * 0.95:
        scale = (width * 0.95) / cw
        new_w = int(cw * scale)
        new_h = int(ch * scale)

    resized = cloth_crop.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    ox = (width - new_w) // 2
    oy = (height - new_h) // 2
    canvas.paste(resized, (ox, oy))

    t = transforms.ToTensor()(canvas)
    t = transforms.Normalize([0.5] * 3, [0.5] * 3)(t)
    return t.unsqueeze(0)


import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# ==============================
# CONSTANTS
# ==============================
GREY_OVERLAP_PX = 20
GREY = (128, 128, 128)  # RGB


# ==============================
# HELPER: 4-POINT GEOMETRY
# ==============================
def get_4_points_and_height(mask_np):
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return {
        "shoulders": [(x, y), (x + w, y)],
        "bottom":    [(x, y + h), (x + w, y + h)],
        "height":    h,
        "y_bottom":  y + h,
        "x_left":    x,
        "x_right":   x + w
    }

# ==============================
# HELPER: HAND COMPOSITING
# ==============================
def composite_hands(result_img_rgb, person_img_rgb, identity_mask_path):
    if not identity_mask_path or not os.path.exists(identity_mask_path):
        return result_img_rgb

    identity_mask = cv2.imread(identity_mask_path, cv2.IMREAD_GRAYSCALE)
    if identity_mask is None:
        return result_img_rgb

    if identity_mask.ndim == 3:
        identity_mask = identity_mask[:, :, 0]

    if identity_mask.shape[:2] != result_img_rgb.shape[:2]:
        identity_mask = cv2.resize(
            identity_mask,
            (result_img_rgb.shape[1], result_img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    identity_mask_blur = cv2.GaussianBlur(identity_mask, (11, 11), 0)
    alpha = identity_mask_blur.astype(np.float32) / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=-1)

    person_float = person_img_rgb.astype(np.float32)
    result_float = result_img_rgb.astype(np.float32)

    composited = person_float * alpha + result_float * (1.0 - alpha)
    return composited.astype(np.uint8)


# ==============================
# HELPER: SUBTRACT HAND REGION FROM VTON MASK
# ==============================
def subtract_hands_from_mask(mask_np, identity_mask_path):
    if not identity_mask_path or not os.path.exists(identity_mask_path):
        return mask_np

    hand_mask = cv2.imread(identity_mask_path, cv2.IMREAD_GRAYSCALE)
    if hand_mask is None:
        return mask_np

    if hand_mask.ndim == 3:
        hand_mask = hand_mask[:, :, 0]

    if hand_mask.shape[:2] != mask_np.shape[:2]:
        hand_mask = cv2.resize(
            hand_mask,
            (mask_np.shape[1], mask_np.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    result = mask_np.copy()
    result[hand_mask > 127] = 0
    return result


def preprocess_cloth_pil(pil_img, height, width, scale=0.65):
    arr = np.array(pil_img)
    is_cloth = ~((arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235))
    rows = np.any(is_cloth, axis=1)
    cols = np.any(is_cloth, axis=0)
    if rows.any():
        r0 = int(np.argmax(rows));     r1 = len(rows) - int(np.argmax(rows[::-1]))
        c0 = int(np.argmax(cols));     c1 = len(cols) - int(np.argmax(cols[::-1]))
        pil_img = pil_img.crop((c0, r0, c1, r1))

    cw, ch = pil_img.size
    # Always force to exactly scale% of the frame
    target_h = int(height * scale)
    target_w = int(width  * scale)
    r = min(target_w / cw, target_h / ch)  # preserve aspect ratio
    new_w = int(cw * r)
    new_h = int(ch * r)

    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    ox = (width  - new_w) // 2
    oy = (height - new_h) // 2
    canvas.paste(resized, (ox, oy))

    t = transforms.ToTensor()(canvas)
    t = transforms.Normalize([0.5] * 3, [0.5] * 3)(t)
    return t.unsqueeze(0)


# ==============================
# MAIN INFERENCE
# ==============================

@torch.no_grad()
def run_inference(
        person_path, cloth_path, mask_path, agnostic_path, output_path,
        vae, unet, scheduler, sd_pipe=True, identity_mask_path=None,
        save_grid=True,
):
    # ==============================
    # DEBUG HELPER SETUP
    # ==============================
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(output_path)), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_step = 0

    def save_debug(img_data, name):
        """Helper to save intermediate PIL images or NumPy arrays."""
        nonlocal debug_step
        debug_step += 1
        filename = f"{debug_step:02d}_{name}.png"
        filepath = os.path.join(debug_dir, filename)

        if isinstance(img_data, np.ndarray):
            # If it's a 2D array, treat as grayscale mask
            if len(img_data.shape) == 2:
                img = Image.fromarray(img_data.astype(np.uint8), mode='L')
            else:
                img = Image.fromarray(img_data.astype(np.uint8))
        else:
            img = img_data.copy()

        img.save(filepath)
        print(f"  [Debug] Saved -> {filename}")

    # ==============================
    # INITIAL INPUTS
    # ==============================
    person_pil_original = Image.open(person_path).convert("RGB")
    cloth_pil = Image.open(cloth_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")
    agnostic_pil = Image.open(agnostic_path).convert("RGB")

    save_debug(person_pil_original, "inputs_person")
    save_debug(mask_pil, "inputs_mask")
    save_debug(agnostic_pil, "inputs_agnostic")

    inpainted_person_pil = person_pil_original.copy()
    height_diff = 0
    use_fill_cloth_preprocess = True

    # ==============================
    # PRE-FILL: GEOMETRY & AGNOSTIC EXTENSION
    # ==============================
    if sd_pipe is not None:
        print("Analyzing garment geometries...")

        source_mask_np = np.array(mask_pil)
        cloth_np = np.array(cloth_pil)

        gray_cloth = cv2.cvtColor(cloth_np, cv2.COLOR_RGB2GRAY)
        _, target_mask_np = cv2.threshold(gray_cloth, 240, 255, cv2.THRESH_BINARY_INV)

        source_geo = get_4_points_and_height(source_mask_np)
        target_geo = get_4_points_and_height(target_mask_np)

        if source_geo and target_geo:
            img_width, img_height = person_pil_original.size
            person_np = np.array(person_pil_original)
            agnostic_np = np.array(agnostic_pil)

            if agnostic_np.shape != person_np.shape:
                agnostic_np = cv2.resize(agnostic_np, (img_width, img_height))

            shoulder_y = source_geo["shoulders"][0][1]
            height_diff = source_geo["height"] - target_geo["height"]
            source_bottom = source_geo["y_bottom"]
            target_bottom = int(min(shoulder_y + target_geo["height"], img_height))
            cutoff_y = target_bottom

            print(f"  shoulder_y={shoulder_y}, source_bottom={source_bottom}, "
                  f"target_bottom={target_bottom}, height_diff={height_diff}")

            # ==========================================
            # CASE A: Target SHORTER — SD fills the gap
            # ========================================
            if height_diff > 15:
                print(f"[Case A] Target shorter by {height_diff}px. Running SD inpainting...")

                from diffusers import StableDiffusionInpaintPipeline

                dEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                DTYPE = torch.float16

                print("Loading SD Inpainting Pipeline into VRAM...")
                sd = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=DTYPE,
                    variant="fp16",
                    cache_dir="D:/.cache/huggingface/hub",
                    local_files_only=True,
                    safety_checker=None,
                    use_safetensors=False
                ).to(dEVICE)

                clean_composite = person_np.copy()
                clean_composite[0:cutoff_y, :] = agnostic_np[0:cutoff_y, :]
                save_debug(clean_composite, "caseA_clean_composite")  # DEBUG

                # ---------------------------------------------------------
                # 1. CREATE A MUCH TALLER, HEAVILY BLURRED MASK
                # ---------------------------------------------------------
                # We will expand the masked area significantly higher into the sherwani hem
                # and slightly lower into the actual trousers to give SD room to transition fabric.
                extended_cutoff = max(0, cutoff_y - 5)  # Larger expansion upwards
                extended_bottom = min(img_height, source_bottom + 10)  # Slight expansion downwards

                inpaint_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                inpaint_mask[extended_cutoff:extended_bottom, source_geo["x_left"]:source_geo["x_right"]] = 255

                # Apply massive blur (use a large, odd kernel like 51x51 or even larger if needed)
                # This creates smooth feathering rather than a harsh seam.
                inpaint_mask = cv2.GaussianBlur(inpaint_mask, (51, 51), 0)

                # CRUCIAL: Keep the upper face/body perfectly black so we don't accidentally blur
                # anything SD might interpret as part of the new top.
                inpaint_mask[0:extended_cutoff - 30, :] = 0  # Black out significantly higher than cutoff

                save_debug(inpaint_mask, "caseA_inpaint_mask_blurred")  # DEBUG

                torch.cuda.empty_cache()

                # ---------------------------------------------------------
                # 2. AGGRESSIVE, DYNAMIC PRE-FILL & SMOOTHING
                # ---------------------------------------------------------
                # By sampling the trousers dynamically and aggressively painting over the gap,
                # we delete the sherwani embroidery completely before SD can see it.
                sd_input_np = np.array(person_pil_original)

                # Dynamically sample the average color of the actual trousers *below* the garment
                # Sampling a slightly larger and dynamically calculated range for robustness
                sample_y_start = min(source_bottom + 10, img_height - 1)
                sample_y_end = min(source_bottom + 40, img_height)

                trouser_color = np.mean(
                    sd_input_np[sample_y_start:sample_y_end, source_geo["x_left"]:source_geo["x_right"]],
                    axis=(0, 1)
                ).astype(np.uint8)

                # Paint the sampled trouser color over the *entire gap and the extended mask area*
                # physically destroying the embroidery and gold pattern under the blurred mask.
                sd_input_np[extended_cutoff:source_bottom, source_geo["x_left"]:source_geo["x_right"]] = trouser_color

                # Add a slight blur just to the edges of our painted block for extra smoothness
                # inside the AI input image itself.
                block_mask = np.zeros_like(sd_input_np)
                block_mask[extended_cutoff:source_bottom, source_geo["x_left"]:source_geo["x_right"]] = 1
                sd_input_blurred = cv2.GaussianBlur(sd_input_np, (31, 31), 0)
                sd_input_np = np.where(block_mask == 1, sd_input_blurred, sd_input_np)

                save_debug(sd_input_np, "caseA_sd_input_prefilled_blurred")  # Check this debug image!
                # ---------------------------------------------------------

                # ---------------------------------------------------------
                # 3. RUN INPAINTING WITH REINFORCED PROMPTS
                # ---------------------------------------------------------
                prompt = (
                    "white trousers, baggy trousers, continuous fabric folds, seamless background, high quality"
                )
                negative_prompt = (
                    "sherwani, embroidery, pattern, gold, brown, shirt tail, kurta, top garment, blurry, mutated, seams, hands"
                )

                inpainted_image = sd(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=Image.fromarray(sd_input_np),  # Use the pre-filled, slightly smoothed input image
                    mask_image=Image.fromarray(inpaint_mask),  # Use the large, heavily blurred mask
                    num_inference_steps=25,
                    strength=0.99  # Maximum replacement strength
                ).images[0]
                save_debug(inpainted_image, "caseA_raw_sd_inpaint")  # DEBUG

                inpainted_np = np.array(inpainted_image)
                if inpainted_np.shape[:2] != (img_height, img_width):
                    inpainted_np = cv2.resize(inpainted_np, (img_width, img_height))

                # inpainted_np[0:cutoff_y, :] = clean_composite[0:cutoff_y, :]
                inpainted_np[0:cutoff_y, :] = agnostic_np[0:cutoff_y, :]
                save_debug(inpainted_np, "caseA_stitched_inpaint")  # DEBUG

                agnostic_pil = Image.fromarray(inpainted_np)
                inpainted_person_pil = agnostic_pil.copy()

                new_mask_np = np.zeros_like(source_mask_np)
                new_mask_np[0:cutoff_y, :] = source_mask_np[0:cutoff_y, :]
                new_mask_np = subtract_hands_from_mask(new_mask_np, identity_mask_path)
                save_debug(new_mask_np, "caseA_new_mask")  # DEBUG
                mask_pil = Image.fromarray(new_mask_np)

            # ==========================================
            # CASE B: Target LONGER — grey fill extension
            # ==========================================
            else:
                print(f"[Case B] Target longer by {abs(height_diff)}px. Applying grey fill...")
                use_fill_cloth_preprocess = False
                extended = agnostic_np.copy()

                fill_y_start = max(0, source_bottom - GREY_OVERLAP_PX)  # Assuming GREY_OVERLAP_PX is defined
                fill_y_end = min(target_bottom, img_height)
                fill_x_left = source_geo["x_left"]
                fill_x_right = source_geo["x_right"]

                print(f"  fill_y=[{fill_y_start}:{fill_y_end}], "
                      f"fill_x=[{fill_x_left}:{fill_x_right}]")

                if fill_y_start < fill_y_end:
                    extended[fill_y_start:fill_y_end,
                    fill_x_left:fill_x_right] = GREY  # Assuming GREY is defined

                save_debug(extended, "caseB_grey_fill_extended")  # DEBUG

                agnostic_with_hands = composite_hands(extended, person_np, identity_mask_path)
                save_debug(agnostic_with_hands, "caseB_agnostic_with_hands")  # DEBUG

                agnostic_pil = Image.fromarray(agnostic_with_hands)

                new_mask_np = source_mask_np.copy()
                if fill_y_start < fill_y_end:
                    new_mask_np[source_bottom:fill_y_end,
                    fill_x_left:fill_x_right] = 255
                new_mask_np = subtract_hands_from_mask(new_mask_np, identity_mask_path)
                save_debug(new_mask_np, "caseB_new_mask")  # DEBUG
                mask_pil = Image.fromarray(new_mask_np)

    # ==============================
    # STANDARD VTON LATENT PREP
    # ==============================
    person_t = preprocess_image(person_pil_original, IMG_HEIGHT, IMG_WIDTH).to(DEVICE, dtype=torch.float16)
    agnostic_t = preprocess_image(agnostic_pil, IMG_HEIGHT, IMG_WIDTH).to(DEVICE, dtype=torch.float16)
    mask_t = preprocess_mask(mask_pil, IMG_HEIGHT, IMG_WIDTH).to(DEVICE, dtype=torch.float16)
    if use_fill_cloth_preprocess:
        # Case A (shorter target) or no geometry — scale cloth to fill frame
        cloth_t = preprocess_cloth_image(cloth_pil, IMG_HEIGHT, IMG_WIDTH).to(DEVICE, dtype=torch.float16)
    else:
        # Case B (longer target) — keep natural proportions
        cloth_t = preprocess_image(cloth_pil, IMG_HEIGHT, IMG_WIDTH).to(DEVICE, dtype=torch.float16)

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

        generator = torch.Generator(device=torch.device(DEVICE))
        scheduler.set_timesteps(DDIM_STEPS)
        latents = randn_tensor(
            masked_concat.shape, generator=generator,
            device=torch.device(DEVICE), dtype=masked_concat.dtype
        ) * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            lmi = torch.cat([latents] * 2)
            lmi = scheduler.scale_model_input(lmi, t)
            ui = torch.cat([lmi, mask_concat_cfg, masked_concat_cfg], dim=1)
            pred = unet(ui, t, encoder_hidden_states=None, return_dict=False)[0]
            u, c = pred.chunk(2)
            pred = u + GUIDANCE * (c - u)
            latents = scheduler.step(pred, t, latents).prev_sample

        result_latent = latents.split(latents.shape[-2] // 2, dim=-2)[0]
        scaled_latent = (result_latent / vae.config.scaling_factor).to(DEVICE, dtype=torch.float16)
        result = vae.decode(scaled_latent).sample.clamp(-1, 1)

    # ==============================
    # COMPOSITING
    # ==============================
    result_pil_raw = tensor_to_pil(result[0].float())
    save_debug(result_pil_raw, "vton_raw_result")  # DEBUG

    # Resize raw output to match original image dimensions
    result_np = np.array(result_pil_raw.resize(
        (person_pil_original.width, person_pil_original.height), Image.LANCZOS
    ))

    # Directly composite identity (hands and face) onto the raw output
    print("Compositing identity (hands/face) onto raw output...")
    final_np = composite_hands(result_np, np.array(person_pil_original), identity_mask_path)

    save_debug(final_np, "final_composited_identity_only")  # DEBUG

    # Save final result
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result_pil = Image.fromarray(final_np)

    from PIL import ImageEnhance
    result_pil = ImageEnhance.Color(result_pil).enhance(1.4)
    result_pil.save(output_path)

    # ==============================
    # GRID GENERATION
    # ==============================
    if save_grid:
        cols = [
            ("Person", person_pil_original),
            ("Agnostic", agnostic_pil),
            ("Cloth", cloth_pil),
            ("Raw out", result_pil_raw),
            ("Final", result_pil),
        ]
        lh = 18
        W, H = IMG_WIDTH, IMG_HEIGHT
        grid = Image.new("RGB", (W * len(cols) + 2 * (len(cols) - 1), H + lh + 2), (40, 40, 40))
        try:
            font = ImageFont.truetype("arial.ttf", lh)
        except Exception:
            font = ImageFont.load_default()

        for i, (lbl, img) in enumerate(cols):
            cell = Image.new("RGB", (W, H + lh + 2), (30, 30, 30))
            cell.paste(img.resize((W, H), Image.LANCZOS), (0, lh + 2))
            ImageDraw.Draw(cell).text((2, 1), lbl, fill=(220, 220, 220), font=font)
            grid.paste(cell, (i * (W + 2), 0))

        grid_path = output_path.rsplit(".", 1)[0] + "_grid.jpg"
        grid.save(grid_path, quality=92)
        print(f"Grid  : {grid_path}")

    return result_pil

# ==============================
# BATCH INFERENCE & EVALUATION
# ==============================

def run_batch(pairs_file, vae, unet, scheduler):
    """Run inference on all pairs and compute evaluation metrics."""
    with open(os.path.join(DATASET_DIR, pairs_file)) as f:
        pairs = [l.strip().split() for l in f if l.strip()]

    print(f"Running batch inference on {len(pairs)} pairs …\n")

    # Initialize metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    # LPIPS uses pre-trained VGG network to compute perceptual similarity
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(DEVICE)

    successful_pairs = 0

    for person_name, cloth_name in pairs:
        base_p = os.path.splitext(person_name)[0]
        base_c = os.path.splitext(cloth_name)[0]

        person_path = os.path.join(DATASET_DIR, "images", person_name)
        cloth_path = os.path.join(DATASET_DIR, "garments", cloth_name)
        mask_path = os.path.join(DATASET_DIR, "agnostic_mask", f"{base_p}_inpaint_mask.png")
        composite_mask_path = os.path.join(DATASET_DIR, "agnostic_mask", f"{base_p}_mask.png")
        agnostic_path = os.path.join(DATASET_DIR, "agnostic", person_name)
        output_path = os.path.join(RESULTS_DIR, f"{base_p}_{base_c}.png")

        # fallbacks
        if not os.path.exists(mask_path):
            mask_path = os.path.join(DATASET_DIR, "agnostic_mask", f"{base_p}.jpg")
        if not os.path.exists(agnostic_path):
            agnostic_path = os.path.join(DATASET_DIR, "agnostic", f"{base_p}.png")

        try:
            # Generate the image
            result_pil = run_inference(
                person_path, cloth_path, mask_path, agnostic_path,
                output_path, vae, unet, scheduler,
                save_grid=False,
            )
            print(f"Completed {person_name} + {cloth_name}")

            # --- EVALUATION BLOCK ---
            # Load Ground Truth and resize to match output
            gt_pil = Image.open(person_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)

            # Convert both to tensors in [0, 1] range
            gt_tensor = TF.to_tensor(gt_pil).unsqueeze(0).to(DEVICE)
            pred_tensor = TF.to_tensor(result_pil).unsqueeze(0).to(DEVICE)

            # Update metric states
            psnr_metric.update(pred_tensor, gt_tensor)
            ssim_metric.update(pred_tensor, gt_tensor)
            lpips_metric.update(pred_tensor, gt_tensor)

            successful_pairs += 1

        except Exception as e:
            print(f"  ERROR {person_name} + {cloth_name}: {e}")

    # Calculate final averages
    if successful_pairs > 0:
        final_psnr = psnr_metric.compute().item()
        final_ssim = ssim_metric.compute().item()
        final_lpips = lpips_metric.compute().item()

        print("\n" + "=" * 40)
        print(f"EVALUATION METRICS (Averaged over {successful_pairs} pairs):")
        print(f"  SSIM  (Higher is better): {final_ssim:.4f}")
        print(f"  PSNR  (Higher is better): {final_psnr:.4f} dB")
        print(f"  LPIPS (Lower is better):  {final_lpips:.4f}")
        print("=" * 40 + "\n")
    else:
        print("\nNo pairs were successfully processed for evaluation.")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    vae, unet, scheduler = load_models(epoch=EPOCH)

    if BATCH_MODE:
        run_batch(TEST_PAIRS, vae, unet, scheduler)

    else:

        # ── Single inference ───────────────────────
        run_inference(
            person_path="../own/images/test1.jpg",
            cloth_path="../own/garments/1.jpg",
            mask_path="../own/agnostic_mask/test1_inpaint_mask.png",
            agnostic_path="../own/agnostic/test1_agnostic.jpg",
            identity_mask_path="../own/identity_masks/test1.png",
            output_path="catvton_finetuned_v4/results/result_debug_1.png",
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            sd_pipe = True
        )