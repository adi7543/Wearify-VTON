# Wearify

**Wearify** is a virtual try-on (VTON) system for South Asian garments (kurta, sherwani) built on top of [CatVTON](https://github.com/Zheng-Chong/CatVTON), a diffusion-based garment try-on model. It wraps a full data preprocessing pipeline, custom model fine-tuning, and a geometry-aware inference engine — designed specifically for the proportions and styles of Eastern wear.

---

## Features

- **CatVTON fine-tuned on Eastern wear** — LoRA fine-tuning on a custom Pakistani kurta dataset
- **Geometry-aware inference** — automatically handles two cases:
  - **Case A** (target garment shorter) — SD inpainting fills the gap with trouser fabric
  - **Case B** (target garment longer) — grey-fill extension and natural-proportion cloth scaling
- **Identity preservation** — GroundingDINO + SAM composites the original hands, face, and neck back onto the result
- **Custom garment segmentation** — SegFormer fine-tuned specifically on kurta/sherwani to generate garment masks
- **Human parsing** — OOTDiffusion's ONNX humanparser for body region labeling
- **Full dataset pipeline** — image scaling, segmentation, garment extraction, and dataset splitting all included

---

## How It Works

```
Raw Person Images
       │
       ▼
Image Scaling (512×512 padded)    ← image_scaling.py
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
Human Parsing (OOTDiffusion ONNX)     Garment Segmentation (SegFormer)
Parsing.py                            Preprocessing_new.py
       │                                      │
       ▼                                      ▼
Parsing label maps               ref_cloth_masks + Cloth extraction
                                       extract_cloth.py
                                              │
                                              ▼
                                        garments/ (white-bg)

       ┌──────────────────────────────────────┘
       ▼
Identity Mask Extraction (GroundingDINO + SAM)   ← SAM_hands.py
[face, hands, neck → identity_masks/]

       ▼
Dataset Split (80/20 train/test)     ← dataset_split.py

       ▼
Fine-tuning (CatVTON + LoRA)         ← CATvton_train.py

       ▼
Inference (geometry-aware VTON)      ← CATvton_inference.py

       ▼
Final Try-On Result
```

---

## Preprocessing Pipeline

The following pretrained components are required. They are **not included** in this repo due to size and licensing constraints.

| Component | Purpose | Source |
|---|---|---|
| **OOTDiffusion Human Parser** | Body region parsing (ONNX) | [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) |
| **SegFormer (mit-b0)** | Base for custom garment segmentation | [HuggingFace](https://huggingface.co/nvidia/mit-b0) |
| **GroundingDINO** | Open-vocabulary object detection (face, hands, neck) | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| **SAM (vit_h)** | Pixel-perfect segmentation from DINO boxes | [Segment Anything](https://github.com/facebookresearch/segment-anything) |
| **CatVTON weights** | Pretrained attention weights for try-on | [zhengchong/CatVTON](https://huggingface.co/zhengchong/CatVTON) |

### Setup Instructions

#### 1. OOTDiffusion Human Parser

Clone OOTDiffusion and place the humanparsing ONNX models in:
```
OOTDiffusion/preprocess/humanparsing/checkpoints/humanparsing/
    parsing_atr.onnx
    parsing_lip.onnx
```
Run parsing on your person images:
```bash
python Parsing.py
```
Expected output:
```
own/parsed_labels/     ← per-image label maps (.png)
```

#### 2. SegFormer (Custom Kurta Segmentation)

Install the base model via HuggingFace:
```bash
pip install transformers
```
The base checkpoint (`nvidia/mit-b0`) is downloaded automatically. To fine-tune your own:
```bash
python my_segformer.py
```
This trains a binary segmenter (background vs. kurta) with 10× class weight on the garment.
Trained model saved to: `my_custom_segformer_v4/`

Run preprocessing to generate masks:
```bash
python Preprocessing_new.py
```
Expected outputs:
```
own/images/            ← resized person images
own/ref_cloth_masks/   ← binary garment masks
```

Then extract garments onto a white background:
```bash
python extract_cloth.py
```
Expected output:
```
own/garments/          ← white-background garment images
```

#### 3. GroundingDINO + SAM

Download weights and place them as follows:
```
GroundingDINO/
    groundingdino_swint_ogc.pth
    sam_vit_h_4b8939.pth
    groundingdino/config/GroundingDINO_SwinT_OGC.py
```

Run identity mask extraction:
```bash
python SAM_hands.py
```
Detects face, hands, and neck.

Expected output:
```
own/identity_masks/    ← per-image identity masks (.png)
```

---

## Dataset Structure

```
dataset_final/
├── images/              # Person images (e.g. 11.jpg)
├── agnostic/            # Agnostic images (clothing region greyed out)
├── agnostic_mask/       # Inpaint masks
│   ├── 11_inpaint_mask.png   ← used for training/inference
└── garments/            # White-background garment images
```

Generate train/test pair files:
```bash
python dataset_split.py
```
Produces `train_pairs.txt` and `test_pairs.txt` at 80/20 split.

---

## Training

```bash
python CATvton_train.py
```

| Setting | Value |
|---|---|
| Base model | `runwayml/stable-diffusion-inpainting` |
| VAE | `stabilityai/sd-vae-ft-mse` |
| Resolution | 512 × 512 |
| LoRA rank / alpha | 16 / 32 |
| LoRA targets | `attn1` self-attention (Q, K, V, Out) |
| Effective batch size | 4 (1 × 4 gradient accumulation) |
| Epochs | 30 |
| Optimizer | AdamW 8-bit (`bitsandbytes`) |
| Loss | MSE + L1 + garment-region L1 (weighted 1.0 / 0.5 / 1.0) |
| Mixed precision | fp16 (AMP) |

Checkpoints saved per epoch to `catvton_finetuned_v4/lora_epoch_N/`.
Visualizations generated every 3 epochs to `catvton_finetuned_v4/visualizations/`.

---

## Inference

```bash
python CATvton_inference.py
```

Configure the paths for single-image or batch mode.

**Single image:**
```python
run_inference(
    person_path="own/images/test1.jpg",
    cloth_path="own/garments/1.jpg",
    mask_path="own/agnostic_mask/test1_inpaint_mask.png",
    agnostic_path="own/agnostic/test1_agnostic.jpg",
    identity_mask_path="own/identity_masks/test1.png",
    output_path="results/result.png",
    ...
)
```

**Batch mode** (set `BATCH_MODE = True`): runs on all test pairs and outputs evaluation metrics.

### Geometry-Aware Inference

The pipeline detects garment height relative to the person and routes accordingly:

| Case | Condition | Behaviour |
|---|---|---|
| **Case A** | Target garment shorter than source | SD inpainting fills the trouser gap; cloth scaled to fill frame |
| **Case B** | Target garment longer than source | Grey-fill extends the agnostic region downward; natural-proportion cloth scaling |

### Evaluation Metrics (Batch Mode)

| Metric | Direction |
|---|---|
| SSIM | Higher is better |
| PSNR | Higher is better (dB) |
| LPIPS | Lower is better |

---

## Environment

Tested with:

| Dependency | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.x |
| CUDA | 12 |
| cuDNN | via `nvidia-cudnn` package |
| ONNX Runtime | 1.16.x (GPU) |

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Key packages: `diffusers`, `transformers`, `peft`, `bitsandbytes`, `segment-anything`, `groundingdino`, `torchmetrics`, `safetensors`, `opencv-python`, `Pillow`

---

## Project Structure

```
wearify/
├── CATvton_train.py         # Fine-tuning pipeline (LoRA on CatVTON)
├── CATvton_inference.py     # Geometry-aware VTON inference
├── my_segformer.py          # Custom SegFormer training (kurta segmentation)
├── Preprocessing_new.py     # Garment mask generation using SegFormer
├── Parsing.py               # Human parsing via OOTDiffusion ONNX
├── SAM_hands.py             # Identity mask extraction (GroundingDINO + SAM)
├── extract_cloth.py         # Garment extraction onto white background
├── image_scaling.py         # Resize & pad images to 512×512
├── data_augmentation.py     # Color jitter augmentation
├── dataset_split.py         # Train/test pair generation
├── filter_images.py         # Retrieve originals for bad-mask correction
└── README.md
```
