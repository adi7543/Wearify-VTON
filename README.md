## Preprocessing Pipeline

This project uses the following pretrained components:

- **OpenPose** – human pose and skeleton extraction
- **SegFormer** – human parsing / semantic segmentation
- **BLIP** – automatic text prompt generation for diffusion conditioning

Due to size and licensing constraints, pretrained models and binaries
are not included in this repository.

### Setup Instructions

#### 1. OpenPose
Download and install OpenPose from:
https://github.com/CMU-Perceptual-Computing-Lab/openpose

Expected output:
openpose-json/
openpose-img/

The RGB skeleton was extracted using a python script.
Expected output:
openpose-skeleton

#### 2. SegFormer
Install SegFormer via HuggingFace:
https://huggingface.co/nvidia/segformer

Used for:
- Human parsing

#### 3. BLIP
BLIP is used to generate text prompts from garment images:
https://huggingface.co/Salesforce/blip-image-captioning-base

Generated prompts are saved as:
prompts.json


## Environment
Tested with:
- Python 3.10
- PyTorch 2.x
- CUDA 11.8