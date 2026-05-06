import os
import sys

# ==========================================
# 0. CUDA FIX (MUST BE FIRST)
# ==========================================
import nvidia.cudnn as cudnn
import nvidia.cuda_runtime as cuda_runtime

cudnn_lib = cudnn.__path__[0]
cuda_bin = os.path.join(cuda_runtime.__path__[0], "bin")
if hasattr(os, 'add_dll_directory'):
    if os.path.exists(cudnn_lib): os.add_dll_directory(cudnn_lib)
    if os.path.exists(cuda_bin): os.add_dll_directory(cuda_bin)
os.environ["PATH"] = cudnn_lib + os.pathsep + cuda_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["CUDA_PATH"] = os.path.dirname(cuda_bin)

# ==========================================
# 1. SERVER LOGIC
# ==========================================
import io
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import Preprocessing
import inference

models_cache = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up GPU and loading ALL models into VRAM... This will take a minute.")
    models_cache.update(Preprocessing.load_preprocessing_models())
    models_cache.update(inference.load_inference_models())
    print("API is Live and Ready on port 8000!")
    yield
    models_cache.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/vton")
async def process_vton(
        person_image: UploadFile = File(...),
        cloth_image: UploadFile = File(...)
):
    # 1. Read Inputs
    person_bytes = await person_image.read()
    cloth_bytes = await cloth_image.read()

    person_pil = Image.open(io.BytesIO(person_bytes)).convert("RGB")
    raw_cloth_pil = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")
    person_bgr = cv2.cvtColor(np.array(person_pil), cv2.COLOR_RGB2BGR)

    # ---------------------------------------------------------
    # PART A: PERSON PREPROCESSING
    # ---------------------------------------------------------
    person_cloth_mask = Preprocessing.get_cloth_mask(person_pil, models_cache["seg_proc"], models_cache["seg_model"])
    parse_map = Preprocessing.get_parse_map(person_pil, models_cache["parser"])
    identity_mask = Preprocessing.get_identity_mask(person_bgr, models_cache["dino"], models_cache["sam"])

    agnostic_pil, tight_mask_pil = Preprocessing.build_agnostic(person_bgr, person_cloth_mask, parse_map, identity_mask)

    # ---------------------------------------------------------
    # PART B: GARMENT PREPROCESSING
    # ---------------------------------------------------------
    garment_mask = Preprocessing.get_cloth_mask(raw_cloth_pil, models_cache["seg_proc"], models_cache["seg_model"])
    clean_cloth_pil = Preprocessing.extract_pure_garment(raw_cloth_pil, garment_mask)

    import torch
    models_cache["sam"].model.to("cpu")
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PART C: INFERENCE
    # ---------------------------------------------------------
    final_result_pil = inference.run_vton(
        person_pil=person_pil,
        cloth_pil=clean_cloth_pil,
        mask_pil=tight_mask_pil,
        agnostic_pil=agnostic_pil,
        identity_mask_np=identity_mask,
        models=models_cache
    )

    models_cache["sam"].model.to("CUDA")

    # Return Final Image bytes to Frontend
    output_buffer = io.BytesIO()
    final_result_pil.save(output_buffer, format="JPEG", quality=95)
    return Response(content=output_buffer.getvalue(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)