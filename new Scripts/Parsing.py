# import sys
# import os
# import numpy as np
# from PIL import Image
# import torch
#
#
#
# # ==========================================
# # 0. THE FORCE-LOAD FIX
# # ==========================================
# import nvidia.cudnn as cudnn
# import nvidia.cuda_runtime as cuda_runtime
#
# cudnn_lib = cudnn.__path__[0]
# cuda_bin = os.path.join(cuda_runtime.__path__[0], "bin")
#
# if hasattr(os, 'add_dll_directory'):
#     if os.path.exists(cudnn_lib):
#         os.add_dll_directory(cudnn_lib)
#     if os.path.exists(cuda_bin):
#         os.add_dll_directory(cuda_bin)
#
# os.environ["PATH"] = cudnn_lib + os.pathsep + cuda_bin + os.pathsep + os.environ.get("PATH", "")
# print(f"cuDNN path: {cudnn_lib}")
# print(f"CUDA bin path: {cuda_bin}")

import os, sys

# Add ALL CUDA 11.8 DLL locations
CUDA_118_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
CUDNN_BIN = r"C:\Users\PC\venv_ort116\lib\site-packages\nvidia\cudnn\bin"
CUDA_RT_BIN = r"C:\Users\PC\venv_ort116\lib\site-packages\nvidia\cuda_runtime\bin"

for dll_path in [CUDA_118_BIN, CUDNN_BIN, CUDA_RT_BIN]:
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)
        os.environ["PATH"] = dll_path + os.pathsep + os.environ.get("PATH", "")
        print(f"Added: {dll_path}")
    else:
        print(f"WARNING: Not found: {dll_path}")

# Remove any CUDA 12 path interference
os.environ["CUDA_PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"






import os
import sys

# ==========================================
# 0. MUST BE FIRST - before torch/onnx imports
# ==========================================
import nvidia.cudnn as cudnn
import nvidia.cuda_runtime as cuda_runtime

cudnn_lib = cudnn.__path__[0]
cuda_bin = os.path.join(cuda_runtime.__path__[0], "bin")

if hasattr(os, 'add_dll_directory'):
    if os.path.exists(cudnn_lib):
        os.add_dll_directory(cudnn_lib)
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)

os.environ["PATH"] = cudnn_lib + os.pathsep + cuda_bin + os.pathsep + os.environ.get("PATH", "")
os.environ["CUDA_PATH"] = os.path.dirname(cuda_bin)  # override wrong CUDA_PATH

# ==========================================
# NOW safe to import everything else
# ==========================================
import numpy as np
from PIL import Image
import torch










# # ==========================================
# # 0. THE FORCE-LOAD FIX
# # ==========================================
# import nvidia.cudnn as cudnn
# import nvidia.cuda_runtime as cuda_runtime
#
# # This finds the actual folder containing the .dll files
# cudnn_lib = os.path.join(os.path.dirname(cudnn.__file__), "lib")
# cuda_bin = os.path.join(os.path.dirname(cuda_runtime.__file__), "bin")
#
# # Add to the DLL search path
# if hasattr(os, 'add_dll_directory'):
#     os.add_dll_directory(cudnn_lib)
#     os.add_dll_directory(cuda_bin)
#
# # Add to the system PATH for this specific terminal window
# os.environ["PATH"] = cudnn_lib + os.pathsep + cuda_bin + os.pathsep + os.environ.get("PATH", "")
#
# # Check if the file actually exists (Sanity Check)
# dll_check = os.path.join(cuda_bin, "cudart64_12.dll")
# if os.path.exists(dll_check):
#     print(f"Found CUDA DLL at: {dll_check}")
# else:
#     print("WARNING: Could not find cudart64_12.dll in the expected folder!")


# # ==========================================
# # 0. THE DEFINITIVE CUDA DLL LINK
# # ==========================================
# def link_cuda_libs():
#     try:
#         import nvidia.cudnn as cudnn
#         import nvidia.cuda_runtime as cuda_runtime
#
#         # Paths to the newly installed nvidia libraries
#         cudnn_lib_path = os.path.join(os.path.dirname(cudnn.__file__), 'lib')
#         cuda_bin_path = os.path.join(os.path.dirname(cuda_runtime.__file__), 'bin')
#
#         # Add to Windows DLL Search Path
#         if hasattr(os, 'add_dll_directory'):
#             if os.path.exists(cudnn_lib_path):
#                 os.add_dll_directory(cudnn_lib_path)
#             if os.path.exists(cuda_bin_path):
#                 os.add_dll_directory(cuda_bin_path)
#
#         # Also update System PATH environment variable for this session
#         os.environ["PATH"] = cudnn_lib_path + os.pathsep + cuda_bin_path + os.pathsep + os.environ.get("PATH", "")
#         print("Successfully linked NVIDIA libraries.")
#     except Exception as e:
#         print(f"Warning: Could not link NVIDIA libraries: {e}")
#
#
# link_cuda_libs()


# ==========================================
# 1. PATH SETUP (Making it unbreakable)
# ==========================================
# Get the exact folder where this script lives ("new Scripts")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to Wearify, then down into OOTDiffusion
PARSER_PATH = os.path.join(SCRIPT_DIR, "..", "OOTDiffusion", "preprocess", "humanparsing")
sys.path.append(os.path.abspath(PARSER_PATH))

# Define Input/Output folders relative to this script
INPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_new", "images"))
OUTPUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_new", "parsed_labels"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. INITIALIZE PARSER
# ==========================================
try:
    from run_parsing import Parsing  # type: ignore
# except ModuleNotFoundError:
#     print("\n[ERROR] Could not find run_parsing.py.")
#     print(f"Looked in: {os.path.abspath(PARSER_PATH)}")
#     sys.exit(1)

except Exception as e:
    print(f"\n[ERROR] Import failed: {e}")
    print(f"Looked in: {os.path.abspath(PARSER_PATH)}")
    sys.exit(1)

print("Loading ONNX Models into VRAM...")
try:
    parser = Parsing(gpu_id=0)
except Exception as e:
    print(
        f"\n[ERROR] Failed to load ONNX models. Make sure parsing_atr.onnx and parsing_lip.onnx are in checkpoints/humanparsing/")
    print(f"Details: {e}")
    sys.exit(1)

# ==========================================
# 3. RUN INFERENCE
# ==========================================
valid_exts = ('.png', '.jpg', '.jpeg')
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_exts)]

if not image_files:
    print(f"\n[WARNING] No images found in {INPUT_DIR}")
    sys.exit(0)

print(f"\nFound {len(image_files)} images. Starting parsing...")

for img_name in image_files:
    img_path = os.path.join(INPUT_DIR, img_name)

    # Force output to be .png so it saves as lossless data, not compressed jpeg
    base_name = os.path.splitext(img_name)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")

    print(f"Processing: {img_name}...")

    try:
        # The parser returns a tuple: (body_map, face_mask). We only want the body_map.
        parsed_data, _ = parser(img_path)

        # Convert PyTorch Tensor -> NumPy Array -> PIL Image
        if hasattr(parsed_data, 'cpu'):
            mask_np = parsed_data.squeeze().cpu().numpy()
        else:
            mask_np = np.array(parsed_data).squeeze()

        # Ensure it is a 2D array of integers (Class IDs)
        mask_np = mask_np.astype(np.uint8)

        # Save the image
        Image.fromarray(mask_np).save(output_path)
        print(f"  -> Saved successfully!")

    except Exception as e:
        print(f"  -> [ERROR] Failed to process {img_name}: {e}")

print("\nAll done! Check your parsed_labels folder.")