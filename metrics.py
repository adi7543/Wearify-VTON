import torch
import time
import numpy as np
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import os

# --- CONFIG ---
model_name = "mattmdjaga/segformer_b2_clothes"
input_folder = r"data\train\image-resized"  # Your images
num_test_images = 50  # Run on first 50 images for the test

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device)

images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))][:num_test_images]
confidences = []

print(f"--- Benchmarking SegFormer on {len(images)} images ---")

# Start Timer
start_time = time.time()

for img_name in images:
    path = os.path.join(input_folder, img_name)
    image = Image.open(path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

        # Calculate Confidence
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)  # Highest probability per pixel
        mean_conf = torch.mean(max_probs).item()
        confidences.append(mean_conf)

end_time = time.time()
total_time = end_time - start_time
fps = len(images) / total_time

print(f"Model: {model_name}")
print(f"Average Inference Speed: {fps:.2f} FPS")
print(f"Mean Model Confidence: {np.mean(confidences) * 100:.2f}%")
print(f"Min Confidence: {np.min(confidences) * 100:.2f}%")