import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
from tqdm import tqdm

source = r"../raw images/images-resized/"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

image_files = [f for f in os.listdir(source) if f.lower().endswith((".jpg","jpeg",".png"))]

for img in tqdm(image_files):
    raw_image = os.path.join(source,img)
    raw_image = Image.open(raw_image).convert('RGB')
    inputs = processor(raw_image, return_tensors ="pt").to("cuda")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    if img .endswith((".jpg")):
        captioned_img = img.replace(".jpg" , ".txt")
        output_img = os.path.join(source, captioned_img)
        with open(output_img, "w") as file:
            file.write(caption)
    else:
        captioned_img = img.replace(".png", ".txt")
        output_img = os.path.join(source, captioned_img)
        with open(output_img, "w") as file:
            file.write(caption)