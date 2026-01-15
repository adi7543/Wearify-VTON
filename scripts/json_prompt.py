import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

source = r"our dataset/images-resized/"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

images = os.listdir(source)

for img in images:
    if img .endswith((".JPG")):
        raw_image = os.path.join(source,img)
        raw_image = Image.open(raw_image).convert('RGB')
        inputs = processor(raw_image, return_tensors ="pt").to("cuda")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        captioned_img = img.replace(".JPG" , ".txt")
        output_img = os.path.join(source, captioned_img)
        with open(output_img, "w") as file:
            file.write(caption)
            # print(caption)