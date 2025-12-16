import torch
from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel

model_id = "facebook/nougat-base"
device = "mps" if torch.backends.mps.is_available() else "cpu"

processor = NougatProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)

image = Image.open("page.png").convert("RGB")
inputs = processor(image, return_tensors="pt").to(device)

out = model.generate(**inputs, max_new_tokens=4096)
text = processor.batch_decode(out, skip_special_tokens=True)[0]
print(text)
