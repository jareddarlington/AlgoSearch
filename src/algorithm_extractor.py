from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device.type == "mps" else torch.float32  # safer fallback on CPU

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,
)
model.to(device)

inputs = tokenizer("The secret to baking a good cake is ", return_tensors="pt").to(device)

with torch.no_grad():
    out_ids = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=False,
    )

print(tokenizer.decode(out_ids[0], skip_special_tokens=True))
