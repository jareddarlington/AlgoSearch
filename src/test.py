from llama_cpp import Llama
import time

with open("prompts/extraction.txt", "r") as f:
    prompt = f.read()

with open("tex.txt", "r") as f:
    latex_content = f.read()

prompt = prompt.replace("{latex}", latex_content)

t0 = time.perf_counter()
llm = Llama(
    model_path="models/qwen2.5-1.5b-instruct-q5_k_m.gguf",
    n_ctx=4096,
    verbose=False,
    chat_format="qwen",
)
t1 = time.perf_counter()
print(f"Model load time: {t1 - t0:.2f}s")

messages = [
    {"role": "system", "content": "You are a precise information extraction engine."},
    {"role": "user", "content": prompt},
]

t0 = time.perf_counter()
out = llm.create_chat_completion(
    messages=messages,
    max_tokens=1024,
    temperature=0.0,
    top_p=1.0,
    repeat_penalty=1.1,
    stop=["</s>", "<|im_end|>", "<|endoftext|>"],
)
t1 = time.perf_counter()
print(f"Generation time: {t1 - t0:.2f}s\n")

print(out["choices"][0]["message"]["content"])
