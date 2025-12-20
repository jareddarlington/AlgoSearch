# TODO: maybe switch to langchain idk, outlines is kind of a pain - maybe use colab gpu for actual extraction

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import outlines
from pydantic import BaseModel
from typing import List
import time
from outlines.samplers import greedy
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


MODEL = "Qwen/Qwen2-1.5B-Instruct"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
is_mps = device.type == "mps"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

t0 = time.perf_counter()
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16 if is_mps else torch.float32,
    low_cpu_mem_usage=True,
)
hf_model.to(device)
hf_model.eval()

t1 = time.perf_counter()
print(f"Model load time: {t1 - t0:.2f}s")

hf_model.generation_config.use_cache = True

model = outlines.models.Transformers(hf_model, tokenizer)


class AlgoInfo(BaseModel):
    name: str
    problem: str
    time_complexity: str
    space_complexity: str


class AlgoInfoList(BaseModel):
    algorithms: List[AlgoInfo]


generator = outlines.generate.json(
    model,
    AlgoInfoList,
    sampler=greedy(),
)

prompt = "List DFS, BFS, and Dijkstra. For each: name, what it solves, time complexity, space complexity."

t0 = time.perf_counter()
with torch.inference_mode():
    algos = generator(prompt, max_tokens=256)
t1 = time.perf_counter()
print(f"Generation time: {t1 - t0:.2f}s\n")

for algo in algos.algorithms:
    print(algo.name)
    print(algo.problem)
    print(algo.time_complexity)
    print(algo.space_complexity)
