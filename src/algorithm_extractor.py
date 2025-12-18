# TODO: maybe switch to langchain idk, outlines is kind of a pain - also maybe use gemini api or something, might be easier

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import outlines
from pydantic import BaseModel
from enum import Enum
import time
from outlines.samplers import greedy
from transformers.utils import logging as hf_logging
from typing import List

hf_logging.set_verbosity_error()

t0 = time.perf_counter()

MODEL = "Qwen/Qwen2-1.5B-Instruct"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16 if device == "mps" else torch.float32,
    low_cpu_mem_usage=True,
)
hf_model.to(device)
hf_model.eval()

model = outlines.models.Transformers(
    hf_model,
    tokenizer,
)

t1 = time.perf_counter()
print(f"Model load time: {t1 - t0:.2f}s")


class AlgoInfo(BaseModel):
    name: str
    problem: str
    time_complexity: str
    space_complexity: str


class AlgoInfoList(BaseModel):
    algorithms: List[AlgoInfo]


t0 = time.perf_counter()
with torch.no_grad():
    generator = outlines.generate.json(
        model,
        AlgoInfoList,
        sampler=greedy(),
    )
    algos = generator("List DFS, BFS, and Dijkstra. For each: name, what it solves, time complexity, space complexity.")
t1 = time.perf_counter()
print(f"Generation time: {t1 - t0:.2f}s\n")

for algo in algos.algorithms:
    print(algo.name)
    print(algo.problem)
    print(algo.time_complexity)
    print(algo.space_complexity)
