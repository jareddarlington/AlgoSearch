from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
from db_utils import get_db_connection

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# Disable tqdm progress bars
from functools import partialmethod
import tqdm

tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)

QUERY_BATCH_SIZE = 1
QUERY_MAX_TOKENS = 32

CORPUS_BATCH_SIZE = 64
CORPUS_MAX_TOKENS = 256

EMBEDDER_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCTION = "Instruct: Given a search query, retrieve relevant algorithm descriptions\nQuery: "

ALGO_TEMPLATE = """\
Name: {name}
Categories: {categories}
Description: {description}
"""

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SentenceTransformer(EMBEDDER_MODEL_NAME, device=str(device), model_kwargs={"torch_dtype": torch.float16})
model.max_seq_length = CORPUS_MAX_TOKENS

# Warmup: trigger Metal shader compilation so the first real query isn't artificially slow
with torch.inference_mode():
    model.encode("warmup", prompt="", convert_to_numpy=True)


def get_algo_data(algo_path: str):
    with get_db_connection(algo_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT algo_id, name, description, categories FROM algorithms")
        rows = cursor.fetchall()

        algo_ids = [row[0] for row in rows]
        embed_texts = [
            ALGO_TEMPLATE.format(
                name=row[1],
                categories="; ".join(json.loads(row[3])),
                description=row[2],
            )
            for row in rows
        ]

        return embed_texts, algo_ids


def embed(text: list[str] | str, is_query: bool) -> np.ndarray:
    with torch.inference_mode():
        if is_query:
            model.max_seq_length = QUERY_MAX_TOKENS
            embeddings = model.encode(
                text,
                prompt=QUERY_INSTRUCTION,
                batch_size=QUERY_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        else:
            model.max_seq_length = CORPUS_MAX_TOKENS
            embeddings = model.encode(
                text,
                prompt="",
                batch_size=CORPUS_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

    return embeddings


def embed_db(algo_path: str):
    texts, ids = get_algo_data(algo_path)

    embeddings = embed(texts, is_query=False)

    embeddings = np.array(embeddings, dtype=np.float32)
    ids = np.array(ids, dtype=np.int64)

    return embeddings, ids
