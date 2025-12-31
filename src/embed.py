from FlagEmbedding import BGEM3FlagModel
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

MODEL_BATCH_SIZE = 64
MAX_DB_TOKENS = 256
MAX_QUERY_TOKENS = 128

EMBED_TEMPLATE = """\
Name: {name}
Categories: {categories}
Description: {description}
"""


def get_algo_data(algo_path: str):
    with get_db_connection(algo_path) as conn:
        # Gather relevant info from database
        cursor = conn.cursor()
        cursor.execute("SELECT algo_id, name, description, categories FROM algorithms")
        rows = cursor.fetchall()

        # Extract columns from rows (transpose)
        algo_ids, names, descriptions, categories = zip(*rows)

        # Format algorithm information for embedding
        embed_texts = [
            EMBED_TEMPLATE.format(
                name=name,
                categories="; ".join(json.loads(category)) if category else "",
                description=description,
            )
            for name, description, category in zip(names, descriptions, categories)
        ]

        return embed_texts, algo_ids


def embed(text: list[str] | str, model_name: str, is_query: bool):
    # Create model
    model = BGEM3FlagModel(model_name, use_fp16=True)

    # Move model to device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.model.to(device)

    if is_query:  # queries typically aren't as long
        max_tokens = MAX_QUERY_TOKENS
    else:
        max_tokens = MAX_DB_TOKENS

    result = model.encode(
        text,
        batch_size=MODEL_BATCH_SIZE,
        max_length=max_tokens,
        return_dense=True,
        return_colbert_vecs=False,
        return_sparse=False,
    )

    embeddings = result['dense_vecs']

    # Convert to numpy array with correct dtype for FAISS
    embeddings = np.array(embeddings, dtype=np.float32)

    return embeddings


def embed_db(algo_path: str, model_name: str):
    # Grab algorithm data
    texts, ids = get_algo_data(algo_path)

    embeddings = embed(texts, model_name)

    # Convert to numpy array with correct dtype for FAISS
    ids = np.array(ids, dtype=np.int64)

    return embeddings, ids
