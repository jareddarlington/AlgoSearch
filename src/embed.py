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

QUERY_BATCH_SIZE = 1
QUERY_MAX_TOKENS = 32

CORPUS_BATCH_SIZE = 64
CORPUS_MAX_TOKENS = 256

EMBEDDER_MODEL_NAME = "BAAI/bge-m3"

ALGO_TEMPLATE = """\
Name: {name}
Categories: {categories}
Description: {description}
"""

# Create model and move to device
model = BGEM3FlagModel(EMBEDDER_MODEL_NAME, use_fp16=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.model = model.model.to(device, dtype=torch.float16)


def get_algo_data(algo_path: str):
    with get_db_connection(algo_path) as conn:
        # Gather relevant info from database
        cursor = conn.cursor()
        cursor.execute("SELECT algo_id, name, description, categories FROM algorithms")
        rows = cursor.fetchall()

        # Extract algorithm IDs and format texts for embedding
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


@torch.inference_mode()
def embed(text: list[str] | str, is_query: bool):
    if is_query:  # embed query
        embeddings = model.encode_queries(
            text,
            batch_size=None,
            max_length=QUERY_MAX_TOKENS,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )['dense_vecs']
    else:  # embed corpus
        embeddings = model.encode_corpus(
            text,
            batch_size=CORPUS_BATCH_SIZE,
            max_length=CORPUS_MAX_TOKENS,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )['dense_vecs']

    return embeddings


def embed_db(algo_path: str):
    # Grab algorithm data
    texts, ids = get_algo_data(algo_path)

    embeddings = embed(texts, is_query=False)

    # TODO: confirm that I actually need this
    # Convert to numpy array with correct dtype for FAISS
    embeddings = np.array(embeddings, dtype=np.float32)
    ids = np.array(ids, dtype=np.int64)

    return embeddings, ids
