from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
import torch
import numpy as np
import json
from db_utils import get_db_connection

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

MODEL_BATCH_SIZE = 32
MODEL_MAX_TOKENS = 256

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


def embed(algo_path: str, model_name: str):
    # Grab algorithm data
    texts, ids = get_algo_data(algo_path)

    # Create model
    # model = BGEM3FlagModel(model_name, use_fp16=True)

    # # Move model to device
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # model.model.to(device)

    model = BGEM3FlagModel(model_name, use_fp16=False, devices='cpu')

    result = model.encode(
        texts,
        batch_size=MODEL_BATCH_SIZE,
        max_length=MODEL_MAX_TOKENS,
        return_dense=True,
        return_colbert_vecs=False,
        return_sparse=False,
    )

    embeddings = result['dense_vecs']

    # Convert to numpy array with correct dtype for FAISS
    embeddings = np.array(embeddings, dtype=np.float32)
    ids = np.array(ids, dtype=np.int64)

    return embeddings, ids
