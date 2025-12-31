from typing import List, Tuple
import numpy as np
import torch
from sentence_transformers import CrossEncoder

BATCH_SIZE = 32

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L2-v2', device=device, max_length=256)


def rerank_candidates(query: str, candidates: List[str]) -> List[Tuple[str, float]]:
    pairs = [(query, c) for c in candidates]

    with torch.inference_mode():
        with torch.autocast(device_type="mps", dtype=torch.float16):
            scores = model.predict(pairs, batch_size=BATCH_SIZE, convert_to_numpy=True)

    order = np.argsort(-scores)
    # return [(candidates[i], float(scores[i])) for i in order]
    return [candidates[i] for i in order]
