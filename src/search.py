import faiss
from embed import embed
import numpy as np
from gemini import create_model, generate_content
from schemas import Queries
from db_utils import get_db_connection
import argparse
import json
from typing import List, Dict
from rerank import rerank_candidates
import time

DEFAULT_INDEX_PATH = "data/index.faiss"
DEFAULT_ALGORITHMS_DB_PATH = "data/algorithms.db"
DEFAULT_LLM_NAME = "gemini-2.5-flash"
DEFAULT_MAX_TOKENS = 1024
PROMPT_PATH = "prompts/query-rewrite.txt"
PROMPT_TEMPLATE = open(PROMPT_PATH, encoding="utf-8").read()
SEARCH_K = 20
RERANK_K = 5

ALGO_TEMPLATE = """\
Name: {name}
Categories: {categories}
Description: {description}
"""


def get_candidates_from_db(ids: List[int]) -> List[str]:
    with get_db_connection(DEFAULT_ALGORITHMS_DB_PATH) as conn:
        cursor = conn.cursor()

        # Create placeholders for SQL IN clause
        placeholders = ','.join('?' * len(ids))
        query = f"""
            SELECT algo_id, name, description, categories
            FROM algorithms
            WHERE algo_id IN ({placeholders})
        """

        cursor.execute(query, ids)
        rows = cursor.fetchall()

        # Convert to list of formatted strings
        results = []
        for row in rows:
            formatted = ALGO_TEMPLATE.format(name=row[1], categories=row[3], description=row[2])
            results.append(formatted)

        return results


def main():
    index = faiss.read_index(DEFAULT_INDEX_PATH)

    query = input("Search for an algorithm: ")

    start_time = time.time()
    embedding = embed(query, is_query=True)
    embedding_time = time.time() - start_time
    print(f"Embedding time: {embedding_time:.3f}s")

    start_time = time.time()
    _, ids = index.search(embedding.reshape(1, -1), SEARCH_K)
    search_time = time.time() - start_time
    print(f"Search time: {search_time:.3f}s")

    # Reranking
    start_time = time.time()
    candidates = get_candidates_from_db(ids[0].tolist())
    reranked_candidates = rerank_candidates(query, candidates)
    topk = reranked_candidates[0:RERANK_K]
    reranking_time = time.time() - start_time
    print(f"Reranking time: {reranking_time:.3f}s")

    # for item in topk:
    #     print(item)
    #     print()


if __name__ == "__main__":
    main()
