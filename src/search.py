import faiss
from embed import embed
import numpy as np
from gemini import create_model, generate_content
from schemas import Queries
import argparse
import json

DEFAULT_INDEX_PATH = "data/index.faiss"
DEFAULT_EMBEDDER_NAME = "BAAI/bge-m3"
DEFAULT_LLM_NAME = "gemini-2.5-flash"
DEFAULT_MAX_TOKENS = 1024
PROMPT_PATH = "prompts/query-rewrite.txt"
PROMPT_TEMPLATE = open(PROMPT_PATH, encoding="utf-8").read()


def main():
    index = faiss.read_index(DEFAULT_INDEX_PATH)

    user_query = input("Search for an algorithm: ")

    schema = Queries.model_json_schema()
    client, config = create_model(
        model_name=DEFAULT_LLM_NAME,
        temperature=0.0,
        max_output_tokens=DEFAULT_MAX_TOKENS,
        response_schema=schema,
    )

    prompt = PROMPT_TEMPLATE.replace("{query}", user_query)
    output = generate_content(client, DEFAULT_LLM_NAME, config, prompt)
    new_queries = Queries.model_validate_json(output.strip())
    new_queries.queries.insert(0, user_query)

    print()

    for query in new_queries.queries:
        print(query)
        embedding = embed(query, DEFAULT_EMBEDDER_NAME, is_query=True).reshape(1, -1)
        D, I = index.search(embedding, 5)
        print(D)
        print(I)
        print()


if __name__ == "__main__":
    main()
