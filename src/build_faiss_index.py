from pathlib import Path
import faiss
import numpy as np
import argparse
from embed import embed_db


DEFAULT_INDEX_PATH = "data/index.faiss"
DEFAULT_ALGO_PATH = "data/algorithms.db"


def build_index(index_path, algo_path):
    embeddings, ids = embed_db(algo_path)  # create vector embeddings from algorithm data

    # Ensure proper numpy types for FAISS
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int64)

    # Build index (with ids)
    dim = len(embeddings[0])  # dimension of embeddings
    index = faiss.index_factory(dim, "IDMap,L2norm,Flat")
    # index = faiss.index_factory(dim, "IDMap,PCA512,L2norm,Flat")
    # index.train(embeddings)  # train PCA
    index.add_with_ids(embeddings, ids)  # add vectors to index (with ids)

    faiss.write_index(index, index_path)


def main():
    # Handle script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX_PATH, help="Path to FAISS index")
    parser.add_argument("--algo_path", type=str, default=DEFAULT_ALGO_PATH, help="Path to algorithms database")
    args = parser.parse_args()

    algo_path = args.algo_path
    if not Path(algo_path).is_file():
        parser.error(f"--algo_path not found: {args.algo_path}")

    index_path = args.index_path
    if Path(index_path).is_file():  # index already exists, ask to rebuild it
        valid_response = False
        while not valid_response:
            response = input(f"FAISS index already exists at \"{index_path}\". Would you like to rebuild it (y/N)? ")

            if response.lower() == "y":
                valid_response = True

                print("Rebuilding FAISS index...")
                build_index(index_path, algo_path)  # rebuild index
                print("Complete: FAISS index rebuilt")
            elif response.lower() == "n" or not response:
                valid_response = True

                print("Skipping: not rebuilding FAISS index")

    else:  # ask to build index if it doesn't exist
        valid_response = False
        while not valid_response:
            response = input(f"No FAISS index found at \"{index_path}\". Would you like to build the index (Y/n)? ")
            if response.lower() == "y" or not response:
                valid_response = True

                print("Building FAISS index...")
                build_index(index_path, algo_path)
                print("Complete: FAISS index built")
            elif response.lower() == "n":
                valid_response = True

                print("Skipping: not building FAISS index")

    print("Exiting")


if __name__ == "__main__":
    main()
