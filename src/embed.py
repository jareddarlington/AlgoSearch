from FlagEmbedding import BGEM3FlagModel
import torch

# Use MPS (Metal Performance Shaders) for GPU acceleration on macOS
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.model.to(device)
print(f"Model device: {next(model.model.parameters()).device}")

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
]

embeddings_1 = model.encode(
    sentences_1,
    batch_size=12,
    max_length=8192,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
)['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# [[0.6265, 0.3477], [0.3499, 0.678 ]]
