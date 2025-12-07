# AlgoSearch: Semantic Search for Algorithms on arXiv

## 1. System Design

### 1.1. Building the Vector Database

1. Scrape arXiv for CS papers
    - Only in the "Data Structures and Algorithms" category for now
    - Not sure the best option for this yet, but [arXiv's API](https://info.arxiv.org/help/api/) provides abstracts, metadata, and links to html/pdfs - html/pdfs are harder to parse for algorithms than LaTeX files in this case though
2. Store each algorithm with its paper's abstract, metadata, and possibly its surrounding context
3. Pass algorithm data into summarization model
    - Probably use deepseek-coder-1.3b-instruct to start, maybe switch to a larger or newer model later (DeepSeek-Coder-V2-Lite-Instruct)
    - Consider doing multiple pases of summarization
4. Pass summarizations into embedder
    - Considering specter2_base or bge-large-en-v1.5
5. Store embeddings in FAISS
    - Also need mapping between FAISS embeddings and algorithm data
    - Replacing FAISS with my own search implementation could also be fun (HNSW or IVF+PQ) - might be a whole other project though

### 1.2. Search

1. Take in natural language request
2.

## x. Additions

-   Create custom algorithm retrieval benchmark
