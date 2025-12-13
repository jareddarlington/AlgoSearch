# AlgoSearch: Semantic Search for Algorithms on arXiv

## 1. System Design

### 1.1. Building Vector Database

1. Scrape arXiv for CS papers
    - Only in the "Data Structures and Algorithms" category for now
    - Not sure the best option for this yet, but [arXiv's API](https://info.arxiv.org/help/api/) provides abstracts, metadata, and links to html/pdfs - html/pdfs are harder to parse for algorithms than LaTeX files in this case though - Claude Sonnet 4.5 seems to be really consistent at parsing algorithms via pdfs (no direct LaTeX), so likely use that or a smaller Claude model
2. Store each algorithm with its paper's abstract and metadata (also surrounding context for embedding but not long term)
3. Pass algorithm data into summarization model
    - Probably use deepseek-coder-1.3b-instruct to start, maybe switch to a larger or newer model later (DeepSeek-Coder-V2-Lite-Instruct)
    - Consider doing multiple pases of summarization
4. Pass summarizations into embedder
    - Considering specter2_base or bge-large-en-v1.5
    - IDEA: convert algorithms to actual code during summarization step (along with normal summarization), then use a code embedder (like Codestral Embed) + late fusion during search (two aligned embedding spaces in this case)
5. Store embeddings in FAISS
    - Also need mapping between FAISS embeddings and algorithm data
    - Look into Pinecone for a real database database

### 1.2. Search

1. Take in natural language query
2. Rewrite and normalize query
3. Create query embedding
4. Filter candidates (metadata filtering)
5. Get candidates from FAISS
6. Re-rank candidates (heuristics + BM25, probably won't use neural re-ranker)
7. Return results
8. Collect ratings for fine-tuning

### 1.3 Fine-Tuning

-   TODO

## x. Additions

-   Create custom algorithm retrieval benchmark
-   Replacing FAISS with my own search implementation could also be fun (HNSW or IVF+PQ) - might be a whole other project though
-   Could add intent detection (for lexical vs semantic search decision - right now the system is only semantic)
-   Collect more data for fine-tuning (click-through, etc)

## Acknowledgements

Thank you to arXiv for use of its open access interoperability.
