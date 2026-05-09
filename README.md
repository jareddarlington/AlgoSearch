# AlgoSearch: Semantic Search for Algorithms on arXiv

<!-- ## What is Semantic Search?

TODO

PUT DEMONSTRATIVE IMAGE HERE

## Usage

TODO -->

## System Design

### Building Vector Database

1. Build papers database
    - Scrape arXiv papers; see `scrape_arxiv.py`
    - Store paper metadata in `data/papers.db`
    - If LaTeX source contains an algorithm environment, store it temporarily in `data/temp/{id}.txt`

2. Build algorithms database
    - Pass LaTeX source into an LLM and build a profile for each algorithm defined in the paper; see `extract_algorithms.py`
    - Store algorithm profiles in `data/algorithms.db`

3. Build FAISS index
    - See `build_faiss_index.py`
    - Embed algorithm profile (name, description, categories)
    - Build and store FAISS index in `data/index.faiss`

<!-- \* Caveat on the database: ... -->

### Search

 <!-- 3. Metadata candidate filtering (TODO) -->

1. Take in natural language query
2. Create query embedding (Qwen3 dense embeddings)
3. Retrieve top-k candidates (FAISS)
4. Rerank candidates (cross-encoder reranking)
5. Return top-n results

<!-- ### Fine-Tuning (TODO)

- Possible relevance signals: y/n relevance question to user, paper click-through rate, etc -->

## Design Priorities & Philosophy

The following were my main priorities and constraints while building AlgoSearch:

1. **Cheap**: I'm a student and don't have money to spend on fun projects like this, so, I knew if I wanted to make the project happen, I needed to keep everything **free**. I was able to accomplish this by using open source tools and running models locally.

2. **Low resource**: I don't own a GPU...

3. **Fast**: From personal experience, I know that people hate waiting more than 3 seconds for anything to load, so making the system fast was high priority for me. I honestly didn't expect this one to be as big of a challenge until I finished my basic implementation and realized it was taking 15+ seconds per query to return results. In the end, I was able to accomplish consistent **sub-150ms latency** on an Mac mini.

4. **Performant**: I considered performance lower priority mostly due to my resource and speed constraints, but also because the reason I started this project was to challenge my system design skills and build my understanding of search pipelines, not necessarily to ace benchmarks.

<!-- ## Ideas & Additions

### Easier Implementations

1. Create demo website (probably host on either GitHub Pages or Hugging Face Spaces)
2. Add back LLM query spell correction / normalization and possibly multiple query rewriting for higher recall if latency allows

### Harder Implementations

1. Write my own ANN implementation / FAISS replacement
2. Embed raw LaTeX source or code snippets using a code embedding model for late fusion (could improve recall and playing around with mulitmodal search techniques would fun + I would learn a lot)
3. Create algorithm-specific retrieval benchmark -->

<!-- ## What I learned

TODO

system design is hard, compute is expensive, search and rec sys is fun -->

## Acknowledgements

**Tools & Libraries**: FAISS, PyTorch, sentence-transformers (ms-marco-MiniLM-L2-v2), FlagEmbedding (BGE-M3), Google Gemini API (gemini-2.5-flash), SQLite

Thank you to arXiv for use of its open access interoperability.
