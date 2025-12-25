# AI Search Engine

This is an AI-powered search engine using embeddings and vector search.

## Installation

1. Install requirements: `pip install -r requirements.txt`


## Usage

1. Run indexing: `python run_indexing.py`

2. Run the API: `python src/api/app.py`

3. Search via API: `GET /search?q=your_query`

## Files Structure

- `README.md`: This file
- `requirements.txt`: Python dependencies
- `data/sample_docs.json`: Sample documents for testing
- `src/embeddings/embedder.py`: Embedding generation using Sentence Transformers
- `src/index/vector_store.py`: Vector storage using FAISS
- `src/search/search_engine.py`: Main search engine logic
- `src/api/app.py`: Flask API for search
- `src/utils/text_cleaner.py`: Text preprocessing utilities
- `run_indexing.py`: Script to index the sample documents
