import json
from src.embeddings.embedder import Embedder
from src.index.vector_store import VectorStore
from src.search.search_engine import SearchEngine

def main():
    # Load documents
    with open('data/sample_docs.json', 'r') as f:
        documents = json.load(f)

    # Initialize search engine
    search_engine = SearchEngine()

    # Index documents
    search_engine.index_documents(documents)

    print("Indexing complete.")

if __name__ == '__main__':
    main()