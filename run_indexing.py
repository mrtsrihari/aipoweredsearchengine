import json
from src.embeddings.embedder import TextEmbedder
from src.index.vector_store import VectorDatabase
from src.search.search_engine import SemanticSearchEngine

def main():
    # Load documents
    with open('data/sample_docs.json', 'r') as f:
        documents = json.load(f)

    # Add IDs if missing
    for idx, doc in enumerate(documents):
        if 'id' not in doc:
            doc['id'] = f"doc_{idx}"

    # Initialize search engine
    search_engine = SemanticSearchEngine()

    # Index documents
    indexed_count = search_engine.index_documents(documents)
    print(f"Successfully indexed {indexed_count} documents.")

if __name__ == '__main__':
    main()