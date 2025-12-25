#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

print("=" * 60)
print("AI SEARCH ENGINE - TESTING & INITIALIZATION")
print("=" * 60)

# Test 1: Check imports
print("\n[1] Testing imports...")
try:
    from src.embeddings.embedder import TextEmbedder
    print("    [OK] TextEmbedder imported successfully")
    from src.index.vector_store import VectorDatabase
    print("    [OK] VectorDatabase imported successfully")
    from src.search.search_engine import SemanticSearchEngine
    print("    [OK] SemanticSearchEngine imported successfully")
    from src.api.app import app
    print("    [OK] Flask app imported successfully")
except Exception as e:
    print(f"    [ERROR] Import error: {e}")
    sys.exit(1)

# Test 2: Initialize search engine
print("\n[2] Initializing search engine...")
try:
    search_engine = SemanticSearchEngine()
    print("    [OK] Search engine initialized")
except Exception as e:
    print(f"    [ERROR] Initialization error: {e}")
    sys.exit(1)

# Test 3: Load and index documents
print("\n[3] Loading and indexing documents...")
try:
    import json
    with open('data/sample_docs.json', 'r') as f:
        documents = json.load(f)
    
    # Add IDs if missing
    for idx, doc in enumerate(documents):
        if 'id' not in doc:
            doc['id'] = f"doc_{idx}"
    
    indexed_count = search_engine.index_documents(documents)
    print(f"    [OK] Successfully indexed {indexed_count} documents")
except Exception as e:
    print(f"    [ERROR] Indexing error: {e}")
    sys.exit(1)

# Test 4: Perform a test search
print("\n[4] Performing test search...")
try:
    test_query = "machine learning"
    results = search_engine.search(test_query, k=3)
    print(f"    [OK] Search completed! Found {len(results)} results for '{test_query}':")
    for i, result in enumerate(results, 1):
        print(f"      {i}. {result.document.get('title', 'N/A')} (Score: {result.score:.4f})")
except Exception as e:
    print(f"    [ERROR] Search error: {e}")
    sys.exit(1)

print("\n[5] Starting Flask server...")
print("=" * 60)
print("  Server running at: http://0.0.0.0:5000")
print("  Frontend at:       http://localhost:5000/")
print("  Health check:      http://localhost:5000/health")
print("  Search API:        http://localhost:5000/search?q=<query>")
print("=" * 60 + "\n")

# Start the Flask app
app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
