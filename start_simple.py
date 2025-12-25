#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

# Disable torch 
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add the project the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 60)
print("AI SEARCH ENGINE - STARTING")
print("=" * 60)

print("\n[1] Importing modules...")
try:
    from src.embeddings.embedder import TextEmbedder
    from src.index.vector_store import VectorDatabase
    from src.search.search_engine import SemanticSearchEngine
    from src.api.app import app
    print("    [OK] All modules imported")
except Exception as e:
    print(f"    [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2] Initializing components...")
try:
    search_engine = SemanticSearchEngine()
    print("    [OK] Search engine ready")
except Exception as e:
    print(f"    [ERROR] {e}")
    sys.exit(1)

print("\n[3] Loading documents...")
try:
    import json
    with open('data/sample_docs.json', 'r') as f:
        documents = json.load(f)
    
    for idx, doc in enumerate(documents):
        if 'id' not in doc:
            doc['id'] = f"doc_{idx}"
    
    count = search_engine.index_documents(documents)
    print(f"    [OK] {count} documents indexed")
except Exception as e:
    print(f"    [ERROR] {e}")
    sys.exit(1)

print("\n[4] Running Flask server...")
print("=" * 60)
print("  http://localhost:5000/           - Web interface")
print("  http://localhost:5000/health     - Health check")
print("  http://localhost:5000/search?q=X - Search API")
print("=" * 60 + "\n")

try:
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)
