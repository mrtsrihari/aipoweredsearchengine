#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the AI Search Engine Server"""
import os
import sys

os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 70)
print(" " * 20 + "AI SEARCH ENGINE")
print("=" * 70)
print("\n[INIT] Importing Flask app...")

from src.api.app import app

print("[OK] App loaded successfully")
print("[OK] Starting server on http://0.0.0.0:5000")
print("-" * 70)
print("Endpoints:")
print("  GET /health           - Health check")
print("  GET /search?q=<query> - Search documents")
print("-" * 70 + "\n")


app.config['JSON_SORT_KEYS'] = False

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped")
        sys.exit(0)
