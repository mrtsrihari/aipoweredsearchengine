#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Search Engine Server - Production Startup
Initializes and runs the Flask API server
"""
import os
import sys

# Environment setup
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 70)
print(" "*20 + "AI SEARCH ENGINE SERVER")
print("=" * 70)

# Step 1: Import modules
print("\n[INIT] Importing modules...", end=" ", flush=True)
try:
    from src.api.app import app
    print("[OK]")
except Exception as e:
    print(f"[ERROR] FAILED: {e}")
    sys.exit(1)

# Step 2: Run server
print("[INIT] Starting Flask server on http://0.0.0.0:5000")
print("-" * 70)
print("API Endpoints:")
print("  GET  /health              - Health check")
print("  GET  /search?q=<query>    - Search documents")
print("-" * 70 + "\n")

# Start Flask app
try:
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        use_reloader=False,
        threaded=True,
        use_debugger=False
    )
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)
