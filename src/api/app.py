from flask import Flask, request, jsonify
from flask_cors import CORS
from src.search.search_engine import SemanticSearchEngine
import json
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "AI in Education",
        "content": "Artificial intelligence is transforming education through personalized learning, adaptive assessments, and intelligent tutoring systems."
    },
    {
        "id": "doc_2",
        "title": "Machine Learning Basics",
        "content": "Machine learning models learn from data without being explicitly programmed, enabling systems to improve predictions over time."
    },
    {
        "id": "doc_3",
        "title": "Climate Change Impact",
        "content": "Climate change affects global ecosystems, biodiversity, sea levels, and weather patterns, posing risks to human societies."
    },
    {
        "id": "doc_4",
        "title": "Semantic Search Explained",
        "content": "Semantic search understands user intent and context rather than relying on exact keyword matches, producing more meaningful results."
    },
    {
        "id": "doc_5",
        "title": "AI in Healthcare",
        "content": "Artificial intelligence is revolutionizing healthcare through medical imaging analysis, disease prediction, and personalized treatment plans."
    },
    {
        "id": "doc_6",
        "title": "Natural Language Processing",
        "content": "Natural language processing enables machines to understand, interpret, and generate human language using linguistic and statistical methods."
    },
    {
        "id": "doc_7",
        "title": "Deep Learning",
        "content": "Deep learning uses neural networks with multiple layers to model complex patterns in data such as images, speech, and text."
    },
    {
        "id": "doc_8",
        "title": "AI Ethics",
        "content": "AI ethics focuses on fairness, transparency, accountability, and reducing bias in artificial intelligence systems."
    },
    {
        "id": "doc_9",
        "title": "Renewable Energy",
        "content": "Renewable energy sources like solar and wind power help reduce carbon emissions and mitigate climate change."
    },
    {
        "id": "doc_10",
        "title": "AI in Finance",
        "content": "AI is widely used in finance for fraud detection, algorithmic trading, credit scoring, and risk management."
    },
    {
        "id": "doc_11",
        "title": "Search Engines",
        "content": "Modern search engines combine information retrieval, machine learning, and natural language understanding to rank results effectively."
    },
    {
        "id": "doc_12",
        "title": "Big Data Analytics",
        "content": "Big data analytics processes large datasets to uncover patterns, trends, and insights for informed decision-making."
    },
    {
        "id": "doc_13",
        "title": "Artificial Neural Networks",
        "content": "Artificial neural networks are inspired by the human brain and consist of interconnected nodes that learn representations from data."
    },
    {
        "id": "doc_14",
        "title": "Sustainable Development",
        "content": "Sustainable development balances economic growth, environmental protection, and social well-being for future generations."
    },
    {
        "id": "doc_15",
        "title": "AI and Automation",
        "content": "AI-driven automation improves productivity by handling repetitive tasks in industries such as manufacturing and logistics."
    }
]


search_engine = SemanticSearchEngine()
initialized = False


def initialize_documents():
    global initialized

    if initialized:
        return

    try:
        with open("data/sample_docs.json", "r") as f:
            documents = json.load(f)
            logger.info("Loaded documents from data/sample_docs.json")
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Using built-in sample documents")
        documents = SAMPLE_DOCUMENTS

    for idx, doc in enumerate(documents):
        doc.setdefault("id", f"doc_{idx}")
        doc.setdefault("content", doc.get("title", ""))

    count = search_engine.index_documents(documents)
    logger.info(f"Indexed {count} documents")

    initialized = True


@app.before_request
def ensure_initialized():
    initialize_documents()


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Semantix API running",
        "endpoints": ["/health", "/search"]
    })


@app.route("/health", methods=["GET"])
def health():
    initialize_documents()
    return jsonify({
        "status": "ok",
        "indexed": search_engine.is_indexed,
        "document_count": len(search_engine.documents)
    })


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q")

    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    if not search_engine.is_indexed:
        return jsonify({"error": "Search engine not ready"}), 503

    results = search_engine.search(query, k=10)

    response = [
        {
            "score": round(r.score, 4),
            "document": r.document
        }
        for r in results
    ]

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
