from flask import Flask, request, jsonify
from src.search.search_engine import SearchEngine

app = Flask(__name__)

search_engine = SearchEngine()

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    results = search_engine.search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)