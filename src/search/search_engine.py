"""
Semantic Search Engine Module

This module provides a complete search engine implementation that performs
semantic search on a collection of documents. It uses text embeddings to
understand document meaning and returns results ranked by semantic similarity.

The search engine handles document indexing, embedding generation, and
efficient similarity-based retrieval through vector search.
"""

from src.embeddings.embedder import TextEmbedder
from src.index.vector_store import VectorStore
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

# Configure logging for tracking operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchResult:
    """
    Represents a single search result with document info and relevance score.
    
    Attributes:
        doc_id: Unique identifier for the document
        document: The original document content
        score: Similarity score (0-1, where 1 is most similar)
        rank: Position in the result ranking (1-indexed)
    """
    
    def __init__(self, doc_id: str, document: Dict[str, Any], 
                 score: float, rank: int) -> None:
        """
        Initialize a search result.
        
        Args:
            doc_id: Unique document identifier
            document: Original document dictionary
            score: Similarity score
            rank: Result ranking position
        """
        self.doc_id = doc_id
        self.document = document
        self.score = score
        self.rank = rank
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format for easy serialization.
        
        Returns:
            Dict containing document info, score, and rank
        """
        return {
            'rank': self.rank,
            'doc_id': self.doc_id,
            'score': float(self.score),
            'document': self.document
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"SearchResult(rank={self.rank}, doc_id={self.doc_id}, score={self.score:.4f})"


class SemanticSearchEngine:
    """
    A semantic search engine for finding similar documents by meaning.
    
    This engine converts documents into semantic embeddings and uses
    vector similarity search to find the most relevant documents for a query.
    It supports batch indexing, single and batch queries, and filtering.
    
    The search process:
    1. Documents are indexed with unique IDs
    2. Text is converted to embeddings (vectors capturing semantic meaning)
    3. Embeddings are stored in an efficient vector index
    4. Queries are embedded using the same method
    5. Top-k most similar documents are retrieved by vector distance
    
    Attributes:
        embedder: TextEmbedder instance for generating embeddings
        vector_store: VectorStore instance for similarity search
        documents: Dictionary mapping document IDs to document content
        is_indexed: Boolean indicating if documents have been indexed
    """
    
    def __init__(self, 
                 embedder: Optional[TextEmbedder] = None,
                 vector_store: Optional[VectorStore] = None) -> None:
        """
        Initialize the semantic search engine.
        
        Args:
            embedder: TextEmbedder instance. If None, creates default embedder.
            vector_store: VectorStore instance. If None, creates default store.
        
        Example:
            >>> engine = SemanticSearchEngine()
            >>> engine = SemanticSearchEngine(
            ...     embedder=TextEmbedder('all-mpnet-base-v2')
            ... )
        """
        # Initialize embedder for converting text to vectors
        self.embedder = embedder or TextEmbedder()
        
        # Initialize vector store for efficient similarity search
        embedding_dim = self.embedder.get_embedding_dimension()
        self.vector_store = vector_store or VectorStore(dim=embedding_dim)
        
        # Dictionary to store original documents by ID for result reconstruction
        self.documents: Dict[str, Dict[str, Any]] = {}
        
        # Track indexing state
        self.is_indexed = False
        
        logger.info("SemanticSearchEngine initialized successfully")
    
    def index_documents(self, documents: List[Dict[str, Any]], 
                       id_field: str = 'id',
                       text_fields: List[str] = ['content', 'text', 'title']) -> int:
        """
        Index a collection of documents for searching.
        
        This method:
        1. Validates documents have required ID and text fields
        2. Extracts and combines text from specified fields
        3. Generates embeddings for each document
        4. Stores embeddings in the vector index
        
        Args:
            documents: List of document dictionaries to index
            id_field: Field name containing document IDs. Default 'id'
            text_fields: List of field names to use for text content.
                        Uses the first available field found. Default is
                        ['content', 'text', 'title']
        
        Returns:
            int: Number of documents successfully indexed
        
        Raises:
            ValueError: If documents list is empty
            KeyError: If ID field not found in documents
        
        Example:
            >>> engine = SemanticSearchEngine()
            >>> docs = [
            ...     {'id': '1', 'content': 'Python programming guide'},
            ...     {'id': '2', 'content': 'JavaScript tutorial'}
            ... ]
            >>> indexed = engine.index_documents(docs)
            >>> print(f"Indexed {indexed} documents")
        """
        # Validate input
        if not documents:
            raise ValueError("Cannot index empty document collection")
        
        doc_ids = []
        text_content = []
        
        # Extract IDs and text from documents
        for doc in documents:
            # Validate ID field exists
            if id_field not in doc:
                raise KeyError(f"Document missing required field '{id_field}'")
            
            doc_id = str(doc[id_field])
            doc_ids.append(doc_id)
            
            # Extract text from first available text field
            text = self._extract_text(doc, text_fields)
            if not text:
                logger.warning(f"Document {doc_id} has no extractable text")
                text = f"Document {doc_id}"
            
            text_content.append(text)
            # Store original document for later retrieval
            self.documents[doc_id] = doc
        
        # Generate embeddings for all documents
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedder.embed_batch(text_content)
        
        # Add embeddings to vector store
        logger.info("Adding embeddings to vector index...")
        self.vector_store.add(embeddings)
        
        self.is_indexed = True
        logger.info(f"Successfully indexed {len(doc_ids)} documents")
        
        return len(doc_ids)
    
    def search(self, query: str, k: int = 5, 
              return_scores: bool = True) -> List[SearchResult]:
        """
        Search for documents similar to the query.
        
        This method:
        1. Embeds the query text to a vector
        2. Searches for the k nearest neighbors in vector space
        3. Reconstructs and ranks results with similarity scores
        
        Args:
            query: Query text string
            k: Number of top results to return. Default is 5.
            return_scores: Whether to include similarity scores.
                          Default is True.
        
        Returns:
            List of SearchResult objects ranked by similarity (best first)
        
        Raises:
            RuntimeError: If documents haven't been indexed yet
            ValueError: If k <= 0 or query is empty
        
        Example:
            >>> engine = SemanticSearchEngine()
            >>> engine.index_documents(docs)
            >>> results = engine.search("python tutorial", k=3)
            >>> for result in results:
            ...     print(f"{result.rank}. {result.doc_id} (score: {result.score:.3f})")
        """
        # Validate engine state
        if not self.is_indexed:
            raise RuntimeError("Documents must be indexed before searching. "
                             "Call index_documents() first.")
        
        # Validate inputs
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if k <= 0:
            raise ValueError(f"k must be positive integer, got {k}")
        
        # Limit k to number of indexed documents
        k = min(k, len(self.documents))
        
        # Embed the query using same method as documents
        query_embedding = self.embedder.embed(query)
        
        # Search for nearest neighbors in vector space
        # Returns distances and indices
        distances, indices = self.vector_store.search(
            query_embedding, 
            k=k
        )
        
        # Reconstruct results with metadata
        results = []
        doc_ids_list = list(self.documents.keys())
        
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), 1):
            # Validate index is within bounds
            if idx >= len(doc_ids_list):
                logger.warning(f"Index {idx} out of bounds, skipping result")
                continue
            
            doc_id = doc_ids_list[idx]
            document = self.documents[doc_id]
            
            # Convert distance to similarity score (0-1 range)
            # For L2 distance: similarity ≈ 1 / (1 + distance)
            similarity_score = self._distance_to_similarity(float(distance))
            
            result = SearchResult(
                doc_id=doc_id,
                document=document,
                score=similarity_score,
                rank=rank
            )
            results.append(result)
        
        logger.info(f"Search returned {len(results)} results")
        return results
    
    def search_batch(self, queries: List[str], k: int = 5) -> Dict[str, List[SearchResult]]:
        """
        Perform multiple searches efficiently in batch.
        
        This method searches for multiple queries and returns results
        organized by query string.
        
        Args:
            queries: List of query strings
            k: Number of top results per query. Default is 5.
        
        Returns:
            Dictionary mapping query strings to lists of SearchResult objects
        
        Example:
            >>> engine = SemanticSearchEngine()
            >>> engine.index_documents(docs)
            >>> queries = ["python", "javascript", "web development"]
            >>> all_results = engine.search_batch(queries, k=3)
            >>> for query, results in all_results.items():
            ...     print(f"Query: {query}")
            ...     for r in results:
            ...         print(f"  - {r.document['title']}")
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, k=k)
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the original document by ID.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Original document dictionary, or None if not found
        
        Example:
            >>> doc = engine.get_document('doc_1')
            >>> print(doc['content'])
        """
        return self.documents.get(doc_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed documents and engine state.
        
        Returns:
            Dictionary with document count, embedding dimension, and indexing status
        """
        return {
            'num_documents': len(self.documents),
            'is_indexed': self.is_indexed,
            'embedding_dimension': self.embedder.get_embedding_dimension(),
            'model_name': self.embedder.model_name
        }
    
    @staticmethod
    def _extract_text(document: Dict[str, Any], 
                     fields: List[str]) -> str:
        """
        Extract text from document using list of possible field names.
        
        Returns the content of the first field found in the document.
        
        Args:
            document: Document dictionary
            fields: List of field names to try in order
        
        Returns:
            Extracted text or empty string if no fields found
        """
        for field in fields:
            if field in document and document[field]:
                content = document[field]
                # Ensure content is string
                return str(content)
        return ""
    
    @staticmethod
    def _distance_to_similarity(distance: float, method: str = 'inverse') -> float:
        """
        Convert L2 distance to similarity score in range [0, 1].
        
        Args:
            distance: L2 distance value
            method: Conversion method ('inverse' or 'exponential'). Default 'inverse'.
        
        Returns:
            Similarity score between 0 and 1
        """
        if method == 'inverse':
            # Inverse relationship: smaller distances → higher similarity
            # Avoid division by zero
            return 1.0 / (1.0 + distance)
        else:
            # Exponential decay method
            return float(np.exp(-distance))