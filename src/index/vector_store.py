"""
In-Memory Vector Database Module

This module implements a lightweight, custom vector database that stores
embeddings and performs similarity search without external dependencies.
It uses manual cosine similarity computation and efficient data structures
for fast nearest-neighbor retrieval.

Key Features:
- Store vectors with optional metadata
- Perform cosine similarity search manually
- Support top-k nearest neighbor queries
- In-memory storage for speed
- No external vector DB dependencies
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from heapq import nsmallest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    A custom in-memory vector database for semantic search.
    
    This database stores embedding vectors and provides efficient similarity
    search capabilities. It implements cosine similarity manually to avoid
    external dependencies while maintaining fast search performance.
    
    The similarity search works by:
    1. Computing cosine similarity between query and all stored vectors
    2. Sorting by similarity score
    3. Returning top-k most similar vectors with their indices
    
    Attributes:
        vector_storage_collection: NumPy array storing all embedding vectors
        total_vectors_count: Number of vectors currently stored
        vector_embedding_dimension: Dimensionality of vectors
        vector_metadata_mapping: Optional dict mapping indices to metadata
        vector_norm_cache: Cached L2 norms for efficient similarity computation
    """
    
    def __init__(self, embedding_dimension: int) -> None:
        """
        Initialize an empty vector database.
        
        Args:
            embedding_dimension: The fixed dimensionality of vectors to store.
                                All vectors must have this dimension.
        
        Raises:
            ValueError: If dimension is <= 0
        
        Example:
            >>> db = VectorDatabase(embedding_dimension=384)
        """
        if embedding_dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {embedding_dimension}")
        
        # Initialize storage structures
        self.vector_storage_collection = np.empty((0, embedding_dimension), dtype=np.float32)
        self.total_vectors_count = 0
        self.vector_embedding_dimension = embedding_dimension
        
        # Storage for optional metadata associated with vectors
        self.vector_metadata_mapping: Dict[int, Dict[str, Any]] = {}
        
        # Cache normalized vector norms for fast similarity computation
        # Storing norms avoids recomputation during repeated searches
        self.vector_norm_cache: List[float] = []
        
        logger.info(f"VectorDatabase initialized: dimension={embedding_dimension}")
    
    def add_vectors(self, embedding_vectors: np.ndarray, 
                   metadata_list: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Add embedding vectors to the database.
        
        This method:
        1. Validates vector dimensions match database
        2. Normalizes vectors to unit length (if needed)
        3. Caches vector norms for similarity computation
        4. Stores associated metadata
        
        Args:
            embedding_vectors: NumPy array of shape (num_vectors, dimension)
                              or (dimension,) for single vector
            metadata_list: Optional list of metadata dicts, one per vector.
                          Each dict can contain any key-value pairs.
        
        Returns:
            int: Number of vectors successfully added
        
        Raises:
            ValueError: If vector dimension doesn't match database dimension
            TypeError: If input is not numpy array or convertible to array
        
        Example:
            >>> db = VectorDatabase(384)
            >>> vectors = np.random.randn(5, 384).astype(np.float32)
            >>> added = db.add_vectors(vectors)
            >>> print(f"Added {added} vectors")
            
            >>> # With metadata
            >>> metadata = [{'id': 'doc1'}, {'id': 'doc2'}]
            >>> db.add_vectors(vectors[:2], metadata)
        """
        # Convert to numpy array if needed
        if not isinstance(embedding_vectors, np.ndarray):
            try:
                embedding_vectors = np.array(embedding_vectors, dtype=np.float32)
            except Exception as conversion_error:
                raise TypeError(f"Cannot convert input to numpy array: {conversion_error}")
        
        # Handle single vector case
        if embedding_vectors.ndim == 1:
            embedding_vectors = embedding_vectors.reshape(1, -1)
        
        # Validate shape
        if embedding_vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {embedding_vectors.shape}")
        
        num_vectors_to_add, vector_dimension = embedding_vectors.shape
        
        # Validate dimension matches database
        if vector_dimension != self.vector_embedding_dimension:
            raise ValueError(
                f"Vector dimension {vector_dimension} doesn't match "
                f"database dimension {self.vector_embedding_dimension}"
            )
        
        # Ensure vectors are float32 for consistency
        if embedding_vectors.dtype != np.float32:
            embedding_vectors = embedding_vectors.astype(np.float32)
        
        # Validate metadata if provided
        if metadata_list is not None:
            if len(metadata_list) != num_vectors_to_add:
                raise ValueError(
                    f"Metadata list length {len(metadata_list)} doesn't match "
                    f"vectors count {num_vectors_to_add}"
                )
        
        # Compute and cache L2 norms for each vector
        # This enables fast cosine similarity: cos_sim = dot_product / (norm1 * norm2)
        vector_norms_batch = np.linalg.norm(embedding_vectors, axis=1)
        
        # Add vectors to storage
        current_start_index = self.total_vectors_count
        self.vector_storage_collection = np.vstack([
            self.vector_storage_collection,
            embedding_vectors
        ])
        
        # Cache norms
        self.vector_norm_cache.extend(vector_norms_batch.tolist())
        
        # Store metadata if provided
        if metadata_list is not None:
            for offset, metadata_entry in enumerate(metadata_list):
                vector_index = current_start_index + offset
                self.vector_metadata_mapping[vector_index] = metadata_entry
        
        # Update count
        self.total_vectors_count += num_vectors_to_add
        
        logger.info(f"Added {num_vectors_to_add} vectors. "
                   f"Total vectors: {self.total_vectors_count}")
        
        return num_vectors_to_add
    
    def _compute_cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and all stored vectors.
        
        Cosine similarity measures angular distance in vector space.
        Formula: cos_sim(u, v) = (u · v) / (||u|| * ||v||)
        
        This implementation:
        1. Computes dot product with all stored vectors (vectorized)
        2. Gets query vector norm
        3. Divides by product of norms
        
        Args:
            query_vector: Single vector of shape (dimension,)
        
        Returns:
            np.ndarray: Array of similarity scores for each stored vector
                       Values between -1 and 1
        
        Raises:
            ValueError: If query dimension doesn't match database
        """
        # Validate query dimension
        if len(query_vector) != self.vector_embedding_dimension:
            raise ValueError(
                f"Query dimension {len(query_vector)} doesn't match "
                f"database dimension {self.vector_embedding_dimension}"
            )
        
        # Handle empty database
        if self.total_vectors_count == 0:
            return np.array([])
        
        # Compute L2 norm of query vector
        query_norm = np.linalg.norm(query_vector)
        
        # Handle zero-norm edge case
        if query_norm == 0:
            logger.warning("Query vector has zero norm")
            return np.zeros(self.total_vectors_count)
        
        # Compute dot products with all stored vectors (vectorized operation)
        dot_products = np.dot(self.vector_storage_collection, query_vector)
        
        # Compute cosine similarity: dot_product / (query_norm * vector_norms)
        # Convert norms to array for element-wise division
        stored_norms_array = np.array(self.vector_norm_cache, dtype=np.float32)
        
        # Avoid division by zero for zero-norm vectors
        denominator = query_norm * stored_norms_array
        denominator = np.where(denominator == 0, 1, denominator)
        
        similarity_scores = dot_products / denominator
        
        return similarity_scores
    
    def search(self, query_vector: np.ndarray, 
              top_k: int = 5,
              return_distances: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find top-k most similar vectors to query.
        
        This method:
        1. Computes cosine similarity with all vectors
        2. Identifies top-k highest similarity scores
        3. Returns indices and optionally distances
        
        Args:
            query_vector: Query vector of shape (dimension,)
            top_k: Number of top results to return. Default 5.
            return_distances: If True, returns similarity scores instead of distances.
                            Default False.
        
        Returns:
            Tuple of:
            - distances: Array of shape (1, top_k) with scores/distances
            - indices: Array of shape (1, top_k) with vector indices
        
        Raises:
            ValueError: If top_k > number of stored vectors
            RuntimeError: If no vectors are stored
        
        Example:
            >>> db = VectorDatabase(384)
            >>> vectors = np.random.randn(100, 384).astype(np.float32)
            >>> db.add_vectors(vectors)
            >>> query = np.random.randn(384).astype(np.float32)
            >>> scores, indices = db.search(query, top_k=5)
            >>> print(f"Top match index: {indices[0][0]}")
        """
        # Validate database is not empty
        if self.total_vectors_count == 0:
            raise RuntimeError("Cannot search empty database")
        
        # Limit top_k to available vectors
        top_k = min(top_k, self.total_vectors_count)
        
        # Compute similarity with all vectors
        similarity_scores = self._compute_cosine_similarity(query_vector)
        
        # Find indices of top-k highest similarities
        # Use nsmallest with negative scores to get largest values efficiently
        if return_distances:
            # Return similarity scores (not distances)
            top_k_indices = np.argsort(-similarity_scores)[:top_k]
            top_k_scores = similarity_scores[top_k_indices]
        else:
            # Return distances (inverse of similarity)
            distances = 1.0 - similarity_scores  # Convert similarity to distance
            top_k_indices = np.argsort(distances)[:top_k]
            top_k_scores = distances[top_k_indices]
        
        # Format output as 2D arrays matching expected interface
        distances_output = top_k_scores.reshape(1, -1)
        indices_output = top_k_indices.reshape(1, -1)
        
        return distances_output, indices_output
    
    def search_with_metadata(self, query_vector: np.ndarray, 
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search and return results with metadata and similarity scores.
        
        This is a convenience method that combines search results with
        stored metadata for easier interpretation.
        
        Args:
            query_vector: Query vector of shape (dimension,)
            top_k: Number of top results. Default 5.
        
        Returns:
            List of dicts with keys: 'index', 'score', 'metadata'
        
        Example:
            >>> results = db.search_with_metadata(query_vector, top_k=5)
            >>> for result in results:
            ...     print(f"Index: {result['index']}, Score: {result['score']:.4f}")
            ...     print(f"Metadata: {result['metadata']}")
        """
        # Get top-k search results with similarity scores
        similarity_scores, indices = self.search(query_vector, top_k, return_distances=True)
        
        # Build results list with metadata
        results_list = []
        for rank, (score, idx) in enumerate(zip(similarity_scores[0], indices[0]), 1):
            idx_int = int(idx)
            result_entry = {
                'rank': rank,
                'index': idx_int,
                'score': float(score),
                'metadata': self.vector_metadata_mapping.get(idx_int, {})
            }
            results_list.append(result_entry)
        
        return results_list
    
    def get_vector(self, vector_index: int) -> Optional[np.ndarray]:
        """
        Retrieve a specific vector by index.
        
        Args:
            vector_index: Index of vector to retrieve
        
        Returns:
            Vector array or None if index is invalid
        """
        if not (0 <= vector_index < self.total_vectors_count):
            return None
        
        return self.vector_storage_collection[vector_index].copy()
    
    def get_metadata(self, vector_index: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific vector.
        
        Args:
            vector_index: Index of vector
        
        Returns:
            Metadata dict or None if not found
        """
        return self.vector_metadata_mapping.get(vector_index)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict with vector count, dimension, and memory usage estimate
        """
        # Estimate memory usage (rough approximation)
        vector_memory_bytes = self.total_vectors_count * self.vector_embedding_dimension * 4  # float32 = 4 bytes
        memory_mb = vector_memory_bytes / (1024 * 1024)
        
        return {
            'total_vectors': self.total_vectors_count,
            'embedding_dimension': self.vector_embedding_dimension,
            'estimated_memory_mb': round(memory_mb, 2),
            'vectors_with_metadata': len(self.vector_metadata_mapping)
        }
    
    def clear(self) -> None:
        """Remove all vectors from database."""
        self.vector_storage_collection = np.empty((0, self.vector_embedding_dimension), dtype=np.float32)
        self.total_vectors_count = 0
        self.vector_metadata_mapping.clear()
        self.vector_norm_cache.clear()
        logger.info("Database cleared")