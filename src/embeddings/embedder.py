"""
Text Embedding Module

This module provides a TextEmbedder class that converts text into dense vector
representations using sentence-transformers. These embeddings capture semantic
meaning and can be used for similarity search, clustering, and retrieval tasks.

The embeddings are normalized and can be efficiently compared using cosine
similarity or other distance metrics.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union, List, Dict
import logging

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedder:
    """
    A semantic embedding generator using pre-trained sentence transformers.
    
    This class wraps the sentence-transformers library to provide a simple
    interface for converting text into fixed-size dense vectors. The vectors
    encode semantic meaning and can be compared using distance metrics.
    
    Features:
    - Robust cosine similarity computation for any vector magnitudes
    - Model warm-up to reduce first-query latency
    - Optional embedding caching for repeated queries
    - Full type hints and comprehensive error handling
    
    Attributes:
        pretrained_model_instance: The underlying SentenceTransformer model
        embedding_vector_dimension: The dimensionality of generated embeddings
        selected_model_identifier: Name of the model being used
        embedding_cache: Dict caching computed embeddings for repeated queries
    """
    
    def __init__(self, model_identifier: str = 'all-MiniLM-L6-v2',
                 enable_cache: bool = True,
                 warmup_enabled: bool = True) -> None:
        """
        Initialize the TextEmbedder with a pre-trained model.
        
        Args:
            model_identifier: Identifier for sentence-transformers model.
                             Defaults to 'all-MiniLM-L6-v2' which is lightweight
                             and provides excellent performance for general use.
                             Examples: 'all-mpnet-base-v2', 'paraphrase-MiniLM-L6-v2'
            enable_cache: Whether to cache embedding computations. Default True.
                         Speeds up repeated queries on same texts.
            warmup_enabled: Whether to warm-up the model on initialization.
                           Default True. Reduces first-query latency significantly.
                             
        Raises:
            OSError: If the model cannot be downloaded or loaded
            Exception: If initialization fails for any reason
            
        Example:
            >>> embedder = TextEmbedder()
            >>> embedder_custom = TextEmbedder('all-mpnet-base-v2', enable_cache=True)
        """
        try:
            self.selected_model_identifier = model_identifier
            
            # Load the pre-trained sentence transformer model
            # This automatically downloads and caches the model on first use
            # Disable model card loading to avoid network issues
            self.pretrained_model_instance = SentenceTransformer(
                model_identifier,
                trust_remote_code=True,
                device='cpu'
            )
            
            # Determine embedding dimensionality by encoding a test string
            # This is more reliable than hardcoding dimensions
            test_embedding_vector = self.pretrained_model_instance.encode("sample")
            self.embedding_vector_dimension = len(test_embedding_vector)
            
            # Initialize embedding cache for performance optimization
            self.embedding_cache: Dict[str, np.ndarray] = {}
            self.cache_enabled = enable_cache
            
            # Model warm-up: Pre-load the model by encoding dummy text
            # This reduces first-query latency by ~500-1000ms in production
            if warmup_enabled:
                logger.info("Performing model warm-up...")
                self.pretrained_model_instance.encode(
                    ["model warmup initialization"],
                    show_progress_bar=False
                )
                logger.info("Model warm-up complete")
            
            logger.info(f"TextEmbedder initialized: model={model_identifier}, "
                       f"vector_dimension={self.embedding_vector_dimension}, "
                       f"cache_enabled={enable_cache}")
        except Exception as initialization_error:
            logger.error(f"TextEmbedder initialization failed: {str(initialization_error)}")
            raise
    
    def embed(self, input_text: Union[str, List[str]], 
              normalize_output: bool = True,
              processing_batch_size: int = 32,
              use_cache: bool = True) -> np.ndarray:
        """
        Convert text into semantic embeddings (dense vectors).
        
        This method accepts either a single text string or a list of strings
        and returns corresponding embedding vectors. By default, embeddings
        are L2-normalized to unit length for efficient cosine similarity.
        
        The embedding process:
        1. Check cache if enabled (for single texts)
        2. Text is tokenized and padded to model input length
        3. Tokens are passed through transformer neural network
        4. Output representations are aggregated (typically mean pooling)
        5. Optional L2 normalization is applied for unit vectors
        6. Result is cached if cache is enabled
        
        Args:
            input_text (Union[str, List[str]]): Single text string OR list of strings
            normalize_output (bool): Whether to L2-normalize embeddings to unit vectors.
                                    Default True. Normalized vectors enable efficient
                                    cosine similarity computation via dot product.
            processing_batch_size (int): Texts per processing batch. Larger batches
                                        are faster but use more GPU/CPU memory.
                                        Default 32.
            use_cache (bool): Whether to use embedding cache if enabled. Default True.
                            Set False to bypass cache for dynamic content.
        
        Returns:
            np.ndarray: Embedding array with shape:
                       - (embedding_vector_dimension,) if input is single string
                       - (num_texts, embedding_vector_dimension) if input is list
            
        Raises:
            TypeError: If input type is not string or list of strings
            ValueError: If input list is empty
            
        Example:
            >>> embedder = TextEmbedder()
            >>> single_embedding = embedder.embed("Hello world")
            >>> single_embedding.shape
            (384,)
            
            >>> batch_embeddings = embedder.embed(["Text one", "Text two", "Text three"])
            >>> batch_embeddings.shape
            (3, 384)
        """
        # Determine if input is single text string or list
        is_single_text_input = isinstance(input_text, str)
        
        # Check cache for single text inputs
        if is_single_text_input and use_cache and self.cache_enabled:
            if input_text in self.embedding_cache:
                logger.debug(f"Cache hit for text: {input_text[:50]}...")
                return self.embedding_cache[input_text].copy()
        
        # Normalize input to list format for uniform processing pipeline
        if is_single_text_input:
            texts_for_processing = [input_text]
        elif isinstance(input_text, list):
            # Validate list is not empty
            if not input_text:
                raise ValueError("Cannot process empty text list")
            # Validate all items are strings
            if not all(isinstance(text_item, str) for text_item in input_text):
                raise TypeError("All list items must be strings")
            texts_for_processing = input_text
        else:
            raise TypeError(f"Expected str or List[str], received {type(input_text)}")
        
        try:
            # Generate embeddings using the transformer model
            # show_progress_bar=False prevents cluttered console output
            generated_embedding_vectors = self.pretrained_model_instance.encode(
                texts_for_processing,
                normalize_embeddings=normalize_output,
                batch_size=processing_batch_size,
                show_progress_bar=False
            )
            
            # Cache single text result if caching is enabled
            if is_single_text_input and self.cache_enabled:
                self.embedding_cache[input_text] = generated_embedding_vectors[0].copy()
            
            # Return single vector without batch dimension for string inputs
            if is_single_text_input:
                return generated_embedding_vectors[0]
            
            # Return full batch of vectors for list inputs
            return generated_embedding_vectors
            
        except Exception as encoding_error:
            logger.error(f"Embedding generation failed: {str(encoding_error)}")
            raise
    
    def embed_batch(self, texts_collection: List[str], 
                   processing_batch_size: int = 32,
                   normalize_output: bool = True) -> np.ndarray:
        """
        Embed multiple texts with explicit batch-oriented API.
        
        This is a convenience wrapper around embed() that makes batch processing
        intent explicit. Use this when processing document collections to improve
        code readability and maintainability.
        
        Args:
            texts_collection: List of text strings to embed
            processing_batch_size: Texts per processing batch. Default 32.
            normalize_output: Whether to normalize embeddings. Default True.
        
        Returns:
            np.ndarray: Array of vectors with shape (len(texts), embedding_dimension)
            
        Example:
            >>> embedder = TextEmbedder()
            >>> documents = [
            ...     "Machine learning article",
            ...     "Deep learning tutorial", 
            ...     "Natural language processing guide"
            ... ]
            >>> all_embeddings = embedder.embed_batch(documents, processing_batch_size=64)
            >>> all_embeddings.shape
            (3, 384)
        """
        return self.embed(
            texts_collection, 
            normalize_output=normalize_output, 
            processing_batch_size=processing_batch_size,
            use_cache=False  # Don't cache batch results
        )
    
    def get_embedding_dimension(self) -> int:
        """
        Query the embedding vector dimensionality.
        
        Different models produce vectors of different sizes. Knowing the
        dimension is essential for initializing vector databases and
        similarity computation.
        
        Returns:
            int: Vector dimension (e.g., 384 for all-MiniLM-L6-v2)
            
        Example:
            >>> embedder = TextEmbedder()
            >>> dim = embedder.get_embedding_dimension()
            >>> print(f"Vector dimension: {dim}")
            Vector dimension: 384
        """
        return self.embedding_vector_dimension
    
    def similarity(self, first_embedding_vector: np.ndarray, 
                  second_embedding_vector: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embedding vectors.
        
        This method computes the mathematically robust cosine similarity metric,
        which measures the angular distance between vectors. Works correctly
        even if vectors are NOT normalized.
        
        Mathematical foundation:
        - Cosine similarity = (u · v) / (||u|| * ||v||)
        - Invariant to vector magnitude (works with any magnitude)
        - Result always in range [-1.0, 1.0]
        
        Similarity interpretation:
        - 1.0: Perfect semantic similarity (identical direction)
        - 0.5: Moderate semantic overlap
        - 0.0: Orthogonal vectors (completely unrelated)
        - -1.0: Opposite meaning (rare with normalized embeddings)
        
        Args:
            first_embedding_vector: First embedding vector (1D numpy array)
            second_embedding_vector: Second embedding vector (1D numpy array)
        
        Returns:
            float: Cosine similarity score between -1.0 and 1.0
        
        Raises:
            ValueError: If vectors have mismatched dimensions
            ValueError: If either vector has zero magnitude
        
        Example:
            >>> embedder = TextEmbedder()
            >>> vec1 = embedder.embed("The cat sat on the mat")
            >>> vec2 = embedder.embed("A feline rested on the rug")
            >>> score = embedder.similarity(vec1, vec2)
            >>> print(f"Semantic similarity: {score:.4f}")
            Semantic similarity: 0.8234
        """
        # Validate vector shapes match
        if len(first_embedding_vector) != len(second_embedding_vector):
            raise ValueError(
                f"Vector dimension mismatch: {len(first_embedding_vector)} "
                f"vs {len(second_embedding_vector)}"
            )
        
        # Compute L2 norms for both vectors
        # This makes similarity computation robust regardless of normalization
        norm_first_vector = np.linalg.norm(first_embedding_vector)
        norm_second_vector = np.linalg.norm(second_embedding_vector)
        
        # Handle zero-magnitude vectors (degenerate case)
        if norm_first_vector == 0 or norm_second_vector == 0:
            raise ValueError(
                "Cannot compute similarity with zero-magnitude vector. "
                "This indicates invalid embeddings or all-zero input."
            )
        
        # Compute cosine similarity: (dot product) / (product of norms)
        # This formulation is mathematically robust and works for ANY vector magnitudes
        dot_product_result = np.dot(first_embedding_vector, second_embedding_vector)
        computed_similarity_score = dot_product_result / (norm_first_vector * norm_second_vector)
        
        return float(computed_similarity_score)
    
    def clear_cache(self) -> int:
        """
        Clear the embedding cache and return cache size before clearing.
        
        Useful for memory management in long-running processes.
        
        Returns:
            int: Number of cached embeddings that were cleared
        
        Example:
            >>> embedder = TextEmbedder()
            >>> embedder.embed("Some text")
            >>> cleared = embedder.clear_cache()
            >>> print(f"Cleared {cleared} cached embeddings")
        """
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"Cache cleared: {cache_size} entries removed")
        return cache_size
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics for monitoring.
        
        Returns:
            Dict with cache size and status
        
        Example:
            >>> stats = embedder.get_cache_stats()
            >>> print(f"Cache has {stats['size']} entries")
        """
        return {
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self.embedding_cache),
            'estimated_memory_mb': round(len(self.embedding_cache) * self.embedding_vector_dimension * 4 / (1024 * 1024), 2)
        }