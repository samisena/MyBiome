"""
Embedding Engine for Hierarchical Semantic Normalization

Uses Ollama's nomic-embed-text model for generating semantic embeddings.
Supports caching, batch processing, and similarity calculations.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Generate and manage semantic embeddings using Ollama's nomic-embed-text model.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        cache_path: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize the embedding engine.

        Args:
            model: Ollama embedding model name
            base_url: Ollama server URL
            cache_path: Path to embedding cache file (pickle)
            batch_size: Number of texts to process in parallel
        """
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size

        # Set up cache
        self.cache_path = cache_path
        self.cache: Dict[str, np.ndarray] = {}
        if cache_path and os.path.exists(cache_path):
            self._load_cache()

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_embeddings_generated = 0

        logger.info(f"EmbeddingEngine initialized with model: {model}")

    def _load_cache(self):
        """Load embedding cache from disk."""
        try:
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
            logger.info(f"Loaded {len(self.cache)} embeddings from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}

    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.cache_path:
            return

        try:
            # Create cache directory if needed
            cache_dir = os.path.dirname(self.cache_path)
            os.makedirs(cache_dir, exist_ok=True)

            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array (768 dimensions)
        """
        # Check cache first
        if text in self.cache:
            self.cache_hits += 1
            return self.cache[text]

        # Generate new embedding
        self.cache_misses += 1
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()

            embedding = np.array(response.json()['embedding'], dtype=np.float32)

            # Cache the result
            self.cache[text] = embedding
            self.total_embeddings_generated += 1

            # Save cache periodically
            if self.total_embeddings_generated % 50 == 0:
                self._save_cache()

            return embedding

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate embedding for '{text}': {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")

            for text in batch:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)

        return embeddings

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-9)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-9)

        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)

        # Clip to [0, 1] range (handle numerical errors)
        return float(np.clip(similarity, 0.0, 1.0))

    def find_similar(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Find most similar texts to a query.

        Args:
            query_text: Query text to match against
            candidate_texts: List of candidate texts to compare
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (text, similarity_score) tuples, sorted by similarity (descending)
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query_text)

        # Generate candidate embeddings
        candidate_embeddings = self.generate_embeddings_batch(candidate_texts)

        # Calculate similarities
        similarities = []
        for text, embedding in zip(candidate_texts, candidate_embeddings):
            similarity = self.cosine_similarity(query_embedding, embedding)
            if similarity >= min_similarity and text != query_text:
                similarities.append((text, similarity))

        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def batch_similarity_matrix(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Generate full similarity matrix for a list of texts.

        Args:
            texts: List of texts to compare

        Returns:
            NxN similarity matrix where N = len(texts)
        """
        # Generate all embeddings
        embeddings = self.generate_embeddings_batch(texts)

        # Calculate similarity matrix
        n = len(embeddings)
        similarity_matrix = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = self.cosine_similarity(embeddings[i], embeddings[j])
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim  # Symmetric

        return similarity_matrix

    def get_stats(self) -> Dict[str, any]:
        """Get embedding engine statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_embeddings_generated': self.total_embeddings_generated
        }

    def save_cache_now(self):
        """Force save cache to disk."""
        self._save_cache()

    def __del__(self):
        """Cleanup: save cache on destruction."""
        if hasattr(self, 'cache') and self.cache_path:
            self._save_cache()


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_embedding_engine(config_path: Optional[str] = None) -> EmbeddingEngine:
    """
    Load embedding engine with configuration.

    Args:
        config_path: Path to YAML config file (optional)

    Returns:
        Configured EmbeddingEngine instance
    """
    # Default configuration
    default_config = {
        'model': 'nomic-embed-text',
        'base_url': 'http://localhost:11434',
        'cache_path': 'c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/embeddings.pkl',
        'batch_size': 32
    }

    # Load config from file if provided
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            embedding_config = config.get('embedding', {})
            default_config.update(embedding_config)

    return EmbeddingEngine(**default_config)


if __name__ == "__main__":
    # Test the embedding engine
    print("Testing EmbeddingEngine...")

    engine = EmbeddingEngine(
        cache_path="c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/embeddings_test.pkl"
    )

    # Test single embedding
    text = "Lactobacillus reuteri DSM 17938"
    embedding = engine.generate_embedding(text)
    print(f"Generated embedding for '{text}': shape={embedding.shape}, dtype={embedding.dtype}")

    # Test similarity
    text1 = "vitamin D"
    text2 = "cholecalciferol"
    text3 = "chemotherapy"

    emb1 = engine.generate_embedding(text1)
    emb2 = engine.generate_embedding(text2)
    emb3 = engine.generate_embedding(text3)

    sim_12 = engine.cosine_similarity(emb1, emb2)
    sim_13 = engine.cosine_similarity(emb1, emb3)

    print(f"\nSimilarity '{text1}' vs '{text2}': {sim_12:.3f}")
    print(f"Similarity '{text1}' vs '{text3}': {sim_13:.3f}")

    # Test find_similar
    candidates = ["vitamin D3", "vitamin C", "metformin", "probiotics", "cholecalciferol"]
    similar = engine.find_similar(text1, candidates, top_k=3)

    print(f"\nTop 3 similar to '{text1}':")
    for text, score in similar:
        print(f"  - {text}: {score:.3f}")

    # Print stats
    stats = engine.get_stats()
    print(f"\nEngine stats: {stats}")
