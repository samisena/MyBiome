"""
Base Embedder Abstract Class

Defines the interface for all embedding engines in the unified Phase 3 pipeline.
Supports caching, batch processing, and performance tracking.
"""

import json
import pickle
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import requests

from back_end.src.data.constants import OLLAMA_API_URL

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding engines.

    Subclasses must implement:
    - _generate_embedding_batch(texts: List[str]) -> np.ndarray
    """

    def __init__(
        self,
        model: str,
        dimension: int,
        batch_size: int = 32,
        cache_path: Optional[str] = None,
        normalization: str = "l2"
    ):
        """
        Initialize the base embedder.

        Args:
            model: Model name (e.g., 'nomic-embed-text', 'mxbai-embed-large')
            dimension: Embedding dimension (768, 1024, etc.)
            batch_size: Number of texts to process in parallel
            cache_path: Path to cache file (JSON or pickle)
            normalization: Normalization method ('l2', 'unit_sphere', 'none')
        """
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.cache_path = Path(cache_path) if cache_path else None
        self.normalization = normalization

        # Cache: {text_hash: embedding_array}
        self.cache: Dict[str, np.ndarray] = {}
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_embeddings_generated': 0,
            'total_texts_embedded': 0,
            'cache_saves': 0
        }

        logger.info(f"Initialized {self.__class__.__name__}: model={model}, dim={dimension}, batch_size={batch_size}")

    @abstractmethod
    def _generate_embedding_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        This method must be implemented by subclasses.

        Args:
            texts: List of input texts

        Returns:
            np.ndarray: Embeddings array of shape (len(texts), dimension)
        """
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts (with caching).

        Args:
            texts: List of input texts

        Returns:
            np.ndarray: Embeddings array of shape (len(texts), dimension)
        """
        embeddings = []
        texts_to_generate = []
        text_indices = []

        # Check cache
        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            if text_hash in self.cache:
                embeddings.append((i, self.cache[text_hash]))
                self.stats['cache_hits'] += 1
            else:
                texts_to_generate.append(text)
                text_indices.append(i)
                self.stats['cache_misses'] += 1

        # Generate missing embeddings
        if texts_to_generate:
            logger.debug(f"Generating {len(texts_to_generate)} new embeddings (cache misses)")
            new_embeddings = self._generate_embeddings_with_batching(texts_to_generate)

            # Cache new embeddings
            for text, embedding in zip(texts_to_generate, new_embeddings):
                text_hash = self._hash_text(text)
                self.cache[text_hash] = embedding

            # Add to results
            for idx, embedding in zip(text_indices, new_embeddings):
                embeddings.append((idx, embedding))

            self.stats['total_embeddings_generated'] += len(texts_to_generate)

            # Save cache periodically
            if len(texts_to_generate) >= 50:
                self.save_cache()

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        embeddings_array = np.array([emb for _, emb in embeddings], dtype=np.float32)

        self.stats['total_texts_embedded'] += len(texts)

        return embeddings_array

    def _generate_embeddings_with_batching(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings with batching support.

        Args:
            texts: List of texts to embed

        Returns:
            np.ndarray: Embeddings array
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._generate_embedding_batch(batch)

            # Normalize if requested
            if self.normalization == "l2":
                batch_embeddings = self._normalize_l2(batch_embeddings)
            elif self.normalization == "unit_sphere":
                batch_embeddings = self._normalize_unit_sphere(batch_embeddings)

            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def _normalize_l2(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalization."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-9)

    def _normalize_unit_sphere(self, embeddings: np.ndarray) -> np.ndarray:
        """Project onto unit sphere."""
        return embeddings / np.sqrt(np.sum(embeddings**2, axis=1, keepdims=True))

    def _hash_text(self, text: str) -> str:
        """Generate hash for text (for caching)."""
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _call_ollama_api(
        self,
        text: str,
        model: str,
        api_endpoint: str,
        timeout: int,
        dimension: int,
        rate_limit_delay: float = 0.01
    ) -> np.ndarray:
        """
        Call Ollama API to generate embedding for a single text.

        This method centralizes Ollama API calls to eliminate duplication
        across InterventionEmbedder, ConditionEmbedder, and MechanismEmbedder.

        Args:
            text: Input text to embed
            model: Ollama model name (e.g., 'mxbai-embed-large')
            api_endpoint: Full API endpoint URL
            timeout: Request timeout in seconds
            dimension: Expected embedding dimension
            rate_limit_delay: Delay after request (seconds)

        Returns:
            np.ndarray: Embedding vector of shape (dimension,)
        """
        try:
            response = requests.post(
                api_endpoint,
                json={
                    "model": model,
                    "prompt": text
                },
                timeout=timeout
            )
            response.raise_for_status()

            embedding = response.json().get('embedding', [])

            # Dimension validation and correction
            if len(embedding) != dimension:
                logger.warning(f"Unexpected embedding dimension: {len(embedding)} (expected {dimension})")
                if len(embedding) < dimension:
                    # Pad with zeros
                    embedding = embedding + [0.0] * (dimension - len(embedding))
                else:
                    # Truncate
                    embedding = embedding[:dimension]

            # Rate limiting
            time.sleep(rate_limit_delay)

            return np.array(embedding, dtype=np.float32)

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to embed text '{text[:50]}...': {e}")
            # Fallback: zero vector
            return np.zeros(dimension, dtype=np.float32)

    def _load_cache(self):
        """Load cache from disk."""
        try:
            if self.cache_path.suffix == '.json':
                with open(self.cache_path, 'r') as f:
                    cache_data = json.load(f)
                    self.cache = {
                        k: np.array(v, dtype=np.float32)
                        for k, v in cache_data.items()
                    }
            else:  # pickle
                with open(self.cache_path, 'rb') as f:
                    self.cache = pickle.load(f)

            logger.info(f"Loaded {len(self.cache)} embeddings from cache: {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}

    def save_cache(self):
        """Save cache to disk."""
        if not self.cache_path:
            return

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            if self.cache_path.suffix == '.json':
                cache_data = {k: v.tolist() for k, v in self.cache.items()}
                with open(self.cache_path, 'w') as f:
                    json.dump(cache_data, f)
            else:  # pickle
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.cache, f)

            self.stats['cache_saves'] += 1
            logger.debug(f"Saved {len(self.cache)} embeddings to cache: {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_stats(self) -> Dict:
        """Get embedding statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0.0

        return {
            'model': self.model,
            'dimension': self.dimension,
            'cache_size': len(self.cache),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'total_embeddings_generated': self.stats['total_embeddings_generated'],
            'total_texts_embedded': self.stats['total_texts_embedded'],
            'cache_saves': self.stats['cache_saves']
        }

    def clear_cache(self):
        """Clear in-memory cache."""
        self.cache.clear()
        logger.info("Cache cleared")

    def __del__(self):
        """Cleanup: save cache on destruction."""
        if hasattr(self, 'cache') and self.cache_path:
            self.save_cache()
