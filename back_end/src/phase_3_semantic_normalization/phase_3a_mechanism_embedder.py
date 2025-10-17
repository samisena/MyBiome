"""
Mechanism Embedder - Semantic embeddings for intervention mechanisms

Uses Ollama API (mxbai-embed-large by default, 1024-dim) to generate embeddings
for mechanism descriptions (often longer text than interventions/conditions).
"""

import logging
from typing import List, Optional, Tuple
import numpy as np

from .phase_3a_base_embedder import BaseEmbedder
from back_end.src.data.constants import OLLAMA_API_URL

logger = logging.getLogger(__name__)


class MechanismEmbedder(BaseEmbedder):
    """
    Embedder for mechanism descriptions using Ollama API.

    Supports:
    - mxbai-embed-large (1024-dim, current default, better for longer text)
    - nomic-embed-text (768-dim, legacy, good for short text)
    """

    def __init__(
        self,
        model: str = "mxbai-embed-large",
        dimension: int = 1024,
        batch_size: int = 10,  # Smaller batches for potentially longer text
        cache_path: Optional[str] = None,
        normalization: str = "l2",
        base_url: str = OLLAMA_API_URL,
        include_context: bool = False,
        timeout: int = 60,  # Longer timeout for mxbai
        max_mechanism_length: Optional[int] = None
    ):
        """
        Initialize mechanism embedder.

        Args:
            model: Ollama model name (default: mxbai-embed-large)
            dimension: Embedding dimension (default: 1024)
            batch_size: Texts per batch (default: 10 for mechanisms)
            cache_path: Path to cache file
            normalization: Normalization method (l2, unit_sphere, none)
            base_url: Ollama API URL
            include_context: Include intervention/condition context
            timeout: API request timeout in seconds
            max_mechanism_length: Truncate mechanisms longer than this (chars)
        """
        super().__init__(model, dimension, batch_size, cache_path, normalization)

        self.base_url = base_url
        self.include_context = include_context
        self.timeout = timeout
        self.max_mechanism_length = max_mechanism_length
        self.api_endpoint = f"{base_url}/api/embeddings"

        logger.info(f"MechanismEmbedder initialized: model={model}, dim={dimension}, max_len={max_mechanism_length}")

    def _generate_embedding_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of mechanism descriptions via Ollama API.

        Args:
            texts: List of mechanism descriptions

        Returns:
            np.ndarray: Embeddings array of shape (len(texts), dimension)
        """
        embeddings = []

        # Determine rate limit delay based on model
        rate_limit_delay = 0.1 if self.model == "mxbai-embed-large" else 0.01

        for text in texts:
            # Optional: Truncate long mechanisms
            if self.max_mechanism_length and len(text) > self.max_mechanism_length:
                logger.debug(f"Truncating mechanism from {len(text)} to {self.max_mechanism_length} chars")
                text = text[:self.max_mechanism_length]

            # Optional: Enhance with context
            if self.include_context:
                text = self._enhance_with_context(text)

            # Use centralized Ollama API caller from base class
            embedding = self._call_ollama_api(
                text=text,
                model=self.model,
                api_endpoint=self.api_endpoint,
                timeout=self.timeout,
                dimension=self.dimension,
                rate_limit_delay=rate_limit_delay
            )
            embeddings.append(embedding)

        return np.vstack(embeddings)

    def _enhance_with_context(self, mechanism_text: str) -> str:
        """
        Enhance mechanism description with additional context.

        Example:
        - Input: "reduces inflammation"
        - Output: "reduces inflammation anti-inflammatory biological mechanism"

        Args:
            mechanism_text: Original mechanism description

        Returns:
            str: Enhanced text with context
        """
        enhanced = mechanism_text

        # Add general biological context
        if len(mechanism_text) < 100:  # Only for short mechanisms
            enhanced += " biological mechanism pathway"

        return enhanced

    def embed_mechanisms_from_db(self, db_path: str, limit: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load unique mechanism descriptions from database and generate embeddings.

        Args:
            db_path: Path to intervention_research.db
            limit: Maximum number of mechanisms to load (for testing)

        Returns:
            Tuple of (embeddings_array, mechanism_texts)
        """
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT mechanism
            FROM interventions
            WHERE mechanism IS NOT NULL
              AND mechanism != ''
              AND mechanism != 'N/A'
            ORDER BY mechanism
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)

        mechanism_texts = [row[0] for row in cursor.fetchall()]
        conn.close()

        logger.info(f"Loaded {len(mechanism_texts)} unique mechanism descriptions from database")

        # Log mechanism length statistics
        lengths = [len(m) for m in mechanism_texts]
        logger.info(f"Mechanism length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

        embeddings = self.embed(mechanism_texts)

        return embeddings, mechanism_texts
