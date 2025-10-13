"""
Condition Embedder - Semantic embeddings for medical condition names

Uses Ollama API (nomic-embed-text by default) to generate embeddings for condition names.
Supports optional context inclusion (symptoms, related conditions).
"""

import logging
import time
from typing import List, Optional, Tuple
import numpy as np
import requests

from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class ConditionEmbedder(BaseEmbedder):
    """
    Embedder for medical condition names using Ollama API.

    Supports:
    - nomic-embed-text (768-dim, fast)
    - Optional context enhancement (symptoms, ICD codes)
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dimension: int = 768,
        batch_size: int = 32,
        cache_path: Optional[str] = None,
        normalization: str = "l2",
        base_url: str = "http://localhost:11434",
        include_context: bool = False,
        timeout: int = 30
    ):
        """
        Initialize condition embedder.

        Args:
            model: Ollama model name (default: nomic-embed-text)
            dimension: Embedding dimension (default: 768)
            batch_size: Texts per batch (default: 32)
            cache_path: Path to cache file
            normalization: Normalization method (l2, unit_sphere, none)
            base_url: Ollama API URL
            include_context: Include symptom/ICD context in embedding
            timeout: API request timeout in seconds
        """
        super().__init__(model, dimension, batch_size, cache_path, normalization)

        self.base_url = base_url
        self.include_context = include_context
        self.timeout = timeout
        self.api_endpoint = f"{base_url}/api/embeddings"

        logger.info(f"ConditionEmbedder initialized: model={model}, context={include_context}")

    def _generate_embedding_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of condition names via Ollama API.

        Args:
            texts: List of condition names

        Returns:
            np.ndarray: Embeddings array of shape (len(texts), dimension)
        """
        embeddings = []

        for text in texts:
            # Optional: Enhance with context
            if self.include_context:
                text = self._enhance_with_context(text)

            try:
                response = requests.post(
                    self.api_endpoint,
                    json={
                        "model": self.model,
                        "prompt": text
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()

                embedding = response.json().get('embedding', [])

                if len(embedding) != self.dimension:
                    logger.warning(f"Unexpected embedding dimension: {len(embedding)} (expected {self.dimension})")
                    if len(embedding) < self.dimension:
                        embedding = embedding + [0.0] * (self.dimension - len(embedding))
                    else:
                        embedding = embedding[:self.dimension]

                embeddings.append(embedding)

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to embed condition '{text[:50]}...': {e}")
                embeddings.append([0.0] * self.dimension)

            time.sleep(0.01)  # Rate limiting

        embeddings_array = np.array(embeddings, dtype=np.float32)
        return embeddings_array

    def _enhance_with_context(self, condition_name: str) -> str:
        """
        Enhance condition name with medical context.

        Example:
        - Input: "diabetes"
        - Output: "diabetes chronic metabolic disease blood glucose"

        Args:
            condition_name: Original condition name

        Returns:
            str: Enhanced text with context
        """
        enhanced = condition_name

        # Cardiovascular context
        if any(word in condition_name.lower() for word in ['heart', 'cardiac', 'hypertension', 'arrhythmia']):
            enhanced += " cardiovascular heart disease"

        # Neurological context
        elif any(word in condition_name.lower() for word in ['alzheimer', 'parkinson', 'stroke', 'dementia']):
            enhanced += " neurological brain nervous system disorder"

        # Digestive context
        elif any(word in condition_name.lower() for word in ['ibs', 'crohn', 'colitis', 'bowel']):
            enhanced += " gastrointestinal digestive system disorder"

        # Metabolic context
        elif any(word in condition_name.lower() for word in ['diabetes', 'thyroid', 'obesity', 'metabolic']):
            enhanced += " metabolic endocrine disorder"

        return enhanced

    def embed_conditions_from_db(self, db_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load unique condition names from database and generate embeddings.

        Args:
            db_path: Path to intervention_research.db

        Returns:
            Tuple of (embeddings_array, condition_names)
        """
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT health_condition
            FROM interventions
            WHERE health_condition IS NOT NULL
              AND health_condition != ''
            ORDER BY health_condition
        """)

        condition_names = [row[0] for row in cursor.fetchall()]
        conn.close()

        logger.info(f"Loaded {len(condition_names)} unique condition names from database")

        embeddings = self.embed(condition_names)

        return embeddings, condition_names
