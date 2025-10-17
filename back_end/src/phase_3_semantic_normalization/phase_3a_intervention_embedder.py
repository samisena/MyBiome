"""
Intervention Embedder - Semantic embeddings for intervention names

Uses Ollama API (mxbai-embed-large by default, 1024-dim) to generate embeddings for intervention names.
Supports optional context inclusion (dosage, duration, etc.).
"""

import logging
from typing import List, Optional
import numpy as np

from .phase_3a_base_embedder import BaseEmbedder
from back_end.src.data.constants import OLLAMA_API_URL

logger = logging.getLogger(__name__)


class InterventionEmbedder(BaseEmbedder):
    """
    Embedder for intervention names using Ollama API.

    Supports:
    - mxbai-embed-large (1024-dim, current default, better semantic separation)
    - nomic-embed-text (768-dim, legacy)
    - Optional context enhancement (dosage, duration)
    """

    def __init__(
        self,
        model: str = "mxbai-embed-large",
        dimension: int = 1024,
        batch_size: int = 32,
        cache_path: Optional[str] = None,
        normalization: str = "l2",
        base_url: str = OLLAMA_API_URL,
        include_context: bool = False,
        timeout: int = 30
    ):
        """
        Initialize intervention embedder.

        Args:
            model: Ollama model name (default: mxbai-embed-large)
            dimension: Embedding dimension (default: 1024)
            batch_size: Texts per batch (default: 32)
            cache_path: Path to cache file
            normalization: Normalization method (l2, unit_sphere, none)
            base_url: Ollama API URL
            include_context: Include dosage/duration context in embedding
            timeout: API request timeout in seconds
        """
        super().__init__(model, dimension, batch_size, cache_path, normalization)

        self.base_url = base_url
        self.include_context = include_context
        self.timeout = timeout
        self.api_endpoint = f"{base_url}/api/embeddings"

        logger.info(f"InterventionEmbedder initialized: model={model}, context={include_context}")

    def _generate_embedding_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of intervention names via Ollama API.

        Args:
            texts: List of intervention names

        Returns:
            np.ndarray: Embeddings array of shape (len(texts), dimension)
        """
        embeddings = []

        for text in texts:
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
                rate_limit_delay=0.01
            )
            embeddings.append(embedding)

        return np.vstack(embeddings)

    def _enhance_with_context(self, intervention_name: str) -> str:
        """
        Enhance intervention name with context (dosage, duration, etc.).

        Example:
        - Input: "vitamin D"
        - Output: "vitamin D supplement oral dosage"

        Args:
            intervention_name: Original intervention name

        Returns:
            str: Enhanced text with context
        """
        # Simple heuristic: add common context based on name patterns
        enhanced = intervention_name

        # Supplement context
        if any(word in intervention_name.lower() for word in ['vitamin', 'mineral', 'probiotic', 'omega']):
            enhanced += " supplement oral dosage"

        # Medication context
        elif any(word in intervention_name.lower() for word in ['mg', 'tablet', 'pill', 'drug']):
            enhanced += " medication prescribed treatment"

        # Exercise context
        elif any(word in intervention_name.lower() for word in ['exercise', 'training', 'yoga', 'walking']):
            enhanced += " exercise physical activity routine"

        # Diet context
        elif any(word in intervention_name.lower() for word in ['diet', 'fasting', 'nutrition']):
            enhanced += " dietary intervention nutrition plan"

        return enhanced

    def embed_interventions_from_db(self, db_path: str) -> np.ndarray:
        """
        Load unique intervention names from database and generate embeddings.

        Args:
            db_path: Path to intervention_research.db

        Returns:
            np.ndarray: Embeddings array
        """
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT intervention_name
            FROM interventions
            WHERE intervention_name IS NOT NULL
              AND intervention_name != ''
            ORDER BY intervention_name
        """)

        intervention_names = [row[0] for row in cursor.fetchall()]
        conn.close()

        logger.info(f"Loaded {len(intervention_names)} unique intervention names from database")

        embeddings = self.embed(intervention_names)

        return embeddings, intervention_names
