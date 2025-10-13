"""
Base Clusterer Abstract Class

Defines the interface for all clustering algorithms in the unified Phase 3 pipeline.
Supports hyperparameter tracking, evaluation metrics, and result caching.
"""

import json
import pickle
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

logger = logging.getLogger(__name__)


class BaseClusterer(ABC):
    """
    Abstract base class for all clustering algorithms.

    Subclasses must implement:
    - _perform_clustering(embeddings: np.ndarray) -> np.ndarray
    """

    def __init__(
        self,
        algorithm_name: str,
        hyperparameters: Dict,
        cache_path: Optional[str] = None
    ):
        """
        Initialize the base clusterer.

        Args:
            algorithm_name: Name of clustering algorithm (e.g., 'hdbscan', 'hierarchical')
            hyperparameters: Dict of algorithm-specific hyperparameters
            cache_path: Path to cache clustering results
        """
        self.algorithm_name = algorithm_name
        self.hyperparameters = hyperparameters
        self.cache_path = Path(cache_path) if cache_path else None

        # Results cache: {embeddings_hash: cluster_labels}
        self.cache: Dict[str, np.ndarray] = {}
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

        # Statistics
        self.stats = {
            'clusterings_performed': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info(f"Initialized {self.__class__.__name__}: algorithm={algorithm_name}")

    @abstractmethod
    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform clustering on embeddings.

        This method must be implemented by subclasses.

        Args:
            embeddings: Input embeddings array of shape (n_samples, n_features)

        Returns:
            np.ndarray: Cluster labels of shape (n_samples,)
                       -1 indicates unassigned (noise)
        """
        pass

    def cluster(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Cluster embeddings with caching support.

        Args:
            embeddings: Input embeddings array

        Returns:
            Tuple of (cluster_labels, metadata)
            - cluster_labels: np.ndarray of shape (n_samples,)
            - metadata: Dict with clustering info (num_clusters, etc.)
        """
        # Check cache
        embeddings_hash = self._hash_embeddings(embeddings)
        if embeddings_hash in self.cache:
            logger.debug("Cache hit for clustering")
            self.stats['cache_hits'] += 1
            cluster_labels = self.cache[embeddings_hash]
        else:
            logger.debug("Cache miss, performing clustering")
            self.stats['cache_misses'] += 1
            cluster_labels = self._perform_clustering(embeddings)

            # Cache results
            self.cache[embeddings_hash] = cluster_labels
            self.stats['clusterings_performed'] += 1

            # Save cache periodically
            if self.stats['clusterings_performed'] % 5 == 0:
                self.save_cache()

        # Compute metadata
        metadata = self._compute_metadata(embeddings, cluster_labels)

        return cluster_labels, metadata

    def _compute_metadata(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Compute clustering metadata and quality metrics.

        Args:
            embeddings: Input embeddings
            labels: Cluster labels

        Returns:
            Dict with clustering metadata
        """
        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_noise = np.sum(labels == -1)
        cluster_sizes = {int(label): int(np.sum(labels == label)) for label in unique_labels if label != -1}

        metadata = {
            'algorithm': self.algorithm_name,
            'hyperparameters': self.hyperparameters,
            'num_samples': len(labels),
            'num_clusters': num_clusters,
            'num_noise': num_noise,
            'cluster_sizes': cluster_sizes,
            'assignment_rate': (len(labels) - num_noise) / len(labels) if len(labels) > 0 else 0.0
        }

        # Compute quality metrics (if enough clusters)
        if num_clusters >= 2 and num_noise < len(labels):
            try:
                # Silhouette score (higher is better, range: -1 to 1)
                if num_noise > 0:
                    # Only compute on assigned points
                    assigned_mask = labels != -1
                    silhouette = silhouette_score(
                        embeddings[assigned_mask],
                        labels[assigned_mask]
                    )
                else:
                    silhouette = silhouette_score(embeddings, labels)

                metadata['silhouette_score'] = float(silhouette)
            except Exception as e:
                logger.warning(f"Failed to compute silhouette score: {e}")
                metadata['silhouette_score'] = None

            try:
                # Davies-Bouldin index (lower is better, range: 0 to inf)
                if num_noise > 0:
                    assigned_mask = labels != -1
                    davies_bouldin = davies_bouldin_score(
                        embeddings[assigned_mask],
                        labels[assigned_mask]
                    )
                else:
                    davies_bouldin = davies_bouldin_score(embeddings, labels)

                metadata['davies_bouldin_score'] = float(davies_bouldin)
            except Exception as e:
                logger.warning(f"Failed to compute Davies-Bouldin score: {e}")
                metadata['davies_bouldin_score'] = None

        # Cluster size statistics
        if cluster_sizes:
            sizes = list(cluster_sizes.values())
            metadata['cluster_size_stats'] = {
                'min': min(sizes),
                'max': max(sizes),
                'mean': np.mean(sizes),
                'median': np.median(sizes)
            }

        return metadata

    def _hash_embeddings(self, embeddings: np.ndarray) -> str:
        """Generate hash for embeddings (for caching)."""
        import hashlib
        return hashlib.sha256(embeddings.tobytes()).hexdigest()

    def _load_cache(self):
        """Load cache from disk."""
        try:
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
            logger.info(f"Loaded {len(self.cache)} cached clustering results from {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}

    def save_cache(self):
        """Save cache to disk."""
        if not self.cache_path:
            return

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.debug(f"Saved {len(self.cache)} clustering results to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_hyperparameters(self) -> Dict:
        """Get clustering hyperparameters."""
        return self.hyperparameters.copy()

    def get_stats(self) -> Dict:
        """Get clustering statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0.0

        return {
            'algorithm': self.algorithm_name,
            'hyperparameters': self.hyperparameters,
            'cache_size': len(self.cache),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'clusterings_performed': self.stats['clusterings_performed']
        }

    def __del__(self):
        """Cleanup: save cache on destruction."""
        if hasattr(self, 'cache') and self.cache_path:
            self.save_cache()
