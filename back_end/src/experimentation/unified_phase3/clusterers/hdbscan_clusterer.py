"""
HDBSCAN Clusterer - Density-based clustering with hierarchical structure

Uses HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
for automatic cluster discovery. Suitable for medical entity clustering where cluster
counts are unknown.

Key features:
- Automatic cluster discovery (no need to specify number of clusters)
- Handles noise/outliers (assigns -1 label)
- Hierarchical structure
- Robust to varying density
"""

import logging
from typing import Optional, Dict
import numpy as np

from .base_clusterer import BaseClusterer

logger = logging.getLogger(__name__)


class HDBSCANClusterer(BaseClusterer):
    """
    HDBSCAN clustering for entity embeddings.

    Hyperparameters:
    - min_cluster_size: Minimum number of samples in a cluster (default: 2)
    - min_samples: Conservativeness of clustering (default: 1)
    - cluster_selection_epsilon: Distance threshold for merging (default: 0.0)
    - metric: Distance metric (default: 'euclidean')
    - cluster_selection_method: 'eom' or 'leaf' (default: 'eom')
    """

    def __init__(
        self,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.0,
        metric: str = 'euclidean',
        cluster_selection_method: str = 'eom',
        cache_path: Optional[str] = None
    ):
        """
        Initialize HDBSCAN clusterer.

        Args:
            min_cluster_size: Minimum cluster size (lower = more clusters)
            min_samples: Minimum samples for dense region (lower = more permissive)
            cluster_selection_epsilon: Distance threshold for cluster merging
            metric: Distance metric ('euclidean', 'cosine', 'manhattan')
            cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'
            cache_path: Path to cache clustering results
        """
        hyperparameters = {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'cluster_selection_epsilon': cluster_selection_epsilon,
            'metric': metric,
            'cluster_selection_method': cluster_selection_method
        }

        super().__init__('hdbscan', hyperparameters, cache_path)

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method

        # Check if HDBSCAN is installed
        try:
            import hdbscan
            self.hdbscan = hdbscan
            logger.info("HDBSCAN library loaded successfully")
        except ImportError:
            logger.error("HDBSCAN not installed. Install with: pip install hdbscan")
            raise ImportError("HDBSCAN library not found. Install with: pip install hdbscan")

    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform HDBSCAN clustering.

        Args:
            embeddings: Input embeddings array of shape (n_samples, n_features)

        Returns:
            np.ndarray: Cluster labels of shape (n_samples,)
                       -1 indicates noise/unassigned
        """
        logger.info(f"Running HDBSCAN clustering on {len(embeddings)} samples...")
        logger.debug(f"Hyperparameters: min_cluster_size={self.min_cluster_size}, "
                    f"min_samples={self.min_samples}, metric={self.metric}")

        clusterer = self.hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True  # Enable prediction for new points
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        num_noise = np.sum(cluster_labels == -1)

        logger.info(f"HDBSCAN discovered {num_clusters} clusters, {num_noise} noise points")

        # Store clusterer for potential future use
        self.clusterer_ = clusterer

        return cluster_labels

    def get_cluster_probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get cluster membership probabilities (soft clustering).

        Args:
            embeddings: Input embeddings

        Returns:
            np.ndarray: Cluster probabilities of shape (n_samples,)
        """
        if not hasattr(self, 'clusterer_'):
            raise ValueError("Must run clustering first before getting probabilities")

        return self.clusterer_.probabilities_

    def get_cluster_hierarchy(self) -> Dict:
        """
        Get hierarchical cluster structure.

        Returns:
            Dict with condensed tree and linkage information
        """
        if not hasattr(self, 'clusterer_'):
            raise ValueError("Must run clustering first before getting hierarchy")

        return {
            'condensed_tree': self.clusterer_.condensed_tree_,
            'single_linkage_tree': self.clusterer_.single_linkage_tree_,
            'min_spanning_tree': self.clusterer_.minimum_spanning_tree_
        }
