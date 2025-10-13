"""
Hierarchical Clusterer - Agglomerative clustering with distance-based merging

Uses scikit-learn's AgglomerativeClustering for bottom-up hierarchical clustering.
Alternative to HDBSCAN when density-based approach doesn't work well.

Key features:
- Deterministic results (same input = same clusters)
- Control over number of clusters OR distance threshold
- Multiple linkage methods (ward, complete, average, single)
- No noise points (100% assignment by design)
"""

import logging
from typing import Optional, Dict
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .base_clusterer import BaseClusterer

logger = logging.getLogger(__name__)


class HierarchicalClusterer(BaseClusterer):
    """
    Hierarchical (Agglomerative) clustering for entity embeddings.

    Hyperparameters:
    - linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
    - distance_threshold: Distance threshold for stopping (default: None)
    - n_clusters: Number of clusters (default: None, auto-determined by threshold)
    - metric: Distance metric (default: 'euclidean')
    """

    def __init__(
        self,
        linkage: str = 'ward',
        distance_threshold: Optional[float] = None,
        n_clusters: Optional[int] = None,
        metric: str = 'euclidean',
        cache_path: Optional[str] = None
    ):
        """
        Initialize Hierarchical clusterer.

        Args:
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            distance_threshold: Distance threshold for cluster merging (None = use n_clusters)
            n_clusters: Number of clusters (None = use distance_threshold)
            metric: Distance metric ('euclidean', 'cosine', 'manhattan')
            cache_path: Path to cache clustering results

        Note:
            - If distance_threshold is set, n_clusters must be None (and vice versa)
            - Ward linkage requires metric='euclidean'
        """
        if distance_threshold is not None and n_clusters is not None:
            raise ValueError("Only one of distance_threshold or n_clusters can be set")

        if distance_threshold is None and n_clusters is None:
            # Default: use distance threshold
            distance_threshold = 0.5
            logger.info("Neither distance_threshold nor n_clusters set, using distance_threshold=0.5")

        if linkage == 'ward' and metric != 'euclidean':
            logger.warning("Ward linkage requires euclidean metric, overriding")
            metric = 'euclidean'

        hyperparameters = {
            'linkage': linkage,
            'distance_threshold': distance_threshold,
            'n_clusters': n_clusters,
            'metric': metric
        }

        super().__init__('hierarchical', hyperparameters, cache_path)

        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.n_clusters = n_clusters
        self.metric = metric

    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform hierarchical clustering.

        Args:
            embeddings: Input embeddings array of shape (n_samples, n_features)

        Returns:
            np.ndarray: Cluster labels of shape (n_samples,)
                       No -1 labels (100% assignment)
        """
        logger.info(f"Running Hierarchical clustering on {len(embeddings)} samples...")
        logger.debug(f"Hyperparameters: linkage={self.linkage}, "
                    f"distance_threshold={self.distance_threshold}, n_clusters={self.n_clusters}")

        # Handle edge case: single sample
        if len(embeddings) == 1:
            logger.warning("Only 1 sample, returning single cluster")
            return np.array([0])

        clusterer = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
            metric=self.metric,
            compute_full_tree=True  # Needed for distance_threshold
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        num_clusters = len(set(cluster_labels))

        logger.info(f"Hierarchical clustering created {num_clusters} clusters (100% assignment)")

        # Store clusterer for potential future use
        self.clusterer_ = clusterer

        return cluster_labels

    def get_cluster_hierarchy(self) -> Dict:
        """
        Get hierarchical cluster structure.

        Returns:
            Dict with children, distances, and number of leaves
        """
        if not hasattr(self, 'clusterer_'):
            raise ValueError("Must run clustering first before getting hierarchy")

        return {
            'children': self.clusterer_.children_,
            'n_clusters': self.clusterer_.n_clusters_,
            'n_leaves': self.clusterer_.n_leaves_,
            'distances': getattr(self.clusterer_, 'distances_', None)
        }

    def get_dendrogram_data(self, embeddings: np.ndarray) -> Dict:
        """
        Get dendrogram data for visualization.

        Args:
            embeddings: Original embeddings used for clustering

        Returns:
            Dict with linkage matrix suitable for scipy.cluster.hierarchy.dendrogram
        """
        from scipy.cluster.hierarchy import linkage as scipy_linkage

        if not hasattr(self, 'clusterer_'):
            raise ValueError("Must run clustering first")

        # Create linkage matrix from AgglomerativeClustering results
        from scipy.cluster.hierarchy import dendrogram

        # Get linkage matrix
        counts = np.zeros(self.clusterer_.children_.shape[0])
        n_samples = len(embeddings)
        for i, merge in enumerate(self.clusterer_.children_):
            counts[i] = (
                (1 if merge[0] < n_samples else counts[merge[0] - n_samples]) +
                (1 if merge[1] < n_samples else counts[merge[1] - n_samples])
            )

        linkage_matrix = np.column_stack([
            self.clusterer_.children_,
            self.clusterer_.distances_,
            counts
        ]).astype(float)

        return {
            'linkage_matrix': linkage_matrix,
            'dendrogram_data': dendrogram(linkage_matrix, no_plot=True)
        }
