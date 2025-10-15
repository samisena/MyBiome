"""
Singleton Handler - Ensures 100% Assignment Guarantee

Takes cluster labels with -1 (unassigned/noise) and creates singleton clusters
for each unassigned sample. This guarantees that every entity gets categorized.

Key features:
- 100% assignment guarantee (no -1 labels after processing)
- Preserves existing cluster structure
- Creates unique singleton clusters for outliers
- Tracks assignment provenance (original cluster vs singleton)
"""

import logging
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SingletonHandler:
    """
    Handles unassigned points by creating singleton clusters.

    Ensures 100% assignment rate - no entity left uncategorized.
    """

    def __init__(self):
        """Initialize singleton handler."""
        self.stats = {
            'total_processed': 0,
            'original_noise_count': 0,
            'singletons_created': 0,
            'original_clusters_preserved': 0
        }

        logger.info("SingletonHandler initialized")

    def process_labels(
        self,
        cluster_labels: np.ndarray,
        entity_names: List[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process cluster labels to ensure 100% assignment.

        Args:
            cluster_labels: Original cluster labels (may contain -1 for noise)
            entity_names: Optional list of entity names (for logging)

        Returns:
            Tuple of (new_labels, metadata)
            - new_labels: np.ndarray with no -1 labels
            - metadata: Dict with assignment information
        """
        self.stats['total_processed'] += len(cluster_labels)

        # Find noise points
        noise_mask = cluster_labels == -1
        num_noise = np.sum(noise_mask)

        self.stats['original_noise_count'] += num_noise

        if num_noise == 0:
            logger.info("No noise points found, 100% assignment already achieved")
            return cluster_labels, {
                'method': 'no_singletons_needed',
                'original_clusters': len(set(cluster_labels)),
                'singletons_created': 0,
                'total_clusters': len(set(cluster_labels)),
                'assignment_rate': 1.0
            }

        logger.info(f"Found {num_noise} noise points, creating singleton clusters...")

        # Copy labels to avoid modifying original
        new_labels = cluster_labels.copy()

        # Find next available cluster ID
        max_cluster_id = int(np.max(cluster_labels)) if len(cluster_labels) > 0 else -1
        next_cluster_id = max_cluster_id + 1

        # Assign singleton cluster IDs to noise points
        noise_indices = np.where(noise_mask)[0]

        for i, idx in enumerate(noise_indices):
            singleton_id = next_cluster_id + i
            new_labels[idx] = singleton_id

            if entity_names and idx < len(entity_names):
                logger.debug(f"Singleton cluster {singleton_id}: '{entity_names[idx]}'")

        self.stats['singletons_created'] += num_noise

        # Count original clusters (excluding noise)
        original_clusters = len(set(cluster_labels[~noise_mask]))
        self.stats['original_clusters_preserved'] += original_clusters

        # Verify 100% assignment
        assert -1 not in new_labels, "Failed to achieve 100% assignment!"

        total_clusters = len(set(new_labels))

        metadata = {
            'method': 'singleton_creation',
            'original_clusters': original_clusters,
            'singletons_created': int(num_noise),
            'total_clusters': total_clusters,
            'assignment_rate': 1.0,
            'singleton_percentage': num_noise / len(cluster_labels),
            'noise_indices': noise_indices.tolist()
        }

        logger.info(f"100% assignment achieved: {original_clusters} natural clusters + "
                   f"{num_noise} singletons = {total_clusters} total clusters")

        return new_labels, metadata

    def identify_singletons(self, cluster_labels: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Identify which clusters are singletons.

        Args:
            cluster_labels: Cluster labels (after singleton processing)
            metadata: Metadata from process_labels()

        Returns:
            np.ndarray: Boolean mask where True = singleton cluster
        """
        if metadata['method'] == 'no_singletons_needed':
            return np.zeros(len(cluster_labels), dtype=bool)

        noise_indices = metadata.get('noise_indices', [])
        singleton_mask = np.zeros(len(cluster_labels), dtype=bool)
        singleton_mask[noise_indices] = True

        return singleton_mask

    def get_stats(self) -> Dict:
        """Get singleton handler statistics."""
        return {
            'total_processed': self.stats['total_processed'],
            'original_noise_count': self.stats['original_noise_count'],
            'singletons_created': self.stats['singletons_created'],
            'original_clusters_preserved': self.stats['original_clusters_preserved'],
            'singleton_rate': (
                self.stats['singletons_created'] / self.stats['total_processed']
                if self.stats['total_processed'] > 0 else 0.0
            )
        }

    @staticmethod
    def merge_singletons_if_similar(
        cluster_labels: np.ndarray,
        embeddings: np.ndarray,
        singleton_mask: np.ndarray,
        similarity_threshold: float = 0.8
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optional: Merge singleton clusters that are very similar.

        This reduces the number of singleton clusters by merging those that
        are semantically close (based on embedding similarity).

        Args:
            cluster_labels: Current cluster labels
            embeddings: Entity embeddings
            singleton_mask: Boolean mask identifying singletons
            similarity_threshold: Cosine similarity threshold for merging

        Returns:
            Tuple of (merged_labels, merge_info)
        """
        from sklearn.metrics.pairwise import cosine_similarity

        singleton_indices = np.where(singleton_mask)[0]

        if len(singleton_indices) < 2:
            logger.info("Less than 2 singletons, no merging needed")
            return cluster_labels, {'merges': 0}

        logger.info(f"Checking {len(singleton_indices)} singletons for potential merging...")

        # Compute pairwise similarities between singletons
        singleton_embeddings = embeddings[singleton_indices]
        similarities = cosine_similarity(singleton_embeddings)

        # Merge singletons above threshold
        merged_labels = cluster_labels.copy()
        merge_count = 0

        for i in range(len(singleton_indices)):
            for j in range(i + 1, len(singleton_indices)):
                if similarities[i, j] >= similarity_threshold:
                    # Merge j into i
                    idx_i = singleton_indices[i]
                    idx_j = singleton_indices[j]

                    label_i = merged_labels[idx_i]
                    label_j = merged_labels[idx_j]

                    # Update all instances of label_j to label_i
                    merged_labels[merged_labels == label_j] = label_i
                    merge_count += 1

                    logger.debug(f"Merged singleton {label_j} into {label_i} (similarity={similarities[i, j]:.3f})")

        merge_info = {
            'merges': merge_count,
            'original_singletons': len(singleton_indices),
            'remaining_singletons': len(set(merged_labels[singleton_indices]))
        }

        logger.info(f"Merged {merge_count} singleton pairs, {merge_info['remaining_singletons']} remain")

        return merged_labels, merge_info
