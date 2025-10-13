"""
Hierarchical Sub-Clustering for Mechanism Clusters - Phase 3.2

Implements 2-level hierarchical clustering for large/heterogeneous clusters:
- Identifies clusters requiring sub-clustering (> 30 members OR low coherence)
- Recursively applies HDBSCAN within clusters
- Creates parent-child relationships in database
- Validates hierarchy coherence

Architecture:
- Level 0: Root (primary mechanisms)
- Level 1: Children (sub-mechanisms)
- Level 2+: Reserved for future expansion

Usage:
    from mechanism_hierarchical_clustering import HierarchicalClusterer

    clusterer = HierarchicalClusterer()
    hierarchy = clusterer.create_hierarchy(primary_clusters, embeddings_dict)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Import HDBSCAN
try:
    import hdbscan
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("hdbscan or sklearn not available")

# Import canonical extractor
from .mechanism_canonical_extractor import MechanismCanonicalExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HierarchyNode:
    """Represents a node in the mechanism hierarchy."""
    cluster_id: int
    canonical_name: str
    members: List[str]
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    hierarchy_level: int = 0
    silhouette_score: Optional[float] = None


@dataclass
class HierarchyResult:
    """Result of hierarchical clustering."""
    nodes: Dict[int, HierarchyNode]  # cluster_id -> HierarchyNode
    parent_child_map: Dict[int, List[int]]  # parent_id -> [child_ids]
    total_levels: int
    subclustered_count: int
    avg_children_per_parent: float


class HierarchicalClusterer:
    """
    Creates 2-level hierarchical clustering for large/heterogeneous clusters.

    Triggers:
    1. Large clusters (> member_threshold)
    2. Low coherence clusters (silhouette < coherence_threshold)
    """

    def __init__(
        self,
        member_threshold: int = 30,
        coherence_threshold: float = 0.3,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        canonical_extractor: Optional[MechanismCanonicalExtractor] = None
    ):
        """
        Initialize hierarchical clusterer.

        Args:
            member_threshold: Trigger sub-clustering if members > threshold
            coherence_threshold: Trigger sub-clustering if silhouette < threshold
            min_cluster_size: HDBSCAN min_cluster_size for sub-clustering
            min_samples: HDBSCAN min_samples for sub-clustering
            canonical_extractor: Optional canonical name extractor
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("hdbscan and scikit-learn required")

        self.member_threshold = member_threshold
        self.coherence_threshold = coherence_threshold
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

        if canonical_extractor is None:
            self.canonical_extractor = MechanismCanonicalExtractor()
        else:
            self.canonical_extractor = canonical_extractor

        logger.info("HierarchicalClusterer initialized")
        logger.info(f"  Thresholds: members>{member_threshold}, coherence<{coherence_threshold}")

    def should_subcluster(
        self,
        cluster_members: List[str],
        cluster_embeddings: np.ndarray,
        cluster_labels: Optional[np.ndarray] = None
    ) -> Tuple[bool, List[str]]:
        """
        Determine if cluster should be sub-clustered.

        Args:
            cluster_members: List of mechanism texts in cluster
            cluster_embeddings: Embeddings for cluster members
            cluster_labels: Optional cluster labels (for silhouette computation)

        Returns:
            Tuple of (should_subcluster, reasons)
        """
        reasons = []

        # Check member count
        if len(cluster_members) > self.member_threshold:
            reasons.append(f"large_size ({len(cluster_members)} > {self.member_threshold})")

        # Check coherence (silhouette score)
        if cluster_labels is not None and len(set(cluster_labels)) > 1:
            try:
                silhouette = silhouette_score(cluster_embeddings, cluster_labels)
                if silhouette < self.coherence_threshold:
                    reasons.append(f"low_coherence (sil={silhouette:.2f} < {self.coherence_threshold})")
            except Exception as e:
                logger.warning(f"Failed to compute silhouette: {e}")

        should_trigger = len(reasons) > 0

        return should_trigger, reasons

    def subcluster(
        self,
        cluster_members: List[str],
        cluster_embeddings: np.ndarray,
        parent_id: int
    ) -> Optional[Dict[int, List[str]]]:
        """
        Sub-cluster a large/heterogeneous cluster.

        Args:
            cluster_members: List of mechanism texts in cluster
            cluster_embeddings: Embeddings for cluster members
            parent_id: Parent cluster ID

        Returns:
            Dict mapping sub_cluster_id to list of members (or None if failed)
        """
        logger.info(f"Sub-clustering parent cluster {parent_id} ({len(cluster_members)} members)...")

        try:
            # Run HDBSCAN on cluster members
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=0.0,
                metric='euclidean'
            )

            labels = clusterer.fit_predict(cluster_embeddings)

            # Build sub-clusters
            sub_clusters = defaultdict(list)
            for member, label in zip(cluster_members, labels):
                sub_clusters[int(label)].append(member)

            # Count valid sub-clusters (excluding noise)
            valid_subclusters = {k: v for k, v in sub_clusters.items() if k != -1}

            if not valid_subclusters:
                logger.warning(f"  No valid sub-clusters found for parent {parent_id}")
                return None

            logger.info(f"  Created {len(valid_subclusters)} sub-clusters")

            return dict(valid_subclusters)

        except Exception as e:
            logger.error(f"Failed to sub-cluster parent {parent_id}: {e}")
            return None

    def create_hierarchy(
        self,
        primary_clusters: Dict[int, List[str]],
        embeddings_dict: Dict[str, np.ndarray],
        next_cluster_id: int = 1000
    ) -> HierarchyResult:
        """
        Create 2-level hierarchical clustering.

        Args:
            primary_clusters: Dict mapping cluster_id to list of members (Level 0)
            embeddings_dict: Dict mapping mechanism_text to embedding vector
            next_cluster_id: Starting ID for child clusters (to avoid conflicts)

        Returns:
            HierarchyResult with hierarchy information
        """
        logger.info(f"\nCreating 2-level hierarchy for {len(primary_clusters)} primary clusters...")

        nodes: Dict[int, HierarchyNode] = {}
        parent_child_map: Dict[int, List[int]] = {}
        subclustered_count = 0
        current_cluster_id = next_cluster_id

        # Step 1: Create Level 0 nodes (parents)
        logger.info("\nStep 1: Creating Level 0 (parent) nodes...")

        for cluster_id, members in primary_clusters.items():
            if cluster_id == -1:
                # Skip singleton cluster
                continue

            # Get embeddings for cluster members
            cluster_embeddings = np.array([embeddings_dict[m] for m in members if m in embeddings_dict])

            # Extract canonical name
            canonical_result = self.canonical_extractor.extract_canonical(members, cluster_id)

            # Create node
            node = HierarchyNode(
                cluster_id=cluster_id,
                canonical_name=canonical_result.canonical_name,
                members=members,
                parent_id=None,
                children_ids=[],
                hierarchy_level=0,
                silhouette_score=None
            )

            nodes[cluster_id] = node

        logger.info(f"  Created {len(nodes)} Level 0 nodes")

        # Step 2: Identify clusters for sub-clustering
        logger.info("\nStep 2: Identifying clusters for sub-clustering...")

        for cluster_id, node in list(nodes.items()):
            if len(node.members) == 0:
                continue

            # Get embeddings
            cluster_embeddings = np.array([embeddings_dict[m] for m in node.members if m in embeddings_dict])

            if len(cluster_embeddings) == 0:
                continue

            # Check if should sub-cluster
            should_trigger, reasons = self.should_subcluster(node.members, cluster_embeddings)

            if not should_trigger:
                continue

            logger.info(f"  Cluster {cluster_id} ({len(node.members)} members): {', '.join(reasons)}")

            # Sub-cluster
            sub_clusters = self.subcluster(node.members, cluster_embeddings, cluster_id)

            if sub_clusters is None or len(sub_clusters) < 2:
                # Sub-clustering failed or no meaningful split
                logger.info(f"    No meaningful sub-clusters created, keeping as flat cluster")
                continue

            # Step 3: Create child nodes
            child_ids = []

            for local_sub_id, sub_members in sub_clusters.items():
                # Assign global cluster ID
                child_cluster_id = current_cluster_id
                current_cluster_id += 1

                # Extract canonical name for child
                child_canonical_result = self.canonical_extractor.extract_canonical(
                    sub_members,
                    child_cluster_id
                )

                # Create child node
                child_node = HierarchyNode(
                    cluster_id=child_cluster_id,
                    canonical_name=child_canonical_result.canonical_name,
                    members=sub_members,
                    parent_id=cluster_id,
                    children_ids=[],
                    hierarchy_level=1
                )

                nodes[child_cluster_id] = child_node
                child_ids.append(child_cluster_id)

                logger.info(f"    Created child cluster {child_cluster_id}: {child_canonical_result.canonical_name} ({len(sub_members)} members)")

            # Update parent node
            node.children_ids = child_ids
            parent_child_map[cluster_id] = child_ids
            subclustered_count += 1

        logger.info(f"\nHierarchy creation complete:")
        logger.info(f"  Total nodes: {len(nodes)}")
        logger.info(f"  Level 0 (parents): {sum(1 for n in nodes.values() if n.hierarchy_level == 0)}")
        logger.info(f"  Level 1 (children): {sum(1 for n in nodes.values() if n.hierarchy_level == 1)}")
        logger.info(f"  Parents with children: {subclustered_count}")

        # Compute statistics
        if parent_child_map:
            avg_children = sum(len(children) for children in parent_child_map.values()) / len(parent_child_map)
        else:
            avg_children = 0.0

        return HierarchyResult(
            nodes=nodes,
            parent_child_map=parent_child_map,
            total_levels=2,
            subclustered_count=subclustered_count,
            avg_children_per_parent=avg_children
        )

    def validate_hierarchy(
        self,
        hierarchy_result: HierarchyResult,
        sample_size: int = 5
    ) -> Dict[str, Any]:
        """
        Validate hierarchy coherence.

        Args:
            hierarchy_result: HierarchyResult to validate
            sample_size: Number of parent-child pairs to sample

        Returns:
            Validation metrics
        """
        logger.info("\nValidating hierarchy coherence...")

        # Sample parent-child pairs
        parent_child_pairs = list(hierarchy_result.parent_child_map.items())
        if not parent_child_pairs:
            logger.warning("No parent-child relationships to validate")
            return {
                'coherence_score': 0.0,
                'sample_count': 0
            }

        import random
        sample_pairs = random.sample(parent_child_pairs, min(sample_size, len(parent_child_pairs)))

        coherence_scores = []

        for parent_id, child_ids in sample_pairs:
            parent_node = hierarchy_result.nodes[parent_id]

            logger.info(f"\nParent {parent_id}: {parent_node.canonical_name} ({len(parent_node.members)} members)")
            logger.info(f"  Children ({len(child_ids)}):")

            for child_id in child_ids:
                child_node = hierarchy_result.nodes[child_id]
                logger.info(f"    - Cluster {child_id}: {child_node.canonical_name} ({len(child_node.members)} members)")

            # Heuristic coherence scoring
            # Good hierarchy: children capture distinct subsets, parent captures all
            total_children_members = sum(len(hierarchy_result.nodes[cid].members) for cid in child_ids)
            parent_coverage = total_children_members / len(parent_node.members)

            # Expected: close to 1.0 (all parent members in children)
            coherence = min(parent_coverage, 1.0)
            coherence_scores.append(coherence)

            logger.info(f"  Coherence: {coherence:.2f} (coverage: {parent_coverage:.2f})")

        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

        logger.info(f"\nAverage coherence: {avg_coherence:.2f}")

        return {
            'coherence_score': avg_coherence,
            'sample_count': len(sample_pairs),
            'min_coherence': min(coherence_scores) if coherence_scores else 0.0,
            'max_coherence': max(coherence_scores) if coherence_scores else 0.0
        }


def main():
    """Command-line interface."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Hierarchical Mechanism Clustering (Phase 3.2)")
    parser.add_argument('--clusters-file', required=True, help='Path to primary clusters JSON')
    parser.add_argument('--embeddings-file', required=True, help='Path to embeddings JSON')
    parser.add_argument('--action', choices=['create', 'validate'], default='create')
    parser.add_argument('--member-threshold', type=int, default=30, help='Member threshold for sub-clustering')
    parser.add_argument('--sample-size', type=int, default=5, help='Sample size for validation')

    args = parser.parse_args()

    # Load primary clusters
    with open(args.clusters_file, 'r') as f:
        primary_clusters = json.load(f)

    # Convert string keys to int
    primary_clusters = {int(k): v for k, v in primary_clusters.items()}

    # Load embeddings
    with open(args.embeddings_file, 'r') as f:
        embeddings_data = json.load(f)

    # Convert to numpy arrays
    embeddings_dict = {k: np.array(v) for k, v in embeddings_data.items()}

    # Create clusterer
    clusterer = HierarchicalClusterer(member_threshold=args.member_threshold)

    if args.action == 'create':
        # Create hierarchy
        hierarchy = clusterer.create_hierarchy(primary_clusters, embeddings_dict)

        # Print summary
        print(f"\nHierarchy Summary:")
        print(f"  Total nodes: {len(hierarchy.nodes)}")
        print(f"  Parents with children: {hierarchy.subclustered_count}")
        print(f"  Avg children per parent: {hierarchy.avg_children_per_parent:.1f}")

        # Validate
        validation = clusterer.validate_hierarchy(hierarchy, sample_size=args.sample_size)
        print(f"\nValidation:")
        print(f"  Coherence score: {validation['coherence_score']:.2f}")

    elif args.action == 'validate':
        # Create and validate
        hierarchy = clusterer.create_hierarchy(primary_clusters, embeddings_dict)
        validation = clusterer.validate_hierarchy(hierarchy, sample_size=args.sample_size)

        print(f"\nValidation Results:")
        print(f"  Coherence score: {validation['coherence_score']:.2f}")
        print(f"  Sample count: {validation['sample_count']}")


if __name__ == "__main__":
    main()
