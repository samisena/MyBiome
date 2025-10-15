"""
Validation Metrics for Phase 3d Hierarchical Clustering

Quality scoring functions for evaluating hierarchy configurations.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """Represents a cluster with members and metadata."""
    cluster_id: int
    canonical_name: str
    members: List[str]
    parent_id: Optional[int] = None
    hierarchy_level: int = 0


@dataclass
class HierarchyMetrics:
    """Metrics for evaluating hierarchy quality."""
    reduction_score: float  # 0-25 points
    depth_score: float  # 0-15 points
    size_distribution_score: float  # 0-20 points
    coherence_score: float  # 0-25 points
    separation_score: float  # 0-15 points
    composite_score: float  # 0-100 points

    # Raw values
    initial_count: int
    final_count: int
    reduction_ratio: float
    max_depth: int
    singleton_ratio: float
    median_cluster_size: float
    avg_coherence: float
    avg_separation: float


def compute_reduction_score(
    initial_count: int,
    final_count: int,
    target_range: tuple = (0.40, 0.60)
) -> tuple:
    """
    Score based on cluster count reduction.

    Args:
        initial_count: Number of clusters before merging
        final_count: Number of clusters after merging
        target_range: Desired reduction ratio range (min, max)

    Returns:
        Tuple of (score, reduction_ratio)
    """
    reduction_ratio = 1 - (final_count / initial_count)

    min_target, max_target = target_range

    if min_target <= reduction_ratio <= max_target:
        score = 25.0
    elif (min_target - 0.10) <= reduction_ratio <= (max_target + 0.10):
        score = 20.0
    elif (min_target - 0.20) <= reduction_ratio <= (max_target + 0.20):
        score = 15.0
    else:
        score = 10.0

    return score, reduction_ratio


def compute_depth_score(
    hierarchy: Dict[str, List[Cluster]],
    target_range: tuple = (2, 3)
) -> tuple:
    """
    Score based on hierarchy depth.

    Args:
        hierarchy: Dict mapping level names to cluster lists
        target_range: Desired depth range (min, max)

    Returns:
        Tuple of (score, max_depth)
    """
    # Count how many levels exist
    max_depth = 0
    for level_name in hierarchy.keys():
        if level_name.startswith('level_'):
            level_num = int(level_name.split('_')[1])
            max_depth = max(max_depth, level_num)

    # Actual depth is the difference from root to deepest leaf
    actual_depth = max_depth

    min_target, max_target = target_range

    if min_target <= actual_depth <= max_target:
        score = 15.0
    elif actual_depth == min_target - 1 or actual_depth == max_target + 1:
        score = 12.0
    elif actual_depth == 1 or actual_depth == 4:
        score = 10.0
    else:
        score = 5.0

    return score, actual_depth


def compute_size_distribution_score(
    top_level_clusters: List[Cluster],
    singleton_threshold: float = 0.50
) -> tuple:
    """
    Score based on cluster size distribution.

    Args:
        top_level_clusters: List of top-level (root) clusters
        singleton_threshold: Max acceptable singleton ratio

    Returns:
        Tuple of (score, singleton_ratio, median_size, max_size)
    """
    if not top_level_clusters:
        return 0.0, 0.0, 0.0, 0

    cluster_sizes = [len(c.members) for c in top_level_clusters]

    singleton_ratio = sum(1 for s in cluster_sizes if s == 1) / len(cluster_sizes)
    median_size = float(np.median(cluster_sizes))
    max_size = max(cluster_sizes)

    # Start with max score
    score = 20.0

    # Penalize high singleton ratio
    if singleton_ratio > singleton_threshold:
        score -= 5.0
    if singleton_ratio > 0.60:
        score -= 3.0

    # Penalize poor median size
    if median_size < 2 or median_size > 10:
        score -= 5.0

    # Penalize mega-clusters
    if max_size > 100:
        score -= 5.0
    elif max_size > 50:
        score -= 2.0

    return max(0.0, score), singleton_ratio, median_size, max_size


def compute_coherence_score(
    top_level_clusters: List[Cluster],
    embeddings: Dict[str, np.ndarray],
    target_min: float = 0.65
) -> tuple:
    """
    Score based on intra-cluster coherence (avg similarity within clusters).

    Args:
        top_level_clusters: List of clusters to evaluate
        embeddings: Dict mapping member IDs to embedding vectors
        target_min: Minimum acceptable coherence

    Returns:
        Tuple of (score, avg_coherence)
    """
    coherence_scores = []

    for cluster in top_level_clusters:
        if len(cluster.members) < 2:
            continue

        # Get embeddings for cluster members
        member_embeddings = []
        for member in cluster.members:
            if member in embeddings:
                member_embeddings.append(embeddings[member])

        if len(member_embeddings) < 2:
            continue

        # Compute pairwise similarities
        pairwise_sims = []
        for i in range(len(member_embeddings)):
            for j in range(i+1, len(member_embeddings)):
                sim = cosine_similarity(member_embeddings[i], member_embeddings[j])
                pairwise_sims.append(sim)

        if pairwise_sims:
            coherence_scores.append(np.mean(pairwise_sims))

    avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 0.5

    # Score based on coherence level
    if avg_coherence >= 0.70:
        score = 25.0
    elif avg_coherence >= 0.65:
        score = 22.0
    elif avg_coherence >= 0.60:
        score = 20.0
    elif avg_coherence >= 0.50:
        score = 15.0
    else:
        score = 10.0

    return score, avg_coherence


def compute_separation_score(
    top_level_clusters: List[Cluster],
    embeddings: Dict[str, np.ndarray],
    target_min: float = 0.35
) -> tuple:
    """
    Score based on inter-cluster separation (distinctness).

    Args:
        top_level_clusters: List of clusters to evaluate
        embeddings: Dict mapping member IDs to embedding vectors
        target_min: Minimum acceptable separation

    Returns:
        Tuple of (score, avg_separation)
    """
    if len(top_level_clusters) < 2:
        return 15.0, 1.0  # Perfect separation if only 1 cluster

    # Compute centroids for each cluster
    centroids = []
    for cluster in top_level_clusters:
        member_embeddings = [embeddings[m] for m in cluster.members if m in embeddings]
        if member_embeddings:
            centroid = np.mean(member_embeddings, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            centroids.append(centroid)

    if len(centroids) < 2:
        return 15.0, 1.0

    # Compute inter-cluster similarities
    inter_cluster_sims = []
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            sim = cosine_similarity(centroids[i], centroids[j])
            inter_cluster_sims.append(sim)

    avg_similarity = float(np.mean(inter_cluster_sims))
    avg_separation = 1.0 - avg_similarity

    # Score based on separation level
    if avg_separation >= 0.40:
        score = 15.0
    elif avg_separation >= 0.35:
        score = 13.0
    elif avg_separation >= 0.30:
        score = 12.0
    else:
        score = 8.0

    return score, avg_separation


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Normalize
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-9)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-9)

    # Dot product
    similarity = np.dot(vec1_norm, vec2_norm)

    # Clip to [0, 1]
    return float(np.clip(similarity, 0.0, 1.0))


def evaluate_hierarchy_quality(
    hierarchy: Dict[str, List[Cluster]],
    embeddings: Dict[str, np.ndarray],
    config: Dict[str, Any]
) -> HierarchyMetrics:
    """
    Comprehensive quality evaluation of hierarchy.

    Args:
        hierarchy: Dict mapping level names to cluster lists
        embeddings: Dict mapping member IDs to embeddings
        config: Configuration dict with target values

    Returns:
        HierarchyMetrics object with all scores
    """
    # Get initial and final counts
    initial_count = len(hierarchy.get('level_3', []))
    final_count = len(hierarchy.get('level_0', []))

    # Compute individual scores
    reduction_score, reduction_ratio = compute_reduction_score(
        initial_count,
        final_count,
        target_range=config.get('target_reduction_range', (0.40, 0.60))
    )

    depth_score, max_depth = compute_depth_score(
        hierarchy,
        target_range=config.get('target_depth_range', (2, 3))
    )

    top_level_clusters = hierarchy.get('level_0', [])
    size_score, singleton_ratio, median_size, max_size = compute_size_distribution_score(
        top_level_clusters,
        singleton_threshold=config.get('target_singleton_threshold', 0.50)
    )

    coherence_score, avg_coherence = compute_coherence_score(
        top_level_clusters,
        embeddings,
        target_min=config.get('target_coherence_min', 0.65)
    )

    separation_score, avg_separation = compute_separation_score(
        top_level_clusters,
        embeddings,
        target_min=config.get('target_separation_min', 0.35)
    )

    # Composite score (sum of all scores)
    composite_score = (
        reduction_score +
        depth_score +
        size_score +
        coherence_score +
        separation_score
    )

    metrics = HierarchyMetrics(
        reduction_score=reduction_score,
        depth_score=depth_score,
        size_distribution_score=size_score,
        coherence_score=coherence_score,
        separation_score=separation_score,
        composite_score=composite_score,
        initial_count=initial_count,
        final_count=final_count,
        reduction_ratio=reduction_ratio,
        max_depth=max_depth,
        singleton_ratio=singleton_ratio,
        median_cluster_size=median_size,
        avg_coherence=avg_coherence,
        avg_separation=avg_separation
    )

    return metrics


def print_metrics_summary(metrics: HierarchyMetrics):
    """Print human-readable summary of metrics."""
    print("\n" + "="*60)
    print("HIERARCHY QUALITY METRICS")
    print("="*60)

    print(f"\n{'Metric':<30s} {'Score':<10s} {'Value':<20s}")
    print("-"*60)

    print(f"{'Cluster Reduction':<30s} {metrics.reduction_score:>6.1f}/25 {metrics.reduction_ratio:>18.1%}")
    print(f"{'Hierarchy Depth':<30s} {metrics.depth_score:>6.1f}/15 {metrics.max_depth:>18d} levels")
    print(f"{'Size Distribution':<30s} {metrics.size_distribution_score:>6.1f}/20 {'singleton='+str(round(metrics.singleton_ratio*100, 1))+'%':>18s}")
    print(f"{'Intra-Cluster Coherence':<30s} {metrics.coherence_score:>6.1f}/25 {metrics.avg_coherence:>18.2f}")
    print(f"{'Inter-Cluster Separation':<30s} {metrics.separation_score:>6.1f}/15 {metrics.avg_separation:>18.2f}")

    print("-"*60)
    print(f"{'COMPOSITE SCORE':<30s} {metrics.composite_score:>6.1f}/100")
    print("="*60)

    print(f"\nCluster Count: {metrics.initial_count} → {metrics.final_count} ({metrics.reduction_ratio:.1%} reduction)")
    print(f"Median Cluster Size: {metrics.median_cluster_size:.1f}")
