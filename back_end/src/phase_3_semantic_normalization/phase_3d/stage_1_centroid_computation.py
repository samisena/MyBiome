"""
Stage 1: Centroid Computation

Compute cluster-level embeddings (centroids) from member embeddings.
Centroids are used for similarity calculations in Stage 2.
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from .validation_metrics import Cluster

logger = logging.getLogger(__name__)


@dataclass
class CentroidResult:
    """Result of centroid computation."""
    cluster_id: int
    centroid: np.ndarray
    member_count: int
    missing_embeddings: int
    normalization_applied: bool


def compute_centroids(
    clusters: List[Cluster],
    embeddings: Dict[str, np.ndarray],
    embedding_dim: int = 1024,
    normalize: bool = True
) -> Dict[int, np.ndarray]:
    """
    Compute centroid embeddings for a list of clusters.

    Args:
        clusters: List of Cluster objects
        embeddings: Dict mapping member IDs to embedding vectors
        embedding_dim: Embedding dimension (default: 1024 for mxbai-embed-large)
        normalize: Whether to normalize centroids to unit vectors

    Returns:
        Dict mapping cluster_id to centroid vector
    """
    centroids = {}

    for cluster in clusters:
        centroid = compute_single_centroid(
            cluster=cluster,
            embeddings=embeddings,
            embedding_dim=embedding_dim,
            normalize=normalize
        )
        centroids[cluster.cluster_id] = centroid

    logger.info(f"Computed centroids for {len(centroids)} clusters")

    return centroids


def compute_single_centroid(
    cluster: Cluster,
    embeddings: Dict[str, np.ndarray],
    embedding_dim: int = 1024,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute centroid for a single cluster.

    Args:
        cluster: Cluster object
        embeddings: Dict mapping member IDs to embeddings
        embedding_dim: Embedding dimension
        normalize: Whether to normalize to unit vector

    Returns:
        Centroid vector (numpy array)
    """
    member_embeddings = []

    for member in cluster.members:
        if member in embeddings:
            member_embeddings.append(embeddings[member])

    if not member_embeddings:
        # No embeddings available - return zero vector
        logger.warning(f"Cluster {cluster.cluster_id} ({cluster.canonical_name}): No embeddings found, using zero vector")
        return np.zeros(embedding_dim, dtype=np.float32)

    # Compute mean
    centroid = np.mean(member_embeddings, axis=0)

    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm
        else:
            logger.warning(f"Cluster {cluster.cluster_id}: Zero-norm centroid, cannot normalize")

    return centroid.astype(np.float32)


def compute_centroids_with_details(
    clusters: List[Cluster],
    embeddings: Dict[str, np.ndarray],
    embedding_dim: int = 1024,
    normalize: bool = True
) -> List[CentroidResult]:
    """
    Compute centroids with detailed diagnostic information.

    Args:
        clusters: List of Cluster objects
        embeddings: Dict mapping member IDs to embeddings
        embedding_dim: Embedding dimension
        normalize: Whether to normalize

    Returns:
        List of CentroidResult objects with diagnostic info
    """
    results = []

    for cluster in clusters:
        member_embeddings = []
        missing_count = 0

        for member in cluster.members:
            if member in embeddings:
                member_embeddings.append(embeddings[member])
            else:
                missing_count += 1

        if not member_embeddings:
            centroid = np.zeros(embedding_dim, dtype=np.float32)
            normalization_applied = False
        else:
            centroid = np.mean(member_embeddings, axis=0)

            if normalize:
                norm = np.linalg.norm(centroid)
                if norm > 1e-9:
                    centroid = centroid / norm
                    normalization_applied = True
                else:
                    normalization_applied = False
            else:
                normalization_applied = False

        result = CentroidResult(
            cluster_id=cluster.cluster_id,
            centroid=centroid.astype(np.float32),
            member_count=len(cluster.members),
            missing_embeddings=missing_count,
            normalization_applied=normalization_applied
        )

        results.append(result)

    return results


def compute_weighted_centroid(
    cluster: Cluster,
    embeddings: Dict[str, np.ndarray],
    weights: Dict[str, float],
    embedding_dim: int = 1024,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute weighted centroid for a cluster.

    Useful for weighting members by importance (e.g., frequency, correlation strength).

    Args:
        cluster: Cluster object
        embeddings: Dict mapping member IDs to embeddings
        weights: Dict mapping member IDs to weight values
        embedding_dim: Embedding dimension
        normalize: Whether to normalize

    Returns:
        Weighted centroid vector
    """
    member_embeddings = []
    member_weights = []

    for member in cluster.members:
        if member in embeddings:
            member_embeddings.append(embeddings[member])
            weight = weights.get(member, 1.0)  # Default weight = 1.0
            member_weights.append(weight)

    if not member_embeddings:
        return np.zeros(embedding_dim, dtype=np.float32)

    # Convert to numpy arrays
    member_embeddings = np.array(member_embeddings)
    member_weights = np.array(member_weights)

    # Normalize weights
    member_weights = member_weights / (np.sum(member_weights) + 1e-9)

    # Compute weighted average
    centroid = np.average(member_embeddings, axis=0, weights=member_weights)

    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm

    return centroid.astype(np.float32)


def validate_centroids(
    centroids: Dict[int, np.ndarray],
    expected_dim: int = 1024
) -> Dict[str, any]:
    """
    Validate computed centroids.

    Checks:
    - All centroids have correct dimension
    - All centroids are normalized (if expected)
    - No NaN or Inf values

    Args:
        centroids: Dict mapping cluster_id to centroid
        expected_dim: Expected embedding dimension

    Returns:
        Dict with validation results
    """
    issues = []

    for cluster_id, centroid in centroids.items():
        # Check dimension
        if len(centroid) != expected_dim:
            issues.append(f"Cluster {cluster_id}: Wrong dimension ({len(centroid)} != {expected_dim})")

        # Check for NaN/Inf
        if np.any(np.isnan(centroid)):
            issues.append(f"Cluster {cluster_id}: Contains NaN values")

        if np.any(np.isinf(centroid)):
            issues.append(f"Cluster {cluster_id}: Contains Inf values")

        # Check if normalized (norm should be ~1.0 or 0.0 for zero vectors)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9 and abs(norm - 1.0) > 0.01:
            issues.append(f"Cluster {cluster_id}: Not normalized (norm={norm:.3f})")

    return {
        'valid': len(issues) == 0,
        'total_centroids': len(centroids),
        'issues': issues
    }


def print_centroid_statistics(results: List[CentroidResult]):
    """
    Print statistics about computed centroids.

    Args:
        results: List of CentroidResult objects
    """
    print("\n" + "="*60)
    print("CENTROID COMPUTATION STATISTICS")
    print("="*60)

    total_clusters = len(results)
    total_members = sum(r.member_count for r in results)
    total_missing = sum(r.missing_embeddings for r in results)
    normalized_count = sum(1 for r in results if r.normalization_applied)
    zero_vectors = sum(1 for r in results if np.allclose(r.centroid, 0.0))

    print(f"\nTotal clusters: {total_clusters}")
    print(f"Total members: {total_members}")
    print(f"Missing embeddings: {total_missing} ({total_missing/total_members*100:.1f}%)")
    print(f"Normalized centroids: {normalized_count}/{total_clusters} ({normalized_count/total_clusters*100:.1f}%)")
    print(f"Zero vectors: {zero_vectors}/{total_clusters} ({zero_vectors/total_clusters*100:.1f}%)")

    # Cluster size distribution
    sizes = [r.member_count for r in results]
    print(f"\nCluster size distribution:")
    print(f"  Min: {min(sizes)}")
    print(f"  Median: {np.median(sizes):.1f}")
    print(f"  Mean: {np.mean(sizes):.1f}")
    print(f"  Max: {max(sizes)}")

    # Clusters with most missing embeddings
    if total_missing > 0:
        top_missing = sorted(results, key=lambda r: r.missing_embeddings, reverse=True)[:5]
        print(f"\nTop 5 clusters with missing embeddings:")
        for r in top_missing:
            if r.missing_embeddings > 0:
                print(f"  Cluster {r.cluster_id}: {r.missing_embeddings}/{r.member_count} missing")

    print("="*60)


if __name__ == "__main__":
    # Test the centroid computation
    logging.basicConfig(level=logging.INFO)

    # Create test data
    test_clusters = [
        Cluster(0, "cluster_0", ["m1", "m2", "m3"], None, 0),
        Cluster(1, "cluster_1", ["m4", "m5"], None, 0),
        Cluster(2, "cluster_2", ["m6"], None, 0)
    ]

    test_embeddings = {
        'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
        'm2': np.array([0.9, 0.1, 0.0], dtype=np.float32),
        'm3': np.array([0.8, 0.2, 0.0], dtype=np.float32),
        'm4': np.array([0.0, 1.0, 0.0], dtype=np.float32),
        'm5': np.array([0.0, 0.9, 0.1], dtype=np.float32),
        'm6': np.array([0.0, 0.0, 1.0], dtype=np.float32)
    }

    print("Computing centroids...")
    centroids = compute_centroids(test_clusters, test_embeddings, embedding_dim=3)

    print("\nCentroids computed:")
    for cluster_id, centroid in centroids.items():
        print(f"  Cluster {cluster_id}: {centroid}")

    # Validate
    validation = validate_centroids(centroids, expected_dim=3)
    print(f"\nValidation: {'PASS' if validation['valid'] else 'FAIL'}")
    if not validation['valid']:
        for issue in validation['issues']:
            print(f"  - {issue}")

    # Detailed computation
    print("\n\nComputing with details...")
    results = compute_centroids_with_details(test_clusters, test_embeddings, embedding_dim=3)
    print_centroid_statistics(results)
