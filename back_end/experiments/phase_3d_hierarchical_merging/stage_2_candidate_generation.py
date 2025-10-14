"""
Stage 2: Candidate Generation

Find merge candidate pairs based on centroid similarity.
Filters and ranks pairs above threshold for LLM validation.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from validation_metrics import Cluster, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class MergeCandidate:
    """Represents a potential cluster merge."""
    cluster_a_id: int
    cluster_b_id: int
    cluster_a: Cluster
    cluster_b: Cluster
    similarity: float
    confidence_tier: str  # 'HIGH', 'MEDIUM', 'LOW'


def generate_merge_candidates(
    clusters: List[Cluster],
    centroids: Dict[int, np.ndarray],
    similarity_threshold: float = 0.85,
    max_candidates: Optional[int] = None
) -> List[MergeCandidate]:
    """
    Generate merge candidate pairs based on centroid similarity.

    Args:
        clusters: List of Cluster objects
        centroids: Dict mapping cluster_id to centroid vector
        similarity_threshold: Minimum similarity to consider
        max_candidates: Maximum number of candidates to return (None = all)

    Returns:
        List of MergeCandidate objects, sorted by similarity (descending)
    """
    logger.info(f"Generating merge candidates (threshold={similarity_threshold})...")

    # Build cluster lookup
    cluster_lookup = {c.cluster_id: c for c in clusters}

    candidates = []

    # Compute pairwise similarities
    cluster_ids = list(centroids.keys())

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            id_a = cluster_ids[i]
            id_b = cluster_ids[j]

            # Compute similarity
            centroid_a = centroids[id_a]
            centroid_b = centroids[id_b]

            sim = cosine_similarity(centroid_a, centroid_b)

            if sim >= similarity_threshold:
                # Determine confidence tier
                confidence_tier = get_confidence_tier(sim)

                candidate = MergeCandidate(
                    cluster_a_id=id_a,
                    cluster_b_id=id_b,
                    cluster_a=cluster_lookup[id_a],
                    cluster_b=cluster_lookup[id_b],
                    similarity=sim,
                    confidence_tier=confidence_tier
                )

                candidates.append(candidate)

    # Sort by similarity (descending)
    candidates.sort(key=lambda c: c.similarity, reverse=True)

    # Limit if requested
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    logger.info(f"  Found {len(candidates)} candidates above threshold")

    # Log tier distribution
    tier_counts = {}
    for candidate in candidates:
        tier_counts[candidate.confidence_tier] = tier_counts.get(candidate.confidence_tier, 0) + 1

    for tier, count in sorted(tier_counts.items()):
        logger.info(f"    {tier}: {count} candidates")

    return candidates


def get_confidence_tier(similarity: float) -> str:
    """
    Classify similarity into confidence tier.

    Args:
        similarity: Cosine similarity score (0.0 to 1.0)

    Returns:
        Confidence tier string: 'HIGH', 'MEDIUM', or 'LOW'
    """
    if similarity >= 0.90:
        return 'HIGH'
    elif similarity >= 0.85:
        return 'MEDIUM'
    else:
        return 'LOW'


def filter_candidates_by_tier(
    candidates: List[MergeCandidate],
    min_tier: str = 'MEDIUM'
) -> List[MergeCandidate]:
    """
    Filter candidates by minimum confidence tier.

    Args:
        candidates: List of MergeCandidate objects
        min_tier: Minimum tier to keep ('HIGH', 'MEDIUM', 'LOW')

    Returns:
        Filtered list of candidates
    """
    tier_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    min_level = tier_order[min_tier]

    filtered = [c for c in candidates if tier_order[c.confidence_tier] >= min_level]

    logger.info(f"Filtered to {len(filtered)} candidates (min tier: {min_tier})")

    return filtered


def filter_candidates_by_size_imbalance(
    candidates: List[MergeCandidate],
    max_size_ratio: float = 10.0
) -> List[MergeCandidate]:
    """
    Filter out candidates with large size imbalance.

    Prevents merging tiny clusters into huge clusters.

    Args:
        candidates: List of MergeCandidate objects
        max_size_ratio: Maximum ratio of larger/smaller cluster size

    Returns:
        Filtered list of candidates
    """
    filtered = []

    for candidate in candidates:
        size_a = len(candidate.cluster_a.members)
        size_b = len(candidate.cluster_b.members)

        size_ratio = max(size_a, size_b) / max(min(size_a, size_b), 1)

        if size_ratio <= max_size_ratio:
            filtered.append(candidate)

    removed = len(candidates) - len(filtered)
    if removed > 0:
        logger.info(f"Filtered out {removed} candidates due to size imbalance (ratio>{max_size_ratio})")

    return filtered


def remove_duplicate_pairs(candidates: List[MergeCandidate]) -> List[MergeCandidate]:
    """
    Remove duplicate candidate pairs.

    Ensures (A, B) and (B, A) are not both present.

    Args:
        candidates: List of MergeCandidate objects

    Returns:
        Deduplicated list
    """
    seen_pairs = set()
    deduplicated = []

    for candidate in candidates:
        # Create normalized pair key (smaller ID first)
        pair_key = tuple(sorted([candidate.cluster_a_id, candidate.cluster_b_id]))

        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            deduplicated.append(candidate)

    removed = len(candidates) - len(deduplicated)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate pairs")

    return deduplicated


def group_candidates_by_cluster(
    candidates: List[MergeCandidate]
) -> Dict[int, List[MergeCandidate]]:
    """
    Group candidates by cluster involvement.

    Useful for analyzing which clusters have many potential merges.

    Args:
        candidates: List of MergeCandidate objects

    Returns:
        Dict mapping cluster_id to list of candidates involving that cluster
    """
    cluster_groups = {}

    for candidate in candidates:
        # Add to cluster A's group
        if candidate.cluster_a_id not in cluster_groups:
            cluster_groups[candidate.cluster_a_id] = []
        cluster_groups[candidate.cluster_a_id].append(candidate)

        # Add to cluster B's group
        if candidate.cluster_b_id not in cluster_groups:
            cluster_groups[candidate.cluster_b_id] = []
        cluster_groups[candidate.cluster_b_id].append(candidate)

    return cluster_groups


def print_candidate_summary(candidates: List[MergeCandidate], top_n: int = 10):
    """
    Print summary of candidate pairs.

    Args:
        candidates: List of MergeCandidate objects
        top_n: Number of top candidates to display
    """
    print("\n" + "="*80)
    print(f"MERGE CANDIDATES SUMMARY ({len(candidates)} total)")
    print("="*80)

    if len(candidates) == 0:
        print("No candidates found.")
        return

    # Tier distribution
    tier_counts = {}
    for candidate in candidates:
        tier_counts[candidate.confidence_tier] = tier_counts.get(candidate.confidence_tier, 0) + 1

    print("\nConfidence Tier Distribution:")
    for tier in ['HIGH', 'MEDIUM', 'LOW']:
        count = tier_counts.get(tier, 0)
        percentage = count / len(candidates) * 100
        print(f"  {tier:8s}: {count:4d} ({percentage:5.1f}%)")

    # Similarity statistics
    similarities = [c.similarity for c in candidates]
    print(f"\nSimilarity Statistics:")
    print(f"  Min: {min(similarities):.3f}")
    print(f"  Mean: {np.mean(similarities):.3f}")
    print(f"  Median: {np.median(similarities):.3f}")
    print(f"  Max: {max(similarities):.3f}")

    # Top candidates
    print(f"\nTop {min(top_n, len(candidates))} Candidates:")
    print(f"{'Sim':>6s} | {'Tier':8s} | {'Cluster A':30s} | {'Cluster B':30s}")
    print("-" * 80)

    for i, candidate in enumerate(candidates[:top_n]):
        name_a = candidate.cluster_a.canonical_name[:28]
        name_b = candidate.cluster_b.canonical_name[:28]
        print(f"{candidate.similarity:6.3f} | {candidate.confidence_tier:8s} | {name_a:30s} | {name_b:30s}")

    print("="*80)


def analyze_candidate_network(candidates: List[MergeCandidate]) -> Dict[str, any]:
    """
    Analyze candidate network properties.

    Identifies highly connected clusters and potential merge chains.

    Args:
        candidates: List of MergeCandidate objects

    Returns:
        Dict with network analysis results
    """
    cluster_groups = group_candidates_by_cluster(candidates)

    # Count connections per cluster
    connection_counts = {cid: len(cands) for cid, cands in cluster_groups.items()}

    # Identify highly connected clusters (>= 5 connections)
    highly_connected = {cid: count for cid, count in connection_counts.items() if count >= 5}

    # Compute average connections
    avg_connections = np.mean(list(connection_counts.values())) if connection_counts else 0

    analysis = {
        'total_clusters_involved': len(cluster_groups),
        'total_connections': len(candidates),
        'avg_connections_per_cluster': avg_connections,
        'highly_connected_clusters': highly_connected,
        'max_connections': max(connection_counts.values()) if connection_counts else 0,
        'isolated_clusters': sum(1 for count in connection_counts.values() if count == 1)
    }

    return analysis


def save_candidates_to_file(candidates: List[MergeCandidate], output_path: str):
    """
    Save candidates to JSON file for inspection.

    Args:
        candidates: List of MergeCandidate objects
        output_path: Path to output file
    """
    import json
    from datetime import datetime

    data = {
        'timestamp': datetime.now().isoformat(),
        'total_candidates': len(candidates),
        'candidates': [
            {
                'cluster_a_id': c.cluster_a_id,
                'cluster_a_name': c.cluster_a.canonical_name,
                'cluster_a_size': len(c.cluster_a.members),
                'cluster_b_id': c.cluster_b_id,
                'cluster_b_name': c.cluster_b.canonical_name,
                'cluster_b_size': len(c.cluster_b.members),
                'similarity': float(c.similarity),
                'confidence_tier': c.confidence_tier
            }
            for c in candidates
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved candidates to: {output_path}")


if __name__ == "__main__":
    # Test candidate generation
    logging.basicConfig(level=logging.INFO)

    # Create test clusters
    test_clusters = [
        Cluster(0, "cluster_0", ["m1", "m2"], None, 0),
        Cluster(1, "cluster_1", ["m3", "m4"], None, 0),
        Cluster(2, "cluster_2", ["m5"], None, 0),
        Cluster(3, "cluster_3", ["m6", "m7", "m8"], None, 0)
    ]

    # Create test centroids (similar pairs: 0-1, 2-3)
    test_centroids = {
        0: np.array([0.9, 0.1, 0.0], dtype=np.float32),
        1: np.array([0.95, 0.05, 0.0], dtype=np.float32),  # Very similar to 0
        2: np.array([0.0, 0.9, 0.1], dtype=np.float32),
        3: np.array([0.0, 0.85, 0.15], dtype=np.float32)  # Somewhat similar to 2
    }

    print("Generating candidates...")
    candidates = generate_merge_candidates(
        test_clusters,
        test_centroids,
        similarity_threshold=0.80
    )

    print_candidate_summary(candidates)

    # Test filtering
    print("\n\nFiltering by tier (HIGH only)...")
    high_tier = filter_candidates_by_tier(candidates, min_tier='HIGH')
    print(f"  Remaining: {len(high_tier)} candidates")

    # Network analysis
    print("\n\nNetwork Analysis:")
    analysis = analyze_candidate_network(candidates)
    for key, value in analysis.items():
        print(f"  {key}: {value}")
