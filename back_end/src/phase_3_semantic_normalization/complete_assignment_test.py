"""
Complete Assignment Clustering Test

Guarantees 100% mechanism assignment by:
1. Running HDBSCAN to group similar mechanisms
2. Creating singleton clusters for any unassigned mechanisms

Every mechanism gets categorized - no mechanism is ignored.
"""

import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Tuple

try:
    import hdbscan
    from sklearn.metrics import silhouette_score, davies_bouldin_score
except ImportError:
    print("[ERROR] Required packages not installed!")
    exit(1)

# Configuration
DB_PATH = "c:/Users\samis/Desktop/MyBiome/back_end/data/processed/intervention_research.db"
CACHE_PATH = "back_end/data/semantic_normalization_cache/mechanism_embeddings_nomic.json"

def load_embeddings_cache(cache_path: str) -> Tuple[List[str], np.ndarray]:
    """Load embeddings from cache file."""
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        mechanisms = cache_data['mechanisms']
        embeddings = np.array(cache_data['embeddings'], dtype=np.float32)

        print(f"Loaded {len(mechanisms)} mechanisms from cache")
        return mechanisms, embeddings

    except FileNotFoundError:
        print(f"[ERROR] Cache file not found: {cache_path}")
        print("Run real_embedding_test.py first to generate embeddings")
        return None, None

def cluster_with_complete_assignment(embeddings: np.ndarray,
                                    min_cluster_size: int = 2,
                                    min_samples: int = 1,
                                    epsilon: float = 0.05) -> np.ndarray:
    """
    Cluster mechanisms with guaranteed 100% assignment.

    1. Run HDBSCAN to find natural groupings
    2. Assign any unassigned mechanisms to singleton clusters
    """

    print(f"\n[Clustering with complete assignment]")
    print(f"  Hyperparameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}")

    # Step 1: Run HDBSCAN
    print(f"  Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=epsilon,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    cluster_labels = clusterer.fit_predict(embeddings)

    # Step 2: Find unassigned mechanisms
    unassigned_mask = cluster_labels == -1
    num_unassigned = np.sum(unassigned_mask)
    num_hdbscan_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    print(f"  HDBSCAN results: {num_hdbscan_clusters} clusters, {num_unassigned} unassigned")

    # Step 3: Create singleton clusters for unassigned mechanisms
    if num_unassigned > 0:
        print(f"  Creating {num_unassigned} singleton clusters for unassigned mechanisms...")

        # Get next available cluster ID
        next_cluster_id = max(cluster_labels) + 1

        # Assign each unassigned mechanism to its own cluster
        for i, is_unassigned in enumerate(unassigned_mask):
            if is_unassigned:
                cluster_labels[i] = next_cluster_id
                next_cluster_id += 1

    # Final statistics
    final_num_clusters = len(set(cluster_labels))
    num_still_unassigned = np.sum(cluster_labels == -1)

    print(f"  Final results: {final_num_clusters} total clusters, {num_still_unassigned} unassigned (should be 0)")

    return cluster_labels

def analyze_clusters(mechanisms: List[str], cluster_labels: np.ndarray):
    """Analyze cluster composition."""

    # Group mechanisms by cluster
    cluster_dict = {}
    for mechanism, label in zip(mechanisms, cluster_labels):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(mechanism)

    # Sort by size
    clusters_sorted = sorted(
        cluster_dict.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    # Statistics
    sizes = [len(v) for k, v in clusters_sorted]

    print("\n" + "="*80)
    print("CLUSTER STATISTICS")
    print("="*80)

    print(f"\nTotal clusters: {len(clusters_sorted)}")
    print(f"Total mechanisms: {len(mechanisms)}")
    print(f"Mechanisms assigned: {sum(sizes)} (should be {len(mechanisms)})")

    print(f"\nCluster size distribution:")
    print(f"  Average size: {np.mean(sizes):.2f}")
    print(f"  Median size: {np.median(sizes):.0f}")
    print(f"  Largest cluster: {max(sizes)} members")
    print(f"  Smallest cluster: {min(sizes)} members")

    # Size buckets
    buckets = {
        "Singleton (1)": sum(1 for s in sizes if s == 1),
        "Pair (2)": sum(1 for s in sizes if s == 2),
        "Small (3-5)": sum(1 for s in sizes if 3 <= s <= 5),
        "Medium (6-10)": sum(1 for s in sizes if 6 <= s <= 10),
        "Large (11-20)": sum(1 for s in sizes if 11 <= s <= 20),
        "Very Large (>20)": sum(1 for s in sizes if s > 20)
    }

    print("\nCluster size buckets:")
    for bucket, count in buckets.items():
        percentage = (count / len(sizes)) * 100
        print(f"  {bucket}: {count} clusters ({percentage:.1f}%)")

    return clusters_sorted

def display_sample_clusters(clusters_sorted: List[Tuple[int, List[str]]],
                           num_multi_member: int = 10,
                           num_singleton: int = 5):
    """Display sample clusters."""

    # Separate multi-member and singleton clusters
    multi_member = [(k, v) for k, v in clusters_sorted if len(v) > 1]
    singletons = [(k, v) for k, v in clusters_sorted if len(v) == 1]

    print("\n" + "="*80)
    print(f"TOP {num_multi_member} MULTI-MEMBER CLUSTERS")
    print("="*80)

    for cluster_id, members in multi_member[:num_multi_member]:
        print(f"\nCluster {cluster_id} ({len(members)} members):")
        print("-"*80)
        for i, mech in enumerate(members[:7], 1):
            print(f"  {i}. {mech[:75]}{'...' if len(mech) > 75 else ''}")
        if len(members) > 7:
            print(f"  ... and {len(members) - 7} more")

    if singletons:
        print("\n" + "="*80)
        print(f"SAMPLE SINGLETON CLUSTERS (first {num_singleton} of {len(singletons)})")
        print("="*80)
        print("\nThese are unique mechanisms without close semantic matches:")

        for cluster_id, members in singletons[:num_singleton]:
            mech = members[0]
            print(f"  Cluster {cluster_id}: {mech[:75]}{'...' if len(mech) > 75 else ''}")

def compute_quality_metrics(embeddings: np.ndarray, cluster_labels: np.ndarray):
    """Compute clustering quality metrics."""

    print("\n" + "="*80)
    print("CLUSTERING QUALITY METRICS")
    print("="*80)

    # Filter out singleton clusters for quality metrics
    # (silhouette score requires at least 2 samples per cluster)
    cluster_sizes = {}
    for label in cluster_labels:
        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

    # Mask for non-singleton clusters
    non_singleton_mask = np.array([cluster_sizes[label] > 1 for label in cluster_labels])

    if np.sum(non_singleton_mask) > 10:  # Need sufficient points
        filtered_embeddings = embeddings[non_singleton_mask]
        filtered_labels = cluster_labels[non_singleton_mask]

        silhouette = silhouette_score(filtered_embeddings, filtered_labels)
        davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)

        print(f"\nQuality metrics (for multi-member clusters only):")
        print(f"  Silhouette score: {silhouette:.3f} (target: >0.40)")
        print(f"  Davies-Bouldin index: {davies_bouldin:.3f} (target: <1.0)")

        if silhouette > 0.40 and davies_bouldin < 1.0:
            print(f"  Status: GOOD - Clusters are well-separated and coherent")
        elif silhouette > 0.30:
            print(f"  Status: ACCEPTABLE - Reasonable cluster quality")
        else:
            print(f"  Status: NEEDS IMPROVEMENT - Consider adjusting epsilon")
    else:
        print("\nNot enough multi-member clusters for quality metrics")

def main():
    """Run complete assignment clustering test."""

    print("="*80)
    print("COMPLETE ASSIGNMENT CLUSTERING TEST")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("Goal: Ensure 100% mechanism assignment")
    print("Method: HDBSCAN + singleton clusters for unassigned mechanisms\n")

    # Load embeddings
    print("[1/4] Loading embeddings...")
    mechanisms, embeddings = load_embeddings_cache(CACHE_PATH)

    if mechanisms is None:
        return

    # Run clustering with complete assignment
    print("\n[2/4] Clustering with guaranteed assignment...")
    cluster_labels = cluster_with_complete_assignment(
        embeddings,
        min_cluster_size=2,
        min_samples=1,
        epsilon=0.05  # Test value - will optimize via experiments
    )

    # Analyze results
    print("\n[3/4] Analyzing cluster composition...")
    clusters_sorted = analyze_clusters(mechanisms, cluster_labels)

    # Compute quality metrics
    print("\n[4/4] Computing quality metrics...")
    compute_quality_metrics(embeddings, cluster_labels)

    # Display samples
    display_sample_clusters(clusters_sorted, num_multi_member=10, num_singleton=5)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    print(f"\nTotal mechanisms: {len(mechanisms)}")
    print(f"Total clusters: {len(clusters_sorted)}")
    print(f"Assignment rate: 100% (guaranteed)")

    print("\nKey insight:")
    print("  Every mechanism is categorized - either grouped with similar mechanisms")
    print("  or assigned to its own singleton cluster if truly unique.")

    print("\nNext steps:")
    print("  1. Run hyperparameter experiments to optimize epsilon")
    print("  2. Find balance between grouping similar mechanisms and preserving uniqueness")
    print("  3. Integrate optimal configuration into main pipeline")

    print("\n" + "="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
