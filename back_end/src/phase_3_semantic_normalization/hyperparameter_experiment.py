"""
Hyperparameter Grid Search for Mechanism Clustering

Tests multiple combinations of HDBSCAN hyperparameters to find optimal
clustering configuration. Uses real nomic-embed-text embeddings.

Hyperparameters tested:
- min_cluster_size: [3, 5, 7, 10]
- min_samples: [2, 3, 5]
- cluster_selection_epsilon: [0.0, 0.05, 0.10]

Total: 36 configurations
"""

import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import List, Dict, Tuple
import time

try:
    import hdbscan
except ImportError:
    print("[ERROR] hdbscan not installed!")
    exit(1)

try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
except ImportError:
    print("[ERROR] sklearn not installed!")
    exit(1)

# Configuration
DB_PATH = "c:/Users/samis/Desktop/MyBiome/back_end/data/processed/intervention_research.db"
CACHE_PATH = "back_end/data/semantic_normalization_cache/mechanism_embeddings_nomic.json"
RESULTS_PATH = "back_end/data/semantic_normalization_results/hyperparameter_results.json"

# Hyperparameter grid - allow ALL mechanisms to be assigned
# Every mechanism is unique and meaningful - no mechanism should be left unassigned
# min_cluster_size=2 is HDBSCAN's minimum, min_samples=1 allows single-mechanism clusters
PARAM_GRID = {
    'min_cluster_size': [2],  # Minimum allowed by HDBSCAN (cannot be 1)
    'min_samples': [1],  # Allow single mechanisms to form their own cluster
    'cluster_selection_epsilon': [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]  # Test various epsilon values
}

def load_embeddings_cache(cache_path: str) -> Tuple[List[str], np.ndarray]:
    """Load embeddings from cache file."""
    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        mechanisms = cache_data['mechanisms']
        embeddings = np.array(cache_data['embeddings'], dtype=np.float32)

        return mechanisms, embeddings

    except FileNotFoundError:
        print(f"[ERROR] Cache file not found: {cache_path}")
        print("Run real_embedding_test.py first to generate embeddings cache")
        return None, None
    except Exception as e:
        print(f"[ERROR] Failed to load cache: {e}")
        return None, None

def run_single_experiment(embeddings: np.ndarray, min_cluster_size: int,
                         min_samples: int, cluster_selection_epsilon: float) -> Dict:
    """Run single clustering experiment with given hyperparameters."""

    # Run clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    cluster_labels = clusterer.fit_predict(embeddings)

    # Basic statistics
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = list(cluster_labels).count(-1)
    singleton_percentage = (num_noise / len(cluster_labels)) * 100

    result = {
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'cluster_selection_epsilon': cluster_selection_epsilon,
        'num_clusters': num_clusters,
        'num_noise': num_noise,
        'singleton_percentage': singleton_percentage,
        'silhouette_score': None,
        'davies_bouldin_index': None,
        'calinski_harabasz_score': None
    }

    # Quality metrics (only if we have clusters)
    if num_clusters > 1:
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) > 10:  # Need at least 10 points
            try:
                silhouette = silhouette_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
                davies_bouldin = davies_bouldin_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
                calinski_harabasz = calinski_harabasz_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])

                result['silhouette_score'] = float(silhouette)
                result['davies_bouldin_index'] = float(davies_bouldin)
                result['calinski_harabasz_score'] = float(calinski_harabasz)
            except:
                pass

    # Cluster sizes
    cluster_dict = {}
    for label in cluster_labels:
        if label != -1:
            cluster_dict[label] = cluster_dict.get(label, 0) + 1

    if cluster_dict:
        sizes = list(cluster_dict.values())
        result['avg_cluster_size'] = float(np.mean(sizes))
        result['max_cluster_size'] = int(max(sizes))
        result['min_cluster_size_actual'] = int(min(sizes))
    else:
        result['avg_cluster_size'] = 0
        result['max_cluster_size'] = 0
        result['min_cluster_size_actual'] = 0

    return result

def generate_all_configurations():
    """Generate all hyperparameter combinations."""
    configs = []

    for min_cluster_size in PARAM_GRID['min_cluster_size']:
        for min_samples in PARAM_GRID['min_samples']:
            for epsilon in PARAM_GRID['cluster_selection_epsilon']:
                configs.append({
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'cluster_selection_epsilon': epsilon
                })

    return configs

def score_configuration(result: Dict) -> float:
    """
    Score a configuration based on multiple criteria.

    Priority: Assign ALL mechanisms (no mechanism should be ignored)

    Good clustering should have:
    - Very low unassigned percentage (target: 0% - all mechanisms assigned)
    - High silhouette score (>0.40)
    - Reasonable number of clusters
    - Low Davies-Bouldin index (<1.0)
    """

    score = 0.0

    # Unassigned percentage (weight: 50% - HIGHEST PRIORITY)
    # Every mechanism is meaningful and should be assigned
    unassigned_pct = result['singleton_percentage']
    if unassigned_pct == 0:
        unassigned_score = 1.0
    elif unassigned_pct < 5:
        unassigned_score = 0.9
    elif unassigned_pct < 10:
        unassigned_score = 0.7
    else:
        unassigned_score = max(0, 1.0 - (unassigned_pct / 100))
    score += unassigned_score * 50

    # Silhouette score (weight: 30%)
    if result['silhouette_score'] is not None:
        score += min(result['silhouette_score'] / 0.50, 1.0) * 30

    # Davies-Bouldin index (weight: 10%, inverted)
    if result['davies_bouldin_index'] is not None:
        db_score = max(0, 1.0 - (result['davies_bouldin_index'] / 2.0))
        score += db_score * 10

    # Number of clusters (weight: 10%)
    # Prefer meaningful groupings (50-200 clusters for 666 mechanisms)
    num_clusters = result['num_clusters']
    if 50 <= num_clusters <= 200:
        cluster_score = 1.0
    elif num_clusters < 50:
        cluster_score = num_clusters / 50.0
    else:
        cluster_score = max(0, 1.0 - ((num_clusters - 200) / 400))
    score += cluster_score * 10

    return score

def main():
    """Run hyperparameter grid search."""
    print("="*80)
    print("HYPERPARAMETER GRID SEARCH FOR MECHANISM CLUSTERING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load embeddings
    print("[1/3] Loading embeddings cache...")
    mechanisms, embeddings = load_embeddings_cache(CACHE_PATH)

    if mechanisms is None:
        return

    print(f"  Loaded {len(mechanisms)} mechanisms with embeddings")

    # Generate configurations
    print("\n[2/3] Generating hyperparameter configurations...")
    configs = generate_all_configurations()
    print(f"  Total configurations to test: {len(configs)}")

    print("\n  Grid:")
    for param, values in PARAM_GRID.items():
        print(f"    {param}: {values}")

    # Run experiments
    print("\n[3/3] Running experiments...")
    results = []

    for i, config in enumerate(configs, 1):
        print(f"\n  Experiment {i}/{len(configs)}: "
              f"min_cluster_size={config['min_cluster_size']}, "
              f"min_samples={config['min_samples']}, "
              f"epsilon={config['cluster_selection_epsilon']}")

        start_time = time.time()

        result = run_single_experiment(
            embeddings,
            config['min_cluster_size'],
            config['min_samples'],
            config['cluster_selection_epsilon']
        )

        elapsed = time.time() - start_time

        # Calculate score
        result['score'] = score_configuration(result)

        results.append(result)

        # Print summary
        silhouette_str = f"{result['silhouette_score']:.3f}" if result['silhouette_score'] else 'N/A'
        print(f"    Clusters: {result['num_clusters']}, "
              f"Unassigned: {result['singleton_percentage']:.1f}%, "
              f"Silhouette: {silhouette_str}, "
              f"Score: {result['score']:.1f}/100, "
              f"Time: {elapsed:.1f}s")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_data = {
        'experiment_timestamp': datetime.now().isoformat(),
        'total_mechanisms': len(mechanisms),
        'total_configurations': len(configs),
        'param_grid': PARAM_GRID,
        'results': results
    }

    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")

    # Analyze results
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS")
    print("="*80)

    # Sort by score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    for i, result in enumerate(sorted_results[:10], 1):
        print(f"\n{i}. Score: {result['score']:.1f}/100")
        print(f"   Hyperparameters:")
        print(f"     min_cluster_size: {result['min_cluster_size']}")
        print(f"     min_samples: {result['min_samples']}")
        print(f"     cluster_selection_epsilon: {result['cluster_selection_epsilon']}")
        print(f"   Metrics:")
        print(f"     Clusters: {result['num_clusters']}")
        print(f"     Singletons: {result['singleton_percentage']:.1f}%")
        if result['silhouette_score']:
            print(f"     Silhouette: {result['silhouette_score']:.3f}")
            print(f"     Davies-Bouldin: {result['davies_bouldin_index']:.3f}")
            print(f"     Calinski-Harabasz: {result['calinski_harabasz_score']:.1f}")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    best_result = sorted_results[0]

    print("\nBest configuration:")
    print(f"  min_cluster_size: {best_result['min_cluster_size']}")
    print(f"  min_samples: {best_result['min_samples']}")
    print(f"  cluster_selection_epsilon: {best_result['cluster_selection_epsilon']}")

    print("\nExpected performance:")
    print(f"  Clusters: {best_result['num_clusters']}")
    print(f"  Singleton percentage: {best_result['singleton_percentage']:.1f}%")
    if best_result['silhouette_score']:
        print(f"  Silhouette score: {best_result['silhouette_score']:.3f}")

    print("\nNext steps:")
    print("  1. Review top configurations manually")
    print("  2. Test best configuration with sample cluster output")
    print("  3. Update mechanism_normalizer.py with optimal hyperparameters")
    print("  4. Proceed to Phase 6-8 integration")

    print("\n" + "="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()
