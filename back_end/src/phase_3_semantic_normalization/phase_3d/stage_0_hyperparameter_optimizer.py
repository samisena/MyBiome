"""
Stage 0: Hyperparameter Optimization

Grid search across threshold combinations to find optimal similarity thresholds
for hierarchical merging. Uses fast simulation (no LLM calls).
"""

import time
import logging
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict

from .config import Phase3dConfig, get_config
from .validation_metrics import (
    Cluster,
    evaluate_hierarchy_quality,
    print_metrics_summary,
    cosine_similarity
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a single threshold configuration test."""
    thresholds: Dict[int, float]  # level -> threshold
    metrics: Dict[str, Any]
    composite_score: float
    execution_time: float


class HyperparameterOptimizer:
    """
    Grid search optimizer for similarity thresholds.

    Tests all combinations from config.threshold_search_space
    and evaluates quality metrics.
    """

    def __init__(self, db_path: str = None, entity_type: str = 'mechanism', config: Phase3dConfig = None):
        """
        Initialize optimizer.

        Args:
            db_path: Path to database
            entity_type: 'mechanism', 'intervention', or 'condition'
            config: Configuration object (uses global if None)
        """
        self.config = config or get_config()
        self.db_path = db_path or self.config.db_path
        self.entity_type = entity_type

        # Load data
        self.clusters = []
        self.embeddings = {}

        logger.info(f"Initialized HyperparameterOptimizer for {entity_type}")

    def load_data(self):
        """Load clusters and embeddings from database/cache."""
        logger.info(f"Loading {self.entity_type} data...")

        if self.entity_type == 'mechanism':
            self._load_mechanism_data()
        elif self.entity_type in ['intervention', 'condition']:
            self._load_entity_data(self.entity_type)
        else:
            raise ValueError(f"Unknown entity type: {self.entity_type}")

        logger.info(f"  Loaded {len(self.clusters)} clusters")
        logger.info(f"  Loaded {len(self.embeddings)} embeddings")

    def _load_mechanism_data(self):
        """Load mechanism clusters from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load clusters
        cursor.execute("""
            SELECT cluster_id, canonical_name, member_count, hierarchy_level, parent_cluster_id
            FROM mechanism_clusters
            ORDER BY cluster_id
        """)

        for row in cursor.fetchall():
            cluster_id, canonical_name, member_count, hierarchy_level, parent_id = row

            # Get members
            cursor.execute("""
                SELECT mechanism_text
                FROM mechanism_cluster_membership
                WHERE cluster_id = ?
            """, (cluster_id,))

            members = [r[0] for r in cursor.fetchall()]

            self.clusters.append(Cluster(
                cluster_id=cluster_id,
                canonical_name=canonical_name,
                members=members,
                parent_id=parent_id,
                hierarchy_level=hierarchy_level
            ))

        conn.close()

        # Load embeddings from cache
        cache_path = Path(self.config.cache_dir) / "mechanism_embeddings_nomic.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                mechanisms = cache_data.get('mechanisms', [])
                embeddings_list = cache_data.get('embeddings', [])

                for mech, emb in zip(mechanisms, embeddings_list):
                    self.embeddings[mech] = np.array(emb, dtype=np.float32)

            logger.info(f"  Loaded embeddings from cache: {cache_path}")
        else:
            logger.warning(f"  Embeddings cache not found: {cache_path}")

    def _load_entity_data(self, entity_type: str):
        """Load intervention/condition entities from semantic_hierarchy."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load from canonical_groups
        cursor.execute("""
            SELECT id, canonical_name, entity_type, member_count
            FROM canonical_groups
            WHERE entity_type = ?
            ORDER BY id
        """, (entity_type,))

        for row in cursor.fetchall():
            group_id, canonical_name, ent_type, member_count = row

            # Get members from semantic_hierarchy
            cursor.execute("""
                SELECT entity_name
                FROM semantic_hierarchy
                WHERE entity_type = ? AND layer_1_canonical = ?
            """, (entity_type, canonical_name))

            members = [r[0] for r in cursor.fetchall()]

            self.clusters.append(Cluster(
                cluster_id=group_id,
                canonical_name=canonical_name,
                members=members,
                parent_id=None,
                hierarchy_level=0
            ))

        conn.close()

        # Load embeddings from semantic_hierarchy
        cursor = conn.cursor()
        cursor.execute("""
            SELECT entity_name, embedding_vector
            FROM semantic_hierarchy
            WHERE entity_type = ? AND embedding_vector IS NOT NULL
        """, (entity_type,))

        for row in cursor.fetchall():
            entity_name, embedding_blob = row
            if embedding_blob:
                # Deserialize BLOB to numpy array
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                self.embeddings[entity_name] = embedding

    def optimize(self) -> List[OptimizationResult]:
        """
        Run grid search optimization.

        Returns:
            List of OptimizationResult objects, sorted by composite score (best first)
        """
        logger.info("="*80)
        logger.info("HYPERPARAMETER OPTIMIZATION: Grid Search")
        logger.info("="*80)

        # Load data
        self.load_data()

        if len(self.clusters) == 0:
            logger.error("No clusters found - cannot optimize")
            return []

        # Get search space
        search_space = self.config.threshold_search_space

        level_3_to_2_values = search_space['level_3_to_2']
        level_2_to_1_values = search_space['level_2_to_1']
        level_1_to_0_values = search_space['level_1_to_0']

        total_configs = (
            len(level_3_to_2_values) *
            len(level_2_to_1_values) *
            len(level_1_to_0_values)
        )

        logger.info(f"Testing {total_configs} threshold combinations...")
        logger.info(f"  Level 3→2: {level_3_to_2_values}")
        logger.info(f"  Level 2→1: {level_2_to_1_values}")
        logger.info(f"  Level 1→0: {level_1_to_0_values}")

        results = []
        config_num = 0

        start_time = time.time()

        # Grid search
        for t3_to_2 in level_3_to_2_values:
            for t2_to_1 in level_2_to_1_values:
                for t1_to_0 in level_1_to_0_values:
                    config_num += 1

                    thresholds = {
                        3: t3_to_2,
                        2: t2_to_1,
                        1: t1_to_0
                    }

                    logger.info(f"\n[Config {config_num}/{total_configs}] Testing: {thresholds}")

                    config_start = time.time()

                    # Run simulation
                    try:
                        hierarchy = self.simulate_hierarchy_building(thresholds)

                        # Evaluate quality
                        metrics = evaluate_hierarchy_quality(
                            hierarchy,
                            self.embeddings,
                            config={
                                'target_reduction_range': self.config.target_reduction_range,
                                'target_depth_range': self.config.target_depth_range,
                                'target_singleton_threshold': self.config.target_singleton_threshold,
                                'target_coherence_min': self.config.target_coherence_min,
                                'target_separation_min': self.config.target_separation_min
                            }
                        )

                        config_time = time.time() - config_start

                        result = OptimizationResult(
                            thresholds=thresholds,
                            metrics=asdict(metrics),
                            composite_score=metrics.composite_score,
                            execution_time=config_time
                        )

                        results.append(result)

                        logger.info(f"  Composite score: {metrics.composite_score:.1f}/100")
                        logger.info(f"  Reduction: {metrics.reduction_ratio:.1%} ({metrics.initial_count}→{metrics.final_count})")
                        logger.info(f"  Time: {config_time:.2f}s")

                    except Exception as e:
                        logger.error(f"  Failed: {e}")
                        continue

        total_time = time.time() - start_time

        # Sort by composite score (best first)
        results.sort(key=lambda x: x.composite_score, reverse=True)

        logger.info("\n" + "="*80)
        logger.info(f"OPTIMIZATION COMPLETE ({total_time:.1f}s total)")
        logger.info("="*80)

        # Print top 5
        self.print_top_results(results[:5])

        # Save results
        self.save_results(results)

        return results

    def simulate_hierarchy_building(self, thresholds: Dict[int, float]) -> Dict[str, List[Cluster]]:
        """
        Simulate hierarchy building using only embedding similarity.

        No LLM calls - just pure similarity-based merging.

        Args:
            thresholds: Dict mapping level to threshold value

        Returns:
            Dict mapping level names to cluster lists
        """
        current_clusters = [c for c in self.clusters if c.hierarchy_level == 0]  # Start with flat clusters
        hierarchy = {'level_3': current_clusters.copy()}

        for level in [3, 2, 1]:
            threshold = thresholds.get(level)
            if threshold is None:
                continue

            # Compute centroids
            centroids = self._compute_centroids(current_clusters)

            # Find merge pairs
            merge_pairs = []
            for i in range(len(current_clusters)):
                for j in range(i+1, len(current_clusters)):
                    sim = cosine_similarity(centroids[i], centroids[j])
                    if sim >= threshold:
                        merge_pairs.append((i, j, sim))

            if not merge_pairs:
                # No merges possible at this level
                hierarchy[f'level_{level-1}'] = current_clusters.copy()
                continue

            # Apply greedy merging
            parent_clusters = self._apply_greedy_merges(current_clusters, merge_pairs, level)

            hierarchy[f'level_{level-1}'] = parent_clusters
            current_clusters = parent_clusters

        return hierarchy

    def _compute_centroids(self, clusters: List[Cluster]) -> List[np.ndarray]:
        """Compute centroid embeddings for clusters."""
        centroids = []

        for cluster in clusters:
            member_embeddings = []
            for member in cluster.members:
                if member in self.embeddings:
                    member_embeddings.append(self.embeddings[member])

            if member_embeddings:
                centroid = np.mean(member_embeddings, axis=0)
                # Normalize
                centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
                centroids.append(centroid)
            else:
                # No embeddings available - use zero vector
                centroids.append(np.zeros(self.config.embedding_dimension, dtype=np.float32))

        return centroids

    def _apply_greedy_merges(
        self,
        clusters: List[Cluster],
        merge_pairs: List[Tuple[int, int, float]],
        level: int
    ) -> List[Cluster]:
        """
        Apply greedy merging strategy.

        Merges highest similarity pairs first, updates indices.

        Args:
            clusters: List of clusters to merge
            merge_pairs: List of (index_i, index_j, similarity) tuples
            level: Current hierarchy level

        Returns:
            List of parent clusters
        """
        # Sort by similarity (descending)
        merge_pairs.sort(key=lambda x: x[2], reverse=True)

        # Track which clusters have been merged
        merged_into = {}  # cluster_idx -> parent_idx
        parent_clusters = []
        next_cluster_id = max(c.cluster_id for c in clusters) + 1

        for idx_i, idx_j, similarity in merge_pairs:
            # Check if either cluster already merged
            if idx_i in merged_into or idx_j in merged_into:
                continue

            # Create parent cluster
            cluster_i = clusters[idx_i]
            cluster_j = clusters[idx_j]

            parent = Cluster(
                cluster_id=next_cluster_id,
                canonical_name=f"merged_{next_cluster_id}",
                members=cluster_i.members + cluster_j.members,
                parent_id=None,
                hierarchy_level=level - 1
            )

            parent_clusters.append(parent)
            merged_into[idx_i] = len(parent_clusters) - 1
            merged_into[idx_j] = len(parent_clusters) - 1

            next_cluster_id += 1

        # Add unmerged clusters as-is
        for idx, cluster in enumerate(clusters):
            if idx not in merged_into:
                parent_clusters.append(cluster)

        return parent_clusters

    def print_top_results(self, top_results: List[OptimizationResult]):
        """Print top configuration results."""
        print("\n" + "="*80)
        print(f"TOP {len(top_results)} CONFIGURATIONS")
        print("="*80)

        for rank, result in enumerate(top_results, 1):
            print(f"\nRank {rank}: Composite Score = {result.composite_score:.1f}/100")
            print(f"  Thresholds: L3→L2: {result.thresholds[3]:.2f}, " +
                  f"L2→L1: {result.thresholds[2]:.2f}, " +
                  f"L1→L0: {result.thresholds[1]:.2f}")

            m = result.metrics
            print(f"  Reduction: {m['reduction_ratio']:.1%} ({m['initial_count']}→{m['final_count']} clusters)")
            print(f"  Depth: {m['max_depth']} levels")
            print(f"  Coherence: {m['avg_coherence']:.2f}")
            print(f"  Separation: {m['avg_separation']:.2f}")
            print(f"  Execution: {result.execution_time:.2f}s")

        print("\n" + "="*80)
        print(f"SELECTED: Rank 1 configuration")
        print("="*80)

    def save_results(self, results: List[OptimizationResult]):
        """Save optimization results to file."""
        output_file = Path(self.config.results_dir) / f"hyperparameter_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        results_data = {
            'entity_type': self.entity_type,
            'total_configs_tested': len(results),
            'search_space': self.config.threshold_search_space,
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'rank': idx + 1,
                    'thresholds': r.thresholds,
                    'composite_score': r.composite_score,
                    'metrics': r.metrics,
                    'execution_time': r.execution_time
                }
                for idx, r in enumerate(results)
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    # Test the optimizer
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    optimizer = HyperparameterOptimizer(entity_type='mechanism')
    results = optimizer.optimize()

    if results:
        print(f"\n✓ Optimization successful!")
        print(f"  Best score: {results[0].composite_score:.1f}/100")
        print(f"  Best thresholds: {results[0].thresholds}")
