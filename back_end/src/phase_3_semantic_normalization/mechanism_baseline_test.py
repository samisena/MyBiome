"""
Baseline Clustering Test - Phase 2.1

Runs baseline clustering on raw mechanism texts to establish performance baseline.
This provides the comparison point for preprocessing experiments.

Test Procedure:
1. Load mechanisms from database
2. Run clustering with default config
3. Record baseline metrics
4. Save results for comparison

Usage:
    python mechanism_baseline_test.py --db-path <path>
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Import components
from .mechanism_normalizer import MechanismNormalizer
from .mechanism_db_manager import MechanismDatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineTest:
    """
    Baseline clustering test runner.

    Establishes performance baseline for raw mechanism texts.
    """

    def __init__(
        self,
        db_path: str,
        cache_dir: Optional[str] = None,
        results_dir: Optional[str] = None
    ):
        """
        Initialize baseline test.

        Args:
            db_path: Path to intervention_research.db
            cache_dir: Cache directory for embeddings and LLM decisions
            results_dir: Directory for test results
        """
        self.db_path = db_path

        # Setup directories
        if cache_dir is None:
            from .config import CACHE_DIR
            cache_dir = CACHE_DIR

        if results_dir is None:
            from .config import RESULTS_DIR
            results_dir = RESULTS_DIR
        else:
            results_dir = Path(results_dir)

        results_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = cache_dir
        self.results_dir = results_dir

        logger.info("BaselineTest initialized")

    def run_baseline(self) -> Dict[str, Any]:
        """
        Run baseline clustering test.

        Returns:
            Baseline results dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("BASELINE CLUSTERING TEST - PHASE 2.1")
        logger.info("="*80)

        start_time = datetime.now()

        # Step 1: Load mechanisms
        logger.info("\nStep 1: Loading mechanisms from database...")
        normalizer = MechanismNormalizer(
            db_path=self.db_path,
            cache_dir=self.cache_dir,
            # Default hyperparameters (from implementation plan)
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_epsilon=0.0
        )

        mechanisms = normalizer.load_mechanisms_from_db()

        if not mechanisms:
            logger.error("No mechanisms found in database")
            return {
                'success': False,
                'error': 'No mechanisms in database'
            }

        logger.info(f"  Loaded {len(mechanisms)} unique mechanisms")

        # Step 2: Run clustering
        logger.info("\nStep 2: Running HDBSCAN clustering (baseline)...")
        clustering_result = normalizer.cluster_all_mechanisms(mechanisms=mechanisms)

        if not clustering_result.success:
            logger.error(f"Clustering failed: {clustering_result.error}")
            return {
                'success': False,
                'error': clustering_result.error
            }

        # Step 3: Record metrics
        logger.info("\nStep 3: Recording baseline metrics...")

        baseline_results = {
            'success': True,
            'test_type': 'baseline',
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_seconds': (datetime.now() - start_time).total_seconds(),

            # Dataset info
            'total_mechanisms': len(mechanisms),

            # Hyperparameters
            'hyperparameters': {
                'min_cluster_size': normalizer.min_cluster_size,
                'min_samples': normalizer.min_samples,
                'cluster_selection_epsilon': normalizer.cluster_selection_epsilon,
                'primary_threshold': normalizer.primary_threshold,
                'secondary_threshold': normalizer.secondary_threshold
            },

            # Clustering results
            'num_clusters': clustering_result.num_clusters,

            # Metrics
            'metrics': {
                'silhouette_score': clustering_result.metrics.silhouette_score,
                'davies_bouldin_index': clustering_result.metrics.davies_bouldin_index,
                'singleton_count': clustering_result.metrics.singleton_count,
                'singleton_percentage': clustering_result.metrics.singleton_percentage,
                'avg_cluster_size': clustering_result.metrics.avg_cluster_size
            },

            # Thresholds
            'passed_thresholds': clustering_result.metrics.passes_thresholds(),

            # Summary
            'summary': clustering_result.metrics.summary()
        }

        # Step 4: Save results
        logger.info("\nStep 4: Saving baseline results...")
        self._save_results(baseline_results)

        # Step 5: Print summary
        self._print_summary(baseline_results)

        return baseline_results

    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"baseline_clustering_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"  Results saved to: {results_file}")

    def _print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        logger.info("\n" + "="*80)
        logger.info("BASELINE CLUSTERING SUMMARY")
        logger.info("="*80)

        logger.info(f"\nDataset:")
        logger.info(f"  Total mechanisms: {results['total_mechanisms']}")

        logger.info(f"\nHyperparameters:")
        hp = results['hyperparameters']
        logger.info(f"  min_cluster_size: {hp['min_cluster_size']}")
        logger.info(f"  min_samples: {hp['min_samples']}")
        logger.info(f"  cluster_selection_epsilon: {hp['cluster_selection_epsilon']}")

        logger.info(f"\nClustering Results:")
        logger.info(f"  Clusters: {results['num_clusters']}")

        logger.info(f"\nMetrics:")
        m = results['metrics']
        logger.info(f"  Silhouette: {m['silhouette_score']:.3f}")
        logger.info(f"  Davies-Bouldin: {m['davies_bouldin_index']:.3f}")
        logger.info(f"  Singletons: {m['singleton_count']} ({m['singleton_percentage']:.1%})")
        logger.info(f"  Avg cluster size: {m['avg_cluster_size']:.1f}")

        logger.info(f"\nValidation:")
        logger.info(f"  Passed thresholds: {'✓ YES' if results['passed_thresholds'] else '✗ NO'}")
        logger.info(f"  Summary: {results['summary']}")

        logger.info(f"\nTime:")
        logger.info(f"  Elapsed: {results['elapsed_time_seconds']:.1f}s")

        logger.info("\n" + "="*80)


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Baseline Clustering Test (Phase 2.1)")
    parser.add_argument('--db-path', required=True, help='Path to intervention_research.db')
    parser.add_argument('--cache-dir', help='Cache directory (default: auto)')
    parser.add_argument('--results-dir', help='Results directory (default: auto)')

    args = parser.parse_args()

    # Run test
    test = BaselineTest(
        db_path=args.db_path,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir
    )

    results = test.run_baseline()

    # Exit code
    exit_code = 0 if results.get('success') else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
