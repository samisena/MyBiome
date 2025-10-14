"""
Preprocessing Comparison Framework - Phase 2.2

Compares baseline clustering vs. preprocessing with hierarchical decomposition.

Test Procedure:
1. Load baseline results
2. Run preprocessing (selective hierarchical decomposition)
3. Re-cluster on enhanced dataset
4. Compare metrics (silhouette improvement)
5. Decision: Keep or revert to baseline

Decision Criteria:
- Silhouette improvement > 0.05 → Keep preprocessing
- Otherwise → Revert to baseline

Usage:
    python mechanism_preprocessing_comparison.py --db-path <path> --baseline-file <path>
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Import components
from .mechanism_normalizer import MechanismNormalizer
from .mechanism_preprocessor import MechanismPreprocessor
from .mechanism_baseline_test import BaselineTest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingComparison:
    """
    Preprocessing comparison framework.

    Compares baseline vs. preprocessing with hierarchical decomposition.
    """

    SILHOUETTE_IMPROVEMENT_THRESHOLD = 0.05  # 5% improvement required

    def __init__(
        self,
        db_path: str,
        cache_dir: Optional[str] = None,
        results_dir: Optional[str] = None
    ):
        """
        Initialize comparison framework.

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

        logger.info("PreprocessingComparison initialized")

    def run_comparison(
        self,
        baseline_file: Optional[str] = None,
        run_baseline_first: bool = True,
        max_decompositions: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run full comparison test.

        Args:
            baseline_file: Path to existing baseline results (optional)
            run_baseline_first: If True and baseline_file is None, run baseline first
            max_decompositions: Limit decompositions for testing

        Returns:
            Comparison results dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("PREPROCESSING COMPARISON TEST - PHASE 2.2")
        logger.info("="*80)

        start_time = datetime.now()

        # Step 1: Load or run baseline
        logger.info("\nStep 1: Loading baseline results...")
        baseline_results = self._load_baseline(baseline_file, run_baseline_first)

        if not baseline_results or not baseline_results.get('success'):
            logger.error("Failed to load baseline results")
            return {
                'success': False,
                'error': 'Baseline results not available'
            }

        baseline_silhouette = baseline_results['metrics']['silhouette_score']
        logger.info(f"  Baseline silhouette: {baseline_silhouette:.3f}")

        # Step 2: Load mechanisms
        logger.info("\nStep 2: Loading mechanisms from database...")
        normalizer = MechanismNormalizer(
            db_path=self.db_path,
            cache_dir=self.cache_dir,
            min_cluster_size=baseline_results['hyperparameters']['min_cluster_size'],
            min_samples=baseline_results['hyperparameters']['min_samples'],
            cluster_selection_epsilon=baseline_results['hyperparameters']['cluster_selection_epsilon']
        )

        mechanisms = normalizer.load_mechanisms_from_db()
        logger.info(f"  Loaded {len(mechanisms)} mechanisms")

        # Step 3: Run preprocessing
        logger.info("\nStep 3: Running selective hierarchical decomposition...")
        preprocessor = MechanismPreprocessor(cache_dir=self.cache_dir)

        preprocessing_result = preprocessor.preprocess_mechanisms(
            mechanisms,
            selective=True,
            max_decompositions=max_decompositions
        )

        logger.info(f"  Decomposed: {preprocessing_result.decomposed_count}")
        logger.info(f"  Success rate: {preprocessing_result.decomposition_success_rate:.1%}")

        # Step 4: Validate decomposition quality
        logger.info("\nStep 4: Validating decomposition quality...")
        decomposition_accuracy = preprocessor.validate_decomposition_quality(
            preprocessing_result.decompositions,
            sample_size=20
        )

        logger.info(f"  Decomposition accuracy: {decomposition_accuracy:.1%}")

        # Check if accuracy meets threshold
        if decomposition_accuracy < 0.85:
            logger.warning(f"  Decomposition accuracy ({decomposition_accuracy:.1%}) < 85% threshold")
            logger.warning("  Consider reverting to baseline")

        # Step 5: Generate enhanced dataset
        logger.info("\nStep 5: Generating enhanced dataset...")
        enhanced_mechanisms = preprocessor.generate_enhanced_dataset(
            mechanisms,
            preprocessing_result.decompositions,
            include_originals=True
        )

        # Step 6: Re-cluster on enhanced dataset
        logger.info("\nStep 6: Re-clustering on enhanced dataset...")
        enhanced_clustering_result = normalizer.cluster_all_mechanisms(mechanisms=enhanced_mechanisms)

        if not enhanced_clustering_result.success:
            logger.error(f"Enhanced clustering failed: {enhanced_clustering_result.error}")
            return {
                'success': False,
                'error': enhanced_clustering_result.error,
                'baseline_results': baseline_results,
                'preprocessing_result': preprocessing_result
            }

        enhanced_silhouette = enhanced_clustering_result.metrics.silhouette_score
        logger.info(f"  Enhanced silhouette: {enhanced_silhouette:.3f}")

        # Step 7: Compare metrics
        logger.info("\nStep 7: Comparing metrics...")
        silhouette_improvement = enhanced_silhouette - baseline_silhouette
        improvement_percentage = (silhouette_improvement / baseline_silhouette) * 100 if baseline_silhouette > 0 else 0

        logger.info(f"  Silhouette improvement: {silhouette_improvement:+.3f} ({improvement_percentage:+.1f}%)")

        # Decision
        should_keep_preprocessing = silhouette_improvement > self.SILHOUETTE_IMPROVEMENT_THRESHOLD
        decision = "KEEP PREPROCESSING" if should_keep_preprocessing else "REVERT TO BASELINE"

        logger.info(f"\n  Decision: {decision}")

        # Step 8: Compile results
        comparison_results = {
            'success': True,
            'test_type': 'preprocessing_comparison',
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_seconds': (datetime.now() - start_time).total_seconds(),

            # Baseline
            'baseline': {
                'silhouette': baseline_silhouette,
                'davies_bouldin': baseline_results['metrics']['davies_bouldin_index'],
                'num_clusters': baseline_results['num_clusters'],
                'singleton_percentage': baseline_results['metrics']['singleton_percentage']
            },

            # Preprocessing
            'preprocessing': {
                'total_mechanisms': preprocessing_result.total_mechanisms,
                'complex_count': preprocessing_result.complex_count,
                'decomposed_count': preprocessing_result.decomposed_count,
                'success_rate': preprocessing_result.decomposition_success_rate,
                'avg_children': preprocessing_result.avg_children_per_decomposition,
                'decomposition_accuracy': decomposition_accuracy
            },

            # Enhanced clustering
            'enhanced': {
                'dataset_size': len(enhanced_mechanisms),
                'silhouette': enhanced_silhouette,
                'davies_bouldin': enhanced_clustering_result.metrics.davies_bouldin_index,
                'num_clusters': enhanced_clustering_result.num_clusters,
                'singleton_percentage': enhanced_clustering_result.metrics.singleton_percentage
            },

            # Comparison
            'comparison': {
                'silhouette_improvement': silhouette_improvement,
                'improvement_percentage': improvement_percentage,
                'threshold': self.SILHOUETTE_IMPROVEMENT_THRESHOLD,
                'should_keep_preprocessing': should_keep_preprocessing,
                'decision': decision
            }
        }

        # Step 9: Save results
        logger.info("\nStep 9: Saving comparison results...")
        self._save_results(comparison_results)

        # Step 10: Print summary
        self._print_summary(comparison_results)

        return comparison_results

    def _load_baseline(
        self,
        baseline_file: Optional[str],
        run_baseline_first: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Load baseline results from file or run baseline test.

        Args:
            baseline_file: Path to existing baseline results
            run_baseline_first: If True, run baseline if file not provided

        Returns:
            Baseline results dictionary
        """
        if baseline_file and Path(baseline_file).exists():
            # Load from file
            with open(baseline_file, 'r') as f:
                baseline_results = json.load(f)
            logger.info(f"  Loaded baseline from: {baseline_file}")
            return baseline_results

        elif run_baseline_first:
            # Run baseline test
            logger.info("  Running baseline test...")
            baseline_test = BaselineTest(
                db_path=self.db_path,
                cache_dir=self.cache_dir,
                results_dir=self.results_dir
            )
            baseline_results = baseline_test.run_baseline()
            return baseline_results

        else:
            logger.error("  No baseline file provided and run_baseline_first=False")
            return None

    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"preprocessing_comparison_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"  Results saved to: {results_file}")

    def _print_summary(self, results: Dict[str, Any]):
        """Print comparison summary."""
        logger.info("\n" + "="*80)
        logger.info("PREPROCESSING COMPARISON SUMMARY")
        logger.info("="*80)

        # Baseline
        logger.info("\nBaseline Clustering:")
        b = results['baseline']
        logger.info(f"  Silhouette: {b['silhouette']:.3f}")
        logger.info(f"  Davies-Bouldin: {b['davies_bouldin']:.3f}")
        logger.info(f"  Clusters: {b['num_clusters']}")
        logger.info(f"  Singletons: {b['singleton_percentage']:.1%}")

        # Preprocessing
        logger.info("\nPreprocessing (Hierarchical Decomposition):")
        p = results['preprocessing']
        logger.info(f"  Complex mechanisms: {p['complex_count']}/{p['total_mechanisms']} ({p['complex_count']/p['total_mechanisms']:.1%})")
        logger.info(f"  Decomposed: {p['decomposed_count']}")
        logger.info(f"  Success rate: {p['success_rate']:.1%}")
        logger.info(f"  Avg children per decomposition: {p['avg_children']:.1f}")
        logger.info(f"  Decomposition accuracy: {p['decomposition_accuracy']:.1%}")

        # Enhanced clustering
        logger.info("\nEnhanced Clustering:")
        e = results['enhanced']
        logger.info(f"  Dataset size: {e['dataset_size']} (vs {results['preprocessing']['total_mechanisms']} baseline)")
        logger.info(f"  Silhouette: {e['silhouette']:.3f}")
        logger.info(f"  Davies-Bouldin: {e['davies_bouldin']:.3f}")
        logger.info(f"  Clusters: {e['num_clusters']}")
        logger.info(f"  Singletons: {e['singleton_percentage']:.1%}")

        # Comparison
        logger.info("\nComparison:")
        c = results['comparison']
        logger.info(f"  Silhouette improvement: {c['silhouette_improvement']:+.3f} ({c['improvement_percentage']:+.1f}%)")
        logger.info(f"  Threshold: {c['threshold']:.2f}")
        logger.info(f"  Decision: {c['decision']}")
        logger.info(f"  Keep preprocessing: {'✓ YES' if c['should_keep_preprocessing'] else '✗ NO'}")

        logger.info(f"\nTime:")
        logger.info(f"  Elapsed: {results['elapsed_time_seconds']:.1f}s")

        logger.info("\n" + "="*80)


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing Comparison Test (Phase 2.2)")
    parser.add_argument('--db-path', required=True, help='Path to intervention_research.db')
    parser.add_argument('--baseline-file', help='Path to baseline results JSON')
    parser.add_argument('--no-baseline', action='store_true', help='Do not run baseline if file missing')
    parser.add_argument('--max-decompositions', type=int, help='Limit decompositions (for testing)')
    parser.add_argument('--cache-dir', help='Cache directory (default: auto)')
    parser.add_argument('--results-dir', help='Results directory (default: auto)')

    args = parser.parse_args()

    # Run comparison
    comparison = PreprocessingComparison(
        db_path=args.db_path,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir
    )

    results = comparison.run_comparison(
        baseline_file=args.baseline_file,
        run_baseline_first=not args.no_baseline,
        max_decompositions=args.max_decompositions
    )

    # Exit code
    exit_code = 0 if results.get('success') else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
