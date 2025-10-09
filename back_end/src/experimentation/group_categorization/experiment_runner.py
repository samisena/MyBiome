"""
Experiment Runner for Group-Based Categorization

Runs controlled experiments to compare group-based vs individual categorization.
Measures: LLM calls, time, accuracy, coverage.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .group_categorizer import GroupBasedCategorizer
from .validation import validate_all

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs group-based categorization experiments.
    """

    def __init__(self, db_path: str, results_dir: str = "back_end/src/experimentation/group_categorization/results"):
        """
        Initialize experiment runner.

        Args:
            db_path: Path to SQLite database
            results_dir: Directory to save results
        """
        self.db_path = db_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ExperimentRunner initialized")
        logger.info(f"Database: {db_path}")
        logger.info(f"Results dir: {results_dir}")

    def get_database_stats(self) -> Dict:
        """Get current database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Interventions
        cursor.execute("SELECT COUNT(DISTINCT intervention_name) FROM interventions")
        stats['total_interventions'] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(DISTINCT intervention_name)
            FROM interventions
            WHERE intervention_category IS NOT NULL AND intervention_category != ''
        """)
        stats['categorized_interventions'] = cursor.fetchone()[0]

        # Canonical groups
        cursor.execute("SELECT COUNT(*) FROM canonical_groups WHERE entity_type = 'intervention'")
        stats['total_groups'] = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM canonical_groups
            WHERE entity_type = 'intervention'
            AND (layer_0_category IS NOT NULL AND layer_0_category != '')
        """)
        stats['categorized_groups'] = cursor.fetchone()[0]

        # Papers
        cursor.execute("SELECT COUNT(*) FROM papers")
        stats['total_papers'] = cursor.fetchone()[0]

        conn.close()

        return stats

    def run_full_experiment(self, batch_size: int = 20) -> Dict:
        """
        Run full group-based categorization experiment.

        Args:
            batch_size: Batch size for LLM calls

        Returns:
            Experiment results dict
        """
        logger.info("=" * 80)
        logger.info("STARTING GROUP-BASED CATEGORIZATION EXPERIMENT")
        logger.info("=" * 80)

        start_time = time.time()

        # Get initial stats
        initial_stats = self.get_database_stats()
        logger.info(f"\nInitial Database Stats:")
        logger.info(f"  Total interventions: {initial_stats['total_interventions']}")
        logger.info(f"  Categorized interventions: {initial_stats['categorized_interventions']}")
        logger.info(f"  Total groups: {initial_stats['total_groups']}")
        logger.info(f"  Categorized groups: {initial_stats['categorized_groups']}")

        # Step 1: Categorize groups
        logger.info("\n" + "-" * 80)
        logger.info("Step 1: Categorizing canonical groups")
        logger.info("-" * 80)

        categorizer = GroupBasedCategorizer(
            db_path=self.db_path,
            batch_size=batch_size,
            include_members=True,
            max_members_in_prompt=10
        )

        group_stats = categorizer.categorize_all_groups()
        logger.info(f"Group categorization: {group_stats}")

        # Step 2: Propagate to interventions
        logger.info("\n" + "-" * 80)
        logger.info("Step 2: Propagating categories to interventions")
        logger.info("-" * 80)

        propagate_stats = categorizer.propagate_to_interventions()
        logger.info(f"Propagation: {propagate_stats}")

        # Step 3: Handle orphans
        logger.info("\n" + "-" * 80)
        logger.info("Step 3: Categorizing orphan interventions")
        logger.info("-" * 80)

        orphan_stats = categorizer.categorize_orphan_interventions()
        logger.info(f"Orphan categorization: {orphan_stats}")

        # Step 4: Validation
        logger.info("\n" + "-" * 80)
        logger.info("Step 4: Validating results")
        logger.info("-" * 80)

        validation_results = validate_all(self.db_path)

        # Get final stats
        final_stats = self.get_database_stats()

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Compile results
        results = {
            'experiment_id': f"group_categorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_seconds': elapsed_time,
            'initial_stats': initial_stats,
            'final_stats': final_stats,
            'group_categorization': group_stats,
            'propagation': propagate_stats,
            'orphan_categorization': orphan_stats,
            'validation': validation_results,
            'performance': {
                'total_llm_calls': group_stats['llm_calls'] + orphan_stats.get('llm_calls', 0),
                'groups_per_call': batch_size,
                'estimated_individual_calls': initial_stats['total_interventions'] // batch_size,
                'reduction_rate': 1 - (group_stats['llm_calls'] / (initial_stats['total_interventions'] // batch_size)) if initial_stats['total_interventions'] > 0 else 0
            }
        }

        # Calculate efficiency metrics
        if initial_stats['total_interventions'] > 0:
            results['efficiency'] = {
                'interventions_per_group_call': initial_stats['total_interventions'] / max(group_stats['llm_calls'], 1),
                'coverage_rate': final_stats['categorized_interventions'] / initial_stats['total_interventions'],
                'orphan_rate': orphan_stats['total'] / initial_stats['total_interventions'],
                'time_per_intervention': elapsed_time / initial_stats['total_interventions']
            }

        # Save results
        results_file = self.results_dir / f"{results['experiment_id']}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"\nKey Metrics:")
        logger.info(f"  Total time: {elapsed_time:.1f} seconds")
        logger.info(f"  LLM calls: {results['performance']['total_llm_calls']}")
        logger.info(f"  Reduction vs individual: {results['performance']['reduction_rate']*100:.1f}%")
        logger.info(f"  Coverage: {validation_results['coverage']['coverage_rate']*100:.1f}%")
        logger.info(f"  Orphan rate: {results['efficiency']['orphan_rate']*100:.1f}%")
        logger.info(f"  Validation: {validation_results['all_passed'] and 'PASSED' or 'FAILED'}")

        return results

    def run_subset_experiment(self, subset_size: int = 200, random_seed: int = 42) -> Dict:
        """
        Run experiment on a subset of interventions.

        Args:
            subset_size: Number of interventions to test
            random_seed: Random seed for reproducibility

        Returns:
            Experiment results dict
        """
        logger.info(f"Running subset experiment (size={subset_size}, seed={random_seed})")

        # Create temporary database with subset
        # For now, just run on full database (simplified for initial experiment)
        # TODO: Implement subset sampling if needed

        return self.run_full_experiment()

    def compare_with_individual(self) -> Dict:
        """
        Compare group-based categorization with individual categorization.

        Returns:
            Comparison results dict
        """
        logger.info("Comparing group-based vs individual categorization")

        # This would require running Phase 2.5 (individual) first for comparison
        # For now, we'll rely on existing categories in the database

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get interventions with existing categories
        cursor.execute("""
            SELECT COUNT(DISTINCT intervention_name)
            FROM interventions
            WHERE intervention_category IS NOT NULL AND intervention_category != ''
        """)
        existing_count = cursor.fetchone()[0]

        # Get groups
        cursor.execute("SELECT COUNT(*) FROM canonical_groups WHERE entity_type = 'intervention'")
        group_count = cursor.fetchone()[0]

        conn.close()

        # Estimate LLM calls for individual approach
        individual_calls = existing_count // 20  # Assuming batch_size=20

        # Group approach calls (from last experiment)
        group_calls = group_count // 20

        comparison = {
            'individual_approach': {
                'total_items': existing_count,
                'estimated_llm_calls': individual_calls,
                'batch_size': 20
            },
            'group_approach': {
                'total_items': group_count,
                'estimated_llm_calls': group_calls,
                'batch_size': 20
            },
            'reduction': {
                'absolute': individual_calls - group_calls,
                'percentage': (1 - group_calls / individual_calls) * 100 if individual_calls > 0 else 0
            }
        }

        logger.info(f"\nComparison:")
        logger.info(f"  Individual: {existing_count} interventions → ~{individual_calls} LLM calls")
        logger.info(f"  Group-based: {group_count} groups → ~{group_calls} LLM calls")
        logger.info(f"  Reduction: {comparison['reduction']['percentage']:.1f}%")

        return comparison


def run_experiment(
    db_path: Optional[str] = None,
    subset_size: Optional[int] = None,
    batch_size: int = 20
) -> Dict:
    """
    Convenience function to run experiment.

    Args:
        db_path: Path to database (uses config default if None)
        subset_size: Run on subset (None = full database)
        batch_size: Batch size for LLM calls

    Returns:
        Experiment results dict
    """
    if db_path is None:
        from back_end.src.data.config import config
        db_path = config.db_path

    runner = ExperimentRunner(db_path)

    if subset_size:
        return runner.run_subset_experiment(subset_size)
    else:
        return runner.run_full_experiment(batch_size=batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run group-based categorization experiment")
    parser.add_argument("--subset", type=int, help="Run on subset of interventions")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for LLM calls")
    parser.add_argument("--db-path", type=str, help="Path to database")

    args = parser.parse_args()

    results = run_experiment(
        db_path=args.db_path,
        subset_size=args.subset,
        batch_size=args.batch_size
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(json.dumps(results, indent=2))
