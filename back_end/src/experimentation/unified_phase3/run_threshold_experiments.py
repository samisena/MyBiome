"""
Distance Threshold Experiment Runner

Runs hierarchical clustering experiments with different distance_threshold values
(0.4, 0.5, 0.6, 0.7, 0.8) for interventions, conditions, and mechanisms.

Tests which threshold produces the best cluster quality based on:
- Silhouette score (higher is better)
- Davies-Bouldin score (lower is better)
- Cluster size distribution
- Manual inspection of cluster members

Usage:
    python run_threshold_experiments.py \\
        --db back_end/data/processed/intervention_research.db \\
        --thresholds 0.7 0.8 \\
        --entity-types interventions conditions mechanisms
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experimentation.unified_phase3.experiment_runner import ExperimentRunner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run distance threshold optimization experiments'
    )
    parser.add_argument(
        '--db',
        required=True,
        help='Path to intervention_research.db'
    )
    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        default=[0.4, 0.5, 0.6, 0.7],
        help='Distance thresholds to test (default: 0.4 0.5 0.6 0.7)'
    )
    parser.add_argument(
        '--entity-types',
        nargs='+',
        choices=['interventions', 'conditions', 'mechanisms'],
        default=['interventions', 'conditions', 'mechanisms'],
        help='Entity types to test (default: all three)'
    )
    parser.add_argument(
        '--exp-db',
        help='Path to experiment database (default: unified_phase3/experiment_results.db)'
    )
    parser.add_argument(
        '--cache-dir',
        help='Cache directory (default: from base config)'
    )

    args = parser.parse_args()

    # Build list of config paths
    config_dir = Path(__file__).parent / 'config' / 'experiment_configs'
    config_paths = []

    # Map entity types to experiment number ranges
    entity_ranges = {
        'interventions': (10, 13),  # exp_010-013 (0.4-0.7), exp_022 (0.8)
        'conditions': (14, 17),      # exp_014-017 (0.4-0.7), exp_023 (0.8)
        'mechanisms': (18, 21)       # exp_018-021 (0.4-0.7), exp_024 (0.8)
    }

    # Threshold to experiment mapping
    threshold_offsets = {
        0.4: 0,
        0.5: 1,
        0.6: 2,
        0.7: 3,
        0.8: 12  # Special offset for 0.8 (exp_022-024)
    }

    # Special mapping for 0.8 threshold
    entity_exp_0_8 = {
        'interventions': 22,
        'conditions': 23,
        'mechanisms': 24
    }

    for entity_type in args.entity_types:
        start_exp, end_exp = entity_ranges[entity_type]

        for threshold in args.thresholds:
            if threshold not in threshold_offsets:
                logger.warning(f"Skipping invalid threshold: {threshold}")
                continue

            # Special handling for threshold 0.8
            if threshold == 0.8:
                exp_num = entity_exp_0_8[entity_type]
            else:
                exp_num = start_exp + threshold_offsets[threshold]

            config_name = f"exp_{exp_num:03d}_threshold_{threshold}_{entity_type}.yaml"
            config_path = config_dir / config_name

            if config_path.exists():
                config_paths.append(str(config_path))
            else:
                logger.warning(f"Config not found: {config_path}")

    if not config_paths:
        logger.error("No valid config files found!")
        return 1

    logger.info(f"Running {len(config_paths)} threshold experiments...")
    logger.info(f"Thresholds: {args.thresholds}")
    logger.info(f"Entity types: {args.entity_types}")

    # Set default experiment DB path if not provided
    exp_db = args.exp_db or str(Path(__file__).parent / 'experiment_results.db')

    # Create experiment runner
    runner = ExperimentRunner(
        config_paths=config_paths,
        db_path=args.db,
        experiment_db_path=exp_db,
        cache_dir=args.cache_dir
    )

    # Run all experiments
    results = runner.run_all(sequential=True)

    # Summary
    successful = sum(1 for r in results if r['success'])
    logger.info("")
    logger.info("="*80)
    logger.info(f"THRESHOLD EXPERIMENTS COMPLETED: {successful}/{len(results)} successful")
    logger.info("="*80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run: python analyze_threshold_results.py --exp-db experiment_results.db")
    logger.info("  2. Review cluster member lists for quality assessment")
    logger.info("  3. Select optimal thresholds per entity type")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
