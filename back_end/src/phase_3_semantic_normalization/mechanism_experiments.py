"""
Hyperparameter Experimentation Framework for Mechanism Clustering

Implements grid search over HDBSCAN parameters with comprehensive validation:
- Multiple hyperparameter configurations
- Clustering quality metrics (silhouette, Davies-Bouldin)
- Manual coherence sampling
- Quick validation checkpoints
- Experiment logging and comparison

Usage:
    from mechanism_experiments import MechanismExperimentRunner

    runner = MechanismExperimentRunner(db_path=config.db_path)

    # Run grid search
    best_config = runner.run_grid_search()

    # Run single experiment
    result = runner.run_experiment(min_cluster_size=5, min_samples=3)
"""

import os
import json
import logging
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Import mechanism normalizer
from .mechanism_normalizer import MechanismNormalizer, ClusterMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    experiment_id: str
    min_cluster_size: int
    min_samples: int
    cluster_selection_epsilon: float
    primary_threshold: float = 0.75
    secondary_threshold: float = 0.60


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    experiment_id: str
    config: ExperimentConfig
    metrics: ClusterMetrics
    manual_coherence: Optional[float] = None
    cluster_count: int = 0
    mechanism_count: int = 0
    elapsed_time_seconds: float = 0.0
    passed_thresholds: bool = False
    notes: str = ""

    def summary(self) -> str:
        """Generate 1-line experiment summary."""
        status = "✓ PASS" if self.passed_thresholds else "✗ FAIL"
        return (f"{status} | {self.experiment_id} | "
                f"sil={self.metrics.silhouette_score:.2f}, "
                f"clusters={self.cluster_count}, "
                f"singleton={self.metrics.singleton_percentage:.1%}, "
                f"coherence={self.manual_coherence or 0:.1f}/5")


class MechanismExperimentRunner:
    """
    Experiment runner for hyperparameter tuning.

    Runs grid search over HDBSCAN parameters and evaluates clustering quality.
    """

    def __init__(
        self,
        db_path: str,
        cache_dir: Optional[str] = None,
        results_dir: Optional[str] = None
    ):
        """
        Initialize experiment runner.

        Args:
            db_path: Path to intervention_research.db
            cache_dir: Cache directory for embeddings and LLM decisions
            results_dir: Directory for experiment results
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

        # Experiment tracking
        self.experiments: List[ExperimentResult] = []
        self.best_config: Optional[ExperimentConfig] = None

        logger.info("MechanismExperimentRunner initialized")

    def generate_experiment_configs(self) -> List[ExperimentConfig]:
        """
        Generate grid of experiment configurations.

        Returns:
            List of ExperimentConfig objects
        """
        configs = []

        # Grid search parameters (from implementation plan)
        min_cluster_sizes = [3, 5, 7, 10]
        min_samples_list = [2, 3, 5]
        cluster_selection_epsilons = [0.0, 0.05, 0.10]

        experiment_id = 1
        for mcs in min_cluster_sizes:
            for ms in min_samples_list:
                for eps in cluster_selection_epsilons:
                    config = ExperimentConfig(
                        experiment_id=f"exp_{experiment_id:02d}",
                        min_cluster_size=mcs,
                        min_samples=ms,
                        cluster_selection_epsilon=eps
                    )
                    configs.append(config)
                    experiment_id += 1

        logger.info(f"Generated {len(configs)} experiment configurations")
        return configs

    def run_experiment(
        self,
        config: ExperimentConfig,
        mechanisms: Optional[List[str]] = None,
        sample_coherence: bool = True,
        sample_size: int = 5
    ) -> ExperimentResult:
        """
        Run a single experiment with given configuration.

        Args:
            config: Experiment configuration
            mechanisms: Optional list of mechanisms (loads from DB if None)
            sample_coherence: Whether to manually review sample clusters
            sample_size: Number of clusters to sample for manual review

        Returns:
            ExperimentResult with metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {config.experiment_id}")
        logger.info(f"{'='*60}")
        logger.info(f"  min_cluster_size: {config.min_cluster_size}")
        logger.info(f"  min_samples: {config.min_samples}")
        logger.info(f"  cluster_selection_epsilon: {config.cluster_selection_epsilon}")

        start_time = datetime.now()

        try:
            # Create normalizer with experiment config
            normalizer = MechanismNormalizer(
                db_path=self.db_path,
                cache_dir=self.cache_dir,
                min_cluster_size=config.min_cluster_size,
                min_samples=config.min_samples,
                cluster_selection_epsilon=config.cluster_selection_epsilon,
                primary_threshold=config.primary_threshold,
                secondary_threshold=config.secondary_threshold
            )

            # Run clustering
            clustering_result = normalizer.cluster_all_mechanisms(mechanisms=mechanisms)

            if not clustering_result.success:
                return ExperimentResult(
                    experiment_id=config.experiment_id,
                    config=config,
                    metrics=clustering_result.metrics,
                    passed_thresholds=False,
                    notes=f"Clustering failed: {clustering_result.error}"
                )

            # Manual coherence review (if requested)
            manual_coherence = None
            if sample_coherence and clustering_result.num_clusters > 0:
                manual_coherence = self._manual_coherence_review(
                    clustering_result.cluster_members,
                    sample_size=sample_size
                )

            # Check if thresholds passed
            passed_thresholds = clustering_result.metrics.passes_thresholds()
            if manual_coherence is not None:
                passed_thresholds = passed_thresholds and manual_coherence >= 4.0

            elapsed_time = (datetime.now() - start_time).total_seconds()

            result = ExperimentResult(
                experiment_id=config.experiment_id,
                config=config,
                metrics=clustering_result.metrics,
                manual_coherence=manual_coherence,
                cluster_count=clustering_result.num_clusters,
                mechanism_count=clustering_result.num_mechanisms,
                elapsed_time_seconds=elapsed_time,
                passed_thresholds=passed_thresholds
            )

            # Log summary
            logger.info(f"\n{result.summary()}")

            # Store result
            self.experiments.append(result)

            return result

        except Exception as e:
            logger.error(f"Experiment {config.experiment_id} failed: {e}")
            return ExperimentResult(
                experiment_id=config.experiment_id,
                config=config,
                metrics=ClusterMetrics(0, 0, 0, 0, 0, 0),
                passed_thresholds=False,
                notes=f"Exception: {str(e)}"
            )

    def _manual_coherence_review(
        self,
        cluster_members: Dict[int, List[str]],
        sample_size: int = 5
    ) -> float:
        """
        Manually review cluster coherence (sample clusters).

        For automation purposes, this uses heuristics. In production,
        this would involve human review.

        Args:
            cluster_members: Dict mapping cluster_id to list of members
            sample_size: Number of clusters to sample

        Returns:
            Average coherence score (1.0-5.0)
        """
        # Filter out singleton cluster (-1) and empty clusters
        valid_clusters = {cid: members for cid, members in cluster_members.items()
                         if cid != -1 and len(members) > 1}

        if not valid_clusters:
            logger.warning("No valid clusters for coherence review")
            return 3.0  # Neutral score

        # Sample clusters
        sample_clusters = random.sample(
            list(valid_clusters.items()),
            min(sample_size, len(valid_clusters))
        )

        coherence_scores = []

        logger.info(f"\nManual Coherence Review (sampling {len(sample_clusters)} clusters):")

        for cluster_id, members in sample_clusters:
            # Heuristic coherence score based on:
            # 1. Cluster size (optimal: 3-10 members)
            # 2. Text similarity (measure avg length similarity as proxy)

            size_score = self._heuristic_size_score(len(members))
            similarity_score = self._heuristic_similarity_score(members)

            coherence = (size_score + similarity_score) / 2.0
            coherence_scores.append(coherence)

            # Log sample
            logger.info(f"  Cluster {cluster_id} ({len(members)} members):")
            for i, member in enumerate(members[:3]):  # Show first 3
                logger.info(f"    - {member[:60]}...")
            if len(members) > 3:
                logger.info(f"    ... and {len(members)-3} more")
            logger.info(f"    Coherence: {coherence:.1f}/5.0")

        avg_coherence = np.mean(coherence_scores)
        logger.info(f"\nAverage Manual Coherence: {avg_coherence:.1f}/5.0")

        return avg_coherence

    def _heuristic_size_score(self, size: int) -> float:
        """Heuristic score based on cluster size (optimal: 3-10)."""
        if size < 2:
            return 2.0
        elif size <= 3:
            return 4.0
        elif size <= 10:
            return 5.0
        elif size <= 20:
            return 4.0
        else:
            return 3.0  # Too large, may need sub-clustering

    def _heuristic_similarity_score(self, members: List[str]) -> float:
        """Heuristic score based on text length similarity."""
        if len(members) < 2:
            return 3.0

        lengths = [len(m) for m in members]
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)

        # Low variation in length suggests similar mechanisms
        cv = std_length / avg_length if avg_length > 0 else 1.0

        if cv < 0.3:
            return 5.0
        elif cv < 0.5:
            return 4.0
        elif cv < 0.7:
            return 3.0
        else:
            return 2.0

    def run_grid_search(
        self,
        mechanisms: Optional[List[str]] = None,
        sample_coherence: bool = True,
        save_results: bool = True
    ) -> Optional[ExperimentConfig]:
        """
        Run grid search over all experiment configurations.

        Args:
            mechanisms: Optional list of mechanisms (loads from DB if None)
            sample_coherence: Whether to manually review sample clusters
            save_results: Whether to save results to file

        Returns:
            Best configuration (or None if all failed)
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING GRID SEARCH")
        logger.info("="*60)

        # Generate configs
        configs = self.generate_experiment_configs()

        # Run experiments
        for config in configs:
            self.run_experiment(
                config=config,
                mechanisms=mechanisms,
                sample_coherence=sample_coherence
            )

        # Find best configuration
        self.best_config = self._select_best_config()

        # Save results
        if save_results:
            self._save_results()

        # Print summary
        self._print_summary()

        return self.best_config

    def _select_best_config(self) -> Optional[ExperimentConfig]:
        """
        Select best configuration based on multiple criteria.

        Scoring:
        - Silhouette score (weight: 0.4)
        - Manual coherence (weight: 0.3)
        - Singleton percentage (weight: 0.2, inverted)
        - Cluster count in range (weight: 0.1, bonus for 15-30)

        Returns:
            Best ExperimentConfig
        """
        if not self.experiments:
            return None

        # Filter passing experiments
        passing = [exp for exp in self.experiments if exp.passed_thresholds]

        if not passing:
            logger.warning("No experiments passed thresholds, selecting best from all")
            passing = self.experiments

        # Score each experiment
        scores = []
        for exp in passing:
            # Silhouette (0.0-1.0 → 0-40 points)
            silhouette_points = exp.metrics.silhouette_score * 40

            # Manual coherence (1.0-5.0 → 0-30 points)
            coherence_points = (exp.manual_coherence or 3.0) / 5.0 * 30

            # Singleton percentage (0.0-1.0 → 20-0 points, inverted)
            singleton_points = (1.0 - exp.metrics.singleton_percentage) * 20

            # Cluster count bonus (15-30 clusters → 10 points)
            cluster_bonus = 10 if 15 <= exp.cluster_count <= 30 else 0

            total_score = silhouette_points + coherence_points + singleton_points + cluster_bonus

            scores.append((exp, total_score))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)

        best_exp, best_score = scores[0]

        logger.info(f"\nBest configuration: {best_exp.experiment_id} (score: {best_score:.1f})")
        logger.info(f"  min_cluster_size: {best_exp.config.min_cluster_size}")
        logger.info(f"  min_samples: {best_exp.config.min_samples}")
        logger.info(f"  cluster_selection_epsilon: {best_exp.config.cluster_selection_epsilon}")

        return best_exp.config

    def _save_results(self):
        """Save experiment results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"mechanism_experiments_{timestamp}.json"

        results_data = {
            'timestamp': timestamp,
            'total_experiments': len(self.experiments),
            'experiments': [
                {
                    'experiment_id': exp.experiment_id,
                    'config': asdict(exp.config),
                    'metrics': {
                        'silhouette_score': exp.metrics.silhouette_score,
                        'davies_bouldin_index': exp.metrics.davies_bouldin_index,
                        'num_clusters': exp.metrics.num_clusters,
                        'singleton_count': exp.metrics.singleton_count,
                        'singleton_percentage': exp.metrics.singleton_percentage,
                        'avg_cluster_size': exp.metrics.avg_cluster_size
                    },
                    'manual_coherence': exp.manual_coherence,
                    'cluster_count': exp.cluster_count,
                    'mechanism_count': exp.mechanism_count,
                    'elapsed_time_seconds': exp.elapsed_time_seconds,
                    'passed_thresholds': exp.passed_thresholds,
                    'notes': exp.notes
                }
                for exp in self.experiments
            ],
            'best_config': asdict(self.best_config) if self.best_config else None
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to: {results_file}")

    def _print_summary(self):
        """Print experiment summary table."""
        logger.info("\n" + "="*80)
        logger.info("GRID SEARCH SUMMARY")
        logger.info("="*80)

        # Header
        logger.info(f"{'ID':<8} {'mcs':<5} {'ms':<5} {'eps':<6} {'sil':<6} {'clust':<7} {'sing%':<7} {'coh':<5} {'pass':<5}")
        logger.info("-"*80)

        # Results
        for exp in self.experiments:
            logger.info(
                f"{exp.experiment_id:<8} "
                f"{exp.config.min_cluster_size:<5} "
                f"{exp.config.min_samples:<5} "
                f"{exp.config.cluster_selection_epsilon:<6.2f} "
                f"{exp.metrics.silhouette_score:<6.3f} "
                f"{exp.cluster_count:<7} "
                f"{exp.metrics.singleton_percentage*100:<6.1f}% "
                f"{exp.manual_coherence or 0:<5.1f} "
                f"{'✓' if exp.passed_thresholds else '✗':<5}"
            )

        logger.info("="*80)

        # Best config
        if self.best_config:
            logger.info(f"\nBest Config: {self.best_config.experiment_id}")
            logger.info(f"  Parameters: mcs={self.best_config.min_cluster_size}, "
                       f"ms={self.best_config.min_samples}, "
                       f"eps={self.best_config.cluster_selection_epsilon}")


def main():
    """Command-line interface for experiment runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Mechanism Clustering Hyperparameter Experiments")
    parser.add_argument('--db-path', required=True, help='Path to intervention_research.db')
    parser.add_argument('--cache-dir', help='Cache directory (default: auto)')
    parser.add_argument('--results-dir', help='Results directory (default: auto)')
    parser.add_argument('--grid-search', action='store_true', help='Run full grid search')
    parser.add_argument('--single', action='store_true', help='Run single experiment')
    parser.add_argument('--min-cluster-size', type=int, default=5)
    parser.add_argument('--min-samples', type=int, default=3)
    parser.add_argument('--cluster-selection-epsilon', type=float, default=0.0)

    args = parser.parse_args()

    runner = MechanismExperimentRunner(
        db_path=args.db_path,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir
    )

    if args.grid_search:
        best_config = runner.run_grid_search()
        if best_config:
            print(f"\nBest Configuration:")
            print(f"  min_cluster_size: {best_config.min_cluster_size}")
            print(f"  min_samples: {best_config.min_samples}")
            print(f"  cluster_selection_epsilon: {best_config.cluster_selection_epsilon}")

    elif args.single:
        config = ExperimentConfig(
            experiment_id="single_exp",
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon
        )

        result = runner.run_experiment(config)
        print(f"\n{result.summary()}")


if __name__ == "__main__":
    main()
