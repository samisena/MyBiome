"""
Evaluation Module - Compare Experiments and Analyze Results

Provides tools for:
1. Comparing experiments across different configurations
2. Analyzing temperature effects on naming quality
3. Computing consistency metrics
4. Selecting optimal configuration
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemperatureMetrics:
    """Metrics for a specific temperature."""
    temperature: float
    num_experiments: int
    avg_clusters: float
    avg_singletons: float
    avg_silhouette: float
    avg_failures: float
    avg_cache_hit_rate: float
    consistency_score: Optional[float] = None


class ExperimentEvaluator:
    """
    Evaluate and compare experiment results.

    Features:
    - Temperature comparison
    - Naming consistency analysis
    - Quality metrics computation
    - Best configuration selection
    """

    def __init__(self, experiment_db_path: str):
        """
        Initialize evaluator.

        Args:
            experiment_db_path: Path to experiment database
        """
        self.experiment_db_path = Path(experiment_db_path)

        if not self.experiment_db_path.exists():
            raise FileNotFoundError(f"Experiment database not found: {self.experiment_db_path}")

        logger.info(f"ExperimentEvaluator initialized with database: {self.experiment_db_path}")

    def get_all_experiments(self, status: Optional[str] = 'completed') -> List[Dict]:
        """
        Get all experiments from database.

        Args:
            status: Filter by status (completed, failed, running, pending)

        Returns:
            List of experiment dicts
        """
        conn = sqlite3.connect(self.experiment_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if status:
            cursor.execute("""
                SELECT * FROM experiments
                WHERE status = ?
                ORDER BY created_at DESC
            """, (status,))
        else:
            cursor.execute("""
                SELECT * FROM experiments
                ORDER BY created_at DESC
            """)

        experiments = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return experiments

    def compare_temperatures(
        self,
        entity_type: str = 'intervention',
        temperatures: Optional[List[float]] = None
    ) -> Dict[float, TemperatureMetrics]:
        """
        Compare results across different temperatures.

        Args:
            entity_type: Entity type to analyze
            temperatures: Specific temperatures to compare (None = all)

        Returns:
            Dict mapping temperature to TemperatureMetrics
        """
        logger.info(f"Comparing temperatures for {entity_type}s...")

        conn = sqlite3.connect(self.experiment_db_path)
        cursor = conn.cursor()

        # Build query
        query = """
            SELECT
                e.naming_temperature,
                COUNT(*) as num_experiments,
                AVG(er.num_clusters) as avg_clusters,
                AVG(er.num_singleton_clusters) as avg_singletons,
                AVG(er.silhouette_score) as avg_silhouette,
                AVG(er.naming_failures) as avg_failures,
                AVG(er.naming_cache_hit_rate) as avg_cache_hit_rate
            FROM experiments e
            JOIN experiment_results er ON e.experiment_id = er.experiment_id
            WHERE e.status = 'completed' AND er.entity_type = ?
        """

        params = [entity_type]

        if temperatures:
            query += f" AND e.naming_temperature IN ({','.join('?'*len(temperatures))})"
            params.extend(temperatures)

        query += " GROUP BY e.naming_temperature ORDER BY e.naming_temperature"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = {}
        for row in rows:
            temp = row[0]
            results[temp] = TemperatureMetrics(
                temperature=temp,
                num_experiments=row[1],
                avg_clusters=row[2],
                avg_singletons=row[3],
                avg_silhouette=row[4] if row[4] is not None else 0.0,
                avg_failures=row[5],
                avg_cache_hit_rate=row[6]
            )

        return results

    def compute_naming_consistency(
        self,
        experiment_ids: List[int],
        entity_type: str = 'intervention'
    ) -> Dict[str, Any]:
        """
        Compute naming consistency across experiments.

        Measures how consistently the same cluster gets the same name
        across different experiment runs (e.g., different temperatures).

        Args:
            experiment_ids: List of experiment IDs to compare
            entity_type: Entity type to analyze

        Returns:
            Dict with consistency metrics
        """
        logger.info(f"Computing naming consistency for {len(experiment_ids)} experiments...")

        conn = sqlite3.connect(self.experiment_db_path)
        cursor = conn.cursor()

        # Get cluster details for each experiment
        cluster_names = defaultdict(lambda: defaultdict(str))

        for exp_id in experiment_ids:
            cursor.execute("""
                SELECT cluster_id, canonical_name, member_entities
                FROM cluster_details
                WHERE experiment_id = ? AND entity_type = ?
            """, (exp_id, entity_type))

            for cluster_id, canonical_name, members_json in cursor.fetchall():
                # Use sorted members as cluster key
                members = tuple(sorted(json.loads(members_json)))
                cluster_names[members][exp_id] = canonical_name

        conn.close()

        # Analyze consistency
        consistency_scores = []
        cluster_analysis = []

        for members, exp_names in cluster_names.items():
            # Only analyze clusters present in multiple experiments
            if len(exp_names) < 2:
                continue

            # Get all names for this cluster
            names = list(exp_names.values())
            unique_names = set(names)

            # Calculate Levenshtein distance between all pairs
            distances = []
            if len(unique_names) > 1:
                import Levenshtein  # pip install python-Levenshtein
                name_list = list(unique_names)
                for i in range(len(name_list)):
                    for j in range(i + 1, len(name_list)):
                        dist = Levenshtein.distance(name_list[i], name_list[j])
                        distances.append(dist)

            avg_distance = np.mean(distances) if distances else 0.0

            # Consistency score: 1.0 if identical, decreases with distance
            consistency = 1.0 / (1.0 + avg_distance)
            consistency_scores.append(consistency)

            cluster_analysis.append({
                'member_count': len(members),
                'members_preview': list(members)[:3],
                'num_experiments': len(exp_names),
                'unique_names': list(unique_names),
                'avg_levenshtein_distance': avg_distance,
                'consistency_score': consistency
            })

        # Overall metrics
        results = {
            'entity_type': entity_type,
            'num_experiments': len(experiment_ids),
            'num_clusters_analyzed': len(cluster_analysis),
            'overall_consistency': np.mean(consistency_scores) if consistency_scores else 0.0,
            'consistency_std': np.std(consistency_scores) if consistency_scores else 0.0,
            'min_consistency': min(consistency_scores) if consistency_scores else 0.0,
            'max_consistency': max(consistency_scores) if consistency_scores else 0.0,
            'clusters': cluster_analysis[:20]  # Top 20 for reporting
        }

        logger.info(f"Overall consistency: {results['overall_consistency']:.2%}")
        logger.info(f"Analyzed {results['num_clusters_analyzed']} clusters")

        return results

    def select_optimal_temperature(
        self,
        entity_type: str = 'intervention',
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Select optimal temperature based on multiple metrics.

        Args:
            entity_type: Entity type to optimize for
            weights: Metric weights (silhouette, failures, consistency)

        Returns:
            Tuple of (optimal_temperature, analysis)
        """
        logger.info(f"Selecting optimal temperature for {entity_type}s...")

        # Default weights
        if weights is None:
            weights = {
                'silhouette': 0.4,  # Clustering quality
                'failures': 0.3,    # Naming success
                'consistency': 0.3  # Naming consistency
            }

        # Get temperature metrics
        temp_metrics = self.compare_temperatures(entity_type)

        if not temp_metrics:
            logger.error("No temperature data available")
            return None, {}

        # Normalize metrics to 0-1 range
        temps = list(temp_metrics.keys())

        silhouettes = [temp_metrics[t].avg_silhouette for t in temps]
        failures = [temp_metrics[t].avg_failures for t in temps]

        # Normalize (handle division by zero)
        max_silhouette = max(silhouettes) if silhouettes and max(silhouettes) > 0 else 1.0
        max_failures = max(failures) if failures and max(failures) > 0 else 1.0

        scores = {}
        for temp in temps:
            m = temp_metrics[temp]

            # Silhouette: higher is better (normalize to 0-1)
            silhouette_score = m.avg_silhouette / max_silhouette if max_silhouette > 0 else 0.0

            # Failures: lower is better (invert and normalize)
            failure_score = 1.0 - (m.avg_failures / max_failures) if max_failures > 0 else 1.0

            # Consistency: get from naming consistency analysis (if available)
            consistency_score = m.consistency_score if m.consistency_score is not None else 0.5

            # Weighted score
            total_score = (
                weights['silhouette'] * silhouette_score +
                weights['failures'] * failure_score +
                weights['consistency'] * consistency_score
            )

            scores[temp] = {
                'silhouette_score': silhouette_score,
                'failure_score': failure_score,
                'consistency_score': consistency_score,
                'total_score': total_score,
                'metrics': m
            }

        # Select best temperature
        optimal_temp = max(scores.keys(), key=lambda t: scores[t]['total_score'])

        analysis = {
            'optimal_temperature': optimal_temp,
            'weights': weights,
            'scores': scores,
            'recommendation': {
                'temperature': optimal_temp,
                'total_score': scores[optimal_temp]['total_score'],
                'silhouette': temp_metrics[optimal_temp].avg_silhouette,
                'failures': temp_metrics[optimal_temp].avg_failures,
                'consistency': scores[optimal_temp]['consistency_score']
            }
        }

        logger.info(f"\nOptimal temperature: {optimal_temp:.1f}")
        logger.info(f"  - Total score: {scores[optimal_temp]['total_score']:.3f}")
        logger.info(f"  - Silhouette: {temp_metrics[optimal_temp].avg_silhouette:.3f}")
        logger.info(f"  - Failures: {temp_metrics[optimal_temp].avg_failures:.2f}")

        return optimal_temp, analysis

    def generate_comparison_report(
        self,
        entity_types: Optional[List[str]] = None,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.

        Args:
            entity_types: Entity types to include (None = all)
            output_path: Path to save JSON report (optional)

        Returns:
            Dict with full report
        """
        logger.info("Generating comparison report...")

        if entity_types is None:
            entity_types = ['intervention', 'condition', 'mechanism']

        report = {
            'generated_at': str(Path.ctime),
            'entity_types': {},
            'experiments': self.get_all_experiments('completed')
        }

        for entity_type in entity_types:
            # Temperature comparison
            temp_metrics = self.compare_temperatures(entity_type)

            # Optimal temperature
            optimal_temp, analysis = self.select_optimal_temperature(entity_type)

            report['entity_types'][entity_type] = {
                'temperature_metrics': {
                    str(k): {
                        'temperature': v.temperature,
                        'num_experiments': v.num_experiments,
                        'avg_clusters': v.avg_clusters,
                        'avg_singletons': v.avg_singletons,
                        'avg_silhouette': v.avg_silhouette,
                        'avg_failures': v.avg_failures
                    }
                    for k, v in temp_metrics.items()
                },
                'optimal_temperature': optimal_temp,
                'optimization_analysis': analysis
            }

        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")

        return report


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Unified Phase 3 experiments")
    parser.add_argument('--exp-db', required=True, help='Path to experiment database')
    parser.add_argument('--entity-type', default='intervention', help='Entity type to analyze')
    parser.add_argument('--report', help='Path to save JSON report')

    args = parser.parse_args()

    # Create evaluator
    evaluator = ExperimentEvaluator(args.exp_db)

    # Compare temperatures
    temp_metrics = evaluator.compare_temperatures(args.entity_type)

    print(f"\nTemperature Comparison ({args.entity_type}s):")
    print(f"{'Temp':<6} {'Experiments':<12} {'Clusters':<10} {'Silhouette':<12} {'Failures':<10}")
    print("-" * 60)
    for temp, metrics in temp_metrics.items():
        print(f"{temp:<6.1f} {metrics.num_experiments:<12} {metrics.avg_clusters:<10.1f} "
              f"{metrics.avg_silhouette:<12.3f} {metrics.avg_failures:<10.1f}")

    # Select optimal
    optimal_temp, analysis = evaluator.select_optimal_temperature(args.entity_type)

    print(f"\nOptimal temperature: {optimal_temp:.1f}")
    print(f"Recommendation score: {analysis['recommendation']['total_score']:.3f}")

    # Generate report
    if args.report:
        evaluator.generate_comparison_report(output_path=args.report)
