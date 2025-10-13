"""
Experiment Runner - Execute Multiple Configurations and Compare Results

Runs multiple experiment configurations in sequence, tracks results,
and generates comparison reports. Supports temperature experimentation
and hyperparameter tuning.
"""

import json
import sqlite3
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from .orchestrator import UnifiedPhase3Orchestrator

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Run multiple experiments and compare results.

    Features:
    - Sequential or parallel execution
    - Progress tracking
    - Result comparison
    - Temperature analysis
    - Best configuration selection
    """

    def __init__(
        self,
        config_paths: List[str],
        db_path: str,
        experiment_db_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize experiment runner.

        Args:
            config_paths: List of paths to YAML configuration files
            db_path: Path to intervention_research.db
            experiment_db_path: Path to experiment database
            cache_dir: Cache directory for all experiments
        """
        self.config_paths = [Path(p) for p in config_paths]
        self.db_path = Path(db_path)
        self.experiment_db_path = Path(experiment_db_path) if experiment_db_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.results = []

        logger.info(f"ExperimentRunner initialized with {len(self.config_paths)} configurations")

    def run_all(self, sequential: bool = True) -> List[Dict[str, Any]]:
        """
        Run all experiments.

        Args:
            sequential: Run experiments sequentially (True) or in parallel (False)

        Returns:
            List of experiment results
        """
        logger.info("="*60)
        logger.info(f"RUNNING {len(self.config_paths)} EXPERIMENTS")
        logger.info("="*60)

        if sequential:
            return self._run_sequential()
        else:
            # TODO: Implement parallel execution
            logger.warning("Parallel execution not yet implemented, falling back to sequential")
            return self._run_sequential()

    def _run_sequential(self) -> List[Dict[str, Any]]:
        """Run experiments sequentially."""
        self.results = []

        for i, config_path in enumerate(self.config_paths):
            logger.info(f"\n{'='*60}")
            logger.info(f"EXPERIMENT {i+1}/{len(self.config_paths)}: {config_path.name}")
            logger.info(f"{'='*60}")

            try:
                # Create orchestrator
                orchestrator = UnifiedPhase3Orchestrator(
                    config_path=str(config_path),
                    db_path=str(self.db_path),
                    experiment_db_path=str(self.experiment_db_path) if self.experiment_db_path else None,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None
                )

                # Run experiment
                result = orchestrator.run()
                self.results.append(result)

                if result['success']:
                    logger.info(f"[SUCCESS] Experiment {i+1} completed in {result['duration_seconds']:.1f}s")
                else:
                    logger.error(f"[FAILED] Experiment {i+1}: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"[FAILED] Experiment {i+1} crashed: {e}", exc_info=True)
                self.results.append({
                    'success': False,
                    'config_path': str(config_path),
                    'error': str(e)
                })

        # Summary
        successful = sum(1 for r in self.results if r['success'])
        logger.info(f"\n{'='*60}")
        logger.info(f"ALL EXPERIMENTS COMPLETED: {successful}/{len(self.results)} successful")
        logger.info(f"{'='*60}")

        return self.results

    def compare_experiments(
        self,
        metric: str = 'silhouette_score',
        entity_type: str = 'intervention'
    ) -> Dict[str, Any]:
        """
        Compare experiments by a specific metric.

        Args:
            metric: Metric to compare (silhouette_score, naming_failures, etc.)
            entity_type: Entity type to compare (intervention, condition, mechanism)

        Returns:
            Dict with comparison results
        """
        if not self.results:
            logger.warning("No results to compare")
            return {}

        logger.info(f"\nComparing experiments by {metric} ({entity_type})")

        comparison = {
            'metric': metric,
            'entity_type': entity_type,
            'experiments': []
        }

        for result in self.results:
            if not result['success']:
                continue

            entity_results = result['results'].get(f'{entity_type}s')
            if not entity_results:
                continue

            metric_value = getattr(entity_results, metric, None)

            comparison['experiments'].append({
                'experiment_name': result['experiment_name'],
                'experiment_id': result['experiment_id'],
                'metric_value': metric_value,
                'duration_seconds': result['duration_seconds']
            })

        # Sort by metric (higher is better for silhouette, lower for failures)
        if 'failure' in metric or 'error' in metric:
            comparison['experiments'].sort(key=lambda x: x['metric_value'] if x['metric_value'] is not None else float('inf'))
        else:
            comparison['experiments'].sort(key=lambda x: x['metric_value'] if x['metric_value'] is not None else float('-inf'), reverse=True)

        if comparison['experiments']:
            comparison['best_experiment'] = comparison['experiments'][0]
            logger.info(f"Best experiment: {comparison['best_experiment']['experiment_name']} "
                       f"({metric}={comparison['best_experiment']['metric_value']})")

        return comparison

    def compare_temperatures(self, entity_type: str = 'intervention') -> Dict[str, Any]:
        """
        Compare naming results across different temperatures.

        Args:
            entity_type: Entity type to compare

        Returns:
            Dict with temperature comparison
        """
        logger.info(f"\nComparing temperatures for {entity_type}s")

        if not self.experiment_db_path:
            logger.error("No experiment database specified")
            return {}

        conn = sqlite3.connect(self.experiment_db_path)
        cursor = conn.cursor()

        # Query temperature comparison
        cursor.execute("""
            SELECT
                e.naming_temperature,
                COUNT(*) as num_experiments,
                AVG(er.num_clusters) as avg_clusters,
                AVG(er.num_singleton_clusters) as avg_singletons,
                AVG(er.silhouette_score) as avg_silhouette,
                AVG(er.naming_failures) as avg_failures,
                AVG(er.naming_cache_hit_rate) as avg_cache_hit_rate,
                AVG(er.naming_duration_seconds) as avg_naming_duration
            FROM experiments e
            JOIN experiment_results er ON e.experiment_id = er.experiment_id
            WHERE e.status = 'completed' AND er.entity_type = ?
            GROUP BY e.naming_temperature
            ORDER BY e.naming_temperature
        """, (entity_type,))

        rows = cursor.fetchall()
        conn.close()

        comparison = {
            'entity_type': entity_type,
            'temperatures': []
        }

        for row in rows:
            comparison['temperatures'].append({
                'temperature': row[0],
                'num_experiments': row[1],
                'avg_clusters': row[2],
                'avg_singletons': row[3],
                'avg_silhouette': row[4],
                'avg_failures': row[5],
                'avg_cache_hit_rate': row[6],
                'avg_naming_duration': row[7]
            })

        # Print summary
        logger.info("\nTemperature Comparison:")
        logger.info(f"{'Temp':<6} {'Experiments':<12} {'Clusters':<10} {'Silhouette':<12} {'Failures':<10} {'Duration':<10}")
        logger.info("-" * 70)
        for temp_data in comparison['temperatures']:
            logger.info(f"{temp_data['temperature']:<6.1f} {temp_data['num_experiments']:<12} "
                       f"{temp_data['avg_clusters']:<10.1f} {temp_data['avg_silhouette']:<12.3f} "
                       f"{temp_data['avg_failures']:<10.1f} {temp_data['avg_naming_duration']:<10.1f}s")

        # Select best temperature (lowest failures, highest silhouette)
        if comparison['temperatures']:
            best = min(comparison['temperatures'], key=lambda x: (x['avg_failures'], -x['avg_silhouette']))
            comparison['best_temperature'] = best['temperature']
            logger.info(f"\nRecommended temperature: {best['temperature']:.1f}")
            logger.info(f"  - Avg failures: {best['avg_failures']:.2f}")
            logger.info(f"  - Avg silhouette: {best['avg_silhouette']:.3f}")

        return comparison

    def analyze_naming_consistency(
        self,
        experiment_ids: List[int],
        entity_type: str = 'intervention'
    ) -> Dict[str, Any]:
        """
        Analyze naming consistency across experiments (for same clusters).

        Args:
            experiment_ids: List of experiment IDs to compare
            entity_type: Entity type to analyze

        Returns:
            Dict with consistency analysis
        """
        if not self.experiment_db_path:
            logger.error("No experiment database specified")
            return {}

        logger.info(f"\nAnalyzing naming consistency for {len(experiment_ids)} experiments")

        conn = sqlite3.connect(self.experiment_db_path)
        cursor = conn.cursor()

        # Get cluster details for each experiment
        cluster_names = defaultdict(lambda: defaultdict(list))

        for exp_id in experiment_ids:
            cursor.execute("""
                SELECT cluster_id, canonical_name, member_entities
                FROM cluster_details
                WHERE experiment_id = ? AND entity_type = ?
                ORDER BY cluster_id
            """, (exp_id, entity_type))

            for cluster_id, canonical_name, members_json in cursor.fetchall():
                # Use member entities as cluster key (to match across experiments)
                members = tuple(sorted(json.loads(members_json)))
                cluster_names[members][exp_id].append(canonical_name)

        conn.close()

        # Analyze consistency
        analysis = {
            'entity_type': entity_type,
            'num_experiments': len(experiment_ids),
            'total_clusters_analyzed': len(cluster_names),
            'clusters': []
        }

        for members, exp_names in cluster_names.items():
            # Only analyze clusters present in multiple experiments
            if len(exp_names) < 2:
                continue

            # Get all names for this cluster
            all_names = []
            for exp_id in experiment_ids:
                if exp_id in exp_names:
                    all_names.extend(exp_names[exp_id])

            # Calculate consistency (same name across experiments)
            unique_names = set(all_names)
            consistency = 1.0 if len(unique_names) == 1 else 1.0 / len(unique_names)

            analysis['clusters'].append({
                'member_count': len(members),
                'members_preview': list(members)[:3],  # First 3 members
                'num_experiments_with_cluster': len(exp_names),
                'unique_names': list(unique_names),
                'consistency_score': consistency
            })

        # Calculate overall consistency
        if analysis['clusters']:
            analysis['overall_consistency'] = sum(c['consistency_score'] for c in analysis['clusters']) / len(analysis['clusters'])
            logger.info(f"Overall naming consistency: {analysis['overall_consistency']:.2%}")

        return analysis

    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive experiment report.

        Args:
            output_path: Path to save JSON report (optional)

        Returns:
            Dict with full report
        """
        logger.info("\nGenerating experiment report...")

        report = {
            'report_generated_at': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful_experiments': sum(1 for r in self.results if r['success']),
            'failed_experiments': sum(1 for r in self.results if not r['success']),
            'experiments': []
        }

        # Add experiment details
        for result in self.results:
            if result['success']:
                exp_summary = {
                    'experiment_name': result['experiment_name'],
                    'experiment_id': result['experiment_id'],
                    'duration_seconds': result['duration_seconds'],
                    'entity_types': {}
                }

                for entity_type in ['interventions', 'conditions', 'mechanisms']:
                    entity_results = result['results'].get(entity_type)
                    if entity_results:
                        exp_summary['entity_types'][entity_type] = {
                            'num_clusters': entity_results.num_clusters,
                            'num_singletons': entity_results.num_singleton_clusters,
                            'silhouette_score': entity_results.silhouette_score,
                            'naming_failures': entity_results.naming_failures
                        }

                report['experiments'].append(exp_summary)

        # Add comparisons
        if self.experiment_db_path:
            report['temperature_comparison'] = self.compare_temperatures('intervention')

        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")

        return report

    def cleanup(self):
        """Cleanup resources."""
        logger.info("ExperimentRunner cleanup complete")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run multiple Unified Phase 3 experiments")
    parser.add_argument('--db', required=True, help='Path to intervention_research.db')
    parser.add_argument('--configs', nargs='+', required=True, help='Paths to config YAML files')
    parser.add_argument('--exp-db', help='Path to experiment database')
    parser.add_argument('--cache-dir', help='Cache directory')
    parser.add_argument('--report', help='Path to save JSON report')

    args = parser.parse_args()

    # Run experiments
    runner = ExperimentRunner(
        config_paths=args.configs,
        db_path=args.db,
        experiment_db_path=args.exp_db,
        cache_dir=args.cache_dir
    )

    results = runner.run_all()

    # Generate report
    if args.report:
        runner.generate_report(args.report)

    # Compare temperatures
    runner.compare_temperatures('intervention')
