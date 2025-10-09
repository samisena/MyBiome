"""
Main experiment runner for batch size optimization.
"""

import json
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from back_end.src.data.config import config, setup_logging
from back_end.src.llm_processing.single_model_analyzer import SingleModelAnalyzer
from back_end.src.orchestration.rotation_llm_categorization import RotationLLMCategorizer
from back_end.src.experimentation.config.experiment_config import ExperimentConfig
from back_end.src.experimentation.evaluation.system_monitor import SystemMonitor

logger = setup_logging(__name__, 'experiment_runner.log')


class ExperimentRunner:
    """Run batch size optimization experiments."""

    def __init__(self, papers: List[Dict], output_dir: str):
        """
        Initialize experiment runner.

        Args:
            papers: List of test papers
            output_dir: Directory to save results
        """
        self.papers = papers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = SingleModelAnalyzer()
        self.categorizer = RotationLLMCategorizer(batch_size=20)
        self.monitor = SystemMonitor(max_temp=85.0)

    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Run a single experiment.

        Args:
            config: Experiment configuration

        Returns:
            Results dictionary
        """
        logger.info(f"Starting {config.experiment_id}: {config.description}")
        logger.info(f"Batch size: {config.batch_size}, Papers: {len(self.papers)}")

        # Initialize results
        results = {
            'experiment_id': config.experiment_id,
            'batch_size': config.batch_size,
            'num_papers': len(self.papers),
            'description': config.description,
            'start_time': datetime.now().isoformat(),
            'phase2_results': {},
            'phase2_5_results': {},
            'system_metrics': {},
            'errors': []
        }

        try:
            # Check thermal status before starting
            initial_metrics = self.monitor.get_metrics()
            if initial_metrics:
                logger.info(f"Initial GPU temp: {initial_metrics.gpu_temp:.1f}°C")
                if not initial_metrics.is_safe:
                    logger.warning("GPU temperature too high, waiting for cooling...")
                    self.monitor.wait_for_cooling()

            # Phase 2: Extraction
            logger.info(f"Phase 2: Starting extraction with batch size {config.batch_size}")
            phase2_start = time.time()

            try:
                phase2_result = self.analyzer.process_papers_batch(
                    papers=self.papers,
                    save_to_db=False,  # Don't save to main DB (test only)
                    batch_size=config.batch_size
                )

                phase2_time = time.time() - phase2_start

                # Extract detailed intervention information
                all_interventions = []
                for paper_result in phase2_result.get('paper_results', []):
                    for intervention in paper_result.get('interventions', []):
                        all_interventions.append({
                            'intervention_name': intervention.get('intervention_name'),
                            'health_condition': intervention.get('health_condition'),
                            'mechanism': intervention.get('mechanism'),
                            'correlation_type': intervention.get('correlation_type'),
                            'correlation_strength': intervention.get('correlation_strength'),
                            'study_type': intervention.get('study_type'),
                            'sample_size': intervention.get('sample_size'),
                            'paper_id': paper_result.get('pmid')
                        })

                results['phase2_results'] = {
                    'duration_seconds': phase2_time,
                    'successful_papers': phase2_result['successful_papers'],
                    'failed_papers': phase2_result['failed_papers'],
                    'total_interventions': phase2_result['total_interventions'],
                    'interventions_by_category': phase2_result['interventions_by_category'],
                    'model_statistics': phase2_result['model_statistics'],
                    'interventions': all_interventions  # Save full intervention details
                }

                logger.info(f"Phase 2 complete: {phase2_time:.1f}s, {phase2_result['total_interventions']} interventions")

            except Exception as e:
                logger.error(f"Phase 2 failed: {e}")
                logger.error(traceback.format_exc())
                results['errors'].append(f"Phase 2 error: {str(e)}")
                results['phase2_results']['error'] = str(e)
                phase2_result = None

            # Phase 2.5: Categorization (if Phase 2 succeeded)
            if phase2_result and phase2_result['total_interventions'] > 0:
                logger.info(f"Phase 2.5: Categorizing {phase2_result['total_interventions']} interventions")
                phase2_5_start = time.time()

                try:
                    # Mock categorization (would normally categorize extracted interventions)
                    # For this experiment, we'll estimate based on batch size
                    num_interventions = phase2_result['total_interventions']
                    batches_needed = (num_interventions + 19) // 20  # Round up
                    estimated_time = batches_needed * 5  # ~5s per batch

                    time.sleep(min(estimated_time, 60))  # Cap at 60s for testing

                    phase2_5_time = time.time() - phase2_5_start

                    results['phase2_5_results'] = {
                        'duration_seconds': phase2_5_time,
                        'interventions_categorized': num_interventions,
                        'batches_processed': batches_needed,
                        'estimated': True  # Flag that this is estimated
                    }

                    logger.info(f"Phase 2.5 complete: {phase2_5_time:.1f}s")

                except Exception as e:
                    logger.error(f"Phase 2.5 failed: {e}")
                    results['errors'].append(f"Phase 2.5 error: {str(e)}")
                    results['phase2_5_results']['error'] = str(e)

            # Collect final system metrics
            final_metrics = self.monitor.get_metrics()
            if final_metrics:
                logger.info(f"Final GPU temp: {final_metrics.gpu_temp:.1f}°C")

            results['system_metrics'] = self.monitor.get_summary()
            results['end_time'] = datetime.now().isoformat()

            # Calculate total time
            total_time = results['phase2_results'].get('duration_seconds', 0) + \
                        results['phase2_5_results'].get('duration_seconds', 0)
            results['total_pipeline_seconds'] = total_time
            results['papers_per_hour'] = (len(self.papers) / total_time * 3600) if total_time > 0 else 0

            logger.info(f"{config.experiment_id} complete: {total_time:.1f}s total, {results['papers_per_hour']:.1f} papers/hour")

        except Exception as e:
            logger.error(f"Experiment {config.experiment_id} failed: {e}")
            logger.error(traceback.format_exc())
            results['errors'].append(f"Experiment error: {str(e)}")
            results['end_time'] = datetime.now().isoformat()

        # Save results immediately (error recovery)
        self.save_results(config.experiment_id, results)

        return results

    def save_results(self, experiment_id: str, results: Dict):
        """
        Save experiment results to JSON file.

        Args:
            experiment_id: Experiment ID
            results: Results dictionary
        """
        output_file = self.output_dir / f"{experiment_id}_results.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")

    def run_all_experiments(self, experiments: List[ExperimentConfig], cooling_period: int = 0):
        """
        Run all experiments (no cooling periods by default).

        Args:
            experiments: List of experiment configurations
            cooling_period: Seconds to wait between experiments (default 0 = no cooling)
        """
        all_results = []

        for i, exp_config in enumerate(experiments, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {i}/{len(experiments)}: {exp_config.experiment_id}")
            logger.info(f"{'='*60}\n")

            result = self.run_experiment(exp_config)
            all_results.append(result)

            # Optional cooling period between experiments
            if cooling_period > 0 and i < len(experiments):
                logger.info(f"\nCooling period: {cooling_period}s before next experiment...")
                self.monitor.wait_for_cooling(target_temp=75.0, timeout=cooling_period)
                logger.info(f"Cooling complete\n")

        # Save summary
        summary_file = self.output_dir / "experiment_summary.json"
        summary = {
            'total_experiments': len(all_results),
            'test_papers': len(self.papers),
            'experiments': all_results,
            'completed_at': datetime.now().isoformat()
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"\nAll experiments complete. Summary saved to {summary_file}")

        return all_results


if __name__ == "__main__":
    """Run all batch size experiments."""
    from back_end.src.experimentation.config.experiment_config import EXPERIMENTS
    from back_end.src.experimentation.runners.dataset_selector import DatasetSelector

    # Load test dataset
    data_dir = Path(__file__).parent.parent / "data"
    dataset_file = data_dir / "test_dataset.json"

    if not dataset_file.exists():
        print("Test dataset not found. Creating...")
        selector = DatasetSelector(num_papers=16)
        papers = selector.select_papers()
        selector.save_dataset(papers, str(dataset_file))
    else:
        print(f"Loading test dataset from {dataset_file}")
        selector = DatasetSelector()
        papers = selector.load_dataset(str(dataset_file))

    print(f"Loaded {len(papers)} test papers")

    # Create runner and run all experiments
    results_dir = data_dir / "results"
    runner = ExperimentRunner(papers, str(results_dir))

    print("\nStarting batch size optimization experiments...")
    print(f"Experiments to run: {len(EXPERIMENTS)}")
    print(f"Papers per experiment: {len(papers)}")
    print(f"Results directory: {results_dir}\n")

    runner.run_all_experiments(EXPERIMENTS, cooling_period=0)  # No cooling periods

    print("\nExperimentation complete!")
