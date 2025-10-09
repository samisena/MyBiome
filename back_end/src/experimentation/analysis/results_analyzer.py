"""
Analyze experiment results and generate recommendations.
"""

import json
from pathlib import Path
from typing import Dict, List


class ResultsAnalyzer:
    """Analyze batch size optimization experiment results."""

    def __init__(self, results_dir: str):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.experiments = []

    def load_results(self):
        """Load all experiment results."""
        result_files = sorted(self.results_dir.glob("EXP-*.json"))

        for result_file in result_files:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.experiments.append(data)

        print(f"Loaded {len(self.experiments)} experiment results")

    def analyze(self) -> Dict:
        """
        Analyze results and generate recommendations.

        Returns:
            Analysis dictionary with recommendations
        """
        if not self.experiments:
            self.load_results()

        print("\n" + "=" * 70)
        print("BATCH SIZE OPTIMIZATION RESULTS")
        print("=" * 70 + "\n")

        # Extract metrics for each experiment
        results = []
        for exp in self.experiments:
            exp_id = exp['experiment_id']
            batch_size = exp['batch_size']
            total_time = exp.get('total_pipeline_seconds', 0)
            papers_hour = exp.get('papers_per_hour', 0)
            errors = len(exp.get('errors', []))
            gpu_temp_max = exp.get('system_metrics', {}).get('gpu_temp_max', 0)
            gpu_memory_peak = exp.get('system_metrics', {}).get('gpu_memory_peak_mb', 0)

            results.append({
                'experiment_id': exp_id,
                'batch_size': batch_size,
                'total_time_seconds': total_time,
                'papers_per_hour': papers_hour,
                'error_count': errors,
                'gpu_temp_max': gpu_temp_max,
                'gpu_memory_peak_mb': gpu_memory_peak
            })

            print(f"{exp_id} - Batch Size {batch_size}")
            print(f"  Total Pipeline Time: {total_time:.1f}s")
            print(f"  Papers per Hour: {papers_hour:.1f}")
            print(f"  Errors: {errors}")
            print(f"  Max GPU Temp: {gpu_temp_max:.1f}°C")
            print(f"  Peak GPU Memory: {gpu_memory_peak:.1f} MB")
            print()

        # Find optimal batch size
        print("=" * 70)
        print("ANALYSIS")
        print("=" * 70 + "\n")

        # Filter safe results (temp <85°C, errors <5%)
        safe_results = [r for r in results if r['gpu_temp_max'] < 85.0 and r['error_count'] / 16 < 0.05]

        if not safe_results:
            print("WARNING: No experiments met safety criteria!")
            safe_results = results

        # Find fastest
        fastest = max(safe_results, key=lambda x: x['papers_per_hour'])

        print(f"Optimal Batch Size: {fastest['batch_size']}")
        print(f"  Papers per Hour: {fastest['papers_per_hour']:.1f}")
        print(f"  Total Time: {fastest['total_time_seconds']:.1f}s")
        print(f"  Max GPU Temp: {fastest['gpu_temp_max']:.1f}°C")
        print(f"  Errors: {fastest['error_count']}")
        print()

        # Compare with baseline (batch=8)
        baseline = next((r for r in results if r['batch_size'] == 8), None)
        if baseline and fastest['batch_size'] != 8:
            improvement = ((fastest['papers_per_hour'] - baseline['papers_per_hour']) / baseline['papers_per_hour']) * 100
            print(f"Improvement over baseline (batch=8): {improvement:+.1f}%")
            print()

        # Recommendation
        print("=" * 70)
        print("RECOMMENDATION")
        print("=" * 70 + "\n")

        if fastest['batch_size'] == 4:
            print("Recommendation: REDUCE batch size to 4")
            print("Reason: Smaller batches provide better stability or performance")
        elif fastest['batch_size'] == 8:
            print("Recommendation: KEEP batch size at 8")
            print("Reason: Current configuration is already optimal")
        elif fastest['batch_size'] == 12:
            print("Recommendation: INCREASE batch size to 12")
            print("Reason: Larger batches improve throughput without compromising safety")
        elif fastest['batch_size'] == 16:
            print("Recommendation: INCREASE batch size to 16")
            print("Reason: Maximum throughput achieved with acceptable safety margins")
            print("Note: Monitor thermal status closely in production")

        print("\nAction Required:")
        print(f"  Update config.py: intervention_batch_size = {fastest['batch_size']}")

        return {
            'optimal_batch_size': fastest['batch_size'],
            'papers_per_hour': fastest['papers_per_hour'],
            'improvement_vs_baseline': improvement if baseline and fastest['batch_size'] != 8 else 0,
            'all_results': results
        }


if __name__ == "__main__":
    """Analyze experiment results."""
    results_dir = Path(__file__).parent.parent / "data" / "results"

    analyzer = ResultsAnalyzer(str(results_dir))
    analysis = analyzer.analyze()

    # Save analysis
    output_file = results_dir / "analysis_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis saved to: {output_file}")
