"""
Quick script to check experiment progress.
"""

import json
from pathlib import Path

results_dir = Path(__file__).parent / "data" / "results"

print("Checking experiment progress...")
print("=" * 60)

if not results_dir.exists():
    print("Results directory not found yet.")
else:
    results_files = list(results_dir.glob("EXP-*.json"))
    print(f"Found {len(results_files)} experiment result files\n")

    for result_file in sorted(results_files):
        with open(result_file, 'r') as f:
            data = json.load(f)

        exp_id = data.get('experiment_id', 'Unknown')
        batch_size = data.get('batch_size', '?')
        total_time = data.get('total_pipeline_seconds', 0)
        papers_hour = data.get('papers_per_hour', 0)
        errors = len(data.get('errors', []))

        print(f"{exp_id} (Batch={batch_size}):")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Papers/hour: {papers_hour:.1f}")
        print(f"  Errors: {errors}")
        print()

    # Check summary
    summary_file = results_dir / "experiment_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"\nAll {summary['total_experiments']} experiments complete!")
        print(f"Completed at: {summary['completed_at']}")

print("=" * 60)
