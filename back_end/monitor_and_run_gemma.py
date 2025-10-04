"""
Monitor script: Waits for qwen test to complete, then auto-runs Gemma test.
"""

import time
import subprocess
import os
from pathlib import Path

def check_qwen_test_done():
    """Check if qwen test has created its output file."""
    output_file = Path(__file__).parent / "model_comparison_results.json"
    return output_file.exists()

def main():
    print("Monitoring for qwen test completion...")
    print("Waiting for: model_comparison_results.json")

    check_interval = 10  # seconds
    max_wait_time = 1800  # 30 minutes max wait
    elapsed = 0

    while elapsed < max_wait_time:
        if check_qwen_test_done():
            print(f"\n✓ Qwen test completed after {elapsed}s!")
            print("Reading results...")

            # Show qwen results briefly
            try:
                import json
                with open(Path(__file__).parent / "model_comparison_results.json", 'r') as f:
                    data = json.load(f)

                print("\nQwen Test Summary:")
                for model_key in ['qwen2.5_14b', 'qwen3_14b']:
                    if model_key in data:
                        result = data[model_key]
                        model_name = result.get('model', model_key)
                        count = len(result.get('interventions', []))
                        time_taken = result.get('extraction_time', 0)
                        print(f"  {model_name}: {count} interventions in {time_taken:.2f}s")

                        # Show conditions extracted
                        if result.get('interventions'):
                            for i, interv in enumerate(result['interventions'], 1):
                                condition = interv.get('health_condition', 'N/A')
                                print(f"    #{i}: {condition}")
            except Exception as e:
                print(f"Could not parse results: {e}")

            print("\n" + "="*80)
            print("Starting Gemma model test...")
            print("="*80 + "\n")

            # Wait a bit for GPU cleanup
            print("Pausing 10 seconds for GPU cleanup...")
            time.sleep(10)

            # Run Gemma test
            os.chdir(Path(__file__).parent.parent)
            result = subprocess.run(
                ['python', '-m', 'back_end.test_4model_comparison'],
                capture_output=False
            )

            if result.returncode == 0:
                print("\n✅ Gemma test completed successfully!")
            else:
                print(f"\n❌ Gemma test failed with code {result.returncode}")

            return

        # Still waiting
        elapsed += check_interval
        print(f"  {elapsed}s elapsed... still waiting")
        time.sleep(check_interval)

    print(f"\n⚠️  Timeout after {max_wait_time}s - qwen test may have failed")
    print("You can manually run: python -m back_end.test_4model_comparison")

if __name__ == "__main__":
    main()
