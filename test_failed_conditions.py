"""
Test script to manually categorize the failed conditions
to understand why they failed.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.orchestration.rotation_llm_categorization import RotationLLMCategorizer

# Test with failed conditions
failed_conditions = [
    "female infertility",
    "metabolic adverse effects",
    "metabolic conditions",
    "metabolic pathways in adipose tissue",
    "moderate or severe hypercholesterolemia",
    "moderate or severe hypercholesterolemia, amyloidosis"
]

categorizer = RotationLLMCategorizer(batch_size=6)

print("Testing failed conditions...")
print("=" * 60)

try:
    # Try to categorize these as a batch
    result = categorizer._categorize_condition_batch(failed_conditions)

    print(f"\nResults: {len(result)} conditions categorized")
    for condition_name, category in result.items():
        print(f"  {condition_name} -> {category}")

    # Check which ones didn't get categorized
    missing = [c for c in failed_conditions if c not in result]
    if missing:
        print(f"\nMissing categories:")
        for c in missing:
            print(f"  - {c}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
