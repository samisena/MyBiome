"""
Test script to manually categorize one of the failed interventions
to understand why it failed.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.orchestration.rotation_llm_categorization import RotationLLMCategorizer

# Test with one of the failed interventions
failed_interventions = [
    {"id": 1483, "name": "self-care education based on conceptual mapping"},
    {"id": 1789, "name": "remote rehabilitation training"},
    {"id": 1792, "name": "postoperative home-based pulmonary rehabilitation"},
    {"id": 1872, "name": "Sodium-glucose cotransporter 2 inhibitors"},
    {"id": 1879, "name": "Sodium-Glucose Cotransporter-2 inhibitors"},
    {"id": 1881, "name": "BI 1595043"},
]

categorizer = RotationLLMCategorizer(batch_size=6)

print("Testing failed interventions...")
print("=" * 60)

try:
    # Try to categorize these as a batch
    result = categorizer._categorize_intervention_batch(failed_interventions)

    print(f"\nResults: {len(result)} interventions categorized")
    for intervention_id, category in result.items():
        intervention_name = next(i["name"] for i in failed_interventions if i["id"] == intervention_id)
        print(f"  ID {intervention_id}: {intervention_name} -> {category}")

    # Check which ones didn't get categorized
    missing = [i["id"] for i in failed_interventions if i["id"] not in result]
    if missing:
        print(f"\nMissing categories for IDs: {missing}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
