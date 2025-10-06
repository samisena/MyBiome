import json
import shutil
from pathlib import Path

# File paths
ground_truth_file = Path("c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/ground_truth/labeling_session_hierarchical_ground_truth_20251005_184757.json")
backup_file = ground_truth_file.with_name(ground_truth_file.stem + "_BACKUP.json")
corrected_file = ground_truth_file.with_name(ground_truth_file.stem + "_CORRECTED.json")

print("Loading ground truth file...")
with open(ground_truth_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create backup
print(f"Creating backup at: {backup_file}")
shutil.copy2(ground_truth_file, backup_file)

# Extract labeled pairs
labeled_pairs = data['labeled_pairs']
print(f"Total labeled pairs: {len(labeled_pairs)}")

# Apply corrections
corrections_applied = 0

# Fix #1: Pair 3 - Cetuximab canonical (monoclonal antibody -> cetuximab)
print("\n[Fix 1/9] Pair 3: Changing canonical from 'monoclonal antibody' to 'cetuximab'")
labeled_pairs[2]['relationship']['hierarchy']['layer_1_canonical'] = 'cetuximab'
corrections_applied += 1

# Fix #2: Pair 4 - Change DOSAGE_VARIANT to VARIANT
print("[Fix 2/9] Pair 4: Changing DOSAGE_VARIANT to VARIANT")
labeled_pairs[3]['relationship']['type_code'] = 'VARIANT'
labeled_pairs[3]['relationship']['type_display'] = 'Variant (same concept, different formulation)'
labeled_pairs[3]['relationship']['aggregation_rule'] = 'share_layer_1_link_layer_2'
corrections_applied += 1

# Fix #3: Pair 13 - Change VARIANT to EXACT_MATCH
print("[Fix 3/9] Pair 13: Changing VARIANT to EXACT_MATCH")
labeled_pairs[12]['relationship']['type_code'] = 'EXACT_MATCH'
labeled_pairs[12]['relationship']['type_display'] = 'Exact Match (same intervention, same formulation)'
labeled_pairs[12]['relationship']['aggregation_rule'] = 'merge_completely'
labeled_pairs[12]['relationship']['hierarchy']['same_variant_layer_2'] = True
corrections_applied += 1

# Fix #4: Pair 18 - Change VARIANT to SAME_CATEGORY
print("[Fix 4/9] Pair 18: Changing VARIANT to SAME_CATEGORY")
labeled_pairs[17]['relationship']['type_code'] = 'SAME_CATEGORY'
labeled_pairs[17]['relationship']['type_display'] = 'Same Category (different entities in same class)'
labeled_pairs[17]['relationship']['aggregation_rule'] = 'separate_all_layers'
labeled_pairs[17]['relationship']['hierarchy']['same_variant_layer_2'] = False
corrections_applied += 1

# Fix #5: Pair 22 - Change EXACT_MATCH to VARIANT
print("[Fix 5/9] Pair 22: Changing EXACT_MATCH to VARIANT")
labeled_pairs[21]['relationship']['type_code'] = 'VARIANT'
labeled_pairs[21]['relationship']['type_display'] = 'Variant (same concept, different formulation)'
labeled_pairs[21]['relationship']['aggregation_rule'] = 'share_layer_1_link_layer_2'
labeled_pairs[21]['relationship']['hierarchy']['same_variant_layer_2'] = False
corrections_applied += 1

# Fix #6: Pair 29 - Refine canonical (interventions -> nutritional interventions)
print("[Fix 6/9] Pair 29: Refining canonical from 'interventions' to 'nutritional interventions'")
labeled_pairs[28]['relationship']['hierarchy']['layer_1_canonical'] = 'nutritional interventions'
corrections_applied += 1

# Fix #7: Pair 32 - Change VARIANT to DIFFERENT
print("[Fix 7/9] Pair 32: Changing VARIANT to DIFFERENT")
labeled_pairs[31]['relationship']['type_code'] = 'DIFFERENT'
labeled_pairs[31]['relationship']['type_display'] = 'Different (completely unrelated interventions)'
labeled_pairs[31]['relationship']['aggregation_rule'] = 'no_relationship'
labeled_pairs[31]['relationship']['hierarchy'] = {}
corrections_applied += 1

# Fix #8: Pair 35 - Change VARIANT to DIFFERENT
print("[Fix 8/9] Pair 35: Changing VARIANT to DIFFERENT")
labeled_pairs[34]['relationship']['type_code'] = 'DIFFERENT'
labeled_pairs[34]['relationship']['type_display'] = 'Different (completely unrelated interventions)'
labeled_pairs[34]['relationship']['aggregation_rule'] = 'no_relationship'
labeled_pairs[34]['relationship']['hierarchy'] = {}
corrections_applied += 1

# Fix #9: Pair 48 - Refine canonical (auricular acupressure -> acupressure)
print("[Fix 9/9] Pair 48: Refining canonical from 'auricular acupressure' to 'acupressure'")
labeled_pairs[47]['relationship']['hierarchy']['layer_1_canonical'] = 'acupressure'
corrections_applied += 1

# Save corrected file
print(f"\n[OK] All {corrections_applied} corrections applied successfully!")
print(f"Saving corrected file to: {corrected_file}")

with open(corrected_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("\n=== Summary ===")
print(f"Backup file: {backup_file}")
print(f"Corrected file: {corrected_file}")
print(f"Total corrections applied: {corrections_applied}/9")
print("\n[OK] Correction process complete!")
