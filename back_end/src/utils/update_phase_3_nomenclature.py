"""
Update Phase 3 nomenclature from 3.5/3.6 to 3b/3c.

Changes:
- phase_3_5 → phase_3b (group categorization)
- phase_3_6 → phase_3c (mechanism clustering)
- phase_3d remains as-is
"""

import re
from pathlib import Path

REPLACEMENTS = [
    # File names in imports
    ('phase_3b_intervention_categorizer', 'phase_3b_intervention_categorizer'),
    ('phase_3b_condition_categorizer', 'phase_3b_condition_categorizer'),
    ('phase_3b_group_categorization', 'phase_3b_group_categorization'),
    ('phase_3c_mechanism_clustering', 'phase_3c_mechanism_clustering'),
    ('phase_3c_mechanism_pipeline_orchestrator', 'phase_3c_mechanism_pipeline_orchestrator'),
    ('phase_3c_mechanism_hierarchical_clustering', 'phase_3c_mechanism_hierarchical_clustering'),
]

def update_file(file_path: Path) -> tuple[int, list[str]]:
    """Update Phase 3 nomenclature in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        for old_name, new_name in REPLACEMENTS:
            # Match imports (from/import statements)
            pattern = rf'\b{old_name}\b'
            matches = list(re.finditer(pattern, content))
            if matches:
                content = re.sub(pattern, new_name, content)
                changes.append(f"{old_name} -> {new_name} ({len(matches)} occurrences)")

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return len(changes), changes

        return 0, []
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, []

def main():
    """Update all files in back_end directory."""
    back_end_dir = Path(__file__).parent.parent.parent

    print("="*80)
    print("UPDATING PHASE 3 NOMENCLATURE (3.5 -> 3b, 3.6 -> 3c)")
    print("="*80)

    total_files = 0
    updated_files = 0

    # Find all Python files
    for py_file in back_end_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        total_files += 1
        change_count, changes = update_file(py_file)

        if change_count > 0:
            updated_files += 1
            print(f"\n{py_file.relative_to(back_end_dir)}:")
            for change in set(changes):
                print(f"  - {change}")

    print("\n" + "="*80)
    print(f"SUMMARY: {updated_files}/{total_files} files updated")
    print("="*80)

if __name__ == "__main__":
    main()
