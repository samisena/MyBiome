"""
Update relative imports after folder renaming.
"""

import os
import re
from pathlib import Path

# Define replacement patterns for relative imports
RELATIVE_REPLACEMENTS = [
    (r'from \.\.data_collection', 'from ..phase_1_data_collection'),
    (r'from \.\.llm_processing', 'from ..phase_2_llm_processing'),
    (r'from \.\.semantic_normalization', 'from ..phase_3_semantic_normalization'),
]

def update_file(file_path: Path) -> tuple[int, list[str]]:
    """Update relative imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        for pattern, replacement in RELATIVE_REPLACEMENTS:
            matches = re.findall(pattern, content)
            if matches:
                content = re.sub(pattern, replacement, content)
                changes.append(f"{pattern} -> {replacement} ({len(matches)} occurrences)")

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return len(changes), changes

        return 0, []
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, []

def main():
    """Update all Python files in the src directory."""
    src_dir = Path(__file__).parent.parent

    print("="*80)
    print("UPDATING RELATIVE IMPORTS")
    print("="*80)

    total_files = 0
    updated_files = 0

    # Find all Python files in src
    for py_file in src_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        total_files += 1
        change_count, changes = update_file(py_file)

        if change_count > 0:
            updated_files += 1
            print(f"\n{py_file.relative_to(src_dir)}:")
            for change in changes:
                print(f"  - {change}")

    print("\n" + "="*80)
    print(f"SUMMARY: {updated_files}/{total_files} files updated")
    print("="*80)

if __name__ == "__main__":
    main()
