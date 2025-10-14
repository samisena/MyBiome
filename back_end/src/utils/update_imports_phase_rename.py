"""
Script to update all imports after renaming folders to reflect phases.

Renames:
- data_collection → phase_1_data_collection
- llm_processing → phase_2_llm_processing
- semantic_normalization → phase_3_semantic_normalization
"""

import os
import re
from pathlib import Path

# Define replacement patterns
REPLACEMENTS = [
    # Folder renames
    (r'from back_end\.src\.data_collection', 'from back_end.src.phase_1_data_collection'),
    (r'import back_end\.src\.data_collection', 'import back_end.src.phase_1_data_collection'),
    (r'from back_end\.src\.llm_processing', 'from back_end.src.phase_2_llm_processing'),
    (r'import back_end\.src\.llm_processing', 'import back_end.src.phase_2_llm_processing'),
    (r'from back_end\.src\.semantic_normalization', 'from back_end.src.phase_3_semantic_normalization'),
    (r'import back_end\.src\.semantic_normalization', 'import back_end.src.phase_3_semantic_normalization'),
]

def update_file(file_path: Path) -> tuple[int, list[str]]:
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        for pattern, replacement in REPLACEMENTS:
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
    """Update all Python files in the back_end directory."""
    back_end_dir = Path(__file__).parent.parent.parent

    print("="*80)
    print("UPDATING IMPORTS AFTER PHASE RENAMING")
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
            for change in changes:
                print(f"  - {change}")

    # Also update markdown files in experiments folder
    for md_file in back_end_dir.glob("experiments/**/*.md"):
        change_count, changes = update_file(md_file)
        if change_count > 0:
            updated_files += 1
            print(f"\n{md_file.relative_to(back_end_dir)}:")
            for change in changes:
                print(f"  - {change}")

    print("\n" + "="*80)
    print(f"SUMMARY: {updated_files}/{total_files} files updated")
    print("="*80)

if __name__ == "__main__":
    main()
