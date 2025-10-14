"""
Update all imports after renaming individual files within phase folders.
"""

import re
from pathlib import Path

# Define all file renames (old_name, new_name)
FILE_RENAMES = [
    # Phase 1 renames
    ('pubmed_collector', 'phase_1_pubmed_collector'),
    ('fulltext_retriever', 'phase_1_fulltext_retriever'),
    ('paper_parser', 'phase_1_paper_parser'),
    ('semantic_scholar_enrichment', 'phase_1_semantic_scholar_enrichment'),

    # Phase 2 renames
    ('single_model_analyzer', 'phase_2_single_model_analyzer'),
    ('batch_entity_processor', 'phase_2_batch_entity_processor'),
    ('entity_operations', 'phase_2_entity_operations'),
    ('entity_utils', 'phase_2_entity_utils'),
    ('prompt_service', 'phase_2_prompt_service'),
    ('export_to_json', 'phase_2_export_to_json'),

    # Phase 3 renames
    ('embedding_engine', 'phase_3_embedding_engine'),
    ('llm_classifier', 'phase_3_llm_classifier'),
    ('hierarchy_manager', 'phase_3_hierarchy_manager'),
    ('normalizer', 'phase_3_normalizer'),
    ('group_categorizer', 'phase_3b_intervention_categorizer'),
    ('condition_group_categorizer', 'phase_3b_condition_categorizer'),
]

def create_import_patterns(old_name, new_name):
    """Create regex patterns for all import variations."""
    patterns = []

    # Absolute imports from phase folders
    patterns.append((
        rf'from back_end\.src\.phase_1_data_collection\.{old_name}',
        f'from back_end.src.phase_1_data_collection.{new_name}'
    ))
    patterns.append((
        rf'from back_end\.src\.phase_2_llm_processing\.{old_name}',
        f'from back_end.src.phase_2_llm_processing.{new_name}'
    ))
    patterns.append((
        rf'from back_end\.src\.phase_3_semantic_normalization\.{old_name}',
        f'from back_end.src.phase_3_semantic_normalization.{new_name}'
    ))

    # Relative imports with dot notation
    patterns.append((
        rf'from \.{old_name}',
        f'from .{new_name}'
    ))
    patterns.append((
        rf'from \.\.{old_name}',
        f'from ..{new_name}'
    ))

    # Import statements
    patterns.append((
        rf'import {old_name}',
        f'import {new_name}'
    ))

    return patterns

def update_file(file_path: Path) -> tuple[int, list[str]]:
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        for old_name, new_name in FILE_RENAMES:
            patterns = create_import_patterns(old_name, new_name)

            for pattern, replacement in patterns:
                matches = list(re.finditer(pattern, content))
                if matches:
                    content = re.sub(pattern, replacement, content)
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
    """Update all Python files in back_end directory."""
    back_end_dir = Path(__file__).parent.parent.parent

    print("="*80)
    print("UPDATING IMPORTS AFTER FILE RENAMING")
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
            for change in set(changes):  # Unique changes only
                print(f"  - {change}")

    print("\n" + "="*80)
    print(f"SUMMARY: {updated_files}/{total_files} files updated")
    print("="*80)

if __name__ == "__main__":
    main()
