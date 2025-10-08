"""
Drop legacy Phase 3 canonical tables from the database.

These tables are deprecated and replaced by Phase 3.5 hierarchical semantic normalization:
- canonical_entities (378 entities) - replaced by semantic_hierarchy
- entity_mappings (204 mappings) - replaced by entity_relationships
- llm_normalization_cache - replaced by cache files in data/semantic_normalization_cache/
- normalized_terms_cache - replaced by cache files

This script will:
1. Show you what will be deleted
2. Ask for confirmation
3. Back up the tables before dropping (optional)
4. Drop the legacy tables

WARNING: This action cannot be easily undone without a backup!
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import shutil

def get_database_path() -> Path:
    """Get the path to the intervention research database."""
    return Path(__file__).parent.parent.parent / "data" / "processed" / "intervention_research.db"

def backup_database(db_path: Path) -> Path:
    """Create a backup of the database before making changes."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"intervention_research_backup_{timestamp}.db"

    print(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    print(f"Backup created successfully!")

    return backup_path

def show_legacy_table_info(cursor):
    """Show information about legacy tables before dropping."""
    legacy_tables = [
        'canonical_entities',
        'entity_mappings',
        'llm_normalization_cache',
        'normalized_terms_cache'
    ]

    print("\n" + "="*70)
    print("LEGACY TABLES TO BE DROPPED (Phase 3 - Deprecated)")
    print("="*70)

    for table in legacy_tables:
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
        if cursor.fetchone():
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"\n[X] {table}")
            print(f"  - Records: {count:,}")
            print(f"  - Status: DEPRECATED")
        else:
            print(f"\n[O] {table}")
            print(f"  - Status: Already removed")

    print("\n" + "="*70)
    print("REPLACEMENT TABLES (Phase 3.5 - Current)")
    print("="*70)

    # Show replacement tables
    cursor.execute("SELECT COUNT(*) FROM semantic_hierarchy WHERE entity_type = 'intervention'")
    semantic_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT layer_1_canonical) FROM semantic_hierarchy WHERE entity_type = 'intervention' AND layer_1_canonical IS NOT NULL")
    canonical_groups = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM entity_relationships")
    relationships = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM canonical_groups")
    groups_count = cursor.fetchone()[0]

    print(f"\n[+] semantic_hierarchy")
    print(f"  - Records: {semantic_count:,} interventions")
    print(f"  - Canonical groups: {canonical_groups:,}")

    print(f"\n[+] entity_relationships")
    print(f"  - Records: {relationships:,} relationships")

    print(f"\n[+] canonical_groups")
    print(f"  - Records: {groups_count:,} groups")

    print("\n" + "="*70)

def drop_legacy_tables(cursor):
    """Drop the legacy Phase 3 tables."""
    legacy_tables = [
        'canonical_entities',
        'entity_mappings',
        'llm_normalization_cache',
        'normalized_terms_cache'
    ]

    dropped_count = 0

    for table in legacy_tables:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"[+] Dropped table: {table}")
            dropped_count += 1
        except Exception as e:
            print(f"[X] Error dropping {table}: {e}")

    return dropped_count

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("DROP LEGACY CANONICAL TABLES")
    print("="*70)
    print("\nThis script will remove deprecated Phase 3 tables and replace them")
    print("with the new Phase 3.5 hierarchical semantic normalization system.")

    db_path = get_database_path()

    if not db_path.exists():
        print(f"\nError: Database not found at {db_path}")
        return

    # Connect to database
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Show what will be dropped
    show_legacy_table_info(cursor)

    print("\n" + "="*70)
    print("CONFIRMATION")
    print("="*70)

    # Ask for confirmation
    response = input("\nDo you want to create a backup first? (recommended) [Y/n]: ").strip().lower()

    if response != 'n':
        backup_path = backup_database(db_path)
        print(f"\nBackup saved to: {backup_path}")
    else:
        print("\nSkipping backup...")

    print("\n" + "="*70)
    response = input("\nAre you sure you want to drop these legacy tables? [y/N]: ").strip().lower()

    if response == 'y':
        print("\nDropping legacy tables...")
        dropped = drop_legacy_tables(cursor)
        conn.commit()
        print(f"\n[+] Successfully dropped {dropped} legacy tables!")

        print("\n" + "="*70)
        print("CLEANUP COMPLETE")
        print("="*70)
        print("\nThe database now uses only Phase 3.5 hierarchical semantic normalization.")
        print("Legacy Phase 3 tables have been removed.")

    else:
        print("\nOperation cancelled. No changes made.")

    conn.close()

if __name__ == "__main__":
    main()
