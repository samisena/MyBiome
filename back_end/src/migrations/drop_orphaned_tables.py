"""
Database Migration: Drop Orphaned and Legacy Tables
Date: October 16, 2025
Migration Type: Schema Cleanup - Round 2

This migration removes:
1. 8 orphaned analytics tables (created but never populated)
2. 4 legacy normalization tables (replaced by Phase 3 semantic_hierarchy)

Total tables dropped: 12
Total schema reduction: ~30% of unused tables

Safety: Creates automatic backup before dropping tables
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime


# Tables to drop - organized by category
ORPHANED_ANALYTICS_TABLES = [
    'treatment_recommendations',
    'research_gaps',
    'innovation_tracking',
    'biological_patterns',
    'condition_similarities',
    'intervention_combinations',
    'failed_interventions',
    'data_mining_sessions'
]

LEGACY_NORMALIZATION_TABLES = [
    'canonical_entities',
    'entity_mappings',
    'normalized_terms_cache',
    'llm_normalization_cache'
]


def create_backup(db_path: Path) -> Path:
    """Create timestamped backup of database before migration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = db_path.parent / f"{db_path.stem}_backup_before_table_drop_{timestamp}.db"

    print(f"Creating backup: {backup_path.name}")
    shutil.copy2(db_path, backup_path)

    backup_size_mb = backup_path.stat().st_size / (1024 * 1024)
    print(f"Backup created successfully ({backup_size_mb:.2f} MB)")

    return backup_path


def verify_tables_exist(conn: sqlite3.Connection, table_names: list) -> dict:
    """Check which tables actually exist in the database"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}

    results = {}
    for table in table_names:
        results[table] = table in existing_tables

    return results


def get_table_row_count(conn: sqlite3.Connection, table_name: str) -> int:
    """Get row count for a table"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]
    except sqlite3.Error:
        return -1


def drop_tables(conn: sqlite3.Connection, table_names: list) -> dict:
    """Drop specified tables and return results"""
    cursor = conn.cursor()
    results = {}

    for table in table_names:
        try:
            # Check row count before dropping
            row_count = get_table_row_count(conn, table)

            # Drop table
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()

            results[table] = {
                'status': 'dropped',
                'row_count': row_count
            }
            print(f"  [OK] Dropped {table} ({row_count} rows)")

        except sqlite3.Error as e:
            results[table] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"  [ERROR] Error dropping {table}: {e}")

    return results


def verify_drops(conn: sqlite3.Connection, table_names: list) -> bool:
    """Verify all tables were successfully dropped"""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    remaining_tables = {row[0] for row in cursor.fetchall()}

    failed_drops = [table for table in table_names if table in remaining_tables]

    if failed_drops:
        print(f"\nWARNING: {len(failed_drops)} tables were not dropped:")
        for table in failed_drops:
            print(f"  - {table}")
        return False

    return True


def print_migration_summary(orphaned_results: dict, legacy_results: dict):
    """Print detailed migration summary"""
    print("\n" + "="*70)
    print("MIGRATION SUMMARY")
    print("="*70)

    print("\nOrphaned Analytics Tables (8 tables):")
    print("-" * 70)
    total_orphaned_rows = 0
    for table, result in orphaned_results.items():
        if result['status'] == 'dropped':
            total_orphaned_rows += result.get('row_count', 0)
            status = "DROPPED" if result['row_count'] == 0 else f"DROPPED ({result['row_count']} rows)"
        else:
            status = f"ERROR: {result.get('error', 'Unknown')}"
        print(f"  {table:40} {status}")

    print(f"\n  Total rows in orphaned tables: {total_orphaned_rows}")

    print("\nLegacy Normalization Tables (4 tables):")
    print("-" * 70)
    total_legacy_rows = 0
    for table, result in legacy_results.items():
        if result['status'] == 'dropped':
            total_legacy_rows += result.get('row_count', 0)
            status = "DROPPED" if result['row_count'] == 0 else f"DROPPED ({result['row_count']} rows)"
        else:
            status = f"ERROR: {result.get('error', 'Unknown')}"
        print(f"  {table:40} {status}")

    print(f"\n  Total rows in legacy tables: {total_legacy_rows}")

    print("\n" + "="*70)
    total_dropped = sum(1 for r in orphaned_results.values() if r['status'] == 'dropped')
    total_dropped += sum(1 for r in legacy_results.values() if r['status'] == 'dropped')
    print(f"TOTAL TABLES DROPPED: {total_dropped} / 12")
    print(f"TOTAL DATA LOSS: {total_orphaned_rows + total_legacy_rows} rows")
    print("="*70 + "\n")


def main():
    """Main migration entry point"""
    print("\n" + "="*70)
    print("DATABASE MIGRATION: Drop Orphaned and Legacy Tables")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Migration: Round 2 Cleanup")
    print("="*70 + "\n")

    # Database path
    db_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'intervention_research.db'

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return

    print(f"Database: {db_path}")
    print(f"Size: {db_path.stat().st_size / (1024 * 1024):.2f} MB\n")

    # Step 1: Create backup
    print("Step 1: Creating backup...")
    backup_path = create_backup(db_path)
    print()

    # Step 2: Connect to database
    print("Step 2: Connecting to database...")
    conn = sqlite3.connect(db_path)
    print("Connected successfully\n")

    # Step 3: Verify tables exist
    print("Step 3: Verifying tables exist...")
    all_tables = ORPHANED_ANALYTICS_TABLES + LEGACY_NORMALIZATION_TABLES
    table_status = verify_tables_exist(conn, all_tables)

    existing_count = sum(1 for exists in table_status.values() if exists)
    print(f"Found {existing_count} / {len(all_tables)} tables to drop\n")

    # Step 4: Drop orphaned analytics tables
    print("Step 4: Dropping orphaned analytics tables...")
    orphaned_results = drop_tables(conn, ORPHANED_ANALYTICS_TABLES)
    print()

    # Step 5: Drop legacy normalization tables
    print("Step 5: Dropping legacy normalization tables...")
    legacy_results = drop_tables(conn, LEGACY_NORMALIZATION_TABLES)
    print()

    # Step 6: Verify drops
    print("Step 6: Verifying drops...")
    verification_success = verify_drops(conn, all_tables)
    if verification_success:
        print("All tables successfully dropped\n")
    else:
        print("Some tables were not dropped - check warnings above\n")

    # Step 7: Vacuum database to reclaim space
    print("Step 7: Vacuuming database to reclaim space...")
    old_size = db_path.stat().st_size / (1024 * 1024)
    conn.execute("VACUUM")
    conn.commit()
    new_size = db_path.stat().st_size / (1024 * 1024)
    space_saved = old_size - new_size
    print(f"Database vacuumed successfully")
    print(f"Size before: {old_size:.2f} MB")
    print(f"Size after: {new_size:.2f} MB")
    print(f"Space saved: {space_saved:.2f} MB\n")

    # Close connection
    conn.close()

    # Print summary
    print_migration_summary(orphaned_results, legacy_results)

    print("Migration completed successfully!")
    print(f"Backup saved at: {backup_path}\n")


if __name__ == "__main__":
    main()
