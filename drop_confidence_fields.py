#!/usr/bin/env python3
"""
Database Migration: Remove correlation_strength and extraction_confidence columns

This script removes the deprecated confidence fields from the interventions table:
- correlation_strength: Removed because it was arbitrary and not used in Phase 4b Bayesian scoring
- extraction_confidence: Removed because it was LLM's self-assessment, not objective study quality

IMPORTANT: This migration requires SQLite 3.35.0+ for ALTER TABLE DROP COLUMN support.
For older versions, the script will recreate the table (slower but compatible).

Usage:
    python drop_confidence_fields.py

The script will:
1. Check SQLite version
2. Create backup of database
3. Remove both columns using the appropriate method
4. Verify the migration
5. Print rollback instructions if needed
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime


def get_sqlite_version():
    """Get SQLite version."""
    conn = sqlite3.connect(":memory:")
    version = conn.execute("SELECT sqlite_version()").fetchone()[0]
    conn.close()
    return tuple(map(int, version.split('.')))


def backup_database(db_path):
    """Create a backup of the database."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
    shutil.copy2(db_path, backup_path)
    print(f"[OK] Backup created: {backup_path}")
    return backup_path


def drop_columns_modern(conn, cursor):
    """Drop columns using ALTER TABLE DROP COLUMN (SQLite 3.35.0+)."""
    print("\nUsing modern SQLite DROP COLUMN method...")

    # Step 1: Save and drop all views that might reference the columns
    cursor.execute("""
        SELECT name, sql FROM sqlite_master
        WHERE type='view'
        ORDER BY name
    """)
    views = cursor.fetchall()

    print(f"  [OK] Found {len(views)} views to temporarily drop")

    for view_name, view_sql in views:
        cursor.execute(f"DROP VIEW IF EXISTS {view_name}")
        print(f"  [OK] Dropped view: {view_name}")

    # Step 2: Drop the columns
    cursor.execute("ALTER TABLE interventions DROP COLUMN correlation_strength")
    print("  [OK] Dropped correlation_strength column")

    cursor.execute("ALTER TABLE interventions DROP COLUMN extraction_confidence")
    print("  [OK] Dropped extraction_confidence column")

    # Step 3: Recreate views (excluding references to dropped columns)
    for view_name, view_sql in views:
        try:
            # Remove references to dropped columns from view SQL
            updated_sql = view_sql.replace('i.correlation_strength,', '').replace(', i.correlation_strength', '')
            updated_sql = updated_sql.replace('i.extraction_confidence,', '').replace(', i.extraction_confidence', '')

            cursor.execute(updated_sql)
            print(f"  [OK] Recreated view: {view_name}")
        except sqlite3.OperationalError as e:
            print(f"  [WARNING] Could not recreate view {view_name}: {e}")
            print(f"           (View will need manual recreation)")

    conn.commit()


def drop_columns_legacy(conn, cursor):
    """Drop columns by recreating the table (SQLite < 3.35.0)."""
    print("\nUsing legacy table recreation method...")

    # Get current table schema (excluding the columns to drop)
    cursor.execute("PRAGMA table_info(interventions)")
    columns = cursor.fetchall()

    # Filter out the columns we want to drop
    kept_columns = [col for col in columns if col[1] not in ('correlation_strength', 'extraction_confidence')]

    # Build new schema
    column_defs = []
    for col in kept_columns:
        col_def = f"{col[1]} {col[2]}"
        if col[3]:  # NOT NULL
            col_def += " NOT NULL"
        if col[4] is not None:  # DEFAULT
            col_def += f" DEFAULT {col[4]}"
        if col[5]:  # PRIMARY KEY
            col_def += " PRIMARY KEY"
        column_defs.append(col_def)

    column_names = [col[1] for col in kept_columns]

    # Create new table
    new_schema = f"CREATE TABLE interventions_new ({', '.join(column_defs)})"
    cursor.execute(new_schema)
    print("  [OK] Created new table schema")

    # Copy data (excluding dropped columns)
    column_list = ', '.join(column_names)
    cursor.execute(f"""
        INSERT INTO interventions_new ({column_list})
        SELECT {column_list}
        FROM interventions
    """)
    print("  [OK] Copied data to new table")

    # Drop old table
    cursor.execute("DROP TABLE interventions")
    print("  [OK] Dropped old table")

    # Rename new table
    cursor.execute("ALTER TABLE interventions_new RENAME TO interventions")
    print("  [OK] Renamed new table to 'interventions'")

    # Recreate indexes (if any)
    cursor.execute("""
        SELECT sql FROM sqlite_master
        WHERE type='index' AND tbl_name='interventions'
        AND sql IS NOT NULL
    """)
    indexes = cursor.fetchall()

    for index_sql, in indexes:
        cursor.execute(index_sql)

    if indexes:
        print(f"  [OK] Recreated {len(indexes)} indexes")

    conn.commit()


def verify_migration(cursor):
    """Verify that the columns were removed."""
    print("\nVerifying migration...")

    cursor.execute("PRAGMA table_info(interventions)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]

    # Check that removed columns are gone
    if 'correlation_strength' in column_names:
        print("  [ERROR] correlation_strength column still exists!")
        return False
    else:
        print("  [OK] correlation_strength column removed")

    if 'extraction_confidence' in column_names:
        print("  [ERROR] extraction_confidence column still exists!")
        return False
    else:
        print("  [OK] extraction_confidence column removed")

    # Check that study_confidence still exists
    if 'study_confidence' not in column_names:
        print("  [ERROR] study_confidence column was accidentally removed!")
        return False
    else:
        print("  [OK] study_confidence column preserved")

    # Check row count
    cursor.execute("SELECT COUNT(*) FROM interventions")
    row_count = cursor.fetchone()[0]
    print(f"  [OK] Row count: {row_count}")

    return True


def main():
    """Main migration function."""
    print("=" * 70)
    print("Database Migration: Remove correlation_strength & extraction_confidence")
    print("=" * 70)

    # Database path (check both locations)
    db_path = Path(__file__).parent / "back_end" / "data" / "processed" / "intervention_research.db"

    # Fall back to main data folder if processed doesn't exist
    if not db_path.exists():
        db_path = Path(__file__).parent / "back_end" / "data" / "intervention_research.db"

    if not db_path.exists():
        print(f"\n[ERROR] Database not found at {db_path}")
        print("Please check the database path and try again.")
        return

    print(f"\nDatabase: {db_path}")

    # Check SQLite version
    version = get_sqlite_version()
    print(f"SQLite version: {'.'.join(map(str, version))}")

    # Create backup
    backup_path = backup_database(db_path)

    # Perform migration
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Choose migration method based on SQLite version
        if version >= (3, 35, 0):
            drop_columns_modern(conn, cursor)
        else:
            drop_columns_legacy(conn, cursor)

        # Verify migration
        if verify_migration(cursor):
            print("\n" + "=" * 70)
            print("[SUCCESS] MIGRATION COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"\nBackup location: {backup_path}")
            print("\nIf you need to rollback, run:")
            print(f"  copy {backup_path} {db_path}")
        else:
            print("\n" + "=" * 70)
            print("[FAILED] MIGRATION VERIFICATION FAILED!")
            print("=" * 70)
            print("\nRolling back to backup...")
            conn.close()
            shutil.copy2(backup_path, db_path)
            print(f"[OK] Restored from backup: {backup_path}")

        conn.close()

    except Exception as e:
        print(f"\n[ERROR] during migration: {e}")
        print("\nRolling back to backup...")
        try:
            conn.close()
        except:
            pass
        shutil.copy2(backup_path, db_path)
        print(f"[OK] Restored from backup: {backup_path}")
        raise


if __name__ == "__main__":
    main()
