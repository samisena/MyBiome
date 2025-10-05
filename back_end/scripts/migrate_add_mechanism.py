"""
Migration script to add mechanism field to interventions table.

This migration:
1. Adds 'mechanism' TEXT column to interventions table
2. Preserves all existing data
3. Sets mechanism to NULL for existing records (can be backfilled later)

Usage:
    python -m back_end.scripts.migrate_add_mechanism
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from back_end.src.data.config import config, setup_logging

logger = setup_logging(__name__, 'migration_add_mechanism.log')


def check_column_exists(cursor, table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def migrate_add_mechanism(db_path: Path):
    """Add mechanism column to interventions table."""

    logger.info(f"Starting migration: Adding mechanism column to {db_path}")

    try:
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if column already exists
        if check_column_exists(cursor, 'interventions', 'mechanism'):
            logger.warning("Column 'mechanism' already exists in interventions table. Skipping migration.")
            conn.close()
            return True

        # Get current row count before migration
        cursor.execute("SELECT COUNT(*) FROM interventions")
        row_count_before = cursor.fetchone()[0]
        logger.info(f"Current interventions count: {row_count_before}")

        # Add mechanism column
        logger.info("Adding 'mechanism' column to interventions table...")
        cursor.execute("""
            ALTER TABLE interventions
            ADD COLUMN mechanism TEXT
        """)

        conn.commit()

        # Verify migration success
        cursor.execute("SELECT COUNT(*) FROM interventions")
        row_count_after = cursor.fetchone()[0]

        if row_count_before != row_count_after:
            raise Exception(f"Data loss detected! Before: {row_count_before}, After: {row_count_after}")

        # Verify column was added
        if not check_column_exists(cursor, 'interventions', 'mechanism'):
            raise Exception("Failed to add mechanism column")

        logger.info(f"Migration successful! Column 'mechanism' added to interventions table.")
        logger.info(f"All {row_count_after} existing interventions preserved (mechanism set to NULL)")
        logger.info("New extractions will populate the mechanism field automatically.")

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False


def main():
    """Run the migration on both database locations."""

    # Check both possible database locations
    db_locations = [
        Path(config.db_path),  # Main database path from config
        Path("back_end/data/processed/intervention_research.db"),  # Processed data path
        Path("back_end/data/intervention_research.db"),  # Alternative path
    ]

    success_count = 0

    for db_path in db_locations:
        if db_path.exists():
            logger.info(f"\nFound database at: {db_path}")
            if migrate_add_mechanism(db_path):
                success_count += 1
        else:
            logger.info(f"Database not found at: {db_path} (skipping)")

    if success_count > 0:
        print(f"\n[SUCCESS] Migration completed successfully for {success_count} database(s)")
        print("The 'mechanism' field is now available for new extractions.")
        print("Existing interventions have mechanism=NULL (can be backfilled later if needed).")
        return 0
    else:
        print("\n[FAILED] Migration failed or no databases found")
        return 1


if __name__ == "__main__":
    sys.exit(main())
