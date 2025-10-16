"""
Migration script to remove correlation_strength column from interventions table.

This migration removes the deprecated correlation_strength field which has been
replaced by the findings array field that contains actual quantitative results
from papers.

Run this migration after updating all code to remove correlation_strength references.

Usage:
    python -m back_end.src.migrations.drop_correlation_strength_column
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from back_end.src.data.config import config, setup_logging

logger = setup_logging(__name__, 'migration_drop_correlation_strength.log')


def backup_database(db_path: Path) -> Path:
    """Create a backup of the database before migration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_before_drop_correlation_strength_{timestamp}{db_path.suffix}"

    logger.info(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    logger.info(f"Backup created successfully")

    return backup_path


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def drop_correlation_strength_column():
    """
    Drop the correlation_strength column from the interventions table.

    SQLite doesn't support DROP COLUMN directly (before version 3.35.0),
    so we need to:
    1. Create a new table without the column
    2. Copy data from old table to new table
    3. Drop old table
    4. Rename new table to old name
    5. Recreate indexes and constraints
    """
    db_path = Path(config.db_path)

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return False

    # Create backup first
    backup_path = backup_database(db_path)
    logger.info(f"Backup saved to: {backup_path}")

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if column exists
        if not column_exists(conn, 'interventions', 'correlation_strength'):
            logger.info("Column 'correlation_strength' does not exist. Migration not needed.")
            conn.close()
            return True

        logger.info("Starting migration to drop correlation_strength column...")

        # Step 1: Create new table without correlation_strength
        logger.info("Step 1: Creating new interventions table without correlation_strength...")
        cursor.execute('''
            CREATE TABLE interventions_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                intervention_category TEXT,
                intervention_name TEXT NOT NULL,
                intervention_details TEXT,
                health_condition TEXT NOT NULL,
                mechanism TEXT,
                correlation_type TEXT CHECK(correlation_type IN ('positive', 'negative', 'neutral', 'inconclusive')),

                -- Dual confidence metrics
                extraction_confidence REAL CHECK(extraction_confidence >= 0 AND extraction_confidence <= 1),
                study_confidence REAL CHECK(study_confidence >= 0 AND study_confidence <= 1),

                -- Study details
                sample_size INTEGER,
                study_duration TEXT,
                study_type TEXT,
                population_details TEXT,
                supporting_quote TEXT,

                -- Additional optional fields
                delivery_method TEXT,
                severity TEXT,
                adverse_effects TEXT,
                cost_category TEXT,

                -- New hierarchical format fields (Phase 2.5)
                study_focus TEXT,
                measured_metrics TEXT,
                findings TEXT,
                study_location TEXT,
                publisher TEXT,

                -- LLM metadata
                extraction_model TEXT,

                -- Consensus metadata (from dual-model extraction - DEPRECATED)
                consensus_confidence REAL,
                model_agreement REAL,
                models_used TEXT,
                raw_extraction_count INTEGER,
                models_contributing TEXT,

                -- Normalization metadata
                intervention_canonical_id INTEGER,
                condition_canonical_id INTEGER,
                condition_category TEXT,
                normalized BOOLEAN DEFAULT FALSE,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers(pmid)
            )
        ''')

        # Step 2: Copy data from old table to new table (excluding correlation_strength)
        logger.info("Step 2: Copying data to new table...")
        cursor.execute('''
            INSERT INTO interventions_new
            (id, paper_id, intervention_category, intervention_name, intervention_details,
             health_condition, mechanism, correlation_type,
             extraction_confidence, study_confidence,
             sample_size, study_duration, study_type, population_details, supporting_quote,
             delivery_method, severity, adverse_effects, cost_category,
             study_focus, measured_metrics, findings, study_location, publisher,
             extraction_model,
             consensus_confidence, model_agreement, models_used, raw_extraction_count, models_contributing,
             intervention_canonical_id, condition_canonical_id, condition_category, normalized,
             created_at)
            SELECT
             id, paper_id, intervention_category, intervention_name, intervention_details,
             health_condition, mechanism, correlation_type,
             extraction_confidence, study_confidence,
             sample_size, study_duration, study_type, population_details, supporting_quote,
             delivery_method, severity, adverse_effects, cost_category,
             study_focus, measured_metrics, findings, study_location, publisher,
             extraction_model,
             consensus_confidence, model_agreement, models_used, raw_extraction_count, models_contributing,
             intervention_canonical_id, condition_canonical_id, condition_category, normalized,
             created_at
            FROM interventions
        ''')

        rows_copied = cursor.rowcount
        logger.info(f"Copied {rows_copied} rows to new table")

        # Step 3: Drop old table
        logger.info("Step 3: Dropping old interventions table...")
        cursor.execute('DROP TABLE interventions')

        # Step 4: Rename new table to old name
        logger.info("Step 4: Renaming interventions_new to interventions...")
        cursor.execute('ALTER TABLE interventions_new RENAME TO interventions')

        # Step 5: Recreate indexes
        logger.info("Step 5: Recreating indexes...")
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_interventions_paper_id
            ON interventions(paper_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_interventions_category
            ON interventions(intervention_category)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_interventions_condition
            ON interventions(health_condition)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_interventions_canonical
            ON interventions(intervention_canonical_id, condition_canonical_id)
        ''')

        # Commit changes
        conn.commit()
        logger.info("Migration completed successfully!")

        # Verify the column is gone
        if column_exists(conn, 'interventions', 'correlation_strength'):
            logger.error("ERROR: Column still exists after migration!")
            conn.close()
            return False

        logger.info("Verification passed: correlation_strength column successfully removed")
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        logger.error("Database backup is available at: {backup_path}")
        logger.error("You can restore the database by copying the backup over the current database.")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False


def main():
    """Run the migration."""
    print("=" * 70)
    print("MIGRATION: Drop correlation_strength column from interventions table")
    print("=" * 70)
    print()
    print("This migration will:")
    print("  1. Create a backup of your database")
    print("  2. Create a new interventions table without correlation_strength")
    print("  3. Copy all data to the new table")
    print("  4. Replace the old table with the new one")
    print("  5. Recreate indexes")
    print()

    response = input("Do you want to proceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Migration cancelled.")
        return

    print("\nStarting migration...")
    success = drop_correlation_strength_column()

    if success:
        print("\nMigration completed successfully!")
        print("The correlation_strength column has been removed.")
    else:
        print("\nMigration failed. Check the logs for details.")
        print("Your database backup is safe and can be restored if needed.")


if __name__ == '__main__':
    main()
