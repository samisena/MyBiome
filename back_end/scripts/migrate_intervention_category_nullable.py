"""
Migration script to make intervention_category nullable.
This allows intervention categorization to happen as a separate phase.
"""

import sqlite3
import os
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "processed" / "intervention_research.db"

def migrate_intervention_category_nullable():
    """Remove NOT NULL constraint from intervention_category field."""

    print(f"Migrating database: {DB_PATH}")

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Step 1: Check current schema
        cursor.execute("PRAGMA table_info(interventions)")
        schema = cursor.fetchall()

        print("\nCurrent schema for intervention_category:")
        for col in schema:
            if col[1] == 'intervention_category':
                print(f"  Name: {col[1]}, Type: {col[2]}, NotNull: {col[3]}, Default: {col[4]}")

        # Step 2: Create new table with updated schema
        print("\nCreating new table with nullable intervention_category...")
        cursor.execute('''
            CREATE TABLE interventions_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id TEXT NOT NULL,
                intervention_category TEXT,
                intervention_name TEXT NOT NULL,
                intervention_details TEXT,
                health_condition TEXT NOT NULL,
                correlation_type TEXT CHECK(correlation_type IN ('positive', 'negative', 'neutral', 'inconclusive')),
                correlation_strength REAL CHECK(correlation_strength >= 0 AND correlation_strength <= 1),
                extraction_confidence REAL CHECK(extraction_confidence >= 0 AND extraction_confidence <= 1),
                study_confidence REAL CHECK(study_confidence >= 0 AND study_confidence <= 1),
                sample_size INTEGER,
                study_duration TEXT,
                study_type TEXT,
                population_details TEXT,
                supporting_quote TEXT,
                delivery_method TEXT,
                severity TEXT CHECK(severity IN ('mild', 'moderate', 'severe')),
                adverse_effects TEXT,
                cost_category TEXT CHECK(cost_category IN ('low', 'medium', 'high')),
                extraction_model TEXT NOT NULL,
                extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                validation_issues TEXT,
                verification_model TEXT,
                verification_timestamp TIMESTAMP,
                verification_confidence REAL CHECK(verification_confidence >= 0 AND verification_confidence <= 1),
                human_reviewed BOOLEAN DEFAULT FALSE,
                human_reviewer TEXT,
                review_timestamp TIMESTAMP,
                review_notes TEXT,
                consensus_confidence REAL,
                models_contributing TEXT,
                intervention_canonical_id INTEGER,
                condition_canonical_id INTEGER,
                normalized BOOLEAN,
                condition_wording_source TEXT,
                condition_wording_confidence REAL,
                original_condition_wordings TEXT,
                model_agreement TEXT,
                models_used TEXT,
                raw_extraction_count INTEGER DEFAULT 1,
                condition_category TEXT,
                FOREIGN KEY (intervention_canonical_id) REFERENCES canonical_entities(id),
                FOREIGN KEY (condition_canonical_id) REFERENCES canonical_entities(id)
            )
        ''')

        # Step 3: Copy all data from old table to new table
        print("Copying data from old table to new table...")
        cursor.execute('''
            INSERT INTO interventions_new
            SELECT * FROM interventions
        ''')

        rows_copied = cursor.rowcount
        print(f"  Copied {rows_copied} rows")

        # Step 4: Drop old table
        print("Dropping old table...")
        cursor.execute('DROP TABLE interventions')

        # Step 5: Rename new table to original name
        print("Renaming new table...")
        cursor.execute('ALTER TABLE interventions_new RENAME TO interventions')

        # Step 6: Verify new schema
        cursor.execute("PRAGMA table_info(interventions)")
        new_schema = cursor.fetchall()

        print("\nNew schema for intervention_category:")
        for col in new_schema:
            if col[1] == 'intervention_category':
                print(f"  Name: {col[1]}, Type: {col[2]}, NotNull: {col[3]}, Default: {col[4]}")

        # Step 7: Verify data integrity
        cursor.execute("SELECT COUNT(*) FROM interventions")
        final_count = cursor.fetchone()[0]
        print(f"\nFinal row count: {final_count}")

        if final_count == rows_copied:
            print("  Data integrity verified!")
        else:
            print(f"  WARNING: Row count mismatch! Original: {rows_copied}, Final: {final_count}")
            conn.rollback()
            return False

        # Commit changes
        conn.commit()
        print("\nMigration completed successfully!")
        return True

    except Exception as e:
        print(f"\nERROR during migration: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()

if __name__ == "__main__":
    success = migrate_intervention_category_nullable()
    exit(0 if success else 1)
