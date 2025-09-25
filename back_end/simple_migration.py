#!/usr/bin/env python3
"""
Simple Migration Script - Normalize all records using compatible methods
"""

import sqlite3
import logging
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Set up logging for migration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'data/logs/simple_migration_{timestamp}.log'

    os.makedirs('data/logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def create_backup(db_path):
    """Create backup before migration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{db_path.replace('.db', '')}_simple_migration_backup_{timestamp}.db"

    # Copy database file
    import shutil
    shutil.copy2(db_path, backup_path)
    logging.info(f"Created backup: {backup_path}")
    return backup_path

def find_or_create_canonical(cursor, term, entity_type):
    """Simple normalization - exact match or create new canonical"""
    normalized_term = term.strip().lower()

    # Look for exact match (case insensitive)
    cursor.execute("""
        SELECT id, canonical_name
        FROM canonical_entities
        WHERE LOWER(TRIM(canonical_name)) = ? AND entity_type = ?
    """, (normalized_term, entity_type))

    result = cursor.fetchone()
    if result:
        return result[0], result[1], 'exact_match', 1.0

    # Create new canonical entity
    cursor.execute("""
        INSERT INTO canonical_entities (canonical_name, entity_type, confidence_score)
        VALUES (?, ?, ?)
    """, (term.strip(), entity_type, 1.0))

    canonical_id = cursor.lastrowid
    logging.info(f"Created new canonical: '{term}' -> ID {canonical_id}")
    return canonical_id, term.strip(), 'new_canonical', 1.0

def migrate_records():
    """Migrate all records using simple normalization"""

    db_path = 'data/processed/intervention_research.db'

    # Create backup
    backup_path = create_backup(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all records that need normalization
        cursor.execute("""
            SELECT id, intervention_name, health_condition
            FROM interventions
            WHERE normalized IS NULL OR normalized = 0
        """)

        records = cursor.fetchall()
        total_records = len(records)
        logging.info(f"Found {total_records} records to migrate")

        if total_records == 0:
            logging.info("No records need migration")
            return

        # Process records
        success_count = 0
        error_count = 0

        for i, record in enumerate(records, 1):
            try:
                record_id = record['id']
                intervention_name = record['intervention_name']
                health_condition = record['health_condition']

                # Process intervention
                intervention_canonical_id = None
                if intervention_name and intervention_name.strip():
                    canonical_id, canonical_name, method, confidence = find_or_create_canonical(
                        cursor, intervention_name, 'intervention'
                    )
                    intervention_canonical_id = canonical_id

                    # Create mapping entry
                    cursor.execute("""
                        INSERT OR IGNORE INTO entity_mappings
                        (raw_text, canonical_id, entity_type, confidence_score)
                        VALUES (?, ?, ?, ?)
                    """, (intervention_name, canonical_id, 'intervention', confidence))

                # Process condition
                condition_canonical_id = None
                if health_condition and health_condition.strip():
                    canonical_id, canonical_name, method, confidence = find_or_create_canonical(
                        cursor, health_condition, 'condition'
                    )
                    condition_canonical_id = canonical_id

                    # Create mapping entry
                    cursor.execute("""
                        INSERT OR IGNORE INTO entity_mappings
                        (raw_text, canonical_id, entity_type, confidence_score)
                        VALUES (?, ?, ?, ?)
                    """, (health_condition, canonical_id, 'condition', confidence))

                # Update intervention record
                cursor.execute("""
                    UPDATE interventions
                    SET intervention_canonical_id = ?, condition_canonical_id = ?, normalized = 1
                    WHERE id = ?
                """, (intervention_canonical_id, condition_canonical_id, record_id))

                success_count += 1

                if i % 100 == 0:
                    logging.info(f"Processed {i}/{total_records} records...")
                    conn.commit()

            except Exception as e:
                error_count += 1
                logging.error(f"Error processing record {record_id}: {str(e)}")
                continue

        # Final commit
        conn.commit()

        logging.info(f"""
================================================================================
MIGRATION COMPLETE
================================================================================
Total records processed: {total_records}
Successful: {success_count}
Errors: {error_count}
Success rate: {(success_count/total_records)*100:.1f}%
Backup created: {backup_path}
================================================================================
        """)

if __name__ == "__main__":
    log_file = setup_logging()

    try:
        migrate_records()
        print(f"\nMigration complete! Check log file: {log_file}")
    except Exception as e:
        logging.error(f"Migration failed: {str(e)}")
        raise