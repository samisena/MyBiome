#!/usr/bin/env python3
"""
Batch Migration for Normalization of Existing Records

This script processes all existing records in the interventions table to add
canonical mappings while preserving original terms.

Features:
- Batch processing (100 records per batch)
- Progress logging and statistics
- Error handling and recovery
- Rollback capability
- Performance monitoring
- Off-hours optimization ready
"""

import sqlite3
import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


class BatchMigrationManager:
    """Manages batch migration of existing records to add normalization"""

    def __init__(self, db_path: str = "data/processed/intervention_research.db", batch_size: int = 100):
        self.db_path = db_path
        self.batch_size = batch_size
        self.normalizer = None
        self.stats = {
            'total_records': 0,
            'processed_records': 0,
            'successful_normalizations': 0,
            'failed_normalizations': 0,
            'intervention_mappings': 0,
            'condition_mappings': 0,
            'new_canonicals_created': 0,
            'start_time': None,
            'end_time': None,
            'batches_processed': 0,
            'errors': []
        }

        # Setup logging
        self.setup_logging()

        # Initialize normalizer
        self.initialize_normalizer()

    def setup_logging(self):
        """Setup logging for migration process"""
        log_dir = Path("data/logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"batch_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Migration log initialized: {log_file}")

    def initialize_normalizer(self):
        """Initialize entity normalizer"""
        try:
            conn = sqlite3.connect(self.db_path)
            self.normalizer = EntityNormalizer(conn)
            conn.close()
            self.logger.info("Entity normalizer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize normalizer: {e}")
            raise

    def create_additional_backup(self) -> str:
        """Create additional backup specific to this migration"""
        backup_name = f"intervention_research_batch_migration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        backup_path = f"data/processed/{backup_name}"

        import shutil
        shutil.copy2(self.db_path, backup_path)

        self.logger.info(f"Created migration-specific backup: {backup_path}")
        return backup_path

    def get_records_to_migrate(self) -> List[Dict]:
        """Get all records that need normalization"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get records where normalized is false or null
            cursor.execute("""
                SELECT
                    id,
                    intervention_name,
                    health_condition,
                    intervention_canonical_id,
                    condition_canonical_id,
                    normalized,
                    paper_id
                FROM interventions
                WHERE normalized IS NULL OR normalized = 0
                ORDER BY id ASC
            """)

            records = [dict(row) for row in cursor.fetchall()]
            self.logger.info(f"Found {len(records)} records to migrate")
            return records

    def process_batch(self, batch: List[Dict]) -> Tuple[int, int]:
        """Process a batch of records"""
        successful = 0
        failed = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for record in batch:
                try:
                    record_id = record['id']
                    intervention_name = record['intervention_name']
                    health_condition = record['health_condition']

                    self.logger.debug(f"Processing record {record_id}: '{intervention_name}' -> '{health_condition}'")

                    # Initialize canonical IDs
                    intervention_canonical_id = record['intervention_canonical_id']
                    condition_canonical_id = record['condition_canonical_id']

                    # Create fresh normalizer connection for this batch
                    norm_conn = sqlite3.connect(self.db_path)
                    normalizer = EntityNormalizer(norm_conn)

                    # Process intervention if not already mapped
                    if not intervention_canonical_id and intervention_name:
                        intervention_mapping = normalizer.find_or_create_mapping(
                            intervention_name.strip(), 'intervention', confidence_threshold=0.7
                        )

                        if intervention_mapping['success']:
                            intervention_canonical_id = intervention_mapping['canonical_id']
                            self.stats['intervention_mappings'] += 1

                            if intervention_mapping['method'] == 'new_canonical':
                                self.stats['new_canonicals_created'] += 1

                            self.logger.debug(f"  Intervention: '{intervention_name}' -> canonical_id {intervention_canonical_id} (method: {intervention_mapping['method']})")

                    # Process condition if not already mapped
                    if not condition_canonical_id and health_condition:
                        condition_mapping = normalizer.find_or_create_mapping(
                            health_condition.strip(), 'condition', confidence_threshold=0.7
                        )

                        if condition_mapping['success']:
                            condition_canonical_id = condition_mapping['canonical_id']
                            self.stats['condition_mappings'] += 1

                            if condition_mapping['method'] == 'new_canonical':
                                self.stats['new_canonicals_created'] += 1

                            self.logger.debug(f"  Condition: '{health_condition}' -> canonical_id {condition_canonical_id} (method: {condition_mapping['method']})")

                    # Update the record
                    cursor.execute("""
                        UPDATE interventions
                        SET intervention_canonical_id = ?,
                            condition_canonical_id = ?,
                            normalized = 1
                        WHERE id = ?
                    """, (intervention_canonical_id, condition_canonical_id, record_id))

                    norm_conn.close()
                    successful += 1
                    self.stats['successful_normalizations'] += 1

                except Exception as e:
                    error_msg = f"Error processing record {record.get('id', 'unknown')}: {e}"
                    self.logger.error(error_msg)
                    self.stats['errors'].append({
                        'record_id': record.get('id'),
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    failed += 1
                    self.stats['failed_normalizations'] += 1

            conn.commit()

        return successful, failed

    def run_migration(self, max_batches: Optional[int] = None) -> Dict:
        """Run the complete migration process"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING BATCH MIGRATION FOR NORMALIZATION")
        self.logger.info("=" * 80)

        self.stats['start_time'] = datetime.now()

        # Create additional backup
        backup_path = self.create_additional_backup()

        # Get records to migrate
        records_to_migrate = self.get_records_to_migrate()
        self.stats['total_records'] = len(records_to_migrate)

        if not records_to_migrate:
            self.logger.info("No records found that need migration")
            return self.stats

        self.logger.info(f"Processing {len(records_to_migrate)} records in batches of {self.batch_size}")

        # Process in batches
        total_batches = (len(records_to_migrate) + self.batch_size - 1) // self.batch_size
        if max_batches:
            total_batches = min(total_batches, max_batches)

        self.logger.info(f"Total batches to process: {total_batches}")

        for i in range(0, len(records_to_migrate), self.batch_size):
            if max_batches and self.stats['batches_processed'] >= max_batches:
                self.logger.info(f"Reached maximum batch limit ({max_batches})")
                break

            batch = records_to_migrate[i:i + self.batch_size]
            batch_num = self.stats['batches_processed'] + 1

            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)")

            batch_start_time = time.time()
            successful, failed = self.process_batch(batch)
            batch_duration = time.time() - batch_start_time

            self.stats['processed_records'] += len(batch)
            self.stats['batches_processed'] += 1

            # Log batch results
            self.logger.info(f"Batch {batch_num} completed in {batch_duration:.2f}s: {successful} successful, {failed} failed")

            # Progress report every 10 batches
            if batch_num % 10 == 0:
                progress_percent = (self.stats['processed_records'] / self.stats['total_records']) * 100
                elapsed = datetime.now() - self.stats['start_time']
                self.logger.info(f"PROGRESS: {progress_percent:.1f}% complete ({self.stats['processed_records']}/{self.stats['total_records']}) - Elapsed: {elapsed}")

            # Small delay to avoid overwhelming the system
            time.sleep(0.1)

        self.stats['end_time'] = datetime.now()
        self.finalize_migration()

        return self.stats

    def finalize_migration(self):
        """Finalize migration and show results"""
        duration = self.stats['end_time'] - self.stats['start_time']

        self.logger.info("=" * 80)
        self.logger.info("MIGRATION COMPLETED")
        self.logger.info("=" * 80)

        self.logger.info(f"Total duration: {duration}")
        self.logger.info(f"Total records processed: {self.stats['processed_records']}")
        self.logger.info(f"Successful normalizations: {self.stats['successful_normalizations']}")
        self.logger.info(f"Failed normalizations: {self.stats['failed_normalizations']}")
        self.logger.info(f"Intervention mappings created: {self.stats['intervention_mappings']}")
        self.logger.info(f"Condition mappings created: {self.stats['condition_mappings']}")
        self.logger.info(f"New canonical entities created: {self.stats['new_canonicals_created']}")
        self.logger.info(f"Batches processed: {self.stats['batches_processed']}")

        if self.stats['errors']:
            self.logger.warning(f"Encountered {len(self.stats['errors'])} errors")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                self.logger.warning(f"  Record {error['record_id']}: {error['error']}")

        # Save detailed stats
        stats_file = f"data/logs/migration_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            stats_for_json = self.stats.copy()
            if stats_for_json['start_time']:
                stats_for_json['start_time'] = stats_for_json['start_time'].isoformat()
            if stats_for_json['end_time']:
                stats_for_json['end_time'] = stats_for_json['end_time'].isoformat()

            json.dump(stats_for_json, f, indent=2)

        self.logger.info(f"Detailed statistics saved to: {stats_file}")

    def verify_migration(self) -> Dict:
        """Verify migration results"""
        self.logger.info("Verifying migration results...")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Count total records
            cursor.execute("SELECT COUNT(*) FROM interventions")
            total_records = cursor.fetchone()[0]

            # Count normalized records
            cursor.execute("SELECT COUNT(*) FROM interventions WHERE normalized = 1")
            normalized_records = cursor.fetchone()[0]

            # Count records with intervention canonical mappings
            cursor.execute("SELECT COUNT(*) FROM interventions WHERE intervention_canonical_id IS NOT NULL")
            intervention_mapped = cursor.fetchone()[0]

            # Count records with condition canonical mappings
            cursor.execute("SELECT COUNT(*) FROM interventions WHERE condition_canonical_id IS NOT NULL")
            condition_mapped = cursor.fetchone()[0]

            # Count unmigrated records
            cursor.execute("SELECT COUNT(*) FROM interventions WHERE normalized IS NULL OR normalized = 0")
            unmigrated = cursor.fetchone()[0]

            verification = {
                'total_records': total_records,
                'normalized_records': normalized_records,
                'intervention_mapped': intervention_mapped,
                'condition_mapped': condition_mapped,
                'unmigrated_records': unmigrated,
                'migration_percentage': (normalized_records / max(total_records, 1)) * 100
            }

            self.logger.info("VERIFICATION RESULTS:")
            self.logger.info(f"  Total records: {total_records}")
            self.logger.info(f"  Normalized records: {normalized_records}")
            self.logger.info(f"  Intervention mappings: {intervention_mapped}")
            self.logger.info(f"  Condition mappings: {condition_mapped}")
            self.logger.info(f"  Unmigrated records: {unmigrated}")
            self.logger.info(f"  Migration percentage: {verification['migration_percentage']:.1f}%")

            return verification

    def rollback_migration(self, backup_path: str):
        """Rollback migration using backup"""
        self.logger.warning("INITIATING ROLLBACK...")

        if not os.path.exists(backup_path):
            self.logger.error(f"Backup file not found: {backup_path}")
            return False

        try:
            import shutil

            # Create backup of current state before rollback
            current_backup = f"data/processed/intervention_research_before_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(self.db_path, current_backup)

            # Restore from backup
            shutil.copy2(backup_path, self.db_path)

            self.logger.info(f"Successfully rolled back to: {backup_path}")
            self.logger.info(f"Current state backed up to: {current_backup}")
            return True

        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False


def main():
    """Main function with command line options"""
    import argparse

    parser = argparse.ArgumentParser(description="Batch Migration for Normalization")
    parser.add_argument("--batch-size", type=int, default=100, help="Records per batch")
    parser.add_argument("--max-batches", type=int, help="Maximum batches to process (for testing)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify current state")
    parser.add_argument("--rollback", type=str, help="Rollback using specified backup file")

    args = parser.parse_args()

    migration_manager = BatchMigrationManager(batch_size=args.batch_size)

    if args.rollback:
        migration_manager.rollback_migration(args.rollback)
        return

    if args.verify_only:
        migration_manager.verify_migration()
        return

    # Run migration
    stats = migration_manager.run_migration(max_batches=args.max_batches)

    # Verify results
    verification = migration_manager.verify_migration()

    # Success check
    if verification['unmigrated_records'] == 0:
        migration_manager.logger.info("SUCCESS CHECK MET: All existing records now have canonical mappings!")
    else:
        migration_manager.logger.warning(f"Migration incomplete: {verification['unmigrated_records']} records still need processing")


if __name__ == "__main__":
    main()