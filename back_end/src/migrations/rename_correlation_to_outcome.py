"""
Database migration: Rename correlation_type to outcome_type with health-impact semantics.

Migration Strategy:
1. Add new outcome_type column
2. Copy and transform data from correlation_type
3. Validate all rows have valid outcome_type values
4. Drop old correlation_type column

Value Mapping:
- positive → improves (intervention makes patient healthier)
- negative → worsens (intervention makes patient sicker)
- neutral → no_effect (no measurable health impact)
- inconclusive → inconclusive (mixed/unclear evidence)
"""

import sqlite3
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from back_end.src.data.config import setup_logging

logger = setup_logging(__name__, 'migration_correlation_to_outcome.log')


class CorrelationToOutcomeMigration:
    """Migrate correlation_type column to outcome_type with health-impact semantics."""

    # Value mapping from old to new terminology
    VALUE_MAPPING = {
        'positive': 'improves',
        'negative': 'worsens',
        'neutral': 'no_effect',
        'inconclusive': 'inconclusive',
        # Handle variations
        'positive_correlation': 'improves',
        'negative_correlation': 'worsens',
        'no_correlation': 'no_effect'
    }

    VALID_NEW_VALUES = ['improves', 'worsens', 'no_effect', 'inconclusive']

    def __init__(self, db_path: str, dry_run: bool = False):
        """
        Initialize migration.

        Args:
            db_path: Path to SQLite database
            dry_run: If True, preview changes without applying them
        """
        self.db_path = db_path
        self.dry_run = dry_run
        self.conn = None

    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database: {self.db_path}")

    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def create_backup(self):
        """Create backup of interventions table before migration."""
        cursor = self.conn.cursor()

        # Drop backup table if exists
        cursor.execute("DROP TABLE IF EXISTS interventions_backup_correlation_migration")

        # Create backup table
        cursor.execute("""
            CREATE TABLE interventions_backup_correlation_migration AS
            SELECT * FROM interventions
        """)

        count = cursor.execute("SELECT COUNT(*) FROM interventions_backup_correlation_migration").fetchone()[0]
        logger.info(f"Created backup table with {count} rows")

        self.conn.commit()

    def analyze_current_values(self) -> Dict[str, int]:
        """Analyze distribution of current correlation_type values."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT correlation_type, COUNT(*) as count
            FROM interventions
            GROUP BY correlation_type
            ORDER BY count DESC
        """)

        distribution = {}
        for row in cursor.fetchall():
            value = row['correlation_type'] if row['correlation_type'] else 'NULL'
            distribution[value] = row['count']

        logger.info("Current correlation_type distribution:")
        for value, count in distribution.items():
            logger.info(f"  {value}: {count} rows")

        return distribution

    def preview_migration(self) -> Dict[str, Tuple[str, int]]:
        """Preview how values will be transformed."""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT correlation_type, COUNT(*) as count
            FROM interventions
            GROUP BY correlation_type
        """)

        preview = {}
        unmapped = []

        for row in cursor.fetchall():
            old_value = row['correlation_type']
            count = row['count']

            if old_value is None:
                new_value = 'no_effect'  # Default for NULL values
            else:
                new_value = self.VALUE_MAPPING.get(old_value.lower())

            if new_value is None:
                unmapped.append((old_value, count))
                new_value = 'UNMAPPED'

            preview[old_value if old_value else 'NULL'] = (new_value, count)

        logger.info("\nMigration Preview:")
        for old, (new, count) in preview.items():
            logger.info(f"  {old} → {new} ({count} rows)")

        if unmapped:
            logger.warning(f"\nWARNING: {len(unmapped)} unmapped values found!")
            for value, count in unmapped:
                logger.warning(f"  {value}: {count} rows")

        return preview

    def add_outcome_type_column(self):
        """Add new outcome_type column."""
        cursor = self.conn.cursor()

        try:
            cursor.execute("""
                ALTER TABLE interventions
                ADD COLUMN outcome_type TEXT
            """)
            logger.info("Added outcome_type column")
            self.conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                logger.warning("outcome_type column already exists")
            else:
                raise

    def migrate_values(self) -> Dict[str, int]:
        """Migrate values from correlation_type to outcome_type."""
        cursor = self.conn.cursor()

        stats = {
            'migrated': 0,
            'unmapped': 0,
            'null_defaults': 0
        }

        # Get all rows
        cursor.execute("SELECT id, correlation_type FROM interventions")
        rows = cursor.fetchall()

        for row in rows:
            intervention_id = row['id']
            old_value = row['correlation_type']

            if old_value is None:
                new_value = 'no_effect'
                stats['null_defaults'] += 1
            else:
                new_value = self.VALUE_MAPPING.get(old_value.lower())

            if new_value is None:
                logger.warning(f"Unmapped value for intervention {intervention_id}: {old_value}")
                new_value = 'no_effect'  # Safe default
                stats['unmapped'] += 1
            else:
                stats['migrated'] += 1

            # Update row
            cursor.execute("""
                UPDATE interventions
                SET outcome_type = ?
                WHERE id = ?
            """, (new_value, intervention_id))

        self.conn.commit()

        logger.info(f"\nMigration Statistics:")
        logger.info(f"  Successfully migrated: {stats['migrated']}")
        logger.info(f"  NULL → no_effect: {stats['null_defaults']}")
        logger.info(f"  Unmapped (defaulted): {stats['unmapped']}")

        return stats

    def validate_migration(self) -> bool:
        """Validate that all rows have valid outcome_type values."""
        cursor = self.conn.cursor()

        # Check for NULL values
        cursor.execute("SELECT COUNT(*) FROM interventions WHERE outcome_type IS NULL")
        null_count = cursor.fetchone()[0]

        # Check for invalid values
        placeholders = ','.join('?' * len(self.VALID_NEW_VALUES))
        cursor.execute(f"""
            SELECT COUNT(*) FROM interventions
            WHERE outcome_type NOT IN ({placeholders})
        """, self.VALID_NEW_VALUES)
        invalid_count = cursor.fetchone()[0]

        # Get distribution of new values
        cursor.execute("""
            SELECT outcome_type, COUNT(*) as count
            FROM interventions
            GROUP BY outcome_type
            ORDER BY count DESC
        """)

        logger.info("\nValidation Results:")
        logger.info(f"  NULL values: {null_count}")
        logger.info(f"  Invalid values: {invalid_count}")
        logger.info("\nFinal outcome_type distribution:")

        for row in cursor.fetchall():
            logger.info(f"  {row['outcome_type']}: {row['count']} rows")

        validation_passed = null_count == 0 and invalid_count == 0

        if validation_passed:
            logger.info("\n✓ Validation PASSED")
        else:
            logger.error("\n✗ Validation FAILED")

        return validation_passed

    def drop_old_column(self):
        """Drop old correlation_type column."""
        # SQLite doesn't support DROP COLUMN directly, need to recreate table
        cursor = self.conn.cursor()

        logger.info("\nRecreating interventions table without correlation_type column...")

        # Step 1: Save and drop dependent views
        cursor.execute("""
            SELECT name, sql FROM sqlite_master
            WHERE type='view' AND sql LIKE '%interventions%'
        """)
        views = cursor.fetchall()
        logger.info(f"Found {len(views)} dependent views to recreate")

        for view_name, _ in views:
            logger.info(f"  Dropping view: {view_name}")
            cursor.execute(f"DROP VIEW IF EXISTS {view_name}")

        # Get current table schema
        cursor.execute("PRAGMA table_info(interventions)")
        columns = cursor.fetchall()

        # Build new schema (exclude correlation_type)
        new_columns = []
        for col in columns:
            col_name = col['name']
            if col_name != 'correlation_type':
                col_type = col['type']
                not_null = ' NOT NULL' if col['notnull'] else ''
                default = f" DEFAULT {col['dflt_value']}" if col['dflt_value'] else ''
                pk = ' PRIMARY KEY' if col['pk'] else ''
                new_columns.append(f"{col_name} {col_type}{not_null}{default}{pk}")

        new_schema = ', '.join(new_columns)

        # Create new table
        cursor.execute(f"""
            CREATE TABLE interventions_new (
                {new_schema}
            )
        """)

        # Copy data
        column_names = [col['name'] for col in columns if col['name'] != 'correlation_type']
        columns_str = ', '.join(column_names)

        cursor.execute(f"""
            INSERT INTO interventions_new ({columns_str})
            SELECT {columns_str} FROM interventions
        """)

        # Drop old table
        cursor.execute("DROP TABLE interventions")

        # Rename new table
        cursor.execute("ALTER TABLE interventions_new RENAME TO interventions")

        logger.info("Old correlation_type column dropped")

        # Step 5: Recreate views with updated column names
        for view_name, view_sql in views:
            if view_sql:
                logger.info(f"  Recreating view: {view_name}")
                # Replace correlation_type with outcome_type in view definition
                updated_sql = view_sql.replace('correlation_type', 'outcome_type')
                try:
                    cursor.execute(updated_sql)
                    logger.info(f"    ✓ View recreated successfully")
                except Exception as e:
                    logger.warning(f"    ⚠ Could not recreate view automatically: {e}")
                    logger.warning(f"    Original SQL: {view_sql}")

        self.conn.commit()

    def run_migration(self):
        """Execute complete migration process."""
        try:
            self.connect()

            logger.info("="*60)
            logger.info("STARTING MIGRATION: correlation_type → outcome_type")
            logger.info("="*60)

            # Step 1: Analyze current state
            logger.info("\n[1/7] Analyzing current data...")
            self.analyze_current_values()

            # Step 2: Preview migration
            logger.info("\n[2/7] Previewing migration...")
            preview = self.preview_migration()

            if self.dry_run:
                logger.info("\n[DRY RUN] Migration preview complete. No changes applied.")
                return

            # Step 3: Create backup
            logger.info("\n[3/7] Creating backup...")
            self.create_backup()

            # Step 4: Add new column
            logger.info("\n[4/7] Adding outcome_type column...")
            self.add_outcome_type_column()

            # Step 5: Migrate values
            logger.info("\n[5/7] Migrating values...")
            stats = self.migrate_values()

            # Step 6: Validate
            logger.info("\n[6/7] Validating migration...")
            validation_passed = self.validate_migration()

            if not validation_passed:
                logger.error("\nValidation failed! Rolling back...")
                self.conn.rollback()
                logger.info("Rollback complete. Database unchanged.")
                return

            # Step 7: Drop old column
            logger.info("\n[7/7] Dropping old correlation_type column...")
            self.drop_old_column()

            logger.info("\n" + "="*60)
            logger.info("MIGRATION COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total rows migrated: {stats['migrated'] + stats['null_defaults']}")
            logger.info("Backup table: interventions_backup_correlation_migration")

        except Exception as e:
            logger.error(f"\nMigration failed with error: {e}")
            if self.conn:
                self.conn.rollback()
                logger.info("Rolled back changes")
            raise
        finally:
            self.disconnect()


def main():
    """Run migration with command-line options."""
    import argparse

    parser = argparse.ArgumentParser(description='Migrate correlation_type to outcome_type')
    parser.add_argument('--db-path', default='back_end/data/medical_research.db',
                       help='Path to SQLite database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without applying them')

    args = parser.parse_args()

    # Convert to absolute path
    db_path = Path(args.db_path)
    if not db_path.is_absolute():
        db_path = project_root / db_path

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    logger.info(f"Database path: {db_path}")

    migration = CorrelationToOutcomeMigration(str(db_path), dry_run=args.dry_run)
    migration.run_migration()


if __name__ == '__main__':
    main()
