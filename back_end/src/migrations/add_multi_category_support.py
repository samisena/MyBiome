"""
Migration: Add Multi-Category Support

Enables interventions, conditions, and mechanisms to belong to multiple categories.
Creates junction tables for many-to-many category relationships.

Example use cases:
- Probiotics: "supplement" (primary) + "gut flora modulator" (functional)
- FMT: "procedure" (primary) + "gut flora modulator" (functional)
- Antacids + LES surgery: Both in "GERD treatment" (therapeutic)

Created: 2025-10-14
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MultiCategoryMigration:
    """
    Migrates database to support multi-category membership for entities.
    """

    def __init__(self, db_path: str):
        """
        Initialize migration.

        Args:
            db_path: Path to intervention_research.db
        """
        self.db_path = Path(db_path)

    def run(self, dry_run: bool = False):
        """
        Run migration.

        Args:
            dry_run: If True, only check what would be done
        """
        logger.info("=" * 60)
        logger.info("MULTI-CATEGORY SUPPORT MIGRATION")
        logger.info("=" * 60)

        if dry_run:
            logger.info("DRY RUN MODE - No changes will be made")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Step 1: Create junction tables
            logger.info("\n[1/5] Creating junction tables...")
            self._create_junction_tables(cursor, dry_run)

            # Step 2: Migrate existing single-category data
            logger.info("\n[2/5] Migrating existing single-category data...")
            stats = self._migrate_existing_data(cursor, dry_run)

            # Step 3: Create indexes
            logger.info("\n[3/5] Creating performance indexes...")
            self._create_indexes(cursor, dry_run)

            # Step 4: Create compatibility views
            logger.info("\n[4/5] Creating compatibility views...")
            self._create_compatibility_views(cursor, dry_run)

            # Step 5: Validate migration
            if not dry_run:
                logger.info("\n[5/5] Validating migration...")
                self._validate_migration(cursor)
            else:
                logger.info("\n[5/5] Validation skipped in dry-run mode")

            if not dry_run:
                conn.commit()
                logger.info("\n" + "=" * 60)
                logger.info("MIGRATION COMPLETE")
                logger.info("=" * 60)
                logger.info(f"Interventions migrated: {stats['interventions']}")
                logger.info(f"Conditions migrated: {stats['conditions']}")
                logger.info(f"Mechanisms ready: {stats['mechanisms']}")
            else:
                logger.info("\nDRY RUN COMPLETE - No changes made")

        except Exception as e:
            conn.rollback()
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            conn.close()

    def _create_junction_tables(self, cursor, dry_run: bool):
        """Create 3 junction tables for multi-category support."""

        tables = {
            'intervention_category_mapping': """
                CREATE TABLE IF NOT EXISTS intervention_category_mapping (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intervention_id INTEGER NOT NULL,

                    -- Category details
                    category_type TEXT NOT NULL CHECK(category_type IN ('primary', 'functional', 'therapeutic', 'experimental')),
                    category_name TEXT NOT NULL,

                    -- Metadata
                    confidence REAL DEFAULT 1.0 CHECK(confidence >= 0 AND confidence <= 1),
                    assigned_by TEXT DEFAULT 'system',  -- 'llm_extraction', 'semantic_grouping', 'phase_3d', 'manual'
                    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,

                    FOREIGN KEY (intervention_id) REFERENCES interventions(id) ON DELETE CASCADE,
                    UNIQUE(intervention_id, category_name)
                )
            """,
            'condition_category_mapping': """
                CREATE TABLE IF NOT EXISTS condition_category_mapping (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    -- Condition reference (no condition table, use name directly)
                    condition_name TEXT NOT NULL,

                    -- Category details
                    category_type TEXT NOT NULL CHECK(category_type IN ('primary', 'system', 'comorbidity', 'therapeutic')),
                    category_name TEXT NOT NULL,

                    -- Metadata
                    confidence REAL DEFAULT 1.0 CHECK(confidence >= 0 AND confidence <= 1),
                    assigned_by TEXT DEFAULT 'system',
                    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,

                    UNIQUE(condition_name, category_name)
                )
            """,
            'mechanism_category_mapping': """
                CREATE TABLE IF NOT EXISTS mechanism_category_mapping (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mechanism_cluster_id INTEGER NOT NULL,

                    -- Category details
                    category_type TEXT NOT NULL CHECK(category_type IN ('primary', 'pathway', 'target', 'functional')),
                    category_name TEXT NOT NULL,

                    -- Metadata
                    confidence REAL DEFAULT 1.0 CHECK(confidence >= 0 AND confidence <= 1),
                    assigned_by TEXT DEFAULT 'system',
                    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,

                    FOREIGN KEY (mechanism_cluster_id) REFERENCES mechanism_clusters(cluster_id) ON DELETE CASCADE,
                    UNIQUE(mechanism_cluster_id, category_name)
                )
            """
        }

        for table_name, create_sql in tables.items():
            if not dry_run:
                cursor.execute(create_sql)
            logger.info(f"  Created table: {table_name}")

    def _migrate_existing_data(self, cursor, dry_run: bool) -> dict:
        """
        Migrate existing single-category data to junction tables.

        Returns:
            Dict with migration statistics
        """
        stats = {
            'interventions': 0,
            'conditions': 0,
            'mechanisms': 0
        }

        # Check if interventions table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='interventions'
        """)

        if not cursor.fetchone():
            logger.info("  interventions table not found (empty database)")
            return stats

        # Migrate intervention categories
        cursor.execute("""
            SELECT id, intervention_category
            FROM interventions
            WHERE intervention_category IS NOT NULL
            AND intervention_category != ''
        """)
        interventions = cursor.fetchall()

        logger.info(f"  Found {len(interventions)} interventions with categories")

        if not dry_run:
            for row in interventions:
                intervention_id = row['id']
                category = row['intervention_category']

                # Insert as PRIMARY category
                cursor.execute("""
                    INSERT OR IGNORE INTO intervention_category_mapping
                    (intervention_id, category_type, category_name, assigned_by)
                    VALUES (?, 'primary', ?, 'migration')
                """, (intervention_id, category))

            stats['interventions'] = len(interventions)

        # Migrate condition categories
        cursor.execute("""
            SELECT DISTINCT health_condition, condition_category
            FROM interventions
            WHERE condition_category IS NOT NULL
            AND condition_category != ''
        """)
        conditions = cursor.fetchall()

        logger.info(f"  Found {len(conditions)} unique conditions with categories")

        if not dry_run:
            for row in conditions:
                condition_name = row['health_condition']
                category = row['condition_category']

                # Insert as PRIMARY category
                cursor.execute("""
                    INSERT OR IGNORE INTO condition_category_mapping
                    (condition_name, category_type, category_name, assigned_by)
                    VALUES (?, 'primary', ?, 'migration')
                """, (condition_name, category))

            stats['conditions'] = len(conditions)

        # Check mechanism_clusters table existence
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='mechanism_clusters'
        """)

        if cursor.fetchone():
            cursor.execute("SELECT COUNT(*) as cnt FROM mechanism_clusters")
            mechanism_count = cursor.fetchone()['cnt']
            stats['mechanisms'] = mechanism_count
            logger.info(f"  Found {mechanism_count} mechanism clusters (no migration needed yet)")
        else:
            logger.info("  mechanism_clusters table not found (Phase 3.6 not run yet)")

        return stats

    def _create_indexes(self, cursor, dry_run: bool):
        """Create indexes for performance."""

        indexes = [
            # Intervention category mapping indexes
            "CREATE INDEX IF NOT EXISTS idx_icm_intervention ON intervention_category_mapping(intervention_id)",
            "CREATE INDEX IF NOT EXISTS idx_icm_category_name ON intervention_category_mapping(category_name)",
            "CREATE INDEX IF NOT EXISTS idx_icm_category_type ON intervention_category_mapping(category_type)",
            "CREATE INDEX IF NOT EXISTS idx_icm_type_name ON intervention_category_mapping(category_type, category_name)",

            # Condition category mapping indexes
            "CREATE INDEX IF NOT EXISTS idx_ccm_condition ON condition_category_mapping(condition_name)",
            "CREATE INDEX IF NOT EXISTS idx_ccm_category_name ON condition_category_mapping(category_name)",
            "CREATE INDEX IF NOT EXISTS idx_ccm_category_type ON condition_category_mapping(category_type)",
            "CREATE INDEX IF NOT EXISTS idx_ccm_type_name ON condition_category_mapping(category_type, category_name)",

            # Mechanism category mapping indexes
            "CREATE INDEX IF NOT EXISTS idx_mcm_cluster ON mechanism_category_mapping(mechanism_cluster_id)",
            "CREATE INDEX IF NOT EXISTS idx_mcm_category_name ON mechanism_category_mapping(category_name)",
            "CREATE INDEX IF NOT EXISTS idx_mcm_category_type ON mechanism_category_mapping(category_type)",
        ]

        for idx_sql in indexes:
            if not dry_run:
                cursor.execute(idx_sql)
            idx_name = idx_sql.split('idx_')[1].split(' ')[0] if 'idx_' in idx_sql else 'unknown'
            logger.info(f"  Created index: idx_{idx_name}")

    def _create_compatibility_views(self, cursor, dry_run: bool):
        """Create views for backward compatibility."""

        views = {
            'interventions_with_primary_category': """
                CREATE VIEW IF NOT EXISTS interventions_with_primary_category AS
                SELECT
                    i.*,
                    (SELECT category_name
                     FROM intervention_category_mapping
                     WHERE intervention_id = i.id
                     AND category_type = 'primary'
                     LIMIT 1) AS primary_category
                FROM interventions i
            """,
            'conditions_with_primary_category': """
                CREATE VIEW IF NOT EXISTS conditions_with_primary_category AS
                SELECT
                    DISTINCT health_condition,
                    (SELECT category_name
                     FROM condition_category_mapping
                     WHERE condition_name = interventions.health_condition
                     AND category_type = 'primary'
                     LIMIT 1) AS primary_category
                FROM interventions
            """
        }

        for view_name, view_sql in views.items():
            if not dry_run:
                cursor.execute(view_sql)
            logger.info(f"  Created view: {view_name}")

    def _validate_migration(self, cursor):
        """Validate migration integrity."""

        # Check junction tables exist
        required_tables = [
            'intervention_category_mapping',
            'condition_category_mapping',
            'mechanism_category_mapping'
        ]

        for table in required_tables:
            cursor.execute(f"""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='{table}'
            """)
            if not cursor.fetchone():
                raise ValueError(f"Migration validation failed: {table} not created")

        # Check data integrity
        cursor.execute("""
            SELECT COUNT(*) as cnt
            FROM intervention_category_mapping
            WHERE category_type = 'primary'
        """)
        primary_count = cursor.fetchone()['cnt']

        logger.info(f"  Validation passed")
        logger.info(f"  Primary intervention categories: {primary_count}")


def main():
    """Run migration standalone."""
    import argparse

    parser = argparse.ArgumentParser(description='Add multi-category support to database')
    parser.add_argument('--db-path', default='back_end/data/intervention_research.db',
                       help='Path to database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode (no changes)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    migration = MultiCategoryMigration(args.db_path)
    migration.run(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
