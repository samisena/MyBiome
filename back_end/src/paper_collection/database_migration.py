#!/usr/bin/env python3
"""
Database Migration Manager for MyBiome Research Platform

Handles database schema migrations to add new tables for data mining results
while preserving existing data and maintaining backward compatibility.

Features:
- Safe incremental migrations
- Rollback capability
- Data preservation
- Version tracking
- Validation and testing

Usage:
    python database_migration.py --migrate
    python database_migration.py --rollback
    python database_migration.py --status
    python database_migration.py --validate
"""

import sys
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.data.config import config, setup_logging
    from src.paper_collection.database_manager import database_manager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logger = setup_logging(__name__, 'database_migration.log')


class DatabaseMigration:
    """Manages database schema migrations."""

    def __init__(self):
        self.db_path = config.db_path
        self.migration_table = 'schema_migrations'
        self.current_version = self._get_current_version()
        self.target_version = 2  # Version 2 includes data mining tables

    def _get_current_version(self) -> int:
        """Get current database schema version."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Check if migration table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name=?
                """, (self.migration_table,))

                if not cursor.fetchone():
                    # No migration table means version 1 (original schema)
                    return 1

                # Get latest version
                cursor.execute(f"""
                    SELECT version FROM {self.migration_table}
                    ORDER BY version DESC LIMIT 1
                """)

                result = cursor.fetchone()
                return result[0] if result else 1

        except Exception as e:
            logger.error(f"Error getting current version: {e}")
            return 1

    def _create_migration_table(self):
        """Create migration tracking table."""
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.migration_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER UNIQUE NOT NULL,
                    migration_name TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rollback_sql TEXT,
                    notes TEXT
                )
            """)
            conn.commit()

    def _record_migration(self, version: int, name: str, rollback_sql: str = None, notes: str = None):
        """Record a successful migration."""
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {self.migration_table}
                (version, migration_name, rollback_sql, notes)
                VALUES (?, ?, ?, ?)
            """, (version, name, rollback_sql, notes))
            conn.commit()

    def _backup_database(self) -> Path:
        """Create a backup of the database before migration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.parent / f"{self.db_path.stem}_backup_{timestamp}.db"

        logger.info(f"Creating database backup: {backup_path}")

        # Use SQLite backup API for consistent backup
        with sqlite3.connect(str(self.db_path)) as source:
            with sqlite3.connect(str(backup_path)) as backup:
                source.backup(backup)

        logger.info(f"Database backup created: {backup_path}")
        return backup_path

    def _validate_existing_schema(self) -> bool:
        """Validate that existing schema is as expected."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Check for required tables
                required_tables = ['papers', 'interventions']
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name IN ('papers', 'interventions')
                """)

                existing_tables = [row[0] for row in cursor.fetchall()]
                missing_tables = set(required_tables) - set(existing_tables)

                if missing_tables:
                    logger.error(f"Missing required tables: {missing_tables}")
                    return False

                # Validate papers table structure
                cursor.execute("PRAGMA table_info(papers)")
                papers_columns = {row[1] for row in cursor.fetchall()}
                required_papers_columns = {'pmid', 'title', 'abstract'}

                if not required_papers_columns.issubset(papers_columns):
                    missing_cols = required_papers_columns - papers_columns
                    logger.error(f"Papers table missing columns: {missing_cols}")
                    return False

                # Validate interventions table structure
                cursor.execute("PRAGMA table_info(interventions)")
                interventions_columns = {row[1] for row in cursor.fetchall()}
                required_interventions_columns = {'intervention_name', 'health_condition', 'paper_id'}

                if not required_interventions_columns.issubset(interventions_columns):
                    missing_cols = required_interventions_columns - interventions_columns
                    logger.error(f"Interventions table missing columns: {missing_cols}")
                    return False

                logger.info("Existing schema validation passed")
                return True

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def _apply_migration_v2(self):
        """Apply migration to version 2 (add data mining tables)."""
        logger.info("Applying migration to version 2: Adding data mining tables")

        # Read the schema SQL file
        schema_file = Path(__file__).parent / "enhanced_database_schema.sql"

        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        # Split into individual statements
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]

        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Apply each statement
            for i, statement in enumerate(statements):
                try:
                    # Skip comments and empty statements
                    if statement.startswith('--') or not statement:
                        continue

                    logger.debug(f"Executing statement {i+1}: {statement[:100]}...")
                    cursor.execute(statement)

                except sqlite3.Error as e:
                    # Log but continue for statements that might already exist
                    if "already exists" in str(e).lower():
                        logger.debug(f"Statement {i+1} skipped (already exists): {e}")
                    else:
                        logger.error(f"Error in statement {i+1}: {e}")
                        logger.error(f"Statement: {statement}")
                        raise

            conn.commit()
            logger.info("Migration v2 applied successfully")

    def _generate_rollback_v2(self) -> str:
        """Generate rollback SQL for version 2."""
        # List of tables added in version 2
        v2_tables = [
            'knowledge_graph_nodes',
            'knowledge_graph_edges',
            'bayesian_scores',
            'treatment_recommendations',
            'research_gaps',
            'innovation_tracking',
            'biological_patterns',
            'condition_similarities',
            'intervention_combinations',
            'failed_interventions',
            'data_mining_sessions'
        ]

        # List of views added in version 2
        v2_views = [
            'intervention_insights',
            'research_opportunities'
        ]

        # List of triggers added in version 2
        v2_triggers = [
            'update_kg_nodes_timestamp',
            'update_innovation_timestamp'
        ]

        rollback_statements = []

        # Drop views first (they may depend on tables)
        for view in v2_views:
            rollback_statements.append(f"DROP VIEW IF EXISTS {view}")

        # Drop triggers
        for trigger in v2_triggers:
            rollback_statements.append(f"DROP TRIGGER IF EXISTS {trigger}")

        # Drop tables
        for table in v2_tables:
            rollback_statements.append(f"DROP TABLE IF EXISTS {table}")

        return "; ".join(rollback_statements)

    def migrate(self) -> bool:
        """Perform database migration."""
        try:
            logger.info(f"Starting migration from version {self.current_version} to {self.target_version}")

            if self.current_version >= self.target_version:
                logger.info("Database is already at target version or newer")
                return True

            # Create migration tracking table
            self._create_migration_table()

            # Validate existing schema
            if not self._validate_existing_schema():
                logger.error("Schema validation failed, aborting migration")
                return False

            # Create backup
            backup_path = self._backup_database()

            try:
                if self.current_version == 1 and self.target_version >= 2:
                    self._apply_migration_v2()
                    rollback_sql = self._generate_rollback_v2()
                    self._record_migration(
                        version=2,
                        name="Add data mining tables",
                        rollback_sql=rollback_sql,
                        notes=f"Backup created at: {backup_path}"
                    )

                # Update current version
                self.current_version = self._get_current_version()
                logger.info(f"Migration completed successfully. Current version: {self.current_version}")
                return True

            except Exception as e:
                logger.error(f"Migration failed: {e}")
                logger.error(f"Database backup available at: {backup_path}")
                raise

        except Exception as e:
            logger.error(f"Migration error: {e}")
            logger.error(traceback.format_exc())
            return False

    def rollback(self, target_version: int = None) -> bool:
        """Rollback to a previous version."""
        try:
            if target_version is None:
                target_version = self.current_version - 1

            if target_version >= self.current_version:
                logger.error("Cannot rollback to same or newer version")
                return False

            logger.info(f"Rolling back from version {self.current_version} to {target_version}")

            # Create backup before rollback
            backup_path = self._backup_database()

            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get migrations to rollback (in reverse order)
                cursor.execute(f"""
                    SELECT version, migration_name, rollback_sql
                    FROM {self.migration_table}
                    WHERE version > ?
                    ORDER BY version DESC
                """, (target_version,))

                migrations_to_rollback = cursor.fetchall()

                for version, name, rollback_sql in migrations_to_rollback:
                    if rollback_sql:
                        logger.info(f"Rolling back migration {version}: {name}")

                        # Execute rollback statements
                        statements = [stmt.strip() for stmt in rollback_sql.split(';') if stmt.strip()]
                        for statement in statements:
                            cursor.execute(statement)

                        # Remove migration record
                        cursor.execute(f"DELETE FROM {self.migration_table} WHERE version = ?", (version,))
                    else:
                        logger.warning(f"No rollback SQL for migration {version}: {name}")

                conn.commit()

            # Update current version
            self.current_version = self._get_current_version()
            logger.info(f"Rollback completed. Current version: {self.current_version}")
            return True

        except Exception as e:
            logger.error(f"Rollback error: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get migration status."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get basic info
                status = {
                    'database_path': str(self.db_path),
                    'current_version': self.current_version,
                    'target_version': self.target_version,
                    'migration_needed': self.current_version < self.target_version
                }

                # Get migration history if table exists
                try:
                    cursor.execute(f"SELECT * FROM {self.migration_table} ORDER BY version")
                    migrations = cursor.fetchall()
                    status['migration_history'] = [
                        {
                            'version': m[1],
                            'name': m[2],
                            'applied_at': m[3],
                            'notes': m[5]
                        }
                        for m in migrations
                    ]
                except sqlite3.OperationalError:
                    status['migration_history'] = []

                # Get table counts
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                status['table_count'] = len(tables)
                status['tables'] = tables

                # Check for data mining tables
                data_mining_tables = [
                    'knowledge_graph_nodes', 'knowledge_graph_edges', 'bayesian_scores',
                    'treatment_recommendations', 'research_gaps', 'innovation_tracking',
                    'biological_patterns', 'condition_similarities', 'intervention_combinations',
                    'failed_interventions', 'data_mining_sessions'
                ]

                existing_dm_tables = [t for t in data_mining_tables if t in tables]
                status['data_mining_tables_present'] = len(existing_dm_tables)
                status['data_mining_tables_total'] = len(data_mining_tables)
                status['data_mining_ready'] = len(existing_dm_tables) == len(data_mining_tables)

                return status

        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'error': str(e)}

    def validate_migration(self) -> Dict[str, Any]:
        """Validate that migration was successful."""
        validation_results = {
            'success': True,
            'errors': [],
            'warnings': [],
            'table_checks': {},
            'data_integrity': {}
        }

        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Check that all expected tables exist
                expected_tables = [
                    # Original tables
                    'papers', 'interventions', 'intervention_categories',
                    # Data mining tables
                    'knowledge_graph_nodes', 'knowledge_graph_edges', 'bayesian_scores',
                    'treatment_recommendations', 'research_gaps', 'innovation_tracking',
                    'biological_patterns', 'condition_similarities', 'intervention_combinations',
                    'failed_interventions', 'data_mining_sessions'
                ]

                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}

                for table in expected_tables:
                    if table in existing_tables:
                        validation_results['table_checks'][table] = 'present'
                    else:
                        validation_results['table_checks'][table] = 'missing'
                        validation_results['errors'].append(f"Table {table} is missing")
                        validation_results['success'] = False

                # Check foreign key constraints
                cursor.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                if fk_violations:
                    validation_results['errors'].append(f"Foreign key violations: {fk_violations}")
                    validation_results['success'] = False

                # Check data integrity for original tables
                cursor.execute("SELECT COUNT(*) FROM papers")
                papers_count = cursor.fetchone()[0]
                validation_results['data_integrity']['papers_count'] = papers_count

                cursor.execute("SELECT COUNT(*) FROM interventions")
                interventions_count = cursor.fetchone()[0]
                validation_results['data_integrity']['interventions_count'] = interventions_count

                # Verify indexes exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                indexes = {row[0] for row in cursor.fetchall()}

                expected_indexes = [
                    'idx_kg_nodes_type', 'idx_bayesian_intervention', 'idx_recommendations_condition'
                ]

                missing_indexes = []
                for idx in expected_indexes:
                    if idx not in indexes:
                        missing_indexes.append(idx)

                if missing_indexes:
                    validation_results['warnings'].append(f"Missing indexes: {missing_indexes}")

                logger.info(f"Validation completed. Success: {validation_results['success']}")

        except Exception as e:
            validation_results['success'] = False
            validation_results['errors'].append(f"Validation error: {e}")
            logger.error(f"Validation error: {e}")

        return validation_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Database Migration Manager")
    parser.add_argument('--migrate', action='store_true',
                       help='Apply pending migrations')
    parser.add_argument('--rollback', type=int, metavar='VERSION',
                       help='Rollback to specified version')
    parser.add_argument('--status', action='store_true',
                       help='Show migration status')
    parser.add_argument('--validate', action='store_true',
                       help='Validate database after migration')
    parser.add_argument('--force', action='store_true',
                       help='Force migration even with warnings')

    args = parser.parse_args()

    migration = DatabaseMigration()

    try:
        if args.migrate:
            print("Starting database migration...")
            if migration.migrate():
                print("✅ Migration completed successfully")

                # Auto-validate after migration
                print("Validating migration...")
                validation = migration.validate_migration()
                if validation['success']:
                    print("✅ Validation passed")
                else:
                    print("❌ Validation failed:")
                    for error in validation['errors']:
                        print(f"  - {error}")
                    return 1
            else:
                print("❌ Migration failed")
                return 1

        elif args.rollback is not None:
            print(f"Rolling back to version {args.rollback}...")
            if migration.rollback(args.rollback):
                print("✅ Rollback completed successfully")
            else:
                print("❌ Rollback failed")
                return 1

        elif args.validate:
            print("Validating database...")
            validation = migration.validate_migration()

            print(f"Validation result: {'✅ PASSED' if validation['success'] else '❌ FAILED'}")

            if validation['errors']:
                print("Errors:")
                for error in validation['errors']:
                    print(f"  - {error}")

            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")

            print(f"Tables: {len([t for t in validation['table_checks'].values() if t == 'present'])}/{len(validation['table_checks'])} present")

            return 0 if validation['success'] else 1

        else:  # Default to status
            print("Database Migration Status")
            print("=" * 40)

            status = migration.get_status()

            if 'error' in status:
                print(f"❌ Error: {status['error']}")
                return 1

            print(f"Database: {status['database_path']}")
            print(f"Current version: {status['current_version']}")
            print(f"Target version: {status['target_version']}")
            print(f"Migration needed: {'Yes' if status['migration_needed'] else 'No'}")
            print(f"Tables: {status['table_count']}")
            print(f"Data mining ready: {'Yes' if status['data_mining_ready'] else 'No'}")
            print(f"Data mining tables: {status['data_mining_tables_present']}/{status['data_mining_tables_total']}")

            if status['migration_history']:
                print("\nMigration History:")
                for migration in status['migration_history']:
                    print(f"  v{migration['version']}: {migration['name']} ({migration['applied_at']})")

        return 0

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Migration error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)