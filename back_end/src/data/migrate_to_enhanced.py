"""
Migration script to transition from the old architecture to the enhanced architecture.
This script helps migrate existing data and provides a transition path.
"""

import sys
from pathlib import Path
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .config import config, setup_logging
from .database_manager_enhanced import database_manager as enhanced_db
from .database_manager import DatabaseManager as OldDatabaseManager
from .utils import log_execution_time

logger = setup_logging(__name__, 'migration.log')


class DatabaseMigrator:
    """Handles migration from old to enhanced database structure."""
    
    def __init__(self):
        self.enhanced_db = enhanced_db
        
        # Try to connect to old database if it exists
        try:
            self.old_db = OldDatabaseManager()
            logger.info("Old database connection established")
        except Exception as e:
            logger.warning(f"Could not connect to old database: {e}")
            self.old_db = None
    
    @log_execution_time
    def migrate_data(self) -> bool:
        """
        Migrate data from old database structure to enhanced structure.
        
        Returns:
            True if migration was successful, False otherwise
        """
        if not self.old_db:
            logger.error("No old database available for migration")
            return False
        
        try:
            logger.info("Starting database migration...")
            
            # Step 1: Migrate papers
            papers_migrated = self._migrate_papers()
            logger.info(f"Migrated {papers_migrated} papers")
            
            # Step 2: Migrate correlations
            correlations_migrated = self._migrate_correlations()
            logger.info(f"Migrated {correlations_migrated} correlations")
            
            # Step 3: Update processing status based on existing correlations
            self._update_processing_status()
            
            logger.info("Database migration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def _migrate_papers(self) -> int:
        """Migrate papers from old to new database."""
        old_papers = self.old_db.get_all_papers()
        migrated_count = 0
        
        for paper in old_papers:
            try:
                # The enhanced database manager will handle validation
                if self.enhanced_db.insert_paper(paper):
                    migrated_count += 1
            except Exception as e:
                logger.warning(f"Failed to migrate paper {paper.get('pmid', 'unknown')}: {e}")
        
        return migrated_count
    
    def _migrate_correlations(self) -> int:
        """Migrate correlations from old to new database."""
        migrated_count = 0
        
        try:
            with self.old_db.get_connection() as old_conn:
                old_cursor = old_conn.cursor()
                old_cursor.execute('SELECT * FROM correlations')
                
                for row in old_cursor.fetchall():
                    correlation = dict(row)
                    
                    try:
                        # Add any missing fields with defaults
                        if 'validation_status' not in correlation:
                            correlation['validation_status'] = 'pending'
                        
                        if self.enhanced_db.insert_correlation(correlation):
                            migrated_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to migrate correlation {correlation.get('id', 'unknown')}: {e}")
            
        except Exception as e:
            logger.error(f"Error accessing old correlations: {e}")
        
        return migrated_count
    
    def _update_processing_status(self):
        """Update processing status based on existing correlations."""
        try:
            with self.enhanced_db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Mark papers as processed if they have correlations
                cursor.execute('''
                    UPDATE papers 
                    SET processing_status = 'processed'
                    WHERE pmid IN (
                        SELECT DISTINCT paper_id FROM correlations
                    )
                ''')
                
                processed_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Updated processing status for {processed_count} papers")
                
        except Exception as e:
            logger.error(f"Error updating processing status: {e}")
    
    def validate_migration(self) -> Dict:
        """Validate the migration by comparing record counts."""
        if not self.old_db:
            return {"error": "No old database available for validation"}
        
        try:
            # Get counts from old database
            old_stats = self.old_db.get_database_stats()
            
            # Get counts from enhanced database
            enhanced_stats = self.enhanced_db.get_database_stats()
            
            validation_report = {
                "old_papers": old_stats.get('total_papers', 0),
                "new_papers": enhanced_stats.get('total_papers', 0),
                "old_correlations": old_stats.get('total_correlations', 0),
                "new_correlations": enhanced_stats.get('total_correlations', 0),
                "papers_match": old_stats.get('total_papers', 0) == enhanced_stats.get('total_papers', 0),
                "correlations_match": old_stats.get('total_correlations', 0) == enhanced_stats.get('total_correlations', 0)
            }
            
            logger.info("Migration validation:")
            for key, value in validation_report.items():
                logger.info(f"  {key}: {value}")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {"error": str(e)}


class ConfigurationMigrator:
    """Handles migration of configuration and file structure."""
    
    def __init__(self):
        self.config = config
    
    @log_execution_time  
    def setup_enhanced_structure(self):
        """Set up the enhanced directory structure and configuration."""
        logger.info("Setting up enhanced directory structure...")
        
        # The config.__init__ already creates directories
        # Just validate they exist
        directories_created = []
        
        required_dirs = [
            self.config.paths.raw_data,
            self.config.paths.processed_data,
            self.config.paths.papers_dir,
            self.config.paths.metadata_dir,
            self.config.paths.fulltext_dir,
            self.config.paths.pmc_dir,
            self.config.paths.pdf_dir,
            self.config.paths.logs_dir
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                directories_created.append(str(directory))
            
        if directories_created:
            logger.info(f"Created directories: {directories_created}")
        else:
            logger.info("All required directories already exist")
        
        # Validate configuration
        validation = self.config.validate()
        
        if validation['valid']:
            logger.info("Configuration validation passed")
        else:
            logger.error("Configuration validation failed:")
            for issue in validation['issues']:
                logger.error(f"  - {issue}")
        
        if validation['warnings']:
            logger.warning("Configuration warnings:")
            for warning in validation['warnings']:
                logger.warning(f"  - {warning}")
        
        return validation


def main():
    """Main migration function."""
    logger.info("=== MyBiome Enhanced Architecture Migration ===")
    
    # Step 1: Set up enhanced structure
    config_migrator = ConfigurationMigrator()
    config_validation = config_migrator.setup_enhanced_structure()
    
    if not config_validation['valid']:
        logger.error("Configuration validation failed. Please fix issues before proceeding.")
        return False
    
    # Step 2: Migrate database
    db_migrator = DatabaseMigrator()
    migration_success = db_migrator.migrate_data()
    
    if not migration_success:
        logger.error("Database migration failed")
        return False
    
    # Step 3: Validate migration
    validation_report = db_migrator.validate_migration()
    
    if 'error' in validation_report:
        logger.error(f"Migration validation error: {validation_report['error']}")
        return False
    
    # Step 4: Final summary
    logger.info("=== Migration Summary ===")
    logger.info(f"Papers migrated: {validation_report['new_papers']}")
    logger.info(f"Correlations migrated: {validation_report['new_correlations']}")
    logger.info(f"Papers match: {validation_report['papers_match']}")
    logger.info(f"Correlations match: {validation_report['correlations_match']}")
    
    if validation_report['papers_match'] and validation_report['correlations_match']:
        logger.info("✅ Migration completed successfully!")
        logger.info("\nYou can now use the enhanced modules:")
        logger.info("  - config.py for centralized configuration")
        logger.info("  - database_manager_enhanced.py with connection pooling")
        logger.info("  - api_clients.py for centralized API management")
        logger.info("  - Enhanced collectors, parsers, and analyzers")
        return True
    else:
        logger.warning("⚠️ Migration completed but validation shows mismatches")
        logger.warning("Please review the data carefully before using enhanced modules")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)