"""
Database manager with thread-safe connection handling.

This module now uses the DAO (Data Access Object) pattern for better separation of concerns.
All database operations are delegated to specialized DAO classes.

IMPORTANT: This class maintains backward compatibility while internally using DAOs.
"""

from contextlib import contextmanager
from typing import List, Dict, Optional, Any
from pathlib import Path

from back_end.src.data.config import config, setup_logging
from .dao import PapersDAO, InterventionsDAO, AnalyticsDAO, SchemaDAO

logger = setup_logging(__name__, 'database.log')

# Optional import for entity normalization - graceful fallback if not available
try:
    from ..phase_2_llm_processing.batch_entity_processor import BatchEntityProcessor as EntityNormalizer
    NORMALIZATION_AVAILABLE = True
except ImportError:
    NORMALIZATION_AVAILABLE = False
    logger.warning("Entity normalization not available - install required dependencies")


class DatabaseManager:
    """
    Database manager with thread-safe connection handling using DAO pattern.

    This class delegates all operations to specialized DAOs:
    - PapersDAO: Paper CRUD operations
    - InterventionsDAO: Intervention CRUD operations
    - AnalyticsDAO: Statistics and analytics queries
    - SchemaDAO: Table creation and migrations
    """

    def __init__(self, db_config=None, enable_normalization: bool = False):
        self.db_config = db_config or type('DatabaseConfig', (), {
            'name': config.db_name,
            'path': config.db_path
        })()

        # Ensure db_path is a Path object
        self.db_path = Path(self.db_config.path) if not isinstance(self.db_config.path, Path) else self.db_config.path

        # Set up normalization capability
        self.enable_normalization = enable_normalization and NORMALIZATION_AVAILABLE
        if enable_normalization and not NORMALIZATION_AVAILABLE:
            logger.warning("Normalization requested but not available - running without normalization")

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize DAOs
        self.schema = SchemaDAO(self.db_path)
        self.papers = PapersDAO(self.db_path)
        self.interventions = InterventionsDAO(self.db_path)
        self.analytics = AnalyticsDAO(self.db_path)

        # Create tables and run migrations
        self.schema.create_all_tables()
        self.schema.migrate_llm_processed_flag()
        self.schema.migrate_study_confidence()
        self.schema.add_optional_intervention_columns()
        self.schema.add_semantic_scholar_columns()

        # Set up intervention categories
        self.interventions.setup_intervention_categories()

        logger.info(f"Thread-safe database manager initialized at {self.db_path} (normalization: {self.enable_normalization})")

    @contextmanager
    def get_connection(self):
        """
        Get a thread-safe database connection (delegates to SchemaDAO).

        This creates a fresh connection for each context, ensuring thread safety.
        """
        with self.schema.get_connection() as conn:
            yield conn

    # ================================================================
    # SCHEMA & MIGRATION OPERATIONS (delegate to SchemaDAO)
    # ================================================================

    def create_tables(self):
        """Create all necessary database tables."""
        return self.schema.create_all_tables()

    def migrate_to_llm_processed_flag(self):
        """Add llm_processed flag for Phase 2 optimization."""
        return self.schema.migrate_llm_processed_flag()

    def migrate_to_study_confidence(self):
        """Add study_confidence column."""
        return self.schema.migrate_study_confidence()

    def check_data_mining_tables_exist(self) -> bool:
        """Check if data mining tables exist."""
        return self.schema.check_data_mining_tables_exist()

    def initialize_data_mining_schema(self) -> bool:
        """Initialize data mining schema if not already present."""
        if self.check_data_mining_tables_exist():
            logger.info("Data mining tables already exist")
            return True

        try:
            self.schema.create_data_mining_tables()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize data mining schema: {e}")
            return False

    def get_data_mining_connection(self):
        """Get a connection specifically for data mining operations."""
        return self.get_connection()

    # ================================================================
    # PAPER OPERATIONS (delegate to PapersDAO)
    # ================================================================

    def insert_paper(self, paper: Dict) -> bool:
        """Insert a paper with validation."""
        return self.papers.insert_paper(paper)

    def insert_papers_batch(self, papers: List[Dict]) -> tuple[int, int]:
        """Insert multiple papers efficiently using executemany()."""
        return self.papers.insert_papers_batch(papers)

    def get_paper_by_pmid(self, pmid: str) -> Optional[Dict]:
        """Get a paper by PMID."""
        return self.papers.get_paper_by_pmid(pmid)

    def get_all_papers(self, limit: Optional[int] = None,
                      processing_status: Optional[str] = None) -> List[Dict]:
        """Get all papers with optional filtering."""
        return self.papers.get_all_papers(limit, processing_status)

    def get_papers_for_processing(self, extraction_model: str,
                                  limit: Optional[int] = None) -> List[Dict]:
        """Get papers that need LLM processing."""
        return self.papers.get_papers_for_processing(extraction_model, limit)

    def mark_paper_llm_processed(self, pmid: str) -> bool:
        """Mark a paper as LLM processed."""
        return self.papers.mark_paper_llm_processed(pmid)

    def update_paper_processing_status(self, pmid: str, status: str) -> bool:
        """Update paper processing status."""
        return self.papers.update_paper_processing_status(pmid, status)

    def get_papers_by_condition(self, condition: str, limit: Optional[int] = None) -> List[Dict]:
        """Get papers related to a specific health condition."""
        return self.papers.get_papers_by_condition(condition, limit)

    # ================================================================
    # INTERVENTION OPERATIONS (delegate to InterventionsDAO)
    # ================================================================

    def insert_intervention(self, intervention: Dict) -> bool:
        """Insert an intervention with validation."""
        return self.interventions.insert_intervention(intervention)

    def setup_intervention_categories(self):
        """Set up the intervention categories table with taxonomy data."""
        return self.interventions.setup_intervention_categories()

    def clean_placeholder_interventions(self) -> Dict[str, int]:
        """Remove interventions with placeholder names from the database."""
        return self.interventions.clean_placeholder_interventions()

    def assign_category(
        self,
        entity_type: str,
        entity_id: Any,
        category_name: str,
        category_type: str = 'primary',
        confidence: float = 1.0,
        assigned_by: str = 'system',
        notes: Optional[str] = None
    ) -> bool:
        """Assign a category to an entity (supports multi-category)."""
        return self.interventions.assign_category(
            entity_type, entity_id, category_name, category_type, confidence, assigned_by, notes
        )

    def get_entity_categories(
        self,
        entity_type: str,
        entity_id: Any,
        category_type_filter: Optional[str] = None
    ) -> List[Dict]:
        """Get all categories for an entity."""
        return self.interventions.get_entity_categories(entity_type, entity_id, category_type_filter)

    def get_entities_by_category(
        self,
        category_name: str,
        entity_type: str = 'intervention',
        category_type_filter: Optional[str] = None
    ) -> List[Dict]:
        """Get all entities in a category (supports multi-category)."""
        return self.interventions.get_entities_by_category(category_name, entity_type, category_type_filter)

    def get_primary_category(
        self,
        entity_type: str,
        entity_id: Any
    ) -> Optional[str]:
        """Get the primary category for an entity (backward compatibility)."""
        return self.interventions.get_primary_category(entity_type, entity_id)

    # ================================================================
    # ANALYTICS OPERATIONS (delegate to AnalyticsDAO)
    # ================================================================

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        return self.analytics.get_database_stats()

    # ================================================================
    # NORMALIZATION OPERATIONS (kept in DatabaseManager for now)
    # These could be moved to a NormalizationDAO in the future
    # ================================================================

    def insert_intervention_normalized(self, intervention: dict) -> bool:
        """Insert intervention with automatic entity normalization."""
        if not self.enable_normalization:
            return self.insert_intervention(intervention)

        try:
            normalized_intervention = intervention.copy()

            # Normalize intervention_name
            intervention_name = intervention.get('intervention_name', '').strip()
            if intervention_name:
                with self.get_connection() as conn:
                    normalizer = EntityNormalizer(conn)
                    intervention_mapping = normalizer.find_or_create_mapping(
                        intervention_name, 'intervention', confidence_threshold=0.7
                    )
                    normalized_intervention['intervention_canonical_id'] = intervention_mapping['canonical_id']

                    if intervention_mapping['is_new']:
                        logger.info(f"Created new intervention canonical: {intervention_mapping['canonical_name']}")
                    elif intervention_mapping['method'] != 'exact_canonical':
                        logger.info(f"Normalized '{intervention_name}' -> '{intervention_mapping['canonical_name']}' "
                                  f"(method: {intervention_mapping['method']}, confidence: {intervention_mapping['confidence']:.2f})")

            # Normalize health_condition
            health_condition = intervention.get('health_condition', '').strip()
            if health_condition:
                with self.get_connection() as conn:
                    normalizer = EntityNormalizer(conn)
                    condition_mapping = normalizer.find_or_create_mapping(
                        health_condition, 'condition', confidence_threshold=0.7
                    )
                    normalized_intervention['condition_canonical_id'] = condition_mapping['canonical_id']

                    if condition_mapping['is_new']:
                        logger.info(f"Created new condition canonical: {condition_mapping['canonical_name']}")
                    elif condition_mapping['method'] != 'exact_canonical':
                        logger.info(f"Normalized '{health_condition}' -> '{condition_mapping['canonical_name']}' "
                                  f"(method: {condition_mapping['method']}, confidence: {condition_mapping['confidence']:.2f})")

            normalized_intervention['normalized'] = True
            return self._insert_intervention_with_normalization(normalized_intervention)

        except Exception as e:
            logger.error(f"Error in normalized insertion: {e}")
            return self.insert_intervention(intervention)

    def _insert_intervention_with_normalization(self, intervention: dict) -> bool:
        """Insert intervention including normalization fields."""
        # This is complex enough to keep inline for now
        # Could be moved to InterventionsDAO in the future
        try:
            from back_end.src.interventions.category_validators import category_validator
            import json

            validated_intervention = category_validator.validate_intervention(intervention)

            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO interventions
                    (paper_id, intervention_category, intervention_name, intervention_details,
                     health_condition, mechanism, condition_category, outcome_type,
                     study_confidence,
                     sample_size, study_duration, study_type, population_details,
                     supporting_quote, delivery_method, severity, adverse_effects, cost_category,
                     extraction_model, consensus_confidence, model_agreement,
                     models_used, raw_extraction_count, models_contributing,
                     intervention_canonical_id, condition_canonical_id, normalized)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validated_intervention['paper_id'] if 'paper_id' in validated_intervention else validated_intervention.get('pmid'),
                    validated_intervention['intervention_category'],
                    validated_intervention['intervention_name'],
                    json.dumps(validated_intervention.get('intervention_details', {})),
                    validated_intervention['health_condition'],
                    validated_intervention.get('mechanism'),
                    validated_intervention.get('condition_category'),
                    validated_intervention['outcome_type'],
                    validated_intervention.get('study_confidence'),
                    validated_intervention.get('sample_size'),
                    validated_intervention.get('study_duration'),
                    validated_intervention.get('study_type'),
                    validated_intervention.get('population_details'),
                    validated_intervention.get('supporting_quote'),
                    validated_intervention.get('delivery_method'),
                    validated_intervention.get('severity'),
                    validated_intervention.get('adverse_effects'),
                    validated_intervention.get('cost_category'),
                    validated_intervention.get('extraction_model', 'qwen3:14b'),
                    validated_intervention.get('consensus_confidence'),
                    validated_intervention.get('model_agreement', 'single'),
                    validated_intervention.get('models_used', 'qwen3:14b'),
                    validated_intervention.get('raw_extraction_count', 1),
                    validated_intervention.get('models_contributing'),
                    validated_intervention.get('intervention_canonical_id'),
                    validated_intervention.get('condition_canonical_id'),
                    validated_intervention.get('normalized', False)
                ))

                return True

        except Exception as e:
            logger.error(f"Error in normalized database insertion: {e}")
            return False

    def batch_normalize_existing_interventions(self, limit: int = 100) -> dict:
        """Normalize existing interventions that haven't been normalized yet."""
        if not self.enable_normalization:
            return {'error': 'Normalization not enabled'}

        results = {
            'processed': 0,
            'normalized_interventions': 0,
            'normalized_conditions': 0,
            'new_canonicals_created': 0,
            'errors': []
        }

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, intervention_name, health_condition
                    FROM interventions
                    WHERE (normalized IS NULL OR normalized = FALSE)
                    AND intervention_name IS NOT NULL
                    AND health_condition IS NOT NULL
                    LIMIT ?
                """, (limit,))

                interventions = cursor.fetchall()

                for row in interventions:
                    intervention_id = row[0]
                    intervention_name = row[1]
                    health_condition = row[2]

                    try:
                        with self.get_connection() as norm_conn:
                            normalizer = EntityNormalizer(norm_conn)

                            intervention_mapping = normalizer.find_or_create_mapping(
                                intervention_name, 'intervention'
                            )

                            condition_mapping = normalizer.find_or_create_mapping(
                                health_condition, 'condition'
                            )

                        cursor.execute("""
                            UPDATE interventions
                            SET intervention_canonical_id = ?,
                                condition_canonical_id = ?,
                                normalized = TRUE
                            WHERE id = ?
                        """, (
                            intervention_mapping['canonical_id'],
                            condition_mapping['canonical_id'],
                            intervention_id
                        ))

                        results['processed'] += 1

                        if intervention_mapping['is_new']:
                            results['new_canonicals_created'] += 1

                        if intervention_mapping['method'] != 'exact_canonical':
                            results['normalized_interventions'] += 1

                        if condition_mapping['method'] != 'exact_canonical':
                            results['normalized_conditions'] += 1

                    except Exception as e:
                        results['errors'].append(f"Error processing intervention {intervention_id}: {e}")

                conn.commit()

        except Exception as e:
            results['errors'].append(f"Batch normalization error: {e}")

        return results

    def close(self):
        """
        Close method for API compatibility.

        Note: With thread-local connections, there's no persistent pool to close.
        This method is kept for backward compatibility.
        """
        logger.info("Database manager close() called (connections auto-close per context)")


# Global instance for dependency injection
database_manager = DatabaseManager()
