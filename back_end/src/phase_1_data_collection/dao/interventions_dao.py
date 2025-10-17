"""
InterventionsDAO - Data access object for intervention operations.

Handles all CRUD operations for interventions and their categories.
"""

import json
from typing import List, Dict, Optional, Any
from back_end.src.data.config import setup_logging
from back_end.src.data.constants import PLACEHOLDER_PATTERNS
from back_end.src.interventions.category_validators import category_validator
from .base_dao import BaseDAO

logger = setup_logging(__name__, 'database.log')


class InterventionsDAO(BaseDAO):
    """Data Access Object for intervention operations."""

    def insert_intervention(self, intervention: Dict) -> bool:
        """
        Insert an intervention with validation.

        Args:
            intervention: Intervention dictionary with all fields

        Returns:
            True if intervention was inserted successfully
        """
        try:
            # Validate intervention data
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
                     study_focus, measured_metrics, findings, study_location, publisher,
                     extraction_model)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validated_intervention['paper_id'] if 'paper_id' in validated_intervention else validated_intervention.get('pmid'),
                    validated_intervention.get('intervention_category'),  # Changed: allow NULL
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
                    json.dumps(validated_intervention.get('study_focus', [])) if validated_intervention.get('study_focus') else None,
                    json.dumps(validated_intervention.get('measured_metrics', [])) if validated_intervention.get('measured_metrics') else None,
                    json.dumps(validated_intervention.get('findings', [])) if validated_intervention.get('findings') else None,
                    validated_intervention.get('study_location'),
                    validated_intervention.get('publisher'),
                    validated_intervention.get('extraction_model', 'qwen3:14b')
                ))

                was_new = cursor.rowcount > 0

                if was_new:
                    category = validated_intervention.get('intervention_category', 'uncategorized')
                    logger.info(f"✓ Inserted intervention: {category} - {validated_intervention['intervention_name']} for {validated_intervention['health_condition']}")
                else:
                    logger.debug(f"Intervention already exists (replaced): {validated_intervention['intervention_name']}")

                return was_new

        except Exception as e:
            # Enhanced error logging with details
            paper_id = intervention.get('paper_id') or intervention.get('pmid', 'unknown')
            intervention_name = intervention.get('intervention_name', 'unknown')
            logger.error(f"✗ Failed to insert intervention '{intervention_name}' for paper {paper_id}")
            logger.error(f"  Error details: {str(e)}")
            logger.error(f"  Intervention data keys: {list(intervention.keys())}")

            # Log which required fields are missing
            required = ['intervention_name', 'health_condition', 'mechanism', 'outcome_type', 'paper_id']
            missing = [f for f in required if f not in intervention and f.replace('paper_', '') not in intervention]
            if missing:
                logger.error(f"  Missing required fields: {missing}")

            return False

    def setup_intervention_categories(self):
        """Set up the intervention categories table with taxonomy data."""
        from back_end.src.interventions.taxonomy import intervention_taxonomy
        from back_end.src.interventions.search_terms import search_terms

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                for category_type, category_def in intervention_taxonomy.get_all_categories().items():
                    # Get search terms for this category
                    category_search_terms = search_terms.get_terms_for_category(category_type)

                    cursor.execute('''
                        INSERT OR REPLACE INTO intervention_categories
                        (category, display_name, description, search_terms)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        category_type.value,
                        category_def.display_name,
                        category_def.description,
                        json.dumps(category_search_terms)
                    ))

                logger.info(f"Set up {len(intervention_taxonomy.get_all_categories())} intervention categories")
                return True

        except Exception as e:
            logger.error(f"Error setting up intervention categories: {e}")
            return False

    def clean_placeholder_interventions(self) -> Dict[str, int]:
        """
        Remove interventions with placeholder names from the database.

        Returns:
            Dictionary with count of removed entries
        """
        placeholder_patterns = list(PLACEHOLDER_PATTERNS) + [
            'intervention', 'treatment', 'therapy'
        ]

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Count entries to be removed
                placeholders = "', '".join(placeholder_patterns)
                count_query = f"""
                    SELECT COUNT(*) FROM interventions
                    WHERE intervention_name IN ('{placeholders}')
                    OR LENGTH(TRIM(intervention_name)) < 3
                    OR LOWER(intervention_name) LIKE 'unknown%'
                    OR LOWER(intervention_name) LIKE 'placeholder%'
                    OR LOWER(intervention_name) LIKE 'various%'
                    OR LOWER(intervention_name) LIKE 'multiple%'
                    OR LOWER(health_condition) IN ('{placeholders.lower()}')
                    OR LENGTH(TRIM(health_condition)) < 3
                """

                cursor.execute(count_query)
                count_to_remove = cursor.fetchone()[0]

                if count_to_remove == 0:
                    logger.info("No placeholder interventions found")
                    return {'removed_count': 0}

                # Remove placeholder entries
                delete_query = f"""
                    DELETE FROM interventions
                    WHERE intervention_name IN ('{placeholders}')
                    OR LENGTH(TRIM(intervention_name)) < 3
                    OR LOWER(intervention_name) LIKE 'unknown%'
                    OR LOWER(intervention_name) LIKE 'placeholder%'
                    OR LOWER(intervention_name) LIKE 'various%'
                    OR LOWER(intervention_name) LIKE 'multiple%'
                    OR LOWER(health_condition) IN ('{placeholders.lower()}')
                    OR LENGTH(TRIM(health_condition)) < 3
                """

                cursor.execute(delete_query)
                removed_count = cursor.rowcount

                logger.info(f"Removed {removed_count} placeholder interventions from database")
                return {'removed_count': removed_count}

        except Exception as e:
            logger.error(f"Error cleaning placeholder interventions: {e}")
            return {'removed_count': 0, 'error': str(e)}

    # ================================================================
    # MULTI-CATEGORY SUPPORT API
    # ================================================================

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
        """
        Assign a category to an entity (supports multi-category).

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'
            entity_id: intervention.id, condition name, or mechanism_cluster_id
            category_name: Category name to assign
            category_type: 'primary', 'functional', 'therapeutic', etc.
            confidence: Confidence score (0-1)
            assigned_by: Assignment source
            notes: Optional notes

        Returns:
            True if successful
        """
        valid_entity_types = ['intervention', 'condition', 'mechanism']
        if entity_type not in valid_entity_types:
            raise ValueError(f"entity_type must be one of {valid_entity_types}")

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if entity_type == 'intervention':
                    cursor.execute("""
                        INSERT OR IGNORE INTO intervention_category_mapping
                        (intervention_id, category_type, category_name, confidence, assigned_by, notes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (entity_id, category_type, category_name, confidence, assigned_by, notes))

                elif entity_type == 'condition':
                    cursor.execute("""
                        INSERT OR IGNORE INTO condition_category_mapping
                        (condition_name, category_type, category_name, confidence, assigned_by, notes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (entity_id, category_type, category_name, confidence, assigned_by, notes))

                elif entity_type == 'mechanism':
                    cursor.execute("""
                        INSERT OR IGNORE INTO mechanism_category_mapping
                        (mechanism_cluster_id, category_type, category_name, confidence, assigned_by, notes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (entity_id, category_type, category_name, confidence, assigned_by, notes))

                was_new = cursor.rowcount > 0
                return was_new

        except Exception as e:
            logger.error(f"Error assigning category: {e}")
            return False

    def get_entity_categories(
        self,
        entity_type: str,
        entity_id: Any,
        category_type_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all categories for an entity.

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'
            entity_id: Entity identifier
            category_type_filter: Optional filter by category_type

        Returns:
            List of category dicts
        """
        try:
            query_map = {
                'intervention': """
                    SELECT category_name, category_type, confidence, assigned_by, notes, assigned_at
                    FROM intervention_category_mapping
                    WHERE intervention_id = ?
                """,
                'condition': """
                    SELECT category_name, category_type, confidence, assigned_by, notes, assigned_at
                    FROM condition_category_mapping
                    WHERE condition_name = ?
                """,
                'mechanism': """
                    SELECT category_name, category_type, confidence, assigned_by, notes, assigned_at
                    FROM mechanism_category_mapping
                    WHERE mechanism_cluster_id = ?
                """
            }

            if entity_type not in query_map:
                raise ValueError(f"Invalid entity_type: {entity_type}")

            query = query_map[entity_type]
            params = [entity_id]

            if category_type_filter:
                query += " AND category_type = ?"
                params.append(category_type_filter)

            query += " ORDER BY category_type, category_name"

            return self.execute_query(query, tuple(params))

        except Exception as e:
            logger.error(f"Error getting entity categories: {e}")
            return []

    def get_entities_by_category(
        self,
        category_name: str,
        entity_type: str = 'intervention',
        category_type_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all entities in a category (supports multi-category).

        Args:
            category_name: Category to search
            entity_type: 'intervention', 'condition', or 'mechanism'
            category_type_filter: Optional filter by category_type

        Returns:
            List of entity dicts
        """
        try:
            query_map = {
                'intervention': """
                    SELECT i.*, icm.category_type, icm.confidence as category_confidence
                    FROM interventions i
                    JOIN intervention_category_mapping icm ON i.id = icm.intervention_id
                    WHERE icm.category_name = ?
                """,
                'condition': """
                    SELECT DISTINCT ccm.condition_name, ccm.category_type, ccm.confidence as category_confidence
                    FROM condition_category_mapping ccm
                    WHERE ccm.category_name = ?
                """,
                'mechanism': """
                    SELECT mc.*, mcm.category_type, mcm.confidence as category_confidence
                    FROM mechanism_clusters mc
                    JOIN mechanism_category_mapping mcm ON mc.cluster_id = mcm.mechanism_cluster_id
                    WHERE mcm.category_name = ?
                """
            }

            if entity_type not in query_map:
                raise ValueError(f"Invalid entity_type: {entity_type}")

            query = query_map[entity_type]
            params = [category_name]

            if category_type_filter:
                query += " AND category_type = ?"
                params.append(category_type_filter)

            return self.execute_query(query, tuple(params))

        except Exception as e:
            logger.error(f"Error getting entities by category: {e}")
            return []

    def get_primary_category(
        self,
        entity_type: str,
        entity_id: Any
    ) -> Optional[str]:
        """
        Get the primary category for an entity (backward compatibility).

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'
            entity_id: Entity identifier

        Returns:
            Primary category name or None
        """
        categories = self.get_entity_categories(entity_type, entity_id, category_type_filter='primary')
        return categories[0]['category_name'] if categories else None
