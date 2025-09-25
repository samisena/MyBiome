#!/usr/bin/env python3
"""
Enhanced Database Manager with Entity Normalization Integration
Extends the existing database manager to automatically normalize entities during insertion
"""

import sqlite3
import json
import sys
import os
from typing import Dict, List, Optional, Any

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from paper_collection.database_manager import EnhancedDatabaseManager
from entity_normalizer import EntityNormalizer


class NormalizedDatabaseManager(EnhancedDatabaseManager):
    """Enhanced Database Manager with automatic entity normalization"""

    def __init__(self, db_path: str = None, enable_normalization: bool = True):
        """Initialize with normalization capabilities"""
        super().__init__(db_path)
        self.enable_normalization = enable_normalization
        self.normalizer = None

        if enable_normalization:
            try:
                # Create normalizer with shared connection pool
                conn = sqlite3.connect(self.db_path)
                self.normalizer = EntityNormalizer(conn)
                conn.close()
                print("Entity normalization enabled")
            except Exception as e:
                print(f"Warning: Could not enable entity normalization: {e}")
                self.enable_normalization = False

    def insert_intervention_normalized(self, intervention: Dict) -> bool:
        """Insert intervention with automatic entity normalization"""

        if not self.enable_normalization:
            # Fall back to standard insertion
            return self.insert_intervention(intervention)

        try:
            # Create a copy to avoid modifying the original
            normalized_intervention = intervention.copy()

            # Normalize intervention_name
            intervention_name = intervention.get('intervention_name', '').strip()
            if intervention_name:
                conn = sqlite3.connect(self.db_path)
                normalizer = EntityNormalizer(conn)

                intervention_mapping = normalizer.find_or_create_mapping(
                    intervention_name, 'intervention', confidence_threshold=0.7
                )

                normalized_intervention['intervention_canonical_id'] = intervention_mapping['canonical_id']

                if intervention_mapping['is_new']:
                    print(f"Created new intervention canonical: {intervention_mapping['canonical_name']}")
                elif intervention_mapping['method'] != 'exact_canonical':
                    print(f"Normalized '{intervention_name}' -> '{intervention_mapping['canonical_name']}' "
                          f"(method: {intervention_mapping['method']}, confidence: {intervention_mapping['confidence']:.2f})")

                conn.close()

            # Normalize health_condition
            health_condition = intervention.get('health_condition', '').strip()
            if health_condition:
                conn = sqlite3.connect(self.db_path)
                normalizer = EntityNormalizer(conn)

                condition_mapping = normalizer.find_or_create_mapping(
                    health_condition, 'condition', confidence_threshold=0.7
                )

                normalized_intervention['condition_canonical_id'] = condition_mapping['canonical_id']

                if condition_mapping['is_new']:
                    print(f"Created new condition canonical: {condition_mapping['canonical_name']}")
                elif condition_mapping['method'] != 'exact_canonical':
                    print(f"Normalized '{health_condition}' -> '{condition_mapping['canonical_name']}' "
                          f"(method: {condition_mapping['method']}, confidence: {condition_mapping['confidence']:.2f})")

                conn.close()

            # Mark as normalized
            normalized_intervention['normalized'] = True

            # Insert with normalized data
            return self._insert_intervention_with_normalization(normalized_intervention)

        except Exception as e:
            print(f"Error in normalized insertion: {e}")
            # Fall back to standard insertion
            return self.insert_intervention(intervention)

    def _insert_intervention_with_normalization(self, intervention: Dict) -> bool:
        """Insert intervention including normalization fields"""

        try:
            # Use existing validation
            from data.category_validator import category_validator
            validated_intervention = category_validator.validate_intervention(intervention)

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Enhanced INSERT query with normalization fields
                cursor.execute('''
                    INSERT OR REPLACE INTO interventions
                    (paper_id, intervention_category, intervention_name, intervention_details,
                     health_condition, correlation_type, correlation_strength, confidence_score,
                     sample_size, study_duration, study_type, population_details,
                     supporting_quote, delivery_method, severity, adverse_effects, cost_category,
                     extraction_model, validation_status, consensus_confidence, model_agreement,
                     models_used, raw_extraction_count, models_contributing,
                     intervention_canonical_id, condition_canonical_id, normalized)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validated_intervention['paper_id'] if 'paper_id' in validated_intervention else validated_intervention.get('pmid'),
                    validated_intervention['intervention_category'],
                    validated_intervention['intervention_name'],
                    json.dumps(validated_intervention.get('intervention_details', {})),
                    validated_intervention['health_condition'],
                    validated_intervention['correlation_type'],
                    validated_intervention.get('correlation_strength'),
                    validated_intervention.get('confidence_score'),
                    validated_intervention.get('sample_size'),
                    validated_intervention.get('study_duration'),
                    validated_intervention.get('study_type'),
                    validated_intervention.get('population_details'),
                    validated_intervention.get('supporting_quote'),
                    validated_intervention.get('delivery_method'),
                    validated_intervention.get('severity'),
                    validated_intervention.get('adverse_effects'),
                    validated_intervention.get('cost_category'),
                    validated_intervention.get('extraction_model', 'consensus'),
                    'pending',
                    validated_intervention.get('consensus_confidence'),
                    validated_intervention.get('model_agreement'),
                    validated_intervention.get('models_used'),
                    validated_intervention.get('raw_extraction_count', 1),
                    validated_intervention.get('models_contributing'),
                    # New normalization fields
                    validated_intervention.get('intervention_canonical_id'),
                    validated_intervention.get('condition_canonical_id'),
                    validated_intervention.get('normalized', False)
                ))

                return True

        except Exception as e:
            print(f"Error in normalized database insertion: {e}")
            import traceback
            traceback.print_exc()
            return False

    def batch_normalize_existing_interventions(self, limit: int = 100) -> Dict[str, Any]:
        """Normalize existing interventions that haven't been normalized yet"""

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

                # Get unnormalized interventions
                cursor.execute("""
                    SELECT id, intervention_name, health_condition
                    FROM interventions
                    WHERE (normalized IS NULL OR normalized = FALSE)
                    AND intervention_name IS NOT NULL
                    AND health_condition IS NOT NULL
                    LIMIT ?
                """, (limit,))

                interventions = cursor.fetchall()

                conn_norm = sqlite3.connect(self.db_path)
                normalizer = EntityNormalizer(conn_norm)

                for row in interventions:
                    intervention_id = row[0]
                    intervention_name = row[1]
                    health_condition = row[2]

                    try:
                        # Normalize intervention
                        intervention_mapping = normalizer.find_or_create_mapping(
                            intervention_name, 'intervention'
                        )

                        # Normalize condition
                        condition_mapping = normalizer.find_or_create_mapping(
                            health_condition, 'condition'
                        )

                        # Update the record
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

                conn_norm.close()
                conn.commit()

        except Exception as e:
            results['errors'].append(f"Batch normalization error: {e}")

        return results


def create_test_normalized_manager(db_path: str = None) -> NormalizedDatabaseManager:
    """Create a test instance of the normalized database manager"""

    if not db_path:
        db_path = "data/processed/intervention_research.db"

    return NormalizedDatabaseManager(db_path, enable_normalization=True)


# Test function
def test_normalized_insertion():
    """Test the normalized insertion functionality"""

    print("=== TESTING NORMALIZED DATABASE INSERTION ===\n")

    # Create test manager
    db_manager = create_test_normalized_manager()

    # Test intervention data
    test_interventions = [
        {
            'paper_id': 'test_001',
            'intervention_category': 'dietary',
            'intervention_name': 'probiotic supplements',  # Should map to existing "probiotics"
            'health_condition': 'irritable bowel syndrome',  # Should map to existing "irritable bowel syndrome"
            'correlation_type': 'positive',
            'correlation_strength': 0.8,
            'confidence_score': 0.9,
            'extraction_model': 'test_model'
        },
        {
            'paper_id': 'test_002',
            'intervention_category': 'dietary',
            'intervention_name': 'completely_new_intervention_xyz',  # Should create new canonical
            'health_condition': 'completely_new_condition_abc',  # Should create new canonical
            'correlation_type': 'positive',
            'correlation_strength': 0.6,
            'confidence_score': 0.7,
            'extraction_model': 'test_model'
        }
    ]

    print("Testing normalized insertion of test interventions...")

    for i, intervention in enumerate(test_interventions, 1):
        print(f"\nTest {i}: {intervention['intervention_name']} -> {intervention['health_condition']}")

        success = db_manager.insert_intervention_normalized(intervention)

        if success:
            print("✓ Insertion successful")
        else:
            print("✗ Insertion failed")

    print(f"\n[SUCCESS] Normalized insertion test completed")


if __name__ == "__main__":
    test_normalized_insertion()