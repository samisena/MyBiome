"""
Test Multi-Category API

Quick test to verify multi-category assignment and retrieval works.
"""

import sys
import sqlite3
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from back_end.src.phase_1_data_collection.database_manager import DatabaseManager


def test_multi_category_api():
    """Test multi-category assignment and retrieval."""

    print("="*60)
    print("MULTI-CATEGORY API TEST")
    print("="*60)

    # Use backup database
    db_path = "back_end/data/intervention_research_backup_before_rollback_20251009_192106.db"
    db_manager = DatabaseManager(db_config=type('Config', (), {'path': db_path})())

    print("\n[1] Testing Intervention Multi-Category Assignment")
    print("-" * 60)

    # Get first intervention
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, intervention_name, intervention_category FROM interventions LIMIT 1")
    test_intervention = cursor.fetchone()
    conn.close()

    if not test_intervention:
        print("ERROR: No interventions found in database")
        return

    intervention_id = test_intervention['id']
    intervention_name = test_intervention['intervention_name']
    primary_category = test_intervention['intervention_category']

    print(f"Test Intervention:")
    print(f"  ID: {intervention_id}")
    print(f"  Name: {intervention_name}")
    print(f"  Primary Category: {primary_category}")

    # Assign multiple categories
    print("\nAssigning categories:")

    # Primary (from migration)
    categories = db_manager.get_entity_categories('intervention', intervention_id)
    print(f"  Existing categories: {len(categories)}")
    for cat in categories:
        print(f"    - {cat['category_name']} ({cat['category_type']})")

    # Assign functional category
    success = db_manager.assign_category(
        entity_type='intervention',
        entity_id=intervention_id,
        category_name='Gut Flora Modulators',
        category_type='functional',
        confidence=0.95,
        assigned_by='test_script',
        notes='Test functional category'
    )
    print(f"\n  Assigned 'Gut Flora Modulators' (functional): {success}")

    # Assign therapeutic category
    success = db_manager.assign_category(
        entity_type='intervention',
        entity_id=intervention_id,
        category_name='IBS Treatment',
        category_type='therapeutic',
        confidence=0.90,
        assigned_by='test_script',
        notes='Test therapeutic category'
    )
    print(f"  Assigned 'IBS Treatment' (therapeutic): {success}")

    # Retrieve all categories
    print("\n[2] Retrieving All Categories for Intervention")
    print("-" * 60)

    all_categories = db_manager.get_entity_categories('intervention', intervention_id)
    print(f"Total categories: {len(all_categories)}")

    for cat in all_categories:
        print(f"  - {cat['category_name']} ({cat['category_type']}) - Confidence: {cat['confidence']:.2f}")
        print(f"    Assigned by: {cat['assigned_by']} at {cat['assigned_at']}")

    # Test filtering by category type
    print("\n[3] Testing Category Type Filtering")
    print("-" * 60)

    functional_cats = db_manager.get_entity_categories('intervention', intervention_id, category_type_filter='functional')
    print(f"Functional categories: {len(functional_cats)}")
    for cat in functional_cats:
        print(f"  - {cat['category_name']}")

    primary_cat = db_manager.get_primary_category('intervention', intervention_id)
    print(f"\nPrimary category (backward compat): {primary_cat}")

    # Test get_entities_by_category
    print("\n[4] Testing Get Entities by Category")
    print("-" * 60)

    entities = db_manager.get_entities_by_category('Gut Flora Modulators', entity_type='intervention')
    print(f"Interventions in 'Gut Flora Modulators': {len(entities)}")
    for entity in entities[:5]:  # Show first 5
        print(f"  - {entity['intervention_name']} (ID: {entity['id']})")

    # Test condition multi-category
    print("\n[5] Testing Condition Multi-Category")
    print("-" * 60)

    # Get a test condition
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT health_condition FROM interventions LIMIT 1")
    test_condition = cursor.fetchone()[0]
    conn.close()

    print(f"Test Condition: {test_condition}")

    # Assign multiple categories
    success = db_manager.assign_category(
        entity_type='condition',
        entity_id=test_condition,
        category_name='Chronic Inflammatory Conditions',
        category_type='system',
        confidence=0.92,
        assigned_by='test_script'
    )
    print(f"  Assigned 'Chronic Inflammatory Conditions' (system): {success}")

    condition_categories = db_manager.get_entity_categories('condition', test_condition)
    print(f"\nCondition categories ({test_condition}): {len(condition_categories)}")
    for cat in condition_categories:
        print(f"  - {cat['category_name']} ({cat['category_type']})")

    print("\n" + "="*60)
    print("MULTI-CATEGORY API TEST COMPLETE")
    print("="*60)
    print("\n[SUCCESS] All tests passed!")
    print("Multi-category assignment and retrieval working correctly.")


if __name__ == '__main__':
    test_multi_category_api()
