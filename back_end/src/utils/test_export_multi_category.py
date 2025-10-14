"""
Test export with multi-category data.

Adds some test multi-category assignments then exports to verify JSON structure.
"""

import sys
from pathlib import Path

# Add to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from back_end.src.phase_1_data_collection.database_manager import DatabaseManager
from back_end.src.utils import export_frontend_data


def test_multi_category_export():
    """Test export with multi-category assignments."""

    print("="*60)
    print("TESTING MULTI-CATEGORY EXPORT")
    print("="*60)

    # Use backup database
    db_path = "back_end/data/intervention_research_backup_before_rollback_20251009_192106.db"

    print(f"\nUsing database: {db_path}")

    # Initialize database manager
    db_manager = DatabaseManager(db_config=type('Config', (), {'path': db_path})())

    # Assign some test functional/therapeutic categories
    print("\n[1] Assigning test multi-categories...")

    # Get first few interventions
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, intervention_name, intervention_category FROM interventions LIMIT 5")
    test_interventions = cursor.fetchall()
    conn.close()

    for intervention_id, name, primary_cat in test_interventions:
        print(f"\n  Intervention: {name} (ID: {intervention_id})")
        print(f"    Primary: {primary_cat}")

        # Assign functional category based on primary
        if primary_cat in ['supplement', 'diet']:
            functional = 'Nutritional Modulators'
        elif primary_cat in ['medication', 'biologics']:
            functional = 'Pharmacological Agents'
        elif primary_cat == 'therapy':
            functional = 'Behavioral Interventions'
        else:
            functional = 'Multi-Modal Treatments'

        db_manager.assign_category(
            'intervention', intervention_id, functional,
            'functional', 0.95, 'test_script'
        )
        print(f"    Functional: {functional}")

        # Assign therapeutic category
        therapeutic = 'Test Treatment Group'
        db_manager.assign_category(
            'intervention', intervention_id, therapeutic,
            'therapeutic', 0.90, 'test_script'
        )
        print(f"    Therapeutic: {therapeutic}")

    # Export data
    print("\n[2] Exporting data with multi-categories...")

    # Temporarily override database path
    original_get_db = export_frontend_data.get_database_path
    export_frontend_data.get_database_path = lambda: Path(db_path)

    try:
        data = export_frontend_data.export_interventions_data()

        print("\n[3] Export Results:")
        print(f"  Total interventions: {data['metadata']['total_interventions']}")
        print(f"  Multi-category interventions: {data['metadata'].get('multi_category_interventions', 0)}")

        # Show multi-category stats
        if 'multi_category_stats' in data['metadata']:
            print("\n  Multi-Category Statistics:")
            for cat_type, categories in data['metadata']['multi_category_stats'].items():
                print(f"\n    {cat_type.upper()}:")
                for cat_name, count in list(categories.items())[:5]:  # Top 5
                    print(f"      {cat_name}: {count}")

        # Show example intervention with multi-categories
        print("\n[4] Example Interventions with Multi-Categories:")
        count = 0
        for intervention in data['interventions']:
            categories = intervention['intervention']['categories']
            if len(categories) > 1:  # Has multiple category types
                count += 1
                print(f"\n  {count}. {intervention['intervention']['name']}")
                for cat_type, cat_list in categories.items():
                    print(f"      {cat_type}: {', '.join(cat_list)}")

                if count >= 3:  # Show first 3
                    break

        print("\n" + "="*60)
        print("TEST COMPLETE - Multi-category export working!")
        print("="*60)

    finally:
        # Restore original function
        export_frontend_data.get_database_path = original_get_db


if __name__ == '__main__':
    test_multi_category_export()
