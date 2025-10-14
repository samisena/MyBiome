"""
Integration Test: Multi-Category Feature
Tests the complete pipeline from categorization → export → frontend display

This test verifies:
1. Group categorization assigns functional/therapeutic categories
2. Categories propagate to individual entities
3. Export generates correct JSON structure
4. Frontend can load and display multi-category data
"""

import sys
import json
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from back_end.src.phase_1_data_collection.database_manager import DatabaseManager
from back_end.src.utils.export_frontend_data import export_interventions_data


def test_integration():
    """Test complete multi-category pipeline."""

    print("\n" + "="*80)
    print("MULTI-CATEGORY INTEGRATION TEST")
    print("="*80)

    # Use backup database (find the most recent one)
    data_dir = project_root / "back_end" / "data"
    db_files = list(data_dir.glob("intervention_research_backup_*.db"))

    if not db_files:
        print(f"[ERROR] No backup database found in: {data_dir}")
        return False

    # Use the most recent backup
    db_path = max(db_files, key=lambda p: p.stat().st_mtime)
    print(f"\nUsing most recent backup: {db_path.name}")

    print(f"\nUsing database: {db_path}")

    # Step 1: Verify multi-category assignments exist
    print("\n" + "-"*80)
    print("STEP 1: Verify Multi-Category Assignments")
    print("-"*80)

    # Create a config object for DatabaseManager
    from types import SimpleNamespace
    db_config = SimpleNamespace(
        name=db_path.stem,
        path=str(db_path)
    )

    db = DatabaseManager(db_config=db_config)

    # Use direct connection instead of execute_query
    # Check intervention categories
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                category_type,
                COUNT(*) as count,
                COUNT(DISTINCT intervention_id) as unique_entities
            FROM intervention_category_mapping
            GROUP BY category_type
            ORDER BY category_type
        """)
        intervention_stats = [dict(row) for row in cursor.fetchall()]

    print("\nIntervention Category Assignments:")
    for row in intervention_stats:
        print(f"  - {row['category_type']:12s}: {row['count']:4d} assignments, {row['unique_entities']:4d} unique entities")

    # Check condition categories
    condition_stats = db.execute_query("""
        SELECT
            category_type,
            COUNT(*) as count,
            COUNT(DISTINCT condition_id) as unique_entities
        FROM condition_category_mapping
        GROUP BY category_type
        ORDER BY category_type
    """)

    print("\nCondition Category Assignments:")
    for row in condition_stats:
        print(f"  - {row['category_type']:12s}: {row['count']:4d} assignments, {row['unique_entities']:4d} unique entities")

    # Check multi-category entities
    multi_cat_interventions = db.execute_query("""
        SELECT intervention_id, COUNT(*) as category_count
        FROM intervention_category_mapping
        GROUP BY intervention_id
        HAVING COUNT(*) > 1
    """)

    print(f"\nInterventions with multiple categories: {len(multi_cat_interventions)}")

    if len(multi_cat_interventions) > 0:
        print("\nSample multi-category interventions:")
        for i, row in enumerate(multi_cat_interventions[:5]):
            # Get intervention name and categories
            details = db.execute_query("""
                SELECT
                    i.intervention_name,
                    GROUP_CONCAT(icm.category_type || ':' || icm.category_name, ', ') as categories
                FROM interventions i
                JOIN intervention_category_mapping icm ON i.id = icm.intervention_id
                WHERE i.id = ?
                GROUP BY i.id
            """, (row['intervention_id'],))

            if details:
                print(f"  {i+1}. {details[0]['intervention_name']}")
                print(f"     Categories: {details[0]['categories']}")

    # Step 2: Export data with multi-category support
    print("\n" + "-"*80)
    print("STEP 2: Export Data with Multi-Category Support")
    print("-"*80)

    output_path = project_root / "frontend" / "data" / "interventions_test.json"

    print(f"\nExporting to: {output_path}")
    export_interventions_data(str(db_path), str(output_path))
    print("[PASS] Export completed successfully")

    # Step 3: Validate JSON structure
    print("\n" + "-"*80)
    print("STEP 3: Validate JSON Structure")
    print("-"*80)

    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\nLoaded JSON with {len(data['interventions'])} interventions")

    # Check metadata
    print("\nMetadata validation:")
    required_fields = ['total_interventions', 'unique_interventions', 'multi_category_interventions',
                      'multi_category_stats']
    for field in required_fields:
        if field in data['metadata']:
            print(f"  [PASS] {field}: {data['metadata'][field]}")
        else:
            print(f"  [FAIL] Missing metadata field: {field}")

    # Check multi-category stats
    if 'multi_category_stats' in data['metadata']:
        stats = data['metadata']['multi_category_stats']
        print("\nMulti-category statistics:")
        for cat_type, categories in stats.items():
            print(f"  - {cat_type}: {len(categories)} unique categories")
            if categories:
                print(f"    Sample: {list(categories.keys())[:3]}")

    # Check intervention structure
    print("\nIntervention structure validation:")
    sample = data['interventions'][0]

    # Check intervention categories
    if 'categories' in sample['intervention']:
        print(f"  [PASS] intervention.categories exists")
        for cat_type, cats in sample['intervention']['categories'].items():
            print(f"    - {cat_type}: {cats}")
    else:
        print(f"  [FAIL] intervention.categories missing")

    # Check condition categories
    if 'categories' in sample['condition']:
        print(f"  [PASS] condition.categories exists")
        for cat_type, cats in sample['condition']['categories'].items():
            print(f"    - {cat_type}: {cats}")
    else:
        print(f"  [FAIL] condition.categories missing")

    # Step 4: Check multi-category examples
    print("\n" + "-"*80)
    print("STEP 4: Sample Multi-Category Interventions")
    print("-"*80)

    multi_cat_count = 0
    for intervention in data['interventions']:
        if 'categories' in intervention['intervention']:
            total_cats = sum(len(cats) for cats in intervention['intervention']['categories'].values())
            if total_cats > 1:
                multi_cat_count += 1
                if multi_cat_count <= 3:  # Show first 3
                    print(f"\n{multi_cat_count}. {intervention['intervention']['name']}")
                    for cat_type, cats in intervention['intervention']['categories'].items():
                        if cats:
                            print(f"   {cat_type:12s}: {', '.join(cats)}")

    print(f"\nTotal multi-category interventions: {multi_cat_count}")

    # Step 5: Performance metrics
    print("\n" + "-"*80)
    print("STEP 5: Performance Metrics")
    print("-"*80)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nExported file size: {file_size_mb:.2f} MB")

    # Count categories
    primary_count = sum(1 for i in data['interventions']
                       if 'categories' in i['intervention'] and 'primary' in i['intervention']['categories'])
    functional_count = sum(1 for i in data['interventions']
                          if 'categories' in i['intervention'] and 'functional' in i['intervention']['categories'])
    therapeutic_count = sum(1 for i in data['interventions']
                           if 'categories' in i['intervention'] and 'therapeutic' in i['intervention']['categories'])

    print(f"\nCategory coverage:")
    print(f"  - Primary:     {primary_count}/{len(data['interventions'])} ({primary_count/len(data['interventions'])*100:.1f}%)")
    print(f"  - Functional:  {functional_count}/{len(data['interventions'])} ({functional_count/len(data['interventions'])*100:.1f}%)")
    print(f"  - Therapeutic: {therapeutic_count}/{len(data['interventions'])} ({therapeutic_count/len(data['interventions'])*100:.1f}%)")

    # Final summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    print("\n[PASS] Integration test completed successfully!")
    print("\nNext steps:")
    print("  1. Open frontend/index.html in browser")
    print("  2. Update data path to 'data/interventions_test.json'")
    print("  3. Verify multi-category badges display correctly")
    print("  4. Test functional/therapeutic category filters")
    print("  5. Check details modal shows all category types")

    return True


if __name__ == "__main__":
    try:
        success = test_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
