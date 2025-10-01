"""
Trigger database schema update by initializing DatabaseManager.
"""
from back_end.src.data_collection.database_manager import database_manager

print("Initializing DatabaseManager to trigger schema updates...")
print()

# This will trigger the create_tables() method which includes ALTER TABLE statements
with database_manager.get_connection() as conn:
    cursor = conn.cursor()

    # Verify the columns were added
    cursor.execute("PRAGMA table_info(interventions)")
    columns = cursor.fetchall()

    required_columns = [
        'condition_wording_source',
        'condition_wording_confidence',
        'original_condition_wordings'
    ]

    found_columns = {col[1] for col in columns}

    print("Checking for consensus wording columns...")
    for col in required_columns:
        if col in found_columns:
            print(f"  [OK] {col}")
        else:
            print(f"  [MISSING] {col}")

    print()
    all_found = all(col in found_columns for col in required_columns)

    if all_found:
        print("SUCCESS: All consensus wording columns have been added!")
    else:
        print("ERROR: Some columns were not added. Check database_manager.py")
