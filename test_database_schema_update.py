"""
Test script to verify database schema has been updated with consensus wording columns.
"""
import sqlite3

def check_schema():
    print("=" * 80)
    print("DATABASE SCHEMA CHECK")
    print("=" * 80)
    print()

    # Connect to database
    db_path = 'back_end/data/processed/intervention_research.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get interventions table schema
    cursor.execute("PRAGMA table_info(interventions)")
    columns = cursor.fetchall()

    print("Checking for consensus wording columns in interventions table...")
    print("-" * 80)

    required_columns = [
        'condition_wording_source',
        'condition_wording_confidence',
        'original_condition_wordings'
    ]

    found_columns = {}
    for col in columns:
        col_name = col[1]
        col_type = col[2]
        if col_name in required_columns:
            found_columns[col_name] = col_type
            print(f"✓ Found: {col_name} ({col_type})")

    print()
    print("RESULTS:")
    print("-" * 80)

    all_found = all(col in found_columns for col in required_columns)

    if all_found:
        print("SUCCESS: All consensus wording columns are present in the database schema")
    else:
        missing = [col for col in required_columns if col not in found_columns]
        print(f"MISSING COLUMNS: {missing}")
        print()
        print("These columns will be added automatically on next database connection.")

    print()
    print("Full interventions table schema:")
    print("-" * 80)
    for col in columns:
        print(f"  {col[1]:35} {col[2]:20} {'NOT NULL' if col[3] else ''}")

    conn.close()
    print()
    print("=" * 80)

if __name__ == "__main__":
    check_schema()
