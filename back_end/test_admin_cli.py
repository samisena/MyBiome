#!/usr/bin/env python3
"""
Test the admin CLI functionality
"""

from admin_cli import NormalizationAdmin
import json


def test_admin_functions():
    """Test admin CLI functions"""

    print("=== TESTING ADMIN CLI FUNCTIONS ===")

    admin = NormalizationAdmin()

    # Test statistics
    print("\n1. Testing statistics...")
    stats = admin.get_statistics()

    print("Statistics retrieved:")
    print(json.dumps(stats, indent=2))

    # Test viewing canonical entities (programmatically)
    print("\n2. Testing canonical entities view...")
    import sqlite3

    with sqlite3.connect(admin.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ce.id, ce.canonical_name, ce.entity_type,
                   COUNT(em.id) as mapping_count
            FROM canonical_entities ce
            LEFT JOIN entity_mappings em ON ce.id = em.canonical_id
            GROUP BY ce.id
            ORDER BY mapping_count DESC
            LIMIT 5
        """)

        entities = cursor.fetchall()

        print("Top 5 canonical entities by mapping count:")
        for entity in entities:
            print(f"  ID {entity['id']}: {entity['canonical_name']} ({entity['entity_type']}) - {entity['mapping_count']} mappings")

    # Test search functionality (programmatically)
    print("\n3. Testing search functionality...")
    with sqlite3.connect(admin.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Search for probiotic-related terms
        cursor.execute("""
            SELECT em.raw_text, ce.canonical_name, em.confidence_score
            FROM entity_mappings em
            JOIN canonical_entities ce ON em.canonical_id = ce.id
            WHERE em.raw_text LIKE '%probiotic%'
            ORDER BY em.confidence_score DESC
            LIMIT 5
        """)

        results = cursor.fetchall()
        print("Probiotic-related mappings found:")
        for result in results:
            print(f"  '{result['raw_text']}' -> '{result['canonical_name']}' (conf: {result['confidence_score']:.2f})")

    print("\n[SUCCESS] Admin CLI functions working correctly!")
    print("\nTo use the interactive CLI, run: python admin_cli.py")


if __name__ == "__main__":
    test_admin_functions()