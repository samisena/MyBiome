#!/usr/bin/env python3
"""
Debug find_by_llm method directly
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def debug_find_by_llm():
    """Debug find_by_llm method directly"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    print("=== DEBUGGING find_by_llm METHOD ===\n")

    # Test with acid reflux -> GERD
    term = "acid reflux"
    entity_type = "condition"

    print(f"Testing: '{term}' ({entity_type})")

    # Test the method
    try:
        result = normalizer.find_by_llm(term, entity_type)

        if result:
            print(f"SUCCESS! Found match:")
            print(f"  Canonical name: {result['canonical_name']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Reasoning: {result.get('reasoning', 'No reasoning')}")
            print(f"  Cached: {result.get('cached', False)}")
        else:
            print("No match returned by find_by_llm method")

    except Exception as e:
        print(f"Error in find_by_llm: {e}")
        import traceback
        traceback.print_exc()

    # Test another known case
    print(f"\n--- Testing another case ---")
    term2 = "GERD"
    entity_type2 = "condition"

    print(f"Testing: '{term2}' ({entity_type2})")

    try:
        result2 = normalizer.find_by_llm(term2, entity_type2)

        if result2:
            print(f"SUCCESS! Found match:")
            print(f"  Canonical name: {result2['canonical_name']}")
            print(f"  Confidence: {result2['confidence']}")
            print(f"  Reasoning: {result2.get('reasoning', 'No reasoning')}")
            print(f"  Cached: {result2.get('cached', False)}")
        else:
            print("No match returned by find_by_llm method")

    except Exception as e:
        print(f"Error in find_by_llm: {e}")

    conn.close()
    print(f"\n[SUCCESS] Debug completed")


if __name__ == "__main__":
    debug_find_by_llm()