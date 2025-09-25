#!/usr/bin/env python3
"""
Debug the safe matching method
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def debug_safe_matching():
    """Debug what the safe matching method returns"""

    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    print("=== DEBUGGING SAFE MATCHING ===\n")

    # Test with a known term
    test_terms = [
        ("probiotics", "intervention"),
        ("prebiotics", "intervention"),  # Should have no matches
        ("IBS", "condition"),
    ]

    for term, entity_type in test_terms:
        print(f"Testing: '{term}' ({entity_type})")

        safe_matches = normalizer.find_safe_matches_only(term, entity_type)
        print(f"  Safe matches returned: {len(safe_matches)}")

        for i, match in enumerate(safe_matches):
            print(f"  Match {i+1}: {match}")

        print()

    conn.close()


if __name__ == "__main__":
    debug_safe_matching()