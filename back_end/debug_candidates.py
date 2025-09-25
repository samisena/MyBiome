#!/usr/bin/env python3
"""
Debug candidate selection issue
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_candidates():
    """Debug candidate selection"""

    # Connect to the database with row factory
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable dict-like access to rows

    print("=== DEBUGGING CANDIDATE SELECTION ===\n")

    entity_type = "condition"

    # Test the exact query from find_by_llm
    cursor = conn.cursor()
    cursor.execute("""
        SELECT canonical_name FROM canonical_entities WHERE entity_type = ?
        ORDER BY canonical_name
    """, (entity_type,))

    rows = cursor.fetchall()

    print(f"Number of rows fetched: {len(rows)}")
    print(f"Raw rows: {rows}")

    # Test the exact list comprehension
    candidate_canonicals = [row['canonical_name'] for row in rows]

    print(f"\nCandidate canonicals: {candidate_canonicals}")

    # Test direct access
    print(f"\nDirect database query check:")
    cursor.execute("SELECT canonical_name FROM canonical_entities WHERE entity_type = 'condition' ORDER BY canonical_name;")
    direct_results = cursor.fetchall()

    print(f"Direct results count: {len(direct_results)}")
    for i, row in enumerate(direct_results):
        print(f"  {i}: {row[0]}")

    conn.close()
    print(f"\n[SUCCESS] Debug completed")


if __name__ == "__main__":
    debug_candidates()