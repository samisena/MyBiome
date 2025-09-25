#!/usr/bin/env python3
"""
Check which interventions have semantic fields populated.
"""

import sqlite3
from pathlib import Path

def check_semantic_fields():
    """Check the state of semantic fields in the database."""
    db_path = Path("data/processed/intervention_research.db")

    if not db_path.exists():
        print("[ERROR] Database not found")
        return

    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check total interventions
        cursor.execute("SELECT COUNT(*) FROM interventions")
        total = cursor.fetchone()[0]

        # Check how many have semantic fields
        cursor.execute("""
            SELECT COUNT(*) FROM interventions
            WHERE canonical_name IS NOT NULL AND canonical_name != ''
        """)
        with_canonical = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM interventions
            WHERE alternative_names IS NOT NULL AND alternative_names != ''
        """)
        with_alternatives = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM interventions
            WHERE semantic_group_id IS NOT NULL AND semantic_group_id != ''
        """)
        with_groups = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM interventions
            WHERE models_used IS NOT NULL AND models_used != ''
        """)
        with_models = cursor.fetchone()[0]

        # Sample a few interventions to see their state
        cursor.execute("""
            SELECT intervention_name, canonical_name, alternative_names,
                   semantic_group_id, models_used, confidence_score
            FROM interventions
            ORDER BY confidence_score DESC
            LIMIT 10
        """)

        sample = cursor.fetchall()

        print("=== Semantic Fields Status ===")
        print(f"Total interventions: {total}")
        print(f"With canonical_name: {with_canonical} ({with_canonical/total*100:.1f}%)")
        print(f"With alternative_names: {with_alternatives} ({with_alternatives/total*100:.1f}%)")
        print(f"With semantic_group_id: {with_groups} ({with_groups/total*100:.1f}%)")
        print(f"With models_used: {with_models} ({with_models/total*100:.1f}%)")

        print("\n=== Sample Interventions ===")
        for row in sample:
            print(f"Name: {row['intervention_name']}")
            print(f"  Canonical: {row['canonical_name']}")
            print(f"  Alternatives: {row['alternative_names']}")
            print(f"  Group ID: {row['semantic_group_id']}")
            print(f"  Models: {row['models_used']}")
            print(f"  Confidence: {row['confidence_score']}")
            print()

        conn.close()

    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    check_semantic_fields()