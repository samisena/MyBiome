#!/usr/bin/env python3
"""
Quick test to see what a full analysis would produce
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from generate_mapping_suggestions import MappingSuggestionGenerator


def main():
    """Run a quick analysis to show full potential"""

    db_path = "data/processed/intervention_research.db"
    generator = MappingSuggestionGenerator(db_path)

    try:
        # Load all terms to get counts
        interventions = generator.load_unique_terms('intervention')
        conditions = generator.load_unique_terms('condition')

        print("=== FULL DATABASE ANALYSIS POTENTIAL ===")
        print(f"Total unique interventions: {len(interventions)}")
        print(f"Total unique conditions: {len(conditions)}")
        print(f"Total unique terms: {len(interventions) + len(conditions)}")

        # Show top terms by frequency
        print(f"\nTop 10 interventions by frequency:")
        for i, (term, freq) in enumerate(interventions[:10]):
            print(f"  {i+1}. {term}: {freq}")

        print(f"\nTop 10 conditions by frequency:")
        for i, (term, freq) in enumerate(conditions[:10]):
            print(f"  {i+1}. {term}: {freq}")

        # Get existing mapping counts
        cursor = generator.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM canonical_entities")
        canonical_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM entity_mappings")
        mapping_count = cursor.fetchone()[0]

        print(f"\nExisting normalization system:")
        print(f"  Canonical entities: {canonical_count}")
        print(f"  Term mappings: {mapping_count}")

        print(f"\nTo run full analysis, edit generate_mapping_suggestions.py:")
        print(f"  Remove max_interventions and max_conditions limits")
        print(f"  This will process all {len(interventions) + len(conditions)} terms")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        generator.close()


if __name__ == "__main__":
    main()