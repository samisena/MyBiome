"""Temporary script to query confidence levels from database."""
import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from back_end.src.phase_1_data_collection.database_manager import database_manager

def main():
    print("\n" + "="*80)
    print("EXTRACTION CONFIDENCE ANALYSIS")
    print("="*80)

    with database_manager.get_connection() as conn:
        cursor = conn.cursor()

        # Get sample of each confidence level
        query = """
            SELECT
                i.intervention_name,
                i.extraction_confidence,
                i.correlation_strength,
                i.correlation_type,
                p.title,
                p.abstract
            FROM interventions i
            JOIN papers p ON i.paper_id = p.pmid
            WHERE i.extraction_confidence IS NOT NULL
            ORDER BY
                CASE i.extraction_confidence
                    WHEN 'very high' THEN 5
                    WHEN 'high' THEN 4
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 2
                    WHEN 'very low' THEN 1
                END DESC,
                RANDOM()
        """

        cursor.execute(query)
        results = cursor.fetchall()

        # Group by confidence level
        by_confidence = {}
        for row in results:
            conf = row[1]
            if conf not in by_confidence:
                by_confidence[conf] = []
            by_confidence[conf].append(row)

        # Display examples from each level
        for conf_level in ['very high', 'high', 'medium', 'low', 'very low']:
            if conf_level in by_confidence:
                examples = by_confidence[conf_level][:2]  # Get 2 examples
                print(f"\n{'='*80}")
                print(f"{conf_level.upper()} CONFIDENCE ({len(by_confidence[conf_level])} total)")
                print(f"{'='*80}")

                for idx, row in enumerate(examples, 1):
                    intervention = row[0]
                    strength = row[2] if row[2] else "N/A"
                    corr_type = row[3]
                    title = row[4]
                    abstract = row[5] if row[5] else "No abstract"

                    print(f"\nExample {idx}:")
                    print(f"  Intervention: {intervention}")
                    print(f"  Confidence: {conf_level}")
                    print(f"  Strength: {strength} | Type: {corr_type}")
                    print(f"  Title: {title[:150]}")
                    print(f"  Abstract excerpt: {abstract[:400]}...")
                    print()

if __name__ == '__main__':
    main()
