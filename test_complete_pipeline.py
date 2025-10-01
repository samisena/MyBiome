"""
Complete end-to-end test of the vitamin D deduplication pipeline.
Tests: detection → merging → database storage → verification
"""
import sqlite3
import json
from datetime import datetime

from back_end.src.llm_processing.batch_entity_processor import BatchEntityProcessor
from back_end.src.data_collection.database_manager import database_manager

def setup_test_data():
    """Insert test interventions into database."""
    print("Setting up test data in database...")
    print("-" * 80)

    paper_id = "TEST_41031311"

    # First ensure paper exists
    with database_manager.get_connection() as conn:
        cursor = conn.cursor()

        # Delete any existing test data
        cursor.execute("DELETE FROM interventions WHERE paper_id = ?", (paper_id,))
        cursor.execute("DELETE FROM papers WHERE pmid = ?", (paper_id,))

        # Insert test paper
        cursor.execute("""
            INSERT INTO papers (pmid, title, abstract, publication_date, has_fulltext)
            VALUES (?, ?, ?, ?, ?)
        """, (
            paper_id,
            "Test Paper: Vitamin D and Cognitive Impairment in Diabetes",
            "Study examining the effects of vitamin D supplementation on cognitive impairment in type 2 diabetes patients.",
            "2024",
            0
        ))

        # Insert two interventions (simulating dual extraction)
        interventions = [
            {
                'paper_id': paper_id,
                'intervention_category': 'supplement',
                'intervention_name': 'vitamin D',
                'intervention_details': json.dumps({'dosage': '2000 IU daily'}),
                'health_condition': 'cognitive impairment',
                'correlation_type': 'positive',
                'correlation_strength': 0.75,
                'extraction_confidence': 0.85,
                'study_confidence': 0.80,
                'extraction_model': 'gemma2:9b',
                'supporting_quote': 'Vitamin D supplementation improved cognitive scores.'
            },
            {
                'paper_id': paper_id,
                'intervention_category': 'supplement',
                'intervention_name': 'vitamin D',
                'intervention_details': json.dumps({'dosage': '2000 IU daily'}),
                'health_condition': 'type 2 diabetes mellitus-induced cognitive impairment',
                'correlation_type': 'positive',
                'correlation_strength': 0.80,
                'extraction_confidence': 0.90,
                'study_confidence': 0.85,
                'extraction_model': 'qwen2.5:14b',
                'supporting_quote': 'Vitamin D improved cognitive function in diabetic patients.'
            }
        ]

        for intervention in interventions:
            cursor.execute("""
                INSERT INTO interventions (
                    paper_id, intervention_category, intervention_name, intervention_details,
                    health_condition, correlation_type, correlation_strength,
                    extraction_confidence, study_confidence, extraction_model, supporting_quote
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                intervention['paper_id'],
                intervention['intervention_category'],
                intervention['intervention_name'],
                intervention['intervention_details'],
                intervention['health_condition'],
                intervention['correlation_type'],
                intervention['correlation_strength'],
                intervention['extraction_confidence'],
                intervention['study_confidence'],
                intervention['extraction_model'],
                intervention['supporting_quote']
            ))

        conn.commit()

        print(f"Inserted 2 test interventions for paper {paper_id}")
        print()

    return paper_id

def verify_before_deduplication(paper_id):
    """Verify we have 2 separate interventions before deduplication."""
    print("BEFORE DEDUPLICATION:")
    print("-" * 80)

    with database_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, intervention_name, health_condition, extraction_model
            FROM interventions
            WHERE paper_id = ?
            ORDER BY id
        """, (paper_id,))

        interventions = cursor.fetchall()

        print(f"Found {len(interventions)} intervention(s):")
        for row in interventions:
            print(f"  ID {row[0]}: {row[1]} for {row[2]} (by {row[3]})")

        print()
        return len(interventions)

def run_deduplication(paper_id):
    """Run the deduplication process."""
    print("RUNNING DEDUPLICATION:")
    print("-" * 80)

    with database_manager.get_connection() as conn:
        processor = BatchEntityProcessor(conn, llm_model="qwen2.5:14b")

        # Get interventions for this paper
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM interventions WHERE paper_id = ? ORDER BY id
        """, (paper_id,))

        interventions = [dict(row) for row in cursor.fetchall()]

        # Add canonical names (simulating normalization step)
        for intervention in interventions:
            intervention['canonical_intervention_name'] = intervention['intervention_name'].lower().strip()
            intervention['canonical_condition_name'] = 'cognitive impairment'  # Both map to same canonical condition

        print(f"Processing {len(interventions)} interventions...")

        # Detect duplicates
        duplicate_groups = processor.duplicate_detector.detect_same_paper_duplicates(interventions)

        print(f"Found {len(duplicate_groups)} duplicate group(s)")
        print()

        if duplicate_groups:
            # Get paper info
            cursor.execute("SELECT * FROM papers WHERE pmid = ?", (paper_id,))
            paper = dict(cursor.fetchone())

            # Merge each duplicate group
            for i, group in enumerate(duplicate_groups, 1):
                print(f"Merging duplicate group {i} ({len(group)} interventions)...")

                merged = processor.duplicate_detector.merge_duplicate_group(group, paper)

                # Delete all interventions in the group
                for intervention in group:
                    cursor.execute("DELETE FROM interventions WHERE id = ?", (intervention['id'],))

                # Insert merged intervention
                # Skip id, runtime-only fields, and fields not in database schema
                skip_fields = {
                    'id', 'canonical_intervention_name', 'canonical_condition_name',
                    'duplicate_source', 'cross_model_validation', 'confidence_interval', 'consensus_metadata'
                }
                columns = []
                values = []
                for key, value in merged.items():
                    if key not in skip_fields:
                        columns.append(key)
                        # Convert lists/dicts to JSON strings for SQLite
                        if isinstance(value, (list, dict)):
                            values.append(json.dumps(value))
                        else:
                            values.append(value)

                placeholders = ','.join(['?' for _ in values])
                columns_str = ','.join(columns)

                cursor.execute(f"""
                    INSERT INTO interventions ({columns_str})
                    VALUES ({placeholders})
                """, values)

                print(f"  Merged into single intervention (ID: {cursor.lastrowid})")
                print()

            conn.commit()
            print("Deduplication complete")
        else:
            print("No duplicates found")

        print()

def verify_after_deduplication(paper_id):
    """Verify we have 1 merged intervention with consensus metadata."""
    print("AFTER DEDUPLICATION:")
    print("-" * 80)

    with database_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                id, intervention_name, health_condition, models_used,
                extraction_confidence, condition_wording_source,
                condition_wording_confidence, original_condition_wordings
            FROM interventions
            WHERE paper_id = ?
        """, (paper_id,))

        interventions = cursor.fetchall()

        print(f"Found {len(interventions)} intervention(s):")
        print()

        for row in interventions:
            print(f"Intervention ID: {row[0]}")
            print(f"  Name: {row[1]}")
            print(f"  Condition: {row[2]}")
            print(f"  Models: {row[3]}")
            print(f"  Extraction Confidence: {row[4]}")
            print(f"  Wording Source: {row[5]}")
            print(f"  Wording Confidence: {row[6]}")
            print(f"  Original Wordings: {row[7]}")
            print()

        return len(interventions)

def main():
    print("=" * 80)
    print("COMPLETE PIPELINE TEST")
    print("=" * 80)
    print()

    # Setup test data
    paper_id = setup_test_data()

    # Verify before
    count_before = verify_before_deduplication(paper_id)

    # Run deduplication
    run_deduplication(paper_id)

    # Verify after
    count_after = verify_after_deduplication(paper_id)

    # Test results
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    if count_before == 2 and count_after == 1:
        print("PASS: Successfully reduced 2 interventions to 1")

        # Get final merged intervention
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT models_used, condition_wording_source, original_condition_wordings
                FROM interventions
                WHERE paper_id = ?
            """, (paper_id,))

            row = cursor.fetchone()
            models_used = row[0]
            wording_source = row[1]
            original_wordings = row[2]

            checks = []
            checks.append(('Both models credited', 'gemma2:9b' in models_used and 'qwen2.5:14b' in models_used))
            checks.append(('Has consensus wording metadata', wording_source == 'llm_consensus'))
            checks.append(('Has original wordings', original_wordings is not None))

            print()
            for check_name, result in checks:
                status = "PASS" if result else "FAIL"
                print(f"  [{status}] {check_name}")

            all_passed = all(result for _, result in checks)

            print()
            if all_passed:
                print("SUCCESS: The vitamin D problem has been completely solved!")
                print("The pipeline correctly:")
                print("  1. Detected semantic duplicates from dual extraction")
                print("  2. Merged them into a single intervention")
                print("  3. Credited both models")
                print("  4. Selected consensus wording via LLM")
                print("  5. Stored all metadata in database")
            else:
                print("PARTIAL: Some checks failed")

    else:
        print(f"FAIL: Expected 2 interventions before and 1 after, got {count_before} and {count_after}")

    print("=" * 80)

if __name__ == "__main__":
    main()
