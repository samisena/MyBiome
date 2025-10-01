"""
Test script for verifying two-stage duplicate detection and consensus merging.
Tests the vitamin D problem: same intervention, slightly different condition names.
"""
import sqlite3
import json
from datetime import datetime

from back_end.src.llm_processing.batch_entity_processor import BatchEntityProcessor

def create_test_interventions():
    """Create mock interventions simulating the vitamin D problem."""
    paper_id = "41031311"

    intervention1 = {
        'id': 1,
        'paper_id': paper_id,
        'intervention_name': 'vitamin D',
        'canonical_intervention_name': 'vitamin d',
        'health_condition': 'cognitive impairment',
        'canonical_condition_name': 'cognitive impairment',
        'correlation_type': 'positive',
        'extraction_model': 'gemma2:9b',
        'extraction_confidence': 0.85,
        'created_at': datetime.now().isoformat()
    }

    intervention2 = {
        'id': 2,
        'paper_id': paper_id,
        'intervention_name': 'vitamin D',
        'canonical_intervention_name': 'vitamin d',
        'health_condition': 'type 2 diabetes mellitus-induced cognitive impairment',
        'canonical_condition_name': 'cognitive impairment',
        'correlation_type': 'positive',
        'extraction_model': 'qwen2.5:14b',
        'extraction_confidence': 0.90,
        'created_at': datetime.now().isoformat()
    }

    return [intervention1, intervention2]

def create_test_paper():
    """Create mock paper object for merging."""
    return {
        'paper_id': '41031311',
        'pmid': '41031311',
        'title': 'Test Paper: Vitamin D and Cognitive Impairment',
        'abstract': 'Study on vitamin D effects on cognitive impairment in diabetic patients.'
    }

def main():
    print("=" * 80)
    print("VITAMIN D DEDUPLICATION TEST")
    print("=" * 80)
    print()

    # Create test interventions and paper
    interventions = create_test_interventions()
    paper = create_test_paper()

    print("TEST DATA:")
    print("-" * 80)
    for i, intervention in enumerate(interventions, 1):
        print(f"Intervention {i}:")
        print(f"  Paper ID: {intervention['paper_id']}")
        print(f"  Intervention: {intervention['intervention_name']}")
        print(f"  Condition: {intervention['health_condition']}")
        print(f"  Canonical Condition: {intervention['canonical_condition_name']}")
        print(f"  Model: {intervention['extraction_model']}")
        print(f"  Confidence: {intervention['extraction_confidence']}")
        print()

    # Connect to database
    db_path = 'back_end/data/processed/intervention_research.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Initialize BatchEntityProcessor
    print("Initializing BatchEntityProcessor...")
    processor = BatchEntityProcessor(conn, llm_model="qwen2.5:14b")
    print()

    # STEP 1: Test duplicate detection (Stage 1 + Stage 2)
    print("STEP 1: DUPLICATE DETECTION (Two-Stage)")
    print("-" * 80)

    duplicate_groups = processor.duplicate_detector.detect_same_paper_duplicates(interventions)

    print(f"Found {len(duplicate_groups)} duplicate group(s)")
    print()

    if duplicate_groups:
        for i, group in enumerate(duplicate_groups, 1):
            print(f"Duplicate Group {i} ({len(group)} interventions):")
            for intervention in group:
                print(f"  - ID {intervention['id']}: {intervention['health_condition']} "
                      f"(by {intervention['extraction_model']})")
            print()
    else:
        print("WARNING: No duplicate groups detected!")
        print("This means the two-stage detection did not find the semantic match.")
        print()

    # STEP 2: Test consensus merging
    if duplicate_groups:
        print("STEP 2: CONSENSUS MERGING")
        print("-" * 80)

        for i, group in enumerate(duplicate_groups, 1):
            print(f"Merging Duplicate Group {i}...")
            print()

            merged = processor.duplicate_detector.merge_duplicate_group(group, paper)

            print("MERGED RESULT:")
            print("-" * 80)
            print(f"Intervention Name: {merged.get('intervention_name')}")
            print(f"Health Condition: {merged.get('health_condition')}")
            print(f"Canonical Condition: {merged.get('canonical_condition_name')}")
            print(f"Models Used: {merged.get('models_used', 'N/A')}")
            print(f"Extraction Confidence: {merged.get('extraction_confidence')}")
            print()

            # Check for consensus wording fields
            if 'condition_wording_source' in merged:
                print("CONSENSUS WORDING METADATA:")
                print(f"  Source: {merged['condition_wording_source']}")
                print(f"  Confidence: {merged.get('condition_wording_confidence', 'N/A')}")
                print(f"  Original Wordings: {merged.get('original_condition_wordings', 'N/A')}")
                print()
            else:
                print("WARNING: No consensus wording metadata found!")
                print()

            # Verify both models are credited
            models_used = merged.get('models_used', '')
            if 'gemma2:9b' in models_used and 'qwen2.5:14b' in models_used:
                print("SUCCESS: Both models credited in merged intervention")
            else:
                print(f"WARNING: Models used = '{models_used}' (expected both models)")
            print()

    # STEP 3: Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if duplicate_groups:
        print("PASS: Duplicate detection found semantic matches")

        merged = processor.duplicate_detector.merge_duplicate_group(duplicate_groups[0], paper)

        has_consensus = 'condition_wording_source' in merged
        has_both_models = 'gemma2:9b' in merged.get('models_used', '') and 'qwen2.5:14b' in merged.get('models_used', '')

        if has_consensus and has_both_models:
            print("PASS: Consensus merging includes both models and wording metadata")
            print()
            print("FINAL VERIFICATION:")
            print(f"  Original conditions: 'cognitive impairment' vs 'type 2 diabetes mellitus-induced cognitive impairment'")
            print(f"  Selected condition: '{merged.get('health_condition')}'")
            print(f"  Models credited: {merged.get('models_used')}")
            print()
            print("The vitamin D problem has been SOLVED!")
        else:
            print("PARTIAL: Detection works but merging may be incomplete")
            if not has_consensus:
                print("  - Missing consensus wording metadata")
            if not has_both_models:
                print("  - Missing dual model attribution")
    else:
        print("FAIL: No duplicate groups detected")
        print("This likely means:")
        print("  1. Stage 1 (exact matching) failed due to different canonical conditions, OR")
        print("  2. Stage 2 (LLM semantic matching) did not find equivalence")
        print()
        print("Check the LLM prompt tuning for same-paper context.")

    print("=" * 80)

    conn.close()

if __name__ == "__main__":
    main()
