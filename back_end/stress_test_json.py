#!/usr/bin/env python3
"""
Stress test JSON parsing with multiple LLM comparisons and validation calls.
"""

import sys
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import SemanticMerger, InterventionExtraction

def get_similar_interventions(db_path: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Get interventions that are likely to be duplicates for stress testing."""
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, intervention_name, health_condition, intervention_category,
               correlation_type, confidence_score, correlation_strength,
               supporting_quote, paper_id
        FROM interventions
        WHERE (canonical_name IS NULL OR canonical_name = '')
        AND (intervention_name LIKE '%proton pump%' OR intervention_name LIKE '%PPI%'
             OR intervention_name LIKE '%vonoprazan%'
             OR intervention_name LIKE '%probiotic%'
             OR intervention_name LIKE '%FODMAP%')
        ORDER BY intervention_name, confidence_score DESC NULLS LAST
        LIMIT ?
    """, (limit,))

    interventions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return interventions

def stress_test_json_parsing(interventions: List[Dict[str, Any]], merger: SemanticMerger) -> bool:
    """Stress test JSON parsing with many LLM calls."""
    print(f"=== STRESS TESTING JSON PARSING ({len(interventions)} interventions) ===")

    json_errors = 0
    comparison_count = 0
    validation_count = 0

    # Convert to extractions
    extractions = []
    for intervention in interventions:
        extraction = InterventionExtraction(
            model_name='stress_test',
            intervention_name=intervention['intervention_name'] or '',
            health_condition=intervention['health_condition'] or '',
            intervention_category=intervention['intervention_category'] or 'unknown',
            correlation_type=intervention['correlation_type'] or 'unknown',
            confidence_score=intervention['confidence_score'] or 0.0,
            correlation_strength=intervention['correlation_strength'] or 0.0,
            supporting_quote=intervention['supporting_quote'] or '',
            raw_data={'intervention_id': intervention['id']}
        )
        extractions.append(extraction)

    # Test pairwise comparisons
    for i, ext1 in enumerate(extractions):
        for j, ext2 in enumerate(extractions[i+1:], i+1):
            comparison_count += 1
            print(f"Comparison {comparison_count}: '{ext1.intervention_name}' vs '{ext2.intervention_name}'")

            try:
                # Primary LLM call
                decision = merger.compare_interventions(ext1, ext2)
                print(f"  Primary result: {'DUPLICATE' if decision.is_duplicate else 'NOT DUPLICATE'} (conf: {decision.semantic_confidence})")

                # If duplicate, test validation (this triggers more JSON parsing)
                if decision.is_duplicate:
                    validation_count += 1
                    try:
                        validation = merger.validate_merge_decision(decision, ext1, ext2)
                        print(f"    Validation {validation_count}: agrees={validation.agrees_with_merge}, conf={validation.confidence}")
                    except Exception as val_error:
                        json_errors += 1
                        print(f"    [JSON ERROR] Validation failed: {val_error}")

            except Exception as comp_error:
                json_errors += 1
                print(f"  [JSON ERROR] Comparison failed: {comp_error}")

    print(f"\n=== STRESS TEST RESULTS ===")
    print(f"Total comparisons: {comparison_count}")
    print(f"Total validations: {validation_count}")
    print(f"JSON parsing errors: {json_errors}")

    if json_errors == 0:
        print("[SUCCESS] JSON parsing stress test passed with ZERO errors!")
        return True
    else:
        print(f"[FAILURE] Found {json_errors} JSON parsing errors!")
        return False

def main():
    """Run JSON parsing stress test."""
    db_path = "data/processed/intervention_research.db"

    if not Path(db_path).exists():
        print(f"[ERROR] Database not found: {db_path}")
        return

    print("=== JSON PARSING STRESS TEST ===")
    print(f"Starting at {datetime.now()}")
    print()

    # Get test interventions
    print("Loading interventions for JSON stress testing...")
    interventions = get_similar_interventions(db_path, limit=15)
    print(f"Found {len(interventions)} interventions")

    if len(interventions) < 2:
        print("[ERROR] Need at least 2 interventions")
        return

    print("\nTest interventions:")
    for i, intervention in enumerate(interventions, 1):
        print(f"  {i}. '{intervention['intervention_name']}' ({intervention['health_condition']})")
    print()

    # Initialize merger
    print("Initializing SemanticMerger...")
    try:
        merger = SemanticMerger(
            primary_model='qwen2.5:14b',
            validator_model='gemma2:9b'
        )
        print("[SUCCESS] SemanticMerger initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return

    # Run stress test
    success = stress_test_json_parsing(interventions, merger)

    # Show statistics
    print(f"\nLLM Statistics:")
    stats = merger.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nTest completed at {datetime.now()}")

    if success:
        print("\n[READY] JSON parsing is robust - ready for full migration!")
    else:
        print("\n[NOT READY] JSON parsing has errors - DO NOT run full migration!")

if __name__ == "__main__":
    main()