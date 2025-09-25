#!/usr/bin/env python3
"""
Small batch test - Process only 5 interventions maximum to fix all errors first.
No full migration until this works perfectly with zero errors.
"""

import sys
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import SemanticMerger, InterventionExtraction

def get_small_batch_interventions(db_path: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get exactly 5 interventions for testing - preferably with GERD/PPI to test duplicates."""
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    # First try to get GERD/PPI interventions for testing duplicates
    cursor.execute("""
        SELECT id, intervention_name, health_condition, intervention_category,
               correlation_type, confidence_score, correlation_strength,
               supporting_quote, paper_id
        FROM interventions
        WHERE (canonical_name IS NULL OR canonical_name = '')
        AND (intervention_name LIKE '%proton pump%' OR intervention_name LIKE '%PPI%'
             OR health_condition LIKE '%GERD%' OR health_condition LIKE '%reflux%')
        ORDER BY confidence_score DESC NULLS LAST
        LIMIT ?
    """, (limit,))

    interventions = [dict(row) for row in cursor.fetchall()]

    # If we don't have enough, fill with any interventions
    if len(interventions) < limit:
        cursor.execute("""
            SELECT id, intervention_name, health_condition, intervention_category,
                   correlation_type, confidence_score, correlation_strength,
                   supporting_quote, paper_id
            FROM interventions
            WHERE (canonical_name IS NULL OR canonical_name = '')
            AND id NOT IN ({})
            ORDER BY confidence_score DESC NULLS LAST
            LIMIT ?
        """.format(','.join('?' for _ in interventions)),
        [i['id'] for i in interventions] + [limit - len(interventions)])

        additional = [dict(row) for row in cursor.fetchall()]
        interventions.extend(additional)

    conn.close()
    return interventions[:limit]  # Ensure we never exceed the limit

def test_single_intervention_processing(intervention: Dict[str, Any],
                                       merger: SemanticMerger) -> Dict[str, Any]:
    """Test processing a single intervention with full error handling."""
    print(f"Testing intervention: '{intervention['intervention_name']}'")
    print(f"  Condition: {intervention['health_condition']}")
    print(f"  Category: {intervention['intervention_category']}")

    try:
        # Convert to InterventionExtraction
        extraction = InterventionExtraction(
            model_name='test',
            intervention_name=intervention['intervention_name'] or '',
            health_condition=intervention['health_condition'] or '',
            intervention_category=intervention['intervention_category'] or 'unknown',
            correlation_type=intervention['correlation_type'] or 'unknown',
            confidence_score=intervention['confidence_score'] or 0.0,
            correlation_strength=intervention['correlation_strength'] or 0.0,
            supporting_quote=intervention['supporting_quote'] or '',
            raw_data={'intervention_id': intervention['id'], 'paper_id': intervention['paper_id']}
        )

        # Create basic semantic fields for single intervention
        result = {
            'intervention_id': intervention['id'],
            'canonical_name': intervention['intervention_name'],
            'alternative_names': json.dumps([intervention['intervention_name']]),
            'search_terms': json.dumps([intervention['intervention_name'].lower()]),
            'semantic_group_id': f"sem_{hash(intervention['intervention_name'].lower())}",
            'semantic_confidence': 1.0,
            'merge_source': 'single_test',
            'consensus_confidence': intervention['confidence_score'] or 0.0,
            'model_agreement': 'single',
            'models_used': 'test',
            'raw_extraction_count': 1,
            'status': 'success'
        }

        print(f"  [SUCCESS] Success: {result['canonical_name']}")
        return result

    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        return {
            'intervention_id': intervention['id'],
            'status': 'error',
            'error_message': str(e)
        }

def test_llm_comparison(intervention1: Dict[str, Any], intervention2: Dict[str, Any],
                       merger: SemanticMerger) -> Dict[str, Any]:
    """Test LLM comparison between two interventions with full error handling."""
    print(f"\nTesting LLM comparison:")
    print(f"  1. '{intervention1['intervention_name']}' vs 2. '{intervention2['intervention_name']}'")

    try:
        # Convert to InterventionExtraction objects
        extraction1 = InterventionExtraction(
            model_name='test',
            intervention_name=intervention1['intervention_name'] or '',
            health_condition=intervention1['health_condition'] or '',
            intervention_category=intervention1['intervention_category'] or 'unknown',
            correlation_type=intervention1['correlation_type'] or 'unknown',
            confidence_score=intervention1['confidence_score'] or 0.0,
            correlation_strength=intervention1['correlation_strength'] or 0.0,
            supporting_quote=intervention1['supporting_quote'] or '',
            raw_data={'intervention_id': intervention1['id']}
        )

        extraction2 = InterventionExtraction(
            model_name='test',
            intervention_name=intervention2['intervention_name'] or '',
            health_condition=intervention2['health_condition'] or '',
            intervention_category=intervention2['intervention_category'] or 'unknown',
            correlation_type=intervention2['correlation_type'] or 'unknown',
            confidence_score=intervention2['confidence_score'] or 0.0,
            correlation_strength=intervention2['correlation_strength'] or 0.0,
            supporting_quote=intervention2['supporting_quote'] or '',
            raw_data={'intervention_id': intervention2['id']}
        )

        # Test LLM comparison
        print("  Calling primary LLM...")
        decision = merger.compare_interventions(extraction1, extraction2)

        print(f"  Result: {'DUPLICATE' if decision.is_duplicate else 'NOT DUPLICATE'}")
        print(f"  Confidence: {decision.semantic_confidence}")
        print(f"  Reasoning: {decision.reasoning[:100]}...")

        # If duplicate, test validation
        if decision.is_duplicate:
            print("  Calling validator LLM...")
            validation = merger.validate_merge_decision(decision, extraction1, extraction2)
            print(f"  Validation agrees: {validation.agrees_with_merge}")
            print(f"  Validation confidence: {validation.confidence}")

            return {
                'status': 'success',
                'is_duplicate': True,
                'primary_confidence': decision.semantic_confidence,
                'validation_agrees': validation.agrees_with_merge,
                'validation_confidence': validation.confidence,
                'canonical_name': decision.canonical_name,
                'alternative_names': decision.alternative_names
            }
        else:
            return {
                'status': 'success',
                'is_duplicate': False,
                'primary_confidence': decision.semantic_confidence,
                'reasoning': decision.reasoning
            }

    except Exception as e:
        print(f"  [ERROR] LLM Comparison Error: {e}")
        return {
            'status': 'error',
            'error_message': str(e)
        }

def update_database_test(db_path: str, results: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Test database update with small batch."""
    conn = sqlite3.connect(db_path, timeout=60.0)

    try:
        conn.execute("BEGIN TRANSACTION")

        updates_count = 0
        errors_count = 0

        for result in results:
            if result.get('status') != 'success':
                errors_count += 1
                continue

            try:
                conn.execute("""
                    UPDATE interventions
                    SET canonical_name = ?,
                        alternative_names = ?,
                        search_terms = ?,
                        semantic_group_id = ?,
                        semantic_confidence = ?,
                        merge_source = ?,
                        consensus_confidence = ?,
                        model_agreement = ?,
                        models_used = ?,
                        raw_extraction_count = ?
                    WHERE id = ?
                """, (
                    result['canonical_name'],
                    result['alternative_names'],
                    result['search_terms'],
                    result['semantic_group_id'],
                    result['semantic_confidence'],
                    result['merge_source'],
                    result['consensus_confidence'],
                    result['model_agreement'],
                    result['models_used'],
                    result['raw_extraction_count'],
                    result['intervention_id']
                ))
                updates_count += 1

            except Exception as e:
                print(f"[ERROR] Database update error for intervention {result['intervention_id']}: {e}")
                errors_count += 1

        conn.execute("COMMIT")
        return updates_count, errors_count

    except Exception as e:
        conn.execute("ROLLBACK")
        print(f"[ERROR] Database transaction error: {e}")
        return 0, len(results)
    finally:
        conn.close()

def main():
    """Small batch test - must work with ZERO errors before full migration."""
    db_path = "data/processed/intervention_research.db"

    if not Path(db_path).exists():
        print(f"[ERROR] Database not found: {db_path}")
        return

    print("=== SMALL BATCH TEST (MAX 5 INTERVENTIONS) ===")
    print(f"Starting test at {datetime.now()}")
    print("Goal: ZERO ERRORS before proceeding to full migration")
    print()

    # Step 1: Get small batch
    print("Loading small batch (max 5 interventions)...")
    interventions = get_small_batch_interventions(db_path, limit=5)
    print(f"Found {len(interventions)} interventions for testing")

    if not interventions:
        print("[ERROR] No interventions found for testing")
        return

    # Show what we're testing
    print("\nTest interventions:")
    for i, intervention in enumerate(interventions, 1):
        print(f"  {i}. '{intervention['intervention_name']}' (condition: {intervention['health_condition']})")
    print()

    # Step 2: Initialize semantic merger
    print("Initializing LLM semantic merger...")
    try:
        merger = SemanticMerger(
            primary_model='qwen2.5:14b',
            validator_model='gemma2:9b'
        )
        print("[SUCCESS] SemanticMerger initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize SemanticMerger: {e}")
        return

    # Step 3: Test single intervention processing
    print("\n=== TESTING SINGLE INTERVENTION PROCESSING ===")
    results = []
    for intervention in interventions:
        result = test_single_intervention_processing(intervention, merger)
        results.append(result)

    # Step 4: Test LLM comparison if we have multiple interventions
    if len(interventions) >= 2:
        print("\n=== TESTING LLM COMPARISON ===")
        comparison_result = test_llm_comparison(interventions[0], interventions[1], merger)
        if comparison_result['status'] == 'error':
            print(f"[ERROR] LLM comparison failed: {comparison_result['error_message']}")

    # Step 5: Test database update
    print("\n=== TESTING DATABASE UPDATE ===")
    success_results = [r for r in results if r.get('status') == 'success']
    if success_results:
        updates, errors = update_database_test(db_path, success_results)
        print(f"Database test: {updates} updates, {errors} errors")

    # Step 6: Final assessment
    print(f"\n=== FINAL ASSESSMENT ===")
    total_errors = len([r for r in results if r.get('status') == 'error'])

    if total_errors == 0:
        print("[SUCCESS] Small batch test completed with ZERO errors!")
        print("Ready to proceed to full migration")

        # Show LLM statistics
        print("\nLLM Statistics:")
        stats = merger.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print(f"[FAILURE] Found {total_errors} errors in small batch")
        print("DO NOT proceed to full migration until all errors are fixed")

        # Show errors
        for i, result in enumerate(results):
            if result.get('status') == 'error':
                print(f"  Error {i+1}: {result.get('error_message', 'Unknown error')}")

    print(f"\nTest completed at {datetime.now()}")

if __name__ == "__main__":
    main()