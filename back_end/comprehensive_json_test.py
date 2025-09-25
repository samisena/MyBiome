#!/usr/bin/env python3
"""
Comprehensive test to reproduce and fix JSON parsing errors.
This tests multiple LLM calls including validation to trigger the parsing issues.
"""

import sys
import sqlite3
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import SemanticMerger, InterventionExtraction

def get_test_interventions_for_json_errors(db_path: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get interventions that are likely to trigger JSON parsing errors - similar interventions."""
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    # Get interventions with similar names that should trigger LLM comparisons and validation
    cursor.execute("""
        SELECT id, intervention_name, health_condition, intervention_category,
               correlation_type, confidence_score, correlation_strength,
               supporting_quote, paper_id
        FROM interventions
        WHERE (canonical_name IS NULL OR canonical_name = '')
        AND (
            intervention_name LIKE '%proton pump%' OR
            intervention_name LIKE '%PPI%' OR
            intervention_name LIKE '%probiotic%' OR
            intervention_name LIKE '%FODMAP%' OR
            intervention_name LIKE '%diet%' OR
            intervention_name LIKE '%exercise%'
        )
        ORDER BY intervention_name, confidence_score DESC NULLS LAST
        LIMIT ?
    """, (limit,))

    interventions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return interventions

def stress_test_llm_comparisons(interventions: List[Dict[str, Any]], merger: SemanticMerger) -> List[Dict[str, Any]]:
    """Stress test LLM comparisons to trigger JSON parsing errors."""
    print(f"=== STRESS TESTING {len(interventions)} INTERVENTIONS ===")

    results = []
    comparison_count = 0
    validation_count = 0
    json_errors = []

    # Convert all to InterventionExtraction objects
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
            raw_data={'intervention_id': intervention['id'], 'paper_id': intervention['paper_id']}
        )
        extractions.append(extraction)

    # Test ALL pairwise comparisons to stress test the LLM
    for i, extraction1 in enumerate(extractions):
        for j, extraction2 in enumerate(extractions[i+1:], i+1):
            comparison_count += 1
            print(f"\nComparison {comparison_count}: '{extraction1.intervention_name}' vs '{extraction2.intervention_name}'")

            try:
                # Primary LLM comparison
                print("  Calling primary LLM...")
                decision = merger.compare_interventions(extraction1, extraction2)
                print(f"  Result: {'DUPLICATE' if decision.is_duplicate else 'NOT DUPLICATE'} (conf: {decision.semantic_confidence})")

                # If duplicate found, test validation (this is where JSON errors often occur)
                if decision.is_duplicate:
                    validation_count += 1
                    print(f"    Calling validator LLM (validation #{validation_count})...")
                    try:
                        validation = merger.validate_merge_decision(decision, extraction1, extraction2)
                        print(f"    Validation: agrees={validation.agrees_with_merge}, conf={validation.confidence}")
                    except Exception as validation_error:
                        json_errors.append({
                            'type': 'validation_error',
                            'comparison': f"{extraction1.intervention_name} vs {extraction2.intervention_name}",
                            'error': str(validation_error),
                            'comparison_number': comparison_count
                        })
                        print(f"    [ERROR] Validation failed: {validation_error}")

            except Exception as comparison_error:
                json_errors.append({
                    'type': 'comparison_error',
                    'comparison': f"{extraction1.intervention_name} vs {extraction2.intervention_name}",
                    'error': str(comparison_error),
                    'comparison_number': comparison_count
                })
                print(f"  [ERROR] Comparison failed: {comparison_error}")

    print(f"\n=== STRESS TEST RESULTS ===")
    print(f"Total comparisons attempted: {comparison_count}")
    print(f"Total validations attempted: {validation_count}")
    print(f"Total JSON errors encountered: {len(json_errors)}")

    if json_errors:
        print(f"\n=== JSON ERRORS FOUND ===")
        for i, error in enumerate(json_errors, 1):
            print(f"Error {i} ({error['type']}):")
            print(f"  Comparison: {error['comparison']}")
            print(f"  Error: {error['error']}")
            print(f"  At comparison #{error['comparison_number']}")
            print()

    return json_errors

def analyze_json_parsing_failures():
    """Analyze what causes JSON parsing failures."""
    print("=== ANALYZING JSON PARSING FAILURE PATTERNS ===")
    print()

    print("Common JSON parsing failure modes:")
    print("1. LLM returns malformed JSON (missing quotes, brackets, etc.)")
    print("2. LLM returns valid JSON but wrapped in markdown code blocks")
    print("3. LLM returns explanation text before/after JSON")
    print("4. LLM returns array instead of object")
    print("5. LLM returns None/null values")
    print("6. LLM response gets truncated mid-JSON")
    print("7. LLM includes comments in JSON (not valid JSON)")
    print()

    print("Current JSON repair strategies in utils.py:")
    print("- Strip markdown code blocks")
    print("- Extract JSON from mixed text")
    print("- Fix common JSON syntax errors")
    print("- Handle different quote types")
    print()

    print("If errors found, need to enhance JSON repair strategies...")

def test_json_repair_directly():
    """Test the JSON repair functions directly with problematic inputs."""
    print("=== TESTING JSON REPAIR FUNCTIONS DIRECTLY ===")

    # Import the JSON parsing utilities
    from src.data.utils import parse_json_safely

    # Test cases that might break JSON parsing
    test_cases = [
        # Valid JSON
        '{"is_duplicate": true, "confidence": 0.95}',

        # JSON in markdown code blocks
        '```json\n{"is_duplicate": true, "confidence": 0.95}\n```',

        # JSON with text around it
        'Based on my analysis:\n{"is_duplicate": true, "confidence": 0.95}\nThis is my conclusion.',

        # Malformed JSON - missing quotes
        '{is_duplicate: true, confidence: 0.95}',

        # Array instead of object
        '[{"is_duplicate": true}, {"confidence": 0.95}]',

        # Empty or None
        '',
        'null',
        'None',

        # Truncated JSON
        '{"is_duplicate": true, "confidence": 0.95, "reasoning": "This intervention',

        # JSON with comments (invalid)
        '{\n  // This is a comment\n  "is_duplicate": true,\n  "confidence": 0.95\n}',

        # Unicode characters
        '{"is_duplicate": true, "confidence": 0.95, "reasoning": "These interventions are similar – both target GERD"}',
    ]

    print(f"Testing {len(test_cases)} JSON parsing scenarios...")

    failed_cases = []
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = parse_json_safely(test_case)
            if result is None:
                print(f"Test {i}: FAILED - returned None")
                failed_cases.append((i, test_case[:50] + "..." if len(test_case) > 50 else test_case, "returned None"))
            else:
                print(f"Test {i}: SUCCESS - parsed as {type(result)}")
        except Exception as e:
            print(f"Test {i}: FAILED - {e}")
            failed_cases.append((i, test_case[:50] + "..." if len(test_case) > 50 else test_case, str(e)))

    if failed_cases:
        print(f"\n=== FAILED JSON PARSING CASES ===")
        for case_num, case_text, error in failed_cases:
            print(f"Case {case_num}: '{case_text}'")
            print(f"  Error: {error}")
            print()

    return failed_cases

def main():
    """Comprehensive test to find and fix all JSON parsing issues."""
    db_path = "data/processed/intervention_research.db"

    if not Path(db_path).exists():
        print(f"[ERROR] Database not found: {db_path}")
        return

    print("=== COMPREHENSIVE JSON PARSING ERROR TEST ===")
    print(f"Starting at {datetime.now()}")
    print("Goal: Find and fix ALL JSON parsing issues before full migration")
    print()

    # Step 1: Test JSON repair functions directly
    print("STEP 1: Testing JSON repair functions...")
    json_repair_failures = test_json_repair_directly()

    if json_repair_failures:
        print(f"[CRITICAL] Found {len(json_repair_failures)} JSON repair failures!")
        print("Must fix JSON repair functions before proceeding.")
        return

    # Step 2: Analyze potential failure patterns
    analyze_json_parsing_failures()

    # Step 3: Get test interventions
    print("STEP 2: Loading interventions for stress testing...")
    interventions = get_test_interventions_for_json_errors(db_path, limit=10)
    print(f"Found {len(interventions)} interventions for testing")

    if len(interventions) < 2:
        print("[ERROR] Need at least 2 interventions for comparison testing")
        return

    # Show what we're testing
    print("\nTest interventions:")
    for i, intervention in enumerate(interventions, 1):
        print(f"  {i}. '{intervention['intervention_name']}' ({intervention['health_condition']})")
    print()

    # Step 4: Initialize semantic merger
    print("STEP 3: Initializing LLM semantic merger...")
    try:
        merger = SemanticMerger(
            primary_model='qwen2.5:14b',
            validator_model='gemma2:9b'
        )
        print("[SUCCESS] SemanticMerger initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize SemanticMerger: {e}")
        return

    # Step 5: Stress test LLM calls
    print("STEP 4: Stress testing LLM comparisons...")
    json_errors = stress_test_llm_comparisons(interventions, merger)

    # Step 6: Final assessment
    print(f"\n=== FINAL ASSESSMENT ===")

    if not json_errors:
        print("[SUCCESS] No JSON parsing errors found in stress test!")
        print("The system appears to handle JSON parsing robustly.")

        # Show LLM statistics
        print("\nLLM Statistics:")
        stats = merger.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n[READY] System is ready for full migration!")
    else:
        print(f"[FAILURE] Found {len(json_errors)} JSON parsing errors!")
        print("DO NOT proceed to full migration until these are fixed.")
        print("\nError summary:")
        for error_type in ['comparison_error', 'validation_error']:
            count = len([e for e in json_errors if e['type'] == error_type])
            if count > 0:
                print(f"  {error_type}: {count} errors")

    print(f"\nTest completed at {datetime.now()}")

if __name__ == "__main__":
    main()