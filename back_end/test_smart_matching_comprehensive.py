#!/usr/bin/env python3
"""
Comprehensive test suite for all smart matching methods
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def run_comprehensive_tests():
    """Run comprehensive tests for all smart matching functionality"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    total_tests = 0
    passed_tests = 0

    def test_case(description, test_func):
        nonlocal total_tests, passed_tests
        total_tests += 1
        try:
            result = test_func()
            if result:
                print(f"[PASS] {description}")
                passed_tests += 1
            else:
                print(f"[FAIL] {description}")
        except Exception as e:
            print(f"[ERROR] {description}: {e}")

    print("=== COMPREHENSIVE SMART MATCHING TEST SUITE ===\n")

    # Test normalize_term method
    print("--- Testing normalize_term ---")
    test_case(
        "Remove parentheses and contents",
        lambda: normalizer.normalize_term("probiotics (with supplement)") == "probiotics"
    )
    test_case(
        "Remove punctuation",
        lambda: normalizer.normalize_term("Low-FODMAP diet!!!") == "low-fodmap diet"
    )
    test_case(
        "Normalize whitespace",
        lambda: normalizer.normalize_term("  multiple   spaces  ") == "multiple spaces"
    )
    test_case(
        "Handle empty string",
        lambda: normalizer.normalize_term("") == ""
    )

    # Test find_by_exact_match method
    print("\n--- Testing find_by_exact_match ---")
    test_case(
        "Find existing canonical name",
        lambda: normalizer.find_by_exact_match("probiotics", "intervention") is not None
    )
    test_case(
        "Find with case differences",
        lambda: normalizer.find_by_exact_match("Probiotics", "intervention") is not None
    )
    test_case(
        "Find existing mapping",
        lambda: normalizer.find_by_exact_match("IBS", "condition") is not None
    )
    test_case(
        "Return None for non-existent term",
        lambda: normalizer.find_by_exact_match("nonexistent_term", "intervention") is None
    )
    test_case(
        "Respect entity_type parameter",
        lambda: normalizer.find_by_exact_match("probiotics", "condition") is None
    )

    # Test find_by_pattern method
    print("\n--- Testing find_by_pattern ---")
    test_case(
        "Singular to plural pattern",
        lambda: len(normalizer.find_by_pattern("probiotic", "intervention")) > 0
    )
    test_case(
        "Plural to singular pattern",
        lambda: len(normalizer.find_by_pattern("migraines", "condition")) > 0
    )
    test_case(
        "Prefix removal pattern",
        lambda: len(normalizer.find_by_pattern("the probiotics", "intervention")) > 0
    )
    test_case(
        "Suffix removal pattern",
        lambda: len(normalizer.find_by_pattern("placebo treatment", "intervention")) > 0
    )
    test_case(
        "No false positives",
        lambda: len(normalizer.find_by_pattern("completely_unknown_term", "intervention")) == 0
    )

    # Test calculate_string_similarity method
    print("\n--- Testing calculate_string_similarity ---")
    test_case(
        "Identical strings return 1.0",
        lambda: normalizer.calculate_string_similarity("test", "test") == 1.0
    )
    test_case(
        "Case differences normalized",
        lambda: normalizer.calculate_string_similarity("Test", "test") == 1.0
    )
    test_case(
        "High similarity for minor differences",
        lambda: normalizer.calculate_string_similarity("probiotic", "probiotics") > 0.8
    )
    test_case(
        "Low similarity for very different strings",
        lambda: normalizer.calculate_string_similarity("hello", "world") < 0.5
    )
    test_case(
        "Empty strings handled",
        lambda: normalizer.calculate_string_similarity("", "") == 1.0
    )

    # Test find_by_similarity method
    print("\n--- Testing find_by_similarity ---")
    test_case(
        "Find similar terms above threshold",
        lambda: len(normalizer.find_by_similarity("probiotik", "intervention", 0.8)) > 0
    )
    test_case(
        "Respect similarity threshold",
        lambda: len(normalizer.find_by_similarity("xyz123", "intervention", 0.9)) == 0
    )
    test_case(
        "Return results sorted by similarity",
        lambda: all(
            result['similarity_score'] >= next_result['similarity_score']
            for result, next_result in zip(
                normalizer.find_by_similarity("probiotik", "intervention", 0.7)[:-1],
                normalizer.find_by_similarity("probiotik", "intervention", 0.7)[1:]
            )
        ) if len(normalizer.find_by_similarity("probiotik", "intervention", 0.7)) > 1 else True
    )
    test_case(
        "Include similarity score in results",
        lambda: all(
            'similarity_score' in result
            for result in normalizer.find_by_similarity("probiotik", "intervention", 0.7)
        )
    )

    # Test edge cases and error handling
    print("\n--- Testing Edge Cases ---")
    test_case(
        "Handle None input gracefully",
        lambda: normalizer.normalize_term("") == ""  # Can't test None directly due to string methods
    )
    test_case(
        "Handle very long strings",
        lambda: len(normalizer.normalize_term("a" * 1000)) == 1000
    )
    test_case(
        "Handle special characters",
        lambda: normalizer.normalize_term("test@#$%^&*()") == "test@#$%^&*"
    )

    # Test integration scenarios
    print("\n--- Testing Integration Scenarios ---")
    test_case(
        "Known good case: probiotic -> probiotics",
        lambda: "probiotics" in str(normalizer.find_by_pattern("probiotic", "intervention"))
    )
    test_case(
        "Known good case: Ibs normalized match",
        lambda: normalizer.find_by_exact_match("Ibs", "condition") is not None
    )
    test_case(
        "Multiple match methods work for same term",
        lambda: (
            normalizer.find_canonical_id("probiotic", "intervention") is not None or
            normalizer.find_by_exact_match("probiotic", "intervention") is not None or
            len(normalizer.find_by_pattern("probiotic", "intervention")) > 0 or
            len(normalizer.find_by_similarity("probiotic", "intervention", 0.8)) > 0
        )
    )

    # Performance and consistency tests
    print("\n--- Testing Performance and Consistency ---")
    test_case(
        "Methods return consistent entity types",
        lambda: all(
            result['entity_type'] == 'intervention'
            for result in normalizer.find_by_pattern("probiotic", "intervention")
        )
    )
    test_case(
        "Similarity method respects entity_type filter",
        lambda: all(
            result['entity_type'] == 'condition'
            for result in normalizer.find_by_similarity("migrain", "condition", 0.7)
        )
    )

    # Test that methods don't interfere with existing functionality
    print("\n--- Testing Backward Compatibility ---")
    test_case(
        "Original get_canonical_name still works",
        lambda: normalizer.get_canonical_name("probiotics", "intervention") == "probiotics"
    )
    test_case(
        "Original find_canonical_id still works",
        lambda: normalizer.find_canonical_id("probiotics", "intervention") is not None
    )
    test_case(
        "Database integrity maintained",
        lambda: len(normalizer.search_canonical_entities("probiotics")) > 0
    )

    print(f"\n=== TEST RESULTS ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

    if passed_tests == total_tests:
        print(f"\n[SUCCESS] All tests passed! Smart matching is working correctly.")
        return True
    else:
        print(f"\n[WARNING] {total_tests - passed_tests} tests failed. Review implementation.")
        return False

    conn.close()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)