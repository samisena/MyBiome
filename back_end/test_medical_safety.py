#!/usr/bin/env python3
"""
Test the medical safety improvements to EntityNormalizer
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def test_medical_safety():
    """Test that dangerous medical term matching has been eliminated"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    print("=== MEDICAL SAFETY TEST ===\n")

    # Test 1: Ensure dangerous similarity matches are blocked
    print("--- Test 1: Dangerous Similarity Matches Should Be Blocked ---")

    dangerous_pairs = [
        ("prebiotics", "probiotics", "intervention"),      # Different substances
        ("hyperglycemia", "hypoglycemia", "condition"),    # Opposite conditions
        ("hypertension", "hypotension", "condition"),      # Opposite conditions
        ("tachycardia", "bradycardia", "condition"),       # Opposite heart rates
        ("diarrhea", "constipation", "condition"),         # Opposite symptoms
    ]

    for term1, term2, entity_type in dangerous_pairs:
        similarity = normalizer.calculate_string_similarity(term1, term2)
        print(f"'{term1}' vs '{term2}': similarity = {similarity:.3f} (should be 0.0 or very low)")

        # Test find_by_similarity with low threshold
        similarity_matches = normalizer.find_by_similarity(term1, entity_type, 0.8)
        if similarity_matches:
            print(f"  [WARNING] find_by_similarity found matches for '{term1}': {len(similarity_matches)}")
            for match in similarity_matches:
                if term2 in match['canonical_name'].lower():
                    print(f"    [DANGEROUS] Matched '{term1}' to '{match['canonical_name']}'")
        else:
            print(f"  [SAFE] No similarity matches found for '{term1}'")

    # Test 2: Ensure safe pattern matches still work
    print("\n--- Test 2: Safe Pattern Matches Should Still Work ---")

    safe_pairs = [
        ("probiotics", "probiotic", "intervention"),       # Safe pluralization
        ("IBS", "ibs", "condition"),                       # Case variation
        ("low-FODMAP diet", "low FODMAP diet", "intervention"),  # Spacing/punctuation
        ("the probiotics", "probiotics", "intervention"),  # Article removal
    ]

    for term1, term2, entity_type in safe_pairs:
        # Test pattern matching
        pattern_matches = normalizer.find_by_pattern(term1, entity_type)
        found_safe_match = False

        for match in pattern_matches:
            if term2 in match['canonical_name'].lower() or term2 in normalizer.normalize_term(match['canonical_name']):
                found_safe_match = True
                print(f"  [SAFE] '{term1}' correctly matched to '{match['canonical_name']}' via {match['match_method']}")
                break

        if not found_safe_match:
            # This might be okay if we don't have the canonical entity
            print(f"  [INFO] No pattern match for '{term1}' -> '{term2}' (may not exist in database)")

    # Test 3: Test the new safe matching method
    print("\n--- Test 3: Safe Matching Method ---")

    test_terms = [
        ("probiotics", "intervention"),
        ("probiotic", "intervention"),
        ("prebiotics", "intervention"),
        ("IBS", "condition"),
        ("ibs", "condition"),
        ("migraine", "condition"),
        ("hyperglycemia", "condition"),  # Should not match hypoglycemia
    ]

    for term, entity_type in test_terms:
        safe_matches = normalizer.find_safe_matches_only(term, entity_type)
        if safe_matches:
            for match in safe_matches:
                print(f"  '{term}' -> '{match['canonical_name']}' (method: {match['match_method']}, confidence: {match['confidence_score']:.2f})")
        else:
            print(f"  '{term}' -> No safe matches (good for new terms)")

    # Test 4: Verify specific dangerous case is blocked
    print("\n--- Test 4: Verify Prebiotics/Probiotics Issue is Fixed ---")

    # This was the original dangerous match
    prebiotics_matches = normalizer.find_by_similarity("prebiotics", "intervention", 0.8)
    dangerous_match_found = False

    for match in prebiotics_matches:
        if "probiotics" in match['canonical_name'].lower():
            dangerous_match_found = True
            if match.get('needs_llm_verification', False):
                print(f"  [FLAGGED] '{match['canonical_name']}' match flagged for LLM verification (good)")
            else:
                print(f"  [DANGEROUS] '{match['canonical_name']}' matched without verification flag")

    if not dangerous_match_found:
        print(f"  [SAFE] 'prebiotics' does not match 'probiotics' anymore [PASS]")

    # Test 5: Test similarity calculation improvements
    print("\n--- Test 5: String Similarity Safety Checks ---")

    similarity_tests = [
        ("probiotics", "probiotics"),    # Identical - should be 1.0
        ("probiotics", "Probiotics"),    # Case only - should be 1.0
        ("probiotics", "probiotic"),     # Safe pluralization - should be high
        ("probiotics", "prebiotics"),    # Different terms - should be 0.0
        ("low FODMAP", "low-FODMAP"),    # Spacing only - should be high
        ("hypertension", "hypotension"), # Dangerous pair - should be 0.0
    ]

    for term1, term2 in similarity_tests:
        similarity = normalizer.calculate_string_similarity(term1, term2)
        if "pre" in term1 and "pro" in term2 and similarity > 0.5:
            print(f"  [FAIL] '{term1}' vs '{term2}': {similarity:.3f} (too high for different terms)")
        elif "hyper" in term1 and "hypo" in term2 and similarity > 0.5:
            print(f"  [FAIL] '{term1}' vs '{term2}': {similarity:.3f} (too high for opposite terms)")
        else:
            print(f"  [SAFE] '{term1}' vs '{term2}': {similarity:.3f}")

    conn.close()
    print(f"\n[SUCCESS] Medical safety tests completed!")


if __name__ == "__main__":
    test_medical_safety()