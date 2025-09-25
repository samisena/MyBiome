#!/usr/bin/env python3
"""
Test specific examples mentioned in the requirements:
- "probiotic" should match "probiotics" via pattern
- "Ibs" should match "IBS" via normalization
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def test_specific_examples():
    """Test the specific examples mentioned in the requirements"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    print("=== Testing Specific Examples ===\n")

    # Test 1: "probiotic" should match "probiotics" via pattern
    print("--- Test: 'probiotic' should match 'probiotics' via pattern ---")

    # First, verify "probiotics" exists in our mappings
    probiotics_id = normalizer.find_canonical_id("probiotics", "intervention")
    print(f"Canonical ID for 'probiotics': {probiotics_id}")

    # Test exact match first (should fail)
    exact_match = normalizer.find_by_exact_match("probiotic", "intervention")
    print(f"Exact match for 'probiotic': {exact_match}")

    # Test pattern match (should succeed)
    pattern_matches = normalizer.find_by_pattern("probiotic", "intervention")
    print(f"Pattern matches for 'probiotic':")
    for i, match in enumerate(pattern_matches):
        print(f"  {i+1}. {match['canonical_name']} (method: {match['match_method']}, confidence: {match['confidence_score']:.2f})")

    # Test if we already have "probiotic" mapped (we do!)
    probiotic_canonical = normalizer.get_canonical_name("probiotic", "intervention")
    print(f"get_canonical_name('probiotic'): {probiotic_canonical}")

    print("\n" + "="*60 + "\n")

    # Test 2: "Ibs" should match "IBS" via normalization
    print("--- Test: 'Ibs' should match 'IBS' via normalization ---")

    # First, verify "IBS" exists in our mappings
    ibs_id = normalizer.find_canonical_id("IBS", "condition")
    print(f"Canonical ID for 'IBS': {ibs_id}")

    # Test normalization
    normalized_ibs = normalizer.normalize_term("Ibs")
    normalized_IBS = normalizer.normalize_term("IBS")
    print(f"normalize_term('Ibs'): '{normalized_ibs}'")
    print(f"normalize_term('IBS'): '{normalized_IBS}'")
    print(f"Are they equal after normalization? {normalized_ibs == normalized_IBS}")

    # Test exact match (should succeed due to normalization)
    exact_match = normalizer.find_by_exact_match("Ibs", "condition")
    print(f"Exact match for 'Ibs': {exact_match}")
    if exact_match:
        print(f"  -> Matched canonical: {exact_match['canonical_name']} (ID: {exact_match['id']})")

    # Test get_canonical_name
    ibs_canonical = normalizer.get_canonical_name("Ibs", "condition")
    print(f"get_canonical_name('Ibs'): {ibs_canonical}")

    print("\n" + "="*60 + "\n")

    # Bonus: Test a complete smart matching workflow for both cases
    print("--- Complete Smart Matching Workflow ---")

    def complete_smart_match(normalizer, term, entity_type):
        """Complete workflow showing all matching attempts"""
        print(f"\nFinding matches for '{term}' ({entity_type}):")

        # 1. Check existing mapping
        existing_id = normalizer.find_canonical_id(term, entity_type)
        if existing_id:
            canonical = normalizer.get_canonical_name(term, entity_type)
            print(f"  [PASS] EXISTING MAPPING: {canonical} (ID: {existing_id})")
            return canonical
        else:
            print(f"  [FAIL] No existing mapping found")

        # 2. Try exact normalized match
        exact = normalizer.find_by_exact_match(term, entity_type)
        if exact:
            print(f"  [PASS] EXACT NORMALIZED: {exact['canonical_name']} (ID: {exact['id']})")
            return exact['canonical_name']
        else:
            print(f"  [FAIL] No exact normalized match")

        # 3. Try pattern matching
        patterns = normalizer.find_by_pattern(term, entity_type)
        if patterns:
            best = patterns[0]
            print(f"  [PASS] PATTERN MATCH: {best['canonical_name']} (method: {best['match_method']}, conf: {best['confidence_score']:.2f})")
            return best['canonical_name']
        else:
            print(f"  [FAIL] No pattern matches")

        # 4. Try similarity matching
        similarities = normalizer.find_by_similarity(term, entity_type, 0.8)
        if similarities:
            best = similarities[0]
            print(f"  [PASS] SIMILARITY MATCH: {best['canonical_name']} (sim: {best['similarity_score']:.3f})")
            return best['canonical_name']
        else:
            print(f"  [FAIL] No similarity matches above 0.8")

        return None

    # Test the workflow on our examples
    result1 = complete_smart_match(normalizer, "probiotic", "intervention")
    result2 = complete_smart_match(normalizer, "Ibs", "condition")

    print(f"\n=== RESULTS ===")
    print(f"'probiotic' (intervention) -> '{result1}' [SUCCESS]" if result1 else "'probiotic' -> NO MATCH [FAIL]")
    print(f"'Ibs' (condition) -> '{result2}' [SUCCESS]" if result2 else "'Ibs' -> NO MATCH [FAIL]")

    conn.close()
    print(f"\n[SUCCESS] Specific examples test completed!")


if __name__ == "__main__":
    test_specific_examples()