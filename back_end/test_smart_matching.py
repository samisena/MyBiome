#!/usr/bin/env python3
"""
Test the smart matching functionality of EntityNormalizer
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def test_smart_matching():
    """Test all smart matching methods with seed data"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    print("=== Testing Smart Matching Methods ===\n")

    # Test 1: normalize_term method
    print("--- Test 1: normalize_term ---")
    test_terms = [
        "Probiotics (with supplement)",
        "  IBS  ",
        "Low-FODMAP diet!",
        "Migraine headache???",
        "Type 2 Diabetes Mellitus (T2DM)"
    ]

    for term in test_terms:
        normalized = normalizer.normalize_term(term)
        print(f"'{term}' -> '{normalized}'")

    # Test 2: find_by_exact_match method
    print("\n--- Test 2: find_by_exact_match ---")
    exact_test_terms = [
        ("probiotics", "intervention"),
        ("Probiotics", "intervention"),      # Should match due to normalization
        ("ibs", "condition"),               # Should match "IBS"
        ("Ibs", "condition"),               # Should match "IBS"
        ("irritable bowel syndrome", "condition"),
        ("unknown_term", "intervention")    # Should not match
    ]

    for term, entity_type in exact_test_terms:
        match = normalizer.find_by_exact_match(term, entity_type)
        if match:
            print(f"[MATCH] '{term}' ({entity_type}) -> {match['canonical_name']} (ID: {match['id']})")
        else:
            print(f"[NO MATCH] '{term}' ({entity_type})")

    # Test 3: find_by_pattern method
    print("\n--- Test 3: find_by_pattern ---")
    pattern_test_terms = [
        ("probiotic", "intervention"),       # Should match "probiotics" (singular->plural)
        ("probioticz", "intervention"),      # Should NOT match
        ("migraines", "condition"),          # Should match "migraine" (plural->singular)
        ("the probiotics", "intervention"), # Should match via prefix removal
        ("placebo treatment", "intervention") # Should match via suffix removal
    ]

    for term, entity_type in pattern_test_terms:
        matches = normalizer.find_by_pattern(term, entity_type)
        if matches:
            for match in matches[:1]:  # Show top match only
                method = match.get('match_method', 'unknown')
                confidence = match.get('confidence_score', 0.0)
                print(f"[PATTERN] '{term}' -> {match['canonical_name']} (method: {method}, confidence: {confidence:.2f})")
        else:
            print(f"[NO PATTERN] '{term}' ({entity_type})")

    # Test 4: calculate_string_similarity method
    print("\n--- Test 4: calculate_string_similarity ---")
    similarity_pairs = [
        ("probiotics", "probiotic"),
        ("IBS", "ibs"),
        ("migraine", "migraines"),
        ("placebo", "placebos"),
        ("probiotics", "antibiotics"),  # Should be lower similarity
        ("hello", "world")              # Should be very low similarity
    ]

    for term1, term2 in similarity_pairs:
        similarity = normalizer.calculate_string_similarity(term1, term2)
        print(f"'{term1}' <-> '{term2}': {similarity:.3f}")

    # Test 5: find_by_similarity method
    print("\n--- Test 5: find_by_similarity ---")
    similarity_test_terms = [
        ("probiotic", "intervention", 0.8),   # Should match "probiotics"
        ("Ibs", "condition", 0.8),            # Should match "irritable bowel syndrome"
        ("migrain", "condition", 0.8),        # Should match "migraine" (typo)
        ("plcebo", "intervention", 0.7),      # Should match "placebo" (typo)
        ("xyz123", "intervention", 0.8)       # Should not match anything
    ]

    for term, entity_type, threshold in similarity_test_terms:
        matches = normalizer.find_by_similarity(term, entity_type, threshold)
        if matches:
            for match in matches[:1]:  # Show top match only
                similarity = match.get('similarity_score', 0.0)
                method = match.get('match_method', 'unknown')
                matched_text = match.get('matched_text', '')
                print(f"[SIMILARITY] '{term}' -> {match['canonical_name']} (similarity: {similarity:.3f}, matched: '{matched_text}')")
        else:
            print(f"[NO SIMILARITY] '{term}' ({entity_type}) above {threshold}")

    # Test 6: Combined smart matching workflow
    print("\n--- Test 6: Combined Smart Matching Workflow ---")
    def smart_find_canonical(normalizer, term, entity_type):
        """Demonstrate a complete smart matching workflow"""

        # Step 1: Try existing mapping first (fastest)
        existing_id = normalizer.find_canonical_id(term, entity_type)
        if existing_id:
            canonical_name = normalizer.get_canonical_name(term, entity_type)
            return f"EXISTING: {canonical_name} (ID: {existing_id})"

        # Step 2: Try exact normalized match
        exact_match = normalizer.find_by_exact_match(term, entity_type)
        if exact_match:
            return f"EXACT: {exact_match['canonical_name']} (ID: {exact_match['id']})"

        # Step 3: Try pattern matching
        pattern_matches = normalizer.find_by_pattern(term, entity_type)
        if pattern_matches:
            best_pattern = pattern_matches[0]
            return f"PATTERN: {best_pattern['canonical_name']} (method: {best_pattern['match_method']}, conf: {best_pattern['confidence_score']:.2f})"

        # Step 4: Try similarity matching
        similarity_matches = normalizer.find_by_similarity(term, entity_type, 0.8)
        if similarity_matches:
            best_similarity = similarity_matches[0]
            return f"SIMILARITY: {best_similarity['canonical_name']} (sim: {best_similarity['similarity_score']:.3f})"

        return "NO MATCH FOUND"

    workflow_test_terms = [
        ("probiotics", "intervention"),        # Should find via existing
        ("probiotic", "intervention"),         # Should find via pattern
        ("probiotik", "intervention"),         # Should find via similarity
        ("IBS", "condition"),                  # Should find via existing
        ("ibs", "condition"),                  # Should find via exact
        ("migraines", "condition"),            # Should find via pattern
        ("migrain", "condition"),              # Should find via similarity
        ("random_term", "intervention")        # Should not find
    ]

    for term, entity_type in workflow_test_terms:
        result = smart_find_canonical(normalizer, term, entity_type)
        print(f"'{term}' ({entity_type}): {result}")

    # Test performance on real database terms
    print("\n--- Test 7: Performance on Real Database Terms ---")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT intervention_name
        FROM interventions
        WHERE intervention_name NOT IN (
            SELECT raw_text FROM entity_mappings WHERE entity_type = 'intervention'
        )
        LIMIT 10
    """)

    unmapped_interventions = [row[0] for row in cursor.fetchall()]

    print("Testing smart matching on unmapped interventions from database:")
    for intervention in unmapped_interventions:
        result = smart_find_canonical(normalizer, intervention, "intervention")
        print(f"  '{intervention}': {result}")

    conn.close()
    print(f"\n[SUCCESS] Smart matching tests completed!")


if __name__ == "__main__":
    test_smart_matching()