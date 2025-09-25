#!/usr/bin/env python3
"""
Test the LLM-enhanced entity matching capabilities
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def test_llm_entity_matching():
    """Test LLM entity matching with known synonyms"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    print("=== LLM ENTITY MATCHING TEST ===\n")

    # First, let's add some canonical entities to test against
    print("--- Setting Up Test Canonical Entities ---")

    # Add some medical entities that we can test synonyms for
    test_entities = [
        ("Gastroesophageal Reflux Disease", "condition"),
        ("Proton Pump Inhibitors", "intervention"),
        ("Inflammatory Bowel Disease", "condition"),
        ("Nonsteroidal Anti-Inflammatory Drugs", "intervention"),
    ]

    entity_ids = {}
    for canonical_name, entity_type in test_entities:
        try:
            entity_id = normalizer.create_canonical_entity(canonical_name, entity_type)
            entity_ids[canonical_name] = entity_id
            print(f"  Created: {canonical_name} (ID: {entity_id})")

            # Add the canonical name as its own mapping
            normalizer.add_term_mapping(canonical_name, entity_id, 1.0, "exact_canonical")
        except Exception as e:
            # Might already exist
            existing_id = normalizer.find_canonical_id(canonical_name, entity_type)
            if existing_id:
                entity_ids[canonical_name] = existing_id
                print(f"  Exists: {canonical_name} (ID: {existing_id})")
            else:
                print(f"  Error with {canonical_name}: {e}")

    print(f"\n--- Testing LLM Synonym Detection ---")

    # Test cases: known medical synonyms and non-synonyms
    test_cases = [
        # Should match (synonyms/common names)
        ("acid reflux", "condition", "Gastroesophageal Reflux Disease"),
        ("GERD", "condition", "Gastroesophageal Reflux Disease"),
        ("heartburn", "condition", "Gastroesophageal Reflux Disease"),
        ("stomach acid medication", "intervention", "Proton Pump Inhibitors"),
        ("PPIs", "intervention", "Proton Pump Inhibitors"),
        ("NSAIDs", "intervention", "Nonsteroidal Anti-Inflammatory Drugs"),
        ("anti-inflammatory drugs", "intervention", "Nonsteroidal Anti-Inflammatory Drugs"),
        ("IBD", "condition", "Inflammatory Bowel Disease"),

        # Should NOT match (different concepts)
        ("peptic ulcer", "condition", None),  # Different from GERD
        ("antacids", "intervention", None),   # Different from PPIs
        ("steroids", "intervention", None),   # Different from NSAIDs
        ("IBS", "condition", "irritable bowel syndrome"),  # IBS = Irritable Bowel Syndrome
    ]

    successful_matches = 0
    total_tests = len(test_cases)

    for term, entity_type, expected_match in test_cases:
        print(f"\nTesting: '{term}' ({entity_type})")
        print(f"Expected: {expected_match or 'No match'}")

        # Test LLM matching
        llm_result = normalizer.find_by_llm(term, entity_type)

        if llm_result:
            actual_match = llm_result['canonical_name']
            confidence = llm_result['confidence']
            reasoning = llm_result.get('reasoning', 'No reasoning')
            cached = llm_result.get('cached', False)

            print(f"LLM Result: '{actual_match}' (confidence: {confidence:.2f})")
            print(f"Reasoning: {reasoning}")
            print(f"Cached: {cached}")

            # Check if it matches expectation
            if expected_match and actual_match == expected_match:
                print("[PASS] CORRECT - Expected synonym detected")
                successful_matches += 1
            elif not expected_match and confidence < 0.5:
                print("[PASS] CORRECT - Non-synonym correctly rejected")
                successful_matches += 1
            else:
                if expected_match:
                    print(f"[FAIL] INCORRECT - Expected '{expected_match}', got '{actual_match}'")
                else:
                    print(f"[FAIL] INCORRECT - Should not match, but got '{actual_match}'")
        else:
            print("LLM Result: No match")
            if expected_match:
                print("[FAIL] INCORRECT - Expected synonym not detected")
            else:
                print("[PASS] CORRECT - Non-synonym correctly rejected")
                successful_matches += 1

    print(f"\n--- Testing Comprehensive Matching Workflow ---")

    # Test the comprehensive matching that combines safe + LLM
    comprehensive_test_cases = [
        ("probiotics", "intervention"),      # Should match via existing mapping (safe)
        ("acid reflux", "condition"),        # Should match via LLM (GERD)
        ("stomach acid medication", "intervention"),  # Should match via LLM (PPIs)
        ("random_unknown_term", "intervention"),      # Should not match
    ]

    for term, entity_type in comprehensive_test_cases:
        print(f"\nComprehensive matching for: '{term}' ({entity_type})")

        matches = normalizer.find_comprehensive_matches(term, entity_type, use_llm=True)
        if matches:
            best_match = matches[0]
            method = best_match['match_method']
            canonical_name = best_match['canonical_name']
            confidence = best_match.get('confidence', 0)

            print(f"  Best match: '{canonical_name}' via {method} (confidence: {confidence:.2f})")

            # Show all matches
            for i, match in enumerate(matches):
                if i > 0:  # Skip first one (already shown)
                    print(f"  Alt match {i+1}: '{match['canonical_name']}' via {match['match_method']}")
        else:
            print("  No matches found")

    print(f"\n--- Testing Batch Synonym Finding ---")

    batch_terms = ["acid reflux", "heartburn", "PPIs", "random_term"]
    batch_results = normalizer.batch_find_synonyms(batch_terms, "condition")

    print("Batch results:")
    for term, result in batch_results.items():
        if result:
            print(f"  '{term}' -> '{result['canonical_name']}' ({result['confidence']:.2f})")
        else:
            print(f"  '{term}' -> No match")

    # Test caching
    print(f"\n--- Testing LLM Caching ---")
    print("Running same query twice to test caching:")

    import time

    # First call
    start_time = time.time()
    result1 = normalizer.find_by_llm("acid reflux", "condition")
    time1 = time.time() - start_time

    # Second call (should be cached)
    start_time = time.time()
    result2 = normalizer.find_by_llm("acid reflux", "condition")
    time2 = time.time() - start_time

    print(f"First call: {time1:.3f}s - Cached: {result1.get('cached', False) if result1 else 'N/A'}")
    print(f"Second call: {time2:.3f}s - Cached: {result2.get('cached', False) if result2 else 'N/A'}")

    if result2 and result2.get('cached', False):
        print("[PASS] Caching is working!")
    else:
        print("[WARNING] Caching may not be working")

    # Show cache statistics
    print(f"\n--- LLM Cache Statistics ---")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM llm_normalization_cache")
    cache_count = cursor.fetchone()[0]
    print(f"Total cached LLM decisions: {cache_count}")

    if cache_count > 0:
        cursor.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM llm_normalization_cache
            GROUP BY entity_type
        """)
        for row in cursor.fetchall():
            print(f"  {row[0]}: {row[1]} cached decisions")

    print(f"\n=== TEST RESULTS ===")
    print(f"Successful matches: {successful_matches}/{total_tests}")
    print(f"Success rate: {(successful_matches/total_tests)*100:.1f}%")

    conn.close()

    if successful_matches >= total_tests * 0.7:  # 70% success rate threshold
        print(f"\n[SUCCESS] LLM entity matching is working well!")
        return True
    else:
        print(f"\n[WARNING] LLM entity matching needs improvement.")
        return False


if __name__ == "__main__":
    try:
        success = test_llm_entity_matching()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)