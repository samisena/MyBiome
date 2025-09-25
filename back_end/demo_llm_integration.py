#!/usr/bin/env python3
"""
Demo of LLM-enhanced entity normalization integration
Shows how to use the complete system with real examples
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def demo_integration():
    """Demonstrate the complete LLM-enhanced entity normalization system"""

    print("=== LLM-Enhanced Entity Normalization Demo ===\n")

    # Connect to database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn, llm_model="gemma2:9b")

    # Sample terms to test (mix of easy and challenging cases)
    test_cases = [
        # Easy cases (should match with safe methods)
        ("probiotics", "intervention"),
        ("IBS", "condition"),
        ("probiotics", "intervention"),  # Test caching

        # LLM-required cases (need semantic understanding)
        ("acid reflux", "condition"),
        ("heartburn", "condition"),
        ("stomach acid medication", "intervention"),
        ("PPIs", "intervention"),
        ("anti-inflammatory drugs", "intervention"),

        # Should not match cases
        ("unknown_medical_term", "intervention"),
        ("completely_made_up_condition", "condition"),
    ]

    results = []

    for i, (term, entity_type) in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: '{term}' ({entity_type}) ---")

        # Test comprehensive matching workflow
        matches = normalizer.find_comprehensive_matches(term, entity_type, use_llm=True)

        if matches:
            best_match = matches[0]
            method = best_match['match_method']
            canonical_name = best_match['canonical_name']
            confidence = best_match.get('confidence', 0)
            cached = best_match.get('cached', False)

            print(f"[MATCH FOUND]")
            print(f"   Canonical: {canonical_name}")
            print(f"   Method: {method}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Cached: {cached}")

            if method == 'llm_semantic':
                reasoning = best_match.get('reasoning', 'No reasoning')
                print(f"   LLM Reasoning: {reasoning}")

            results.append({
                'term': term,
                'entity_type': entity_type,
                'matched': True,
                'canonical_name': canonical_name,
                'method': method,
                'confidence': confidence,
                'cached': cached
            })
        else:
            print(f"[NO MATCH]")
            results.append({
                'term': term,
                'entity_type': entity_type,
                'matched': False,
                'canonical_name': None,
                'method': 'no_match',
                'confidence': 0.0,
                'cached': False
            })

    # Show summary
    print(f"\n" + "="*60)
    print("INTEGRATION DEMO SUMMARY")
    print("="*60)

    # Method breakdown
    methods = {}
    for result in results:
        method = result['method']
        methods[method] = methods.get(method, 0) + 1

    print(f"\nMatching Methods Used:")
    for method, count in methods.items():
        method_name = {
            'existing_mapping': 'Existing Database Mapping',
            'exact_normalized': 'Exact Match (normalized)',
            'safe_pattern': 'Safe Pattern Matching',
            'llm_semantic': 'LLM Semantic Matching',
            'no_match': 'No Match Found'
        }.get(method, method)
        print(f"  {method_name}: {count}")

    # Success rate
    successful_matches = sum(1 for r in results if r['matched'])
    success_rate = (successful_matches / len(results)) * 100
    print(f"\nSuccess Rate: {successful_matches}/{len(results)} ({success_rate:.1f}%)")

    # Cache statistics
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

    # Integration workflow demonstration
    print(f"\n--- Integration Workflow Example ---")
    print("This system can be integrated into your data processing pipeline:")
    print()
    print("1. SAFE-FIRST APPROACH:")
    print("   - Check existing mappings (instant)")
    print("   - Try exact normalized matches (fast)")
    print("   - Use conservative pattern matching (fast)")
    print()
    print("2. LLM ENHANCEMENT:")
    print("   - For unmatched terms, use LLM semantic matching")
    print("   - Cache LLM decisions to avoid repeated API calls")
    print("   - Set confidence thresholds for auto vs manual review")
    print()
    print("3. MEDICAL SAFETY:")
    print("   - Conservative matching prevents dangerous false positives")
    print("   - LLM prompts emphasize medical accuracy")
    print("   - High confidence thresholds for automatic application")

    # Example code snippet
    print(f"\n--- Usage Example ---")
    print("""
# Example integration into existing code:
normalizer = EntityNormalizer(db_connection)

def process_intervention_data(intervention_name, health_condition):
    # Normalize both fields
    norm_intervention = normalizer.get_canonical_name(intervention_name, "intervention")
    norm_condition = normalizer.get_canonical_name(health_condition, "condition")

    # Use normalized names in your analysis
    return process_normalized_data(norm_intervention, norm_condition)
    """)

    conn.close()
    print(f"\n[SUCCESS] Integration demo completed")


if __name__ == "__main__":
    demo_integration()