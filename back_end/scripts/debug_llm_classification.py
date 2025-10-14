"""
Debug LLM Classification Phase - Threshold Testing

Tests ONLY the LLM classification step with detailed tracking:
- Outputs after each LLM call
- Tracks inference time per call
- Shows relationship classifications
- No background execution
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from back_end.src.phase_3_semantic_normalization.phase_3_llm_classifier import LLMClassifier

# ==============================================================================
# TEST DATA
# ==============================================================================

# Test pairs with varying similarity scores
TEST_PAIRS = [
    # High similarity - should be EXACT_MATCH or VARIANT
    ("vitamin D", "Vitamin D3", 0.92),
    ("probiotic", "probiotics", 0.95),

    # Medium similarity - should be VARIANT or SUBTYPE
    ("aerobic exercise", "cardio workout", 0.78),
    ("metformin", "metformin ER", 0.85),

    # Lower similarity - could be SUBTYPE or SAME_CATEGORY
    ("meditation", "mindfulness training", 0.72),
    ("omega-3", "fish oil", 0.81),

    # Edge case - around threshold
    ("yoga", "stretching exercises", 0.70),
    ("statins", "atorvastatin", 0.75),
]


# ==============================================================================
# DEBUG LLM CLASSIFICATION
# ==============================================================================

def debug_llm_classification():
    """Test LLM classification with detailed output tracking."""

    print("="*80)
    print("DEBUG: LLM CLASSIFICATION PHASE")
    print("="*80)
    print(f"\nTest configuration:")
    print(f"  Model: qwen3:14b")
    print(f"  Timeout: None (no timeout)")
    print(f"  Test pairs: {len(TEST_PAIRS)}")
    print(f"  Output: Real-time with flush")

    # Initialize LLM classifier
    print(f"\n[INIT] Initializing LLM classifier...")
    start_init = time.time()
    llm_classifier = LLMClassifier(
        model="qwen3:14b",
        base_url="http://localhost:11434",
        temperature=0.0,
        timeout=None,  # No timeout (as configured)
        max_retries=3,
        strip_think_tags=True
    )
    init_time = time.time() - start_init
    print(f"[INIT] OK Initialized in {init_time:.2f}s", flush=True)

    # Test each pair
    results = []
    total_inference_time = 0

    for i, (entity1, entity2, similarity) in enumerate(TEST_PAIRS, 1):
        print(f"\n{'='*80}")
        print(f"TEST PAIR {i}/{len(TEST_PAIRS)}")
        print(f"{'='*80}")
        print(f"Entity 1: {entity1}")
        print(f"Entity 2: {entity2}")
        print(f"Similarity: {similarity:.3f}")
        print(f"\n[LLM] Calling classify_relationship...", flush=True)

        # Time the LLM call
        start_time = time.time()

        try:
            result = llm_classifier.classify_relationship(
                entity1,
                entity2,
                similarity
            )

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # Display result immediately
            print(f"[LLM] OK Response received in {inference_time:.2f}s", flush=True)
            print(f"\nRESULT:")
            print(f"  Relationship Type: {result['relationship_type']}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')[:100]}...")

            # Store result
            results.append({
                'pair': (entity1, entity2),
                'similarity': similarity,
                'relationship': result['relationship_type'],
                'reasoning': result.get('reasoning', ''),
                'inference_time': inference_time
            })

        except Exception as e:
            inference_time = time.time() - start_time
            print(f"[LLM] ERROR after {inference_time:.2f}s: {str(e)}", flush=True)
            results.append({
                'pair': (entity1, entity2),
                'similarity': similarity,
                'relationship': 'ERROR',
                'reasoning': str(e),
                'inference_time': inference_time
            })

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"\nTotal pairs tested: {len(TEST_PAIRS)}")
    print(f"Successful classifications: {sum(1 for r in results if r['relationship'] != 'ERROR')}")
    print(f"Errors: {sum(1 for r in results if r['relationship'] == 'ERROR')}")
    print(f"\nTiming:")
    print(f"  Total inference time: {total_inference_time:.2f}s")
    print(f"  Average per call: {total_inference_time / len(TEST_PAIRS):.2f}s")
    print(f"  Min: {min(r['inference_time'] for r in results):.2f}s")
    print(f"  Max: {max(r['inference_time'] for r in results):.2f}s")

    # Relationship distribution
    print(f"\nRelationship Type Distribution:")
    from collections import Counter
    rel_counts = Counter(r['relationship'] for r in results)
    for rel_type, count in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(TEST_PAIRS) * 100)
        print(f"  {rel_type:<20} {count:>2} ({percentage:>5.1f}%)")

    # Detailed results
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    for i, result in enumerate(results, 1):
        entity1, entity2 = result['pair']
        print(f"\n{i}. {entity1} ↔ {entity2}")
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Relationship: {result['relationship']}")
        print(f"   Inference Time: {result['inference_time']:.2f}s")
        print(f"   Reasoning: {result['reasoning'][:150]}...")

    # Check for hanging/slow calls
    print(f"\n{'='*80}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*80}")

    slow_calls = [r for r in results if r['inference_time'] > 30]
    if slow_calls:
        print(f"\nWARNING: {len(slow_calls)} calls took >30s:")
        for r in slow_calls:
            entity1, entity2 = r['pair']
            print(f"  - {entity1} ↔ {entity2}: {r['inference_time']:.2f}s")
    else:
        print(f"\nOK All calls completed in <30s")

    errors = [r for r in results if r['relationship'] == 'ERROR']
    if errors:
        print(f"\nERRORS DETECTED: {len(errors)} failed calls:")
        for r in errors:
            entity1, entity2 = r['pair']
            print(f"  - {entity1} ↔ {entity2}: {r['reasoning']}")
    else:
        print(f"\nOK No errors detected")

    print(f"\n{'='*80}")
    print("DEBUG TEST COMPLETE")
    print(f"{'='*80}")

    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\nStarting LLM classification debug test...")
    print("This will test 8 intervention pairs with real-time output.\n")

    try:
        results = debug_llm_classification()
        print("\nOK Test completed successfully")

    except KeyboardInterrupt:
        print("\n\nX Test interrupted by user (Ctrl+C)")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nX Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
