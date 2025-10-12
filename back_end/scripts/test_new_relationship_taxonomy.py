"""
Test Script for New Layer-Based Relationship Taxonomy

Tests the new 5-type taxonomy with real intervention pairs and verifies
that the LLM correctly classifies relationships based on layer differences.
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add back_end directory to path
script_dir = Path(__file__).parent
back_end_dir = script_dir.parent
project_root = back_end_dir.parent
sys.path.insert(0, str(project_root))

from back_end.src.semantic_normalization.llm_classifier import LLMClassifier
from back_end.src.semantic_normalization.embedding_engine import EmbeddingEngine

def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def print_result(test_name, int1, int2, similarity, result):
    """Print test result in formatted way."""
    print(f"TEST: {test_name}")
    print(f"  Pair: \"{int1}\" ↔ \"{int2}\"")
    print(f"  Similarity: {similarity:.3f}")
    print(f"  Result:")
    print(f"    Relationship: {result['relationship_type']}")
    print(f"    Layer 1 Canonical: {result.get('layer_1_canonical', 'N/A')}")
    print(f"    Layer 2 Same Variant: {result.get('layer_2_same_variant', False)}")
    print(f"    Reasoning: {result.get('reasoning', 'N/A')}")
    print(f"    Source: {result.get('source', 'N/A')}")
    print()

def main():
    print_header("Testing New Layer-Based Relationship Taxonomy")

    print("Initializing components...")

    # Initialize embedding engine (for similarity calculation)
    embedding_engine = EmbeddingEngine(
        cache_path="c:/Users/samis/Desktop/MyBiome/back_end/data/cache/embeddings_test.pkl"
    )

    # Initialize LLM classifier
    classifier = LLMClassifier(
        model="qwen3:14b",
        temperature=0.0,
        canonical_cache_path="c:/Users/samis/Desktop/MyBiome/back_end/data/cache/canonicals_test.pkl",
        relationship_cache_path="c:/Users/samis/Desktop/MyBiome/back_end/data/cache/relationships_test.pkl"
    )

    print("✓ Components initialized\n")

    # Test cases for each relationship type
    test_cases = [
        # Type 1: EXACT_MATCH (All 4 layers identical)
        {
            'name': 'EXACT_MATCH - Spelling variant',
            'int1': 'Pegylated interferon alpha (Peg-IFNα)',
            'int2': 'pegylated interferon α (PEG-IFNα)',
            'expected': 'EXACT_MATCH'
        },

        # Type 2: DOSAGE_VARIANT (Layer 3 differs)
        {
            'name': 'DOSAGE_VARIANT - Different dosages',
            'int1': 'metformin',
            'int2': 'metformin 500mg',
            'expected': 'DOSAGE_VARIANT'
        },
        {
            'name': 'DOSAGE_VARIANT - Different strains',
            'int1': 'L. reuteri DSM 17938',
            'int2': 'L. reuteri ATCC 55730',
            'expected': 'DOSAGE_VARIANT'
        },

        # Type 3: SAME_CATEGORY_TYPE_VARIANT (Layer 2 differs)
        {
            'name': 'SAME_CATEGORY_TYPE_VARIANT - Vitamin D vs D3',
            'int1': 'vitamin D',
            'int2': 'Vitamin D3',
            'expected': 'SAME_CATEGORY_TYPE_VARIANT'
        },
        {
            'name': 'SAME_CATEGORY_TYPE_VARIANT - Different probiotic species',
            'int1': 'Lactobacillus reuteri',
            'int2': 'Saccharomyces boulardii',
            'expected': 'SAME_CATEGORY_TYPE_VARIANT'
        },
        {
            'name': 'SAME_CATEGORY_TYPE_VARIANT - Biosimilar',
            'int1': 'Cetuximab',
            'int2': 'Cetuximab-β',
            'expected': 'SAME_CATEGORY_TYPE_VARIANT'
        },
        {
            'name': 'SAME_CATEGORY_TYPE_VARIANT - Different statins',
            'int1': 'atorvastatin',
            'int2': 'simvastatin',
            'expected': 'SAME_CATEGORY_TYPE_VARIANT'
        },
        {
            'name': 'SAME_CATEGORY_TYPE_VARIANT - IBS subtypes',
            'int1': 'IBS-D',
            'int2': 'IBS-C',
            'expected': 'SAME_CATEGORY_TYPE_VARIANT'
        },

        # Type 4: SAME_CATEGORY (Layer 1 differs)
        {
            'name': 'SAME_CATEGORY - Different supplement types',
            'int1': 'probiotics',
            'int2': 'magnesium',
            'expected': 'SAME_CATEGORY'
        },
        {
            'name': 'SAME_CATEGORY - Different supplements',
            'int1': 'vitamin D',
            'int2': 'omega-3',
            'expected': 'SAME_CATEGORY'
        },

        # Type 5: DIFFERENT (Layer 0 differs)
        {
            'name': 'DIFFERENT - Completely unrelated',
            'int1': 'vitamin D',
            'int2': 'chemotherapy',
            'expected': 'DIFFERENT'
        },
        {
            'name': 'DIFFERENT - Different categories',
            'int1': 'probiotics',
            'int2': 'cognitive behavioral therapy',
            'expected': 'DIFFERENT'
        }
    ]

    # Run tests
    print_header("Running Test Cases")

    results = []
    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] ", end="")

        try:
            # Calculate similarity
            emb1 = embedding_engine.generate_embedding(test['int1'])
            emb2 = embedding_engine.generate_embedding(test['int2'])
            similarity = embedding_engine.cosine_similarity(emb1, emb2)

            # Classify relationship
            result = classifier.classify_relationship(
                test['int1'],
                test['int2'],
                similarity
            )

            # Print result
            print_result(
                test['name'],
                test['int1'],
                test['int2'],
                similarity,
                result
            )

            # Check if matches expected
            actual = result['relationship_type']
            expected = test['expected']

            if actual == expected:
                print(f"    ✓ PASS - Got expected {expected}")
                passed += 1
            else:
                print(f"    ✗ FAIL - Expected {expected}, got {actual}")
                failed += 1

            results.append({
                'test': test['name'],
                'expected': expected,
                'actual': actual,
                'passed': actual == expected,
                'similarity': similarity
            })

        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            failed += 1
            results.append({
                'test': test['name'],
                'expected': test['expected'],
                'actual': 'ERROR',
                'passed': False,
                'error': str(e)
            })

        print("-" * 80)

    # Summary
    print_header("Test Summary")

    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed} ({passed/len(test_cases)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(test_cases)*100:.1f}%)")
    print()

    if failed > 0:
        print("Failed Tests:")
        for result in results:
            if not result['passed']:
                print(f"  ✗ {result['test']}")
                print(f"    Expected: {result['expected']}")
                print(f"    Actual: {result['actual']}")
                if 'similarity' in result:
                    print(f"    Similarity: {result['similarity']:.3f}")
                print()

    # Print classifier stats
    print_header("Classifier Statistics")
    stats = classifier.get_stats()
    print(f"Canonical Cache:")
    print(f"  Size: {stats['canonical_cache_size']}")
    print(f"  Hits: {stats['canonical_cache_hits']}")
    print(f"  Misses: {stats['canonical_cache_misses']}")
    print(f"  Hit Rate: {stats['canonical_hit_rate']*100:.1f}%")
    print()
    print(f"Relationship Cache:")
    print(f"  Size: {stats['relationship_cache_size']}")
    print(f"  Hits: {stats['relationship_cache_hits']}")
    print(f"  Misses: {stats['relationship_cache_misses']}")
    print(f"  Hit Rate: {stats['relationship_hit_rate']*100:.1f}%")

    print("\n" + "="*80)
    print(f"  {'✓ ALL TESTS PASSED!' if failed == 0 else '✗ SOME TESTS FAILED'}")
    print("="*80 + "\n")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
