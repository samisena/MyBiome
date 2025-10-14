"""
Quick Test: Verify New Relationship Taxonomy Prompt

Tests that the new prompt is correctly formatted and includes all 5 relationship types.
No LLM calls - just validates the prompt structure.
"""

import sys
from pathlib import Path

# Add back_end directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from back_end.src.phase_3_semantic_normalization.prompts import (
    RELATIONSHIP_CLASSIFICATION_PROMPT,
    RELATIONSHIP_CLASSIFICATION_SCHEMA,
    format_relationship_classification_prompt
)

def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def main():
    print_section("Quick Test: New Relationship Taxonomy")

    # Test 1: Check valid relationship types in schema
    print("Test 1: Validate Relationship Types in Schema")
    print("-" * 80)

    expected_types = [
        'EXACT_MATCH',
        'DOSAGE_VARIANT',
        'SAME_CATEGORY_TYPE_VARIANT',
        'SAME_CATEGORY',
        'DIFFERENT'
    ]

    actual_types = RELATIONSHIP_CLASSIFICATION_SCHEMA['valid_relationship_types']

    print(f"Expected types ({len(expected_types)}): {expected_types}")
    print(f"Actual types ({len(actual_types)}):   {actual_types}")

    if actual_types == expected_types:
        print("\n[PASS] Relationship types match!")
    else:
        print("\n[FAIL] Relationship types don't match!")
        print(f"  Missing: {set(expected_types) - set(actual_types)}")
        print(f"  Extra: {set(actual_types) - set(expected_types)}")
        return 1

    # Test 2: Check prompt contains all types
    print_section("Test 2: Validate Prompt Contains All Relationship Types")

    for rel_type in expected_types:
        if rel_type in RELATIONSHIP_CLASSIFICATION_PROMPT:
            print(f"[PASS] {rel_type:30s} found in prompt")
        else:
            print(f"[FAIL] {rel_type:30s} NOT found in prompt")
            return 1

    # Test 3: Check layer-based descriptions
    print_section("Test 3: Validate Layer-Based Descriptions")

    layer_keywords = [
        "Layer 0",
        "Layer 1",
        "Layer 2",
        "Layer 3",
        "4-layer hierarchy",
        "which layer differs"
    ]

    found_keywords = []
    missing_keywords = []

    for keyword in layer_keywords:
        if keyword in RELATIONSHIP_CLASSIFICATION_PROMPT:
            found_keywords.append(keyword)
            print(f"[PASS] '{keyword}' found in prompt")
        else:
            missing_keywords.append(keyword)
            print(f"[FAIL] '{keyword}' NOT found in prompt")

    if missing_keywords:
        print(f"\nMissing keywords: {missing_keywords}")
        return 1

    # Test 4: Check examples are present
    print_section("Test 4: Validate Examples in Prompt")

    example_pairs = [
        ("vitamin D", "Vitamin D3"),  # SAME_CATEGORY_TYPE_VARIANT
        ("metformin", "metformin 500mg"),  # DOSAGE_VARIANT
        ("probiotics", "magnesium"),  # SAME_CATEGORY
        ("L. reuteri DSM 17938", "L. reuteri ATCC 55730"),  # DOSAGE_VARIANT
    ]

    for entity1, entity2 in example_pairs:
        if entity1 in RELATIONSHIP_CLASSIFICATION_PROMPT and entity2 in RELATIONSHIP_CLASSIFICATION_PROMPT:
            print(f"[PASS] Example: '{entity1}' <-> '{entity2}'")
        else:
            print(f"[FAIL] Example missing: '{entity1}' <-> '{entity2}'")
            return 1

    # Test 5: Format a test prompt
    print_section("Test 5: Test Prompt Formatting")

    test_prompt = format_relationship_classification_prompt(
        "vitamin D",
        "Vitamin D3",
        0.78
    )

    if "vitamin D" in test_prompt and "Vitamin D3" in test_prompt and "0.780" in test_prompt:
        print("[PASS] Prompt formatting works correctly")
        print(f"\nSample formatted prompt (first 500 chars):")
        print("-" * 80)
        print(test_prompt[:500] + "...")
    else:
        print("[FAIL] Prompt formatting failed")
        return 1

    # Test 6: Check old types are removed
    print_section("Test 6: Verify Old Relationship Types Are Removed")

    old_types = ['VARIANT', 'SUBTYPE']

    for old_type in old_types:
        # Check if old type is in schema (should NOT be)
        if old_type in RELATIONSHIP_CLASSIFICATION_SCHEMA['valid_relationship_types']:
            print(f"[FAIL] Old type '{old_type}' still in schema!")
            return 1
        else:
            print(f"[PASS] Old type '{old_type}' removed from schema")

    # Final summary
    print_section("Test Summary")
    print("[SUCCESS] All tests passed!")
    print("\nNew 5-type layer-based taxonomy is correctly implemented:")
    print("  1. EXACT_MATCH")
    print("  2. DOSAGE_VARIANT")
    print("  3. SAME_CATEGORY_TYPE_VARIANT")
    print("  4. SAME_CATEGORY")
    print("  5. DIFFERENT")
    print("\nOld types removed:")
    print("  - VARIANT (merged into SAME_CATEGORY_TYPE_VARIANT)")
    print("  - SUBTYPE (merged into SAME_CATEGORY_TYPE_VARIANT)")
    print("\n" + "=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
