#!/usr/bin/env python3
"""
Example usage of EntityNormalizer class

This demonstrates how to use the EntityNormalizer alongside existing code
without interfering with current operations.
"""

import sqlite3
from entity_normalizer import EntityNormalizer


def example_usage():
    """Example of how to use EntityNormalizer"""

    # Connect to your existing database
    conn = sqlite3.connect("../data/processed/intervention_research.db")

    # Create normalizer instance
    normalizer = EntityNormalizer(conn)

    # Example 1: Check if term is already normalized
    canonical_id = normalizer.find_canonical_id("probiotics", "intervention")
    if canonical_id:
        print(f"Term 'probiotics' is already mapped to canonical ID: {canonical_id}")
    else:
        print("Term 'probiotics' is not yet mapped")

    # Example 2: Create canonical entities and map variants
    try:
        # Create canonical entity for probiotics
        probiotic_id = normalizer.create_canonical_entity(
            canonical_name="probiotics",
            entity_type="intervention"
        )
        print(f"Created canonical entity 'probiotics' with ID: {probiotic_id}")

        # Map common variants
        variants = [
            ("Probiotics", 1.0, "exact_match"),
            ("probiotic", 0.95, "singular_form"),
            ("probiotic supplements", 0.9, "variant_form")
        ]

        for variant, confidence, method in variants:
            mapping_id = normalizer.add_term_mapping(variant, probiotic_id, confidence, method)
            print(f"Mapped '{variant}' to probiotics (ID: {mapping_id})")

    except Exception as e:
        print(f"Error creating entities (may already exist): {e}")

    # Example 3: Use normalization in existing workflows
    test_terms = ["probiotics", "Probiotics", "probiotic", "unknown_intervention"]

    print("\n--- Normalization Results ---")
    for term in test_terms:
        canonical_name = normalizer.get_canonical_name(term, "intervention")
        print(f"'{term}' -> '{canonical_name}'")

    # Example 4: Integration with existing intervention processing
    def process_intervention_with_normalization(intervention_name, health_condition, normalizer):
        """Example of how to integrate normalization into existing code"""

        # Normalize the intervention name
        normalized_intervention = normalizer.get_canonical_name(intervention_name, "intervention")
        normalized_condition = normalizer.get_canonical_name(health_condition, "condition")

        print(f"Original: {intervention_name} -> {health_condition}")
        print(f"Normalized: {normalized_intervention} -> {normalized_condition}")

        # Your existing processing logic would go here
        # The rest of your code doesn't need to change!
        return normalized_intervention, normalized_condition

    # Example usage
    print("\n--- Integration Example ---")
    process_intervention_with_normalization("Probiotics", "IBS", normalizer)

    # Example 5: Get statistics
    stats = normalizer.get_mapping_stats()
    print(f"\n--- Current Stats ---")
    print(f"Canonical entities: {stats.get('canonical_entities', {})}")
    print(f"Mappings: {stats.get('mappings', {})}")

    conn.close()


if __name__ == "__main__":
    example_usage()