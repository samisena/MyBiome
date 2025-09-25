#!/usr/bin/env python3
"""
Test the normalization system with real data from the database
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def test_live_normalization():
    """Test normalization against actual terms found in the database"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    print("=== Testing Live Normalization ===\n")

    # Test intervention names that we know exist in the database
    test_interventions = [
        "probiotics",  # Should normalize
        "Probiotics",  # Should normalize
        "placebo",     # Should normalize
        "Placebo",     # Should normalize
        "low FODMAP diet",  # Should normalize
        "FODMAP diet",      # Should normalize
        "rifaximin",   # Not in our mappings - should return as-is
        "antibiotics", # Not in our mappings - should return as-is
    ]

    print("--- Intervention Normalization Test ---")
    for term in test_interventions:
        normalized = normalizer.get_canonical_name(term, "intervention")
        canonical_id = normalizer.find_canonical_id(term, "intervention")
        status = "NORMALIZED" if canonical_id else "UNCHANGED"
        print(f"{status:>10} | '{term}' -> '{normalized}' (ID: {canonical_id})")

    # Test condition names that we know exist in the database
    test_conditions = [
        "irritable bowel syndrome (IBS)",  # Should normalize
        "IBS",                            # Should normalize
        "Irritable Bowel Syndrome (IBS)", # Should normalize
        "SIBO",                           # Should normalize
        "small intestinal bacterial overgrowth (SIBO)",  # Should normalize
        "type 2 diabetes",               # Should normalize
        "migraine headache",              # Should normalize
        "rheumatoid arthritis",           # Not in our mappings - should return as-is
    ]

    print("\n--- Condition Normalization Test ---")
    for term in test_conditions:
        normalized = normalizer.get_canonical_name(term, "condition")
        canonical_id = normalizer.find_canonical_id(term, "condition")
        status = "NORMALIZED" if canonical_id else "UNCHANGED"
        print(f"{status:>10} | '{term}' -> '{normalized}' (ID: {canonical_id})")

    # Show how this would work in practice with real database queries
    print("\n--- Integration with Real Data ---")
    cursor = conn.cursor()

    # Get some real intervention entries from the database
    cursor.execute("""
        SELECT intervention_name, health_condition, COUNT(*) as count
        FROM interventions
        WHERE intervention_name IN ('probiotics', 'Probiotics', 'placebo', 'Placebo', 'IBS', 'irritable bowel syndrome (IBS)')
           OR health_condition IN ('probiotics', 'Probiotics', 'placebo', 'Placebo', 'IBS', 'irritable bowel syndrome (IBS)')
        GROUP BY intervention_name, health_condition
        ORDER BY count DESC
        LIMIT 10
    """)

    results = cursor.fetchall()

    print("Real database entries and their normalized forms:")
    print(f"{'Original Intervention':<25} | {'Original Condition':<35} | {'Normalized Intervention':<25} | {'Normalized Condition':<35} | Count")
    print("-" * 130)

    for row in results:
        original_intervention = row[0]
        original_condition = row[1]
        count = row[2]

        norm_intervention = normalizer.get_canonical_name(original_intervention, "intervention")
        norm_condition = normalizer.get_canonical_name(original_condition, "condition")

        print(f"{original_intervention:<25} | {original_condition:<35} | {norm_intervention:<25} | {norm_condition:<35} | {count}")

    # Show compression statistics
    print(f"\n--- Normalization Impact ---")
    stats = normalizer.get_mapping_stats()
    for entity_type, ratio_data in stats.get('ratios', {}).items():
        unique_terms = ratio_data['unique_terms']
        canonical_count = ratio_data['canonical_entities']
        compression = ratio_data['compression_ratio']
        print(f"{entity_type.title()} entities: {unique_terms} terms -> {canonical_count} canonical ({compression:.1f}x compression)")

    conn.close()
    print(f"\n[SUCCESS] Live normalization test completed!")


if __name__ == "__main__":
    test_live_normalization()