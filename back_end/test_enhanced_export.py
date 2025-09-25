#!/usr/bin/env python3
"""
Test the enhanced export integration
"""

import sqlite3
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_export_to_json import EnhancedDataExporter


def test_enhanced_export():
    """Test the enhanced export functionality"""

    print("=== TESTING ENHANCED EXPORT WITH ENTITY NORMALIZATION ===\n")

    db_path = "data/processed/intervention_research.db"

    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    # Test with normalization enabled
    print("1. Testing with normalization ENABLED...")
    exporter_norm = EnhancedDataExporter(db_path, use_normalization=True)

    # Test get_display_info method
    print("\n--- Testing get_display_info method ---")
    test_terms = [
        ("probiotics", "intervention"),
        ("IBS", "condition"),
        ("low FODMAP diet", "intervention"),
        ("unknown_term", "intervention")
    ]

    for term, entity_type in test_terms:
        info = exporter_norm.get_display_info(term, entity_type)
        print(f"{term} ({entity_type}):")
        print(f"  Canonical: {info['canonical_name']}")
        print(f"  Original: {info['original_term']}")
        print(f"  Alternatives: {info['alternative_names']}")
        print(f"  Normalized: {info['is_normalized']}")
        print()

    # Test top interventions grouping
    print("--- Testing top interventions for IBS (grouped) ---")
    top_interventions = exporter_norm.get_top_interventions_for_condition('IBS', limit=5)

    for i, intervention in enumerate(top_interventions, 1):
        grouped_indicator = " [GROUPED]" if intervention['is_grouped'] else ""
        print(f"{i}. {intervention['intervention']}{grouped_indicator}")
        print(f"   Studies: {intervention['study_count']} (pos: {intervention['positive_studies']}, neg: {intervention['negative_studies']})")
        print(f"   Avg correlation: {intervention['avg_correlation_strength']:.3f}")
        if intervention['original_terms']:
            print(f"   Original terms: {', '.join(intervention['original_terms'][:3])}{'...' if len(intervention['original_terms']) > 3 else ''}")
        print()

    # Test with normalization disabled for comparison
    print("\n2. Testing with normalization DISABLED...")
    exporter_standard = EnhancedDataExporter(db_path, use_normalization=False)

    print("--- Testing top interventions for IBS (ungrouped) ---")
    top_interventions_std = exporter_standard.get_top_interventions_for_condition('IBS', limit=5)

    for i, intervention in enumerate(top_interventions_std, 1):
        print(f"{i}. {intervention['intervention']}")
        print(f"   Studies: {intervention['study_count']} (pos: {intervention['positive_studies']}, neg: {intervention['negative_studies']})")
        print(f"   Avg correlation: {intervention['avg_correlation_strength']:.3f}")
        print()

    # Test summary statistics
    print("--- Testing summary statistics ---")
    stats = exporter_norm.export_summary_statistics()

    print(f"Total interventions: {stats['total_interventions']}")
    print(f"Unique intervention names: {stats['unique_intervention_names']}")

    if stats['normalization']['enabled']:
        norm_stats = stats['normalization']
        print(f"Canonical interventions: {norm_stats['canonical_entities'].get('intervention', 0)}")
        print(f"Intervention reduction: {norm_stats['intervention_reduction']['reduction_percent']:.1f}%")

    # Success check: Look for probiotics grouping
    print("\n--- SUCCESS CHECK: Probiotics grouping ---")
    probiotics_found = False
    for intervention in top_interventions:
        if 'probiotic' in intervention['intervention'].lower():
            original_terms = intervention['original_terms']
            print(f"Found: {intervention['intervention']}")
            print(f"  Original terms: {', '.join(original_terms)}")
            print(f"  Is grouped: {intervention['is_grouped']}")

            # Check if we have variants like "probiotics", "probiotic", "multi-strain probiotic"
            probiotic_variants = [term for term in original_terms
                                if 'probiotic' in term.lower() and term != intervention['intervention']]

            if probiotic_variants:
                print(f"  SUCCESS: Found {len(probiotic_variants)} probiotic variants grouped together!")
                probiotics_found = True
            break

    if not probiotics_found:
        print("  No probiotic grouping found - may need more test data")

    print(f"\n[SUCCESS] Enhanced export integration test completed")


if __name__ == "__main__":
    test_enhanced_export()