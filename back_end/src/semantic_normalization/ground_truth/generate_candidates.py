"""
Generate 500 Candidate Pairs for Ground Truth Labeling

Uses stratified sampling strategy:
- Similarity-based sampling (60%): 300 pairs across similarity ranges
- Random sampling (20%): 100 pairs for DIFFERENT examples
- Targeted sampling (20%): 100 pairs from same drug class/category

Output: labeling_session_hierarchical_candidates_500_YYYYMMDD.json
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pair_generator import SmartPairGenerator
from data_exporter import InterventionDataExporter


def load_intervention_metadata(export_data: dict) -> dict:
    """
    Load intervention metadata including categories from export data.

    Args:
        export_data: Export data from InterventionDataExporter

    Returns:
        Dict mapping intervention names to metadata {category, ...}
    """
    metadata = {}

    # Extract from full_data if available
    if 'full_data' in export_data:
        for intervention in export_data['full_data']:
            name = intervention.get('intervention_name')
            category = intervention.get('intervention_category', 'unknown')

            if name:
                metadata[name] = {
                    'category': category,
                    'health_condition': intervention.get('health_condition', ''),
                    'study_type': intervention.get('study_type', '')
                }

    return metadata


def main():
    """Generate 500 candidate pairs using stratified sampling."""
    print("\n" + "="*80)
    print("GENERATING 500 CANDIDATE PAIRS FOR GROUND TRUTH LABELING")
    print("="*80)
    print("\nStrategy:")
    print("  - Similarity-based (60%): 300 pairs in ranges 0.85-0.95, 0.75-0.85, 0.65-0.75")
    print("  - Random low-similarity (20%): 100 pairs in range 0.40-0.65")
    print("  - Targeted same-category (20%): 100 pairs from same intervention categories")
    print("="*80 + "\n")

    # Load latest intervention export
    print("Loading intervention data...")
    exporter = InterventionDataExporter()
    export_data = exporter.get_latest_export()

    if not export_data:
        print("ERROR: No intervention export found. Please run data export first.")
        print("  python -m back_end.experiments.semantic_normalization.core.data_exporter")
        return

    unique_names = export_data['unique_names']
    print(f"Loaded {len(unique_names)} unique intervention names")

    # Load metadata (categories)
    print("Loading intervention metadata...")
    metadata = load_intervention_metadata(export_data)
    print(f"Loaded metadata for {len(metadata)} interventions")

    # Generate stratified candidates
    print("\nGenerating stratified candidate pairs...")
    generator = SmartPairGenerator()

    candidates = generator.generate_stratified_candidates(
        intervention_names=unique_names,
        intervention_metadata=metadata if metadata else None,
        target_count=500
    )

    # Save candidates
    output_dir = Path(__file__).parent / "data" / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"labeling_session_hierarchical_candidates_500_{timestamp}.json"

    # Categorize for analysis
    categorized = generator.categorize_candidates(candidates)

    output_data = {
        "metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "total_candidates": len(candidates),
            "target_count": 500,
            "sampling_strategy": {
                "similarity_based": "60% (300 pairs)",
                "random_low_similarity": "20% (100 pairs)",
                "targeted_same_category": "20% (100 pairs)"
            },
            "similarity_ranges": {
                "high": "0.85-0.95 (likely EXACT_MATCH or VARIANT)",
                "medium": "0.75-0.85 (likely SUBTYPE or VARIANT)",
                "medium_low": "0.65-0.75 (likely SAME_CATEGORY)",
                "low": "0.40-0.65 (likely DIFFERENT)"
            },
            "algorithm": generator.config['fuzzy_matching']['algorithm']
        },
        "categorized": categorized,
        "all_candidates": candidates
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("CANDIDATE GENERATION COMPLETE")
    print("="*80)
    print(f"Total candidates: {len(candidates)}")
    print(f"Output file: {output_file}")
    print("\nDistribution by similarity:")
    print(f"  - High (>0.75): {len(categorized['likely_match'])} pairs")
    print(f"  - Medium (0.50-0.75): {len(categorized['edge_case'])} pairs")
    print(f"  - Low (<0.50): {len(categorized['likely_no_match'])} pairs")
    print("\nNext steps:")
    print("  1. Start labeling with batch mode:")
    print("     python label_in_batches.py --batch-size 50 --start-from 0")
    print("  2. Check status anytime:")
    print("     python label_in_batches.py --status")
    print("  3. Continue with next batch:")
    print("     python label_in_batches.py --batch-size 50 --start-from 50")
    print("="*80 + "\n")

    # Show sample pairs
    print("Sample candidate pairs:")
    for i, candidate in enumerate(candidates[:10], 1):
        try:
            print(f"{i}. [{candidate['similarity_score']:.2f}] '{candidate['intervention_1']}' vs '{candidate['intervention_2']}'")
        except UnicodeEncodeError:
            print(f"{i}. [{candidate['similarity_score']:.2f}] (pair with special characters)")

    if len(candidates) > 10:
        print(f"... and {len(candidates) - 10} more")

    print()


if __name__ == "__main__":
    main()
