"""
Phase 1 Orchestrator
Runs the complete Phase 1 workflow: Export → Generate → Label
"""

import sys
from pathlib import Path

# Add core module to path
sys.path.insert(0, str(Path(__file__).parent / "core"))

from core.data_exporter import InterventionDataExporter
from core.pair_generator import SmartPairGenerator
from core.labeling_interface import LabelingInterface


def run_phase1_workflow():
    """Execute complete Phase 1 workflow."""
    print("\n" + "="*70)
    print("PHASE 1: SETUP & DATA PREPARATION")
    print("="*70)

    # Step 1: Export intervention data
    print("\n[STEP 1/3] Exporting intervention data from database...")
    exporter = InterventionDataExporter()
    export_data = exporter.export_interventions()

    print(f"\nExported {export_data['metadata']['total_records']} intervention records")
    print(f"Unique intervention names: {export_data['metadata']['unique_intervention_names']}")

    # Step 2: Generate candidate pairs
    print("\n[STEP 2/3] Generating candidate pairs using fuzzy matching...")
    generator = SmartPairGenerator()
    unique_names = export_data['unique_names']

    candidates = generator.generate_candidates(unique_names)
    output_path = generator.save_candidates(candidates)

    print(f"\nGenerated {len(candidates)} candidate pairs")
    print(f"Saved to: {output_path}")

    # Step 3: Launch labeling interface
    print("\n[STEP 3/3] Launching interactive labeling interface...")
    print("\nYou will now label intervention pairs to create ground truth dataset.")

    response = input("\nReady to start labeling? (y/n): ").strip().lower()

    if response == 'y':
        interface = LabelingInterface()
        interface.run_labeling_session()
    else:
        print("\nLabeling skipped. You can run labeling later with:")
        print("  python -m core.labeling_interface")

    print("\n" + "="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review your labeled pairs in data/ground_truth/")
    print("  2. Proceed to Phase 2: Semantic embedding system development")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_phase1_workflow()
