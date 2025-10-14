"""
Ground Truth Workflow CLI

Consolidated command-line interface for ground truth labeling workflow.
Combines functionality from generate_candidates.py, label_in_batches.py, and remove_duplicate_labels.py.

Usage:
    # Generate 500 candidate pairs
    python ground_truth_cli.py generate [--count 500]

    # Start labeling session
    python ground_truth_cli.py label [--batch-size 50] [--start-from 0]

    # Check labeling status
    python ground_truth_cli.py status

    # Clean duplicate labels
    python ground_truth_cli.py clean

For more options:
    python ground_truth_cli.py --help
    python ground_truth_cli.py <command> --help
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pair_generator import SmartPairGenerator
from data_exporter import InterventionDataExporter
from labeling_interface import HierarchicalLabelingInterface


# ==============================================================================
# GENERATE CANDIDATES COMMAND
# ==============================================================================

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


def cmd_generate(args):
    """Generate candidate pairs using stratified sampling."""
    print("\n" + "="*80)
    print("GENERATING CANDIDATE PAIRS FOR GROUND TRUTH LABELING")
    print("="*80)
    print(f"\nTarget: {args.count} pairs")
    print("\nStrategy:")
    print("  - Similarity-based (60%): Pairs across similarity ranges 0.85-0.95, 0.75-0.85, 0.65-0.75")
    print("  - Random low-similarity (20%): Pairs in range 0.40-0.65")
    print("  - Targeted same-category (20%): Pairs from same intervention categories")
    print("="*80 + "\n")

    # Load latest intervention export
    print("Loading intervention data...")
    exporter = InterventionDataExporter()
    export_data = exporter.get_latest_export()

    if not export_data:
        print("ERROR: No intervention export found. Please run data export first.")
        print("  python data_exporter.py")
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
        target_count=args.count
    )

    # Save candidates
    output_dir = Path(__file__).parent / "data" / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"labeling_session_hierarchical_candidates_{args.count}_{timestamp}.json"

    # Categorize for analysis
    categorized = generator.categorize_candidates(candidates)

    output_data = {
        "metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "total_candidates": len(candidates),
            "target_count": args.count,
            "sampling_strategy": {
                "similarity_based": "60%",
                "random_low_similarity": "20%",
                "targeted_same_category": "20%"
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
    print("  1. Start labeling:")
    print(f"     python {Path(__file__).name} label --batch-size 50 --start-from 0")
    print("  2. Check status anytime:")
    print(f"     python {Path(__file__).name} status")
    print("="*80 + "\n")

    # Show sample pairs
    print("Sample candidate pairs:")
    for i, candidate in enumerate(candidates[:10], 1):
        try:
            print(f"{i}. [{candidate['similarity_score']:.2f}] '{candidate['intervention_1']}' vs '{candidate['intervention_2']}'")
        except (UnicodeEncodeError, KeyError):
            print(f"{i}. (pair with special characters or missing data)")

    if len(candidates) > 10:
        print(f"... and {len(candidates) - 10} more")

    print()


# ==============================================================================
# LABEL COMMAND (BATCH LABELING)
# ==============================================================================

class BatchLabelingManager:
    """Manage batch labeling sessions with progress tracking."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize batch labeling manager."""
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "config.yaml"

        self.config_path = config_path
        self.ground_truth_dir = Path(__file__).parent / "data" / "ground_truth"
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_session(self) -> Optional[Dict]:
        """Get the latest labeling session data."""
        session_files = [f for f in self.ground_truth_dir.glob("labeling_session_*.json")
                        if 'candidates' not in f.name]

        if not session_files:
            return None

        latest_file = max(session_files, key=lambda p: p.stat().st_mtime)

        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def display_status(self):
        """Display current labeling session status."""
        print("\n" + "="*80)
        print("BATCH LABELING STATUS")
        print("="*80)

        session = self.get_latest_session()

        if not session:
            print("\nNo active labeling session found.")
            print(f"Start a new session with: python {Path(__file__).name} label --batch-size 50")
            print("="*80 + "\n")
            return

        progress = session.get('progress', {})
        total = progress.get('total', 500)
        labeled = progress.get('labeled', 0)
        percentage = progress.get('percentage', 0)

        print(f"\nSession ID: {session.get('session_id', 'Unknown')}")
        print(f"Created: {session.get('created_at', 'Unknown')}")
        print(f"\nProgress: {labeled}/{total} pairs ({percentage}%)")

        # Show batch breakdown
        if total > 0:
            batch_size = 50
            num_batches = (total + batch_size - 1) // batch_size
            completed_batches = labeled // batch_size
            current_batch = (labeled // batch_size) + 1 if labeled < total else num_batches

            print(f"\nBatch Progress: {completed_batches}/{num_batches} batches complete")
            print(f"Current Batch: #{current_batch} (pairs {completed_batches * batch_size + 1}-{min((completed_batches + 1) * batch_size, total)})")

            # Show batch grid
            print("\nBatch Grid:")
            for i in range(num_batches):
                start_pair = i * batch_size + 1
                end_pair = min((i + 1) * batch_size, total)
                pairs_in_batch = end_pair - start_pair + 1

                if labeled >= end_pair:
                    status = "[DONE]"
                elif labeled >= start_pair:
                    done_in_batch = labeled - start_pair + 1
                    status = f"[IN PROGRESS] ({done_in_batch}/{pairs_in_batch})"
                else:
                    status = "[PENDING]"

                print(f"  Batch {i+1:2d}: Pairs {start_pair:3d}-{end_pair:3d} {status}")

        # Show relationship type distribution
        metadata = session.get('metadata', {})
        rel_counts = metadata.get('relationship_type_counts', {})

        if rel_counts:
            print("\nRelationship Type Distribution:")
            for rel_type, count in sorted(rel_counts.items()):
                print(f"  - {rel_type}: {count}")

        print("\n" + "="*80 + "\n")

    def display_batch_suggestions(self):
        """Suggest next batch to label."""
        session = self.get_latest_session()

        if not session:
            print("\nSuggestion: Start with batch 1")
            print(f"  python {Path(__file__).name} label --batch-size 50 --start-from 0")
            return

        progress = session.get('progress', {})
        labeled = progress.get('labeled', 0)
        total = progress.get('total', 500)

        if labeled >= total:
            print("\nAll pairs labeled! Run 'clean' command to remove duplicates.")
            return

        next_start = (labeled // 50) * 50

        print("\nSuggested Next Batch:")
        print(f"  python {Path(__file__).name} label --batch-size 50 --start-from {next_start}")
        print()

    def run_batch_session(self, batch_size: int = 50, start_from: int = 0):
        """
        Run a batch labeling session.

        Args:
            batch_size: Number of pairs to label in this batch
            start_from: Starting index (0-based)
        """
        print(f"\nStarting batch labeling session:")
        print(f"  Batch size: {batch_size}")
        print(f"  Starting from: {start_from}")
        print(f"  Pairs to label: {start_from} to {start_from + batch_size - 1}")
        print()

        # Find candidates file
        candidates_files = list(self.ground_truth_dir.glob("labeling_session_*candidates*.json"))

        if not candidates_files:
            print(f"ERROR: No candidates file found in {self.ground_truth_dir}")
            print(f"Please generate candidates first: python {Path(__file__).name} generate")
            return

        candidates_file = max(candidates_files, key=lambda p: p.stat().st_mtime)
        print(f"Using candidates file: {candidates_file.name}")

        with open(candidates_file, 'r', encoding='utf-8') as f:
            candidates_data = json.load(f)

        total_candidates = len(candidates_data.get('all_candidates', []))

        # Validate start_from
        if start_from >= total_candidates:
            print(f"\nERROR: start_from ({start_from}) exceeds available candidates ({total_candidates})")
            print(f"Available range: 0 to {total_candidates - 1}")
            return

        # Warn if batch extends beyond available candidates
        if start_from + batch_size > total_candidates:
            actual_batch_size = total_candidates - start_from
            print(f"\nWARNING: Only {actual_batch_size} pairs available in this batch")
            print(f"  (Requested: {batch_size}, Available: {actual_batch_size})")
            print(f"  (This will be the final batch)")

            response = input("\nContinue with reduced batch size? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return

            batch_size = actual_batch_size

        # Create labeling interface with batch mode
        interface = HierarchicalLabelingInterface(
            config_path=str(self.config_path),
            batch_size=batch_size,
            start_from=start_from
        )

        # Run labeling session
        interface.run_labeling_session()

        # Display updated status
        print("\nBatch session complete!")
        self.display_status()
        self.display_batch_suggestions()


def cmd_label(args):
    """Start batch labeling session."""
    manager = BatchLabelingManager()
    manager.run_batch_session(
        batch_size=args.batch_size,
        start_from=args.start_from
    )


# ==============================================================================
# STATUS COMMAND
# ==============================================================================

def cmd_status(args):
    """Display current labeling status."""
    manager = BatchLabelingManager()
    manager.display_status()
    manager.display_batch_suggestions()


# ==============================================================================
# CLEAN COMMAND (REMOVE DUPLICATES)
# ==============================================================================

def remove_duplicates_from_session(session_file: Path):
    """Remove duplicate pair labels from session file."""

    # Load session
    with open(session_file, 'r', encoding='utf-8') as f:
        session = json.load(f)

    labeled_pairs = session.get('labeled_pairs', [])
    print(f"Original labeled pairs: {len(labeled_pairs)}")

    # Find duplicates
    seen_pairs = set()
    unique_pairs = []
    duplicates = []

    for i, label in enumerate(labeled_pairs):
        key = tuple(sorted([label['intervention_1'], label['intervention_2']]))

        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_pairs.append(label)
        else:
            duplicates.append({
                'index': i,
                'pair': key,
                'label': label
            })

    if not duplicates:
        print("\n[OK] No duplicates found!")
        return

    print(f"\nFound {len(duplicates)} duplicate labels:")
    for dup in duplicates[:10]:  # Show first 10
        print(f"  [{dup['index']}] {dup['pair'][0]} vs {dup['pair'][1]}")
        rel_type = dup['label'].get('relationship', {}).get('type_code', 'Unknown')
        print(f"    Label: {rel_type}")

    if len(duplicates) > 10:
        print(f"  ... and {len(duplicates) - 10} more")

    # Ask for confirmation
    response = input(f"\nRemove {len(duplicates)} duplicate labels? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Create backup
    backup_file = session_file.with_suffix('.json.backup')
    shutil.copy(session_file, backup_file)
    print(f"\n[OK] Backup created: {backup_file}")

    # Update session
    session['labeled_pairs'] = unique_pairs
    session['progress']['labeled'] = len(unique_pairs)
    session['progress']['percentage'] = round((len(unique_pairs) / session['progress']['total']) * 100, 1)
    session['metadata']['last_updated'] = datetime.now().isoformat()

    # Save cleaned session
    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Removed {len(duplicates)} duplicate labels")
    print(f"[OK] Unique pairs remaining: {len(unique_pairs)}")
    print(f"[OK] Session saved: {session_file}")


def cmd_clean(args):
    """Remove duplicate labels from session files."""

    # Check ground truth directory
    ground_truth_dir = Path(__file__).parent / "data" / "ground_truth"

    if not ground_truth_dir.exists():
        print("ERROR: Ground truth directory not found.")
        return

    # Find session files (exclude candidates files)
    session_files = [f for f in ground_truth_dir.glob("labeling_session_*.json")
                    if 'candidates' not in f.name and 'backup' not in f.name.lower()]

    if not session_files:
        print("No session files found.")
        return

    print(f"Found {len(session_files)} session file(s):\n")
    for i, f in enumerate(session_files, 1):
        print(f"{i}. {f.name}")

    # Process most recent
    latest = max(session_files, key=lambda p: p.stat().st_mtime)
    print(f"\nProcessing most recent: {latest.name}\n")

    remove_duplicates_from_session(latest)


# ==============================================================================
# MAIN CLI
# ==============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ground Truth Workflow CLI - Consolidated tool for ground truth labeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 500 candidate pairs
  python %(prog)s generate

  # Generate custom number of pairs
  python %(prog)s generate --count 1000

  # Start labeling (batch 1)
  python %(prog)s label --batch-size 50 --start-from 0

  # Continue labeling (batch 2)
  python %(prog)s label --batch-size 50 --start-from 50

  # Check status
  python %(prog)s status

  # Clean duplicates
  python %(prog)s clean
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate command
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate candidate pairs for labeling'
    )
    generate_parser.add_argument(
        '--count',
        type=int,
        default=500,
        help='Number of candidate pairs to generate (default: 500)'
    )
    generate_parser.set_defaults(func=cmd_generate)

    # Label command
    label_parser = subparsers.add_parser(
        'label',
        help='Start batch labeling session'
    )
    label_parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of pairs to label in this batch (default: 50)'
    )
    label_parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Starting pair index, 0-based (default: 0)'
    )
    label_parser.set_defaults(func=cmd_label)

    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show current labeling progress'
    )
    status_parser.set_defaults(func=cmd_status)

    # Clean command
    clean_parser = subparsers.add_parser(
        'clean',
        help='Remove duplicate labels from session'
    )
    clean_parser.set_defaults(func=cmd_clean)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()