"""
Batch Labeling Session Management Script

Allows labeling intervention pairs in manageable batches with:
- Session resume capability
- Progress tracking across batches
- Time estimation
- Batch status overview

Usage:
    python label_in_batches.py --batch-size 50 --start-from 0
    python label_in_batches.py --batch-size 50 --start-from 50
    python label_in_batches.py --status
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from labeling_interface import HierarchicalLabelingInterface


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
        session_files = list(self.ground_truth_dir.glob("labeling_session_*.json"))

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
            print("Start a new session with: python label_in_batches.py --batch-size 50")
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

                print(f"  Batch {i+1:2d}: Pairs {start_pair:3d}-{end_pair:3d} [{status}]")

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
            print("  python label_in_batches.py --batch-size 50 --start-from 0")
            return

        progress = session.get('progress', {})
        labeled = progress.get('labeled', 0)

        next_start = (labeled // 50) * 50

        print("\nSuggested Next Batch:")
        print(f"  python label_in_batches.py --batch-size 50 --start-from {next_start}")
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

        # VALIDATION: Load candidates to check batch boundaries
        candidates_file = self.ground_truth_dir / "hierarchical_candidates_500_pairs.json"

        if not candidates_file.exists():
            print(f"ERROR: Candidates file not found: {candidates_file}")
            print("Please generate candidates first using generate_500_candidates.py")
            return

        with open(candidates_file, 'r', encoding='utf-8') as f:
            candidates_data = json.load(f)

        total_candidates = len(candidates_data.get('all_candidates', []))

        # Validate start_from
        if start_from >= total_candidates:
            print(f"\nERROR: start_from ({start_from}) exceeds available candidates ({total_candidates})")
            print(f"Available range: 0 to {total_candidates - 1}")
            print(f"\nDid you mean to start from 0?")
            return

        # Warn if batch extends beyond available candidates
        if start_from + batch_size > total_candidates:
            actual_batch_size = total_candidates - start_from
            print(f"\nWARNING: Only {actual_batch_size} pairs available in this batch")
            print(f"  (Requested batch size: {batch_size})")
            print(f"  (Total candidates: {total_candidates})")
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


def main():
    """CLI entry point for batch labeling."""
    parser = argparse.ArgumentParser(
        description="Batch labeling session management for intervention pairs"
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Number of pairs to label in this batch (default: 50)'
    )

    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Starting pair index (0-based, default: 0)'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current labeling status without starting a new session'
    )

    parser.add_argument(
        '--suggest',
        action='store_true',
        help='Suggest the next batch to label'
    )

    args = parser.parse_args()

    manager = BatchLabelingManager()

    if args.status:
        manager.display_status()
        manager.display_batch_suggestions()
    elif args.suggest:
        manager.display_batch_suggestions()
    else:
        # Run batch labeling session
        manager.run_batch_session(
            batch_size=args.batch_size,
            start_from=args.start_from
        )


if __name__ == "__main__":
    main()
