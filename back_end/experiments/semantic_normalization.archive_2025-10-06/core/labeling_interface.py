"""
Interactive Labeling Interface (Hierarchical)
Terminal-based interface for manually labeling intervention pairs with hierarchical relationships.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import time


class HierarchicalLabelingInterface:
    """Interactive terminal interface for labeling intervention pairs with hierarchical relationships."""

    def __init__(self, config_path: str = None, batch_size: int = None, start_from: int = 0):
        """Initialize labeling interface with configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.target_pairs = self.config['labeling']['target_pairs']
        self.output_dir = Path(self.config['paths']['ground_truth_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load relationship types
        self.relationship_types = self.config['labeling']['relationship_types']

        # Setup logging
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger(__name__)

        # Session state
        self.session_id = None
        self.session_file = None
        self.labeled_pairs = []
        self.current_index = 0

        # Batch mode settings
        self.batch_size = batch_size
        self.start_from = start_from

        # Undo history (stores last N labeled pairs)
        self.undo_history = []
        self.max_undo_history = 10

        # Performance tracking
        self.session_start_time = None
        self.labels_per_minute = []

    def load_candidates(self, candidates_file: Path = None) -> List[Dict]:
        """Load candidate pairs from JSON file."""
        if candidates_file is None:
            # Find latest candidate file (try multiple patterns)
            candidate_files = []

            # Pattern 1: Old format
            candidate_files.extend(self.output_dir.glob("candidate_pairs_*.json"))

            # Pattern 2: New format with "candidates" in name
            candidate_files.extend([
                f for f in self.output_dir.glob("labeling_session_*candidates*.json")
                if 'candidates' in f.name
            ])

            if not candidate_files:
                raise FileNotFoundError(f"No candidate files found in {self.output_dir}")

            candidates_file = max(candidate_files, key=lambda p: p.stat().st_mtime)

        self.logger.info(f"Loading candidates from {candidates_file}")

        with open(candidates_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('all_candidates', data.get('candidates', []))

    def load_or_create_session(self, session_file: Path = None) -> Dict:
        """Load existing session or create new one."""
        if session_file is None:
            # Check for existing session (exclude candidates files)
            session_files = [
                f for f in self.output_dir.glob("labeling_session_*.json")
                if 'candidates' not in f.name
            ]
            if session_files:
                latest_session = max(session_files, key=lambda p: p.stat().st_mtime)

                # Ask if user wants to resume
                print(f"\nFound existing session: {latest_session.name}")
                response = input("Resume this session? (y/n): ").strip().lower()

                if response == 'y':
                    with open(latest_session, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)

                    # Validate it's a session file
                    if 'session_id' not in session_data:
                        print(f"ERROR: {latest_session.name} is not a valid session file (missing session_id)")
                        print("Creating new session instead...")
                    else:
                        self.session_id = session_data['session_id']
                        self.session_file = latest_session
                        self.labeled_pairs = session_data.get('labeled_pairs', [])
                        self.current_index = len(self.labeled_pairs)

                        self.logger.info(f"Resumed session {self.session_id} with {self.current_index} labeled pairs")
                        return session_data

        # Create new session
        self.session_id = f"hierarchical_ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_file = self.output_dir / f"labeling_session_{self.session_id}.json"
        self.labeled_pairs = []
        self.current_index = 0

        self.logger.info(f"Created new session {self.session_id}")

        return {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "target_pairs": self.target_pairs,
            "labeled_pairs": [],
            "progress": {"total": self.target_pairs, "labeled": 0}
        }

    def save_session(self, auto_save: bool = False):
        """Save current session state to file."""
        session_data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "target_pairs": self.target_pairs,
            "labeled_pairs": self.labeled_pairs,
            "progress": {
                "total": self.target_pairs,
                "labeled": len(self.labeled_pairs),
                "percentage": round((len(self.labeled_pairs) / self.target_pairs) * 100, 1) if self.target_pairs > 0 else 0
            },
            "metadata": {
                "relationship_type_counts": self._count_relationship_types(),
                "batch_mode": self.batch_size is not None,
                "batch_size": self.batch_size,
                "start_from": self.start_from
            }
        }

        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        if auto_save:
            self.logger.debug(f"Auto-saved session with {len(self.labeled_pairs)} labeled pairs")
        else:
            self.logger.info(f"Saved session with {len(self.labeled_pairs)} labeled pairs")

    def _count_relationship_types(self) -> Dict[str, int]:
        """Count occurrences of each relationship type."""
        counts = {}
        for pair in self.labeled_pairs:
            rel_type = pair.get('relationship', {}).get('type_code', 'UNKNOWN')
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts

    def undo_last_label(self) -> bool:
        """Undo the last labeled pair. Returns True if successful."""
        if not self.labeled_pairs:
            print("\nNo labels to undo.")
            return False

        # Remove last label
        undone_pair = self.labeled_pairs.pop()

        # Add to undo history
        self.undo_history.append(undone_pair)
        if len(self.undo_history) > self.max_undo_history:
            self.undo_history.pop(0)

        # Auto-save after undo
        self.save_session(auto_save=True)

        print(f"\nUndone: '{undone_pair['intervention_1']}' vs '{undone_pair['intervention_2']}'")
        print(f"Relationship was: {undone_pair['relationship']['type_code']}")
        return True

    def display_progress_bar(self, current: int, total: int, width: int = 50):
        """Display a progress bar."""
        percentage = (current / total) * 100 if total > 0 else 0
        filled = int(width * current / total) if total > 0 else 0
        bar = '█' * filled + '-' * (width - filled)
        print(f"\nProgress: [{bar}] {current}/{total} ({percentage:.1f}%)")

    def estimate_time_remaining(self) -> str:
        """Estimate time remaining based on labeling speed."""
        if not self.labels_per_minute or len(self.labeled_pairs) == 0:
            return "Unknown"

        avg_labels_per_minute = sum(self.labels_per_minute) / len(self.labels_per_minute)
        remaining_labels = self.target_pairs - len(self.labeled_pairs)

        if avg_labels_per_minute == 0:
            return "Unknown"

        minutes_remaining = remaining_labels / avg_labels_per_minute
        hours = int(minutes_remaining // 60)
        minutes = int(minutes_remaining % 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def display_pair(self, pair: Dict, index: int, total: int):
        """Display intervention pair for labeling."""
        print("\n" + "="*80)
        print(f"PAIR {index + 1} of {total}")

        # Show progress bar
        self.display_progress_bar(len(self.labeled_pairs), self.target_pairs)

        # Show time estimate
        time_est = self.estimate_time_remaining()
        if time_est != "Unknown":
            print(f"Estimated time remaining: {time_est}")

        print("="*80)
        print(f"\nIntervention 1: {pair['intervention_1']}")
        print(f"Intervention 2: {pair['intervention_2']}")
        print(f"\nSimilarity Score: {pair['similarity_score']:.4f}")
        print(f"Length Difference: {pair['length_diff']} characters")
        print(f"Word Counts: {pair['word_count_1']} vs {pair['word_count_2']}")
        print("-"*80)

    def display_relationship_menu(self):
        """Display relationship type menu."""
        print("\nWhat is the relationship between these interventions?")
        print("-"*80)

        for num, rel_info in sorted(self.relationship_types.items()):
            print(f"{num}. {rel_info['code']}")
            print(f"   {rel_info['display']}")
            if 'examples' in rel_info and rel_info['examples']:
                # Handle examples safely
                examples_str = str(rel_info['examples'][0]) if isinstance(rel_info['examples'], list) else str(rel_info['examples'])
                print(f"   Examples: {examples_str}")
            print()

        print("s - SKIP this pair")
        print("u - UNDO last label (Ctrl+Z)")
        print("r - Review later (mark for review)")
        print("q - QUIT and save progress")
        print("-"*80)

    def get_relationship_type(self) -> Optional[Dict]:
        """
        Prompt user for relationship type.

        Returns:
            Dict with relationship info, or None for skip/quit
        """
        while True:
            response = input("\nSelect relationship (1-6/s/u/r/q): ").strip().lower()

            if response == 's':
                return {'action': 'skip'}
            elif response == 'u':
                return {'action': 'undo'}
            elif response == 'r':
                return {'action': 'review'}
            elif response == 'q':
                return {'action': 'quit'}
            elif response in ['1', '2', '3', '4', '5', '6']:
                rel_num = int(response)
                rel_info = self.relationship_types[rel_num]
                return {
                    'action': 'label',
                    'type_code': rel_info['code'],
                    'type_display': rel_info['display'],
                    'aggregation': rel_info['aggregation']
                }
            else:
                print("Invalid input. Please enter 1-6, s, u, r, or q")

    def get_canonical_group(self, intervention_1: str, intervention_2: str, rel_type_code: str) -> Optional[str]:
        """
        Prompt for canonical group (Layer 1) if relationship requires it.

        Returns:
            Canonical group name, or None if not applicable
        """
        # Only ask for canonical group if relationship shares Layer 1
        if rel_type_code in ['EXACT_MATCH', 'VARIANT', 'SUBTYPE', 'SAME_CATEGORY', 'DOSAGE_VARIANT']:
            print(f"\nWhat is the canonical group (Layer 1) for these interventions?")
            print(f"(e.g., 'probiotics', 'statins', 'IBS', 'cetuximab')")
            print(f"Intervention 1: {intervention_1}")
            print(f"Intervention 2: {intervention_2}")

            canonical = input("Canonical group (or press Enter to skip): ").strip()
            return canonical if canonical else None

        return None

    def get_specific_variants(self, intervention_1: str, intervention_2: str, rel_type_code: str) -> Dict[str, str]:
        """
        Prompt for specific variants (Layer 2).

        Returns:
            Dict with variant_1 and variant_2
        """
        # For most relationships, use the original names as variants
        if rel_type_code == 'DIFFERENT':
            return {
                'variant_1': intervention_1,
                'variant_2': intervention_2,
                'same_variant': False
            }

        print(f"\nAre these the SAME specific variant or DIFFERENT variants?")
        print(f"Intervention 1: {intervention_1}")
        print(f"Intervention 2: {intervention_2}")
        print(f"Examples:")
        print(f"  - SAME: 'metformin' = 'metformin therapy' (same drug)")
        print(f"  - DIFFERENT: 'Cetuximab' != 'Cetuximab-β' (biosimilar)")
        print(f"  - DIFFERENT: 'L. reuteri' != 'S. boulardii' (different strains)")

        same = input("Same variant? (y/n): ").strip().lower()

        return {
            'variant_1': intervention_1,
            'variant_2': intervention_2,
            'same_variant': (same == 'y')
        }

    def run_labeling_session(self):
        """Run interactive labeling session."""
        print("\n" + "="*80)
        print("HIERARCHICAL INTERVENTION PAIR LABELING SESSION")
        print("="*80)
        print("\nInstructions:")
        print("  - Review each pair of intervention names")
        print("  - Select the hierarchical relationship type (1-6)")
        print("  - Provide canonical group and variant information")
        print("  - Press 'u' to undo last label, 'r' to mark for review")
        print("\nRelationship Types:")
        print("  1. EXACT_MATCH - Same intervention (synonyms)")
        print("  2. VARIANT - Same concept, different formulation")
        print("  3. SUBTYPE - Related but clinically distinct")
        print("  4. SAME_CATEGORY - Different entities in same class")
        print("  5. DOSAGE_VARIANT - Same intervention, different dose")
        print("  6. DIFFERENT - Completely unrelated")

        if self.batch_size:
            print(f"\nBatch Mode: Labeling {self.batch_size} pairs starting from {self.start_from}")

        print("="*80)

        # Load session
        session_data = self.load_or_create_session()

        # Load candidates
        candidates = self.load_candidates()

        # Set session start time for performance tracking
        self.session_start_time = time.time()
        last_label_time = self.session_start_time

        if self.current_index > 0:
            print(f"\nResuming from pair {self.current_index + 1}")

        # Determine range (batch mode or full range)
        if self.batch_size:
            start_idx = max(self.current_index, self.start_from)
            end_idx = min(len(candidates), self.start_from + self.batch_size)
        else:
            start_idx = self.current_index
            end_idx = min(len(candidates), self.target_pairs)

        # Track pairs to review later
        review_later = []

        # Label pairs
        i = start_idx
        while i < end_idx:
            pair = candidates[i]

            # Display pair
            self.display_pair(pair, i, end_idx)

            # Display relationship menu
            self.display_relationship_menu()

            # Get relationship type
            rel_result = self.get_relationship_type()

            if rel_result['action'] == 'quit':
                print("\nQuitting and saving progress...")
                self.save_session()
                break

            if rel_result['action'] == 'undo':
                if self.undo_last_label():
                    # Don't advance i, stay on current pair
                    continue
                else:
                    # No labels to undo, stay on current pair
                    continue

            if rel_result['action'] == 'review':
                print("Marked for review later.")
                review_later.append(pair)
                i += 1
                continue

            if rel_result['action'] == 'skip':
                print("Skipped.")
                i += 1
                continue

            # Get hierarchical details
            rel_type_code = rel_result['type_code']

            canonical_group = self.get_canonical_group(
                pair['intervention_1'],
                pair['intervention_2'],
                rel_type_code
            )

            variant_info = self.get_specific_variants(
                pair['intervention_1'],
                pair['intervention_2'],
                rel_type_code
            )

            # Record label
            labeled_pair = {
                "pair_id": len(self.labeled_pairs) + 1,
                "intervention_1": pair['intervention_1'],
                "intervention_2": pair['intervention_2'],
                "similarity_score": pair['similarity_score'],

                "relationship": {
                    "type_code": rel_type_code,
                    "type_display": rel_result['type_display'],
                    "aggregation_rule": rel_result['aggregation'],

                    "hierarchy": {
                        "layer_1_canonical": canonical_group,
                        "layer_2_variant_1": variant_info['variant_1'],
                        "layer_2_variant_2": variant_info['variant_2'],
                        "same_variant_layer_2": variant_info['same_variant']
                    }
                },

                "labeled_at": datetime.now().isoformat()
            }

            self.labeled_pairs.append(labeled_pair)

            # Track labeling speed (every 5 labels)
            if len(self.labeled_pairs) % 5 == 0:
                current_time = time.time()
                time_elapsed = (current_time - last_label_time) / 60  # minutes
                labels_per_min = 5 / time_elapsed if time_elapsed > 0 else 0
                self.labels_per_minute.append(labels_per_min)
                last_label_time = current_time

            # Auto-save every 10 labels
            if len(self.labeled_pairs) % 10 == 0:
                self.save_session(auto_save=True)
                print(f"\n[Auto-saved: {len(self.labeled_pairs)} pairs labeled]")

            # Move to next pair
            i += 1

            # Check if target reached
            if len(self.labeled_pairs) >= self.target_pairs:
                print(f"\nTarget of {self.target_pairs} pairs reached!")
                break

        # Save review_later pairs if any
        if review_later:
            review_file = self.output_dir / f"review_later_{self.session_id}.json"
            with open(review_file, 'w', encoding='utf-8') as f:
                json.dump(review_later, f, indent=2, ensure_ascii=False)
            print(f"\nSaved {len(review_later)} pairs for later review: {review_file}")

        # Final save
        self.save_session()

        # Display summary
        self.display_summary()

    def display_summary(self):
        """Display labeling session summary."""
        rel_counts = self._count_relationship_types()

        print("\n" + "="*80)
        print("LABELING SESSION SUMMARY")
        print("="*80)
        print(f"Session ID: {self.session_id}")
        print(f"Total pairs labeled: {len(self.labeled_pairs)}")
        print("\nRelationship Type Distribution:")
        for rel_type, count in sorted(rel_counts.items()):
            print(f"  - {rel_type}: {count}")
        print(f"\nSaved to: {self.session_file}")
        print("="*80 + "\n")


def main():
    """CLI entry point for hierarchical labeling interface."""
    interface = HierarchicalLabelingInterface()
    interface.run_labeling_session()


if __name__ == "__main__":
    main()
