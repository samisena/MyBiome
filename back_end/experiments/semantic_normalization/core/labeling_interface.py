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


class HierarchicalLabelingInterface:
    """Interactive terminal interface for labeling intervention pairs with hierarchical relationships."""

    def __init__(self, config_path: str = None):
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

    def load_candidates(self, candidates_file: Path = None) -> List[Dict]:
        """Load candidate pairs from JSON file."""
        if candidates_file is None:
            # Find latest candidate file
            candidate_files = list(self.output_dir.glob("candidate_pairs_*.json"))
            if not candidate_files:
                raise FileNotFoundError(f"No candidate files found in {self.output_dir}")
            candidates_file = max(candidate_files, key=lambda p: p.stat().st_mtime)

        self.logger.info(f"Loading candidates from {candidates_file}")

        with open(candidates_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data['all_candidates']

    def load_or_create_session(self, session_file: Path = None) -> Dict:
        """Load existing session or create new one."""
        if session_file is None:
            # Check for existing session
            session_files = list(self.output_dir.glob("labeling_session_*.json"))
            if session_files:
                latest_session = max(session_files, key=lambda p: p.stat().st_mtime)

                # Ask if user wants to resume
                print(f"\nFound existing session: {latest_session.name}")
                response = input("Resume this session? (y/n): ").strip().lower()

                if response == 'y':
                    with open(latest_session, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)

                    self.session_id = session_data['session_id']
                    self.session_file = latest_session
                    self.labeled_pairs = session_data['labeled_pairs']
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

    def save_session(self):
        """Save current session state to file."""
        session_data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "target_pairs": self.target_pairs,
            "labeled_pairs": self.labeled_pairs,
            "progress": {
                "total": self.target_pairs,
                "labeled": len(self.labeled_pairs)
            },
            "metadata": {
                "relationship_type_counts": self._count_relationship_types()
            }
        }

        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved session with {len(self.labeled_pairs)} labeled pairs")

    def _count_relationship_types(self) -> Dict[str, int]:
        """Count occurrences of each relationship type."""
        counts = {}
        for pair in self.labeled_pairs:
            rel_type = pair.get('relationship', {}).get('type_code', 'UNKNOWN')
            counts[rel_type] = counts.get(rel_type, 0) + 1
        return counts

    def display_pair(self, pair: Dict, index: int, total: int):
        """Display intervention pair for labeling."""
        print("\n" + "="*80)
        print(f"PAIR {index + 1} of {total}")
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
        print("q - QUIT and save progress")
        print("-"*80)

    def get_relationship_type(self) -> Optional[Dict]:
        """
        Prompt user for relationship type.

        Returns:
            Dict with relationship info, or None for skip/quit
        """
        while True:
            response = input("\nSelect relationship (1-6/s/q): ").strip().lower()

            if response == 's':
                return {'action': 'skip'}
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
                print("Invalid input. Please enter 1-6, s, or q")

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
        print("\nRelationship Types:")
        print("  1. EXACT_MATCH - Same intervention (synonyms)")
        print("  2. VARIANT - Same concept, different formulation")
        print("  3. SUBTYPE - Related but clinically distinct")
        print("  4. SAME_CATEGORY - Different entities in same class")
        print("  5. DOSAGE_VARIANT - Same intervention, different dose")
        print("  6. DIFFERENT - Completely unrelated")
        print("="*80)

        # Load session
        session_data = self.load_or_create_session()

        # Load candidates
        candidates = self.load_candidates()

        if self.current_index > 0:
            print(f"\nResuming from pair {self.current_index + 1}")

        # Label pairs
        for i in range(self.current_index, min(len(candidates), self.target_pairs + self.current_index)):
            pair = candidates[i]

            # Display pair
            self.display_pair(pair, i, self.target_pairs)

            # Display relationship menu
            self.display_relationship_menu()

            # Get relationship type
            rel_result = self.get_relationship_type()

            if rel_result['action'] == 'quit':
                print("\nQuitting and saving progress...")
                self.save_session()
                break

            if rel_result['action'] == 'skip':
                print("Skipped.")
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

            # Auto-save every 5 labels
            if len(self.labeled_pairs) % 5 == 0:
                self.save_session()
                print(f"\n[Auto-saved: {len(self.labeled_pairs)} pairs labeled]")

            # Check if target reached
            if len(self.labeled_pairs) >= self.target_pairs:
                print(f"\nTarget of {self.target_pairs} pairs reached!")
                break

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
