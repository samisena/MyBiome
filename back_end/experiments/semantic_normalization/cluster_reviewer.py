"""
Interactive Cluster Reviewer

Terminal-based tool for manually reviewing and annotating clustering results.
Helps identify:
- False positives (incorrectly clustered together)
- False negatives (should be clustered but aren't)
- Edge cases (dosage, formulation, route differences)
- Good clusters (validation)

Saves annotations for experiment documentation and threshold tuning.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClusterReviewer:
    """
    Interactive terminal tool for reviewing clustering results.
    """

    def __init__(self, test_results_path: str, output_dir: Optional[str] = None):
        """
        Initialize the cluster reviewer.

        Args:
            test_results_path: Path to test results JSON file
            output_dir: Directory for review annotations (default: reviews/)
        """
        self.test_results_path = test_results_path
        self.output_dir = output_dir or "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/reviews"

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load test results
        with open(test_results_path, 'r', encoding='utf-8') as f:
            self.test_results = json.load(f)

        self.clusters = self.test_results['clusters']

        # Review state
        self.review_annotations = []
        self.current_cluster_idx = 0
        self.review_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Statistics
        self.review_stats = {
            'good_clusters': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'edge_cases': 0,
            'singletons_reviewed': 0
        }

        logger.info(f"ClusterReviewer initialized with {len(self.clusters)} clusters")

    def _print_cluster(self, canonical: str, members: List[str]):
        """Print cluster information."""
        print(f"\nCanonical: {canonical}")
        print(f"Members ({len(members)}):")
        for i, member in enumerate(members, 1):
            print(f"  {i}. {member}")

    def _get_user_choice(self, prompt: str, valid_choices: List[str]) -> str:
        """Get user input with validation."""
        while True:
            response = input(f"\n{prompt} ({'/'.join(valid_choices)}): ").strip().lower()
            if response in valid_choices:
                return response
            print(f"Invalid choice. Please enter one of: {', '.join(valid_choices)}")

    def review_multi_member_clusters(self, min_members: int = 2):
        """
        Review clusters with multiple members.

        Args:
            min_members: Minimum cluster size to review
        """
        # Filter clusters
        multi_member_clusters = {
            canonical: members
            for canonical, members in self.clusters.items()
            if len(members) >= min_members
        }

        # Sort by cluster size (descending)
        sorted_clusters = sorted(multi_member_clusters.items(), key=lambda x: len(x[1]), reverse=True)

        print("\n" + "="*80)
        print(f"REVIEWING MULTI-MEMBER CLUSTERS ({len(sorted_clusters)} clusters)")
        print("="*80)

        for i, (canonical, members) in enumerate(sorted_clusters, 1):
            print(f"\n[Cluster {i}/{len(sorted_clusters)}]")
            self._print_cluster(canonical, members)

            # Question 1: Do these all refer to the same intervention?
            same_intervention = self._get_user_choice(
                "Do these all refer to the SAME intervention?",
                ['y', 'n', 's', 'q']  # yes, no, skip, quit
            )

            if same_intervention == 'q':
                print("Quitting review...")
                break
            elif same_intervention == 's':
                continue

            # Question 2: Get notes
            notes = input("Notes (optional, press Enter to skip): ").strip()

            # Classify review
            if same_intervention == 'y':
                review_type = 'good_cluster'
                self.review_stats['good_clusters'] += 1
            else:
                # Ask for specific issue
                issue = self._get_user_choice(
                    "What's the issue? (false_positive/edge_case)",
                    ['fp', 'ec']  # false positive, edge case
                )
                if issue == 'fp':
                    review_type = 'false_positive'
                    self.review_stats['false_positives'] += 1
                else:
                    review_type = 'edge_case'
                    self.review_stats['edge_cases'] += 1

            # Save annotation
            annotation = {
                'cluster_canonical': canonical,
                'cluster_members': members,
                'cluster_size': len(members),
                'review_type': review_type,
                'same_intervention': same_intervention == 'y',
                'notes': notes,
                'reviewed_at': datetime.now().isoformat()
            }

            self.review_annotations.append(annotation)

            # Save progress periodically
            if len(self.review_annotations) % 10 == 0:
                self._save_annotations()
                print(f"\n[Progress saved: {len(self.review_annotations)} clusters reviewed]")

    def review_singleton_clusters(self, limit: int = 50):
        """
        Review interventions that became their own canonical (singletons).

        Args:
            limit: Maximum singletons to review
        """
        # Filter singleton clusters
        singletons = [
            (canonical, members)
            for canonical, members in self.clusters.items()
            if len(members) == 1
        ]

        # Sort alphabetically
        singletons.sort(key=lambda x: x[0])

        # Limit
        singletons = singletons[:limit]

        print("\n" + "="*80)
        print(f"REVIEWING SINGLETON CLUSTERS ({len(singletons)} singletons)")
        print("="*80)
        print("Looking for: common terms that should cluster, typos, abbreviations")

        for i, (canonical, members) in enumerate(singletons, 1):
            intervention_name = members[0]
            print(f"\n[Singleton {i}/{len(singletons)}]")
            print(f"Intervention: {intervention_name}")
            print(f"Canonical: {canonical}")

            # Question: Should this have clustered with something?
            should_cluster = self._get_user_choice(
                "Should this cluster with another intervention?",
                ['y', 'n', 's', 'q']
            )

            if should_cluster == 'q':
                print("Quitting review...")
                break
            elif should_cluster == 's':
                continue

            notes = ""
            if should_cluster == 'y':
                # Ask what it should cluster with
                notes = input("What should it cluster with? ").strip()
                review_type = 'false_negative'
                self.review_stats['false_negatives'] += 1
            else:
                review_type = 'correct_singleton'

            # Save annotation
            annotation = {
                'singleton_intervention': intervention_name,
                'singleton_canonical': canonical,
                'review_type': review_type,
                'should_cluster': should_cluster == 'y',
                'notes': notes,
                'reviewed_at': datetime.now().isoformat()
            }

            self.review_annotations.append(annotation)
            self.review_stats['singletons_reviewed'] += 1

            # Save progress periodically
            if len(self.review_annotations) % 10 == 0:
                self._save_annotations()
                print(f"\n[Progress saved: {len(self.review_annotations)} items reviewed]")

    def find_potential_matches(self, intervention_name: str, top_n: int = 5) -> List[str]:
        """
        Find potential matches for an intervention in other clusters.

        Args:
            intervention_name: Intervention to search for
            top_n: Number of results to return

        Returns:
            List of potential match canonical names
        """
        # Simple fuzzy matching based on shared words
        intervention_words = set(intervention_name.lower().split())

        matches = []
        for canonical, members in self.clusters.items():
            canonical_words = set(canonical.lower().split())
            overlap = len(intervention_words & canonical_words)
            if overlap > 0:
                matches.append((canonical, overlap, members))

        # Sort by overlap
        matches.sort(key=lambda x: x[1], reverse=True)

        return [(canonical, members) for canonical, _, members in matches[:top_n]]

    def _save_annotations(self):
        """Save review annotations to JSON file."""
        output_file = os.path.join(
            self.output_dir,
            f"cluster_review_{self.review_timestamp}.json"
        )

        review_data = {
            'test_results_path': self.test_results_path,
            'test_run_timestamp': self.test_results.get('run_timestamp'),
            'review_timestamp': self.review_timestamp,
            'review_stats': self.review_stats,
            'annotations': self.review_annotations
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Review annotations saved to: {output_file}")

    def print_summary(self):
        """Print review summary."""
        print("\n" + "="*80)
        print("REVIEW SUMMARY")
        print("="*80)

        print(f"\nTotal annotations: {len(self.review_annotations)}")
        print(f"\n--- MULTI-MEMBER CLUSTERS ---")
        print(f"Good clusters: {self.review_stats['good_clusters']}")
        print(f"False positives: {self.review_stats['false_positives']}")
        print(f"Edge cases: {self.review_stats['edge_cases']}")

        print(f"\n--- SINGLETONS ---")
        print(f"Singletons reviewed: {self.review_stats['singletons_reviewed']}")
        print(f"False negatives (should cluster): {self.review_stats['false_negatives']}")

        # Example annotations
        print(f"\n--- EXAMPLE ANNOTATIONS ---")

        # Good clusters
        good_clusters = [a for a in self.review_annotations if a.get('review_type') == 'good_cluster']
        if good_clusters:
            print(f"\nGood Clusters ({len(good_clusters)}):")
            for anno in good_clusters[:3]:
                print(f"  - {anno['cluster_canonical']} ({anno['cluster_size']} members)")

        # False positives
        false_positives = [a for a in self.review_annotations if a.get('review_type') == 'false_positive']
        if false_positives:
            print(f"\nFalse Positives ({len(false_positives)}):")
            for anno in false_positives[:3]:
                print(f"  - {anno['cluster_canonical']} ({anno['cluster_size']} members)")
                if anno.get('notes'):
                    print(f"    Notes: {anno['notes']}")

        # False negatives
        false_negatives = [a for a in self.review_annotations if a.get('review_type') == 'false_negative']
        if false_negatives:
            print(f"\nFalse Negatives ({len(false_negatives)}):")
            for anno in false_negatives[:3]:
                print(f"  - {anno['singleton_intervention']}")
                if anno.get('notes'):
                    print(f"    Should cluster with: {anno['notes']}")

        print("\n" + "="*80)

    def finalize_review(self):
        """Finalize and save review."""
        self._save_annotations()
        self.print_summary()

        output_file = os.path.join(
            self.output_dir,
            f"cluster_review_{self.review_timestamp}.json"
        )

        print(f"\nReview saved to: {output_file}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Cluster Review Tool")
    parser.add_argument(
        'test_results',
        help='Path to test results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/reviews',
        help='Output directory for review annotations'
    )
    parser.add_argument(
        '--mode',
        choices=['multi', 'singleton', 'all'],
        default='all',
        help='Review mode: multi-member clusters, singletons, or all'
    )
    parser.add_argument(
        '--min-members',
        type=int,
        default=2,
        help='Minimum cluster size for multi-member review'
    )
    parser.add_argument(
        '--singleton-limit',
        type=int,
        default=50,
        help='Maximum singletons to review'
    )

    args = parser.parse_args()

    # Create reviewer
    reviewer = ClusterReviewer(
        test_results_path=args.test_results,
        output_dir=args.output_dir
    )

    # Review based on mode
    if args.mode in ['multi', 'all']:
        reviewer.review_multi_member_clusters(min_members=args.min_members)

    if args.mode in ['singleton', 'all']:
        reviewer.review_singleton_clusters(limit=args.singleton_limit)

    # Finalize
    reviewer.finalize_review()


if __name__ == "__main__":
    main()
