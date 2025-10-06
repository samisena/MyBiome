"""
Experiment Logger

Structured logging for normalization experiments.
Tracks:
- Date and configuration
- Statistics
- Examples of successes and failures
- Hypotheses about failure causes
- Ideas for next iteration

Designed for iterative experimentation and threshold tuning.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path


class ExperimentLogger:
    """
    Log experiment details for iterative testing.
    """

    def __init__(self, experiments_dir: Optional[str] = None):
        """
        Initialize the experiment logger.

        Args:
            experiments_dir: Directory for experiment logs (default: experiments/)
        """
        self.experiments_dir = experiments_dir or "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/experiments"
        os.makedirs(self.experiments_dir, exist_ok=True)

        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Experiment data
        self.experiment_log = {
            'experiment_id': self.experiment_id,
            'date': datetime.now().isoformat(),
            'configuration': {},
            'statistics': {},
            'examples': {
                'good_clusters': [],
                'false_positives': [],
                'false_negatives': [],
                'edge_cases': []
            },
            'findings': {
                'what_worked': [],
                'what_failed': [],
                'hypotheses': [],
                'ideas_for_next_iteration': []
            }
        }

    def set_configuration(self, config_dict: Dict):
        """
        Set experiment configuration.

        Args:
            config_dict: Configuration parameters
        """
        self.experiment_log['configuration'] = config_dict

    def set_statistics(self, stats_dict: Dict):
        """
        Set experiment statistics.

        Args:
            stats_dict: Statistics from test run
        """
        self.experiment_log['statistics'] = stats_dict

    def add_good_cluster(self, canonical: str, members: List[str], notes: str = ""):
        """Add example of successful clustering."""
        self.experiment_log['examples']['good_clusters'].append({
            'canonical': canonical,
            'members': members,
            'notes': notes
        })

    def add_false_positive(self, canonical: str, members: List[str], notes: str = ""):
        """Add example of false positive (incorrectly clustered)."""
        self.experiment_log['examples']['false_positives'].append({
            'canonical': canonical,
            'members': members,
            'notes': notes
        })

    def add_false_negative(self, intervention: str, should_cluster_with: str, notes: str = ""):
        """Add example of false negative (should cluster but doesn't)."""
        self.experiment_log['examples']['false_negatives'].append({
            'intervention': intervention,
            'should_cluster_with': should_cluster_with,
            'notes': notes
        })

    def add_edge_case(self, canonical: str, members: List[str], case_type: str, notes: str = ""):
        """
        Add edge case example.

        Args:
            canonical: Canonical name
            members: Cluster members
            case_type: Type of edge case (dosage, formulation, route, etc.)
            notes: Additional notes
        """
        self.experiment_log['examples']['edge_cases'].append({
            'canonical': canonical,
            'members': members,
            'case_type': case_type,
            'notes': notes
        })

    def add_finding(self, category: str, finding: str):
        """
        Add finding to experiment log.

        Args:
            category: Finding category (what_worked, what_failed, hypotheses, ideas_for_next_iteration)
            finding: Finding description
        """
        if category in self.experiment_log['findings']:
            self.experiment_log['findings'][category].append(finding)

    def import_from_test_results(self, test_results_path: str):
        """
        Import statistics from test results JSON.

        Args:
            test_results_path: Path to test results JSON
        """
        with open(test_results_path, 'r', encoding='utf-8') as f:
            test_results = json.load(f)

        # Import statistics
        self.set_statistics(test_results.get('stats_summary', {}))

        # Import configuration
        if 'config_path' in test_results:
            config_path = test_results['config_path']
            if config_path and os.path.exists(config_path):
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.set_configuration({
                        'embedding_model': config.get('embedding', {}).get('model'),
                        'llm_model': config.get('llm', {}).get('model'),
                        'thresholds': config.get('thresholds', {}),
                        'relationship_types': list(config.get('relationship_types', {}).keys())
                    })

    def import_from_cluster_review(self, review_path: str):
        """
        Import examples from cluster review annotations.

        Args:
            review_path: Path to cluster review JSON
        """
        with open(review_path, 'r', encoding='utf-8') as f:
            review_data = json.load(f)

        annotations = review_data.get('annotations', [])

        for anno in annotations:
            review_type = anno.get('review_type')

            if review_type == 'good_cluster':
                self.add_good_cluster(
                    canonical=anno['cluster_canonical'],
                    members=anno['cluster_members'],
                    notes=anno.get('notes', '')
                )
            elif review_type == 'false_positive':
                self.add_false_positive(
                    canonical=anno['cluster_canonical'],
                    members=anno['cluster_members'],
                    notes=anno.get('notes', '')
                )
            elif review_type == 'false_negative':
                self.add_false_negative(
                    intervention=anno['singleton_intervention'],
                    should_cluster_with=anno.get('notes', 'unknown'),
                    notes=""
                )
            elif review_type == 'edge_case':
                self.add_edge_case(
                    canonical=anno['cluster_canonical'],
                    members=anno['cluster_members'],
                    case_type='unknown',
                    notes=anno.get('notes', '')
                )

    def save_log(self) -> str:
        """
        Save experiment log to JSON file.

        Returns:
            Path to saved log file
        """
        output_file = os.path.join(
            self.experiments_dir,
            f"experiment_{self.experiment_id}.json"
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False)

        return output_file

    def generate_markdown_report(self) -> str:
        """
        Generate markdown report of experiment.

        Returns:
            Markdown report string
        """
        report = f"# Experiment {self.experiment_id}\n\n"
        report += f"**Date:** {self.experiment_log['date']}\n\n"

        # Configuration
        report += "## Configuration\n\n"
        config = self.experiment_log['configuration']
        if config:
            report += "```yaml\n"
            for key, value in config.items():
                report += f"{key}: {value}\n"
            report += "```\n\n"

        # Statistics
        report += "## Statistics\n\n"
        stats = self.experiment_log['statistics']
        if stats:
            report += "| Metric | Value |\n"
            report += "|--------|-------|\n"
            for key, value in stats.items():
                if isinstance(value, float):
                    report += f"| {key} | {value:.2%} |\n"
                else:
                    report += f"| {key} | {value} |\n"
            report += "\n"

        # Examples
        examples = self.experiment_log['examples']

        # Good clusters
        if examples['good_clusters']:
            report += "## Good Clusters (Worked Correctly)\n\n"
            for i, example in enumerate(examples['good_clusters'][:10], 1):
                report += f"{i}. **{example['canonical']}** ({len(example['members'])} members)\n"
                for member in example['members'][:5]:
                    report += f"   - {member}\n"
                if len(example['members']) > 5:
                    report += f"   - ... and {len(example['members']) - 5} more\n"
                if example.get('notes'):
                    report += f"   - *Notes:* {example['notes']}\n"
                report += "\n"

        # False positives
        if examples['false_positives']:
            report += "## False Positives (Incorrectly Clustered)\n\n"
            for i, example in enumerate(examples['false_positives'][:10], 1):
                report += f"{i}. **{example['canonical']}** ({len(example['members'])} members)\n"
                for member in example['members']:
                    report += f"   - {member}\n"
                if example.get('notes'):
                    report += f"   - *Issue:* {example['notes']}\n"
                report += "\n"

        # False negatives
        if examples['false_negatives']:
            report += "## False Negatives (Missed Matches)\n\n"
            for i, example in enumerate(examples['false_negatives'][:10], 1):
                report += f"{i}. **{example['intervention']}**\n"
                report += f"   - Should cluster with: {example['should_cluster_with']}\n"
                if example.get('notes'):
                    report += f"   - *Notes:* {example['notes']}\n"
                report += "\n"

        # Edge cases
        if examples['edge_cases']:
            report += "## Edge Cases\n\n"
            for i, example in enumerate(examples['edge_cases'][:10], 1):
                report += f"{i}. **{example['canonical']}** ({example.get('case_type', 'unknown')})\n"
                for member in example['members']:
                    report += f"   - {member}\n"
                if example.get('notes'):
                    report += f"   - *Notes:* {example['notes']}\n"
                report += "\n"

        # Findings
        findings = self.experiment_log['findings']

        if findings['what_worked']:
            report += "## What Worked\n\n"
            for finding in findings['what_worked']:
                report += f"- {finding}\n"
            report += "\n"

        if findings['what_failed']:
            report += "## What Failed\n\n"
            for finding in findings['what_failed']:
                report += f"- {finding}\n"
            report += "\n"

        if findings['hypotheses']:
            report += "## Hypotheses (Why Failures Occurred)\n\n"
            for hypothesis in findings['hypotheses']:
                report += f"- {hypothesis}\n"
            report += "\n"

        if findings['ideas_for_next_iteration']:
            report += "## Ideas for Next Iteration\n\n"
            for idea in findings['ideas_for_next_iteration']:
                report += f"- {idea}\n"
            report += "\n"

        return report

    def save_markdown_report(self) -> str:
        """
        Save markdown report to file.

        Returns:
            Path to saved report file
        """
        report = self.generate_markdown_report()

        output_file = os.path.join(
            self.experiments_dir,
            f"experiment_{self.experiment_id}.md"
        )

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        return output_file


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Experiment Logger")
    parser.add_argument(
        '--test-results',
        help='Path to test results JSON'
    )
    parser.add_argument(
        '--cluster-review',
        help='Path to cluster review JSON'
    )
    parser.add_argument(
        '--output-dir',
        default='c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/experiments',
        help='Output directory for experiment logs'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode for adding findings'
    )

    args = parser.parse_args()

    # Create logger
    logger = ExperimentLogger(experiments_dir=args.output_dir)

    # Import data
    if args.test_results and os.path.exists(args.test_results):
        print(f"Importing test results from: {args.test_results}")
        logger.import_from_test_results(args.test_results)

    if args.cluster_review and os.path.exists(args.cluster_review):
        print(f"Importing cluster review from: {args.cluster_review}")
        logger.import_from_cluster_review(args.cluster_review)

    # Interactive mode
    if args.interactive:
        print("\n=== INTERACTIVE FINDINGS ===")
        print("Add findings to your experiment log. Press Enter with empty input to skip a category.\n")

        # What worked
        print("What worked well in this experiment?")
        while True:
            finding = input("  - ").strip()
            if not finding:
                break
            logger.add_finding('what_worked', finding)

        # What failed
        print("\nWhat failed or didn't work as expected?")
        while True:
            finding = input("  - ").strip()
            if not finding:
                break
            logger.add_finding('what_failed', finding)

        # Hypotheses
        print("\nHypotheses about why failures occurred:")
        while True:
            finding = input("  - ").strip()
            if not finding:
                break
            logger.add_finding('hypotheses', finding)

        # Ideas
        print("\nIdeas for next iteration:")
        while True:
            finding = input("  - ").strip()
            if not finding:
                break
            logger.add_finding('ideas_for_next_iteration', finding)

    # Save logs
    json_path = logger.save_log()
    md_path = logger.save_markdown_report()

    print(f"\nExperiment log saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
