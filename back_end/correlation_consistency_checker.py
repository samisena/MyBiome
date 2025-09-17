#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Consistency Checker - Identify logical inconsistencies in intervention correlations
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re

# Add the src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from src.paper_collection.database_manager import database_manager


class CorrelationConsistencyChecker:
    """Check for logical inconsistencies in intervention correlations."""

    def __init__(self):
        self.db = database_manager

    def get_all_interventions(self) -> List[Dict]:
        """Get all interventions from database."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, paper_id, intervention_name, health_condition,
                       correlation_type, extraction_model, confidence_score
                FROM interventions
                ORDER BY paper_id, intervention_name
            ''')

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def find_logical_inconsistencies(self) -> List[Dict]:
        """Find logical inconsistencies in correlations."""
        interventions = self.get_all_interventions()
        inconsistencies = []

        # Group by paper and condition
        paper_condition_groups = {}

        for intervention in interventions:
            paper_id = intervention['paper_id']
            condition = intervention['health_condition'].lower().strip()
            key = f"{paper_id}_{condition}"

            if key not in paper_condition_groups:
                paper_condition_groups[key] = []
            paper_condition_groups[key].append(intervention)

        # Check each group for logical inconsistencies
        for group_key, group_interventions in paper_condition_groups.items():
            inconsistency = self._check_group_consistency(group_interventions)
            if inconsistency:
                inconsistencies.append(inconsistency)

        return inconsistencies

    def _check_group_consistency(self, interventions: List[Dict]) -> Dict:
        """Check a group of interventions for logical consistency."""
        # Look for opposite intervention pairs
        opposite_pairs = [
            (['gluten-free', 'gluten free'], ['gluten', 'wheat']),
            (['dairy-free', 'dairy free'], ['dairy', 'milk']),
            (['sugar-free', 'sugar free'], ['sugar']),
            (['low-carb', 'low carb'], ['high-carb', 'high carb']),
            (['low-fat', 'low fat'], ['high-fat', 'high fat']),
            (['caffeine-free'], ['caffeine']),
        ]

        inconsistencies = []

        for positive_terms, negative_terms in opposite_pairs:
            positive_interventions = []
            negative_interventions = []

            for intervention in interventions:
                name = intervention['intervention_name'].lower()

                # Check if intervention matches positive terms (should help)
                if any(term in name for term in positive_terms):
                    positive_interventions.append(intervention)

                # Check if intervention matches negative terms (should worsen or be neutral)
                if any(term in name for term in negative_terms):
                    negative_interventions.append(intervention)

            # Check for logical inconsistency
            if positive_interventions and negative_interventions:
                inconsistency = self._analyze_opposite_pair_consistency(
                    positive_interventions, negative_interventions, positive_terms, negative_terms
                )
                if inconsistency:
                    inconsistencies.append(inconsistency)

        if inconsistencies:
            return {
                'paper_id': interventions[0]['paper_id'],
                'condition': interventions[0]['health_condition'],
                'inconsistencies': inconsistencies
            }

        return None

    def _analyze_opposite_pair_consistency(self, positive_interventions: List[Dict],
                                         negative_interventions: List[Dict],
                                         positive_terms: List[str],
                                         negative_terms: List[str]) -> Dict:
        """Analyze if opposite intervention pairs have consistent correlations."""
        # Expected logic:
        # If "gluten-free diet" helps with IBS (positive), then "gluten" should worsen IBS (negative)
        # If "gluten-free diet" worsens IBS (negative), then "gluten" should help with IBS (positive)

        issues = []

        for pos_intervention in positive_interventions:
            for neg_intervention in negative_interventions:
                pos_correlation = pos_intervention['correlation_type']
                neg_correlation = neg_intervention['correlation_type']

                # Check for logical inconsistency
                if pos_correlation == 'positive' and neg_correlation == 'positive':
                    issues.append({
                        'type': 'both_positive',
                        'description': f"Both '{pos_intervention['intervention_name']}' and '{neg_intervention['intervention_name']}' are marked as positive (helping). This is logically inconsistent.",
                        'positive_intervention': pos_intervention,
                        'negative_intervention': neg_intervention
                    })

                elif pos_correlation == 'negative' and neg_correlation == 'negative':
                    issues.append({
                        'type': 'both_negative',
                        'description': f"Both '{pos_intervention['intervention_name']}' and '{neg_intervention['intervention_name']}' are marked as negative (worsening). This is logically inconsistent.",
                        'positive_intervention': pos_intervention,
                        'negative_intervention': neg_intervention
                    })

                elif pos_correlation in ['neutral', 'inconclusive'] and neg_correlation in ['neutral', 'inconclusive']:
                    # This is acceptable - both could be neutral
                    continue

                # Check if they agree when they should disagree
                elif pos_correlation == neg_correlation and pos_correlation in ['positive', 'negative']:
                    issues.append({
                        'type': 'unexpected_agreement',
                        'description': f"'{pos_intervention['intervention_name']}' and '{neg_intervention['intervention_name']}' both show {pos_correlation} correlation. They should likely have opposite effects.",
                        'positive_intervention': pos_intervention,
                        'negative_intervention': neg_intervention
                    })

        if issues:
            return {
                'positive_terms': positive_terms,
                'negative_terms': negative_terms,
                'issues': issues
            }

        return None

    def generate_report(self) -> str:
        """Generate a detailed consistency report."""
        inconsistencies = self.find_logical_inconsistencies()

        if not inconsistencies:
            return "No logical inconsistencies found in correlation data!"

        report = [f"Found {len(inconsistencies)} papers with logical inconsistencies:\n"]

        for i, paper_inconsistency in enumerate(inconsistencies, 1):
            paper_id = paper_inconsistency['paper_id']
            condition = paper_inconsistency['condition']

            report.append(f"{i}. Paper {paper_id} - {condition}")
            report.append("=" * 60)

            for inconsistency in paper_inconsistency['inconsistencies']:
                positive_terms = ', '.join(inconsistency['positive_terms'])
                negative_terms = ', '.join(inconsistency['negative_terms'])

                report.append(f"\nOpposite intervention pair: [{positive_terms}] vs [{negative_terms}]")

                for issue in inconsistency['issues']:
                    pos_int = issue['positive_intervention']
                    neg_int = issue['negative_intervention']

                    report.append(f"\n{issue['type'].upper()}:")
                    report.append(f"   {issue['description']}")
                    report.append(f"   - {pos_int['intervention_name']}: {pos_int['correlation_type']} (Model: {pos_int['extraction_model']})")
                    report.append(f"   - {neg_int['intervention_name']}: {neg_int['correlation_type']} (Model: {neg_int['extraction_model']})")

                    if pos_int['extraction_model'] != neg_int['extraction_model']:
                        report.append(f"   Note: Different models extracted these - may indicate model disagreement")

            report.append("\n" + "=" * 60 + "\n")

        # Add suggestions
        report.append("SUGGESTIONS:")
        report.append("- Review the abstract to determine correct correlation directions")
        report.append("- Consider if different study contexts could explain apparent contradictions")
        report.append("- Check if different models have different interpretations")
        report.append("- Update correlation types to maintain logical consistency")

        return "\n".join(report)

    def fix_gluten_inconsistency(self, paper_id: str = "40706614"):
        """Fix the specific gluten/gluten-free inconsistency."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get current correlations for the paper
            cursor.execute('''
                SELECT id, intervention_name, correlation_type, extraction_model
                FROM interventions
                WHERE paper_id = ? AND health_condition LIKE '%IBS%'
                ORDER BY intervention_name
            ''', (paper_id,))

            interventions = cursor.fetchall()

            print(f"Current correlations for paper {paper_id}:")
            for intervention in interventions:
                print(f"  {intervention[1]}: {intervention[2]} (Model: {intervention[3]})")

            print("\nLogical fix suggestions:")
            print("  If gluten-free diet HELPS with IBS (positive):")
            print("    -> gluten should WORSEN IBS (negative)")
            print("    -> wheat should WORSEN IBS (negative)")

            # Show what needs to be updated
            fixes_needed = []
            for intervention in interventions:
                name = intervention[1].lower()
                if 'gluten-free' in name and intervention[2] != 'positive':
                    fixes_needed.append((intervention[0], 'positive', f"'{intervention[1]}' should be positive (helps with IBS)"))
                elif ('gluten' in name and 'gluten-free' not in name) or 'wheat' in name:
                    if intervention[2] != 'negative':
                        fixes_needed.append((intervention[0], 'negative', f"'{intervention[1]}' should be negative (worsens IBS)"))

            if fixes_needed:
                print(f"\nSuggested fixes for {len(fixes_needed)} interventions:")
                for intervention_id, new_correlation, description in fixes_needed:
                    print(f"  ID {intervention_id}: {description}")
            else:
                print("\nNo fixes needed - correlations are already consistent!")


def main():
    """Run the consistency checker."""
    print("Correlation Consistency Checker")
    print("=" * 50)

    checker = CorrelationConsistencyChecker()

    # Generate and print report
    report = checker.generate_report()
    print(report)

    # Check specific gluten inconsistency
    print("\n" + "=" * 50)
    print("SPECIFIC GLUTEN INCONSISTENCY ANALYSIS")
    print("=" * 50)
    checker.fix_gluten_inconsistency()


if __name__ == "__main__":
    main()