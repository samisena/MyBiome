"""
Stage 4: Cross-Category Detection

Detect merges across intervention/condition categories.
Generates report for investigating potential mis-categorizations.
"""

import json
import logging
import sqlite3
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from stage_3_llm_validation import LLMValidationResult

logger = logging.getLogger(__name__)


@dataclass
class CrossCategoryCase:
    """Represents a cross-category merge case."""
    parent_name: str
    cluster_a_id: int
    cluster_a_name: str
    cluster_a_category: Optional[str]
    cluster_a_members: List[str]
    cluster_b_id: int
    cluster_b_name: str
    cluster_b_category: Optional[str]
    cluster_b_members: List[str]
    similarity: float
    llm_reasoning: str
    suggestion: str  # Recategorization suggestion


class CrossCategoryDetector:
    """
    Detects and reports cross-category merges.

    These cases suggest potential mis-categorization in Phase 3.5.
    """

    def __init__(self, db_path: str, entity_type: str = 'mechanism'):
        """
        Initialize detector.

        Args:
            db_path: Path to database
            entity_type: 'mechanism', 'intervention', or 'condition'
        """
        self.db_path = db_path
        self.entity_type = entity_type

        logger.info(f"Initialized CrossCategoryDetector for {entity_type}")

    def detect_cross_category_merges(
        self,
        approved_merges: List[LLMValidationResult]
    ) -> List[CrossCategoryCase]:
        """
        Detect cross-category cases from approved merges.

        Args:
            approved_merges: List of auto-approved LLMValidationResult objects

        Returns:
            List of CrossCategoryCase objects
        """
        logger.info(f"Detecting cross-category merges in {len(approved_merges)} approved merges...")

        cross_category_cases = []

        for merge in approved_merges:
            candidate = merge.candidate
            cluster_a = candidate.cluster_a
            cluster_b = candidate.cluster_b

            # Get categories for both clusters
            category_a = self._get_cluster_category(cluster_a.cluster_id, cluster_a.members)
            category_b = self._get_cluster_category(cluster_b.cluster_id, cluster_b.members)

            # Check if different
            if category_a and category_b and category_a != category_b:
                # Create cross-category case
                case = CrossCategoryCase(
                    parent_name=merge.suggested_parent_name or f"Merged_{cluster_a.cluster_id}_{cluster_b.cluster_id}",
                    cluster_a_id=cluster_a.cluster_id,
                    cluster_a_name=cluster_a.canonical_name,
                    cluster_a_category=category_a,
                    cluster_a_members=cluster_a.members[:5],  # First 5 examples
                    cluster_b_id=cluster_b.cluster_id,
                    cluster_b_name=cluster_b.canonical_name,
                    cluster_b_category=category_b,
                    cluster_b_members=cluster_b.members[:5],
                    similarity=candidate.similarity,
                    llm_reasoning=merge.llm_reasoning,
                    suggestion=self._generate_suggestion(category_a, category_b, merge)
                )

                cross_category_cases.append(case)

        logger.info(f"  Detected {len(cross_category_cases)} cross-category cases")

        return cross_category_cases

    def _get_cluster_category(
        self,
        cluster_id: int,
        members: List[str]
    ) -> Optional[str]:
        """
        Get category for cluster.

        For mechanisms: check intervention_category of associated interventions
        For interventions/conditions: check from semantic_hierarchy

        Args:
            cluster_id: Cluster ID
            members: List of cluster members

        Returns:
            Category string or None
        """
        if self.entity_type == 'mechanism':
            return self._get_mechanism_category(members)
        elif self.entity_type == 'intervention':
            return self._get_intervention_category(members)
        elif self.entity_type == 'condition':
            return self._get_condition_category(members)
        else:
            return None

    def _get_mechanism_category(self, mechanisms: List[str]) -> Optional[str]:
        """
        Get category for mechanism cluster.

        Looks up intervention_category from interventions table.

        Args:
            mechanisms: List of mechanism texts

        Returns:
            Most common category or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        categories = []

        for mechanism in mechanisms[:10]:  # Sample up to 10
            cursor.execute("""
                SELECT intervention_category
                FROM interventions
                WHERE mechanism = ?
                  AND intervention_category IS NOT NULL
                LIMIT 1
            """, (mechanism,))

            row = cursor.fetchone()
            if row and row[0]:
                categories.append(row[0])

        conn.close()

        if not categories:
            return None

        # Return most common category
        from collections import Counter
        most_common = Counter(categories).most_common(1)[0][0]
        return most_common

    def _get_intervention_category(self, interventions: List[str]) -> Optional[str]:
        """
        Get category for intervention group.

        Args:
            interventions: List of intervention names

        Returns:
            Most common category or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        categories = []

        for intervention in interventions[:10]:
            cursor.execute("""
                SELECT intervention_category
                FROM interventions
                WHERE intervention_name = ?
                  AND intervention_category IS NOT NULL
                LIMIT 1
            """, (intervention,))

            row = cursor.fetchone()
            if row and row[0]:
                categories.append(row[0])

        conn.close()

        if not categories:
            return None

        from collections import Counter
        most_common = Counter(categories).most_common(1)[0][0]
        return most_common

    def _get_condition_category(self, conditions: List[str]) -> Optional[str]:
        """
        Get category for condition group.

        Args:
            conditions: List of condition names

        Returns:
            Most common category or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        categories = []

        for condition in conditions[:10]:
            cursor.execute("""
                SELECT condition_category
                FROM interventions
                WHERE health_condition = ?
                  AND condition_category IS NOT NULL
                LIMIT 1
            """, (condition,))

            row = cursor.fetchone()
            if row and row[0]:
                categories.append(row[0])

        conn.close()

        if not categories:
            return None

        from collections import Counter
        most_common = Counter(categories).most_common(1)[0][0]
        return most_common

    def _generate_suggestion(
        self,
        category_a: str,
        category_b: str,
        merge: LLMValidationResult
    ) -> str:
        """
        Generate recategorization suggestion.

        Args:
            category_a: Category of cluster A
            category_b: Category of cluster B
            merge: LLMValidationResult object

        Returns:
            Suggestion string
        """
        parent_name = merge.suggested_parent_name or "merged cluster"

        # Analyze LLM reasoning for hints
        reasoning_lower = merge.llm_reasoning.lower()

        # Common patterns
        if any(term in reasoning_lower for term in ['same', 'identical', 'equivalent']):
            return f"Consider standardizing both to '{category_a}' or '{category_b}'"

        if any(term in reasoning_lower for term in ['broader', 'parent', 'encompasses']):
            return f"Parent '{parent_name}' may need new category that encompasses both '{category_a}' and '{category_b}'"

        # Default suggestion
        return f"Review categorization: '{category_a}' vs '{category_b}' merged under '{parent_name}'"

    def generate_report(
        self,
        cases: List[CrossCategoryCase],
        output_path: str
    ):
        """
        Generate cross-category report as JSON.

        Args:
            cases: List of CrossCategoryCase objects
            output_path: Path to output file
        """
        if not cases:
            logger.info("No cross-category cases to report")
            return

        report_data = {
            'timestamp': datetime.now().isoformat(),
            'entity_type': self.entity_type,
            'total_cases': len(cases),
            'summary': self._generate_summary_stats(cases),
            'cases': [
                {
                    'parent_name': case.parent_name,
                    'cluster_a': {
                        'id': case.cluster_a_id,
                        'name': case.cluster_a_name,
                        'category': case.cluster_a_category,
                        'members_sample': case.cluster_a_members
                    },
                    'cluster_b': {
                        'id': case.cluster_b_id,
                        'name': case.cluster_b_name,
                        'category': case.cluster_b_category,
                        'members_sample': case.cluster_b_members
                    },
                    'similarity': float(case.similarity),
                    'llm_reasoning': case.llm_reasoning,
                    'suggestion': case.suggestion
                }
                for case in cases
            ],
            'note': (
                'These merges suggest potential mis-categorization in Phase 3.5. '
                'Review cases to determine if recategorization is needed.'
            )
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Cross-category report saved: {output_path}")
        logger.info(f"  Total cases: {len(cases)}")

        # Print summary
        self._print_summary(cases)

    def _generate_summary_stats(self, cases: List[CrossCategoryCase]) -> Dict:
        """Generate summary statistics."""
        from collections import Counter

        category_pairs = [
            tuple(sorted([case.cluster_a_category, case.cluster_b_category]))
            for case in cases
        ]

        pair_counts = Counter(category_pairs)

        return {
            'most_common_pairs': [
                {'categories': list(pair), 'count': count}
                for pair, count in pair_counts.most_common(10)
            ]
        }

    def _print_summary(self, cases: List[CrossCategoryCase]):
        """Print summary to console."""
        print("\n" + "="*80)
        print(f"CROSS-CATEGORY MERGE REPORT ({len(cases)} cases)")
        print("="*80)

        if not cases:
            print("No cross-category cases found.")
            return

        # Most common category pairs
        from collections import Counter

        category_pairs = [
            f"{case.cluster_a_category} + {case.cluster_b_category}"
            for case in cases
        ]

        pair_counts = Counter(category_pairs)

        print("\nMost Common Category Pairs:")
        for pair, count in pair_counts.most_common(10):
            print(f"  {pair}: {count} cases")

        # Sample cases
        print(f"\nSample Cases (first 3):")
        for i, case in enumerate(cases[:3], 1):
            print(f"\n  Case {i}:")
            print(f"    Parent: {case.parent_name}")
            print(f"    Categories: {case.cluster_a_category} + {case.cluster_b_category}")
            print(f"    Suggestion: {case.suggestion}")

        print("="*80)


if __name__ == "__main__":
    # Test cross-category detection
    logging.basicConfig(level=logging.INFO)

    print("Cross-category detection module ready.")
    print("Use detect_cross_category_merges() with approved merges to find cases.")
