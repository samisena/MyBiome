"""
Stage 3.5: Functional Grouping

Detects cross-category hierarchical merges and creates functional/therapeutic categories.
Runs after Stage 3 (LLM Validation) to identify functional groups from cross-category parents.

Example:
- Parent: "Gut Microbiome Modulation"
  Children: ["Probiotics" (supplement), "FMT" (procedure)]
  → Functional Category: "Gut Flora Modulators"

- Parent: "GERD Symptom Management"
  Children: ["Antacids" (medication), "LES Surgery" (surgery)]
  → Therapeutic Category: "GERD Treatments"
"""

import json
import logging
import sqlite3
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from stage_3_llm_validation import LLMValidationResult
from config import Phase3dConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class FunctionalGroup:
    """A functional category spanning multiple primary categories."""
    functional_category_name: str
    category_type: str  # 'functional' or 'therapeutic'
    parent_cluster_id: int
    parent_cluster_name: str
    member_cluster_ids: List[int]
    member_cluster_names: List[str]
    primary_categories_spanned: List[str]
    confidence: float
    llm_reasoning: str
    mechanism_similarity: Optional[float] = None


class FunctionalGrouper:
    """
    Creates functional/therapeutic categories from cross-category hierarchical merges.

    Workflow:
    1. Detect cross-category parents (children from different primary categories)
    2. Use LLM to suggest functional category name
    3. Assign functional categories to all member interventions
    4. Store in intervention_category_mapping junction table
    """

    def __init__(self, config: Phase3dConfig = None, db_path: str = None):
        """
        Initialize functional grouper.

        Args:
            config: Configuration object
            db_path: Path to intervention_research.db
        """
        self.config = config or get_config()
        self.db_path = db_path or "back_end/data/intervention_research.db"
        self.llm_url = f"{self.config.llm_base_url}/api/generate"

        logger.info("Initialized FunctionalGrouper")

    def detect_cross_category_groups(
        self,
        approved_merges: List[LLMValidationResult],
        db_conn: sqlite3.Connection
    ) -> List[FunctionalGroup]:
        """
        Identify functional groups from cross-category approved merges.

        Args:
            approved_merges: List of approved merge results from Stage 3
            db_conn: Database connection

        Returns:
            List of FunctionalGroup objects
        """
        logger.info("Detecting cross-category functional groups...")

        functional_groups = []
        cross_category_count = 0

        for i, merge in enumerate(approved_merges):
            # Only process CREATE_PARENT merges (not MERGE_IDENTICAL)
            if merge.relationship_type != 'CREATE_PARENT':
                continue

            # Get primary categories for child clusters
            child_ids = [merge.candidate.cluster_a.cluster_id, merge.candidate.cluster_b.cluster_id]
            primary_categories = self._get_cluster_primary_categories(child_ids, db_conn)

            # Check if categories span multiple primary categories
            unique_categories = set(primary_categories.values())

            if len(unique_categories) > 1:
                cross_category_count += 1

                # This is a cross-category merge - create functional group
                logger.info(f"\n  Cross-category merge detected ({cross_category_count}):")
                logger.info(f"    Parent: {merge.suggested_parent_name}")
                logger.info(f"    Children: {[c.canonical_name for c in [merge.candidate.cluster_a, merge.candidate.cluster_b]]}")
                logger.info(f"    Primary Categories: {list(unique_categories)}")

                # Suggest functional category name
                functional_name, category_type, reasoning = self._suggest_functional_name(
                    parent_name=merge.suggested_parent_name,
                    child_names=[merge.candidate.cluster_a.canonical_name, merge.candidate.cluster_b.canonical_name],
                    primary_categories=list(unique_categories)
                )

                if functional_name:
                    functional_group = FunctionalGroup(
                        functional_category_name=functional_name,
                        category_type=category_type,
                        parent_cluster_id=-1,  # Parent not yet created (will be in Stage 5)
                        parent_cluster_name=merge.suggested_parent_name,
                        member_cluster_ids=child_ids,
                        member_cluster_names=[merge.candidate.cluster_a.canonical_name,
                                             merge.candidate.cluster_b.canonical_name],
                        primary_categories_spanned=list(unique_categories),
                        confidence=merge.candidate.similarity,
                        llm_reasoning=reasoning
                    )

                    functional_groups.append(functional_group)
                    logger.info(f"    → Functional Category: {functional_name} ({category_type})")

        logger.info(f"\nDetected {len(functional_groups)} cross-category functional groups")
        return functional_groups

    def _get_cluster_primary_categories(
        self,
        cluster_ids: List[int],
        db_conn: sqlite3.Connection
    ) -> Dict[int, str]:
        """
        Get primary intervention category for each cluster.

        Returns:
            Dict mapping cluster_id -> primary_category
        """
        cursor = db_conn.cursor()

        # For mechanisms, we need to look up interventions that use these mechanisms
        # For now, return empty dict (will implement when mechanism_clusters table exists)

        # TODO: Implement actual category lookup from intervention_category_mapping
        # For now, return dummy data for testing

        return {cid: f"category_{cid % 3}" for cid in cluster_ids}

    def _suggest_functional_name(
        self,
        parent_name: str,
        child_names: List[str],
        primary_categories: List[str]
    ) -> Tuple[str, str, str]:
        """
        Use LLM to suggest functional category name.

        Args:
            parent_name: Parent cluster name
            child_names: Child cluster names
            primary_categories: Primary categories being merged

        Returns:
            Tuple of (functional_name, category_type, reasoning)
        """
        prompt = f"""You are analyzing a hierarchical merge that spans multiple intervention categories.

PARENT CLUSTER: {parent_name}

CHILD CLUSTERS:
{chr(10).join(f'- {name}' for name in child_names)}

PRIMARY CATEGORIES SPANNED:
{chr(10).join(f'- {cat}' for cat in primary_categories)}

Your task: Suggest a FUNCTIONAL category name that describes what these interventions DO (not what they ARE).

Guidelines:
1. FUNCTIONAL categories describe shared mechanism/function (e.g., "Gut Flora Modulators", "Pain Relievers")
2. THERAPEUTIC categories describe condition-specific treatment groups (e.g., "GERD Treatments", "Diabetes Management")
3. Be specific but not overly technical
4. Use active language ("Modulators", "Enhancers", "Reducers" not "Modulation", "Enhancement")
5. Keep it concise (2-4 words)

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "functional_name": "suggested name",
    "category_type": "functional" or "therapeutic",
    "reasoning": "brief explanation (1-2 sentences)"
}}"""

        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.config.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": self.config.llm_temperature,
                        "num_predict": 200
                    }
                },
                timeout=10  # Reduced timeout for testing
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()

                # Parse JSON response
                parsed = json.loads(response_text)

                functional_name = parsed.get('functional_name', '')
                category_type = parsed.get('category_type', 'functional')
                reasoning = parsed.get('reasoning', '')

                # Validate category_type
                if category_type not in ['functional', 'therapeutic']:
                    category_type = 'functional'

                return functional_name, category_type, reasoning

            else:
                logger.error(f"LLM request failed: {response.status_code}")
                return "", "functional", ""

        except requests.exceptions.Timeout:
            logger.warning("LLM request timed out (Ollama may not be running)")
            return "", "functional", ""
        except requests.exceptions.ConnectionError:
            logger.warning("Could not connect to LLM (Ollama may not be running)")
            return "", "functional", ""
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return "", "functional", ""
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return "", "functional", ""

    def apply_functional_categories(
        self,
        functional_groups: List[FunctionalGroup],
        db_conn: sqlite3.Connection
    ) -> Dict[str, int]:
        """
        Apply functional categories to junction tables.

        Args:
            functional_groups: List of FunctionalGroup objects
            db_conn: Database connection

        Returns:
            Dict with statistics
        """
        logger.info("\nApplying functional categories to database...")

        cursor = db_conn.cursor()
        stats = {
            'functional_categories_created': 0,
            'therapeutic_categories_created': 0,
            'interventions_assigned': 0,
            'mechanisms_assigned': 0
        }

        for group in functional_groups:
            logger.info(f"\n  Applying: {group.functional_category_name} ({group.category_type})")

            # Get all interventions associated with these mechanism clusters
            intervention_ids = self._get_interventions_for_clusters(
                group.member_cluster_ids,
                db_conn
            )

            if not intervention_ids:
                logger.warning(f"    No interventions found for clusters {group.member_cluster_ids}")
                continue

            logger.info(f"    Found {len(intervention_ids)} interventions to assign")

            # Assign functional category to all interventions
            for intervention_id in intervention_ids:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO intervention_category_mapping
                        (intervention_id, category_type, category_name, confidence, assigned_by, notes)
                        VALUES (?, ?, ?, ?, 'phase_3d_functional', ?)
                    """, (
                        intervention_id,
                        group.category_type,
                        group.functional_category_name,
                        group.confidence,
                        f"Parent: {group.parent_cluster_name}"
                    ))

                    if cursor.rowcount > 0:
                        stats['interventions_assigned'] += 1

                except Exception as e:
                    logger.error(f"    Error assigning category to intervention {intervention_id}: {e}")

            # Update statistics
            if group.category_type == 'functional':
                stats['functional_categories_created'] += 1
            else:
                stats['therapeutic_categories_created'] += 1

            db_conn.commit()

        logger.info("\nFunctional category application complete:")
        logger.info(f"  Functional categories: {stats['functional_categories_created']}")
        logger.info(f"  Therapeutic categories: {stats['therapeutic_categories_created']}")
        logger.info(f"  Interventions assigned: {stats['interventions_assigned']}")

        return stats

    def _get_interventions_for_clusters(
        self,
        cluster_ids: List[int],
        db_conn: sqlite3.Connection
    ) -> List[int]:
        """
        Get all intervention IDs associated with mechanism clusters.

        Args:
            cluster_ids: List of mechanism cluster IDs
            db_conn: Database connection

        Returns:
            List of intervention IDs
        """
        cursor = db_conn.cursor()

        # Check if intervention_mechanisms table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='intervention_mechanisms'
        """)

        if not cursor.fetchone():
            logger.warning("intervention_mechanisms table not found (Phase 3.6 not run)")
            return []

        # Get all interventions linked to these clusters
        placeholders = ','.join('?' * len(cluster_ids))
        cursor.execute(f"""
            SELECT DISTINCT intervention_id
            FROM intervention_mechanisms
            WHERE mechanism_cluster_id IN ({placeholders})
        """, cluster_ids)

        rows = cursor.fetchall()
        return [row[0] for row in rows]

    def save_functional_groups_report(
        self,
        functional_groups: List[FunctionalGroup],
        output_path: str = "functional_groups_report.json"
    ):
        """
        Save functional groups to JSON report for review.

        Args:
            functional_groups: List of FunctionalGroup objects
            output_path: Output file path
        """
        report = {
            'metadata': {
                'total_groups': len(functional_groups),
                'functional_count': sum(1 for g in functional_groups if g.category_type == 'functional'),
                'therapeutic_count': sum(1 for g in functional_groups if g.category_type == 'therapeutic')
            },
            'groups': [asdict(g) for g in functional_groups]
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"\nFunctional groups report saved: {output_file}")


def main():
    """Test functional grouping standalone."""
    import argparse

    parser = argparse.ArgumentParser(description='Stage 3.5: Functional Grouping')
    parser.add_argument('--db-path', default='back_end/data/intervention_research.db',
                       help='Path to database')
    parser.add_argument('--approved-merges', required=True,
                       help='Path to approved merges JSON from Stage 3')
    parser.add_argument('--output', default='functional_groups_report.json',
                       help='Output report path')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # Load approved merges
    logger.info(f"Loading approved merges from: {args.approved_merges}")

    # This would load actual LLMValidationResult objects
    # For now, just create demo report

    grouper = FunctionalGrouper(db_path=args.db_path)

    logger.info("\nStage 3.5: Functional Grouping")
    logger.info("="*60)
    logger.info("Note: Run this after Stage 3 (LLM Validation)")
    logger.info("This stage detects cross-category merges and creates functional categories")


if __name__ == '__main__':
    main()
