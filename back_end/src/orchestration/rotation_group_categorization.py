"""
Phase 3.5: Group-Based Categorization Orchestrator

Categorizes canonical groups created by Phase 3 (Semantic Normalization),
then propagates categories to interventions.

Workflow:
1. Phase 3 creates canonical groups (e.g., "probiotics", "statins")
2. Phase 3.5 categorizes these groups using LLM with semantic context
3. Categories propagate to member interventions via UPDATE-JOIN
4. Orphan interventions (not in groups) get fallback categorization

Usage:
    # Standalone
    python -m back_end.src.orchestration.rotation_group_categorization

    # In pipeline (after Phase 3)
    from back_end.src.orchestration.rotation_group_categorization import RotationGroupCategorizer
    categorizer = RotationGroupCategorizer()
    stats = categorizer.run()
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from back_end.src.data.config import config, setup_logging
from back_end.src.experimentation.group_categorization.group_categorizer import GroupBasedCategorizer
from back_end.src.experimentation.group_categorization.validation import validate_all

logger = setup_logging(__name__)


class RotationGroupCategorizer:
    """
    Phase 3.5: Categorizes canonical groups and propagates to interventions.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        batch_size: int = 20,
        include_members: bool = True,
        max_members_in_prompt: int = 10,
        validate_results: bool = True
    ):
        """
        Initialize Phase 3.5 categorizer.

        Args:
            db_path: Database path (uses config default if None)
            batch_size: Groups per LLM call
            include_members: Include member names in prompt for context
            max_members_in_prompt: Max members to include
            validate_results: Run validation after categorization
        """
        self.db_path = db_path or config.db_path
        self.batch_size = batch_size
        self.include_members = include_members
        self.max_members_in_prompt = max_members_in_prompt
        self.validate_results = validate_results

        logger.info("Phase 3.5: Group-Based Categorization initialized")

    def run(self, condition: Optional[str] = None) -> Dict:
        """
        Run Phase 3.5: Group-based categorization.

        Args:
            condition: Optional condition filter (for single-condition runs)

        Returns:
            Statistics dict
        """
        logger.info("=" * 80)
        logger.info("PHASE 3.5: GROUP-BASED CATEGORIZATION")
        logger.info("=" * 80)

        start_time = time.time()

        # Initialize categorizer
        categorizer = GroupBasedCategorizer(
            db_path=self.db_path,
            batch_size=self.batch_size,
            include_members=self.include_members,
            max_members_in_prompt=self.max_members_in_prompt
        )

        # Step 1: Categorize canonical groups
        logger.info("\nStep 1: Categorizing canonical groups")
        logger.info("-" * 80)

        group_stats = categorizer.categorize_all_groups()
        logger.info(f"Group categorization complete:")
        logger.info(f"  Total groups: {group_stats['total']}")
        logger.info(f"  Successfully categorized: {group_stats['processed']}")
        logger.info(f"  Failed: {group_stats['failed']}")
        logger.info(f"  LLM calls: {group_stats['llm_calls']}")

        # Step 2: Propagate categories to interventions (Option A)
        logger.info("\nStep 2: Propagating categories to interventions")
        logger.info("-" * 80)

        propagate_stats = categorizer.propagate_to_interventions()
        logger.info(f"Category propagation complete:")
        logger.info(f"  Interventions updated: {propagate_stats['updated']}")
        logger.info(f"  Orphan interventions: {propagate_stats['orphans']}")

        # Step 3: Handle orphan interventions (fallback)
        logger.info("\nStep 3: Categorizing orphan interventions (fallback)")
        logger.info("-" * 80)

        orphan_stats = categorizer.categorize_orphan_interventions()
        logger.info(f"Orphan categorization complete:")
        logger.info(f"  Total orphans: {orphan_stats['total']}")
        logger.info(f"  Successfully categorized: {orphan_stats['processed']}")
        logger.info(f"  Failed: {orphan_stats['failed']}")

        # Step 4: Validation (optional)
        validation_results = None
        if self.validate_results:
            logger.info("\nStep 4: Validating results")
            logger.info("-" * 80)

            validation_results = validate_all(self.db_path)

            logger.info(f"Validation complete:")
            logger.info(f"  Coverage: {validation_results['coverage']['coverage_rate']*100:.1f}% - {'PASSED' if validation_results['coverage']['passed'] else 'FAILED'}")
            logger.info(f"  Purity: {validation_results['purity']['purity_rate']*100:.1f}% - {'PASSED' if validation_results['purity']['passed'] else 'FAILED'}")
            if validation_results['comparison']['agreement_rate'] is not None:
                logger.info(f"  Agreement: {validation_results['comparison']['agreement_rate']*100:.1f}% - {'PASSED' if validation_results['comparison']['passed'] else 'FAILED'}")
            logger.info(f"  Overall: {'PASSED' if validation_results['all_passed'] else 'FAILED'}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Compile results
        results = {
            'phase': '3.5',
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_seconds': elapsed_time,
            'group_categorization': group_stats,
            'propagation': propagate_stats,
            'orphan_categorization': orphan_stats,
            'validation': validation_results,
            'performance': {
                'total_llm_calls': group_stats['llm_calls'] + orphan_stats.get('llm_calls', 0),
                'time_per_llm_call': elapsed_time / max(group_stats['llm_calls'], 1)
            }
        }

        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3.5 COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed_time:.1f} seconds")
        logger.info(f"Total LLM calls: {results['performance']['total_llm_calls']}")
        logger.info(f"Groups categorized: {group_stats['processed']}/{group_stats['total']}")
        logger.info(f"Interventions updated: {propagate_stats['updated']}")
        logger.info(f"Orphans handled: {orphan_stats['processed']}/{orphan_stats['total']}")

        if self.validate_results and validation_results:
            logger.info(f"Validation: {'PASSED ✓' if validation_results['all_passed'] else 'FAILED ✗'}")

        return results

    def get_status(self) -> Dict:
        """
        Get current Phase 3.5 status (how many groups categorized).

        Returns:
            Status dict
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Groups status
        cursor.execute("""
            SELECT COUNT(*)
            FROM canonical_groups
            WHERE entity_type = 'intervention'
        """)
        total_groups = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*)
            FROM canonical_groups
            WHERE entity_type = 'intervention'
            AND layer_0_category IS NOT NULL
            AND layer_0_category != ''
        """)
        categorized_groups = cursor.fetchone()[0]

        # Interventions status
        cursor.execute("""
            SELECT COUNT(DISTINCT intervention_name)
            FROM interventions
        """)
        total_interventions = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(DISTINCT intervention_name)
            FROM interventions
            WHERE intervention_category IS NOT NULL
            AND intervention_category != ''
        """)
        categorized_interventions = cursor.fetchone()[0]

        # Orphans
        cursor.execute("""
            SELECT COUNT(DISTINCT intervention_name)
            FROM interventions
            WHERE intervention_name NOT IN (
                SELECT entity_name
                FROM semantic_hierarchy
                WHERE entity_type = 'intervention'
            )
        """)
        orphan_interventions = cursor.fetchone()[0]

        conn.close()

        status = {
            'groups': {
                'total': total_groups,
                'categorized': categorized_groups,
                'remaining': total_groups - categorized_groups,
                'progress': categorized_groups / total_groups if total_groups > 0 else 0
            },
            'interventions': {
                'total': total_interventions,
                'categorized': categorized_interventions,
                'orphans': orphan_interventions,
                'progress': categorized_interventions / total_interventions if total_interventions > 0 else 0
            }
        }

        return status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3.5: Group-Based Categorization")
    parser.add_argument("--condition", type=str, help="Process single condition")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for LLM calls")
    parser.add_argument("--no-validation", action="store_true", help="Skip validation")
    parser.add_argument("--status", action="store_true", help="Show current status")

    args = parser.parse_args()

    categorizer = RotationGroupCategorizer(
        batch_size=args.batch_size,
        validate_results=not args.no_validation
    )

    if args.status:
        status = categorizer.get_status()
        print("\n" + "=" * 80)
        print("PHASE 3.5 STATUS")
        print("=" * 80)
        print(f"\nCanonical Groups:")
        print(f"  Total: {status['groups']['total']}")
        print(f"  Categorized: {status['groups']['categorized']} ({status['groups']['progress']*100:.1f}%)")
        print(f"  Remaining: {status['groups']['remaining']}")
        print(f"\nInterventions:")
        print(f"  Total: {status['interventions']['total']}")
        print(f"  Categorized: {status['interventions']['categorized']} ({status['interventions']['progress']*100:.1f}%)")
        print(f"  Orphans: {status['interventions']['orphans']}")
    else:
        results = categorizer.run(condition=args.condition)
        print(f"\n{json.dumps(results, indent=2)}")
