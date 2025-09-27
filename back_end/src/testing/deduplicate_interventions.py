#!/usr/bin/env python3
"""
Intervention Deduplication Script

Removes duplicate interventions from the database while preserving the best quality version.
Handles cases where both LLM models extracted the same intervention.
"""

import logging
from typing import List, Dict, Tuple
from back_end.src.data.config import config, setup_logging
from back_end.src.data_collection.database_manager import database_manager

logger = setup_logging(__name__, 'deduplication.log')


class InterventionDeduplicator:
    """Handles deduplication of interventions extracted by multiple models."""

    def __init__(self):
        # No longer need to store db_path since we use database_manager
        pass

    def find_duplicates(self) -> List[Tuple]:
        """Find potential duplicate interventions."""
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Find interventions with same name and health condition from same paper
            cursor.execute("""
                SELECT
                    paper_id,
                    intervention_name,
                    health_condition,
                    COUNT(*) as duplicate_count,
                    GROUP_CONCAT(id) as intervention_ids,
                    GROUP_CONCAT(extraction_model) as models,
                    GROUP_CONCAT(confidence_score) as confidences
                FROM interventions
                GROUP BY paper_id, intervention_name, health_condition
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
            """)

            duplicates = cursor.fetchall()

        logger.info(f"Found {len(duplicates)} groups of duplicate interventions")
        return duplicates

    def resolve_duplicate(self, duplicate_group: Tuple) -> int:
        """
        Resolve a duplicate by keeping the best quality intervention.

        Returns the ID of the intervention to keep.
        """
        paper_id, intervention_name, health_condition, count, ids_str, models_str, confidences_str = duplicate_group

        ids = [int(x) for x in ids_str.split(',')]
        models = models_str.split(',')
        confidences = [float(x) if x != 'None' else 0.0 for x in confidences_str.split(',')]

        # Strategy: Keep the intervention with highest confidence score
        # If tied, prefer qwen2.5:14b based on our quality analysis
        best_idx = 0
        best_confidence = confidences[0]

        for i, confidence in enumerate(confidences):
            if confidence > best_confidence:
                best_confidence = confidence
                best_idx = i
            elif confidence == best_confidence and models[i] == 'qwen2.5:14b':
                best_idx = i

        keep_id = ids[best_idx]
        remove_ids = [id for i, id in enumerate(ids) if i != best_idx]

        logger.info(f"Duplicate: {intervention_name} for {health_condition}")
        logger.info(f"  Keeping: ID {keep_id} from {models[best_idx]} (confidence: {confidences[best_idx]})")
        logger.info(f"  Removing: IDs {remove_ids}")

        return keep_id, remove_ids

    def merge_intervention_metadata(self, keep_id: int, remove_ids: List[int]):
        """Merge metadata from duplicate interventions into the kept one."""
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get all models that contributed to this intervention
            cursor.execute("""
                SELECT GROUP_CONCAT(DISTINCT extraction_model) as all_models,
                       COUNT(*) as total_extractions
                FROM interventions
                WHERE id IN ({})
            """.format(','.join(['?'] * (len(remove_ids) + 1))), [keep_id] + remove_ids)

            all_models, total_extractions = cursor.fetchone()

            # Update the kept intervention with consensus metadata
            cursor.execute("""
                UPDATE interventions
                SET models_contributing = ?,
                    raw_extraction_count = ?,
                    model_agreement = 'consensus',
                    consensus_confidence = (
                        SELECT AVG(confidence_score)
                        FROM interventions
                        WHERE id IN ({})
                        AND confidence_score IS NOT NULL
                    )
                WHERE id = ?
            """.format(','.join(['?'] * (len(remove_ids) + 1))),
                          [all_models, total_extractions] + [keep_id] + remove_ids + [keep_id])

            conn.commit()

    def remove_duplicates(self, remove_ids: List[int]):
        """Remove duplicate intervention records."""
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM interventions
                WHERE id IN ({})
            """.format(','.join(['?'] * len(remove_ids))), remove_ids)

            removed_count = cursor.rowcount
            conn.commit()

        logger.info(f"Removed {removed_count} duplicate interventions")
        return removed_count

    def deduplicate_all(self) -> Dict[str, int]:
        """Run complete deduplication process."""
        logger.info("Starting intervention deduplication process")

        # Get initial counts
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM interventions")
            initial_count = cursor.fetchone()[0]

        duplicates = self.find_duplicates()

        total_removed = 0
        total_groups = len(duplicates)

        for duplicate_group in duplicates:
            try:
                keep_id, remove_ids = self.resolve_duplicate(duplicate_group)
                self.merge_intervention_metadata(keep_id, remove_ids)
                removed_count = self.remove_duplicates(remove_ids)
                total_removed += removed_count

            except Exception as e:
                logger.error(f"Error processing duplicate group: {e}")
                continue

        # Get final counts
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM interventions")
            final_count = cursor.fetchone()[0]

        results = {
            'initial_count': initial_count,
            'final_count': final_count,
            'duplicate_groups': total_groups,
            'interventions_removed': total_removed,
            'deduplication_rate': (total_removed / initial_count * 100) if initial_count > 0 else 0
        }

        logger.info(f"Deduplication complete:")
        logger.info(f"  Initial interventions: {initial_count}")
        logger.info(f"  Final interventions: {final_count}")
        logger.info(f"  Duplicate groups resolved: {total_groups}")
        logger.info(f"  Interventions removed: {total_removed}")
        logger.info(f"  Deduplication rate: {results['deduplication_rate']:.1f}%")

        return results


def main():
    """Run the deduplication process."""
    deduplicator = InterventionDeduplicator()
    results = deduplicator.deduplicate_all()

    print("DEDUPLICATION RESULTS:")
    print(f"  Initial interventions: {results['initial_count']}")
    print(f"  Final interventions: {results['final_count']}")
    print(f"  Duplicate groups: {results['duplicate_groups']}")
    print(f"  Interventions removed: {results['interventions_removed']}")
    print(f"  Deduplication rate: {results['deduplication_rate']:.1f}%")


if __name__ == "__main__":
    main()