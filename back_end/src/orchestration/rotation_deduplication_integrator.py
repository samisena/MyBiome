#!/usr/bin/env python3
"""
Rotation Deduplication Integrator

Integrates deduplication with the rotation pipeline.
Runs LLM-based deduplication after each condition is processed
to maintain data quality throughout the rotation cycle.

Features:
- Condition-specific deduplication
- Integration with rotation session manager
- Progress tracking and validation
- Error handling and recovery
- Performance optimization
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

try:
    from ..data.config import config, setup_logging
    from ..llm_processing.batch_entity_processor import create_batch_processor
    from ..data_collection.database_manager import database_manager
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.llm_processing.batch_entity_processor import create_batch_processor
    from back_end.src.data_collection.database_manager import database_manager

logger = setup_logging(__name__, 'rotation_deduplication_integrator.log')


class DeduplicationError(Exception):
    """Custom exception for deduplication errors."""
    pass


class RotationDeduplicationIntegrator:
    """
    Integrates deduplication with the rotation pipeline.
    Focuses on efficient, condition-specific deduplication.
    """

    def __init__(self):
        """Initialize the deduplication integrator."""
        self.max_retries = 2
        self.retry_delays = [30, 60]  # seconds

    def deduplicate_condition_data(self, condition: str) -> Dict[str, Any]:
        """
        Run deduplication for data related to a specific condition.

        Args:
            condition: Medical condition to deduplicate data for

        Returns:
            Deduplication result with statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting deduplication for condition: '{condition}'")

        try:
            # Get pre-deduplication counts
            pre_stats = self._get_condition_entity_counts(condition)

            # Run deduplication with retry logic
            dedup_result = self._run_deduplication_with_retry()

            # Get post-deduplication counts
            post_stats = self._get_condition_entity_counts(condition)

            # Calculate deduplication metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            entities_merged = pre_stats['total_entities'] - post_stats['total_entities']
            deduplication_rate = (entities_merged / pre_stats['total_entities'] * 100) if pre_stats['total_entities'] > 0 else 0

            result = {
                'success': True,
                'condition': condition,
                'entities_before': pre_stats['total_entities'],
                'entities_after': post_stats['total_entities'],
                'entities_merged': entities_merged,
                'deduplication_rate': deduplication_rate,
                'intervention_entities_before': pre_stats['intervention_entities'],
                'intervention_entities_after': post_stats['intervention_entities'],
                'condition_entities_before': pre_stats['condition_entities'],
                'condition_entities_after': post_stats['condition_entities'],
                'processing_time_seconds': processing_time,
                'status': 'completed'
            }

            logger.info(f"Deduplication completed for '{condition}': "
                       f"{entities_merged} entities merged "
                       f"({deduplication_rate:.1f}% reduction)")

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Deduplication failed for '{condition}': {e}")
            logger.error(traceback.format_exc())

            return {
                'success': False,
                'condition': condition,
                'entities_merged': 0,
                'processing_time_seconds': processing_time,
                'error': str(e),
                'status': 'failed'
            }

    def deduplicate_all_data_batch(self) -> Dict[str, Any]:
        """
        Comprehensive deduplication of ALL unprocessed interventions in the database.

        This method mirrors the comprehensive "process all unprocessed" pattern used
        in LLM processing, ensuring no intervention goes unprocessed.

        Returns:
            Comprehensive deduplication result with detailed statistics
        """
        start_time = datetime.now()
        logger.info("Starting comprehensive deduplication of ALL unprocessed interventions")

        try:
            # Get comprehensive pre-deduplication stats
            pre_stats = self._get_comprehensive_unprocessed_stats()
            logger.info(f"Found {pre_stats['unprocessed_interventions']} unprocessed interventions")
            logger.info(f"  - {pre_stats['missing_canonical_mapping']} lacking canonical mapping")
            logger.info(f"  - {pre_stats['missing_normalization']} lacking normalization")
            logger.info(f"  - {pre_stats['missing_consensus_processing']} lacking consensus processing")

            # Run comprehensive deduplication with retry logic
            dedup_result = self._run_comprehensive_deduplication_with_retry()

            # Get post-deduplication stats
            post_stats = self._get_comprehensive_unprocessed_stats()

            # Calculate comprehensive metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            interventions_processed = (
                pre_stats['unprocessed_interventions'] - post_stats['unprocessed_interventions']
            )

            result = {
                'success': True,
                'interventions_before_processing': pre_stats['unprocessed_interventions'],
                'interventions_after_processing': post_stats['unprocessed_interventions'],
                'interventions_processed': interventions_processed,
                'canonical_mappings_created': (
                    pre_stats['missing_canonical_mapping'] - post_stats['missing_canonical_mapping']
                ),
                'normalizations_completed': (
                    pre_stats['missing_normalization'] - post_stats['missing_normalization']
                ),
                'consensus_processing_completed': (
                    pre_stats['missing_consensus_processing'] - post_stats['missing_consensus_processing']
                ),
                'processing_time_seconds': processing_time,
                'batch_deduplication_result': dedup_result,
                'pre_processing_stats': pre_stats,
                'post_processing_stats': post_stats,
                'status': 'completed'
            }

            logger.info(f"Comprehensive deduplication completed: "
                       f"{interventions_processed} interventions processed in {processing_time:.1f}s")

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Comprehensive deduplication failed: {e}")
            logger.error(traceback.format_exc())

            return {
                'success': False,
                'interventions_processed': 0,
                'processing_time_seconds': processing_time,
                'error': str(e),
                'status': 'failed'
            }

    def _get_comprehensive_unprocessed_stats(self) -> Dict[str, int]:
        """Get comprehensive statistics of unprocessed interventions."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Count interventions missing canonical mapping
                cursor.execute("""
                    SELECT COUNT(*) FROM interventions
                    WHERE intervention_canonical_id IS NULL
                """)
                missing_canonical_mapping = cursor.fetchone()[0] or 0

                # Count interventions missing normalization
                cursor.execute("""
                    SELECT COUNT(*) FROM interventions
                    WHERE normalized IS NULL OR normalized = 0
                """)
                missing_normalization = cursor.fetchone()[0] or 0

                # Count interventions missing consensus processing
                cursor.execute("""
                    SELECT COUNT(*) FROM interventions
                    WHERE consensus_confidence IS NULL
                """)
                missing_consensus_processing = cursor.fetchone()[0] or 0

                # Total unprocessed (any missing field)
                cursor.execute("""
                    SELECT COUNT(*) FROM interventions
                    WHERE intervention_canonical_id IS NULL
                       OR normalized IS NULL
                       OR normalized = 0
                       OR consensus_confidence IS NULL
                """)
                total_unprocessed = cursor.fetchone()[0] or 0

                # Total interventions for context
                cursor.execute("SELECT COUNT(*) FROM interventions")
                total_interventions = cursor.fetchone()[0] or 0

                return {
                    'unprocessed_interventions': total_unprocessed,
                    'missing_canonical_mapping': missing_canonical_mapping,
                    'missing_normalization': missing_normalization,
                    'missing_consensus_processing': missing_consensus_processing,
                    'total_interventions': total_interventions,
                    'processing_completion_rate': (
                        ((total_interventions - total_unprocessed) / total_interventions * 100)
                        if total_interventions > 0 else 0
                    )
                }

        except Exception as e:
            logger.error(f"Error getting comprehensive unprocessed stats: {e}")
            return {
                'unprocessed_interventions': 0,
                'missing_canonical_mapping': 0,
                'missing_normalization': 0,
                'missing_consensus_processing': 0,
                'total_interventions': 0,
                'processing_completion_rate': 0
            }

    def _run_comprehensive_deduplication_with_retry(self) -> Dict[str, Any]:
        """Run comprehensive deduplication with retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Comprehensive deduplication attempt {attempt + 1}/{self.max_retries + 1}")

                # Create batch processor and run comprehensive deduplication
                processor = create_batch_processor()

                # Check if batch_deduplicate_entities method exists and call it
                if hasattr(processor, 'batch_deduplicate_entities'):
                    dedup_result = processor.batch_deduplicate_entities()
                    logger.info(f"Merged {dedup_result.get('total_merged', 0)} entities")
                else:
                    # Fallback to comprehensive processing if batch method not available
                    logger.info("Running comprehensive entity processing...")

                    # Get all unprocessed interventions
                    with database_manager.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT id, intervention_name, health_condition, paper_id
                            FROM interventions
                            WHERE intervention_canonical_id IS NULL
                               OR normalized IS NULL
                               OR normalized = 0
                               OR consensus_confidence IS NULL
                        """)
                        unprocessed_interventions = [dict(row) for row in cursor.fetchall()]

                    if unprocessed_interventions:
                        # Process interventions in batches
                        processed_count = 0
                        batch_size = 100

                        for i in range(0, len(unprocessed_interventions), batch_size):
                            batch = unprocessed_interventions[i:i + batch_size]

                            # Process each intervention in the batch
                            for intervention in batch:
                                try:
                                    # Normalize intervention name
                                    normalized_name = processor.get_or_compute_normalized_term(
                                        intervention['intervention_name'], 'intervention'
                                    )

                                    # Find or create canonical entity
                                    canonical_entity = processor.find_canonical_by_name(
                                        normalized_name, 'intervention'
                                    )

                                    if not canonical_entity:
                                        canonical_id = processor.create_canonical_entity(
                                            normalized_name, 'intervention'
                                        )
                                    else:
                                        canonical_id = canonical_entity['id']

                                    # Update intervention with canonical mapping
                                    with database_manager.get_connection() as conn:
                                        cursor = conn.cursor()
                                        cursor.execute("""
                                            UPDATE interventions
                                            SET intervention_canonical_id = ?,
                                                normalized = 1,
                                                consensus_confidence = COALESCE(consensus_confidence, 0.8)
                                            WHERE id = ?
                                        """, (canonical_id, intervention['id']))
                                        conn.commit()

                                    processed_count += 1

                                except Exception as e:
                                    logger.warning(f"Error processing intervention {intervention['id']}: {e}")
                                    continue

                            # Log progress
                            logger.info(f"Processed {processed_count}/{len(unprocessed_interventions)} interventions")

                    dedup_result = {
                        'total_merged': 0,
                        'interventions_processed': processed_count,
                        'method': 'comprehensive_processing'
                    }

                logger.info("Comprehensive deduplication completed successfully")
                return dedup_result

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Comprehensive deduplication attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("All comprehensive deduplication attempts failed")

        # If we get here, all retries failed
        raise DeduplicationError(f"Comprehensive deduplication failed after {self.max_retries + 1} attempts. Last error: {last_error}")

    def _run_deduplication_with_retry(self) -> Dict[str, Any]:
        """Run deduplication with retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Deduplication attempt {attempt + 1}/{self.max_retries + 1}")

                # Run the new batch deduplication
                processor = create_batch_processor()
                dedup_result = processor.batch_deduplicate_entities()
                logger.info(f"Merged {dedup_result['total_merged']} entities")

                logger.info("Deduplication completed successfully")
                return {'success': True}

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Deduplication attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("All deduplication attempts failed")

        # If we get here, all retries failed
        raise DeduplicationError(f"Deduplication failed after {self.max_retries + 1} attempts. Last error: {last_error}")

    def _get_condition_entity_counts(self, condition: str) -> Dict[str, int]:
        """Get entity counts for a specific condition."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Count intervention entities for this condition
                cursor.execute("""
                    SELECT COUNT(DISTINCT intervention_canonical_id)
                    FROM interventions
                    WHERE LOWER(health_condition) LIKE LOWER(?)
                    AND intervention_canonical_id IS NOT NULL
                """, (f"%{condition}%",))

                intervention_entities = cursor.fetchone()[0] or 0

                # Count condition entities for this condition
                cursor.execute("""
                    SELECT COUNT(DISTINCT condition_canonical_id)
                    FROM interventions
                    WHERE LOWER(health_condition) LIKE LOWER(?)
                    AND condition_canonical_id IS NOT NULL
                """, (f"%{condition}%",))

                condition_entities = cursor.fetchone()[0] or 0

                # Get total canonical entities (global count for comparison)
                cursor.execute("""
                    SELECT COUNT(*) FROM canonical_entities
                    WHERE entity_type = 'intervention'
                """)
                total_intervention_entities = cursor.fetchone()[0] or 0

                cursor.execute("""
                    SELECT COUNT(*) FROM canonical_entities
                    WHERE entity_type = 'condition'
                """)
                total_condition_entities = cursor.fetchone()[0] or 0

                return {
                    'intervention_entities': intervention_entities,
                    'condition_entities': condition_entities,
                    'total_entities': total_intervention_entities + total_condition_entities,
                    'total_intervention_entities': total_intervention_entities,
                    'total_condition_entities': total_condition_entities
                }

        except Exception as e:
            logger.error(f"Error getting entity counts for '{condition}': {e}")
            return {
                'intervention_entities': 0,
                'condition_entities': 0,
                'total_entities': 0,
                'total_intervention_entities': 0,
                'total_condition_entities': 0
            }

    def get_deduplication_status(self, condition: str) -> Dict[str, Any]:
        """Get deduplication status for a condition."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get intervention statistics
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_interventions,
                        COUNT(DISTINCT intervention_name) as unique_intervention_names,
                        COUNT(DISTINCT intervention_canonical_id) as canonical_interventions,
                        COUNT(CASE WHEN intervention_canonical_id IS NULL THEN 1 END) as unmapped_interventions
                    FROM interventions
                    WHERE LOWER(health_condition) LIKE LOWER(?)
                """, (f"%{condition}%",))

                intervention_stats = cursor.fetchone()

                # Get condition statistics
                cursor.execute("""
                    SELECT
                        COUNT(DISTINCT health_condition) as unique_condition_names,
                        COUNT(DISTINCT condition_canonical_id) as canonical_conditions,
                        COUNT(CASE WHEN condition_canonical_id IS NULL THEN 1 END) as unmapped_conditions
                    FROM interventions
                    WHERE LOWER(health_condition) LIKE LOWER(?)
                """, (f"%{condition}%",))

                condition_stats = cursor.fetchone()

                # Calculate mapping rates
                intervention_mapping_rate = (
                    (intervention_stats[2] / intervention_stats[1] * 100)
                    if intervention_stats and intervention_stats[1] > 0 else 0
                )

                condition_mapping_rate = (
                    (condition_stats[1] / condition_stats[0] * 100)
                    if condition_stats and condition_stats[0] > 0 else 0
                )

                return {
                    'condition': condition,
                    'total_interventions': intervention_stats[0] if intervention_stats else 0,
                    'unique_intervention_names': intervention_stats[1] if intervention_stats else 0,
                    'canonical_interventions': intervention_stats[2] if intervention_stats else 0,
                    'unmapped_interventions': intervention_stats[3] if intervention_stats else 0,
                    'intervention_mapping_rate': intervention_mapping_rate,
                    'unique_condition_names': condition_stats[0] if condition_stats else 0,
                    'canonical_conditions': condition_stats[1] if condition_stats else 0,
                    'unmapped_conditions': condition_stats[2] if condition_stats else 0,
                    'condition_mapping_rate': condition_mapping_rate,
                    'overall_deduplication_quality': (intervention_mapping_rate + condition_mapping_rate) / 2
                }

        except Exception as e:
            logger.error(f"Error getting deduplication status for '{condition}': {e}")
            return {
                'condition': condition,
                'error': str(e)
            }

    def validate_deduplication_result(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate deduplication result.

        Returns:
            Tuple of (is_valid, validation_message)
        """
        if not isinstance(result, dict):
            return False, "Result is not a dictionary"

        required_fields = ['success', 'condition', 'entities_merged']
        missing_fields = [field for field in required_fields if field not in result]

        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"

        if not result['success']:
            error_msg = result.get('error', 'Unknown error')
            return False, f"Deduplication failed: {error_msg}"

        if result['entities_merged'] < 0:
            return False, f"Invalid entities_merged count: {result['entities_merged']}"

        # Check for reasonable deduplication rate (0-50% is normal)
        dedup_rate = result.get('deduplication_rate', 0)
        if dedup_rate > 50:
            return True, f"Warning: High deduplication rate ({dedup_rate:.1f}%) - verify data quality"

        return True, "Deduplication result is valid"

    def run_condition_cleanup(self, condition: str) -> Dict[str, Any]:
        """
        Run additional cleanup for a condition after deduplication.

        Args:
            condition: Medical condition to clean up

        Returns:
            Cleanup result with statistics
        """
        start_time = datetime.now()
        logger.info(f"Running cleanup for condition: '{condition}'")

        try:
            cleanup_stats = {
                'orphaned_mappings_removed': 0,
                'empty_entities_removed': 0,
                'duplicate_interventions_merged': 0
            }

            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Remove orphaned entity mappings
                cursor.execute("""
                    DELETE FROM entity_mappings
                    WHERE canonical_id NOT IN (SELECT id FROM canonical_entities)
                """)
                cleanup_stats['orphaned_mappings_removed'] = cursor.rowcount

                # Remove canonical entities with no associated interventions
                cursor.execute("""
                    DELETE FROM canonical_entities
                    WHERE entity_type = 'intervention'
                    AND id NOT IN (
                        SELECT DISTINCT intervention_canonical_id
                        FROM interventions
                        WHERE intervention_canonical_id IS NOT NULL
                    )
                """)
                cleanup_stats['empty_entities_removed'] += cursor.rowcount

                cursor.execute("""
                    DELETE FROM canonical_entities
                    WHERE entity_type = 'condition'
                    AND id NOT IN (
                        SELECT DISTINCT condition_canonical_id
                        FROM interventions
                        WHERE condition_canonical_id IS NOT NULL
                    )
                """)
                cleanup_stats['empty_entities_removed'] += cursor.rowcount

                conn.commit()

            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                'success': True,
                'condition': condition,
                'cleanup_stats': cleanup_stats,
                'processing_time_seconds': processing_time,
                'status': 'completed'
            }

            logger.info(f"Cleanup completed for '{condition}': "
                       f"{sum(cleanup_stats.values())} items cleaned up")

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Cleanup failed for '{condition}': {e}")

            return {
                'success': False,
                'condition': condition,
                'processing_time_seconds': processing_time,
                'error': str(e),
                'status': 'failed'
            }


def deduplicate_single_condition(condition: str) -> Dict[str, Any]:
    """
    Convenience function to deduplicate data for a single condition.

    Args:
        condition: Medical condition to deduplicate

    Returns:
        Deduplication result dictionary
    """
    integrator = RotationDeduplicationIntegrator()
    return integrator.deduplicate_condition_data(condition)


if __name__ == "__main__":
    """Test the rotation deduplication integrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Rotation Deduplication Integrator Test")
    parser.add_argument('condition', help='Medical condition to deduplicate data for')
    parser.add_argument('--status-only', action='store_true', help='Show status only, no deduplication')
    parser.add_argument('--cleanup', action='store_true', help='Run cleanup after deduplication')

    args = parser.parse_args()

    integrator = RotationDeduplicationIntegrator()

    if args.status_only:
        # Show status only
        status = integrator.get_deduplication_status(args.condition)
        print(f"\nDeduplication Status for: {args.condition}")
        print("="*50)
        print(f"Total interventions: {status['total_interventions']}")
        print(f"Unique intervention names: {status['unique_intervention_names']}")
        print(f"Canonical interventions: {status['canonical_interventions']}")
        print(f"Intervention mapping rate: {status['intervention_mapping_rate']:.1f}%")
        print(f"Unique condition names: {status['unique_condition_names']}")
        print(f"Canonical conditions: {status['canonical_conditions']}")
        print(f"Condition mapping rate: {status['condition_mapping_rate']:.1f}%")
        print(f"Overall quality: {status['overall_deduplication_quality']:.1f}%")
    else:
        # Run deduplication
        print(f"Running deduplication for: {args.condition}")

        result = integrator.deduplicate_condition_data(args.condition)

        print("\n" + "="*60)
        print("DEDUPLICATION RESULT")
        print("="*60)
        print(f"Success: {result['success']}")
        print(f"Condition: {result['condition']}")
        print(f"Entities before: {result.get('entities_before', 0)}")
        print(f"Entities after: {result.get('entities_after', 0)}")
        print(f"Entities merged: {result.get('entities_merged', 0)}")
        print(f"Deduplication rate: {result.get('deduplication_rate', 0):.1f}%")
        print(f"Processing time: {result.get('processing_time_seconds', 0):.1f} seconds")
        print(f"Status: {result['status']}")

        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")

        # Validate result
        is_valid, message = integrator.validate_deduplication_result(result)
        print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
        print(f"Message: {message}")

        # Run cleanup if requested
        if args.cleanup and result['success']:
            print(f"\nRunning cleanup for: {args.condition}")
            cleanup_result = integrator.run_condition_cleanup(args.condition)

            print(f"Cleanup success: {cleanup_result['success']}")
            if cleanup_result['success']:
                stats = cleanup_result['cleanup_stats']
                print(f"Orphaned mappings removed: {stats['orphaned_mappings_removed']}")
                print(f"Empty entities removed: {stats['empty_entities_removed']}")
                print(f"Total items cleaned: {sum(stats.values())}")