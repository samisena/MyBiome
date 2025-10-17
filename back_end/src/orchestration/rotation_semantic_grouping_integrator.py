#!/usr/bin/env python3
"""
Rotation Semantic Grouping Integrator (Updated October 16, 2025)

MIGRATION NOTE: This file was updated to use Phase 3 semantic normalization
instead of deprecated legacy normalization tables (canonical_entities, entity_mappings).

This integrator now serves as a WRAPPER around Phase3ABCOrchestrator,
maintaining backward compatibility with batch_medical_rotation.py while
using the modern clustering-first architecture.

Legacy tables dropped: canonical_entities, entity_mappings, normalized_terms_cache, llm_normalization_cache
Replacement: semantic_hierarchy + canonical_groups (Phase 3a/3b/3c)
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

try:
    from ..data.config import config, setup_logging
    from .phase_3abc_semantic_normalizer import Phase3ABCOrchestrator
    from ..phase_1_data_collection.database_manager import database_manager
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.orchestration.phase_3abc_semantic_normalizer import Phase3ABCOrchestrator
    from back_end.src.phase_1_data_collection.database_manager import database_manager

logger = setup_logging(__name__, 'rotation_semantic_grouping_integrator.log')


class SemanticGroupingError(Exception):
    """Custom exception for semantic grouping errors."""
    pass


class RotationSemanticGroupingIntegrator:
    """
    Wrapper around Phase3ABCOrchestrator for backward compatibility.

    This class maintains the interface expected by batch_medical_rotation.py
    while delegating all work to the modern Phase 3 clustering-first architecture.

    Key method: group_all_data_semantically_batch()
    """

    def __init__(self):
        """Initialize the semantic grouping integrator."""
        self.max_retries = 2
        self.retry_delays = [30, 60]  # seconds
        self.orchestrator = Phase3ABCOrchestrator()
        logger.info("Initialized RotationSemanticGroupingIntegrator (Phase 3 wrapper)")

    def group_all_data_semantically_batch(self) -> Dict[str, Any]:
        """
        Comprehensive semantic normalization of ALL interventions in the database.

        UPDATED (Oct 16, 2025): Now uses Phase 3abc clustering-first architecture
        instead of legacy LLM-based normalization.

        Phase 3 Pipeline:
        - Phase 3a: Generate embeddings (mxbai-embed-large, 1024-dim)
        - Phase 3b: Cluster embeddings (hierarchical, distance_threshold=0.7)
        - Phase 3c: Name clusters with LLM (qwen3:14b, temperature=0.0)

        Returns:
            Comprehensive normalization result with detailed statistics
        """
        start_time = datetime.now()
        logger.info("Starting comprehensive semantic normalization via Phase 3abc orchestrator")

        try:
            # Get pre-normalization stats
            pre_stats = self._get_pre_normalization_stats()
            logger.info(f"Found {pre_stats['total_interventions']} total interventions")
            logger.info(f"  - {pre_stats['unique_intervention_names']} unique names")
            logger.info(f"  - {pre_stats['normalized_interventions']} already normalized")
            logger.info(f"  - {pre_stats['unnormalized_interventions']} need normalization")

            # Run Phase 3abc orchestrator with retry logic
            normalization_result = self._run_phase3_with_retry()

            # Get post-normalization stats
            post_stats = self._get_post_normalization_stats()

            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            canonical_groups_created = post_stats.get('canonical_groups', 0)
            cluster_reduction_rate = (
                (1 - canonical_groups_created / pre_stats['unique_intervention_names']) * 100
                if pre_stats['unique_intervention_names'] > 0 else 0
            )

            result = {
                'success': True,
                'interventions_before_processing': pre_stats['unnormalized_interventions'],
                'interventions_after_processing': post_stats.get('unnormalized_interventions', 0),
                'interventions_processed': pre_stats['unnormalized_interventions'] - post_stats.get('unnormalized_interventions', 0),
                'canonical_groups_created': canonical_groups_created,
                'cluster_reduction_rate': cluster_reduction_rate,
                'unique_names_before': pre_stats['unique_intervention_names'],
                'canonical_groups_after': canonical_groups_created,
                'processing_time_seconds': processing_time,
                'phase_3_result': normalization_result,
                'pre_processing_stats': pre_stats,
                'post_processing_stats': post_stats,
                'status': 'completed',
                'architecture': 'phase_3abc_clustering_first'
            }

            logger.info(f"Comprehensive normalization completed:")
            logger.info(f"  - {canonical_groups_created} canonical groups created")
            logger.info(f"  - {cluster_reduction_rate:.1f}% cluster reduction")
            logger.info(f"  - Processing time: {processing_time:.1f}s")

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Comprehensive normalization failed: {e}")
            logger.error(traceback.format_exc())

            return {
                'success': False,
                'interventions_processed': 0,
                'processing_time_seconds': processing_time,
                'error': str(e),
                'status': 'failed',
                'architecture': 'phase_3abc_clustering_first'
            }

    def _get_pre_normalization_stats(self) -> Dict[str, int]:
        """Get statistics before normalization."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Total interventions
                cursor.execute("SELECT COUNT(*) FROM interventions")
                total_interventions = cursor.fetchone()[0] or 0

                # Unique intervention names
                cursor.execute("SELECT COUNT(DISTINCT intervention_name) FROM interventions")
                unique_intervention_names = cursor.fetchone()[0] or 0

                # Interventions already in semantic_hierarchy (normalized)
                cursor.execute("""
                    SELECT COUNT(DISTINCT i.intervention_name)
                    FROM interventions i
                    INNER JOIN semantic_hierarchy sh
                        ON LOWER(TRIM(i.intervention_name)) = LOWER(TRIM(sh.original_name))
                    WHERE sh.entity_type = 'intervention'
                """)
                normalized_interventions = cursor.fetchone()[0] or 0

                # Interventions NOT in semantic_hierarchy (need normalization)
                unnormalized_interventions = unique_intervention_names - normalized_interventions

                return {
                    'total_interventions': total_interventions,
                    'unique_intervention_names': unique_intervention_names,
                    'normalized_interventions': normalized_interventions,
                    'unnormalized_interventions': unnormalized_interventions,
                    'normalization_rate': (
                        (normalized_interventions / unique_intervention_names * 100)
                        if unique_intervention_names > 0 else 0
                    )
                }

        except Exception as e:
            logger.error(f"Error getting pre-normalization stats: {e}")
            return {
                'total_interventions': 0,
                'unique_intervention_names': 0,
                'normalized_interventions': 0,
                'unnormalized_interventions': 0,
                'normalization_rate': 0
            }

    def _get_post_normalization_stats(self) -> Dict[str, int]:
        """Get statistics after normalization."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Count canonical groups created (Layer 1 aggregation)
                cursor.execute("""
                    SELECT COUNT(DISTINCT layer_1_canonical)
                    FROM semantic_hierarchy
                    WHERE entity_type = 'intervention'
                    AND layer_1_canonical IS NOT NULL
                """)
                canonical_groups = cursor.fetchone()[0] or 0

                # Count interventions still not normalized
                cursor.execute("""
                    SELECT COUNT(DISTINCT i.intervention_name)
                    FROM interventions i
                    LEFT JOIN semantic_hierarchy sh
                        ON LOWER(TRIM(i.intervention_name)) = LOWER(TRIM(sh.original_name))
                    WHERE sh.original_name IS NULL
                """)
                unnormalized_interventions = cursor.fetchone()[0] or 0

                # Count total semantic hierarchy entries
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM semantic_hierarchy
                    WHERE entity_type = 'intervention'
                """)
                total_hierarchy_entries = cursor.fetchone()[0] or 0

                return {
                    'canonical_groups': canonical_groups,
                    'unnormalized_interventions': unnormalized_interventions,
                    'total_hierarchy_entries': total_hierarchy_entries
                }

        except Exception as e:
            logger.error(f"Error getting post-normalization stats: {e}")
            return {
                'canonical_groups': 0,
                'unnormalized_interventions': 0,
                'total_hierarchy_entries': 0
            }

    def _run_phase3_with_retry(self) -> Dict[str, Any]:
        """Run Phase 3abc orchestrator with retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Phase 3abc normalization attempt {attempt + 1}/{self.max_retries + 1}")

                # Run Phase 3abc orchestrator
                # This runs: 3a (embeddings) → 3b (clustering) → 3c (naming)
                result = self.orchestrator.run_intervention_normalization(
                    batch_size=50,
                    force_rerun=False  # Use cached results when available
                )

                logger.info(f"Phase 3abc completed successfully")
                logger.info(f"  - Embeddings: {result.get('embeddings_generated', 0)}")
                logger.info(f"  - Clusters: {result.get('clusters_created', 0)}")
                logger.info(f"  - Named groups: {result.get('groups_named', 0)}")

                return result

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Phase 3abc attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error("All Phase 3abc attempts failed")

        # If we get here, all retries failed
        raise SemanticGroupingError(
            f"Phase 3abc normalization failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error}"
        )

    def get_normalization_status(self) -> Dict[str, Any]:
        """
        Get current normalization status.

        Returns:
            Dictionary with normalization statistics
        """
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get comprehensive stats
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_interventions,
                        COUNT(DISTINCT intervention_name) as unique_names
                    FROM interventions
                """)
                intervention_stats = cursor.fetchone()

                cursor.execute("""
                    SELECT COUNT(DISTINCT layer_1_canonical)
                    FROM semantic_hierarchy
                    WHERE entity_type = 'intervention'
                """)
                canonical_groups = cursor.fetchone()[0] or 0

                cursor.execute("""
                    SELECT COUNT(*)
                    FROM semantic_hierarchy
                    WHERE entity_type = 'intervention'
                """)
                hierarchy_entries = cursor.fetchone()[0] or 0

                # Calculate reduction rate
                unique_names = intervention_stats[1] if intervention_stats else 0
                reduction_rate = (
                    (1 - canonical_groups / unique_names) * 100
                    if unique_names > 0 else 0
                )

                return {
                    'total_interventions': intervention_stats[0] if intervention_stats else 0,
                    'unique_intervention_names': unique_names,
                    'canonical_groups': canonical_groups,
                    'hierarchy_entries': hierarchy_entries,
                    'cluster_reduction_rate': reduction_rate,
                    'architecture': 'phase_3abc_clustering_first',
                    'status': 'active'
                }

        except Exception as e:
            logger.error(f"Error getting normalization status: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }


def main():
    """Test the rotation semantic grouping integrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Rotation Semantic Grouping Integrator (Phase 3 wrapper)")
    parser.add_argument('--status', action='store_true', help='Show normalization status')
    parser.add_argument('--run', action='store_true', help='Run full normalization')

    args = parser.parse_args()

    integrator = RotationSemanticGroupingIntegrator()

    if args.status:
        # Show status
        status = integrator.get_normalization_status()
        print(f"\nSemantic Normalization Status (Phase 3abc)")
        print("="*60)
        print(f"Total interventions: {status.get('total_interventions', 0)}")
        print(f"Unique names: {status.get('unique_intervention_names', 0)}")
        print(f"Canonical groups: {status.get('canonical_groups', 0)}")
        print(f"Cluster reduction: {status.get('cluster_reduction_rate', 0):.1f}%")
        print(f"Architecture: {status.get('architecture', 'unknown')}")

    elif args.run:
        # Run normalization
        print("Running comprehensive semantic normalization via Phase 3abc...")
        result = integrator.group_all_data_semantically_batch()

        print("\n" + "="*60)
        print("NORMALIZATION RESULT")
        print("="*60)
        print(f"Success: {result['success']}")
        print(f"Canonical groups created: {result.get('canonical_groups_created', 0)}")
        print(f"Cluster reduction: {result.get('cluster_reduction_rate', 0):.1f}%")
        print(f"Processing time: {result.get('processing_time_seconds', 0):.1f} seconds")
        print(f"Status: {result['status']}")

        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
