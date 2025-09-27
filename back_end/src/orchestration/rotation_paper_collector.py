#!/usr/bin/env python3
"""
Rotation Pipeline Paper Collector

Simplified paper collector specifically designed for the medical rotation pipeline.
Collects exactly N papers for a single condition with robust error handling
and integration with the rotation session manager.

Features:
- Single-condition focused collection
- Exact paper count targeting
- Network resilience and retry logic
- Integration with rotation session manager
- Progress tracking and validation
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

try:
    from ..data.config import config, setup_logging
    from ..data_collection.database_manager import database_manager
    from ..data_collection.pubmed_collector import PubMedCollector
    from ..data_collection.semantic_scholar_enrichment import run_semantic_scholar_enrichment
    from .rotation_session_manager import (
        RotationSessionManager, PipelinePhase, session_manager
    )
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.data_collection.database_manager import database_manager
    from back_end.src.data_collection.pubmed_collector import PubMedCollector
    from back_end.src.data_collection.semantic_scholar_enrichment import run_semantic_scholar_enrichment
    from back_end.src.orchestration.rotation_session_manager import (
        RotationSessionManager, PipelinePhase, session_manager
    )

logger = setup_logging(__name__, 'rotation_paper_collector.log')


class RotationPaperCollector:
    """
    Simplified paper collector for rotation pipeline.
    Collects exactly N papers for a single condition.
    """

    def __init__(self):
        """Initialize the rotation paper collector."""
        self.pubmed_collector = PubMedCollector()
        self.max_retries = 3
        self.retry_delays = [30, 60, 120]  # seconds
        self.batch_size = 25  # PubMed API batch size

    def collect_condition_papers(self, condition: str, target_count: int = 10,
                                min_year: int = 2015, max_year: Optional[int] = None,
                                use_s2_enrichment: bool = True) -> Dict[str, Any]:
        """
        Collect exactly target_count papers for a specific condition.

        Args:
            condition: Medical condition to search for
            target_count: Exact number of papers to collect
            min_year: Minimum publication year
            max_year: Maximum publication year (None for current year)
            use_s2_enrichment: Whether to use Semantic Scholar enrichment

        Returns:
            Dictionary with collection results and statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting collection for '{condition}' (target: {target_count} papers)")

        try:
            # Check if we already have enough papers for this condition
            existing_count = self._count_existing_papers(condition)
            if existing_count >= target_count:
                logger.info(f"Already have {existing_count} papers for '{condition}', skipping collection")
                return {
                    'success': True,
                    'condition': condition,
                    'papers_collected': 0,
                    'total_papers': existing_count,
                    'target_reached': True,
                    'collection_time_seconds': 0,
                    'status': 'already_complete'
                }

            needed_papers = target_count - existing_count
            logger.info(f"Need {needed_papers} more papers for '{condition}' (have {existing_count})")

            # Collect papers with retry logic
            collection_result = self._collect_with_retry(
                condition, needed_papers, min_year, max_year
            )

            # Add Semantic Scholar enrichment if enabled and we have new papers
            if use_s2_enrichment and collection_result['papers_collected'] > 0:
                try:
                    logger.info(f"Running Semantic Scholar enrichment for '{condition}'...")
                    s2_result = run_semantic_scholar_enrichment(
                        condition_filter=condition,
                        limit=collection_result['papers_collected']
                    )
                    collection_result['s2_enrichment'] = s2_result
                    logger.info(f"S2 enrichment completed for '{condition}'")
                except Exception as e:
                    logger.warning(f"S2 enrichment failed for '{condition}': {e}")
                    collection_result['s2_enrichment'] = {'error': str(e)}

            # Final count verification
            final_count = self._count_existing_papers(condition)
            collection_time = (datetime.now() - start_time).total_seconds()

            result = {
                'success': True,
                'condition': condition,
                'papers_collected': collection_result['papers_collected'],
                'total_papers': final_count,
                'target_reached': final_count >= target_count,
                'collection_time_seconds': collection_time,
                'pubmed_stats': collection_result.get('pubmed_stats', {}),
                's2_enrichment': collection_result.get('s2_enrichment', {}),
                'status': 'completed'
            }

            if result['target_reached']:
                logger.info(f"Successfully collected papers for '{condition}': "
                           f"{final_count} total papers (target: {target_count})")
            else:
                logger.warning(f"Partial collection for '{condition}': "
                              f"{final_count} total papers (target: {target_count})")

            return result

        except Exception as e:
            collection_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Collection failed for '{condition}': {e}")
            logger.error(traceback.format_exc())

            return {
                'success': False,
                'condition': condition,
                'papers_collected': 0,
                'total_papers': self._count_existing_papers(condition),
                'target_reached': False,
                'collection_time_seconds': collection_time,
                'error': str(e),
                'status': 'failed'
            }

    def _collect_with_retry(self, condition: str, needed_papers: int,
                           min_year: int, max_year: Optional[int]) -> Dict[str, Any]:
        """Collect papers with retry logic for network resilience."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Collection attempt {attempt + 1}/{self.max_retries} for '{condition}'")

                # Use PubMed collector
                result = self.pubmed_collector.collect_interventions_by_condition(
                    condition=condition,
                    min_year=min_year,
                    max_year=max_year,
                    max_results=needed_papers,
                    include_fulltext=True
                )

                # Validate result
                if not isinstance(result, dict):
                    raise ValueError(f"Invalid result type from PubMed collector: {type(result)}")

                papers_collected = result.get('paper_count', 0)
                if papers_collected > 0:
                    logger.info(f"Successfully collected {papers_collected} papers for '{condition}'")
                    return {
                        'papers_collected': papers_collected,
                        'pubmed_stats': {
                            'total_searched': result.get('total_papers_searched', 0),
                            'papers_added': papers_collected,
                            'search_terms': result.get('search_terms_used', [])
                        }
                    }
                else:
                    logger.warning(f"No papers collected for '{condition}' on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delays[min(attempt, len(self.retry_delays) - 1)])
                        continue

                return {'papers_collected': 0, 'pubmed_stats': {}}

            except Exception as e:
                last_error = e
                logger.warning(f"Collection attempt {attempt + 1} failed for '{condition}': {e}")

                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All collection attempts failed for '{condition}'")

        # If we get here, all retries failed
        raise RuntimeError(f"Failed to collect papers after {self.max_retries} attempts. Last error: {last_error}")

    def _count_existing_papers(self, condition: str) -> int:
        """Count existing papers for a condition in the database."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Count papers that have interventions for this condition
                cursor.execute("""
                    SELECT COUNT(DISTINCT p.id)
                    FROM papers p
                    JOIN interventions i ON p.id = i.paper_id
                    WHERE LOWER(i.condition) LIKE LOWER(?)
                """, (f"%{condition}%",))

                count = cursor.fetchone()[0]
                logger.debug(f"Found {count} existing papers for condition '{condition}'")
                return count

        except Exception as e:
            logger.error(f"Error counting existing papers for '{condition}': {e}")
            return 0

    def validate_collection_result(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate collection result and return success status with message.

        Args:
            result: Collection result dictionary

        Returns:
            Tuple of (is_valid, validation_message)
        """
        if not isinstance(result, dict):
            return False, "Result is not a dictionary"

        required_fields = ['success', 'condition', 'papers_collected', 'total_papers', 'target_reached']
        missing_fields = [field for field in required_fields if field not in result]

        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"

        if not result['success']:
            error_msg = result.get('error', 'Unknown error')
            return False, f"Collection failed: {error_msg}"

        if result['papers_collected'] < 0:
            return False, f"Invalid papers_collected count: {result['papers_collected']}"

        if result['total_papers'] < result['papers_collected']:
            return False, f"Total papers ({result['total_papers']}) less than papers collected ({result['papers_collected']})"

        # Check if target was reached (warning, not failure)
        if not result['target_reached']:
            return True, f"Collection completed but target not fully reached: {result['total_papers']} collected"

        return True, "Collection result is valid"

    def get_collection_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for multiple collection results.

        Args:
            results: List of collection result dictionaries

        Returns:
            Summary statistics dictionary
        """
        if not results:
            return {'total_conditions': 0, 'successful_conditions': 0, 'total_papers': 0}

        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]

        total_papers = sum(r.get('papers_collected', 0) for r in results)
        total_time = sum(r.get('collection_time_seconds', 0) for r in results)

        summary = {
            'total_conditions': len(results),
            'successful_conditions': len(successful),
            'failed_conditions': len(failed),
            'success_rate': (len(successful) / len(results)) * 100 if results else 0,
            'total_papers_collected': total_papers,
            'total_collection_time_seconds': total_time,
            'average_papers_per_condition': total_papers / len(results) if results else 0,
            'average_time_per_condition': total_time / len(results) if results else 0,
            'failed_conditions_list': [r.get('condition', 'unknown') for r in failed]
        }

        return summary

    # ================================================================================
    # INTEGRATION METHODS (merged from rotation_collection_integrator.py)
    # ================================================================================

    def collect_current_condition(self, session_mgr: RotationSessionManager = None) -> Dict[str, Any]:
        """Collect papers for the current condition in the rotation session."""
        if session_mgr is None:
            session_mgr = session_manager

        if not session_mgr.session:
            raise Exception("No active rotation session found")

        session = session_mgr.session
        condition = session.current_condition
        target_count = session.papers_per_condition

        logger.info(f"Collecting {target_count} papers for: {condition}")

        # Set interruption state for collection phase
        session_mgr.set_interruption_state(phase=PipelinePhase.COLLECTION)

        try:
            # Simple retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.collect_condition_papers(
                        condition=condition,
                        target_count=target_count,
                        min_year=2015,
                        use_s2_enrichment=True
                    )

                    if result['success']:
                        # Update session progress
                        session_mgr.update_progress(
                            papers_collected=result['papers_collected']
                        )
                        session_mgr.clear_interruption_state()
                        logger.info(f"Collection completed: {result['papers_collected']} papers")
                        return result

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Collection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(30 * (attempt + 1))  # Progressive delay

            return {'success': False, 'error': 'Max retries exceeded'}

        except Exception as e:
            logger.error(f"Collection failed for '{condition}': {e}")
            session_mgr.mark_condition_failed(str(e))
            return {'success': False, 'error': str(e)}

    def get_collection_status(self, session_mgr: RotationSessionManager = None) -> Dict[str, Any]:
        """Get current collection status from session manager."""
        if session_mgr is None:
            session_mgr = session_manager

        if not session_mgr.session:
            return {'error': 'No active session'}

        session = session_mgr.session
        is_collecting = (
            session.interruption_state and
            session.interruption_state.phase == PipelinePhase.COLLECTION.value
        )

        return {
            'current_condition': session.current_condition,
            'current_specialty': session.current_specialty,
            'target_papers': session.papers_per_condition,
            'is_collecting': is_collecting,
            'total_papers_collected': session.total_papers_collected,
            'completed_conditions': len(session.completed_conditions),
            'session_active': session.is_active
        }

    def resume_interrupted_collection(self, session_mgr: RotationSessionManager = None) -> Optional[Dict[str, Any]]:
        """Resume collection if interrupted during collection phase."""
        if session_mgr is None:
            session_mgr = session_manager

        if not session_mgr.session or not session_mgr.session.interruption_state:
            return None

        interruption = session_mgr.session.interruption_state
        if interruption.phase != PipelinePhase.COLLECTION.value:
            return None

        logger.info(f"Resuming interrupted collection for '{session_mgr.session.current_condition}'")
        return self.collect_current_condition(session_mgr)


def create_collection_integrator(session_mgr: RotationSessionManager = None) -> RotationPaperCollector:
    """
    Create and return a collection integrator instance.

    This function provides backward compatibility for code that previously
    used RotationCollectionIntegrator. The RotationPaperCollector now includes
    all integration functionality.
    """
    return RotationPaperCollector()


def collect_single_condition(condition: str, target_count: int = 10,
                            min_year: int = 2015, max_year: Optional[int] = None,
                            use_s2_enrichment: bool = True) -> Dict[str, Any]:
    """
    Convenience function to collect papers for a single condition.

    Args:
        condition: Medical condition to search for
        target_count: Number of papers to collect
        min_year: Minimum publication year
        max_year: Maximum publication year
        use_s2_enrichment: Whether to use Semantic Scholar enrichment

    Returns:
        Collection result dictionary
    """
    collector = RotationPaperCollector()
    return collector.collect_condition_papers(
        condition=condition,
        target_count=target_count,
        min_year=min_year,
        max_year=max_year,
        use_s2_enrichment=use_s2_enrichment
    )


if __name__ == "__main__":
    """Test the rotation paper collector with a single condition."""
    import argparse

    parser = argparse.ArgumentParser(description="Rotation Paper Collector Test")
    parser.add_argument('condition', help='Medical condition to collect papers for')
    parser.add_argument('--count', type=int, default=5, help='Number of papers to collect')
    parser.add_argument('--min-year', type=int, default=2020, help='Minimum publication year')
    parser.add_argument('--no-s2', action='store_true', help='Disable Semantic Scholar enrichment')

    args = parser.parse_args()

    print(f"Testing paper collection for: {args.condition}")
    print(f"Target papers: {args.count}")
    print(f"Min year: {args.min_year}")
    print(f"S2 enrichment: {not args.no_s2}")

    result = collect_single_condition(
        condition=args.condition,
        target_count=args.count,
        min_year=args.min_year,
        use_s2_enrichment=not args.no_s2
    )

    print("\n" + "="*60)
    print("COLLECTION RESULT")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Condition: {result['condition']}")
    print(f"Papers collected: {result['papers_collected']}")
    print(f"Total papers: {result['total_papers']}")
    print(f"Target reached: {result['target_reached']}")
    print(f"Collection time: {result['collection_time_seconds']:.1f} seconds")
    print(f"Status: {result['status']}")

    if not result['success']:
        print(f"Error: {result.get('error', 'Unknown error')}")

    # Validate result
    collector = RotationPaperCollector()
    is_valid, message = collector.validate_collection_result(result)
    print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
    print(f"Message: {message}")