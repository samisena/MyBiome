#!/usr/bin/env python3
"""
Batch Paper Collector for Medical Rotation Pipeline

Optimized batch paper collector that collects papers for all 60 medical conditions
in parallel before processing. Removes Semantic Scholar from orchestration pipeline
for simplified, faster collection.

Features:
- Batch collection for all 60 conditions in parallel
- PubMed-only collection (S2 removed from pipeline)
- Parallel API calls for faster collection
- Bulk database operations
- Progress tracking and validation
- Quality gates before proceeding to processing
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import concurrent.futures
from dataclasses import dataclass

try:
    from ..data.config import config, setup_logging
    from ..data_collection.database_manager import database_manager
    from ..data_collection.pubmed_collector import PubMedCollector
    # Semantic Scholar removed from orchestration pipeline (keeping module intact for future use)
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
    # Semantic Scholar removed from orchestration pipeline (keeping module intact for future use)
    from back_end.src.orchestration.rotation_session_manager import (
        RotationSessionManager, PipelinePhase, session_manager
    )

logger = setup_logging(__name__, 'rotation_paper_collector.log')


@dataclass
class BatchCollectionResult:
    """Result from batch collection of papers across all conditions."""
    total_conditions: int = 0
    successful_conditions: int = 0
    failed_conditions: int = 0
    total_papers_collected: int = 0
    total_collection_time_seconds: float = 0.0
    conditions_results: List[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None

    def __post_init__(self):
        if self.conditions_results is None:
            self.conditions_results = []


class RotationPaperCollector:
    """
    Batch paper collector for rotation pipeline.
    Collects papers for all 60 medical conditions in parallel.
    """

    def __init__(self):
        """Initialize the batch paper collector."""
        self.pubmed_collector = PubMedCollector()
        self.max_retries = 3
        self.retry_delays = [30, 60, 120]  # seconds
        self.batch_size = 25  # PubMed API batch size
        self.max_workers = 8  # Number of parallel API calls

    def get_all_conditions(self) -> List[str]:
        """Get all 60 medical conditions from config."""
        all_conditions = []
        for specialty, conditions in config.medical_specialties.items():
            all_conditions.extend(conditions)
        return all_conditions

    def collect_all_conditions_batch(self, papers_per_condition: int = 10,
                                   min_year: int = 2015, max_year: Optional[int] = None) -> BatchCollectionResult:
        """
        Collect papers for all 60 medical conditions in parallel batches.

        Args:
            papers_per_condition: Target number of papers per condition
            min_year: Minimum publication year
            max_year: Maximum publication year (None for current year)

        Returns:
            BatchCollectionResult with comprehensive collection statistics
        """
        start_time = datetime.now()
        all_conditions = self.get_all_conditions()

        logger.info(f"Starting batch collection for {len(all_conditions)} conditions")
        logger.info(f"Target: {papers_per_condition} papers per condition")
        logger.info(f"Parallel workers: {self.max_workers}")

        # Use ThreadPoolExecutor for parallel API calls
        condition_results = []
        successful_conditions = 0
        failed_conditions = 0
        total_papers_collected = 0

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all collection tasks
                future_to_condition = {
                    executor.submit(
                        self._collect_single_condition_without_s2,
                        condition,
                        papers_per_condition,
                        min_year,
                        max_year
                    ): condition for condition in all_conditions
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_condition):
                    condition = future_to_condition[future]
                    try:
                        result = future.result()
                        condition_results.append(result)

                        if result['success']:
                            successful_conditions += 1
                            total_papers_collected += result['papers_collected']
                            logger.info(f"✓ {condition}: {result['papers_collected']} papers")
                        else:
                            failed_conditions += 1
                            logger.warning(f"✗ {condition}: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        failed_conditions += 1
                        logger.error(f"✗ {condition}: Exception during collection: {e}")
                        condition_results.append({
                            'success': False,
                            'condition': condition,
                            'papers_collected': 0,
                            'total_papers': 0,
                            'target_reached': False,
                            'error': str(e),
                            'status': 'failed'
                        })

            # Calculate final statistics
            total_time = (datetime.now() - start_time).total_seconds()

            # Quality gate check
            success_rate = (successful_conditions / len(all_conditions)) * 100 if all_conditions else 0
            quality_threshold = 80.0  # Require 80% success rate to proceed

            result = BatchCollectionResult(
                total_conditions=len(all_conditions),
                successful_conditions=successful_conditions,
                failed_conditions=failed_conditions,
                total_papers_collected=total_papers_collected,
                total_collection_time_seconds=total_time,
                conditions_results=condition_results,
                success=success_rate >= quality_threshold,
                error=None if success_rate >= quality_threshold else f"Quality gate failed: {success_rate:.1f}% success rate below {quality_threshold}% threshold"
            )

            logger.info(f"Batch collection completed in {total_time:.1f}s")
            logger.info(f"Success rate: {success_rate:.1f}% ({successful_conditions}/{len(all_conditions)})")
            logger.info(f"Total papers collected: {total_papers_collected}")
            logger.info(f"Average papers per condition: {total_papers_collected / len(all_conditions):.1f}")

            if result.success:
                logger.info("✓ Quality gate PASSED - proceeding to processing phase")
            else:
                logger.warning(f"✗ Quality gate FAILED - {result.error}")

            return result

        except Exception as e:
            total_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Batch collection failed: {e}")
            logger.error(traceback.format_exc())

            return BatchCollectionResult(
                total_conditions=len(all_conditions),
                successful_conditions=successful_conditions,
                failed_conditions=failed_conditions,
                total_papers_collected=total_papers_collected,
                total_collection_time_seconds=total_time,
                conditions_results=condition_results,
                success=False,
                error=str(e)
            )

    def _collect_single_condition_without_s2(self, condition: str, target_count: int,
                                           min_year: int, max_year: Optional[int]) -> Dict[str, Any]:
        """
        Collect papers for a single condition without Semantic Scholar enrichment.
        This is the optimized version used in batch collection.
        """
        start_time = datetime.now()
        logger.debug(f"Starting collection for '{condition}' (target: {target_count} papers)")

        try:
            # Check if we already have enough papers for this condition
            existing_count = self._count_existing_papers(condition)
            if existing_count >= target_count:
                logger.debug(f"Already have {existing_count} papers for '{condition}', skipping collection")
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
            logger.debug(f"Need {needed_papers} more papers for '{condition}' (have {existing_count})")

            # Collect papers with retry logic (without S2 enrichment)
            collection_result = self._collect_with_retry(
                condition, needed_papers, min_year, max_year
            )

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
                'status': 'completed'
            }

            return result

        except Exception as e:
            collection_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Collection failed for '{condition}': {e}")

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

    # Old condition-by-condition collection method removed - replaced by collect_all_conditions_batch()

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

    # Old session integration methods removed - replaced by batch processing architecture


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
                            use_s2_enrichment: bool = False) -> Dict[str, Any]:
    """
    Convenience function to collect papers for a single condition.
    NOTE: This uses the internal _collect_single_condition_without_s2 method.

    Args:
        condition: Medical condition to search for
        target_count: Number of papers to collect
        min_year: Minimum publication year
        max_year: Maximum publication year
        use_s2_enrichment: Whether to use Semantic Scholar enrichment (removed from pipeline)

    Returns:
        Collection result dictionary
    """
    collector = RotationPaperCollector()
    return collector._collect_single_condition_without_s2(
        condition=condition,
        target_count=target_count,
        min_year=min_year,
        max_year=max_year
    )


def collect_all_conditions_batch(papers_per_condition: int = 10,
                                min_year: int = 2015, max_year: Optional[int] = None) -> BatchCollectionResult:
    """
    Convenience function to collect papers for all 60 medical conditions in batch.

    Args:
        papers_per_condition: Target number of papers per condition
        min_year: Minimum publication year
        max_year: Maximum publication year (None for current year)

    Returns:
        BatchCollectionResult with comprehensive collection statistics
    """
    collector = RotationPaperCollector()
    return collector.collect_all_conditions_batch(
        papers_per_condition=papers_per_condition,
        min_year=min_year,
        max_year=max_year
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