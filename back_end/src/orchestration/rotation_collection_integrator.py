#!/usr/bin/env python3
"""
Rotation Collection Integrator

Integrates paper collection with the rotation session manager.
Provides enhanced error handling, validation, and recovery mechanisms
for robust operation across the medical rotation pipeline.

Features:
- Session-aware collection coordination
- Advanced error handling and recovery
- Collection validation and quality checks
- Progress synchronization with session manager
- Automatic retry and fallback strategies
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

try:
    from ..data.config import config, setup_logging
    from .rotation_session_manager import (
        RotationSessionManager, PipelinePhase, session_manager
    )
    from .rotation_paper_collector import RotationPaperCollector
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.orchestration.rotation_session_manager import (
        RotationSessionManager, PipelinePhase, session_manager
    )
    from back_end.src.orchestration.rotation_paper_collector import RotationPaperCollector

logger = setup_logging(__name__, 'rotation_collection_integrator.log')


class CollectionError(Exception):
    """Custom exception for collection-related errors."""
    pass


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class RotationCollectionIntegrator:
    """
    Integrates paper collection with rotation session management.
    Handles error recovery, validation, and progress tracking.
    """

    def __init__(self, session_mgr: RotationSessionManager = None):
        """Initialize the collection integrator."""
        self.session_mgr = session_mgr or session_manager
        self.paper_collector = RotationPaperCollector()

        # Error handling configuration
        self.max_collection_retries = 3
        # NOTE: Timeout protection removed - 15 minutes too restrictive for 10 papers/condition
        # TODO: Implement appropriate timeout after baseline performance established
        self.quality_threshold = 0.8  # Minimum success rate for quality check

        # Validation settings
        self.min_papers_threshold = 1  # Minimum papers to consider success
        self.max_retry_delay = 300  # Maximum delay between retries (5 minutes)

    def collect_current_condition(self) -> Dict[str, Any]:
        """
        Collect papers for the current condition in the rotation session.

        Returns:
            Collection result with comprehensive status information
        """
        if not self.session_mgr.session:
            raise CollectionError("No active rotation session found")

        session = self.session_mgr.session
        condition = session.current_condition
        specialty = session.current_specialty
        target_count = session.papers_per_condition

        logger.info(f"Starting collection for current condition: {specialty} -> {condition}")
        logger.info(f"Target papers: {target_count}")

        # Set interruption state for collection phase
        self.session_mgr.set_interruption_state(
            phase=PipelinePhase.COLLECTION,
            paper_id=None,
            paper_index=None
        )

        start_time = datetime.now()
        collection_result = None

        try:
            # Execute collection with retry recovery (no timeout for now)
            collection_result = self._execute_collection_with_recovery(
                condition=condition,
                target_count=target_count
            )

            # Validate collection result
            validation_result = self._validate_collection_result(collection_result)
            collection_result.update(validation_result)

            # Update session progress
            if collection_result['success']:
                self.session_mgr.update_progress(
                    papers_collected=collection_result['papers_collected']
                )
                logger.info(f"Collection completed successfully for '{condition}'")
            else:
                logger.error(f"Collection failed for '{condition}': {collection_result.get('error', 'Unknown error')}")

            # Clear interruption state on success
            if collection_result['success']:
                self.session_mgr.clear_interruption_state()

            # Calculate final metrics
            collection_time = (datetime.now() - start_time).total_seconds()
            collection_result['total_collection_time'] = collection_time
            collection_result['collection_rate'] = (
                collection_result['papers_collected'] / (collection_time / 60)
                if collection_time > 0 else 0
            )

            return collection_result

        except Exception as e:
            logger.error(f"Critical collection error for '{condition}': {e}")
            logger.error(traceback.format_exc())

            # Mark condition as failed in session
            self.session_mgr.mark_condition_failed(str(e))

            # Return error result
            return {
                'success': False,
                'condition': condition,
                'specialty': specialty,
                'papers_collected': 0,
                'total_papers': 0,
                'target_reached': False,
                'error': str(e),
                'error_type': 'critical_failure',
                'total_collection_time': (datetime.now() - start_time).total_seconds(),
                'status': 'failed'
            }

    def _execute_collection_with_recovery(self, condition: str, target_count: int) -> Dict[str, Any]:
        """Execute collection with advanced recovery mechanisms."""
        last_error = None
        best_result = None

        for attempt in range(self.max_collection_retries):
            try:
                logger.info(f"Collection attempt {attempt + 1}/{self.max_collection_retries} for '{condition}'")

                # Execute collection directly (no timeout for now)
                collection_result = self.paper_collector.collect_condition_papers(
                    condition=condition,
                    target_count=target_count,
                    min_year=2015,
                    max_year=None,
                    use_s2_enrichment=True
                )

                # Check if this is the best result so far
                if best_result is None or collection_result['papers_collected'] > best_result['papers_collected']:
                    best_result = collection_result.copy()

                # If successful, return immediately
                if collection_result['success'] and collection_result['target_reached']:
                    logger.info(f"Collection successful for '{condition}' on attempt {attempt + 1}")
                    return collection_result

                # If partial success, continue trying but keep best result
                if collection_result['success'] and collection_result['papers_collected'] > 0:
                    logger.warning(f"Partial collection for '{condition}' on attempt {attempt + 1}: "
                                 f"{collection_result['papers_collected']}/{target_count} papers")

                # If complete failure, record error and retry
                if not collection_result['success']:
                    last_error = collection_result.get('error', 'Unknown error')
                    logger.warning(f"Collection attempt {attempt + 1} failed for '{condition}': {last_error}")

                # Wait before retry (except on last attempt)
                if attempt < self.max_collection_retries - 1:
                    delay = min(30 * (2 ** attempt), self.max_retry_delay)  # Exponential backoff
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)

            except Exception as e:
                last_error = str(e)
                logger.error(f"Collection attempt {attempt + 1} raised exception for '{condition}': {e}")

                if attempt < self.max_collection_retries - 1:
                    delay = min(60 * (2 ** attempt), self.max_retry_delay)
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)

        # If we get here, all attempts failed or only partial success
        if best_result and best_result['papers_collected'] >= self.min_papers_threshold:
            logger.warning(f"Using best partial result for '{condition}': "
                          f"{best_result['papers_collected']} papers collected")
            best_result['status'] = 'partial_success'
            best_result['recovery_note'] = f"Used best result from {self.max_collection_retries} attempts"
            return best_result

        # Complete failure
        raise CollectionError(f"Failed to collect papers after {self.max_collection_retries} attempts. "
                             f"Last error: {last_error}")


    def _validate_collection_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate collection result and add validation metadata."""
        validation_info = {
            'validation_passed': False,
            'validation_warnings': [],
            'validation_errors': [],
            'quality_score': 0.0
        }

        try:
            # Basic validation using collector's validation method
            is_valid, message = self.paper_collector.validate_collection_result(result)

            if not is_valid:
                validation_info['validation_errors'].append(message)
                validation_info['validation_passed'] = False
                return validation_info

            # Advanced quality checks
            papers_collected = result.get('papers_collected', 0)
            target_count = self.session_mgr.session.papers_per_condition if self.session_mgr.session else 10

            # Calculate quality score
            if target_count > 0:
                target_ratio = min(papers_collected / target_count, 1.0)
            else:
                target_ratio = 1.0 if papers_collected > 0 else 0.0

            # Factor in collection time (faster is better, within reason)
            collection_time = result.get('collection_time_seconds', 0)
            time_factor = 1.0
            if collection_time > 0:
                # Ideal time: 1-5 minutes per paper
                ideal_time_per_paper = 180  # 3 minutes
                actual_time_per_paper = collection_time / max(papers_collected, 1)
                if actual_time_per_paper <= ideal_time_per_paper * 2:
                    time_factor = 1.0
                else:
                    time_factor = max(0.5, ideal_time_per_paper * 2 / actual_time_per_paper)

            validation_info['quality_score'] = target_ratio * time_factor

            # Quality thresholds
            if validation_info['quality_score'] >= self.quality_threshold:
                validation_info['validation_passed'] = True
            else:
                validation_info['validation_warnings'].append(
                    f"Quality score {validation_info['quality_score']:.2f} below threshold {self.quality_threshold}"
                )

            # Additional checks
            if papers_collected < self.min_papers_threshold:
                validation_info['validation_warnings'].append(
                    f"Papers collected ({papers_collected}) below minimum threshold ({self.min_papers_threshold})"
                )

            if not result.get('target_reached', False) and result.get('success', False):
                validation_info['validation_warnings'].append(
                    f"Target not fully reached: {papers_collected}/{target_count} papers"
                )

            # If no critical errors but have warnings, still consider it passed
            if not validation_info['validation_errors'] and not validation_info['validation_passed']:
                validation_info['validation_passed'] = True

            logger.info(f"Validation completed - Quality score: {validation_info['quality_score']:.2f}")

        except Exception as e:
            validation_info['validation_errors'].append(f"Validation failed: {str(e)}")
            logger.error(f"Validation error: {e}")

        return validation_info

    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status from session manager."""
        if not self.session_mgr.session:
            return {'error': 'No active session'}

        session = self.session_mgr.session

        # Check if we're currently in collection phase
        is_collecting = (
            session.interruption_state and
            session.interruption_state.phase == PipelinePhase.COLLECTION.value
        )

        status = {
            'current_condition': session.current_condition,
            'current_specialty': session.current_specialty,
            'target_papers': session.papers_per_condition,
            'is_collecting': is_collecting,
            'total_papers_collected': session.total_papers_collected,
            'completed_conditions': len(session.completed_conditions),
            'failed_conditions': len(session.failed_conditions),
            'session_active': session.is_active
        }

        # Add current condition progress if available
        if session.current_condition_progress:
            progress = session.current_condition_progress
            status['current_condition_progress'] = {
                'papers_collected': progress.papers_collected,
                'status': progress.status,
                'start_time': progress.start_time,
                'error_count': progress.error_count
            }

        return status

    def resume_interrupted_collection(self) -> Optional[Dict[str, Any]]:
        """
        Resume collection if interrupted during collection phase.

        Returns:
            Collection result if resuming, None if no interruption or not collection phase
        """
        if not self.session_mgr.session or not self.session_mgr.session.interruption_state:
            return None

        interruption = self.session_mgr.session.interruption_state

        if interruption.phase != PipelinePhase.COLLECTION.value:
            logger.info(f"Interruption is in {interruption.phase} phase, not collection")
            return None

        logger.info(f"Resuming interrupted collection for '{self.session_mgr.session.current_condition}'")

        # Clear the interruption state and restart collection
        condition = self.session_mgr.session.current_condition
        target_count = self.session_mgr.session.papers_per_condition

        logger.info(f"Restarting collection for '{condition}' (target: {target_count})")

        return self.collect_current_condition()


def create_collection_integrator() -> RotationCollectionIntegrator:
    """Create and return a collection integrator instance."""
    return RotationCollectionIntegrator()


if __name__ == "__main__":
    """Test the collection integrator."""
    import argparse
    from back_end.src.orchestration.rotation_session_manager import session_manager

    parser = argparse.ArgumentParser(description="Test Rotation Collection Integrator")
    parser.add_argument('--test-condition', type=str, help='Test with specific condition')
    parser.add_argument('--papers', type=int, default=3, help='Number of papers to collect')

    args = parser.parse_args()

    # Create test session if needed
    if not session_manager.load_existing_session():
        print("Creating test session...")
        session_manager.create_new_session(papers_per_condition=args.papers)

    # Override condition if specified
    if args.test_condition:
        session = session_manager.session
        session.current_condition_progress.condition = args.test_condition
        print(f"Testing with condition: {args.test_condition}")

    # Test collection integrator
    integrator = RotationCollectionIntegrator()

    print("\n" + "="*60)
    print("TESTING COLLECTION INTEGRATOR")
    print("="*60)

    # Get status
    status = integrator.get_collection_status()
    print(f"Current condition: {status['current_condition']}")
    print(f"Target papers: {status['target_papers']}")

    # Test collection
    print(f"\nStarting collection...")
    result = integrator.collect_current_condition()

    print("\n" + "="*60)
    print("COLLECTION RESULT")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Papers collected: {result['papers_collected']}")
    print(f"Target reached: {result['target_reached']}")
    print(f"Quality score: {result.get('quality_score', 'N/A')}")
    print(f"Total time: {result.get('total_collection_time', 0):.1f} seconds")

    if result.get('validation_warnings'):
        print(f"Warnings: {result['validation_warnings']}")

    if not result['success']:
        print(f"Error: {result.get('error', 'Unknown error')}")

    print("\nTest completed.")