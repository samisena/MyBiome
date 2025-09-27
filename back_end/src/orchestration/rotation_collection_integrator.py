#!/usr/bin/env python3
"""
Rotation Collection Integrator

Simple integration between paper collection and session management.
Provides basic coordination and progress tracking.
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


class RotationCollectionIntegrator:
    """Simple integration between paper collection and session management."""

    def __init__(self, session_mgr: RotationSessionManager = None):
        """Initialize the collection integrator."""
        self.session_mgr = session_mgr or session_manager
        self.paper_collector = RotationPaperCollector()
        self.max_retries = 3

    def collect_current_condition(self) -> Dict[str, Any]:
        """Collect papers for the current condition in the rotation session."""
        if not self.session_mgr.session:
            raise Exception("No active rotation session found")

        session = self.session_mgr.session
        condition = session.current_condition
        target_count = session.papers_per_condition

        logger.info(f"Collecting {target_count} papers for: {condition}")

        # Set interruption state for collection phase
        self.session_mgr.set_interruption_state(phase=PipelinePhase.COLLECTION)

        try:
            # Simple retry mechanism
            for attempt in range(self.max_retries):
                try:
                    result = self.paper_collector.collect_condition_papers(
                        condition=condition,
                        target_count=target_count,
                        min_year=2015,
                        use_s2_enrichment=True
                    )

                    if result['success']:
                        # Update session progress
                        self.session_mgr.update_progress(
                            papers_collected=result['papers_collected']
                        )
                        self.session_mgr.clear_interruption_state()
                        logger.info(f"Collection completed: {result['papers_collected']} papers")
                        return result

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    logger.warning(f"Collection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(30 * (attempt + 1))  # Progressive delay

            return {'success': False, 'error': 'Max retries exceeded'}

        except Exception as e:
            logger.error(f"Collection failed for '{condition}': {e}")
            self.session_mgr.mark_condition_failed(str(e))
            return {'success': False, 'error': str(e)}

    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status from session manager."""
        if not self.session_mgr.session:
            return {'error': 'No active session'}

        session = self.session_mgr.session
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

    def resume_interrupted_collection(self) -> Optional[Dict[str, Any]]:
        """Resume collection if interrupted during collection phase."""
        if not self.session_mgr.session or not self.session_mgr.session.interruption_state:
            return None

        interruption = self.session_mgr.session.interruption_state
        if interruption.phase != PipelinePhase.COLLECTION.value:
            return None

        logger.info(f"Resuming interrupted collection for '{self.session_mgr.session.current_condition}'")
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
    if result['success']:
        print(f"Papers collected: {result.get('papers_collected', 0)}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    print("\nTest completed.")
