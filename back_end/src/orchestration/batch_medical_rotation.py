#!/usr/bin/env python3
"""
Batch Medical Rotation Pipeline - Optimized Orchestrator

Simplified batch-oriented pipeline that processes all 60 medical conditions
in four distinct phases: collection → processing → categorization → canonical grouping.
This replaces the complex condition-by-condition approach with an efficient
batch processing workflow.

Pipeline Flow:
1. BATCH COLLECTION: Collect N papers for all 60 conditions in parallel
2. BATCH PROCESSING: Process all papers with single LLM (qwen2.5:14b) - 2x faster!
3. CATEGORIZATION: Classify interventions and conditions using focused LLM prompts
4. CANONICAL GROUPING: Cross-paper semantic merging (e.g., "vitamin D" = "Vitamin D3" = "cholecalciferol")

Features:
- 4 clear phases with natural breakpoints for recovery
- Parallel collection within VRAM constraints
- Single-model processing (qwen2.5:14b) - 2x speed improvement
- Separate categorization phase with focused prompts for higher accuracy
- Canonical name grouping for unified cross-paper analysis
- Simple session management with phase-level recovery
- Quality gates between phases
- Comprehensive progress tracking

Architecture Changes (2025-10):
- Switched from dual-model (gemma2:9b + qwen2.5:14b) to single-model (qwen2.5:14b)
- Eliminated Phase 2 consensus building complexity
- Separated categorization from extraction for simpler, more reliable prompts
- Preserved Qwen's superior extraction detail
- 2x faster processing with simpler error handling

Usage:
    # Run complete batch pipeline
    python batch_medical_rotation.py --papers-per-condition 10

    # Resume from specific phase
    python batch_medical_rotation.py --resume --start-phase categorization

    # Status check
    python batch_medical_rotation.py --status
"""

import sys
import os
import time
import signal
import argparse
import traceback
import json
import platform
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass

# Configure Ollama for optimal GPU usage (95% VRAM with RAM offload)
os.environ.setdefault("OLLAMA_NUM_GPU_LAYERS", "35")
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "30m")

# Platform-specific file locking
if platform.system() == 'Windows':
    import msvcrt
else:
    import fcntl

try:
    from ..data.config import config, setup_logging
    from .rotation_paper_collector import RotationPaperCollector, BatchCollectionResult
    from .rotation_llm_processor import RotationLLMProcessor
    from .rotation_llm_categorization import RotationLLMCategorizer
    from .rotation_semantic_grouping_integrator import RotationSemanticGroupingIntegrator
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.orchestration.rotation_paper_collector import RotationPaperCollector, BatchCollectionResult
    from back_end.src.orchestration.rotation_llm_processor import RotationLLMProcessor
    from back_end.src.orchestration.rotation_llm_categorization import RotationLLMCategorizer
    from back_end.src.orchestration.rotation_semantic_grouping_integrator import RotationSemanticGroupingIntegrator

logger = setup_logging(__name__, 'batch_medical_rotation.log')


class BatchPhase(Enum):
    """Pipeline phases for batch processing."""
    COLLECTION = "collection"
    PROCESSING = "processing"
    CATEGORIZATION = "categorization"
    CANONICAL_GROUPING = "canonical_grouping"
    COMPLETED = "completed"


@dataclass
class BatchSession:
    """Simplified session for batch pipeline."""
    session_id: str
    papers_per_condition: int
    current_phase: BatchPhase
    iteration_number: int
    start_time: str

    # Phase completion tracking
    collection_completed: bool = False
    processing_completed: bool = False
    categorization_completed: bool = False
    canonical_grouping_completed: bool = False

    # Statistics
    total_papers_collected: int = 0
    total_papers_processed: int = 0
    total_interventions_extracted: int = 0
    total_interventions_categorized: int = 0
    total_conditions_categorized: int = 0
    total_canonical_entities_created: int = 0

    # Phase results
    collection_result: Optional[Dict[str, Any]] = None
    processing_result: Optional[Dict[str, Any]] = None
    categorization_result: Optional[Dict[str, Any]] = None
    canonical_grouping_result: Optional[Dict[str, Any]] = None

    def is_completed(self) -> bool:
        """Check if entire pipeline is completed."""
        return self.current_phase == BatchPhase.COMPLETED


class BatchMedicalRotationPipeline:
    """
    Simplified batch medical rotation pipeline.
    Processes all 60 conditions in three distinct phases.
    """

    def __init__(self):
        """Initialize the batch pipeline."""
        # Initialize components
        self.paper_collector = RotationPaperCollector()
        self.llm_processor = RotationLLMProcessor()
        self.categorizer = None  # Lazy loaded
        self.dedup_integrator = RotationSemanticGroupingIntegrator()

        # Control flags
        self.shutdown_requested = False

        # Session management
        self.session_file = Path(config.data_root) / "batch_session.json"
        self.current_session: Optional[BatchSession] = None

        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Batch medical rotation pipeline initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def create_new_session(self, papers_per_condition: int) -> BatchSession:
        """Create a new batch session."""
        session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = BatchSession(
            session_id=session_id,
            papers_per_condition=papers_per_condition,
            current_phase=BatchPhase.COLLECTION,
            iteration_number=1,
            start_time=datetime.now().isoformat()
        )

        self.current_session = session
        self._save_session()

        logger.info(f"Created new batch session: {session_id}")
        return session

    def load_existing_session(self) -> Optional[BatchSession]:
        """Load existing session from file."""
        try:
            if not self.session_file.exists():
                return None

            with open(self.session_file, 'r') as f:
                data = json.load(f)

            session = BatchSession(
                session_id=data['session_id'],
                papers_per_condition=data['papers_per_condition'],
                current_phase=BatchPhase(data['current_phase']),
                iteration_number=data['iteration_number'],
                start_time=data['start_time'],
                collection_completed=data.get('collection_completed', False),
                processing_completed=data.get('processing_completed', False),
                canonical_grouping_completed=data.get('canonical_grouping_completed', False),
                total_papers_collected=data.get('total_papers_collected', 0),
                total_papers_processed=data.get('total_papers_processed', 0),
                total_interventions_extracted=data.get('total_interventions_extracted', 0),
                total_duplicates_removed=data.get('total_duplicates_removed', 0),
                collection_result=data.get('collection_result'),
                processing_result=data.get('processing_result'),
                canonical_grouping_result=data.get('canonical_grouping_result')
            )

            self.current_session = session
            logger.info(f"Loaded existing session: {session.session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def _save_session(self):
        """
        Save current session to file with platform-specific file locking.

        Uses msvcrt on Windows and fcntl on Unix/Linux to prevent race conditions
        when multiple processes try to save simultaneously.
        """
        if not self.current_session:
            return

        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'session_id': self.current_session.session_id,
                'papers_per_condition': self.current_session.papers_per_condition,
                'current_phase': self.current_session.current_phase.value,
                'iteration_number': self.current_session.iteration_number,
                'start_time': self.current_session.start_time,
                'collection_completed': self.current_session.collection_completed,
                'processing_completed': self.current_session.processing_completed,
                'canonical_grouping_completed': self.current_session.canonical_grouping_completed,
                'total_papers_collected': self.current_session.total_papers_collected,
                'total_papers_processed': self.current_session.total_papers_processed,
                'total_interventions_extracted': self.current_session.total_interventions_extracted,
                'total_duplicates_removed': self.current_session.total_duplicates_removed,
                'collection_result': self.current_session.collection_result,
                'processing_result': self.current_session.processing_result,
                'canonical_grouping_result': self.current_session.canonical_grouping_result
            }

            # Write with platform-specific file locking
            with open(self.session_file, 'w') as f:
                try:
                    # Acquire exclusive lock
                    if platform.system() == 'Windows':
                        # Windows file locking
                        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    else:
                        # Unix/Linux file locking
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                    # Write data while holding lock
                    json.dump(data, f, indent=2)

                finally:
                    # Release lock
                    if platform.system() == 'Windows':
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def run_batch_pipeline(self, papers_per_condition: int = 10,
                          resume: bool = False,
                          start_phase: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete batch pipeline.

        Args:
            papers_per_condition: Number of papers to collect per condition
            resume: Whether to resume existing session
            start_phase: Specific phase to start from (collection, processing, categorization, canonical_grouping)

        Returns:
            Pipeline execution summary
        """
        try:
            # Load or create session
            if resume:
                session = self.load_existing_session()
                if not session:
                    logger.warning("No existing session found, creating new session")
                    session = self.create_new_session(papers_per_condition)
                else:
                    logger.info(f"Resumed session: {session.session_id}")
            else:
                session = self.create_new_session(papers_per_condition)

            # Override start phase if specified
            if start_phase:
                try:
                    session.current_phase = BatchPhase(start_phase)
                    logger.info(f"Starting from specified phase: {start_phase}")
                except ValueError:
                    logger.error(f"Invalid phase: {start_phase}")
                    return {'success': False, 'error': f'Invalid phase: {start_phase}'}

            logger.info("="*60)
            logger.info("BATCH MEDICAL ROTATION PIPELINE STARTING")
            logger.info("="*60)
            logger.info(f"Session: {session.session_id}")
            logger.info(f"Papers per condition: {session.papers_per_condition}")
            logger.info(f"Starting phase: {session.current_phase.value}")
            logger.info(f"Total conditions: 60 (12 specialties × 5 conditions)")

            # Execute pipeline phases
            pipeline_start = time.time()

            # Phase 1: Batch Collection
            if session.current_phase == BatchPhase.COLLECTION and not session.collection_completed:
                logger.info("\n" + "="*40)
                logger.info("PHASE 1: BATCH COLLECTION")
                logger.info("="*40)

                collection_result = self._run_collection_phase(session)
                if not collection_result['success']:
                    return collection_result

                session.collection_completed = True
                session.current_phase = BatchPhase.PROCESSING
                self._save_session()

            # Phase 2: Batch Processing
            if session.current_phase == BatchPhase.PROCESSING and not session.processing_completed:
                logger.info("\n" + "="*40)
                logger.info("PHASE 2: BATCH PROCESSING")
                logger.info("="*40)

                processing_result = self._run_processing_phase(session)
                if not processing_result['success']:
                    return processing_result

                session.processing_completed = True
                session.current_phase = BatchPhase.CATEGORIZATION
                self._save_session()

            # Phase 2.5: Categorization
            if session.current_phase == BatchPhase.CATEGORIZATION and not session.categorization_completed:
                logger.info("\n" + "="*40)
                logger.info("PHASE 2.5: CATEGORIZATION")
                logger.info("="*40)

                categorization_result = self._run_categorization_phase(session)
                if not categorization_result['success']:
                    return categorization_result

                session.categorization_completed = True
                session.current_phase = BatchPhase.CANONICAL_GROUPING
                self._save_session()

            # Phase 3: Canonical Name Grouping
            if session.current_phase == BatchPhase.CANONICAL_GROUPING and not session.canonical_grouping_completed:
                logger.info("\n" + "="*40)
                logger.info("PHASE 3: CANONICAL NAME GROUPING")
                logger.info("="*40)

                canonical_grouping_result = self._run_canonical_grouping_phase(session)
                if not canonical_grouping_result['success']:
                    return canonical_grouping_result

                session.canonical_grouping_completed = True
                session.current_phase = BatchPhase.COMPLETED
                self._save_session()

            # Pipeline completed
            total_time = time.time() - pipeline_start

            logger.info("\n" + "="*60)
            logger.info("BATCH PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"Papers collected: {session.total_papers_collected}")
            logger.info(f"Papers processed: {session.total_papers_processed}")
            logger.info(f"Interventions extracted: {session.total_interventions_extracted}")
            logger.info(f"Interventions categorized: {session.total_interventions_categorized}")
            logger.info(f"Conditions categorized: {session.total_conditions_categorized}")
            logger.info(f"Canonical entities created: {session.total_canonical_entities_created}")

            # Prepare for next iteration
            session.iteration_number += 1
            session.current_phase = BatchPhase.COLLECTION
            session.collection_completed = False
            session.processing_completed = False
            session.categorization_completed = False
            session.canonical_grouping_completed = False
            self._save_session()

            logger.info(f"\nPrepared for iteration {session.iteration_number}")

            return {
                'success': True,
                'session_id': session.session_id,
                'iteration_completed': session.iteration_number - 1,
                'total_time_seconds': total_time,
                'statistics': {
                    'papers_collected': session.total_papers_collected,
                    'papers_processed': session.total_papers_processed,
                    'interventions_extracted': session.total_interventions_extracted,
                    'interventions_categorized': session.total_interventions_categorized,
                    'conditions_categorized': session.total_conditions_categorized,
                    'canonical_entities_created': session.total_canonical_entities_created
                }
            }

        except Exception as e:
            logger.error(f"Batch pipeline failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'phase': session.current_phase.value if session else 'unknown'
            }

    def _run_collection_phase(self, session: BatchSession) -> Dict[str, Any]:
        """Run the batch collection phase."""
        logger.info(f"Collecting {session.papers_per_condition} papers for all 60 conditions...")

        try:
            if self.shutdown_requested:
                return {'success': False, 'error': 'Shutdown requested during collection'}

            # Run batch collection
            collection_result = self.paper_collector.collect_all_conditions_batch(
                papers_per_condition=session.papers_per_condition,
                min_year=2015
            )

            # Update session with results
            session.collection_result = {
                'total_conditions': collection_result.total_conditions,
                'successful_conditions': collection_result.successful_conditions,
                'failed_conditions': collection_result.failed_conditions,
                'total_papers_collected': collection_result.total_papers_collected,
                'collection_time_seconds': collection_result.total_collection_time_seconds,
                'success_rate': (collection_result.successful_conditions / collection_result.total_conditions) * 100,
                'quality_gate_passed': collection_result.success
            }

            session.total_papers_collected = collection_result.total_papers_collected

            if not collection_result.success:
                logger.error(f"Collection phase failed: {collection_result.error}")
                return {
                    'success': False,
                    'error': f'Collection quality gate failed: {collection_result.error}',
                    'phase': 'collection'
                }

            logger.info("[SUCCESS] Collection phase completed successfully")
            logger.info(f"  Papers collected: {collection_result.total_papers_collected}")
            logger.info(f"  Success rate: {(collection_result.successful_conditions / collection_result.total_conditions) * 100:.1f}%")

            return {'success': True, 'result': session.collection_result}

        except Exception as e:
            logger.error(f"Collection phase failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'collection'}

    def _run_processing_phase(self, session: BatchSession) -> Dict[str, Any]:
        """Run the batch processing phase with sequential dual LLM."""
        logger.info("Processing all collected papers with sequential dual LLM...")

        try:
            if self.shutdown_requested:
                return {'success': False, 'error': 'Shutdown requested during processing'}

            # Run batch LLM processing using the new method
            processing_result = self.llm_processor.process_all_papers_batch()

            # Update session with results
            session.processing_result = {
                'total_papers_found': processing_result.get('total_papers_found', 0),
                'papers_processed': processing_result.get('papers_processed', 0),
                'papers_failed': processing_result.get('papers_failed', 0),
                'interventions_extracted': processing_result.get('interventions_extracted', 0),
                'processing_time_seconds': processing_result.get('processing_time_seconds', 0),
                'success_rate': processing_result.get('success_rate', 0),
                'model_statistics': processing_result.get('model_statistics', {}),
                'interventions_by_category': processing_result.get('interventions_by_category', {}),
                'failed_papers_count': len(processing_result.get('failed_papers', []))
            }

            session.total_papers_processed = processing_result.get('papers_processed', 0)
            session.total_interventions_extracted = processing_result.get('interventions_extracted', 0)

            if not processing_result['success']:
                logger.error(f"Processing phase failed: {processing_result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': f"Processing failed: {processing_result.get('error', 'Unknown error')}",
                    'phase': 'processing'
                }

            logger.info("[SUCCESS] Processing phase completed successfully")
            logger.info(f"  Papers processed: {session.total_papers_processed}")
            logger.info(f"  Interventions extracted: {session.total_interventions_extracted}")
            logger.info(f"  Success rate: {processing_result.get('success_rate', 0):.1f}%")

            return {'success': True, 'result': session.processing_result}

        except Exception as e:
            logger.error(f"Processing phase failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'processing'}

    def _run_categorization_phase(self, session: BatchSession) -> Dict[str, Any]:
        """Run the categorization phase for interventions and conditions."""
        logger.info("Categorizing interventions and conditions...")

        try:
            if self.shutdown_requested:
                return {'success': False, 'error': 'Shutdown requested during categorization'}

            # Lazy load categorizer
            if self.categorizer is None:
                self.categorizer = RotationLLMCategorizer(batch_size=20)

            # Categorize all interventions and conditions
            categorization_result = self.categorizer.categorize_all()

            # Update session with results
            session.categorization_result = categorization_result
            session.total_interventions_categorized = categorization_result['interventions']['success']
            session.total_conditions_categorized = categorization_result['conditions']['success']

            intervention_success = categorization_result['interventions']['success']
            intervention_total = categorization_result['interventions']['total']
            condition_success = categorization_result['conditions']['success']
            condition_total = categorization_result['conditions']['total']

            logger.info("[SUCCESS] Categorization phase completed successfully")
            logger.info(f"  Interventions categorized: {intervention_success}/{intervention_total}")
            logger.info(f"  Conditions categorized: {condition_success}/{condition_total}")

            return {'success': True, 'result': session.categorization_result}

        except Exception as e:
            logger.error(f"Categorization phase failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'categorization'}

    def _run_canonical_grouping_phase(self, session: BatchSession) -> Dict[str, Any]:
        """
        Run Phase 3: Cross-paper canonical name grouping.

        With single-model extraction (qwen2.5:14b only), same-paper duplicates cannot exist.
        This phase ONLY performs cross-paper semantic analysis to group semantically equivalent
        interventions under canonical names (e.g., "vitamin D" = "Vitamin D3" = "cholecalciferol").
        """
        logger.info("Running Phase 3: Cross-paper canonical name grouping...")

        try:
            if self.shutdown_requested:
                return {'success': False, 'error': 'Shutdown requested during canonical grouping'}

            # Run semantic grouping
            grouping_result = self.dedup_integrator.group_all_data_semantically_batch()

            # Update session with results
            total_processed = grouping_result.get('interventions_processed', 0)
            canonical_entities_created = grouping_result.get('canonical_entities_created', 0)

            session.canonical_grouping_result = {
                'total_interventions_processed': total_processed,
                'canonical_entities_created': canonical_entities_created,
                'interventions_grouped': grouping_result.get('total_merged', 0),
                'processing_time_seconds': grouping_result.get('processing_time_seconds', 0),
                'semantic_groups_found': grouping_result.get('duplicate_groups_found', 0),
                'method': grouping_result.get('method', 'llm_semantic_grouping')
            }

            session.total_canonical_entities_created = canonical_entities_created

            if not grouping_result['success']:
                logger.error(f"Canonical grouping phase failed: {grouping_result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': f"Canonical grouping failed: {grouping_result.get('error', 'Unknown error')}",
                    'phase': 'canonical_grouping'
                }

            logger.info("[SUCCESS] Canonical name grouping phase completed successfully")
            logger.info(f"  Interventions analyzed: {total_processed}")
            logger.info(f"  Canonical entities created: {canonical_entities_created}")
            logger.info(f"  Semantic groups found: {grouping_result.get('duplicate_groups_found', 0)}")
            logger.info(f"  Method: {grouping_result.get('method', 'llm_semantic_grouping')}")

            return {'success': True, 'result': session.canonical_grouping_result}

        except Exception as e:
            logger.error(f"Canonical grouping phase failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'canonical_grouping'}

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        session = self.load_existing_session()

        if not session:
            return {
                'session_exists': False,
                'message': 'No active session found'
            }

        return {
            'session_exists': True,
            'session_id': session.session_id,
            'iteration_number': session.iteration_number,
            'current_phase': session.current_phase.value,
            'papers_per_condition': session.papers_per_condition,
            'progress': {
                'collection_completed': session.collection_completed,
                'processing_completed': session.processing_completed,
                'canonical_grouping_completed': session.canonical_grouping_completed,
                'pipeline_completed': session.is_completed()
            },
            'statistics': {
                'papers_collected': session.total_papers_collected,
                'papers_processed': session.total_papers_processed,
                'interventions_extracted': session.total_interventions_extracted,
                'canonical_entities_created': session.total_canonical_entities_created
            },
            'phase_results': {
                'collection': session.collection_result,
                'processing': session.processing_result,
                'categorization': session.categorization_result,
                'canonical_grouping': session.canonical_grouping_result
            }
        }


def main():
    """Command line interface for batch medical rotation pipeline."""
    parser = argparse.ArgumentParser(
        description="Batch Medical Rotation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete batch pipeline
  python batch_medical_rotation.py --papers-per-condition 10

  # Resume existing session
  python batch_medical_rotation.py --resume

  # Resume from specific phase
  python batch_medical_rotation.py --resume --start-phase processing

  # Check status
  python batch_medical_rotation.py --status
        """
    )

    parser.add_argument('--papers-per-condition', type=int, default=10,
                        help='Number of papers to collect per condition (default: 10)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume existing session')
    parser.add_argument('--start-phase', choices=['collection', 'processing', 'categorization', 'canonical_grouping'],
                        help='Specific phase to start from (use with --resume)')
    parser.add_argument('--status', action='store_true',
                        help='Show current pipeline status')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    try:
        pipeline = BatchMedicalRotationPipeline()

        if args.status:
            status = pipeline.get_status()
            print(json.dumps(status, indent=2))
            return

        # Run pipeline
        result = pipeline.run_batch_pipeline(
            papers_per_condition=args.papers_per_condition,
            resume=args.resume,
            start_phase=args.start_phase
        )

        if result['success']:
            print(f"[SUCCESS] Batch pipeline completed successfully")
            print(f"Session: {result['session_id']}")
            print(f"Iteration: {result['iteration_completed']}")
            print(f"Total time: {result['total_time_seconds']:.1f} seconds")
            print("\nStatistics:")
            for key, value in result['statistics'].items():
                print(f"  {key}: {value}")
        else:
            print(f"[FAILED] Batch pipeline failed: {result['error']}")
            if 'phase' in result:
                print(f"Failed during: {result['phase']} phase")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[FAILED] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[FAILED] Pipeline failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()