#!/usr/bin/env python3
"""
Batch Medical Rotation Pipeline - Optimized Orchestrator

Simplified batch-oriented pipeline that processes all 60 medical conditions
in four distinct phases: collection → processing → semantic normalization → group categorization.
This replaces the complex condition-by-condition approach with an efficient
batch processing workflow.

Pipeline Flow:
1. BATCH COLLECTION: Collect N papers for all 60 conditions in parallel
2. BATCH PROCESSING: Process all papers with single LLM (qwen3:14b) - 2x faster!
3. SEMANTIC NORMALIZATION: Cross-paper semantic merging (e.g., "vitamin D" = "Vitamin D3" = "cholecalciferol")
4. GROUP CATEGORIZATION: Classify canonical groups (not individual interventions) using semantic context

Features:
- 4 clear phases with natural breakpoints for recovery
- Parallel collection within VRAM constraints
- Single-model processing (qwen2.5:14b) - 2x speed improvement
- Separate categorization phase with focused prompts for higher accuracy
- Canonical name grouping for unified cross-paper analysis
- Simple session management with phase-level recovery
- Quality gates between phases
- Comprehensive progress tracking
- **Continuous mode**: Infinite loop that restarts Phase 1 after Phase 3 completion
- **Iteration tracking**: Full history of all completed iterations with statistics
- **Thermal protection**: Configurable delay between iterations

Architecture Changes (2025-10):
- Switched from dual-model (gemma2:9b + qwen2.5:14b) to single-model (qwen3:14b)
- Eliminated Phase 2 consensus building complexity
- Moved categorization AFTER semantic normalization (Phase 2.5 → Phase 3.5)
- Group-based categorization: categorize semantic groups instead of individual interventions
- 80% reduction in LLM calls for categorization (10,000 interventions → ~2,000 groups)
- Better semantic context: group name + member names inform categorization
- Preserved Qwen's superior extraction detail
- 2x faster processing with simpler error handling
- Added continuous mode for unattended multi-iteration data collection

Usage:
    # Run single iteration
    python batch_medical_rotation.py --papers-per-condition 10

    # Run continuous mode (infinite loop until Ctrl+C)
    python batch_medical_rotation.py --papers-per-condition 10 --continuous

    # Run limited iterations (e.g., 5 complete cycles)
    python batch_medical_rotation.py --papers-per-condition 10 --continuous --max-iterations 5

    # Custom delay between iterations (5 minutes)
    python batch_medical_rotation.py --papers-per-condition 10 --continuous --iteration-delay 300

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
from dataclasses import dataclass, field

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
    from .rotation_semantic_grouping_integrator import RotationSemanticGroupingIntegrator
    from .rotation_group_categorization import RotationGroupCategorizer
    from .rotation_mechanism_clustering import RotationMechanismClusterer
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.orchestration.rotation_paper_collector import RotationPaperCollector, BatchCollectionResult
    from back_end.src.orchestration.rotation_llm_processor import RotationLLMProcessor
    from back_end.src.orchestration.rotation_semantic_grouping_integrator import RotationSemanticGroupingIntegrator
    from back_end.src.orchestration.rotation_group_categorization import RotationGroupCategorizer
    from back_end.src.orchestration.rotation_mechanism_clustering import RotationMechanismClusterer

logger = setup_logging(__name__, 'batch_medical_rotation.log')


class BatchPhase(Enum):
    """Pipeline phases for batch processing."""
    COLLECTION = "collection"
    PROCESSING = "processing"
    SEMANTIC_NORMALIZATION = "semantic_normalization"  # Phase 3 (formerly CANONICAL_GROUPING)
    GROUP_CATEGORIZATION = "group_categorization"      # Phase 3.5 (NEW - formerly CATEGORIZATION)
    MECHANISM_CLUSTERING = "mechanism_clustering"      # Phase 3.6 (NEW)
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
    semantic_normalization_completed: bool = False  # Phase 3 (formerly canonical_grouping_completed)
    group_categorization_completed: bool = False     # Phase 3.5 (NEW - formerly categorization_completed)
    mechanism_clustering_completed: bool = False     # Phase 3.6 (NEW)

    # Statistics (current iteration)
    total_papers_collected: int = 0
    total_papers_processed: int = 0
    total_interventions_extracted: int = 0
    total_canonical_groups_created: int = 0         # Phase 3
    total_groups_categorized: int = 0               # Phase 3.5
    total_interventions_categorized: int = 0        # Phase 3.5 (via propagation)
    total_orphans_categorized: int = 0              # Phase 3.5 (fallback)
    total_mechanisms_processed: int = 0             # Phase 3.6 (NEW)
    total_mechanism_clusters: int = 0               # Phase 3.6 (NEW)

    # Continuous mode settings
    continuous_mode: bool = False
    max_iterations: Optional[int] = None
    iteration_delay_seconds: float = 60.0

    # Iteration history (tracks each completed iteration)
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)

    # Phase results
    collection_result: Optional[Dict[str, Any]] = None
    processing_result: Optional[Dict[str, Any]] = None
    semantic_normalization_result: Optional[Dict[str, Any]] = None  # Phase 3
    group_categorization_result: Optional[Dict[str, Any]] = None     # Phase 3.5
    mechanism_clustering_result: Optional[Dict[str, Any]] = None     # Phase 3.6 (NEW)

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
        self.dedup_integrator = RotationSemanticGroupingIntegrator()
        self.group_categorizer = None  # Lazy loaded (Phase 3.5)
        self.mechanism_clusterer = None  # Lazy loaded (Phase 3.6)

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

    def create_new_session(self, papers_per_condition: int,
                          continuous_mode: bool = False,
                          max_iterations: Optional[int] = None,
                          iteration_delay: float = 60.0) -> BatchSession:
        """Create a new batch session."""
        session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = BatchSession(
            session_id=session_id,
            papers_per_condition=papers_per_condition,
            current_phase=BatchPhase.COLLECTION,
            iteration_number=1,
            start_time=datetime.now().isoformat(),
            continuous_mode=continuous_mode,
            max_iterations=max_iterations,
            iteration_delay_seconds=iteration_delay
        )

        self.current_session = session
        self._save_session()

        logger.info(f"Created new batch session: {session_id}")
        if continuous_mode:
            logger.info(f"Continuous mode enabled: max_iterations={max_iterations or 'unlimited'}, delay={iteration_delay}s")
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
                semantic_normalization_completed=data.get('semantic_normalization_completed', False),
                group_categorization_completed=data.get('group_categorization_completed', False),
                mechanism_clustering_completed=data.get('mechanism_clustering_completed', False),
                total_papers_collected=data.get('total_papers_collected', 0),
                total_papers_processed=data.get('total_papers_processed', 0),
                total_interventions_extracted=data.get('total_interventions_extracted', 0),
                total_canonical_groups_created=data.get('total_canonical_groups_created', 0),
                total_groups_categorized=data.get('total_groups_categorized', 0),
                total_interventions_categorized=data.get('total_interventions_categorized', 0),
                total_orphans_categorized=data.get('total_orphans_categorized', 0),
                total_mechanisms_processed=data.get('total_mechanisms_processed', 0),
                total_mechanism_clusters=data.get('total_mechanism_clusters', 0),
                continuous_mode=data.get('continuous_mode', False),
                max_iterations=data.get('max_iterations'),
                iteration_delay_seconds=data.get('iteration_delay_seconds', 60.0),
                iteration_history=data.get('iteration_history', []),
                collection_result=data.get('collection_result'),
                processing_result=data.get('processing_result'),
                semantic_normalization_result=data.get('semantic_normalization_result'),
                group_categorization_result=data.get('group_categorization_result'),
                mechanism_clustering_result=data.get('mechanism_clustering_result')
            )

            self.current_session = session
            logger.info(f"Loaded existing session: {session.session_id}")
            if session.continuous_mode:
                logger.info(f"Continuous mode: max_iterations={session.max_iterations or 'unlimited'}, iteration={session.iteration_number}")
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
                'semantic_normalization_completed': self.current_session.semantic_normalization_completed,
                'group_categorization_completed': self.current_session.group_categorization_completed,
                'mechanism_clustering_completed': self.current_session.mechanism_clustering_completed,
                'total_papers_collected': self.current_session.total_papers_collected,
                'total_papers_processed': self.current_session.total_papers_processed,
                'total_interventions_extracted': self.current_session.total_interventions_extracted,
                'total_canonical_groups_created': self.current_session.total_canonical_groups_created,
                'total_groups_categorized': self.current_session.total_groups_categorized,
                'total_interventions_categorized': self.current_session.total_interventions_categorized,
                'total_orphans_categorized': self.current_session.total_orphans_categorized,
                'total_mechanisms_processed': self.current_session.total_mechanisms_processed,
                'total_mechanism_clusters': self.current_session.total_mechanism_clusters,
                'continuous_mode': self.current_session.continuous_mode,
                'max_iterations': self.current_session.max_iterations,
                'iteration_delay_seconds': self.current_session.iteration_delay_seconds,
                'iteration_history': self.current_session.iteration_history,
                'collection_result': self.current_session.collection_result,
                'processing_result': self.current_session.processing_result,
                'semantic_normalization_result': self.current_session.semantic_normalization_result,
                'group_categorization_result': self.current_session.group_categorization_result,
                'mechanism_clustering_result': self.current_session.mechanism_clustering_result
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
                          start_phase: Optional[str] = None,
                          continuous_mode: bool = False,
                          max_iterations: Optional[int] = None,
                          iteration_delay: float = 60.0) -> Dict[str, Any]:
        """
        Run the complete batch pipeline.

        Args:
            papers_per_condition: Number of papers to collect per condition
            resume: Whether to resume existing session
            start_phase: Specific phase to start from (collection, processing, categorization, canonical_grouping)
            continuous_mode: Enable infinite loop mode (restarts Phase 1 after Phase 3)
            max_iterations: Maximum iterations to run (None = unlimited, only applies in continuous mode)
            iteration_delay: Delay in seconds between iterations (default: 60s)

        Returns:
            Pipeline execution summary
        """
        try:
            # Load or create session
            if resume:
                session = self.load_existing_session()
                if not session:
                    logger.warning("No existing session found, creating new session")
                    session = self.create_new_session(
                        papers_per_condition,
                        continuous_mode=continuous_mode,
                        max_iterations=max_iterations,
                        iteration_delay=iteration_delay
                    )
                else:
                    logger.info(f"Resumed session: {session.session_id}")
                    # Update continuous mode settings if provided
                    if continuous_mode and not session.continuous_mode:
                        session.continuous_mode = True
                        session.max_iterations = max_iterations
                        session.iteration_delay_seconds = iteration_delay
                        logger.info("Enabled continuous mode for resumed session")
            else:
                session = self.create_new_session(
                    papers_per_condition,
                    continuous_mode=continuous_mode,
                    max_iterations=max_iterations,
                    iteration_delay=iteration_delay
                )

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
            logger.info(f"Continuous mode: {'ENABLED' if session.continuous_mode else 'DISABLED'}")
            if session.continuous_mode:
                logger.info(f"Max iterations: {session.max_iterations or 'unlimited'}")
                logger.info(f"Iteration delay: {session.iteration_delay_seconds}s")
            logger.info(f"Total conditions: 60 (12 specialties × 5 conditions)")

            # Execute pipeline phases (infinite loop if continuous_mode enabled)
            pipeline_start = time.time()

            # Iteration loop (runs once if continuous_mode=False, infinite if True)
            while True:
                # Check iteration limit
                if session.continuous_mode and session.max_iterations is not None:
                    if session.iteration_number > session.max_iterations:
                        logger.info(f"Reached max iterations ({session.max_iterations}), stopping")
                        break

                # Check shutdown request
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping after current phase")
                    break

                # Log iteration start
                if session.continuous_mode:
                    logger.info("\n" + "="*60)
                    logger.info(f"ITERATION {session.iteration_number} STARTING")
                    logger.info("="*60)

                iteration_start = time.time()

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
                    session.current_phase = BatchPhase.SEMANTIC_NORMALIZATION
                    self._save_session()

                # Phase 3: Semantic Normalization (formerly Canonical Grouping)
                if session.current_phase == BatchPhase.SEMANTIC_NORMALIZATION and not session.semantic_normalization_completed:
                    logger.info("\n" + "="*40)
                    logger.info("PHASE 3: SEMANTIC NORMALIZATION")
                    logger.info("="*40)

                    semantic_normalization_result = self._run_semantic_normalization_phase(session)
                    if not semantic_normalization_result['success']:
                        return semantic_normalization_result

                    session.semantic_normalization_completed = True
                    session.current_phase = BatchPhase.GROUP_CATEGORIZATION
                    self._save_session()

                # Phase 3.5: Group-Based Categorization (NEW - moved from Phase 2.5)
                if session.current_phase == BatchPhase.GROUP_CATEGORIZATION and not session.group_categorization_completed:
                    logger.info("\n" + "="*40)
                    logger.info("PHASE 3.5: GROUP-BASED CATEGORIZATION")
                    logger.info("="*40)

                    group_categorization_result = self._run_group_categorization_phase(session)
                    if not group_categorization_result['success']:
                        return group_categorization_result

                    session.group_categorization_completed = True
                    session.current_phase = BatchPhase.MECHANISM_CLUSTERING
                    self._save_session()

                # Phase 3.6: Mechanism Clustering (NEW)
                if session.current_phase == BatchPhase.MECHANISM_CLUSTERING and not session.mechanism_clustering_completed:
                    logger.info("\n" + "="*40)
                    logger.info("PHASE 3.6: MECHANISM CLUSTERING")
                    logger.info("="*40)

                    mechanism_clustering_result = self._run_mechanism_clustering_phase(session)
                    if not mechanism_clustering_result['success']:
                        return mechanism_clustering_result

                    session.mechanism_clustering_completed = True
                    session.current_phase = BatchPhase.COMPLETED
                    self._save_session()

                # Iteration completed
                iteration_time = time.time() - iteration_start

                logger.info("\n" + "="*60)
                logger.info(f"ITERATION {session.iteration_number} COMPLETED SUCCESSFULLY")
                logger.info("="*60)
                logger.info(f"Iteration time: {iteration_time:.1f} seconds ({iteration_time/60:.1f} minutes)")
                logger.info(f"Papers collected: {session.total_papers_collected}")
                logger.info(f"Papers processed: {session.total_papers_processed}")
                logger.info(f"Interventions extracted: {session.total_interventions_extracted}")
                logger.info(f"Canonical groups created: {session.total_canonical_groups_created}")
                logger.info(f"Groups categorized: {session.total_groups_categorized}")
                logger.info(f"Interventions categorized: {session.total_interventions_categorized}")
                logger.info(f"Orphans categorized: {session.total_orphans_categorized}")
                logger.info(f"Mechanisms processed: {session.total_mechanisms_processed}")
                logger.info(f"Mechanism clusters created: {session.total_mechanism_clusters}")

                # Save iteration history
                iteration_summary = {
                    'iteration_number': session.iteration_number,
                    'completion_time': datetime.now().isoformat(),
                    'iteration_duration_seconds': iteration_time,
                    'papers_collected': session.total_papers_collected,
                    'papers_processed': session.total_papers_processed,
                    'interventions_extracted': session.total_interventions_extracted,
                    'canonical_groups_created': session.total_canonical_groups_created,
                    'groups_categorized': session.total_groups_categorized,
                    'interventions_categorized': session.total_interventions_categorized,
                    'orphans_categorized': session.total_orphans_categorized,
                    'mechanisms_processed': session.total_mechanisms_processed,
                    'mechanism_clusters_created': session.total_mechanism_clusters
                }
                session.iteration_history.append(iteration_summary)

                # Check if we should continue or exit
                if not session.continuous_mode:
                    # Single iteration mode - exit after first completion
                    total_time = time.time() - pipeline_start
                    self._save_session()

                    return {
                        'success': True,
                        'session_id': session.session_id,
                        'iteration_completed': session.iteration_number,
                        'total_time_seconds': total_time,
                        'statistics': {
                            'papers_collected': session.total_papers_collected,
                            'papers_processed': session.total_papers_processed,
                            'interventions_extracted': session.total_interventions_extracted,
                            'canonical_groups_created': session.total_canonical_groups_created,
                            'groups_categorized': session.total_groups_categorized,
                            'interventions_categorized': session.total_interventions_categorized,
                            'orphans_categorized': session.total_orphans_categorized,
                            'mechanisms_processed': session.total_mechanisms_processed,
                            'mechanism_clusters_created': session.total_mechanism_clusters
                        }
                    }

                # Continuous mode - prepare for next iteration
                logger.info(f"\nPreparing for iteration {session.iteration_number + 1}...")

                # Reset for next iteration
                session.iteration_number += 1
                session.current_phase = BatchPhase.COLLECTION
                session.collection_completed = False
                session.processing_completed = False
                session.semantic_normalization_completed = False
                session.group_categorization_completed = False
                session.mechanism_clustering_completed = False

                # Reset iteration statistics (cumulative tracking happens in iteration_history)
                session.total_papers_collected = 0
                session.total_papers_processed = 0
                session.total_interventions_extracted = 0
                session.total_canonical_groups_created = 0
                session.total_groups_categorized = 0
                session.total_interventions_categorized = 0
                session.total_orphans_categorized = 0
                session.total_mechanisms_processed = 0
                session.total_mechanism_clusters = 0

                self._save_session()

                # Delay before next iteration (thermal protection)
                if session.iteration_delay_seconds > 0:
                    logger.info(f"Waiting {session.iteration_delay_seconds}s before next iteration (thermal protection)...")
                    time.sleep(session.iteration_delay_seconds)

                # Continue to next iteration
                continue

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

    def _run_semantic_normalization_phase(self, session: BatchSession) -> Dict[str, Any]:
        """
        Run Phase 3: Semantic normalization (cross-paper canonical grouping).

        Creates canonical groups that merge semantically equivalent interventions AND conditions.
        Example (interventions): "vitamin D" = "Vitamin D3" = "cholecalciferol" → group "vitamin D"
        Example (conditions): "IBS" → "IBS-C", "IBS-D", "IBS-M"
        """
        logger.info("Running Phase 3: Semantic normalization...")

        try:
            if self.shutdown_requested:
                return {'success': False, 'error': 'Shutdown requested during semantic normalization'}

            # Step 1: Run semantic grouping for interventions
            logger.info("  Step 1: Normalizing interventions...")
            grouping_result = self.dedup_integrator.group_all_data_semantically_batch()

            # Update session with intervention results
            total_interventions_processed = grouping_result.get('interventions_processed', 0)
            intervention_groups_created = grouping_result.get('canonical_entities_created', 0)

            # Step 2: Run semantic grouping for conditions
            logger.info("  Step 2: Normalizing condition entities...")
            from .rotation_semantic_normalizer import SemanticNormalizationOrchestrator
            condition_orchestrator = SemanticNormalizationOrchestrator(db_path=str(config.db_path))
            condition_result = condition_orchestrator.normalize_all_condition_entities(batch_size=50, force=True)

            total_conditions_processed = condition_result.get('processed', 0)
            condition_groups_created = condition_result.get('canonical_groups', 0)

            # Combine results
            total_canonical_groups = intervention_groups_created + condition_groups_created

            session.semantic_normalization_result = {
                'total_interventions_processed': total_interventions_processed,
                'intervention_groups_created': intervention_groups_created,
                'total_conditions_processed': total_conditions_processed,
                'condition_groups_created': condition_groups_created,
                'canonical_groups_created': total_canonical_groups,
                'interventions_grouped': grouping_result.get('total_merged', 0),
                'condition_relationships': condition_result.get('relationships', 0),
                'processing_time_seconds': grouping_result.get('processing_time_seconds', 0),
                'semantic_groups_found': grouping_result.get('duplicate_groups_found', 0),
                'method': grouping_result.get('method', 'llm_semantic_grouping')
            }

            session.total_canonical_groups_created = total_canonical_groups

            if not grouping_result['success']:
                logger.error(f"Intervention semantic normalization failed: {grouping_result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': f"Semantic normalization failed: {grouping_result.get('error', 'Unknown error')}",
                    'phase': 'semantic_normalization'
                }

            if condition_result.get('errors', 0) > 0 and condition_result.get('processed', 0) == 0:
                logger.error(f"Condition semantic normalization failed")
                return {
                    'success': False,
                    'error': f"Condition normalization failed: {condition_result.get('error', 'Unknown error')}",
                    'phase': 'semantic_normalization'
                }

            logger.info("[SUCCESS] Semantic normalization completed successfully")
            logger.info(f"  Interventions analyzed: {total_interventions_processed}")
            logger.info(f"  Intervention groups created: {intervention_groups_created}")
            logger.info(f"  Conditions analyzed: {total_conditions_processed}")
            logger.info(f"  Condition groups created: {condition_groups_created}")
            logger.info(f"  Total canonical groups: {total_canonical_groups}")

            return {'success': True, 'result': session.semantic_normalization_result}

        except Exception as e:
            logger.error(f"Semantic normalization failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'semantic_normalization'}

    def _run_group_categorization_phase(self, session: BatchSession) -> Dict[str, Any]:
        """
        Run Phase 3.5: Group-based categorization (NEW).

        Categorizes canonical groups (not individual interventions) using semantic context.
        Then propagates categories to interventions via UPDATE-JOIN (Option A).
        """
        logger.info("Running Phase 3.5: Group-based categorization...")

        try:
            if self.shutdown_requested:
                return {'success': False, 'error': 'Shutdown requested during group categorization'}

            # Lazy load group categorizer
            if not hasattr(self, 'group_categorizer') or self.group_categorizer is None:
                self.group_categorizer = RotationGroupCategorizer(
                    db_path=config.db_path,
                    batch_size=20,
                    include_members=True,
                    validate_results=True
                )

            # Run group categorization (both interventions AND conditions)
            group_cat_result = self.group_categorizer.run()

            # Update session with results (both intervention and condition stats)
            session.group_categorization_result = {
                # Intervention statistics
                'intervention_groups_categorized': group_cat_result['group_categorization']['processed'],
                'intervention_groups_failed': group_cat_result['group_categorization']['failed'],
                'interventions_updated': group_cat_result['propagation']['updated'],
                'intervention_orphans_found': group_cat_result['propagation']['orphans'],
                'intervention_orphans_categorized': group_cat_result['orphan_categorization']['processed'],
                # Condition statistics (new)
                'condition_groups_categorized': group_cat_result.get('condition_group_categorization', {}).get('processed_groups', 0),
                'condition_groups_failed': group_cat_result.get('condition_group_categorization', {}).get('failed_groups', 0),
                'conditions_updated': group_cat_result.get('condition_propagation', {}).get('updated', 0),
                'condition_orphans_found': group_cat_result.get('condition_propagation', {}).get('orphans', 0),
                'condition_orphans_categorized': group_cat_result.get('condition_orphan_categorization', {}).get('processed', 0),
                # Overall statistics
                'total_llm_calls': group_cat_result['performance']['total_llm_calls'],
                'elapsed_time_seconds': group_cat_result['elapsed_time_seconds'],
                'validation_passed': group_cat_result.get('validation', {}).get('all_passed', False),
                # Legacy fields for backward compatibility
                'groups_categorized': group_cat_result['group_categorization']['processed'],
                'groups_failed': group_cat_result['group_categorization']['failed'],
                'orphans_found': group_cat_result['propagation']['orphans'],
                'orphans_categorized': group_cat_result['orphan_categorization']['processed']
            }

            session.total_groups_categorized = (
                group_cat_result['group_categorization']['processed'] +
                group_cat_result.get('condition_group_categorization', {}).get('processed_groups', 0)
            )
            session.total_interventions_categorized = group_cat_result['propagation']['updated']
            session.total_orphans_categorized = (
                group_cat_result['orphan_categorization']['processed'] +
                group_cat_result.get('condition_orphan_categorization', {}).get('processed', 0)
            )

            logger.info("[SUCCESS] Group-based categorization completed successfully")
            logger.info(f"  Intervention groups categorized: {group_cat_result['group_categorization']['processed']}")
            logger.info(f"  Interventions updated: {session.total_interventions_categorized}")
            logger.info(f"  Intervention orphans categorized: {group_cat_result['orphan_categorization']['processed']}")
            logger.info(f"  Condition groups categorized: {group_cat_result.get('condition_group_categorization', {}).get('processed_groups', 0)}")
            logger.info(f"  Conditions updated: {session.group_categorization_result['conditions_updated']}")
            logger.info(f"  Condition orphans categorized: {group_cat_result.get('condition_orphan_categorization', {}).get('processed', 0)}")
            logger.info(f"  Total LLM calls: {group_cat_result['performance']['total_llm_calls']}")

            if group_cat_result.get('validation'):
                validation_passed = group_cat_result['validation']['all_passed']
                logger.info(f"  Validation: {'PASSED' if validation_passed else 'FAILED'}")

            return {'success': True, 'result': session.group_categorization_result}

        except Exception as e:
            logger.error(f"Group categorization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e), 'phase': 'group_categorization'}

    def _run_mechanism_clustering_phase(self, session: BatchSession) -> Dict[str, Any]:
        """
        Run Phase 3.6: Mechanism clustering (NEW).

        Clusters intervention mechanisms using HDBSCAN + semantic embeddings.
        Ensures 100% assignment - no mechanism left uncategorized.
        """
        logger.info("Running Phase 3.6: Mechanism clustering...")

        try:
            if self.shutdown_requested:
                return {'success': False, 'error': 'Shutdown requested during mechanism clustering'}

            # Lazy load mechanism clusterer
            if not self.mechanism_clusterer:
                self.mechanism_clusterer = RotationMechanismClusterer(
                    db_path=str(config.db_path),
                    cache_dir=str(config.data_root / "semantic_normalization_cache")
                )

            # Run mechanism clustering (100% assignment guaranteed)
            clustering_result = self.mechanism_clusterer.run(force=False)

            # Update session with results
            session.mechanism_clustering_result = {
                'mechanisms_processed': clustering_result['mechanisms_processed'],
                'clusters_created': clustering_result['clusters_created'],
                'natural_clusters': clustering_result.get('natural_clusters', 0),
                'singleton_clusters': clustering_result.get('singleton_clusters', 0),
                'assignment_rate': clustering_result['assignment_rate'],
                'avg_cluster_size': clustering_result.get('avg_cluster_size', 0.0),
                'silhouette_score': clustering_result.get('silhouette_score', 0.0),
                'elapsed_time_seconds': clustering_result['elapsed_time_seconds']
            }

            session.total_mechanisms_processed = clustering_result['mechanisms_processed']
            session.total_mechanism_clusters = clustering_result['clusters_created']

            if not clustering_result['success']:
                logger.error(f"Mechanism clustering failed: {clustering_result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': f"Mechanism clustering failed: {clustering_result.get('error', 'Unknown error')}",
                    'phase': 'mechanism_clustering'
                }

            logger.info("[SUCCESS] Mechanism clustering completed successfully")
            logger.info(f"  Mechanisms processed: {session.total_mechanisms_processed}")
            logger.info(f"  Total clusters created: {session.total_mechanism_clusters}")
            logger.info(f"  Natural clusters: {clustering_result.get('natural_clusters', 0)}")
            logger.info(f"  Singleton clusters: {clustering_result.get('singleton_clusters', 0)}")
            logger.info(f"  Assignment rate: {clustering_result['assignment_rate']:.1%}")
            logger.info(f"  Average cluster size: {clustering_result.get('avg_cluster_size', 0.0):.2f}")

            return {'success': True, 'result': session.mechanism_clustering_result}

        except Exception as e:
            logger.error(f"Mechanism clustering failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e), 'phase': 'mechanism_clustering'}

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
            'continuous_mode': {
                'enabled': session.continuous_mode,
                'max_iterations': session.max_iterations,
                'iteration_delay_seconds': session.iteration_delay_seconds,
                'completed_iterations': len(session.iteration_history)
            },
            'progress': {
                'collection_completed': session.collection_completed,
                'processing_completed': session.processing_completed,
                'semantic_normalization_completed': session.semantic_normalization_completed,
                'group_categorization_completed': session.group_categorization_completed,
                'mechanism_clustering_completed': session.mechanism_clustering_completed,
                'pipeline_completed': session.is_completed()
            },
            'statistics': {
                'papers_collected': session.total_papers_collected,
                'papers_processed': session.total_papers_processed,
                'interventions_extracted': session.total_interventions_extracted,
                'canonical_groups_created': session.total_canonical_groups_created,
                'groups_categorized': session.total_groups_categorized,
                'interventions_categorized': session.total_interventions_categorized,
                'orphans_categorized': session.total_orphans_categorized,
                'mechanisms_processed': session.total_mechanisms_processed,
                'mechanism_clusters_created': session.total_mechanism_clusters
            },
            'iteration_history': session.iteration_history,
            'phase_results': {
                'collection': session.collection_result,
                'processing': session.processing_result,
                'semantic_normalization': session.semantic_normalization_result,
                'group_categorization': session.group_categorization_result,
                'mechanism_clustering': session.mechanism_clustering_result
            }
        }


def main():
    """Command line interface for batch medical rotation pipeline."""
    parser = argparse.ArgumentParser(
        description="Batch Medical Rotation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single iteration
  python batch_medical_rotation.py --papers-per-condition 10

  # Run continuous mode (infinite loop until Ctrl+C)
  python batch_medical_rotation.py --papers-per-condition 10 --continuous

  # Run limited iterations (e.g., 5 complete cycles)
  python batch_medical_rotation.py --papers-per-condition 10 --continuous --max-iterations 5

  # Custom delay between iterations (5 minutes = 300 seconds)
  python batch_medical_rotation.py --papers-per-condition 10 --continuous --iteration-delay 300

  # Resume existing session
  python batch_medical_rotation.py --resume

  # Resume in continuous mode
  python batch_medical_rotation.py --resume --continuous

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
    parser.add_argument('--continuous', action='store_true',
                        help='Enable continuous mode (infinite loop, restarts Phase 1 after Phase 3)')
    parser.add_argument('--max-iterations', type=int,
                        help='Maximum iterations to run in continuous mode (default: unlimited)')
    parser.add_argument('--iteration-delay', type=float, default=60.0,
                        help='Delay in seconds between iterations for thermal protection (default: 60)')
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
            start_phase=args.start_phase,
            continuous_mode=args.continuous,
            max_iterations=args.max_iterations,
            iteration_delay=args.iteration_delay
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