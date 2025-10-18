#!/usr/bin/env python3
"""
Batch Medical Rotation Pipeline - Refactored Orchestrator

Simplified batch-oriented pipeline that processes all 60 medical conditions
in five distinct phases: collection → processing → semantic normalization →
data mining → frontend export.

This is a lean orchestrator that delegates to specialized modules:
- batch_config.py: Configuration and command-line arguments
- batch_session.py: Session state management
- phase_runner.py: Phase execution logic

Pipeline Flow:
1. COLLECTION: Collect N papers for all 60 conditions (Phase 1)
2. PROCESSING: Process all papers with LLM (qwen3:14b) (Phase 2)
3. SEMANTIC NORMALIZATION: Unified Phase 3 - processes ALL entity types (Phase 3)
   - 3a: Embedding (interventions, conditions, mechanisms)
   - 3b: Clustering (interventions, conditions, mechanisms)
   - 3c: LLM Naming + Category Assignment (interventions, conditions, mechanisms)
   - 3d: Hierarchical Merging (optional)
4. DATA MINING: Knowledge graph + Bayesian scoring (Phase 4: 4a + 4b)
5. FRONTEND EXPORT: Export data to frontend JSON files (Phase 5)

Features:
- Modular architecture with clear separation of concerns
- 5 clear phases with natural breakpoints for recovery
- Continuous mode: Infinite loop with iteration tracking
- Thermal protection: Configurable delay between iterations
- Session persistence with platform-specific file locking
- Comprehensive progress tracking and statistics

Usage:
    # Run single iteration
    python batch_medical_rotation.py --papers-per-condition 10

    # Run continuous mode (infinite loop until Ctrl+C)
    python batch_medical_rotation.py --papers-per-condition 10 --continuous

    # Run limited iterations (e.g., 5 complete cycles)
    python batch_medical_rotation.py --papers-per-condition 10 --continuous --max-iterations 5

    # Resume from specific phase
    python batch_medical_rotation.py --resume --start-phase processing

    # Status check
    python batch_medical_rotation.py --status
"""

import sys
import time
import signal
import json
import traceback
from typing import Dict, Any, Optional

from back_end.src.data.config import setup_logging
from .batch_config import parse_command_line_args, BatchPhase, BatchConfig
from .batch_session import SessionManager, BatchSession
from .phase_runner import PhaseRunner

logger = setup_logging(__name__, 'batch_medical_rotation.log')


class BatchMedicalRotationPipeline:
    """
    Simplified batch medical rotation pipeline using modular architecture.

    Delegates to:
    - SessionManager for session state management
    - PhaseRunner for phase execution
    - BatchConfig for configuration
    """

    def __init__(self):
        """Initialize the batch pipeline."""
        self.session_manager = SessionManager()
        self.phase_runner = PhaseRunner()
        self.shutdown_requested = False
        self.current_session: Optional[BatchSession] = None

        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Batch medical rotation pipeline initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def run(self, config: BatchConfig) -> Dict[str, Any]:
        """
        Run the complete batch pipeline.

        Args:
            config: Pipeline configuration

        Returns:
            Pipeline execution summary
        """
        try:
            # Load or create session
            if config.resume:
                session = self.session_manager.load_existing_session()
                if not session:
                    logger.warning("No existing session found, creating new session")
                    session = self.session_manager.create_new_session(
                        config.papers_per_condition,
                        continuous_mode=config.continuous_mode,
                        max_iterations=config.max_iterations,
                        iteration_delay=config.iteration_delay_seconds
                    )
                else:
                    logger.info(f"Resumed session: {session.session_id}")
                    # Update continuous mode settings if provided
                    if config.continuous_mode and not session.continuous_mode:
                        session.continuous_mode = True
                        session.max_iterations = config.max_iterations
                        session.iteration_delay_seconds = config.iteration_delay_seconds
                        logger.info("Enabled continuous mode for resumed session")
            else:
                session = self.session_manager.create_new_session(
                    config.papers_per_condition,
                    continuous_mode=config.continuous_mode,
                    max_iterations=config.max_iterations,
                    iteration_delay=config.iteration_delay_seconds
                )

            self.current_session = session

            # Override start phase if specified
            if config.start_phase:
                try:
                    session.current_phase = BatchPhase(config.start_phase)
                    logger.info(f"Starting from specified phase: {config.start_phase}")
                except ValueError:
                    logger.error(f"Invalid phase: {config.start_phase}")
                    return {'success': False, 'error': f'Invalid phase: {config.start_phase}'}

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

                # Execute all pipeline phases
                result = self._execute_pipeline_phases(session)

                if not result['success']:
                    return result

                # Iteration completed successfully
                iteration_time = time.time() - iteration_start
                self._log_iteration_completion(session, iteration_time)

                # Save iteration history
                session.save_iteration_summary(iteration_time)

                # Check if we should continue or exit
                if not session.continuous_mode:
                    total_time = time.time() - pipeline_start
                    self.session_manager.save_session(session)

                    return {
                        'success': True,
                        'session_id': session.session_id,
                        'iteration_completed': session.iteration_number,
                        'total_time_seconds': total_time,
                        'statistics': self._get_session_statistics(session)
                    }

                # Continuous mode - prepare for next iteration
                logger.info(f"\nPreparing for iteration {session.iteration_number + 1}...")
                session.reset_for_next_iteration()
                self.session_manager.save_session(session)

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

    def _execute_pipeline_phases(self, session: BatchSession) -> Dict[str, Any]:
        """Execute all pipeline phases based on current state."""
        phase_map = {
            BatchPhase.COLLECTION: (
                'collection_completed',
                "PHASE 1: BATCH COLLECTION",
                self.phase_runner.run_collection_phase,
                BatchPhase.PROCESSING
            ),
            BatchPhase.PROCESSING: (
                'processing_completed',
                "PHASE 2: BATCH PROCESSING",
                self.phase_runner.run_processing_phase,
                BatchPhase.SEMANTIC_NORMALIZATION
            ),
            BatchPhase.SEMANTIC_NORMALIZATION: (
                'semantic_normalization_completed',
                "PHASE 3: SEMANTIC NORMALIZATION (3a/3b/3c for all entities)",
                self.phase_runner.run_semantic_normalization_phase,
                BatchPhase.DATA_MINING
            ),
            BatchPhase.DATA_MINING: (
                'data_mining_completed',
                "PHASE 4: DATA MINING",
                self.phase_runner.run_data_mining_phase,
                BatchPhase.FRONTEND_EXPORT
            ),
            BatchPhase.FRONTEND_EXPORT: (
                'frontend_export_completed',
                "PHASE 5: FRONTEND DATA EXPORT",
                self.phase_runner.run_frontend_export_phase,
                BatchPhase.COMPLETED
            )
        }

        for phase, (completion_flag, phase_name, phase_func, next_phase) in phase_map.items():
            if session.current_phase == phase and not getattr(session, completion_flag):
                if self.shutdown_requested:
                    return {'success': False, 'error': f'Shutdown requested during {phase_name}'}

                logger.info("\n" + "="*40)
                logger.info(phase_name)
                logger.info("="*40)

                phase_result = phase_func(session)
                if not phase_result['success']:
                    return phase_result

                setattr(session, completion_flag, True)
                session.current_phase = next_phase
                self.session_manager.save_session(session)

        return {'success': True}

    def _log_iteration_completion(self, session: BatchSession, iteration_time: float):
        """Log iteration completion statistics."""
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
        logger.info(f"Mechanisms processed: {session.total_mechanisms_processed}")
        logger.info(f"Mechanism clusters created: {session.total_mechanism_clusters}")
        logger.info(f"Knowledge graph nodes: {session.total_knowledge_graph_nodes}")
        logger.info(f"Knowledge graph edges: {session.total_knowledge_graph_edges}")
        logger.info(f"Bayesian scores generated: {session.total_bayesian_scores}")
        logger.info(f"Files exported: {session.total_files_exported}")

    def _get_session_statistics(self, session: BatchSession) -> Dict[str, Any]:
        """Get session statistics dictionary."""
        return {
            'papers_collected': session.total_papers_collected,
            'papers_processed': session.total_papers_processed,
            'interventions_extracted': session.total_interventions_extracted,
            'canonical_groups_created': session.total_canonical_groups_created,
            'groups_categorized': session.total_groups_categorized,
            'interventions_categorized': session.total_interventions_categorized,
            'orphans_categorized': session.total_orphans_categorized,
            'mechanisms_processed': session.total_mechanisms_processed,
            'mechanism_clusters_created': session.total_mechanism_clusters,
            'knowledge_graph_nodes': session.total_knowledge_graph_nodes,
            'knowledge_graph_edges': session.total_knowledge_graph_edges,
            'bayesian_scores_generated': session.total_bayesian_scores,
            'files_exported': session.total_files_exported
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        session = self.session_manager.load_existing_session()

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
                'data_mining_completed': session.data_mining_completed,
                'frontend_export_completed': session.frontend_export_completed,
                'pipeline_completed': session.is_completed()
            },
            'statistics': self._get_session_statistics(session),
            'iteration_history': session.iteration_history,
            'phase_results': {
                'collection': session.collection_result,
                'processing': session.processing_result,
                'semantic_normalization': session.semantic_normalization_result,
                'group_categorization': session.group_categorization_result,
                'mechanism_clustering': session.mechanism_clustering_result,
                'data_mining': session.data_mining_result,
                'frontend_export': session.frontend_export_result
            }
        }


def main():
    """Command line interface for batch medical rotation pipeline."""
    try:
        # Parse command-line arguments
        config = parse_command_line_args()

        # Initialize pipeline
        pipeline = BatchMedicalRotationPipeline()

        # Handle status check
        if config.status_only:
            status = pipeline.get_status()
            print(json.dumps(status, indent=2))
            return

        # Run pipeline
        result = pipeline.run(config)

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
        print("\n[INTERRUPTED] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[FAILED] Pipeline failed: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
