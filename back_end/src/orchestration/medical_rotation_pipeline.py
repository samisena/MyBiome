#!/usr/bin/env python3
"""
Medical Rotation Pipeline - Main Orchestrator

Complete orchestrator for the rotating medical condition pipeline.
Coordinates collection, processing, and deduplication across all 60 medical conditions
with robust error handling, session persistence, and resumable operation.

Features:
- Complete workflow orchestration (collect → process → deduplicate)
- Circular rotation through 60 medical conditions (12 specialties × 5 conditions)
- Resumable execution after any interruption
- Comprehensive error handling and auto-recovery
- Session state persistence and progress tracking
- Configurable papers per condition
- Overnight operation capability

Usage:
    # Start new rotation pipeline (default: 10 papers per condition)
    python medical_rotation_pipeline.py

    # Custom papers per condition
    python medical_rotation_pipeline.py --papers-per-condition 5

    # Resume interrupted session
    python medical_rotation_pipeline.py --resume

    # Show current status
    python medical_rotation_pipeline.py --status

    # Run single condition for testing
    python medical_rotation_pipeline.py --test-condition "diabetes mellitus" --papers 3

Examples:
    # Standard overnight operation
    python medical_rotation_pipeline.py --papers-per-condition 10

    # Resume with status monitoring
    python medical_rotation_pipeline.py --resume --verbose

    # Test with specific condition
    python medical_rotation_pipeline.py --test-condition "hypertension" --papers 5
"""

import sys
import time
import signal
import argparse
import traceback
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass

try:
    from ..data.config import config, setup_logging
    from .rotation_session_manager import (
        RotationSessionManager, PipelinePhase, session_manager
    )
    from .rotation_collection_integrator import RotationCollectionIntegrator
    from .rotation_llm_processor import RotationLLMProcessor
    from .rotation_deduplication_integrator import RotationDeduplicationIntegrator
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.orchestration.rotation_session_manager import (
        RotationSessionManager, PipelinePhase, session_manager
    )
    from back_end.src.orchestration.rotation_collection_integrator import RotationCollectionIntegrator
    from back_end.src.orchestration.rotation_llm_processor import RotationLLMProcessor
    from back_end.src.orchestration.rotation_deduplication_integrator import RotationDeduplicationIntegrator

logger = setup_logging(__name__, 'medical_rotation_pipeline.log')


class ErrorSeverity(Enum):
    """Error severity levels for categorized handling."""
    LOW = "low"          # Warnings, partial failures
    MEDIUM = "medium"    # Recoverable errors, retryable failures
    HIGH = "high"        # Critical errors, require intervention
    CRITICAL = "critical" # System failures, complete pipeline stop


class ErrorCategory(Enum):
    """Categories of errors for specialized handling."""
    NETWORK = "network"          # API failures, timeouts
    DATABASE = "database"        # DB connection, query failures
    PROCESSING = "processing"    # LLM processing, extraction errors
    VALIDATION = "validation"    # Data validation, quality issues
    RESOURCE = "resource"        # Memory, disk, GPU issues
    CONFIGURATION = "configuration" # Config, setup errors
    EXTERNAL = "external"        # Third-party service failures
    UNKNOWN = "unknown"          # Uncategorized errors


@dataclass
class ErrorRecord:
    """Detailed error record for tracking and analysis."""
    timestamp: datetime
    error_type: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    condition: Optional[str]
    phase: Optional[str]
    retry_count: int
    recoverable: bool
    recovery_action: Optional[str]
    traceback: Optional[str]


class CircuitBreaker:
    """Circuit breaker pattern for resilient error handling."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == 'open':
            if datetime.now().timestamp() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
                logger.info("Circuit breaker moving to half-open state")
            else:
                raise Exception("Circuit breaker is open - service unavailable")

        try:
            result = func(*args, **kwargs)
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                logger.info("Circuit breaker closed - service recovered")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now().timestamp()

            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            raise e


class MedicalRotationPipeline:
    """
    Main orchestrator for the medical rotation pipeline.
    Coordinates all phases of the rotation workflow with comprehensive error handling.
    """

    def __init__(self, session_mgr: RotationSessionManager = None):
        """Initialize the medical rotation pipeline."""
        self.session_mgr = session_mgr or session_manager

        # Initialize component integrators
        self.collection_integrator = RotationCollectionIntegrator(self.session_mgr)
        self.llm_processor = RotationLLMProcessor()
        self.dedup_integrator = RotationDeduplicationIntegrator()

        # Control flags
        self.shutdown_requested = False
        self.pause_requested = False

        # Performance tracking
        self.start_time = datetime.now()
        self.conditions_completed = 0
        self.total_papers_collected = 0
        self.total_papers_processed = 0
        self.total_interventions_extracted = 0
        self.total_duplicates_removed = 0

        # Error handling and recovery
        self.error_history: List[ErrorRecord] = []
        self.max_consecutive_failures = 3
        self.max_condition_retries = 2
        self.global_failure_threshold = 10
        self.circuit_breakers = {
            'collection': CircuitBreaker(failure_threshold=3, recovery_timeout=300),
            'processing': CircuitBreaker(failure_threshold=3, recovery_timeout=600),
            'deduplication': CircuitBreaker(failure_threshold=5, recovery_timeout=180)
        }

        # Auto-recovery settings
        self.enable_auto_recovery = True
        self.recovery_strategies = {
            ErrorCategory.NETWORK: self._recover_network_error,
            ErrorCategory.DATABASE: self._recover_database_error,
            ErrorCategory.PROCESSING: self._recover_processing_error,
            ErrorCategory.RESOURCE: self._recover_resource_error
        }

        # Condition failure tracking
        self.consecutive_failures = 0
        self.condition_retry_counts = {}

        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Medical rotation pipeline initialized with enhanced error handling")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def run_rotation_pipeline(self, papers_per_condition: int = 10,
                            test_condition: Optional[str] = None,
                            resume: bool = False) -> Dict[str, Any]:
        """
        Run the complete medical rotation pipeline.

        Args:
            papers_per_condition: Number of papers to collect per condition
            test_condition: Single condition to test (overrides rotation)
            resume: Whether to resume existing session

        Returns:
            Pipeline execution summary
        """
        try:
            # Load or create session
            if resume:
                session = self.session_mgr.load_existing_session()
                if not session:
                    logger.warning("No existing session found, creating new session")
                    session = self.session_mgr.create_new_session(papers_per_condition)
                else:
                    logger.info(f"Resumed session: {session.session_id}")
            else:
                session = self.session_mgr.create_new_session(papers_per_condition)

            logger.info("Starting medical rotation pipeline")
            logger.info(f"Session: {session.session_id}")
            logger.info(f"Papers per condition: {session.papers_per_condition}")
            logger.info(f"Total conditions: {session.total_conditions}")

            # Handle test mode
            if test_condition:
                return self._run_test_condition(test_condition, papers_per_condition)

            # Check for interrupted state and resume if needed
            if session.interruption_state:
                logger.info(f"Resuming interrupted pipeline from {session.interruption_state.phase} phase")
                resume_result = self._resume_interrupted_pipeline()
                if resume_result:
                    logger.info("Successfully resumed interrupted pipeline")

            # Main rotation loop
            while session.is_active and not self.shutdown_requested:
                try:
                    # Process current condition
                    current_result = self._process_current_condition()

                    if current_result['success']:
                        # Update statistics
                        self._update_pipeline_statistics(current_result)

                        # Advance to next condition
                        old_condition = session.current_condition
                        new_specialty, new_condition = self.session_mgr.advance_to_next_condition()

                        logger.info(f"Advanced from '{old_condition}' to '{new_specialty} -> {new_condition}'")
                        self.conditions_completed += 1

                        # Check if we completed a full rotation
                        if session.iteration_count > 1 and session.current_specialty_index == 0 and session.current_condition_index == 0:
                            logger.info(f"Completed rotation iteration {session.iteration_count - 1}")

                    else:
                        # Handle condition failure
                        logger.error(f"Condition '{session.current_condition}' failed: {current_result.get('error', 'Unknown error')}")

                        # Advance anyway to avoid getting stuck
                        self.session_mgr.advance_to_next_condition()

                    # Brief pause between conditions
                    if not self.shutdown_requested:
                        time.sleep(2)

                except Exception as e:
                    # Comprehensive error handling
                    error_record = self._record_error(
                        error=e,
                        condition=session.current_condition,
                        phase="pipeline",
                        severity=ErrorSeverity.HIGH
                    )

                    # Attempt error recovery
                    recovery_successful = self._attempt_error_recovery(error_record)

                    if recovery_successful:
                        logger.info(f"Successfully recovered from error for '{session.current_condition}'")
                        self.consecutive_failures = 0
                        continue

                    # Recovery failed - handle based on failure pattern
                    self.consecutive_failures += 1
                    condition_retries = self.condition_retry_counts.get(session.current_condition, 0)

                    if condition_retries < self.max_condition_retries:
                        # Retry current condition
                        self.condition_retry_counts[session.current_condition] = condition_retries + 1
                        logger.warning(f"Retrying condition '{session.current_condition}' (attempt {condition_retries + 2})")
                        time.sleep(30 * (condition_retries + 1))  # Exponential backoff
                        continue

                    # Max retries exceeded - mark as failed and advance
                    logger.error(f"Condition '{session.current_condition}' failed after {self.max_condition_retries + 1} attempts")
                    self.session_mgr.mark_condition_failed(str(e))

                    # Check for global failure threshold
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.critical(f"Too many consecutive failures ({self.consecutive_failures}). Initiating emergency shutdown.")
                        self._emergency_shutdown("Consecutive failure threshold exceeded")
                        break

                    # Advance to next condition
                    self.session_mgr.advance_to_next_condition()

                    # Escalating pause based on failure count
                    pause_time = min(60 * (2 ** self.consecutive_failures), 300)  # Max 5 minutes
                    logger.info(f"Pausing {pause_time} seconds before continuing")
                    time.sleep(pause_time)

            # Generate final summary
            return self._generate_pipeline_summary()

        except Exception as e:
            logger.error(f"Critical pipeline error: {e}")
            logger.error(traceback.format_exc())

            return {
                'success': False,
                'error': str(e),
                'conditions_completed': self.conditions_completed,
                'total_time': (datetime.now() - self.start_time).total_seconds()
            }

    def _process_current_condition(self) -> Dict[str, Any]:
        """Process the current condition through all pipeline phases."""
        session = self.session_mgr.session
        condition = session.current_condition
        specialty = session.current_specialty

        logger.info(f"Processing condition: {specialty} -> {condition}")

        # Phase 1: Collection
        logger.info(f"[1/3] Collection phase for '{condition}'")
        collection_result = self.collection_integrator.collect_current_condition()

        if not collection_result['success']:
            return {
                'success': False,
                'condition': condition,
                'specialty': specialty,
                'failed_phase': 'collection',
                'error': collection_result.get('error', 'Collection failed'),
                'collection_result': collection_result
            }

        logger.info(f"Collection completed: {collection_result['papers_collected']} papers")

        # Phase 2: LLM Processing
        logger.info(f"[2/3] Processing phase for '{condition}'")
        processing_result = self.llm_processor.process_condition_papers(
            condition=condition,
            max_papers=None  # Process all collected papers
        )

        if not processing_result['success']:
            return {
                'success': False,
                'condition': condition,
                'specialty': specialty,
                'failed_phase': 'processing',
                'error': processing_result.get('error', 'Processing failed'),
                'collection_result': collection_result,
                'processing_result': processing_result
            }

        logger.info(f"Processing completed: {processing_result['papers_processed']} papers, "
                   f"{processing_result['interventions_extracted']} interventions")

        # Phase 3: Deduplication
        logger.info(f"[3/3] Deduplication phase for '{condition}'")
        dedup_result = self.dedup_integrator.deduplicate_condition_data(condition)

        if not dedup_result['success']:
            logger.warning(f"Deduplication failed for '{condition}': {dedup_result.get('error', 'Unknown error')}")
            # Don't fail the entire condition for deduplication errors
            dedup_result = {
                'success': True,
                'entities_merged': 0,
                'warning': 'Deduplication skipped due to error'
            }

        logger.info(f"Deduplication completed: {dedup_result.get('entities_merged', 0)} entities merged")

        # All phases completed successfully
        return {
            'success': True,
            'condition': condition,
            'specialty': specialty,
            'collection_result': collection_result,
            'processing_result': processing_result,
            'deduplication_result': dedup_result,
            'papers_collected': collection_result['papers_collected'],
            'papers_processed': processing_result['papers_processed'],
            'interventions_extracted': processing_result['interventions_extracted'],
            'entities_merged': dedup_result.get('entities_merged', 0)
        }

    def _resume_interrupted_pipeline(self) -> bool:
        """Resume pipeline from interrupted state."""
        session = self.session_mgr.session
        if not session or not session.interruption_state:
            return False

        interruption = session.interruption_state
        phase = interruption.phase
        condition = session.current_condition

        logger.info(f"Resuming from {phase} phase for condition '{condition}'")

        try:
            if phase == PipelinePhase.COLLECTION.value:
                # Resume collection
                result = self.collection_integrator.resume_interrupted_collection()
                return result is not None and result.get('success', False)

            elif phase == PipelinePhase.LLM_PROCESSING.value:
                # Resume processing - LLM processor handles this internally
                result = self.llm_processor.process_condition_papers(condition)
                return result.get('success', False)

            elif phase == PipelinePhase.DEDUPLICATION.value:
                # Re-run deduplication (it's idempotent)
                result = self.dedup_integrator.deduplicate_condition_data(condition)
                return result.get('success', False)

            else:
                logger.warning(f"Unknown interruption phase: {phase}")
                return False

        except Exception as e:
            logger.error(f"Error resuming from {phase} phase: {e}")
            return False

    def _run_test_condition(self, condition: str, papers_count: int) -> Dict[str, Any]:
        """Run pipeline for a single test condition."""
        logger.info(f"Running test pipeline for condition: '{condition}'")

        # Create temporary session for testing
        test_session = self.session_mgr.create_new_session(papers_count)

        # Override current condition for testing
        test_session.current_condition_progress.condition = condition
        test_session.current_condition_progress.specialty = "test"

        # Process the test condition
        result = self._process_current_condition()

        logger.info(f"Test completed for '{condition}'")
        return {
            'test_mode': True,
            'test_condition': condition,
            'test_papers': papers_count,
            **result
        }

    def _update_pipeline_statistics(self, result: Dict[str, Any]):
        """Update pipeline-wide statistics."""
        self.total_papers_collected += result.get('papers_collected', 0)
        self.total_papers_processed += result.get('papers_processed', 0)
        self.total_interventions_extracted += result.get('interventions_extracted', 0)
        self.total_duplicates_removed += result.get('entities_merged', 0)

    def _generate_pipeline_summary(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution summary."""
        session = self.session_mgr.session
        total_time = (datetime.now() - self.start_time).total_seconds()

        summary = {
            'success': True,
            'session_id': session.session_id if session else 'unknown',
            'execution_time_seconds': total_time,
            'execution_time_formatted': str(timedelta(seconds=int(total_time))),
            'conditions_completed': self.conditions_completed,
            'total_papers_collected': self.total_papers_collected,
            'total_papers_processed': self.total_papers_processed,
            'total_interventions_extracted': self.total_interventions_extracted,
            'total_duplicates_removed': self.total_duplicates_removed,
            'average_time_per_condition': total_time / max(self.conditions_completed, 1),
            'papers_per_hour': (self.total_papers_processed / (total_time / 3600)) if total_time > 0 else 0,
            'interventions_per_hour': (self.total_interventions_extracted / (total_time / 3600)) if total_time > 0 else 0,
            'shutdown_requested': self.shutdown_requested
        }

        if session:
            summary.update({
                'current_iteration': session.iteration_count,
                'current_specialty': session.current_specialty,
                'current_condition': session.current_condition,
                'session_progress_percent': session.progress_percent,
                'completed_conditions_in_session': len(session.completed_conditions),
                'failed_conditions_in_session': len(session.failed_conditions)
            })

        return summary

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        session = self.session_mgr.session
        if not session:
            return {'error': 'No active session'}

        # Get component statuses
        collection_status = self.collection_integrator.get_collection_status()

        # Calculate runtime statistics
        runtime = (datetime.now() - self.start_time).total_seconds()

        status = {
            'session_id': session.session_id,
            'is_active': session.is_active,
            'current_iteration': session.iteration_count,
            'current_specialty': session.current_specialty,
            'current_condition': session.current_condition,
            'papers_per_condition': session.papers_per_condition,
            'progress_percent': session.progress_percent,
            'completed_conditions': len(session.completed_conditions),
            'failed_conditions': len(session.failed_conditions),
            'total_conditions': session.total_conditions,
            'pipeline_runtime_seconds': runtime,
            'pipeline_runtime_formatted': str(timedelta(seconds=int(runtime))),
            'conditions_completed_this_run': self.conditions_completed,
            'collection_status': collection_status,
            'has_interruption': session.interruption_state is not None,
            'interruption_phase': session.interruption_state.phase if session.interruption_state else None
        }

        return status

    def pause_pipeline(self):
        """Pause the pipeline execution."""
        self.pause_requested = True
        logger.info("Pipeline pause requested")

    def resume_pipeline(self):
        """Resume paused pipeline execution."""
        self.pause_requested = False
        logger.info("Pipeline resumed")

    def stop_pipeline(self):
        """Stop the pipeline execution gracefully."""
        self.shutdown_requested = True
        logger.info("Pipeline stop requested")

    def _record_error(self, error: Exception, condition: Optional[str] = None,
                      phase: Optional[str] = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                      retry_count: int = 0) -> ErrorRecord:
        """Record detailed error information for analysis and recovery."""
        category = self._categorize_error(error)
        recoverable = self._is_recoverable_error(error, category)

        error_record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            severity=severity,
            category=category,
            message=str(error),
            condition=condition,
            phase=phase,
            retry_count=retry_count,
            recoverable=recoverable,
            recovery_action=None,
            traceback=traceback.format_exc()
        )

        self.error_history.append(error_record)

        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR [{category.value}]: {error_record.message}")
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH ERROR [{category.value}]: {error_record.message}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM ERROR [{category.value}]: {error_record.message}")
        else:
            logger.info(f"LOW ERROR [{category.value}]: {error_record.message}")

        return error_record

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error for appropriate handling strategy."""
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()

        # Network-related errors
        if any(keyword in error_msg for keyword in ['timeout', 'connection', 'network', 'http', 'ssl']):
            return ErrorCategory.NETWORK
        if any(keyword in error_type for keyword in ['timeout', 'connection', 'http']):
            return ErrorCategory.NETWORK

        # Database errors
        if any(keyword in error_msg for keyword in ['database', 'sqlite', 'sql', 'lock']):
            return ErrorCategory.DATABASE
        if any(keyword in error_type for keyword in ['database', 'sqlite', 'operational']):
            return ErrorCategory.DATABASE

        # Processing errors
        if any(keyword in error_msg for keyword in ['llm', 'model', 'generation', 'parsing', 'json']):
            return ErrorCategory.PROCESSING
        if any(keyword in error_type for keyword in ['json', 'parse', 'decode']):
            return ErrorCategory.PROCESSING

        # Resource errors
        if any(keyword in error_msg for keyword in ['memory', 'disk', 'space', 'gpu', 'cuda']):
            return ErrorCategory.RESOURCE
        if any(keyword in error_type for keyword in ['memory', 'runtime']):
            return ErrorCategory.RESOURCE

        # Validation errors
        if any(keyword in error_msg for keyword in ['validation', 'invalid', 'format']):
            return ErrorCategory.VALIDATION

        # Configuration errors
        if any(keyword in error_msg for keyword in ['config', 'setting', 'path', 'import']):
            return ErrorCategory.CONFIGURATION
        if any(keyword in error_type for keyword in ['import', 'attribute', 'module']):
            return ErrorCategory.CONFIGURATION

        return ErrorCategory.UNKNOWN

    def _is_recoverable_error(self, error: Exception, category: ErrorCategory) -> bool:
        """Determine if an error is recoverable through automated means."""
        # Definitely recoverable
        recoverable_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.PROCESSING,
            ErrorCategory.EXTERNAL
        }

        # Sometimes recoverable
        if category in recoverable_categories:
            return True

        if category == ErrorCategory.DATABASE:
            # Database locks are recoverable, corruption is not
            return 'lock' in str(error).lower()

        if category == ErrorCategory.RESOURCE:
            # Memory issues might be recoverable with cleanup
            return 'memory' in str(error).lower()

        # Configuration and validation errors usually need manual intervention
        return False

    def _attempt_error_recovery(self, error_record: ErrorRecord) -> bool:
        """Attempt to recover from error using appropriate strategy."""
        if not self.enable_auto_recovery or not error_record.recoverable:
            return False

        logger.info(f"Attempting recovery for {error_record.category.value} error")

        recovery_strategy = self.recovery_strategies.get(error_record.category)
        if recovery_strategy:
            try:
                success = recovery_strategy(error_record)
                error_record.recovery_action = f"Applied {error_record.category.value} recovery strategy"
                return success
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")
                error_record.recovery_action = f"Recovery failed: {str(e)}"
                return False

        return False

    def _recover_network_error(self, error_record: ErrorRecord) -> bool:
        """Recover from network-related errors."""
        logger.info("Applying network error recovery")

        # Wait with exponential backoff
        wait_time = min(30 * (2 ** error_record.retry_count), 300)
        logger.info(f"Waiting {wait_time} seconds for network recovery")
        time.sleep(wait_time)

        # Test network connectivity (basic check)
        try:
            import urllib.request
            urllib.request.urlopen('https://www.google.com', timeout=10)
            logger.info("Network connectivity restored")
            return True
        except:
            logger.warning("Network still unavailable")
            return False

    def _recover_database_error(self, error_record: ErrorRecord) -> bool:
        """Recover from database-related errors."""
        logger.info("Applying database error recovery")

        # Wait for potential lock release
        time.sleep(10)

        # Try to reset database connection
        try:
            # Force reconnection
            if hasattr(self.session_mgr, 'database_manager'):
                self.session_mgr.database_manager.reset_connection()
            logger.info("Database connection reset")
            return True
        except Exception as e:
            logger.error(f"Database recovery failed: {e}")
            return False

    def _recover_processing_error(self, error_record: ErrorRecord) -> bool:
        """Recover from LLM processing errors."""
        logger.info("Applying processing error recovery")

        # Wait and potentially switch models or reduce batch size
        time.sleep(30)

        # Could implement model switching logic here
        logger.info("Processing recovery attempt completed")
        return True

    def _recover_resource_error(self, error_record: ErrorRecord) -> bool:
        """Recover from resource-related errors."""
        logger.info("Applying resource error recovery")

        # Force garbage collection
        import gc
        gc.collect()

        # Wait for resource cleanup
        time.sleep(60)

        logger.info("Resource cleanup completed")
        return True

    def _emergency_shutdown(self, reason: str):
        """Perform emergency shutdown with full state preservation."""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")

        # Save current state
        try:
            if self.session_mgr.session:
                self.session_mgr.save_session()
                logger.info("Session state saved during emergency shutdown")
        except Exception as e:
            logger.error(f"Failed to save session during emergency shutdown: {e}")

        # Generate error report
        self._generate_error_report()

        # Set shutdown flag
        self.shutdown_requested = True

    def _generate_error_report(self) -> str:
        """Generate comprehensive error analysis report."""
        if not self.error_history:
            return "No errors recorded"

        report = ["\n" + "="*60]
        report.append("ERROR ANALYSIS REPORT")
        report.append("="*60)

        # Error summary by category
        category_counts = {}
        severity_counts = {}

        for error in self.error_history:
            category_counts[error.category] = category_counts.get(error.category, 0) + 1
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1

        report.append(f"Total errors: {len(self.error_history)}")
        report.append(f"Error timespan: {self.error_history[0].timestamp} to {self.error_history[-1].timestamp}")
        report.append("\nBy Category:")
        for category, count in category_counts.items():
            report.append(f"  {category.value}: {count}")

        report.append("\nBy Severity:")
        for severity, count in severity_counts.items():
            report.append(f"  {severity.value}: {count}")

        # Recent critical errors
        critical_errors = [e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL]
        if critical_errors:
            report.append("\nRecent Critical Errors:")
            for error in critical_errors[-5:]:  # Last 5 critical errors
                report.append(f"  {error.timestamp}: {error.message}")

        # Recovery success rate
        recoverable_errors = [e for e in self.error_history if e.recoverable]
        successful_recoveries = [e for e in recoverable_errors if e.recovery_action and 'failed' not in e.recovery_action]

        if recoverable_errors:
            recovery_rate = len(successful_recoveries) / len(recoverable_errors) * 100
            report.append(f"\nRecovery success rate: {recovery_rate:.1f}%")

        report.append("="*60)

        error_report = "\n".join(report)
        logger.info(error_report)

        # Save to file
        try:
            report_file = Path(config.data_dir) / "logs" / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_file.write_text(error_report)
            logger.info(f"Error report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")

        return error_report

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {'total_errors': 0}

        # Calculate statistics
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history if (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour

        category_counts = {}
        severity_counts = {}
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1

        # Recovery stats
        recoverable_errors = [e for e in self.error_history if e.recoverable]
        successful_recoveries = [e for e in recoverable_errors if e.recovery_action and 'failed' not in e.recovery_action]

        return {
            'total_errors': total_errors,
            'recent_errors_count': len(recent_errors),
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'recoverable_errors': len(recoverable_errors),
            'successful_recoveries': len(successful_recoveries),
            'recovery_success_rate': len(successful_recoveries) / len(recoverable_errors) * 100 if recoverable_errors else 0,
            'consecutive_failures': self.consecutive_failures,
            'error_history_timespan': {
                'start': self.error_history[0].timestamp.isoformat(),
                'end': self.error_history[-1].timestamp.isoformat()
            } if self.error_history else None
        }

    def _calculate_health_score(self) -> float:
        """Calculate pipeline health score (0-100)."""
        if not self.error_history:
            return 100.0

        # Base score
        score = 100.0

        # Deduct for consecutive failures
        score -= self.consecutive_failures * 15

        # Deduct for recent errors (last hour)
        recent_errors = [e for e in self.error_history if (datetime.now() - e.timestamp).total_seconds() < 3600]
        score -= len(recent_errors) * 5

        # Deduct for critical errors
        critical_errors = [e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL]
        score -= len(critical_errors) * 20

        # Deduct for open circuit breakers
        open_circuits = [cb for cb in self.circuit_breakers.values() if cb.state == 'open']
        score -= len(open_circuits) * 25

        return max(0.0, score)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Medical Rotation Pipeline - Complete orchestrator for rotating medical research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Basic operation modes
    parser.add_argument('--papers-per-condition', type=int, default=10,
                       help='Number of papers to collect per condition (default: 10)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume existing session')
    parser.add_argument('--status', action='store_true',
                       help='Show current pipeline status and exit')

    # Test mode
    parser.add_argument('--test-condition', type=str,
                       help='Run pipeline for single test condition')
    parser.add_argument('--papers', type=int, default=5,
                       help='Number of papers for test condition (default: 5)')

    # Output options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--session-file', type=str,
                       help='Custom session file path')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        # Initialize session manager with custom file if specified
        if args.session_file:
            session_mgr = RotationSessionManager(Path(args.session_file))
        else:
            session_mgr = session_manager

        # Initialize pipeline
        pipeline = MedicalRotationPipeline(session_mgr)

        # Handle status request
        if args.status:
            status = pipeline.get_pipeline_status()
            if 'error' in status:
                print(status['error'])
                return False

            print("Medical Rotation Pipeline Status")
            print("="*50)
            print(f"Session: {status['session_id']}")
            print(f"Active: {status['is_active']}")
            print(f"Iteration: {status['current_iteration']}")
            print(f"Current: {status['current_specialty']} -> {status['current_condition']}")
            print(f"Progress: {status['progress_percent']:.1f}%")
            print(f"Completed: {status['completed_conditions']}/{status['total_conditions']} conditions")
            print(f"Runtime: {status['pipeline_runtime_formatted']}")

            if status['has_interruption']:
                print(f"Interrupted in: {status['interruption_phase']} phase")

            return True

        # Run pipeline
        logger.info("Starting medical rotation pipeline")

        result = pipeline.run_rotation_pipeline(
            papers_per_condition=args.papers_per_condition,
            test_condition=args.test_condition,
            resume=args.resume
        )

        # Print summary
        if result['success']:
            print("\n" + "="*60)
            print("MEDICAL ROTATION PIPELINE COMPLETED")
            print("="*60)

            if result.get('test_mode'):
                print(f"Test condition: {result['test_condition']}")
                print(f"Test papers: {result['test_papers']}")
            else:
                print(f"Session: {result['session_id']}")
                print(f"Runtime: {result['execution_time_formatted']}")
                print(f"Conditions completed: {result['conditions_completed']}")

            print(f"Papers collected: {result['total_papers_collected']}")
            print(f"Papers processed: {result['total_papers_processed']}")
            print(f"Interventions extracted: {result['total_interventions_extracted']}")
            print(f"Duplicates removed: {result['total_duplicates_removed']}")

            if not result.get('test_mode'):
                print(f"Processing rate: {result['papers_per_hour']:.1f} papers/hour")
                print(f"Extraction rate: {result['interventions_per_hour']:.1f} interventions/hour")
        else:
            print(f"\nPipeline failed: {result.get('error', 'Unknown error')}")
            return False

        return True

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\nPipeline interrupted. Session state saved.")
        return True
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        print(f"Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)