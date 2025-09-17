#!/usr/bin/env python3
"""
Autonomous Research Orchestrator - Ultra-robust overnight research automation

This script runs uninterrupted for hours/days, processing multiple health conditions
with comprehensive fault tolerance, automatic recovery, and progress persistence.

Features:
- Multi-condition iterative processing (IBS → IBD → GERD → etc.)
- Automatic restart on any error (network, GPU overheat, API failures)
- Session persistence (can resume after system reboot)
- Thermal protection with automatic cooling
- Network resilience with exponential backoff
- Progress tracking and detailed logging
- Graceful shutdown handling

Usage:
    # Run overnight with default conditions
    python autonomous_research_orchestrator.py

    # Custom conditions and targets
    python autonomous_research_orchestrator.py --conditions "ibs,ibd,gerd,crohns" --target-per-condition 1000

    # Resume existing session
    python autonomous_research_orchestrator.py --resume

    # Status check
    python autonomous_research_orchestrator.py --status
"""

import sys
import json
import time
import signal
import argparse
import traceback
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import queue

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import setup_logging
from src.paper_collection.database_manager import database_manager
from src.paper_collection.pubmed_collector import PubMedCollector
from src.llm.dual_model_analyzer import DualModelAnalyzer

logger = setup_logging(__name__, 'autonomous_orchestrator.log')


@dataclass
class OrchestrationConfig:
    """Configuration for autonomous research orchestration."""
    conditions: List[str]
    target_papers_per_condition: int = 1000
    batch_size_collection: int = 100  # Papers collected per iteration
    batch_size_processing: int = 5    # Papers processed per LLM batch
    min_year: int = 2015
    max_temp_celsius: int = 75
    cooling_temp_celsius: int = 65
    thermal_check_interval: int = 30  # seconds
    network_retry_delays: List[int] = None  # [30, 60, 120, 300, 600] seconds
    max_consecutive_failures: int = 10
    session_save_interval: int = 60   # seconds
    enable_s2_enrichment: bool = True
    auto_restart_on_critical_error: bool = True

    def __post_init__(self):
        if self.network_retry_delays is None:
            self.network_retry_delays = [30, 60, 120, 300, 600, 900]  # Up to 15 min


@dataclass
class ConditionStatus:
    """Tracking status for each condition."""
    condition: str
    papers_collected: int = 0
    papers_processed: int = 0
    target_papers: int = 1000
    last_collection_batch: int = 0
    last_processing_batch: int = 0
    collection_start_time: Optional[str] = None
    processing_start_time: Optional[str] = None
    completion_time: Optional[str] = None
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    status: str = 'pending'  # pending, collecting, processing, completed, failed

    @property
    def is_collection_complete(self) -> bool:
        return self.papers_collected >= self.target_papers

    @property
    def needs_processing(self) -> bool:
        return self.papers_processed < self.papers_collected


@dataclass
class OrchestrationSession:
    """Complete session state for persistence."""
    session_id: str
    config: OrchestrationConfig
    conditions_status: Dict[str, ConditionStatus]
    global_stats: Dict[str, Any]
    start_time: str
    last_update: str
    is_active: bool = True
    total_runtime_hours: float = 0.0
    restart_count: int = 0


class ThermalProtectionSystem:
    """Advanced thermal protection with predictive cooling."""

    def __init__(self, max_temp: int = 75, cooling_temp: int = 65):
        self.max_temp = max_temp
        self.cooling_temp = cooling_temp
        self.is_cooling = False
        self.temp_history = []
        self.last_check = 0

    def get_gpu_temperature(self) -> Optional[float]:
        """Get current GPU temperature."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                temp = float(result.stdout.strip())
                self.temp_history.append((time.time(), temp))
                # Keep only last 10 minutes of history
                cutoff = time.time() - 600
                self.temp_history = [(t, temp) for t, temp in self.temp_history if t > cutoff]
                return temp
        except Exception as e:
            logger.warning(f"GPU temperature check failed: {e}")
        return None

    def is_thermal_safe(self) -> tuple[bool, Optional[float]]:
        """Check if thermal conditions are safe for processing."""
        current_temp = self.get_gpu_temperature()

        if current_temp is None:
            logger.warning("Cannot read GPU temperature, assuming safe")
            return True, None

        is_safe = current_temp <= self.max_temp

        # Predictive cooling: if temperature is rising rapidly, warn early
        if len(self.temp_history) >= 3:
            recent_temps = [temp for _, temp in self.temp_history[-3:]]
            if len(recent_temps) >= 2:
                temp_rate = (recent_temps[-1] - recent_temps[0]) / len(recent_temps)
                if temp_rate > 2 and current_temp > (self.max_temp - 5):
                    logger.warning(f"GPU heating rapidly: {temp_rate:.1f}°C/min, current: {current_temp}°C")

        return is_safe, current_temp

    def wait_for_cooling(self) -> None:
        """Wait for GPU to cool down to safe operating temperature."""
        logger.info(f"🌡️ GPU thermal protection activated! Cooling to {self.cooling_temp}°C...")
        self.is_cooling = True

        cooling_start = time.time()
        while True:
            is_safe, current_temp = self.is_thermal_safe()

            if current_temp is None:
                logger.warning("Cannot read temperature during cooling, resuming")
                break

            if current_temp <= self.cooling_temp:
                cooling_duration = time.time() - cooling_start
                logger.info(f"🌡️ GPU cooled to {current_temp}°C after {cooling_duration/60:.1f} minutes")
                self.is_cooling = False
                break

            logger.info(f"🌡️ Cooling... Current: {current_temp}°C Target: {self.cooling_temp}°C")
            time.sleep(30)


class NetworkResilienceManager:
    """Handles network issues with intelligent retry strategies."""

    def __init__(self, retry_delays: List[int]):
        self.retry_delays = retry_delays
        self.consecutive_failures = 0

    def execute_with_retry(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute operation with exponential backoff retry."""
        for attempt in range(len(self.retry_delays)):
            try:
                result = operation_func(*args, **kwargs)
                if self.consecutive_failures > 0:
                    logger.info(f"✅ {operation_name} recovered after {self.consecutive_failures} failures")
                self.consecutive_failures = 0
                return result

            except Exception as e:
                self.consecutive_failures += 1
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]

                logger.warning(f"⚠️ {operation_name} failed (attempt {attempt + 1}): {e}")

                if attempt < len(self.retry_delays) - 1:
                    logger.info(f"⏳ Retrying {operation_name} in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"💥 {operation_name} failed after {len(self.retry_delays)} attempts")
                    raise

        raise RuntimeError(f"{operation_name} exhausted all retry attempts")


class SessionManager:
    """Manages session persistence and recovery."""

    def __init__(self, session_file: str = "autonomous_session.json"):
        self.session_file = Path(session_file)
        self.session: Optional[OrchestrationSession] = None
        self.save_lock = threading.Lock()

    def create_new_session(self, config: OrchestrationConfig) -> OrchestrationSession:
        """Create a new orchestration session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        conditions_status = {}
        for condition in config.conditions:
            conditions_status[condition] = ConditionStatus(
                condition=condition,
                target_papers=config.target_papers_per_condition
            )

        session = OrchestrationSession(
            session_id=session_id,
            config=config,
            conditions_status=conditions_status,
            global_stats={
                'total_papers_collected': 0,
                'total_papers_processed': 0,
                'total_interventions_extracted': 0,
                'conditions_completed': 0
            },
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat()
        )

        self.session = session
        self.save_session()

        logger.info(f"📋 Created new session: {session_id}")
        logger.info(f"🎯 Target: {len(config.conditions)} conditions, {config.target_papers_per_condition} papers each")

        return session

    def load_existing_session(self) -> Optional[OrchestrationSession]:
        """Load existing session from file."""
        if not self.session_file.exists():
            return None

        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)

            # Reconstruct session object
            config_data = data['config']
            config = OrchestrationConfig(**config_data)

            conditions_status = {}
            for condition, status_data in data['conditions_status'].items():
                conditions_status[condition] = ConditionStatus(**status_data)

            session = OrchestrationSession(
                session_id=data['session_id'],
                config=config,
                conditions_status=conditions_status,
                global_stats=data['global_stats'],
                start_time=data['start_time'],
                last_update=data['last_update'],
                is_active=data.get('is_active', True),
                total_runtime_hours=data.get('total_runtime_hours', 0.0),
                restart_count=data.get('restart_count', 0)
            )

            self.session = session
            logger.info(f"📋 Loaded existing session: {session.session_id}")
            logger.info(f"⏱️ Runtime: {session.total_runtime_hours:.1f} hours, Restarts: {session.restart_count}")

            return session

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def save_session(self) -> None:
        """Save current session to file."""
        if not self.session:
            return

        with self.save_lock:
            try:
                # Update runtime
                start_time = datetime.fromisoformat(self.session.start_time)
                runtime_delta = datetime.now() - start_time
                self.session.total_runtime_hours = runtime_delta.total_seconds() / 3600
                self.session.last_update = datetime.now().isoformat()

                # Convert to serializable format
                session_data = {
                    'session_id': self.session.session_id,
                    'config': asdict(self.session.config),
                    'conditions_status': {k: asdict(v) for k, v in self.session.conditions_status.items()},
                    'global_stats': self.session.global_stats,
                    'start_time': self.session.start_time,
                    'last_update': self.session.last_update,
                    'is_active': self.session.is_active,
                    'total_runtime_hours': self.session.total_runtime_hours,
                    'restart_count': self.session.restart_count
                }

                # Atomic write
                temp_file = self.session_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(session_data, f, indent=2)

                temp_file.replace(self.session_file)

            except Exception as e:
                logger.error(f"Failed to save session: {e}")

    def update_condition_status(self, condition: str, status_update: Dict[str, Any]) -> None:
        """Update status for a specific condition."""
        if not self.session or condition not in self.session.conditions_status:
            return

        condition_status = self.session.conditions_status[condition]
        for key, value in status_update.items():
            if hasattr(condition_status, key):
                setattr(condition_status, key, value)

        self.save_session()


class AutonomousResearchOrchestrator:
    """Ultra-robust autonomous research orchestrator for overnight operation."""

    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.session_manager = SessionManager()
        self.thermal_system = ThermalProtectionSystem(
            max_temp=config.max_temp_celsius,
            cooling_temp=config.cooling_temp_celsius
        )
        self.network_manager = NetworkResilienceManager(config.network_retry_delays)

        # Core components
        self.collector = PubMedCollector()
        self.analyzer = DualModelAnalyzer()

        # Control flags
        self.shutdown_requested = False
        self.pause_requested = False

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Auto-save thread
        self.auto_save_thread = None
        self.auto_save_stop = threading.Event()

        logger.info("🚀 Autonomous Research Orchestrator initialized")

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown signals."""
        logger.info(f"🛑 Shutdown signal received ({signum}). Initiating graceful shutdown...")
        self.shutdown_requested = True

        if self.auto_save_thread:
            self.auto_save_stop.set()

    def _start_auto_save(self):
        """Start automatic session saving thread."""
        def auto_save_worker():
            while not self.auto_save_stop.wait(self.config.session_save_interval):
                self.session_manager.save_session()

        self.auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self.auto_save_thread.start()

    def run_autonomous_research(self, resume: bool = False) -> Dict[str, Any]:
        """Run the complete autonomous research orchestration."""
        try:
            # Load or create session
            if resume:
                session = self.session_manager.load_existing_session()
                if session:
                    session.restart_count += 1
                    logger.info(f"🔄 Resuming session (restart #{session.restart_count})")
                else:
                    logger.warning("No existing session found, creating new one")
                    session = self.session_manager.create_new_session(self.config)
            else:
                session = self.session_manager.create_new_session(self.config)

            # Start auto-save
            self._start_auto_save()

            logger.info("🎯 Starting autonomous research orchestration")
            logger.info(f"📊 Conditions: {list(session.conditions_status.keys())}")

            while not self.shutdown_requested:
                if self._is_orchestration_complete(session):
                    logger.info("🎉 All conditions completed successfully!")
                    break

                # Process each condition iteratively
                for condition in self.config.conditions:
                    if self.shutdown_requested:
                        break

                    try:
                        self._process_condition_cycle(condition, session)
                    except Exception as e:
                        logger.error(f"💥 Error processing condition '{condition}': {e}")
                        logger.error(traceback.format_exc())

                        # Update failure count
                        condition_status = session.conditions_status[condition]
                        condition_status.consecutive_failures += 1
                        condition_status.last_error = str(e)

                        if condition_status.consecutive_failures >= self.config.max_consecutive_failures:
                            logger.error(f"💀 Condition '{condition}' failed too many times, marking as failed")
                            condition_status.status = 'failed'

                        self.session_manager.update_condition_status(condition, asdict(condition_status))

                        # Auto-restart logic
                        if self.config.auto_restart_on_critical_error:
                            logger.info("🔄 Auto-restart enabled, continuing with next condition...")
                            continue
                        else:
                            raise

            # Final session update
            session.is_active = False
            self.session_manager.save_session()

            # Generate final report
            return self._generate_completion_report(session)

        except Exception as e:
            logger.error(f"💥 Critical orchestration error: {e}")
            logger.error(traceback.format_exc())

            if self.config.auto_restart_on_critical_error:
                logger.info("🔄 Auto-restart triggered due to critical error")
                time.sleep(60)  # Brief pause before restart
                return self.run_autonomous_research(resume=True)
            else:
                raise
        finally:
            # Stop auto-save
            if self.auto_save_thread:
                self.auto_save_stop.set()
                self.auto_save_thread.join(timeout=5)

    def _process_condition_cycle(self, condition: str, session: OrchestrationSession) -> None:
        """Process one complete cycle for a condition (collect + process)."""
        condition_status = session.conditions_status[condition]

        if condition_status.status == 'completed':
            logger.debug(f"✅ Condition '{condition}' already completed, skipping")
            return

        if condition_status.status == 'failed':
            logger.debug(f"❌ Condition '{condition}' marked as failed, skipping")
            return

        logger.info(f"🔬 Processing condition: {condition}")
        logger.info(f"📊 Progress: {condition_status.papers_collected}/{condition_status.target_papers} collected, "
                   f"{condition_status.papers_processed} processed")

        try:
            # Collection phase
            if not condition_status.is_collection_complete:
                self._collect_papers_for_condition(condition, session)

            # Processing phase
            if condition_status.needs_processing:
                self._process_papers_for_condition(condition, session)

            # Check completion
            if condition_status.is_collection_complete and not condition_status.needs_processing:
                condition_status.status = 'completed'
                condition_status.completion_time = datetime.now().isoformat()
                session.global_stats['conditions_completed'] += 1

                logger.info(f"🎉 Condition '{condition}' completed successfully!")
                self.session_manager.update_condition_status(condition, asdict(condition_status))

        except Exception as e:
            logger.error(f"Error in condition cycle for '{condition}': {e}")
            raise

    def _collect_papers_for_condition(self, condition: str, session: OrchestrationSession) -> None:
        """Collect papers for a specific condition with network resilience."""
        condition_status = session.conditions_status[condition]

        remaining_papers = condition_status.target_papers - condition_status.papers_collected
        batch_size = min(self.config.batch_size_collection, remaining_papers)

        if batch_size <= 0:
            return

        logger.info(f"📚 Collecting {batch_size} papers for '{condition}'...")

        def collection_operation():
            return self.collector.collect_interventions_by_condition(
                condition=condition,
                min_year=self.config.min_year,
                max_results=batch_size,
                include_fulltext=True,
                use_interleaved_s2=self.config.enable_s2_enrichment
            )

        # Execute with network resilience
        collection_result = self.network_manager.execute_with_retry(
            f"Paper collection for {condition}",
            collection_operation
        )

        # Update status
        papers_collected = collection_result.get('paper_count', 0)
        condition_status.papers_collected += papers_collected
        condition_status.last_collection_batch = papers_collected
        condition_status.status = 'collecting'

        if not condition_status.collection_start_time:
            condition_status.collection_start_time = datetime.now().isoformat()

        session.global_stats['total_papers_collected'] += papers_collected

        logger.info(f"📚 Collected {papers_collected} papers for '{condition}' "
                   f"({condition_status.papers_collected}/{condition_status.target_papers})")

        self.session_manager.update_condition_status(condition, asdict(condition_status))

    def _process_papers_for_condition(self, condition: str, session: OrchestrationSession) -> None:
        """Process papers for a condition with thermal protection."""
        condition_status = session.conditions_status[condition]

        # Get unprocessed papers
        unprocessed_papers = self.analyzer.get_unprocessed_papers()

        if not unprocessed_papers:
            logger.info(f"🤖 No unprocessed papers found for '{condition}'")
            return

        logger.info(f"🤖 Processing {len(unprocessed_papers)} papers for '{condition}'...")

        if not condition_status.processing_start_time:
            condition_status.processing_start_time = datetime.now().isoformat()
            condition_status.status = 'processing'

        # Process in batches with thermal monitoring
        total_processed = 0
        for i in range(0, len(unprocessed_papers), self.config.batch_size_processing):
            if self.shutdown_requested:
                break

            # Thermal safety check
            is_safe, current_temp = self.thermal_system.is_thermal_safe()
            if not is_safe:
                logger.warning(f"🌡️ GPU temperature too high: {current_temp}°C")
                self.thermal_system.wait_for_cooling()

            batch = unprocessed_papers[i:i + self.config.batch_size_processing]

            def processing_operation():
                return self.analyzer.process_papers_batch(
                    papers=batch,
                    save_to_db=True,
                    batch_size=len(batch)
                )

            # Execute with retry
            processing_result = self.network_manager.execute_with_retry(
                f"LLM processing batch for {condition}",
                processing_operation
            )

            batch_processed = processing_result.get('successful_papers', 0)
            total_processed += batch_processed

            logger.info(f"🤖 Processed batch {i//self.config.batch_size_processing + 1}: "
                       f"{batch_processed} papers for '{condition}'")

        # Update status
        condition_status.papers_processed += total_processed
        condition_status.last_processing_batch = total_processed
        session.global_stats['total_papers_processed'] += total_processed

        # Estimate interventions (rough calculation)
        estimated_interventions = total_processed * 2  # Rough estimate
        session.global_stats['total_interventions_extracted'] += estimated_interventions

        logger.info(f"🤖 Processed {total_processed} papers for '{condition}' "
                   f"(total: {condition_status.papers_processed})")

        self.session_manager.update_condition_status(condition, asdict(condition_status))

    def _is_orchestration_complete(self, session: OrchestrationSession) -> bool:
        """Check if all conditions are complete."""
        return all(
            status.status == 'completed'
            for status in session.conditions_status.values()
        )

    def _generate_completion_report(self, session: OrchestrationSession) -> Dict[str, Any]:
        """Generate final completion report."""
        completed_conditions = [
            condition for condition, status in session.conditions_status.items()
            if status.status == 'completed'
        ]

        failed_conditions = [
            condition for condition, status in session.conditions_status.items()
            if status.status == 'failed'
        ]

        report = {
            'session_id': session.session_id,
            'total_runtime_hours': session.total_runtime_hours,
            'restart_count': session.restart_count,
            'conditions_completed': len(completed_conditions),
            'conditions_failed': len(failed_conditions),
            'total_conditions': len(session.conditions_status),
            'completion_rate': len(completed_conditions) / len(session.conditions_status) * 100,
            'global_stats': session.global_stats,
            'completed_conditions': completed_conditions,
            'failed_conditions': failed_conditions,
            'final_status': 'success' if not failed_conditions else 'partial_success'
        }

        logger.info("📊 Final Report:")
        logger.info(f"   Runtime: {report['total_runtime_hours']:.1f} hours")
        logger.info(f"   Restarts: {report['restart_count']}")
        logger.info(f"   Completed: {report['conditions_completed']}/{report['total_conditions']} conditions")
        logger.info(f"   Papers: {session.global_stats['total_papers_collected']} collected, "
                   f"{session.global_stats['total_papers_processed']} processed")

        return report

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        session = self.session_manager.session
        if not session:
            return {'error': 'No active session'}

        # Add thermal status
        is_safe, current_temp = self.thermal_system.is_thermal_safe()

        status = {
            'session_id': session.session_id,
            'runtime_hours': session.total_runtime_hours,
            'restart_count': session.restart_count,
            'conditions_status': {k: asdict(v) for k, v in session.conditions_status.items()},
            'global_stats': session.global_stats,
            'thermal_status': {
                'current_temp': current_temp,
                'is_safe': is_safe,
                'is_cooling': self.thermal_system.is_cooling
            },
            'network_status': {
                'consecutive_failures': self.network_manager.consecutive_failures
            }
        }

        return status


def main():
    """Main entry point for autonomous research orchestrator."""
    parser = argparse.ArgumentParser(
        description='Autonomous Research Orchestrator - Overnight Research Automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run overnight with default conditions
    python autonomous_research_orchestrator.py

    # Custom conditions and targets
    python autonomous_research_orchestrator.py --conditions "ibs,ibd,gerd,crohns" --target 1000

    # Resume existing session
    python autonomous_research_orchestrator.py --resume

    # Status check
    python autonomous_research_orchestrator.py --status
        """
    )

    parser.add_argument('--conditions', type=str,
                       default='ibs,ibd,gerd,constipation,diarrhea,crohns,ulcerative_colitis',
                       help='Comma-separated list of conditions to research')
    parser.add_argument('--target', type=int, default=1000,
                       help='Target papers per condition (default: 1000)')
    parser.add_argument('--batch-collection', type=int, default=100,
                       help='Papers collected per iteration (default: 100)')
    parser.add_argument('--batch-processing', type=int, default=5,
                       help='Papers processed per LLM batch (default: 5)')
    parser.add_argument('--min-year', type=int, default=2015,
                       help='Minimum publication year (default: 2015)')
    parser.add_argument('--max-temp', type=int, default=75,
                       help='Maximum GPU temperature before cooling (default: 75)')
    parser.add_argument('--cooling-temp', type=int, default=65,
                       help='Resume temperature after cooling (default: 65)')
    parser.add_argument('--max-failures', type=int, default=10,
                       help='Max consecutive failures per condition (default: 10)')
    parser.add_argument('--no-s2', action='store_true',
                       help='Disable Semantic Scholar enrichment')
    parser.add_argument('--no-auto-restart', action='store_true',
                       help='Disable automatic restart on critical errors')
    parser.add_argument('--resume', action='store_true',
                       help='Resume existing session')
    parser.add_argument('--status', action='store_true',
                       help='Show current status and exit')

    args = parser.parse_args()

    # Parse conditions
    conditions = [c.strip() for c in args.conditions.split(',')]

    # Create configuration
    config = OrchestrationConfig(
        conditions=conditions,
        target_papers_per_condition=args.target,
        batch_size_collection=args.batch_collection,
        batch_size_processing=args.batch_processing,
        min_year=args.min_year,
        max_temp_celsius=args.max_temp,
        cooling_temp_celsius=args.cooling_temp,
        max_consecutive_failures=args.max_failures,
        enable_s2_enrichment=not args.no_s2,
        auto_restart_on_critical_error=not args.no_auto_restart
    )

    # Initialize orchestrator
    orchestrator = AutonomousResearchOrchestrator(config)

    if args.status:
        status = orchestrator.get_status()
        print("📊 Orchestration Status:")
        print(json.dumps(status, indent=2))
        return

    try:
        logger.info("🚀 Starting Autonomous Research Orchestrator")
        logger.info(f"🎯 Target: {len(conditions)} conditions, {args.target} papers each")
        logger.info(f"📊 Total target: {len(conditions) * args.target} papers")

        results = orchestrator.run_autonomous_research(resume=args.resume)

        print("\n🎉 Orchestration completed!")
        print(f"📊 Runtime: {results['total_runtime_hours']:.1f} hours")
        print(f"✅ Completed: {results['conditions_completed']}/{results['total_conditions']} conditions")
        print(f"📄 Papers: {results['global_stats']['total_papers_collected']} collected, "
              f"{results['global_stats']['total_papers_processed']} processed")

    except KeyboardInterrupt:
        print("\n🛑 Orchestration interrupted by user")
        logger.info("Orchestration interrupted by user")
    except Exception as e:
        print(f"\n💥 Orchestration failed: {e}")
        logger.error(f"Orchestration failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()