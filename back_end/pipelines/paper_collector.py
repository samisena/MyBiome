#!/usr/bin/env python3
"""
Unified Paper Collector - Robust overnight paper collection with comprehensive features

This script provides comprehensive paper collection from PubMed and Semantic Scholar with:
- Multi-condition batch processing
- Network resilience with exponential backoff
- Session persistence and recovery
- Progress tracking and incremental saving
- Auto-restart on errors
- Overnight operation capacity
- Thermal monitoring (if GPU operations needed)

Features:
- Intelligent retry logic for network failures
- Session recovery after interruptions
- Progress checkpoints and auto-save
- Multi-source data collection (PubMed, S2, PMC, Unpaywall)
- Configurable batch sizes and targets
- Comprehensive error handling
- Real-time progress reporting

Usage:
    # Single condition with defaults
    python paper_collector.py "ibs" --max-papers 100

    # Multiple conditions overnight collection
    python paper_collector.py --conditions "ibs,gerd,crohns" --target-per-condition 1000

    # Resume interrupted session
    python paper_collector.py --resume

    # Custom configuration
    python paper_collector.py --config collection_config.json

Examples:
    # Standard collection with S2 enrichment
    python paper_collector.py "inflammatory bowel disease" --max-papers 200 --min-year 2015

    # Overnight multi-condition campaign
    python paper_collector.py --conditions "ibs,gerd,crohns,colitis" --target-per-condition 500 --overnight

    # Resume with status check
    python paper_collector.py --resume --status

    # Traditional mode without interleaved S2
    python paper_collector.py "diabetes" --traditional-mode --max-papers 100
"""

import sys
import json
import time
import signal
import argparse
import traceback
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.data.config import config, setup_logging
    from src.data.repositories import repository_manager
    from src.paper_collection.database_manager import database_manager
    from src.paper_collection.pubmed_collector import PubMedCollector
    from src.paper_collection.semantic_scholar_enrichment import run_semantic_scholar_enrichment

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)

logger = setup_logging(__name__, 'paper_collector.log')


@dataclass
class CollectionConfig:
    """Configuration for paper collection orchestrator."""
    # Target configuration
    conditions: List[str]
    target_papers_per_condition: int = 100
    min_year: int = 2015
    max_year: Optional[int] = None

    # Collection modes
    use_interleaved_s2: bool = True
    include_fulltext: bool = True
    traditional_mode: bool = False

    # Batch processing
    batch_size_collection: int = 50  # Papers collected per API call
    collection_interval: int = 5     # Seconds between batches

    # Network resilience
    network_retry_delays: List[int] = None  # [30, 60, 120, 300, 600] seconds
    max_consecutive_failures: int = 10
    api_timeout: int = 30

    # Session management
    session_file: Path = Path("collection_session.json")
    auto_save_interval: int = 60   # seconds
    checkpoint_interval: int = 10  # papers

    # Output options
    output_dir: Path = Path("collection_results")
    save_intermediate: bool = True
    export_formats: List[str] = None  # ['json', 'csv']

    # Operational modes
    overnight_mode: bool = False
    auto_restart_on_error: bool = True
    enable_status_reporting: bool = True

    def __post_init__(self):
        if self.network_retry_delays is None:
            self.network_retry_delays = [30, 60, 120, 300, 600, 900]  # Up to 15 min
        if self.export_formats is None:
            self.export_formats = ['json']
        self.output_dir = Path(self.output_dir)
        self.session_file = Path(self.session_file)


@dataclass
class ConditionProgress:
    """Progress tracking for individual conditions."""
    condition: str
    target_papers: int = 100
    papers_collected: int = 0
    papers_enriched: int = 0
    last_batch_size: int = 0
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    status: str = 'pending'  # pending, collecting, enriching, completed, failed
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    collection_stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.collection_stats is None:
            self.collection_stats = {
                'pubmed_papers': 0,
                's2_papers': 0,
                'total_searched': 0,
                'expansion_factor': 0.0
            }

    @property
    def is_complete(self) -> bool:
        return self.papers_collected >= self.target_papers

    @property
    def progress_percent(self) -> float:
        if self.target_papers > 0:
            return min((self.papers_collected / self.target_papers) * 100, 100.0)
        return 0.0

    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.completion_time:
            return self.completion_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None


@dataclass
class CollectionSession:
    """Complete session state for persistence."""
    session_id: str
    config: CollectionConfig
    conditions_progress: Dict[str, ConditionProgress]
    global_stats: Dict[str, Any]
    start_time: str
    last_update: str
    is_active: bool = True
    total_runtime_hours: float = 0.0
    restart_count: int = 0


class NetworkResilienceManager:
    """Handles network issues with intelligent retry strategies."""

    def __init__(self, retry_delays: List[int]):
        self.retry_delays = retry_delays
        self.consecutive_failures = 0
        self.total_retries = 0

    def execute_with_retry(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute operation with exponential backoff retry."""
        for attempt in range(len(self.retry_delays)):
            try:
                result = operation_func(*args, **kwargs)
                if self.consecutive_failures > 0:
                    logger.info(f"SUCCESS: {operation_name} recovered after {self.consecutive_failures} failures")
                self.consecutive_failures = 0
                return result

            except Exception as e:
                self.consecutive_failures += 1
                self.total_retries += 1
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]

                logger.warning(f"WARNING: {operation_name} failed (attempt {attempt + 1}): {e}")

                if attempt < len(self.retry_delays) - 1:
                    logger.info(f"⏳ Retrying {operation_name} in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"💥 {operation_name} failed after {len(self.retry_delays)} attempts")
                    raise

        raise RuntimeError(f"{operation_name} exhausted all retry attempts")

    def get_failure_stats(self) -> Dict[str, int]:
        """Get network failure statistics."""
        return {
            'consecutive_failures': self.consecutive_failures,
            'total_retries': self.total_retries
        }


class SessionManager:
    """Manages session persistence and recovery."""

    def __init__(self, session_file: Path):
        self.session_file = session_file
        self.session: Optional[CollectionSession] = None
        self.save_lock = threading.Lock()

    def create_new_session(self, config: CollectionConfig) -> CollectionSession:
        """Create a new collection session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        conditions_progress = {}
        for condition in config.conditions:
            conditions_progress[condition] = ConditionProgress(
                condition=condition,
                target_papers=config.target_papers_per_condition
            )

        session = CollectionSession(
            session_id=session_id,
            config=config,
            conditions_progress=conditions_progress,
            global_stats={
                'total_papers_collected': 0,
                'total_papers_enriched': 0,
                'total_conditions': len(config.conditions),
                'conditions_completed': 0,
                'network_failures': 0,
                'api_calls_made': 0
            },
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat()
        )

        self.session = session
        self.save_session()

        logger.info(f"Created new collection session: {session_id}")
        logger.info(f"Target: {len(config.conditions)} conditions, {config.target_papers_per_condition} papers each")

        return session

    def load_existing_session(self) -> Optional[CollectionSession]:
        """Load existing session from file."""
        if not self.session_file.exists():
            return None

        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)

            # Reconstruct session object
            config_data = data['config']
            # Handle Path objects in config
            if 'session_file' in config_data:
                config_data['session_file'] = Path(config_data['session_file'])
            if 'output_dir' in config_data:
                config_data['output_dir'] = Path(config_data['output_dir'])

            config = CollectionConfig(**config_data)

            conditions_progress = {}
            for condition, progress_data in data['conditions_progress'].items():
                # Convert datetime strings back to datetime objects
                if progress_data.get('start_time'):
                    progress_data['start_time'] = datetime.fromisoformat(progress_data['start_time'])
                if progress_data.get('completion_time'):
                    progress_data['completion_time'] = datetime.fromisoformat(progress_data['completion_time'])

                conditions_progress[condition] = ConditionProgress(**progress_data)

            session = CollectionSession(
                session_id=data['session_id'],
                config=config,
                conditions_progress=conditions_progress,
                global_stats=data['global_stats'],
                start_time=data['start_time'],
                last_update=data['last_update'],
                is_active=data.get('is_active', True),
                total_runtime_hours=data.get('total_runtime_hours', 0.0),
                restart_count=data.get('restart_count', 0)
            )

            self.session = session
            logger.info(f"Loaded existing session: {session.session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def save_session(self):
        """Save current session to file."""
        if not self.session:
            return

        with self.save_lock:
            try:
                # Convert session to JSON-serializable format
                session_dict = asdict(self.session)

                # Convert datetime objects to strings
                for condition, progress in session_dict['conditions_progress'].items():
                    if progress.get('start_time'):
                        progress['start_time'] = progress['start_time'].isoformat() if isinstance(progress['start_time'], datetime) else progress['start_time']
                    if progress.get('completion_time'):
                        progress['completion_time'] = progress['completion_time'].isoformat() if isinstance(progress['completion_time'], datetime) else progress['completion_time']

                # Convert Path objects to strings
                if 'session_file' in session_dict['config']:
                    session_dict['config']['session_file'] = str(session_dict['config']['session_file'])
                if 'output_dir' in session_dict['config']:
                    session_dict['config']['output_dir'] = str(session_dict['config']['output_dir'])

                session_dict['last_update'] = datetime.now().isoformat()

                with open(self.session_file, 'w') as f:
                    json.dump(session_dict, f, indent=2, default=str)

            except Exception as e:
                logger.error(f"Failed to save session: {e}")

    def update_condition_progress(self, condition: str, progress: ConditionProgress):
        """Update progress for a specific condition."""
        if self.session:
            self.session.conditions_progress[condition] = progress
            self.save_session()


class PaperCollector:
    """
    Unified paper collector with comprehensive robustness features.
    """

    def __init__(self, config: CollectionConfig):
        self.config = config
        self.session_start = datetime.now()

        # Initialize components
        self.collector = PubMedCollector()
        self.network_manager = NetworkResilienceManager(config.network_retry_delays)
        self.session_manager = SessionManager(config.session_file)

        # Control flags
        self.shutdown_requested = False
        self.pause_requested = False

        # Auto-save thread
        self.auto_save_thread = None
        self.auto_save_stop = threading.Event()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"📤 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True

    def _start_auto_save(self):
        """Start automatic session saving thread."""
        def auto_save():
            while not self.auto_save_stop.wait(self.config.auto_save_interval):
                try:
                    self.session_manager.save_session()
                    logger.debug("💾 Auto-saved session")
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")

        self.auto_save_thread = threading.Thread(target=auto_save, daemon=True)
        self.auto_save_thread.start()

    def run_collection_campaign(self, resume: bool = False) -> Dict[str, Any]:
        """Run comprehensive paper collection campaign."""
        try:
            # Load or create session
            if resume:
                session = self.session_manager.load_existing_session()
                if session:
                    session.restart_count += 1
                    logger.info(f"Resuming session (restart #{session.restart_count})")
                else:
                    logger.warning("No existing session found, creating new one")
                    session = self.session_manager.create_new_session(self.config)
            else:
                session = self.session_manager.create_new_session(self.config)

            # Start auto-save
            self._start_auto_save()

            logger.info("Starting paper collection campaign")
            logger.info(f"Conditions: {list(session.conditions_progress.keys())}")

            # Main collection loop
            while not self.shutdown_requested:
                if self._is_campaign_complete(session):
                    logger.info("Collection campaign completed successfully!")
                    break

                # Process each condition
                for condition in self.config.conditions:
                    if self.shutdown_requested:
                        break

                    try:
                        self._process_condition(condition, session)
                    except Exception as e:
                        logger.error(f"💥 Error processing condition '{condition}': {e}")
                        logger.error(traceback.format_exc())

                        # Update failure count
                        progress = session.conditions_progress[condition]
                        progress.consecutive_failures += 1
                        progress.last_error = str(e)

                        if progress.consecutive_failures >= self.config.max_consecutive_failures:
                            logger.error(f"💀 Condition '{condition}' failed too many times, marking as failed")
                            progress.status = 'failed'

                        self.session_manager.update_condition_progress(condition, progress)

                        # Auto-restart logic
                        if self.config.auto_restart_on_error:
                            logger.info("Auto-restart enabled, continuing with next condition...")
                            continue
                        else:
                            raise

                # Brief pause between cycles
                if not self._is_campaign_complete(session):
                    time.sleep(self.config.collection_interval)

            # Final session update
            session.is_active = False
            session.total_runtime_hours = (datetime.now() - self.session_start).total_seconds() / 3600
            self.session_manager.save_session()

            # Generate final report
            return self._generate_final_report(session)

        except Exception as e:
            logger.error(f"💥 Critical collection error: {e}")
            logger.error(traceback.format_exc())

            if self.config.auto_restart_on_error:
                logger.info("Auto-restart triggered due to critical error")
                time.sleep(60)  # Brief pause before restart
                return self.run_collection_campaign(resume=True)
            else:
                raise
        finally:
            # Stop auto-save
            if self.auto_save_thread:
                self.auto_save_stop.set()
                self.auto_save_thread.join(timeout=5)

    def _is_campaign_complete(self, session: CollectionSession) -> bool:
        """Check if the entire collection campaign is complete."""
        for progress in session.conditions_progress.values():
            if progress.status not in ['completed', 'failed']:
                return False
        return True

    def _process_condition(self, condition: str, session: CollectionSession):
        """Process collection for a single condition."""
        progress = session.conditions_progress[condition]

        if progress.status == 'completed':
            logger.debug(f"Condition '{condition}' already completed, skipping")
            return

        if progress.status == 'failed':
            logger.debug(f"Condition '{condition}' marked as failed, skipping")
            return

        if progress.is_complete:
            progress.status = 'completed'
            progress.completion_time = datetime.now()
            session.global_stats['conditions_completed'] += 1
            logger.info(f"Condition '{condition}' completed successfully!")
            self.session_manager.update_condition_progress(condition, progress)
            return

        logger.info(f"Processing condition: {condition}")
        logger.info(f"Progress: {progress.papers_collected}/{progress.target_papers} "
                   f"({progress.progress_percent:.1f}%)")

        # Start timing if first collection
        if not progress.start_time:
            progress.start_time = datetime.now()
            progress.status = 'collecting'

        # Calculate batch size for this iteration
        remaining_papers = progress.target_papers - progress.papers_collected
        batch_size = min(self.config.batch_size_collection, remaining_papers)

        if batch_size <= 0:
            return

        # Perform collection
        self._collect_papers_for_condition(condition, batch_size, session)

        # Update global stats
        session.global_stats['api_calls_made'] += 1
        self.session_manager.save_session()

    def _collect_papers_for_condition(self, condition: str, batch_size: int, session: CollectionSession):
        """Collect papers for a specific condition with network resilience."""
        progress = session.conditions_progress[condition]

        logger.info(f"Collecting {batch_size} papers for '{condition}'...")

        def collection_operation():
            return self.collector.collect_interventions_by_condition(
                condition=condition,
                min_year=self.config.min_year,
                max_year=self.config.max_year,
                max_results=batch_size,
                include_fulltext=self.config.include_fulltext,
                use_interleaved_s2=self.config.use_interleaved_s2 and not self.config.traditional_mode
            )

        try:
            # Execute with network resilience
            collection_result = self.network_manager.execute_with_retry(
                f"Paper collection for {condition}",
                collection_operation
            )

            # Update progress
            papers_collected = collection_result.get('paper_count', 0)
            progress.papers_collected += papers_collected
            progress.last_batch_size = papers_collected

            # Update collection stats
            if collection_result.get('interleaved_workflow'):
                progress.collection_stats['pubmed_papers'] += collection_result.get('pubmed_papers', 0)
                progress.collection_stats['s2_papers'] += collection_result.get('s2_similar_papers', 0)
                progress.collection_stats['expansion_factor'] = (
                    papers_collected / collection_result.get('pubmed_papers', 1)
                    if collection_result.get('pubmed_papers', 0) > 0 else 0
                )
            else:
                progress.collection_stats['pubmed_papers'] += papers_collected

            progress.collection_stats['total_searched'] += collection_result.get('total_papers_searched', 0)

            session.global_stats['total_papers_collected'] += papers_collected

            logger.info(f"Collected {papers_collected} papers for '{condition}' "
                       f"({progress.papers_collected}/{progress.target_papers})")

            # Reset failure count on success
            progress.consecutive_failures = 0
            progress.last_error = None

            self.session_manager.update_condition_progress(condition, progress)

            # Save intermediate results if configured
            if self.config.save_intermediate and progress.papers_collected % self.config.checkpoint_interval == 0:
                self._save_intermediate_results(condition, progress)

        except Exception as e:
            logger.error(f"Collection failed for {condition}: {e}")
            progress.consecutive_failures += 1
            progress.last_error = str(e)
            session.global_stats['network_failures'] += 1
            raise

    def _save_intermediate_results(self, condition: str, progress: ConditionProgress):
        """Save intermediate results for a condition."""
        try:
            # Get current database stats for this condition
            stats = self._get_condition_stats(condition)

            result_data = {
                'condition': condition,
                'progress': asdict(progress),
                'database_stats': stats,
                'timestamp': datetime.now().isoformat()
            }

            # Save to output directory
            filename = f"{condition.replace(' ', '_')}_progress.json"
            output_file = self.config.output_dir / filename

            with open(output_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)

            logger.debug(f"💾 Saved intermediate results for {condition}")

        except Exception as e:
            logger.error(f"Failed to save intermediate results for {condition}: {e}")

    def _get_condition_stats(self, condition: str) -> Dict[str, Any]:
        """Get database statistics for a specific condition."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Count papers with interventions for this condition
                cursor.execute("""
                    SELECT COUNT(DISTINCT p.id)
                    FROM papers p
                    JOIN interventions i ON p.id = i.paper_id
                    WHERE LOWER(i.condition) LIKE LOWER(?)
                """, (f"%{condition}%",))
                papers_with_interventions = cursor.fetchone()[0]

                # Count total interventions for this condition
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM interventions
                    WHERE LOWER(condition) LIKE LOWER(?)
                """, (f"%{condition}%",))
                total_interventions = cursor.fetchone()[0]

                # Count papers with S2 enrichment
                cursor.execute("""
                    SELECT COUNT(DISTINCT p.id)
                    FROM papers p
                    JOIN interventions i ON p.id = i.paper_id
                    WHERE LOWER(i.condition) LIKE LOWER(?) AND p.s2_paper_id IS NOT NULL
                """, (f"%{condition}%",))
                s2_enriched_papers = cursor.fetchone()[0]

                return {
                    'papers_with_interventions': papers_with_interventions,
                    'total_interventions': total_interventions,
                    's2_enriched_papers': s2_enriched_papers
                }

        except Exception as e:
            logger.error(f"Failed to get condition stats for {condition}: {e}")
            return {}

    def _generate_final_report(self, session: CollectionSession) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        logger.info("Generating final collection report...")

        # Calculate overall statistics
        total_duration = datetime.now() - self.session_start
        completed_conditions = [p for p in session.conditions_progress.values() if p.status == 'completed']
        failed_conditions = [p for p in session.conditions_progress.values() if p.status == 'failed']

        # Network failure stats
        network_stats = self.network_manager.get_failure_stats()

        report = {
            'session_info': {
                'session_id': session.session_id,
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': str(total_duration),
                'total_duration_hours': session.total_runtime_hours,
                'restart_count': session.restart_count
            },
            'configuration': asdict(self.config),
            'global_statistics': session.global_stats,
            'network_statistics': network_stats,
            'condition_summary': {
                'total_conditions': len(session.conditions_progress),
                'completed_conditions': len(completed_conditions),
                'failed_conditions': len(failed_conditions),
                'success_rate': len(completed_conditions) / len(session.conditions_progress) * 100 if session.conditions_progress else 0
            },
            'condition_details': {},
            'database_summary': self._get_final_database_stats()
        }

        # Add detailed condition information
        for condition, progress in session.conditions_progress.items():
            report['condition_details'][condition] = {
                'status': progress.status,
                'papers_collected': progress.papers_collected,
                'target_papers': progress.target_papers,
                'progress_percent': progress.progress_percent,
                'duration': str(progress.duration) if progress.duration else None,
                'collection_stats': progress.collection_stats,
                'failures': progress.consecutive_failures,
                'last_error': progress.last_error
            }

        # Save final report
        report_file = self.config.output_dir / f"collection_report_{session.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate summary in other formats if requested
        if 'csv' in self.config.export_formats:
            self._export_csv_summary(report, session.session_id)

        logger.info(f"Final report saved to {report_file}")
        logger.info(f"Campaign completed: {len(completed_conditions)}/{len(session.conditions_progress)} conditions successful")
        logger.info(f"Total papers collected: {session.global_stats['total_papers_collected']}")
        logger.info(f"Total duration: {total_duration}")

        return report

    def _get_final_database_stats(self) -> Dict[str, Any]:
        """Get final database statistics."""
        try:
            stats = database_manager.get_database_stats()

            # Add S2 specific stats
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM papers WHERE s2_processed = 1')
                s2_processed = cursor.fetchone()[0]
                cursor.execute('SELECT COUNT(*) FROM papers WHERE s2_paper_id IS NOT NULL')
                s2_enriched = cursor.fetchone()[0]

                stats['s2_processed'] = s2_processed
                stats['s2_enriched'] = s2_enriched

            return stats

        except Exception as e:
            logger.error(f"Failed to get final database stats: {e}")
            return {}

    def _export_csv_summary(self, report: Dict[str, Any], session_id: str):
        """Export summary to CSV format."""
        try:
            import csv

            csv_file = self.config.output_dir / f"collection_summary_{session_id}.csv"

            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Condition', 'Status', 'Papers Collected', 'Target Papers', 'Progress %', 'Duration'])

                for condition, details in report['condition_details'].items():
                    writer.writerow([
                        condition,
                        details['status'],
                        details['papers_collected'],
                        details['target_papers'],
                        f"{details['progress_percent']:.1f}%",
                        details['duration'] or 'N/A'
                    ])

            logger.info(f"CSV summary exported to {csv_file}")

        except Exception as e:
            logger.error(f"Failed to export CSV summary: {e}")

    def show_status(self) -> Dict[str, Any]:
        """Show current collection status."""
        session = self.session_manager.load_existing_session()
        if not session:
            return {"error": "No active session found"}

        status = {
            'session_id': session.session_id,
            'start_time': session.start_time,
            'is_active': session.is_active,
            'global_stats': session.global_stats,
            'conditions': {}
        }

        for condition, progress in session.conditions_progress.items():
            status['conditions'][condition] = {
                'status': progress.status,
                'progress': f"{progress.papers_collected}/{progress.target_papers}",
                'progress_percent': f"{progress.progress_percent:.1f}%",
                'failures': progress.consecutive_failures,
                'last_error': progress.last_error
            }

        return status


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Paper Collector for Medical Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Basic collection options
    parser.add_argument('search_term', nargs='?',
                       help='Single search term/condition (alternative to --conditions)')
    parser.add_argument('--conditions', type=str,
                       help='Comma-separated list of conditions to collect')
    parser.add_argument('--max-papers', type=int, default=100,
                       help='Papers per condition (default: 100)')
    parser.add_argument('--target-per-condition', type=int,
                       help='Papers per condition (alias for --max-papers)')

    # Date filtering
    parser.add_argument('--min-year', type=int, default=2015,
                       help='Minimum publication year (default: 2015)')
    parser.add_argument('--max-year', type=int,
                       help='Maximum publication year (default: current year)')

    # Collection modes
    parser.add_argument('--traditional-mode', action='store_true',
                       help='Use traditional mode without interleaved S2')
    parser.add_argument('--skip-s2', action='store_true',
                       help='Skip Semantic Scholar enrichment')
    parser.add_argument('--no-fulltext', action='store_true',
                       help='Skip fulltext collection')

    # Batch processing
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Papers collected per API call (default: 50)')
    parser.add_argument('--collection-interval', type=int, default=5,
                       help='Seconds between batches (default: 5)')

    # Session management
    parser.add_argument('--resume', action='store_true',
                       help='Resume previous session')
    parser.add_argument('--session-file', type=str, default='collection_session.json',
                       help='Session state file')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save progress every N papers (default: 10)')

    # Output options
    parser.add_argument('--output-dir', type=str, default='collection_results',
                       help='Output directory for results')
    parser.add_argument('--export-formats', type=str, default='json',
                       help='Export formats (comma-separated): json,csv')
    parser.add_argument('--no-intermediate', action='store_true',
                       help='Skip saving intermediate results')

    # Operational modes
    parser.add_argument('--overnight', action='store_true',
                       help='Enable overnight operation mode')
    parser.add_argument('--no-auto-restart', action='store_true',
                       help='Disable auto-restart on errors')

    # Utility options
    parser.add_argument('--status', action='store_true',
                       help='Show current collection status')
    parser.add_argument('--config', type=str,
                       help='Load configuration from JSON file')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            config = CollectionConfig(**config_data)
        else:
            # Build configuration from arguments
            conditions = []
            if args.conditions:
                conditions = [c.strip() for c in args.conditions.split(',')]
            elif args.search_term:
                conditions = [args.search_term]
            else:
                print("Error: Must specify either search_term or --conditions")
                return False

            target_papers = args.target_per_condition or args.max_papers

            config = CollectionConfig(
                conditions=conditions,
                target_papers_per_condition=target_papers,
                min_year=args.min_year,
                max_year=args.max_year,
                use_interleaved_s2=not args.skip_s2,
                include_fulltext=not args.no_fulltext,
                traditional_mode=args.traditional_mode,
                batch_size_collection=args.batch_size,
                collection_interval=args.collection_interval,
                session_file=Path(args.session_file),
                checkpoint_interval=args.checkpoint_interval,
                output_dir=Path(args.output_dir),
                export_formats=args.export_formats.split(','),
                save_intermediate=not args.no_intermediate,
                overnight_mode=args.overnight,
                auto_restart_on_error=not args.no_auto_restart
            )

        # Initialize collector
        collector = PaperCollector(config)

        # Handle different run modes
        if args.status:
            # Show status and exit
            status = collector.show_status()
            if 'error' in status:
                print(status['error'])
            else:
                print("Current collection status:")
                print(f"Session: {status['session_id']}")
                print(f"Active: {status['is_active']}")
                print(f"Total papers collected: {status['global_stats']['total_papers_collected']}")
                print("\nCondition progress:")
                for condition, progress in status['conditions'].items():
                    print(f"  {condition}: {progress['status']} - {progress['progress']} ({progress['progress_percent']})")
            return True

        # Run collection
        logger.info("Starting unified paper collection")
        report = collector.run_collection_campaign(resume=args.resume)

        # Print summary
        print(f"\n=== Collection Campaign Complete ===")
        print(f"Session: {report['session_info']['session_id']}")
        print(f"Duration: {report['session_info']['total_duration']}")
        print(f"Conditions processed: {report['condition_summary']['completed_conditions']}/{report['condition_summary']['total_conditions']}")
        print(f"Total papers collected: {report['global_statistics']['total_papers_collected']}")
        print(f"Success rate: {report['condition_summary']['success_rate']:.1f}%")

        return True

    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        print("\nCollection interrupted. Session state saved.")
        return True
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        logger.error(traceback.format_exc())
        print(f"Collection failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)