#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust LLM Paper Processing Script with Enhanced Features

This script provides an enhanced version of the paper processing pipeline with:
- Frequent progress saving to handle interruptions gracefully
- Thermal monitoring and automatic pausing to prevent overheating
- GPU memory monitoring and optimization
- Recovery from previous interrupted sessions
- Detailed logging and progress tracking
- Configurable safety thresholds

Features:
- Progress checkpoints every few papers
- GPU temperature monitoring with automatic cooling pauses
- Memory usage optimization
- Session recovery on restart
- Comprehensive error handling
- Real-time thermal protection

Usage:
    python robust_llm_processor.py [options]

Examples:
    # Standard processing with thermal protection
    python robust_llm_processor.py --limit 50

    # Process with custom temperature threshold
    python robust_llm_processor.py --max-temp 75 --batch-size 3

    # Resume from previous session
    python robust_llm_processor.py --resume

    # Check thermal status only
    python robust_llm_processor.py --thermal-status
"""

import sys
import json
import time
import signal
import argparse
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Set UTF-8 encoding for stdout/stderr on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import setup_logging
from src.paper_collection.database_manager import database_manager
from src.llm.pipeline import InterventionResearchPipeline
from src.data.utils import format_duration

logger = setup_logging(__name__, 'robust_llm_processor.log')


@dataclass
class ProcessingSession:
    """Represents a processing session with progress tracking."""
    session_id: str
    start_time: float
    total_papers: int
    processed_papers: int
    failed_papers: List[str]
    current_batch: int
    batch_size: int
    last_checkpoint: float
    interventions_extracted: int
    session_config: Dict[str, Any]
    thermal_events: List[Dict[str, Any]]


@dataclass
class ThermalStatus:
    """Current thermal and system status."""
    gpu_temp: float
    gpu_power: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    timestamp: float
    is_safe: bool
    cooling_needed: bool


class ThermalMonitor:
    """Monitor GPU thermal status and manage cooling."""

    def __init__(self, max_temp: float = 80.0, cooling_temp: float = 70.0,
                 max_power: float = 250.0, check_interval: float = 5.0):
        """
        Initialize thermal monitor.

        Args:
            max_temp: Maximum safe GPU temperature (°C)
            cooling_temp: Temperature to resume after cooling (°C)
            max_power: Maximum safe power draw (W)
            check_interval: Monitoring interval (seconds)
        """
        self.max_temp = max_temp
        self.cooling_temp = cooling_temp
        self.max_power = max_power
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        self.thermal_events = []
        self.current_status = None

        logger.info(f"Thermal monitor initialized - Max temp: {max_temp}°C, Cooling: {cooling_temp}°C")

    def get_gpu_status(self) -> Optional[ThermalStatus]:
        """Get current GPU thermal and performance status."""
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                temp = float(values[0])
                power = float(values[1]) if values[1] != '[Not Supported]' else 0.0
                mem_used = float(values[2])
                mem_total = float(values[3])
                utilization = float(values[4])

                is_safe = temp < self.max_temp and (power == 0.0 or power < self.max_power)
                cooling_needed = temp > self.cooling_temp

                status = ThermalStatus(
                    gpu_temp=temp,
                    gpu_power=power,
                    gpu_memory_used=mem_used,
                    gpu_memory_total=mem_total,
                    gpu_utilization=utilization,
                    timestamp=time.time(),
                    is_safe=is_safe,
                    cooling_needed=cooling_needed
                )

                self.current_status = status
                return status

        except Exception as e:
            logger.warning(f"Error getting GPU status: {e}")

        return None

    def start_monitoring(self):
        """Start continuous thermal monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Thermal monitoring started")

    def stop_monitoring(self):
        """Stop thermal monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        logger.info("Thermal monitoring stopped")

    def _monitor_loop(self):
        """Continuous monitoring loop."""
        while self.is_monitoring:
            status = self.get_gpu_status()
            if status and not status.is_safe:
                event = {
                    'timestamp': status.timestamp,
                    'event_type': 'thermal_warning',
                    'temperature': status.gpu_temp,
                    'power': status.gpu_power,
                    'max_temp': self.max_temp,
                    'max_power': self.max_power
                }
                self.thermal_events.append(event)
                logger.warning(f"Thermal warning: {status.gpu_temp}°C (max: {self.max_temp}°C)")

            time.sleep(self.check_interval)

    def wait_for_safe_temperature(self) -> bool:
        """Wait for GPU to cool down to safe temperature."""
        logger.info(f"Waiting for GPU to cool below {self.cooling_temp}°C...")
        start_time = time.time()
        max_wait_time = 300  # 5 minutes maximum wait

        while time.time() - start_time < max_wait_time:
            status = self.get_gpu_status()
            if not status:
                logger.error("Cannot read GPU status during cooling wait")
                return False

            if status.gpu_temp <= self.cooling_temp:
                wait_time = time.time() - start_time
                logger.info(f"GPU cooled to {status.gpu_temp}°C after {wait_time:.1f}s")
                return True

            logger.info(f"Current temperature: {status.gpu_temp}°C, waiting...")
            time.sleep(10)  # Check every 10 seconds during cooling

        logger.error(f"GPU did not cool down within {max_wait_time}s")
        return False

    def is_safe_to_process(self) -> bool:
        """Check if it's safe to continue processing."""
        status = self.get_gpu_status()
        return status is not None and status.is_safe


class RobustLLMProcessor:
    """
    Enhanced LLM processor with progress saving, thermal monitoring, and recovery.
    """

    def __init__(self, session_file: str = "processing_session.json",
                 checkpoint_interval: int = 1, max_temp: float = 80.0):
        """
        Initialize the robust processor.

        Args:
            session_file: File to store session progress
            checkpoint_interval: Save progress every N papers (default: 1 = after each paper)
            max_temp: Maximum safe GPU temperature
        """
        self.session_file = Path(session_file)
        self.checkpoint_interval = checkpoint_interval
        self.pipeline = InterventionResearchPipeline()
        self.thermal_monitor = ThermalMonitor(max_temp=max_temp)
        self.current_session = None
        self.shutdown_requested = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Robust LLM processor initialized - Checkpoint interval: {checkpoint_interval} papers")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.save_session()

    def create_session(self, total_papers: int, batch_size: int, config: Dict[str, Any]) -> ProcessingSession:
        """Create a new processing session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        session = ProcessingSession(
            session_id=session_id,
            start_time=time.time(),
            total_papers=total_papers,
            processed_papers=0,
            failed_papers=[],
            current_batch=0,
            batch_size=batch_size,
            last_checkpoint=time.time(),
            interventions_extracted=0,
            session_config=config,
            thermal_events=[]
        )

        self.current_session = session
        self.save_session()
        logger.info(f"Created new session {session_id} for {total_papers} papers")
        return session

    def load_session(self) -> Optional[ProcessingSession]:
        """Load a previous session from file."""
        if not self.session_file.exists():
            return None

        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            session = ProcessingSession(**data)
            self.current_session = session
            logger.info(f"Loaded session {session.session_id} - {session.processed_papers}/{session.total_papers} papers processed")
            return session

        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return None

    def save_session(self):
        """Save current session to file."""
        if not self.current_session:
            return

        try:
            # Update thermal events
            self.current_session.thermal_events = self.thermal_monitor.thermal_events

            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.current_session), f, indent=2)

            logger.debug(f"Session saved - {self.current_session.processed_papers}/{self.current_session.total_papers} papers")

        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def clear_session(self):
        """Clear the current session file."""
        if self.session_file.exists():
            self.session_file.unlink()
        self.current_session = None
        logger.info("Session cleared")

    def get_unprocessed_papers(self, limit: Optional[int] = None) -> List[Dict]:
        """Get papers that need processing."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                query = """
                    SELECT pmid, title, abstract, journal, publication_date, doi, processing_status
                    FROM papers
                    WHERE (processing_status IS NULL OR processing_status = 'pending' OR processing_status = 'failed')
                    AND abstract IS NOT NULL AND abstract != ''
                    ORDER BY created_at ASC
                """

                if limit:
                    query += f" LIMIT {limit}"

                cursor.execute(query)
                rows = cursor.fetchall()

                papers = []
                for row in rows:
                    papers.append({
                        'pmid': row[0],
                        'title': row[1],
                        'abstract': row[2],
                        'journal': row[3],
                        'publication_date': row[4],
                        'doi': row[5],
                        'processing_status': row[6]
                    })

                return papers

        except Exception as e:
            logger.error(f"Error getting unprocessed papers: {e}")
            return []

    def process_papers_with_thermal_protection(self, papers: List[Dict],
                                             batch_size: int = 5,
                                             resume: bool = False) -> Dict[str, Any]:
        """
        Process papers with thermal protection and progress saving.

        Args:
            papers: List of papers to process
            batch_size: Papers per batch
            resume: Whether resuming from previous session

        Returns:
            Processing results
        """
        start_time = time.time()

        # Create or resume session
        if resume:
            session = self.load_session()
            if not session:
                logger.info("No previous session found, starting new session")
                session = self.create_session(len(papers), batch_size, {
                    'max_temp': self.thermal_monitor.max_temp,
                    'checkpoint_interval': self.checkpoint_interval
                })
        else:
            session = self.create_session(len(papers), batch_size, {
                'max_temp': self.thermal_monitor.max_temp,
                'checkpoint_interval': self.checkpoint_interval
            })

        # Start thermal monitoring
        self.thermal_monitor.start_monitoring()

        try:
            # Skip already processed papers if resuming
            start_index = session.processed_papers if resume else 0
            papers_to_process = papers[start_index:]

            if not papers_to_process:
                logger.info("All papers already processed")
                return self._compile_results(session, time.time() - start_time)

            logger.info(f"Processing {len(papers_to_process)} papers (starting from index {start_index})")

            total_interventions = session.interventions_extracted
            failed_papers = list(session.failed_papers)

            # Process papers in batches
            for i in range(0, len(papers_to_process), batch_size):
                if self.shutdown_requested:
                    logger.info("Shutdown requested, saving progress...")
                    break

                # Check thermal status before processing batch
                if not self.thermal_monitor.is_safe_to_process():
                    logger.warning("GPU temperature too high, waiting for cooling...")
                    if not self.thermal_monitor.wait_for_safe_temperature():
                        logger.error("Could not achieve safe temperature, stopping processing")
                        break

                batch = papers_to_process[i:i + batch_size]
                batch_num = session.current_batch + 1
                total_batches = (len(papers_to_process) + batch_size - 1) // batch_size

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} papers)")

                # Process batch
                try:
                    batch_results = self._process_single_batch(batch)

                    # Update session progress
                    session.processed_papers += len(batch)
                    session.current_batch = batch_num
                    session.interventions_extracted += batch_results.get('interventions_extracted', 0)
                    failed_papers.extend(batch_results.get('failed_papers', []))

                    # Save checkpoint if needed
                    if (session.processed_papers % self.checkpoint_interval == 0 or
                        batch_num == total_batches):
                        session.last_checkpoint = time.time()
                        self.save_session()
                        logger.info(f"Checkpoint saved - {session.processed_papers}/{session.total_papers} papers completed")

                    # Check thermal status after batch
                    status = self.thermal_monitor.get_gpu_status()
                    if status:
                        logger.info(f"GPU status: {status.gpu_temp}°C, {status.gpu_utilization}% util, "
                                  f"{status.gpu_memory_used/1024:.1f}GB/{status.gpu_memory_total/1024:.1f}GB")

                        if status.cooling_needed:
                            logger.info("GPU running warm, adding cooling pause...")
                            time.sleep(5)

                    # Inter-batch delay for thermal management
                    if i + batch_size < len(papers_to_process):
                        time.sleep(2)

                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    # Mark all papers in batch as failed
                    for paper in batch:
                        failed_papers.append(paper['pmid'])
                    session.processed_papers += len(batch)

                    self.save_session()

            return self._compile_results(session, time.time() - start_time)

        finally:
            self.thermal_monitor.stop_monitoring()
            # Final session save
            session.thermal_events = self.thermal_monitor.thermal_events
            self.save_session()

    def _process_single_batch(self, batch: List[Dict]) -> Dict[str, Any]:
        """Process a single batch of papers."""
        try:
            results = self.pipeline.analyze_interventions(
                limit_papers=len(batch),
                batch_size=len(batch)
            )

            return {
                'papers_processed': results.get('papers_processed', 0),
                'interventions_extracted': results.get('interventions_extracted', 0),
                'failed_papers': []  # Pipeline handles individual failures
            }

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {
                'papers_processed': 0,
                'interventions_extracted': 0,
                'failed_papers': [paper['pmid'] for paper in batch]
            }

    def _compile_results(self, session: ProcessingSession, duration: float) -> Dict[str, Any]:
        """Compile final processing results."""
        success_rate = (session.processed_papers - len(session.failed_papers)) / max(session.processed_papers, 1) * 100

        return {
            'session_id': session.session_id,
            'total_papers': session.total_papers,
            'processed_papers': session.processed_papers,
            'successful_papers': session.processed_papers - len(session.failed_papers),
            'failed_papers': session.failed_papers,
            'interventions_extracted': session.interventions_extracted,
            'success_rate': success_rate,
            'duration': duration,
            'formatted_duration': format_duration(duration),
            'thermal_events': session.thermal_events,
            'batch_size_used': session.batch_size,
            'checkpoints_saved': session.processed_papers // self.checkpoint_interval,
            'shutdown_requested': self.shutdown_requested
        }

    def print_session_status(self):
        """Print current session status."""
        session = self.load_session()
        if not session:
            print("No active session found.")
            return

        print(f"\n{'='*60}")
        print("PROCESSING SESSION STATUS")
        print(f"{'='*60}")
        print(f"Session ID: {session.session_id}")
        print(f"Started: {datetime.fromtimestamp(session.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Progress: {session.processed_papers}/{session.total_papers} papers ({session.processed_papers/session.total_papers*100:.1f}%)")
        print(f"Interventions: {session.interventions_extracted}")
        print(f"Failed: {len(session.failed_papers)}")
        print(f"Current Batch: {session.current_batch}")
        print(f"Last Checkpoint: {datetime.fromtimestamp(session.last_checkpoint).strftime('%H:%M:%S')}")

        if session.thermal_events:
            print(f"Thermal Events: {len(session.thermal_events)}")
            for event in session.thermal_events[-3:]:  # Show last 3 events
                time_str = datetime.fromtimestamp(event['timestamp']).strftime('%H:%M:%S')
                print(f"  {time_str}: {event['event_type']} - {event['temperature']:.1f}°C")

        print(f"{'='*60}")

    def print_thermal_status(self):
        """Print current thermal status."""
        status = self.thermal_monitor.get_gpu_status()
        if not status:
            print("Could not read GPU thermal status")
            return

        print(f"\n{'='*60}")
        print("GPU THERMAL STATUS")
        print(f"{'='*60}")
        print(f"Temperature: {status.gpu_temp}°C")
        print(f"Power Draw: {status.gpu_power}W")
        print(f"Memory: {status.gpu_memory_used/1024:.1f}GB / {status.gpu_memory_total/1024:.1f}GB ({status.gpu_memory_used/status.gpu_memory_total*100:.1f}%)")
        print(f"Utilization: {status.gpu_utilization}%")
        print(f"Status: {'SAFE' if status.is_safe else 'WARNING'}")
        print(f"Cooling Needed: {'Yes' if status.cooling_needed else 'No'}")
        print(f"Max Temp Threshold: {self.thermal_monitor.max_temp}°C")
        print(f"Cooling Threshold: {self.thermal_monitor.cooling_temp}°C")
        print(f"{'='*60}")


def main():
    """Main entry point for the robust LLM processing script."""
    parser = argparse.ArgumentParser(
        description="Robust LLM Paper Processing with Thermal Protection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --limit 50                    # Process 50 papers with thermal protection
  %(prog)s --resume                      # Resume previous session
  %(prog)s --max-temp 75 --batch-size 3  # Custom temperature limit and batch size
  %(prog)s --thermal-status              # Check current thermal status
  %(prog)s --session-status              # Check session progress
        """
    )

    parser.add_argument('--limit', type=int, help='Maximum number of papers to process')
    parser.add_argument('--batch-size', type=int, default=5, help='Papers per batch (default: 5)')
    parser.add_argument('--max-temp', type=float, default=80.0, help='Maximum GPU temperature (°C, default: 80)')
    parser.add_argument('--cooling-temp', type=float, default=70.0, help='Resume temperature after cooling (°C, default: 70)')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='Save progress every N papers (default: 1)')
    parser.add_argument('--session-file', default='processing_session.json', help='Session file path')
    parser.add_argument('--resume', action='store_true', help='Resume previous session')
    parser.add_argument('--session-status', action='store_true', help='Show session status and exit')
    parser.add_argument('--thermal-status', action='store_true', help='Show thermal status and exit')
    parser.add_argument('--clear-session', action='store_true', help='Clear session and exit')

    args = parser.parse_args()

    print("MyBiome Robust LLM Processing")
    print("=" * 50)

    try:
        processor = RobustLLMProcessor(
            session_file=args.session_file,
            checkpoint_interval=args.checkpoint_interval,
            max_temp=args.max_temp
        )

        # Handle status checks
        if args.thermal_status:
            processor.print_thermal_status()
            return 0

        if args.session_status:
            processor.print_session_status()
            return 0

        if args.clear_session:
            processor.clear_session()
            print("Session cleared.")
            return 0

        # Check thermal safety before starting
        if not processor.thermal_monitor.is_safe_to_process():
            print(f"\nWarning: GPU temperature too high to start processing safely")
            processor.print_thermal_status()
            return 1

        # Get papers to process
        print("\nGetting papers for processing...")
        papers = processor.get_unprocessed_papers(args.limit)

        if not papers:
            print("No papers need processing.")
            return 0

        print(f"Found {len(papers)} papers to process")
        print(f"Thermal protection: Max {args.max_temp}°C, Cool to {args.cooling_temp}°C")
        print(f"Progress checkpoints every {args.checkpoint_interval} paper{'s' if args.checkpoint_interval != 1 else ''}")

        if args.resume:
            print("Attempting to resume previous session...")

        # Start processing
        results = processor.process_papers_with_thermal_protection(
            papers=papers,
            batch_size=args.batch_size,
            resume=args.resume
        )

        # Print results
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"Session: {results['session_id']}")
        print(f"Papers Processed: {results['successful_papers']}/{results['total_papers']}")
        print(f"Interventions Extracted: {results['interventions_extracted']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Duration: {results['formatted_duration']}")
        print(f"Checkpoints Saved: {results['checkpoints_saved']}")

        if results['thermal_events']:
            print(f"Thermal Events: {len(results['thermal_events'])}")

        if results['failed_papers']:
            print(f"Failed Papers: {len(results['failed_papers'])}")

        if results['shutdown_requested']:
            print("Processing stopped due to shutdown request")

        print(f"{'='*60}")

        # Clean up session if completed successfully
        if not results['shutdown_requested'] and results['processed_papers'] == results['total_papers']:
            processor.clear_session()
            print("Session completed and cleared.")

        return 0 if results['success_rate'] > 0 else 1

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.error(f"Unexpected error in main: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())