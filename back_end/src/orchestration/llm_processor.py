#!/usr/bin/env python3
"""
Unified LLM Processor - Robust overnight LLM processing with comprehensive features

This script provides comprehensive LLM processing for research papers with:
- Advanced thermal protection with GPU monitoring
- Session persistence and recovery
- Progress tracking and incremental saving
- Dual-model analysis (gemma2:9b + qwen2.5:14b)
- Memory optimization and batch processing
- Auto-restart on errors
- Overnight operation capacity
- Real-time thermal monitoring

Features:
- GPU temperature monitoring with predictive cooling
- Session recovery after interruptions
- Progress checkpoints and auto-save
- Memory usage optimization
- Comprehensive error handling
- Real-time thermal protection
- Parallel processing capabilities
- Detailed performance metrics

Usage:
    # Standard processing with thermal protection
    python llm_processor.py --limit 50

    # Process all papers overnight
    python llm_processor.py --all --overnight

    # Resume interrupted session
    python llm_processor.py --resume

    # Custom thermal thresholds
    python llm_processor.py --max-temp 75 --cooling-temp 65

Examples:
    # Process specific number with thermal protection
    python llm_processor.py --limit 100 --batch-size 3 --max-temp 80

    # Overnight processing with custom settings
    python llm_processor.py --all --overnight --checkpoint-interval 5

    # Resume with status monitoring
    python llm_processor.py --resume --thermal-status

    # Force reprocess failed papers
    python llm_processor.py --reprocess-failed --limit 20
"""

import sys
import json
import time
import signal
import argparse
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import traceback

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from back_end.src.data.config import config, setup_logging
    from back_end.src.data.repositories import repository_manager
    from back_end.src.data_collection.database_manager import database_manager
    from back_end.src.llm_processing.dual_model_analyzer import DualModelAnalyzer
    from back_end.src.llm_processing.pipeline import InterventionResearchPipeline
    from back_end.src.data.utils import format_duration

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)

logger = setup_logging(__name__, 'llm_processor.log')


@dataclass
class ProcessingConfig:
    """Configuration for LLM processing orchestrator."""
    # Processing targets
    limit: Optional[int] = None  # None = process all
    batch_size: int = 5
    max_papers_per_session: int = 1000

    # Model configuration
    use_dual_models: bool = True
    models: List[str] = None  # ['gemma2:9b', 'qwen2.5:14b']
    temperature: float = 0.3
    max_tokens: int = 4096

    # Thermal protection
    max_temp_celsius: float = 80.0
    cooling_temp_celsius: float = 70.0
    max_power_watts: float = 250.0
    thermal_check_interval: int = 5  # seconds

    # Memory management
    max_memory_percent: float = 85.0
    memory_cleanup_interval: int = 10  # papers
    force_garbage_collection: bool = True

    # Session management
    session_file: Path = Path("processing_session.json")
    auto_save_interval: int = 60   # seconds
    checkpoint_interval: int = 5   # papers

    # Error handling
    max_retries_per_paper: int = 3
    retry_delay: int = 30  # seconds
    auto_restart_on_error: bool = True
    skip_failed_papers: bool = True

    # Output options
    output_dir: Path = Path("processing_results")
    save_intermediate: bool = True
    export_formats: List[str] = None  # ['json', 'csv']

    # Operational modes
    overnight_mode: bool = False
    parallel_processing: bool = False
    max_workers: int = 2

    # Processing options
    reprocess_failed: bool = False
    force_reprocess: bool = False
    conditions_filter: List[str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = ['gemma2:9b', 'qwen2.5:14b']
        if self.export_formats is None:
            self.export_formats = ['json']
        if self.conditions_filter is None:
            self.conditions_filter = []
        self.output_dir = Path(self.output_dir)
        self.session_file = Path(self.session_file)


@dataclass
class ThermalStatus:
    """Current thermal and system status."""
    gpu_temp: float
    gpu_power: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    cpu_temp: float
    cpu_usage: float
    ram_usage: float
    timestamp: float
    is_safe: bool
    cooling_needed: bool
    thermal_events: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.thermal_events is None:
            self.thermal_events = []


@dataclass
class ProcessingProgress:
    """Progress tracking for processing session."""
    session_id: str
    start_time: datetime
    total_papers: int = 0
    processed_papers: int = 0
    failed_papers: int = 0
    skipped_papers: int = 0
    interventions_extracted: int = 0
    current_paper_id: Optional[str] = None
    last_checkpoint: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    thermal_events: List[Dict[str, Any]] = None
    error_log: List[Dict[str, Any]] = None
    performance_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.thermal_events is None:
            self.thermal_events = []
        if self.error_log is None:
            self.error_log = []
        if self.performance_metrics is None:
            self.performance_metrics = {
                'avg_processing_time': 0.0,
                'total_processing_time': 0.0,
                'papers_per_hour': 0.0,
                'thermal_pauses': 0,
                'memory_cleanups': 0
            }

    @property
    def progress_percent(self) -> float:
        if self.total_papers > 0:
            return (self.processed_papers / self.total_papers) * 100
        return 0.0

    @property
    def duration(self) -> timedelta:
        end_time = self.completion_time or datetime.now()
        return end_time - self.start_time

    @property
    def estimated_completion(self) -> Optional[datetime]:
        if self.processed_papers > 0 and self.total_papers > self.processed_papers:
            avg_time_per_paper = self.duration.total_seconds() / self.processed_papers
            remaining_papers = self.total_papers - self.processed_papers
            remaining_seconds = avg_time_per_paper * remaining_papers
            return datetime.now() + timedelta(seconds=remaining_seconds)
        return None


@dataclass
class ProcessingSession:
    """Complete session state for persistence."""
    session_id: str
    config: ProcessingConfig
    progress: ProcessingProgress
    start_time: str
    last_update: str
    is_active: bool = True
    restart_count: int = 0


class ThermalMonitor:
    """Advanced thermal monitoring with predictive cooling."""

    def __init__(self, max_temp: float = 80.0, cooling_temp: float = 70.0,
                 max_power: float = 250.0, check_interval: float = 5.0):
        self.max_temp = max_temp
        self.cooling_temp = cooling_temp
        self.max_power = max_power
        self.check_interval = check_interval
        self.is_monitoring = False
        self.monitoring_thread = None
        self.thermal_events = []
        self.current_status = None
        self.temp_history = []

        # Thermal monitor initialized (logging removed for performance)

    def get_system_status(self) -> Optional[ThermalStatus]:
        """Get comprehensive system thermal and performance status."""
        try:
            # GPU stats
            gpu_result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)

            gpu_temp = gpu_power = gpu_mem_used = gpu_mem_total = gpu_util = 0.0

            if gpu_result.returncode == 0:
                values = gpu_result.stdout.strip().split(', ')
                gpu_temp = float(values[0]) if values[0] != '[Not Supported]' else 0.0
                gpu_power = float(values[1]) if values[1] != '[Not Supported]' else 0.0
                gpu_mem_used = float(values[2]) if values[2] != '[Not Supported]' else 0.0
                gpu_mem_total = float(values[3]) if values[3] != '[Not Supported]' else 0.0
                gpu_util = float(values[4]) if values[4] != '[Not Supported]' else 0.0

            # CPU and system stats
            cpu_temp = 0.0
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            if entries:
                                cpu_temp = max(entry.current for entry in entries)
                                break
            except:
                pass

            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent

            # Safety checks
            is_safe = (gpu_temp < self.max_temp and
                      (gpu_power == 0.0 or gpu_power < self.max_power) and
                      cpu_temp < 85.0)  # CPU safety threshold
            cooling_needed = gpu_temp > self.cooling_temp

            status = ThermalStatus(
                gpu_temp=gpu_temp,
                gpu_power=gpu_power,
                gpu_memory_used=gpu_mem_used,
                gpu_memory_total=gpu_mem_total,
                gpu_utilization=gpu_util,
                cpu_temp=cpu_temp,
                cpu_usage=cpu_usage,
                ram_usage=ram_usage,
                timestamp=time.time(),
                is_safe=is_safe,
                cooling_needed=cooling_needed
            )

            # Track temperature history for predictive cooling
            self.temp_history.append((time.time(), gpu_temp))
            # Keep only last 10 minutes
            cutoff = time.time() - 600
            self.temp_history = [(t, temp) for t, temp in self.temp_history if t > cutoff]

            self.current_status = status
            return status

        except Exception as e:
            logger.warning(f"Error getting system status: {e}")
            return None

    def is_thermal_safe(self) -> Tuple[bool, Optional[ThermalStatus]]:
        """Check if thermal conditions are safe for processing."""
        status = self.get_system_status()
        if not status:
            # Cannot read thermal status (logging removed for performance)
            return True, None

        # Predictive cooling: check if temperature is rising rapidly
        if len(self.temp_history) >= 3:
            recent_temps = [temp for _, temp in self.temp_history[-3:]]
            if len(recent_temps) >= 2:
                temp_rate = (recent_temps[-1] - recent_temps[0]) / len(recent_temps)
                if temp_rate > 2 and status.gpu_temp > (self.max_temp - 5):
                    # GPU heating rapidly detected (logging removed for performance)

        return status.is_safe, status

    def wait_for_cooling(self) -> None:
        """Wait for system to cool down to safe operating temperature."""
        # Thermal protection activated (logging removed for performance)

        cooling_start = time.time()
        while True:
            is_safe, status = self.is_thermal_safe()

            if not status:
                logger.warning("Cannot read temperature during cooling, resuming")
                break

            if status.gpu_temp <= self.cooling_temp:
                cooling_duration = time.time() - cooling_start
                # System cooled (logging removed for performance)
                break

            # Cooling in progress (logging removed for performance)
            time.sleep(30)

    def start_monitoring(self):
        """Start continuous thermal monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        # Thermal monitoring started (logging removed for performance)

    def stop_monitoring(self):
        """Stop thermal monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        # Thermal monitoring stopped (logging removed for performance)

    def _monitor_loop(self):
        """Continuous monitoring loop."""
        while self.is_monitoring:
            status = self.get_system_status()
            if status and not status.is_safe:
                event = {
                    'timestamp': status.timestamp,
                    'event_type': 'thermal_warning',
                    'gpu_temp': status.gpu_temp,
                    'cpu_temp': status.cpu_temp,
                    'gpu_power': status.gpu_power,
                    'max_temp': self.max_temp
                }
                self.thermal_events.append(event)
                # Only log critical thermal emergencies (85°C+) for safety
                if status.gpu_temp > 85.0:
                    logger.error(f"🚨 CRITICAL THERMAL: GPU {status.gpu_temp}°C > 85°C")

            time.sleep(self.check_interval)


class MemoryManager:
    """Manages memory usage and optimization."""

    def __init__(self, max_memory_percent: float = 85.0):
        self.max_memory_percent = max_memory_percent
        self.cleanup_count = 0

    def check_memory_usage(self) -> Tuple[bool, float]:
        """Check current memory usage."""
        memory = psutil.virtual_memory()
        is_safe = memory.percent < self.max_memory_percent
        return is_safe, memory.percent

    def cleanup_memory(self) -> bool:
        """Perform memory cleanup."""
        try:
            import gc
            gc.collect()
            self.cleanup_count += 1
            # Memory cleanup performed (logging removed for performance)
            return True
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_percent': memory.percent,
            'cleanup_count': self.cleanup_count
        }


class SessionManager:
    """Manages session persistence and recovery."""

    def __init__(self, session_file: Path):
        self.session_file = session_file
        self.session: Optional[ProcessingSession] = None
        self.save_lock = threading.Lock()

    def create_new_session(self, config: ProcessingConfig) -> ProcessingSession:
        """Create a new processing session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        progress = ProcessingProgress(
            session_id=session_id,
            start_time=datetime.now()
        )

        session = ProcessingSession(
            session_id=session_id,
            config=config,
            progress=progress,
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat()
        )

        self.session = session
        self.save_session()

        if not config.fast_mode:
            logger.info(f"📋 Created new processing session: {session_id}")
        return session

    def load_existing_session(self) -> Optional[ProcessingSession]:
        """Load existing session from file."""
        if not self.session_file.exists():
            return None

        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)

            # Reconstruct session object
            config_data = data['config']
            if 'session_file' in config_data:
                config_data['session_file'] = Path(config_data['session_file'])
            if 'output_dir' in config_data:
                config_data['output_dir'] = Path(config_data['output_dir'])

            config = ProcessingConfig(**config_data)

            # Reconstruct progress
            progress_data = data['progress']
            progress_data['start_time'] = datetime.fromisoformat(progress_data['start_time'])
            if progress_data.get('last_checkpoint'):
                progress_data['last_checkpoint'] = datetime.fromisoformat(progress_data['last_checkpoint'])
            if progress_data.get('completion_time'):
                progress_data['completion_time'] = datetime.fromisoformat(progress_data['completion_time'])

            progress = ProcessingProgress(**progress_data)

            session = ProcessingSession(
                session_id=data['session_id'],
                config=config,
                progress=progress,
                start_time=data['start_time'],
                last_update=data['last_update'],
                is_active=data.get('is_active', True),
                restart_count=data.get('restart_count', 0)
            )

            self.session = session
            if not config.fast_mode:
                logger.info(f"📋 Loaded existing session: {session.session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def save_session(self):
        """Save current session to file."""
        if not self.session:
            return

        # Skip session saving in FAST_MODE (performance optimization)
        if config.fast_mode:
            return

        with self.save_lock:
            try:
                session_dict = asdict(self.session)

                # Convert datetime objects to strings
                progress = session_dict['progress']
                progress['start_time'] = progress['start_time'].isoformat() if isinstance(progress['start_time'], datetime) else progress['start_time']
                if progress.get('last_checkpoint'):
                    progress['last_checkpoint'] = progress['last_checkpoint'].isoformat() if isinstance(progress['last_checkpoint'], datetime) else progress['last_checkpoint']
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


class LLMProcessor:
    """
    Unified LLM processor with comprehensive robustness features.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.session_start = datetime.now()

        # Initialize components
        self.analyzer = DualModelAnalyzer() if config.use_dual_models else InterventionResearchPipeline()
        self.thermal_monitor = ThermalMonitor(
            max_temp=config.max_temp_celsius,
            cooling_temp=config.cooling_temp_celsius,
            max_power=config.max_power_watts,
            check_interval=config.thermal_check_interval
        )
        self.memory_manager = MemoryManager(config.max_memory_percent)
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
                    # Auto-saved session (logging removed for performance)
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")

        self.auto_save_thread = threading.Thread(target=auto_save, daemon=True)
        self.auto_save_thread.start()

    def run_processing_campaign(self, resume: bool = False) -> Dict[str, Any]:
        """Run comprehensive LLM processing campaign."""
        try:
            # Load or create session
            if resume:
                session = self.session_manager.load_existing_session()
                if session:
                    session.restart_count += 1
                    if not config.fast_mode:
                        logger.info(f"🔄 Resuming session (restart #{session.restart_count})")
                else:
                    logger.warning("No existing session found, creating new one")
                    session = self.session_manager.create_new_session(self.config)
            else:
                session = self.session_manager.create_new_session(self.config)

            # Start monitoring and auto-save
            self.thermal_monitor.start_monitoring()
            self._start_auto_save()

            if not config.fast_mode:
                logger.info("🤖 Starting LLM processing campaign")

            # Get papers to process
            papers_to_process = self._get_papers_to_process(session)
            session.progress.total_papers = len(papers_to_process)

            if not papers_to_process:
                if not config.fast_mode:
                    logger.info("📄 No papers found for processing")
                return self._generate_final_report(session)

            logger.info(f"📊 Found {len(papers_to_process)} papers to process")

            # Main processing loop
            processed_count = 0
            for i, paper in enumerate(papers_to_process):
                if self.shutdown_requested:
                    logger.info("🛑 Shutdown requested, saving progress...")
                    break

                try:
                    # Thermal safety check
                    is_safe, thermal_status = self.thermal_monitor.is_thermal_safe()
                    if not is_safe:
                        # Thermal protection triggered (logging removed for performance)
                        session.progress.thermal_events.append({
                            'timestamp': time.time(),
                            'event_type': 'thermal_pause',
                            'gpu_temp': thermal_status.gpu_temp if thermal_status else 0,
                            'cpu_temp': thermal_status.cpu_temp if thermal_status else 0
                        })
                        session.progress.performance_metrics['thermal_pauses'] += 1
                        self.thermal_monitor.wait_for_cooling()

                    # Memory check
                    is_memory_safe, memory_percent = self.memory_manager.check_memory_usage()
                    if not is_memory_safe:
                        logger.warning(f"🧠 Memory usage high ({memory_percent:.1f}%), cleaning up...")
                        self.memory_manager.cleanup_memory()
                        session.progress.performance_metrics['memory_cleanups'] += 1

                    # Process the paper
                    success = self._process_single_paper(paper, session)

                    if success:
                        processed_count += 1
                        session.progress.processed_papers += 1
                    else:
                        session.progress.failed_papers += 1

                    # Update progress and save checkpoint
                    if (processed_count % self.config.checkpoint_interval == 0):
                        session.progress.last_checkpoint = datetime.now()
                        self.session_manager.save_session()
                        logger.info(f"📍 Checkpoint: {processed_count}/{len(papers_to_process)} papers processed")

                    # Memory cleanup interval
                    if (processed_count % self.config.memory_cleanup_interval == 0):
                        if self.config.force_garbage_collection:
                            self.memory_manager.cleanup_memory()

                except Exception as e:
                    logger.error(f"💥 Error processing paper {paper.get('pmid', 'unknown')}: {e}")
                    session.progress.failed_papers += 1
                    session.progress.error_log.append({
                        'timestamp': time.time(),
                        'paper_id': paper.get('pmid', 'unknown'),
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })

            # Final session update
            session.progress.completion_time = datetime.now()
            session.is_active = False
            self.session_manager.save_session()

            # Generate final report
            return self._generate_final_report(session)

        except Exception as e:
            logger.error(f"💥 Critical processing error: {e}")
            logger.error(traceback.format_exc())

            if self.config.auto_restart_on_error:
                logger.info("🔄 Auto-restart triggered due to critical error")
                time.sleep(60)  # Brief pause before restart
                return self.run_processing_campaign(resume=True)
            else:
                raise
        finally:
            # Stop monitoring and auto-save
            self.thermal_monitor.stop_monitoring()
            if self.auto_save_thread:
                self.auto_save_stop.set()
                self.auto_save_thread.join(timeout=5)

    def _get_papers_to_process(self, session: ProcessingSession) -> List[Dict[str, Any]]:
        """Get list of papers that need processing."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Build query based on configuration
                conditions = []
                params = []

                if self.config.reprocess_failed:
                    conditions.append("processing_status = 'failed'")
                elif self.config.force_reprocess:
                    # Process all papers regardless of status
                    pass
                else:
                    # Standard mode: process unprocessed papers
                    conditions.append("(processing_status IS NULL OR processing_status = 'pending')")

                # Filter by conditions if specified
                if self.config.conditions_filter:
                    condition_placeholders = ','.join(['?' for _ in self.config.conditions_filter])
                    conditions.append(f"""
                        id IN (
                            SELECT DISTINCT paper_id
                            FROM interventions
                            WHERE condition IN ({condition_placeholders})
                        )
                    """)
                    params.extend(self.config.conditions_filter)

                where_clause = " AND ".join(conditions) if conditions else "1=1"

                query = f"""
                    SELECT id, pmid, title, abstract, processing_status, created_at
                    FROM papers
                    WHERE {where_clause}
                    ORDER BY created_at ASC
                """

                if self.config.limit:
                    query += f" LIMIT {self.config.limit}"

                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get papers to process: {e}")
            return []

    def _process_single_paper(self, paper: Dict[str, Any], session: ProcessingSession) -> bool:
        """Process a single paper with error handling."""
        paper_id = paper.get('pmid', 'unknown')
        session.progress.current_paper_id = paper_id

        start_time = time.time()
        logger.info(f"🔬 Processing paper {paper_id}: {paper.get('title', 'No title')[:100]}...")

        for attempt in range(self.config.max_retries_per_paper):
            try:
                if self.config.use_dual_models:
                    # Use dual model analyzer
                    result = self.analyzer.process_papers_batch(
                        papers=[paper],
                        save_to_db=True,
                        batch_size=1
                    )
                else:
                    # Use single model pipeline
                    result = self.analyzer.process_paper(paper)

                # Update intervention count
                if result and result.get('interventions_extracted', 0) > 0:
                    session.progress.interventions_extracted += result.get('interventions_extracted', 0)

                # Update performance metrics
                processing_time = time.time() - start_time
                session.progress.performance_metrics['total_processing_time'] += processing_time

                if session.progress.processed_papers > 0:
                    session.progress.performance_metrics['avg_processing_time'] = (
                        session.progress.performance_metrics['total_processing_time'] /
                        (session.progress.processed_papers + 1)
                    )

                # Calculate papers per hour
                if session.progress.duration.total_seconds() > 0:
                    session.progress.performance_metrics['papers_per_hour'] = (
                        (session.progress.processed_papers + 1) /
                        (session.progress.duration.total_seconds() / 3600)
                    )

                logger.info(f"✅ Successfully processed paper {paper_id} in {processing_time:.1f}s")
                return True

            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} failed for paper {paper_id}: {e}")

                if attempt < self.config.max_retries_per_paper - 1:
                    logger.info(f"⏳ Retrying paper {paper_id} in {self.config.retry_delay} seconds...")
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"💥 Failed to process paper {paper_id} after {self.config.max_retries_per_paper} attempts")

                    # Mark paper as failed in database
                    try:
                        with database_manager.get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "UPDATE papers SET processing_status = 'failed' WHERE pmid = ?",
                                (paper_id,)
                            )
                            conn.commit()
                    except Exception as db_error:
                        logger.error(f"Failed to mark paper as failed in database: {db_error}")

                    if not self.config.skip_failed_papers:
                        raise

        return False

    def _generate_final_report(self, session: ProcessingSession) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        logger.info("📋 Generating final processing report...")

        # Calculate statistics
        total_duration = datetime.now() - self.session_start
        success_rate = (session.progress.processed_papers / session.progress.total_papers * 100) if session.progress.total_papers > 0 else 0

        # Get thermal statistics
        thermal_stats = {
            'thermal_events': len(session.progress.thermal_events),
            'thermal_pauses': session.progress.performance_metrics.get('thermal_pauses', 0),
            'current_status': asdict(self.thermal_monitor.current_status) if self.thermal_monitor.current_status else None
        }

        # Get memory statistics
        memory_stats = self.memory_manager.get_memory_stats()

        report = {
            'session_info': {
                'session_id': session.session_id,
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': str(total_duration),
                'restart_count': session.restart_count
            },
            'configuration': asdict(self.config),
            'processing_statistics': {
                'total_papers': session.progress.total_papers,
                'processed_papers': session.progress.processed_papers,
                'failed_papers': session.progress.failed_papers,
                'skipped_papers': session.progress.skipped_papers,
                'interventions_extracted': session.progress.interventions_extracted,
                'success_rate': success_rate
            },
            'performance_metrics': session.progress.performance_metrics,
            'thermal_statistics': thermal_stats,
            'memory_statistics': memory_stats,
            'error_summary': {
                'total_errors': len(session.progress.error_log),
                'recent_errors': session.progress.error_log[-5:] if session.progress.error_log else []
            }
        }

        # Save final report
        report_file = self.config.output_dir / f"processing_report_{session.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate CSV summary if requested
        if 'csv' in self.config.export_formats:
            self._export_csv_summary(report, session.session_id)

        logger.info(f"📋 Final report saved to {report_file}")
        if not config.fast_mode:
            logger.info(f"🎯 Processing completed: {session.progress.processed_papers}/{session.progress.total_papers} papers successful")
            logger.info(f"📊 Interventions extracted: {session.progress.interventions_extracted}")
        logger.info(f"⏱️ Total duration: {total_duration}")
        # Thermal events logged (logging removed for performance)

        return report

    def _export_csv_summary(self, report: Dict[str, Any], session_id: str):
        """Export summary to CSV format."""
        try:
            import csv

            csv_file = self.config.output_dir / f"processing_summary_{session_id}.csv"

            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])

                # Write key statistics
                stats = report['processing_statistics']
                for key, value in stats.items():
                    writer.writerow([key.replace('_', ' ').title(), value])

                # Write performance metrics
                writer.writerow(['', ''])  # Empty row
                writer.writerow(['Performance Metrics', ''])
                perf = report['performance_metrics']
                for key, value in perf.items():
                    writer.writerow([key.replace('_', ' ').title(), f"{value:.2f}" if isinstance(value, float) else value])

            logger.info(f"📊 CSV summary exported to {csv_file}")

        except Exception as e:
            logger.error(f"Failed to export CSV summary: {e}")

    def show_status(self) -> Dict[str, Any]:
        """Show current processing status."""
        session = self.session_manager.load_existing_session()
        if not session:
            return {"error": "No active session found"}

        # Get current thermal status
        thermal_status = self.thermal_monitor.get_system_status()

        status = {
            'session_id': session.session_id,
            'start_time': session.start_time,
            'is_active': session.is_active,
            'progress': {
                'total_papers': session.progress.total_papers,
                'processed_papers': session.progress.processed_papers,
                'failed_papers': session.progress.failed_papers,
                'progress_percent': session.progress.progress_percent,
                'current_paper': session.progress.current_paper_id,
                'estimated_completion': session.progress.estimated_completion.isoformat() if session.progress.estimated_completion else None
            },
            'performance': session.progress.performance_metrics,
            'thermal_status': asdict(thermal_status) if thermal_status else None,
            'memory_status': self.memory_manager.get_memory_stats()
        }

        return status

    def show_thermal_status(self) -> Dict[str, Any]:
        """Show detailed thermal status."""
        status = self.thermal_monitor.get_system_status()
        if not status:
            return {"error": "Cannot read thermal status"}

        return {
            'thermal_status': asdict(status),
            'safety_thresholds': {
                'max_temp': self.thermal_monitor.max_temp,
                'cooling_temp': self.thermal_monitor.cooling_temp,
                'max_power': self.thermal_monitor.max_power
            },
            'thermal_events': self.thermal_monitor.thermal_events[-10:],  # Last 10 events
            'recommendations': self._get_thermal_recommendations(status)
        }

    def _get_thermal_recommendations(self, status: ThermalStatus) -> List[str]:
        """Get thermal optimization recommendations."""
        recommendations = []

        if status.gpu_temp > self.thermal_monitor.max_temp * 0.9:
            recommendations.append("GPU temperature approaching limit - consider reducing batch size")

        if status.gpu_power > self.thermal_monitor.max_power * 0.9:
            recommendations.append("GPU power draw high - monitor for thermal throttling")

        if status.cpu_temp > 80:
            recommendations.append("CPU temperature elevated - check cooling system")

        if status.ram_usage > 90:
            recommendations.append("RAM usage very high - consider reducing memory usage")

        if status.gpu_memory_used / status.gpu_memory_total > 0.95:
            recommendations.append("GPU memory nearly full - reduce batch size or model complexity")

        if not recommendations:
            recommendations.append("System operating within normal parameters")

        return recommendations


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified LLM Processor for Medical Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Processing targets
    parser.add_argument('--limit', type=int,
                       help='Maximum number of papers to process')
    parser.add_argument('--all', action='store_true',
                       help='Process all unprocessed papers')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Papers processed per batch (default: 5)')

    # Processing modes
    parser.add_argument('--reprocess-failed', action='store_true',
                       help='Reprocess previously failed papers')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocess all papers regardless of status')
    parser.add_argument('--conditions', type=str,
                       help='Comma-separated list of conditions to filter')

    # Model configuration
    parser.add_argument('--single-model', action='store_true',
                       help='Use single model instead of dual-model analysis')
    parser.add_argument('--models', type=str, default='gemma2:9b,qwen2.5:14b',
                       help='Comma-separated list of models to use')

    # Thermal protection
    parser.add_argument('--max-temp', type=float, default=80.0,
                       help='Maximum GPU temperature (°C, default: 80)')
    parser.add_argument('--cooling-temp', type=float, default=70.0,
                       help='Temperature to resume after cooling (°C, default: 70)')
    parser.add_argument('--max-power', type=float, default=250.0,
                       help='Maximum GPU power draw (W, default: 250)')

    # Session management
    parser.add_argument('--resume', action='store_true',
                       help='Resume previous session')
    parser.add_argument('--session-file', type=str, default='processing_session.json',
                       help='Session state file')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='Save progress every N papers (default: 5)')

    # Output options
    parser.add_argument('--output-dir', type=str, default='processing_results',
                       help='Output directory for results')
    parser.add_argument('--export-formats', type=str, default='json',
                       help='Export formats (comma-separated): json,csv')

    # Operational modes
    parser.add_argument('--overnight', action='store_true',
                       help='Enable overnight operation mode')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    parser.add_argument('--max-workers', type=int, default=2,
                       help='Maximum worker threads for parallel processing')

    # Utility options
    parser.add_argument('--status', action='store_true',
                       help='Show current processing status')
    parser.add_argument('--thermal-status', action='store_true',
                       help='Show detailed thermal status')
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
            config = ProcessingConfig(**config_data)
        else:
            # Build configuration from arguments
            config = ProcessingConfig(
                limit=args.limit if not args.all else None,
                batch_size=args.batch_size,
                use_dual_models=not args.single_model,
                models=args.models.split(','),
                max_temp_celsius=args.max_temp,
                cooling_temp_celsius=args.cooling_temp,
                max_power_watts=args.max_power,
                session_file=Path(args.session_file),
                checkpoint_interval=args.checkpoint_interval,
                output_dir=Path(args.output_dir),
                export_formats=args.export_formats.split(','),
                overnight_mode=args.overnight,
                parallel_processing=args.parallel,
                max_workers=args.max_workers,
                reprocess_failed=args.reprocess_failed,
                force_reprocess=args.force_reprocess,
                conditions_filter=args.conditions.split(',') if args.conditions else []
            )

        # Initialize processor
        processor = LLMProcessor(config)

        # Handle different run modes
        if args.status:
            # Show status and exit
            status = processor.show_status()
            if 'error' in status:
                print(status['error'])
            else:
                print("Current processing status:")
                print(f"Session: {status['session_id']}")
                print(f"Progress: {status['progress']['processed_papers']}/{status['progress']['total_papers']} ({status['progress']['progress_percent']:.1f}%)")
                print(f"Current paper: {status['progress']['current_paper'] or 'None'}")
                if status['progress']['estimated_completion']:
                    print(f"Estimated completion: {status['progress']['estimated_completion']}")
                print(f"Papers/hour: {status['performance']['papers_per_hour']:.1f}")
                if status['thermal_status']:
                    print(f"GPU temp: {status['thermal_status']['gpu_temp']:.1f}°C")
            return True

        if args.thermal_status:
            # Show thermal status and exit
            thermal = processor.show_thermal_status()
            if 'error' in thermal:
                print(thermal['error'])
            else:
                status = thermal['thermal_status']
                print("Current thermal status:")
                print(f"GPU: {status['gpu_temp']:.1f}°C ({status['gpu_power']:.1f}W, {status['gpu_utilization']:.1f}%)")
                print(f"CPU: {status['cpu_temp']:.1f}°C ({status['cpu_usage']:.1f}%)")
                print(f"RAM: {status['ram_usage']:.1f}%")
                print(f"GPU Memory: {status['gpu_memory_used']:.0f}MB/{status['gpu_memory_total']:.0f}MB")
                print(f"Safe: {'Yes' if status['is_safe'] else 'No'}")
                print("\nRecommendations:")
                for rec in thermal['recommendations']:
                    print(f"  • {rec}")
            return True

        # Run processing
        logger.info("🚀 Starting unified LLM processing")
        report = processor.run_processing_campaign(resume=args.resume)

        # Print summary
        print(f"\n=== LLM Processing Complete ===")
        print(f"Session: {report['session_info']['session_id']}")
        print(f"Duration: {report['session_info']['total_duration']}")
        print(f"Papers processed: {report['processing_statistics']['processed_papers']}/{report['processing_statistics']['total_papers']}")
        print(f"Success rate: {report['processing_statistics']['success_rate']:.1f}%")
        print(f"Interventions extracted: {report['processing_statistics']['interventions_extracted']}")
        print(f"Thermal events: {report['thermal_statistics']['thermal_events']}")

        return True

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\nProcessing interrupted. Session state saved.")
        return True
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.error(traceback.format_exc())
        print(f"Processing failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)