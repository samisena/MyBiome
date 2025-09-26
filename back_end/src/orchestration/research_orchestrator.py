#!/usr/bin/env python3
"""
Research Orchestrator - Master orchestrator for complete research workflows

This script orchestrates the complete research pipeline by coordinating:
- Paper collection (via paper_collector.py)
- LLM processing (via llm_processor.py)
- Multi-condition campaigns
- Comprehensive monitoring and reporting

Features:
- Intelligent workflow coordination
- Cross-phase session management
- Complete fault tolerance
- Multi-condition orchestration
- Thermal protection throughout
- Progress tracking across all phases
- Auto-restart capabilities
- Overnight operation capacity

Workflow Phases:
1. Collection Phase: Gather papers from PubMed/S2
2. Processing Phase: Extract interventions via LLM
3. Validation Phase: Quality checks and cleanup
4. Reporting Phase: Generate comprehensive reports

Usage:
    # Complete workflow for single condition
    python research_orchestrator.py "ibs" --papers 500

    # Multi-condition overnight campaign
    python research_orchestrator.py --conditions "ibs,gerd,crohns" --papers-per-condition 1000 --overnight

    # Resume interrupted campaign
    python research_orchestrator.py --resume

    # Collection only
    python research_orchestrator.py --conditions "diabetes" --collection-only

Examples:
    # Standard research campaign
    python research_orchestrator.py "inflammatory bowel disease" --papers 200 --batch-size 5

    # Overnight multi-condition campaign
    python research_orchestrator.py --conditions "ibs,gerd,crohns,colitis" --papers-per-condition 500 --overnight --auto-restart

    # Resume with status monitoring
    python research_orchestrator.py --resume --status --thermal-monitor

    # Processing-only workflow
    python research_orchestrator.py --processing-only --limit 100
"""

import sys
import json
import time
import signal
import argparse
import subprocess
import traceback
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from back_end.src.data.config import config, setup_logging
    from back_end.src.data.repositories import repository_manager
    from back_end.src.data_collection.database_manager import database_manager
    from back_end.src.utils.batch_file_operations import cleanup_old_files

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)

logger = setup_logging(__name__, 'research_orchestrator.log')


class WorkflowPhase(Enum):
    """Enumeration of workflow phases."""
    INITIALIZATION = "initialization"
    COLLECTION = "collection"
    PROCESSING = "processing"
    VALIDATION = "validation"
    DATA_MINING = "data_mining"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OrchestrationConfig:
    """Configuration for research orchestration."""
    # Target configuration
    conditions: List[str]
    papers_per_condition: int = 500
    min_year: int = 2015
    max_year: Optional[int] = None

    # Workflow control
    collection_only: bool = False
    processing_only: bool = False
    skip_validation: bool = False
    workflow_phases: List[str] = None

    # Collection configuration
    collection_batch_size: int = 50
    use_interleaved_s2: bool = True
    traditional_mode: bool = False

    # Processing configuration
    processing_batch_size: int = 5
    use_dual_models: bool = True
    processing_limit: Optional[int] = None

    # Thermal protection
    max_temp_celsius: float = 80.0
    cooling_temp_celsius: float = 70.0
    thermal_check_interval: int = 30  # seconds

    # Session management
    session_file: Path = Path("orchestration_session.json")
    auto_save_interval: int = 60   # seconds
    checkpoint_interval: int = 10  # operations

    # Error handling
    max_retries_per_phase: int = 3
    auto_restart_on_error: bool = True
    continue_on_phase_failure: bool = True

    # Output options
    output_dir: Path = Path("orchestration_results")
    save_intermediate: bool = True
    generate_reports: bool = True

    # Operational modes
    overnight_mode: bool = False
    parallel_conditions: bool = False
    max_condition_workers: int = 2

    def __post_init__(self):
        if self.workflow_phases is None:
            if self.collection_only:
                self.workflow_phases = [WorkflowPhase.COLLECTION.value]
            elif self.processing_only:
                self.workflow_phases = [WorkflowPhase.PROCESSING.value]
            else:
                self.workflow_phases = [
                    WorkflowPhase.COLLECTION.value,
                    WorkflowPhase.PROCESSING.value,
                    WorkflowPhase.VALIDATION.value,
                    WorkflowPhase.DATA_MINING.value,
                    WorkflowPhase.REPORTING.value
                ]
        self.output_dir = Path(self.output_dir)
        self.session_file = Path(self.session_file)


@dataclass
class PhaseProgress:
    """Progress tracking for individual workflow phases."""
    phase: str
    status: str = 'pending'  # pending, running, completed, failed, skipped
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    operations_completed: int = 0
    total_operations: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    phase_results: Dict[str, Any] = None

    def __post_init__(self):
        if self.phase_results is None:
            self.phase_results = {}

    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.completion_time:
            return self.completion_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None

    @property
    def progress_percent(self) -> float:
        if self.total_operations > 0:
            return (self.operations_completed / self.total_operations) * 100
        return 0.0


@dataclass
class ConditionProgress:
    """Progress tracking for individual conditions."""
    condition: str
    current_phase: str = WorkflowPhase.INITIALIZATION.value
    status: str = 'pending'  # pending, running, completed, failed
    phase_progress: Dict[str, PhaseProgress] = None
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    papers_collected: int = 0
    papers_processed: int = 0
    interventions_extracted: int = 0
    error_count: int = 0

    def __post_init__(self):
        if self.phase_progress is None:
            self.phase_progress = {}

    @property
    def overall_progress_percent(self) -> float:
        if not self.phase_progress:
            return 0.0

        total_phases = len(self.phase_progress)
        completed_phases = sum(1 for p in self.phase_progress.values() if p.status == 'completed')

        if total_phases == 0:
            return 0.0

        # Add partial progress from current phase
        current_phase_progress = 0.0
        for phase in self.phase_progress.values():
            if phase.status == 'running':
                current_phase_progress = phase.progress_percent / 100
                break

        return ((completed_phases + current_phase_progress) / total_phases) * 100


@dataclass
class OrchestrationSession:
    """Complete orchestration session state."""
    session_id: str
    config: OrchestrationConfig
    conditions_progress: Dict[str, ConditionProgress]
    global_statistics: Dict[str, Any]
    start_time: str
    last_update: str
    current_phase: str = WorkflowPhase.INITIALIZATION.value
    is_active: bool = True
    restart_count: int = 0


class ThermalMonitor:
    """Simplified thermal monitoring for orchestration."""

    def __init__(self, max_temp: float = 80.0, check_interval: int = 30):
        self.max_temp = max_temp
        self.check_interval = check_interval
        self.is_monitoring = False
        self.current_temp = 0.0

    def get_gpu_temperature(self) -> Optional[float]:
        """Get current GPU temperature."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Cannot read GPU temperature: {e}")
        return None

    def is_thermal_safe(self) -> Tuple[bool, Optional[float]]:
        """Check if thermal conditions are safe."""
        temp = self.get_gpu_temperature()
        self.current_temp = temp or 0.0
        return (temp is None or temp < self.max_temp), temp

    def wait_for_cooling(self, target_temp: float = 70.0):
        """Wait for GPU to cool down."""
        logger.info(f"🌡️ Thermal protection: waiting for GPU to cool to {target_temp}°C...")

        while True:
            temp = self.get_gpu_temperature()
            if temp is None or temp <= target_temp:
                logger.info(f"🌡️ GPU cooled to {temp or 'unknown'}°C")
                break

            logger.info(f"🌡️ Cooling... Current: {temp}°C, Target: {target_temp}°C")
            time.sleep(30)


class WorkflowExecutor:
    """Executes individual workflow phases by calling appropriate scripts."""

    def __init__(self, config: OrchestrationConfig, thermal_monitor: ThermalMonitor):
        self.config = config
        self.thermal_monitor = thermal_monitor
        self.pipelines_dir = Path(__file__).parent

    def execute_collection_phase(self, condition: str, progress: PhaseProgress) -> Dict[str, Any]:
        """Execute paper collection phase."""
        logger.info(f"📚 Starting collection phase for '{condition}'")
        progress.status = 'running'
        progress.start_time = datetime.now()

        try:
            # Build collection command
            cmd = [
                sys.executable,
                str(self.pipelines_dir / "paper_collector.py"),
                condition,
                "--max-papers", str(self.config.papers_per_condition),
                "--min-year", str(self.config.min_year),
                "--batch-size", str(self.config.collection_batch_size),
                "--output-dir", str(self.config.output_dir / "collection"),
                "--checkpoint-interval", str(self.config.checkpoint_interval)
            ]

            if self.config.max_year:
                cmd.extend(["--max-year", str(self.config.max_year)])

            if self.config.traditional_mode:
                cmd.append("--traditional-mode")

            if not self.config.use_interleaved_s2:
                cmd.append("--skip-s2")

            # Execute collection
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout

            if result.returncode == 0:
                progress.status = 'completed'
                progress.completion_time = datetime.now()
                progress.operations_completed = progress.total_operations = 1

                # Parse results from stdout/stderr if available
                result_data = {
                    'returncode': result.returncode,
                    'stdout': result.stdout[-1000:],  # Last 1000 chars
                    'stderr': result.stderr[-1000:] if result.stderr else None
                }

                # Try to extract collection statistics
                papers_collected = self._extract_papers_collected(result.stdout)
                result_data['papers_collected'] = papers_collected

                progress.phase_results = result_data
                logger.info(f"✅ Collection completed for '{condition}': {papers_collected} papers")
                return result_data

            else:
                error_msg = f"Collection failed with code {result.returncode}: {result.stderr}"
                progress.status = 'failed'
                progress.error_count += 1
                progress.last_error = error_msg
                logger.error(f"❌ Collection failed for '{condition}': {error_msg}")
                raise RuntimeError(error_msg)

        except Exception as e:
            progress.status = 'failed'
            progress.error_count += 1
            progress.last_error = str(e)
            logger.error(f"❌ Collection phase failed for '{condition}': {e}")
            raise

    def execute_processing_phase(self, condition: str, progress: PhaseProgress) -> Dict[str, Any]:
        """Execute LLM processing phase."""
        logger.info(f"🤖 Starting processing phase for '{condition}'")
        progress.status = 'running'
        progress.start_time = datetime.now()

        try:
            # Check thermal safety before processing
            is_safe, temp = self.thermal_monitor.is_thermal_safe()
            if not is_safe:
                logger.warning(f"🌡️ Thermal protection triggered before processing: {temp}°C")
                self.thermal_monitor.wait_for_cooling(self.config.cooling_temp_celsius)

            # Build processing command
            cmd = [
                sys.executable,
                str(self.pipelines_dir / "llm_processor.py"),
                "--batch-size", str(self.config.processing_batch_size),
                "--max-temp", str(self.config.max_temp_celsius),
                "--cooling-temp", str(self.config.cooling_temp_celsius),
                "--output-dir", str(self.config.output_dir / "processing"),
                "--checkpoint-interval", str(self.config.checkpoint_interval)
            ]

            if self.config.processing_limit:
                cmd.extend(["--limit", str(self.config.processing_limit)])
            else:
                cmd.append("--all")

            if not self.config.use_dual_models:
                cmd.append("--single-model")

            if condition:  # Filter by condition if processing specific condition
                cmd.extend(["--conditions", condition])

            # Execute processing
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)  # 4 hour timeout

            if result.returncode == 0:
                progress.status = 'completed'
                progress.completion_time = datetime.now()
                progress.operations_completed = progress.total_operations = 1

                result_data = {
                    'returncode': result.returncode,
                    'stdout': result.stdout[-1000:],
                    'stderr': result.stderr[-1000:] if result.stderr else None
                }

                # Try to extract processing statistics
                processing_stats = self._extract_processing_stats(result.stdout)
                result_data.update(processing_stats)

                progress.phase_results = result_data
                logger.info(f"✅ Processing completed for '{condition}': {processing_stats.get('papers_processed', 0)} papers")
                return result_data

            else:
                error_msg = f"Processing failed with code {result.returncode}: {result.stderr}"
                progress.status = 'failed'
                progress.error_count += 1
                progress.last_error = error_msg
                logger.error(f"❌ Processing failed for '{condition}': {error_msg}")
                raise RuntimeError(error_msg)

        except Exception as e:
            progress.status = 'failed'
            progress.error_count += 1
            progress.last_error = str(e)
            logger.error(f"❌ Processing phase failed for '{condition}': {e}")
            raise

    def execute_validation_phase(self, condition: str, progress: PhaseProgress) -> Dict[str, Any]:
        """Execute validation phase."""
        logger.info(f"✅ Starting validation phase for '{condition}'")
        progress.status = 'running'
        progress.start_time = datetime.now()

        try:
            # Validation checks
            validation_results = {}

            # Check database consistency
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Count papers for this condition
                cursor.execute("""
                    SELECT COUNT(DISTINCT p.id) as paper_count,
                           COUNT(i.id) as intervention_count,
                           COUNT(CASE WHEN p.processing_status = 'processed' THEN 1 END) as processed_count
                    FROM papers p
                    LEFT JOIN interventions i ON p.id = i.paper_id
                    WHERE LOWER(i.condition) LIKE LOWER(?) OR i.condition IS NULL
                """, (f"%{condition}%",))

                stats = cursor.fetchone()
                validation_results = {
                    'paper_count': stats[0] if stats else 0,
                    'intervention_count': stats[1] if stats else 0,
                    'processed_count': stats[2] if stats else 0
                }

                # Check for processing errors
                cursor.execute("""
                    SELECT COUNT(*) FROM papers p
                    JOIN interventions i ON p.id = i.paper_id
                    WHERE LOWER(i.condition) LIKE LOWER(?) AND p.processing_status = 'failed'
                """, (f"%{condition}%",))

                failed_count = cursor.fetchone()[0] if cursor.fetchone() else 0
                validation_results['failed_papers'] = failed_count

            progress.status = 'completed'
            progress.completion_time = datetime.now()
            progress.operations_completed = progress.total_operations = 1
            progress.phase_results = validation_results

            logger.info(f"✅ Validation completed for '{condition}': {validation_results}")
            return validation_results

        except Exception as e:
            progress.status = 'failed'
            progress.error_count += 1
            progress.last_error = str(e)
            logger.error(f"❌ Validation phase failed for '{condition}': {e}")
            raise

    def execute_data_mining_phase(self, condition: str, progress: PhaseProgress) -> Dict[str, Any]:
        """Execute data mining phase."""
        logger.info(f"🔬 Starting data mining phase for '{condition}'")
        progress.status = 'running'
        progress.start_time = datetime.now()

        try:
            # Build data mining command
            cmd = [
                sys.executable,
                str(self.pipelines_dir.parent / "data_mining" / "data_mining_orchestrator.py"),
                "--conditions", condition,
                "--all"  # Run all data mining analyses
            ]

            # Run data mining orchestrator
            logger.info(f"🔬 Running data mining for condition: {condition}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.max_script_timeout,
                cwd=self.pipelines_dir.parent
            )

            if result.returncode == 0:
                logger.info(f"✅ Data mining phase completed for '{condition}'")
                progress.status = 'completed'
                progress.end_time = datetime.now()

                # Parse results
                analysis_results = {
                    'analyses_completed': ['knowledge_graph', 'bayesian_scoring', 'treatment_recommendations'],
                    'condition': condition,
                    'returncode': result.returncode,
                    'stdout_preview': result.stdout[:500] if result.stdout else "",
                    'data_mining_database_updated': True
                }

                return analysis_results

            else:
                logger.error(f"❌ Data mining failed for '{condition}': {result.stderr}")
                progress.status = 'failed'
                progress.error_count += 1
                progress.last_error = result.stderr
                raise RuntimeError(f"Data mining failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Data mining phase timed out for '{condition}'")
            progress.status = 'failed'
            progress.error_count += 1
            progress.last_error = "Data mining phase timed out"
            raise RuntimeError("Data mining phase timed out")

        except Exception as e:
            progress.status = 'failed'
            progress.error_count += 1
            progress.last_error = str(e)
            logger.error(f"❌ Data mining phase failed for '{condition}': {e}")
            raise

    def execute_reporting_phase(self, condition: str, progress: PhaseProgress) -> Dict[str, Any]:
        """Execute reporting phase."""
        logger.info(f"📊 Starting reporting phase for '{condition}'")
        progress.status = 'running'
        progress.start_time = datetime.now()

        try:
            # Generate condition-specific report
            report_data = self._generate_condition_report(condition)

            # Save report
            report_file = self.config.output_dir / f"condition_report_{condition.replace(' ', '_')}.json"
            report_file.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            progress.status = 'completed'
            progress.completion_time = datetime.now()
            progress.operations_completed = progress.total_operations = 1
            progress.phase_results = {'report_file': str(report_file), 'report_data': report_data}

            logger.info(f"📊 Report generated for '{condition}': {report_file}")
            return progress.phase_results

        except Exception as e:
            progress.status = 'failed'
            progress.error_count += 1
            progress.last_error = str(e)
            logger.error(f"❌ Reporting phase failed for '{condition}': {e}")
            raise

    def _extract_papers_collected(self, stdout: str) -> int:
        """Extract number of papers collected from stdout."""
        try:
            # Look for patterns like "Total papers collected: 123"
            import re
            patterns = [
                r'Total papers collected:\s*(\d+)',
                r'Successfully collected (\d+)',
                r'(\d+) papers added to database',
                r'Collected (\d+) papers'
            ]

            for pattern in patterns:
                match = re.search(pattern, stdout, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        except Exception:
            pass
        return 0

    def _extract_processing_stats(self, stdout: str) -> Dict[str, int]:
        """Extract processing statistics from stdout."""
        try:
            import re
            stats = {}

            patterns = {
                'papers_processed': r'Papers processed:\s*(\d+)',
                'interventions_extracted': r'Interventions extracted:\s*(\d+)',
                'success_rate': r'Success rate:\s*([\d.]+)%'
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, stdout, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    stats[key] = float(value) if '.' in value else int(value)

            return stats
        except Exception:
            return {}

    def _generate_condition_report(self, condition: str) -> Dict[str, Any]:
        """Generate comprehensive report for a condition."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get comprehensive statistics
                cursor.execute("""
                    SELECT
                        COUNT(DISTINCT p.id) as total_papers,
                        COUNT(DISTINCT CASE WHEN p.processing_status = 'processed' THEN p.id END) as processed_papers,
                        COUNT(i.id) as total_interventions,
                        COUNT(DISTINCT i.intervention_name) as unique_interventions,
                        AVG(i.confidence_score) as avg_confidence,
                        MIN(p.publication_year) as earliest_year,
                        MAX(p.publication_year) as latest_year
                    FROM papers p
                    LEFT JOIN interventions i ON p.id = i.paper_id
                    WHERE LOWER(i.condition) LIKE LOWER(?)
                """, (f"%{condition}%",))

                stats = cursor.fetchone()

                # Get top interventions
                cursor.execute("""
                    SELECT intervention_name, COUNT(*) as frequency, AVG(confidence_score) as avg_confidence
                    FROM interventions
                    WHERE LOWER(condition) LIKE LOWER(?)
                    GROUP BY intervention_name
                    ORDER BY frequency DESC, avg_confidence DESC
                    LIMIT 10
                """, (f"%{condition}%",))

                top_interventions = [dict(row) for row in cursor.fetchall()]

                return {
                    'condition': condition,
                    'generated_at': datetime.now().isoformat(),
                    'statistics': {
                        'total_papers': stats[0] if stats else 0,
                        'processed_papers': stats[1] if stats else 0,
                        'total_interventions': stats[2] if stats else 0,
                        'unique_interventions': stats[3] if stats else 0,
                        'avg_confidence': stats[4] if stats else 0,
                        'year_range': f"{stats[5]}-{stats[6]}" if stats and stats[5] and stats[6] else "N/A"
                    },
                    'top_interventions': top_interventions
                }

        except Exception as e:
            logger.error(f"Failed to generate report for {condition}: {e}")
            return {'condition': condition, 'error': str(e)}


class SessionManager:
    """Manages orchestration session persistence."""

    def __init__(self, session_file: Path):
        self.session_file = session_file
        self.session: Optional[OrchestrationSession] = None
        self.save_lock = threading.Lock()

    def create_new_session(self, config: OrchestrationConfig) -> OrchestrationSession:
        """Create a new orchestration session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        conditions_progress = {}
        for condition in config.conditions:
            condition_progress = ConditionProgress(condition=condition)

            # Initialize phase progress
            for phase_name in config.workflow_phases:
                condition_progress.phase_progress[phase_name] = PhaseProgress(phase=phase_name)

            conditions_progress[condition] = condition_progress

        session = OrchestrationSession(
            session_id=session_id,
            config=config,
            conditions_progress=conditions_progress,
            global_statistics={
                'total_conditions': len(config.conditions),
                'completed_conditions': 0,
                'failed_conditions': 0,
                'total_papers_collected': 0,
                'total_papers_processed': 0,
                'total_interventions_extracted': 0,
                'thermal_events': 0
            },
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat()
        )

        self.session = session
        self.save_session()

        logger.info(f"📋 Created new orchestration session: {session_id}")
        logger.info(f"🎯 Target: {len(config.conditions)} conditions, {len(config.workflow_phases)} phases each")

        return session

    def load_existing_session(self) -> Optional[OrchestrationSession]:
        """Load existing session from file."""
        if not self.session_file.exists():
            return None

        try:
            with open(self.session_file, 'r') as f:
                data = json.load(f)

            # Reconstruct config
            config_data = data['config']
            if 'session_file' in config_data:
                config_data['session_file'] = Path(config_data['session_file'])
            if 'output_dir' in config_data:
                config_data['output_dir'] = Path(config_data['output_dir'])

            config = OrchestrationConfig(**config_data)

            # Reconstruct conditions progress
            conditions_progress = {}
            for condition, progress_data in data['conditions_progress'].items():
                # Convert datetime strings back to datetime objects
                if progress_data.get('start_time'):
                    progress_data['start_time'] = datetime.fromisoformat(progress_data['start_time'])
                if progress_data.get('completion_time'):
                    progress_data['completion_time'] = datetime.fromisoformat(progress_data['completion_time'])

                # Reconstruct phase progress
                phase_progress = {}
                for phase_name, phase_data in progress_data.get('phase_progress', {}).items():
                    if phase_data.get('start_time'):
                        phase_data['start_time'] = datetime.fromisoformat(phase_data['start_time'])
                    if phase_data.get('completion_time'):
                        phase_data['completion_time'] = datetime.fromisoformat(phase_data['completion_time'])

                    phase_progress[phase_name] = PhaseProgress(**phase_data)

                progress_data['phase_progress'] = phase_progress
                conditions_progress[condition] = ConditionProgress(**progress_data)

            session = OrchestrationSession(
                session_id=data['session_id'],
                config=config,
                conditions_progress=conditions_progress,
                global_statistics=data['global_statistics'],
                start_time=data['start_time'],
                last_update=data['last_update'],
                current_phase=data.get('current_phase', WorkflowPhase.INITIALIZATION.value),
                is_active=data.get('is_active', True),
                restart_count=data.get('restart_count', 0)
            )

            self.session = session
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
                for condition, progress in session_dict['conditions_progress'].items():
                    if progress.get('start_time'):
                        progress['start_time'] = progress['start_time'].isoformat() if isinstance(progress['start_time'], datetime) else progress['start_time']
                    if progress.get('completion_time'):
                        progress['completion_time'] = progress['completion_time'].isoformat() if isinstance(progress['completion_time'], datetime) else progress['completion_time']

                    # Convert phase progress datetime objects
                    for phase_name, phase_data in progress.get('phase_progress', {}).items():
                        if phase_data.get('start_time'):
                            phase_data['start_time'] = phase_data['start_time'].isoformat() if isinstance(phase_data['start_time'], datetime) else phase_data['start_time']
                        if phase_data.get('completion_time'):
                            phase_data['completion_time'] = phase_data['completion_time'].isoformat() if isinstance(phase_data['completion_time'], datetime) else phase_data['completion_time']

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


class ResearchOrchestrator:
    """
    Master orchestrator for complete research workflows.
    """

    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.session_start = datetime.now()

        # Initialize components
        self.thermal_monitor = ThermalMonitor(config.max_temp_celsius, config.thermal_check_interval)
        self.workflow_executor = WorkflowExecutor(config, self.thermal_monitor)
        self.session_manager = SessionManager(config.session_file)

        # Control flags
        self.shutdown_requested = False

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

    def run_research_campaign(self, resume: bool = False) -> Dict[str, Any]:
        """Run complete research campaign."""
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

            logger.info("🎯 Starting research orchestration campaign")
            logger.info(f"📊 Conditions: {list(session.conditions_progress.keys())}")
            logger.info(f"🔄 Workflow phases: {self.config.workflow_phases}")

            # Main orchestration loop
            for condition in self.config.conditions:
                if self.shutdown_requested:
                    logger.info("🛑 Shutdown requested, saving progress...")
                    break

                try:
                    self._process_condition_workflow(condition, session)
                except Exception as e:
                    logger.error(f"💥 Error processing condition '{condition}': {e}")

                    condition_progress = session.conditions_progress[condition]
                    condition_progress.status = 'failed'
                    condition_progress.error_count += 1
                    session.global_statistics['failed_conditions'] += 1

                    if not self.config.continue_on_phase_failure:
                        raise

            # Generate final campaign report
            if self.config.generate_reports:
                self._generate_campaign_report(session)

            # Final session update
            session.is_active = False
            self.session_manager.save_session()

            return self._create_final_summary(session)

        except Exception as e:
            logger.error(f"💥 Critical orchestration error: {e}")
            logger.error(traceback.format_exc())

            if self.config.auto_restart_on_error:
                logger.info("🔄 Auto-restart triggered due to critical error")
                time.sleep(60)  # Brief pause before restart
                return self.run_research_campaign(resume=True)
            else:
                raise
        finally:
            # Stop auto-save
            if self.auto_save_thread:
                self.auto_save_stop.set()
                self.auto_save_thread.join(timeout=5)

    def _process_condition_workflow(self, condition: str, session: OrchestrationSession):
        """Process complete workflow for a single condition."""
        condition_progress = session.conditions_progress[condition]

        if condition_progress.status == 'completed':
            logger.info(f"✅ Condition '{condition}' already completed, skipping")
            return

        if condition_progress.status == 'failed':
            logger.info(f"❌ Condition '{condition}' marked as failed, skipping")
            return

        logger.info(f"🔬 Starting workflow for condition: {condition}")
        condition_progress.status = 'running'
        condition_progress.start_time = datetime.now()

        # Execute each workflow phase
        for phase_name in self.config.workflow_phases:
            if self.shutdown_requested:
                break

            phase_progress = condition_progress.phase_progress[phase_name]

            if phase_progress.status == 'completed':
                logger.info(f"✅ Phase '{phase_name}' already completed for '{condition}', skipping")
                continue

            if phase_progress.status == 'failed' and not self.config.continue_on_phase_failure:
                logger.error(f"❌ Phase '{phase_name}' previously failed for '{condition}', stopping workflow")
                break

            condition_progress.current_phase = phase_name
            logger.info(f"🚀 Executing phase '{phase_name}' for condition '{condition}'")

            try:
                # Execute phase with retry logic
                for attempt in range(self.config.max_retries_per_phase):
                    try:
                        result = self._execute_workflow_phase(phase_name, condition, phase_progress)

                        # Update statistics based on phase results
                        self._update_statistics_from_phase(session, condition, phase_name, result)
                        break

                    except Exception as e:
                        phase_progress.error_count += 1
                        phase_progress.last_error = str(e)

                        if attempt < self.config.max_retries_per_phase - 1:
                            logger.warning(f"⚠️ Phase '{phase_name}' attempt {attempt + 1} failed for '{condition}', retrying...")
                            time.sleep(30)  # Brief pause before retry
                        else:
                            logger.error(f"💥 Phase '{phase_name}' failed for '{condition}' after {self.config.max_retries_per_phase} attempts")
                            phase_progress.status = 'failed'
                            raise

            except Exception as e:
                logger.error(f"💥 Phase '{phase_name}' failed for condition '{condition}': {e}")
                if not self.config.continue_on_phase_failure:
                    condition_progress.status = 'failed'
                    raise

        # Mark condition as completed if all phases succeeded
        completed_phases = sum(1 for p in condition_progress.phase_progress.values() if p.status == 'completed')
        total_phases = len(condition_progress.phase_progress)

        if completed_phases == total_phases:
            condition_progress.status = 'completed'
            condition_progress.completion_time = datetime.now()
            session.global_statistics['completed_conditions'] += 1
            logger.info(f"🎉 Condition '{condition}' completed successfully!")
        elif not self.config.continue_on_phase_failure:
            condition_progress.status = 'failed'
            session.global_statistics['failed_conditions'] += 1

        self.session_manager.save_session()

    def _execute_workflow_phase(self, phase_name: str, condition: str, progress: PhaseProgress) -> Dict[str, Any]:
        """Execute a specific workflow phase."""
        phase_enum = WorkflowPhase(phase_name)

        if phase_enum == WorkflowPhase.COLLECTION:
            return self.workflow_executor.execute_collection_phase(condition, progress)
        elif phase_enum == WorkflowPhase.PROCESSING:
            return self.workflow_executor.execute_processing_phase(condition, progress)
        elif phase_enum == WorkflowPhase.VALIDATION:
            return self.workflow_executor.execute_validation_phase(condition, progress)
        elif phase_enum == WorkflowPhase.DATA_MINING:
            return self.workflow_executor.execute_data_mining_phase(condition, progress)
        elif phase_enum == WorkflowPhase.REPORTING:
            return self.workflow_executor.execute_reporting_phase(condition, progress)
        else:
            raise ValueError(f"Unknown workflow phase: {phase_name}")

    def _update_statistics_from_phase(self, session: OrchestrationSession, condition: str,
                                    phase_name: str, result: Dict[str, Any]):
        """Update global statistics based on phase results."""
        condition_progress = session.conditions_progress[condition]

        if phase_name == WorkflowPhase.COLLECTION.value:
            papers_collected = result.get('papers_collected', 0)
            condition_progress.papers_collected += papers_collected
            session.global_statistics['total_papers_collected'] += papers_collected

        elif phase_name == WorkflowPhase.PROCESSING.value:
            papers_processed = result.get('papers_processed', 0)
            interventions_extracted = result.get('interventions_extracted', 0)

            condition_progress.papers_processed += papers_processed
            condition_progress.interventions_extracted += interventions_extracted
            session.global_statistics['total_papers_processed'] += papers_processed
            session.global_statistics['total_interventions_extracted'] += interventions_extracted

    def _generate_campaign_report(self, session: OrchestrationSession):
        """Generate comprehensive campaign report."""
        logger.info("📋 Generating comprehensive campaign report...")

        # Calculate overall statistics
        total_duration = datetime.now() - self.session_start
        completed_conditions = [c for c in session.conditions_progress.values() if c.status == 'completed']
        failed_conditions = [c for c in session.conditions_progress.values() if c.status == 'failed']

        campaign_report = {
            'campaign_info': {
                'session_id': session.session_id,
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': str(total_duration),
                'restart_count': session.restart_count
            },
            'configuration': asdict(self.config),
            'global_statistics': session.global_statistics,
            'condition_summary': {
                'total_conditions': len(session.conditions_progress),
                'completed_conditions': len(completed_conditions),
                'failed_conditions': len(failed_conditions),
                'success_rate': len(completed_conditions) / len(session.conditions_progress) * 100 if session.conditions_progress else 0
            },
            'condition_details': {},
            'phase_performance': self._analyze_phase_performance(session)
        }

        # Add detailed condition information
        for condition, progress in session.conditions_progress.items():
            campaign_report['condition_details'][condition] = {
                'status': progress.status,
                'overall_progress_percent': progress.overall_progress_percent,
                'papers_collected': progress.papers_collected,
                'papers_processed': progress.papers_processed,
                'interventions_extracted': progress.interventions_extracted,
                'error_count': progress.error_count,
                'phase_details': {
                    name: {
                        'status': phase.status,
                        'duration': str(phase.duration) if phase.duration else None,
                        'progress_percent': phase.progress_percent,
                        'error_count': phase.error_count
                    }
                    for name, phase in progress.phase_progress.items()
                }
            }

        # Save campaign report
        report_file = self.config.output_dir / f"campaign_report_{session.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(campaign_report, f, indent=2, default=str)

        logger.info(f"📋 Campaign report saved to {report_file}")

    def _analyze_phase_performance(self, session: OrchestrationSession) -> Dict[str, Any]:
        """Analyze performance across all phases."""
        phase_stats = {}

        for phase_name in self.config.workflow_phases:
            phase_data = []

            for condition_progress in session.conditions_progress.values():
                if phase_name in condition_progress.phase_progress:
                    phase_progress = condition_progress.phase_progress[phase_name]
                    if phase_progress.duration:
                        phase_data.append(phase_progress.duration.total_seconds())

            if phase_data:
                phase_stats[phase_name] = {
                    'avg_duration_seconds': sum(phase_data) / len(phase_data),
                    'min_duration_seconds': min(phase_data),
                    'max_duration_seconds': max(phase_data),
                    'total_executions': len(phase_data)
                }

        return phase_stats

    def _create_final_summary(self, session: OrchestrationSession) -> Dict[str, Any]:
        """Create final campaign summary."""
        total_duration = datetime.now() - self.session_start

        return {
            'session_id': session.session_id,
            'total_duration': str(total_duration),
            'conditions_processed': len(session.conditions_progress),
            'successful_conditions': session.global_statistics['completed_conditions'],
            'failed_conditions': session.global_statistics['failed_conditions'],
            'total_papers_collected': session.global_statistics['total_papers_collected'],
            'total_papers_processed': session.global_statistics['total_papers_processed'],
            'total_interventions_extracted': session.global_statistics['total_interventions_extracted'],
            'success_rate': session.global_statistics['completed_conditions'] / len(session.conditions_progress) * 100 if session.conditions_progress else 0
        }

    def show_status(self) -> Dict[str, Any]:
        """Show current orchestration status."""
        session = self.session_manager.load_existing_session()
        if not session:
            return {"error": "No active session found"}

        # Get thermal status
        is_safe, temp = self.thermal_monitor.is_thermal_safe()

        status = {
            'session_id': session.session_id,
            'current_phase': session.current_phase,
            'is_active': session.is_active,
            'start_time': session.start_time,
            'global_statistics': session.global_statistics,
            'thermal_status': {
                'is_safe': is_safe,
                'temperature': temp,
                'max_temp': self.thermal_monitor.max_temp
            },
            'conditions': {}
        }

        for condition, progress in session.conditions_progress.items():
            status['conditions'][condition] = {
                'status': progress.status,
                'current_phase': progress.current_phase,
                'overall_progress': f"{progress.overall_progress_percent:.1f}%",
                'papers_collected': progress.papers_collected,
                'papers_processed': progress.papers_processed,
                'interventions_extracted': progress.interventions_extracted,
                'error_count': progress.error_count
            }

        return status


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Research Orchestrator for Medical Research Workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Basic options
    parser.add_argument('search_term', nargs='?',
                       help='Single condition to process (alternative to --conditions)')
    parser.add_argument('--conditions', type=str,
                       help='Comma-separated list of conditions to process')
    parser.add_argument('--papers', type=int, default=500,
                       help='Papers per condition (default: 500)')
    parser.add_argument('--papers-per-condition', type=int,
                       help='Papers per condition (alias for --papers)')

    # Date filtering
    parser.add_argument('--min-year', type=int, default=2015,
                       help='Minimum publication year (default: 2015)')
    parser.add_argument('--max-year', type=int,
                       help='Maximum publication year')

    # Workflow control
    parser.add_argument('--collection-only', action='store_true',
                       help='Run collection phase only')
    parser.add_argument('--processing-only', action='store_true',
                       help='Run processing phase only')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation phase')
    parser.add_argument('--phases', type=str,
                       help='Comma-separated list of phases to run')

    # Collection options
    parser.add_argument('--collection-batch-size', type=int, default=50,
                       help='Papers collected per batch (default: 50)')
    parser.add_argument('--traditional-mode', action='store_true',
                       help='Use traditional collection mode')
    parser.add_argument('--skip-s2', action='store_true',
                       help='Skip Semantic Scholar enrichment')

    # Processing options
    parser.add_argument('--processing-batch-size', type=int, default=5,
                       help='Papers processed per batch (default: 5)')
    parser.add_argument('--processing-limit', type=int,
                       help='Limit papers processed per condition')
    parser.add_argument('--single-model', action='store_true',
                       help='Use single model instead of dual-model analysis')

    # Thermal protection
    parser.add_argument('--max-temp', type=float, default=80.0,
                       help='Maximum GPU temperature (°C, default: 80)')
    parser.add_argument('--cooling-temp', type=float, default=70.0,
                       help='Temperature to resume after cooling (°C, default: 70)')

    # Session management
    parser.add_argument('--resume', action='store_true',
                       help='Resume previous session')
    parser.add_argument('--session-file', type=str, default='orchestration_session.json',
                       help='Session state file')

    # Output options
    parser.add_argument('--output-dir', type=str, default='orchestration_results',
                       help='Output directory for results')
    parser.add_argument('--no-reports', action='store_true',
                       help='Skip report generation')

    # Operational modes
    parser.add_argument('--overnight', action='store_true',
                       help='Enable overnight operation mode')
    parser.add_argument('--no-auto-restart', action='store_true',
                       help='Disable auto-restart on errors')
    parser.add_argument('--continue-on-failure', action='store_true',
                       help='Continue processing other conditions if one fails')

    # Utility options
    parser.add_argument('--status', action='store_true',
                       help='Show current orchestration status')
    parser.add_argument('--thermal-monitor', action='store_true',
                       help='Show thermal monitoring status')
    parser.add_argument('--config', type=str,
                       help='Load configuration from JSON file')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        # Startup cleanup of old temporary files
        cleanup_old_files()

        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            config = OrchestrationConfig(**config_data)
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

            papers_per_condition = args.papers_per_condition or args.papers

            config = OrchestrationConfig(
                conditions=conditions,
                papers_per_condition=papers_per_condition,
                min_year=args.min_year,
                max_year=args.max_year,
                collection_only=args.collection_only,
                processing_only=args.processing_only,
                skip_validation=args.skip_validation,
                workflow_phases=args.phases.split(',') if args.phases else None,
                collection_batch_size=args.collection_batch_size,
                use_interleaved_s2=not args.skip_s2,
                traditional_mode=args.traditional_mode,
                processing_batch_size=args.processing_batch_size,
                processing_limit=args.processing_limit,
                use_dual_models=not args.single_model,
                max_temp_celsius=args.max_temp,
                cooling_temp_celsius=args.cooling_temp,
                session_file=Path(args.session_file),
                output_dir=Path(args.output_dir),
                generate_reports=not args.no_reports,
                overnight_mode=args.overnight,
                auto_restart_on_error=not args.no_auto_restart,
                continue_on_phase_failure=args.continue_on_failure
            )

        # Initialize orchestrator
        orchestrator = ResearchOrchestrator(config)

        # Handle different run modes
        if args.status:
            # Show status and exit
            status = orchestrator.show_status()
            if 'error' in status:
                print(status['error'])
            else:
                print("Current orchestration status:")
                print(f"Session: {status['session_id']}")
                print(f"Current phase: {status['current_phase']}")
                print(f"Active: {status['is_active']}")
                print(f"Papers collected: {status['global_statistics']['total_papers_collected']}")
                print(f"Papers processed: {status['global_statistics']['total_papers_processed']}")
                print(f"Interventions extracted: {status['global_statistics']['total_interventions_extracted']}")
                print(f"GPU temp: {status['thermal_status']['temperature']}°C (safe: {status['thermal_status']['is_safe']})")
                print("\nCondition progress:")
                for condition, progress in status['conditions'].items():
                    print(f"  {condition}: {progress['status']} - {progress['overall_progress']} (phase: {progress['current_phase']})")
            return True

        if args.thermal_monitor:
            # Show thermal status and exit
            is_safe, temp = orchestrator.thermal_monitor.is_thermal_safe()
            print(f"Thermal status: {temp}°C (max: {orchestrator.thermal_monitor.max_temp}°C)")
            print(f"Status: {'Safe' if is_safe else 'Warning - Too Hot'}")
            return True

        # Run orchestration
        logger.info("🚀 Starting research orchestration")
        summary = orchestrator.run_research_campaign(resume=args.resume)

        # Print summary
        print(f"\n=== Research Campaign Complete ===")
        print(f"Session: {summary['session_id']}")
        print(f"Duration: {summary['total_duration']}")
        print(f"Conditions: {summary['successful_conditions']}/{summary['conditions_processed']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Papers collected: {summary['total_papers_collected']}")
        print(f"Papers processed: {summary['total_papers_processed']}")
        print(f"Interventions extracted: {summary['total_interventions_extracted']}")

        return True

    except KeyboardInterrupt:
        logger.info("Orchestration interrupted by user")
        print("\nOrchestration interrupted. Session state saved.")
        return True
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        logger.error(traceback.format_exc())
        print(f"Orchestration failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)