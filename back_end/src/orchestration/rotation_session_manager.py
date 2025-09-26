#!/usr/bin/env python3
"""
Medical Rotation Pipeline Session Manager

Manages session state persistence for the rotating medical condition pipeline.
Handles resumable execution across interruptions and tracks progress through
the complete rotation of medical specialties and conditions.

Features:
- Session state persistence to JSON
- Resumable pipeline execution
- Progress tracking across conditions
- Interruption handling at paper/processing level
- Circular iteration through medical specialties
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from ..data.config import config, setup_logging
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging

logger = setup_logging(__name__, 'rotation_session_manager.log')


class PipelinePhase(Enum):
    """Pipeline phases for interruption handling."""
    COLLECTION = "collection"
    LLM_PROCESSING = "llm_processing"
    DEDUPLICATION = "deduplication"
    COMPLETED = "completed"


@dataclass
class InterruptionState:
    """State information for resuming after interruption."""
    phase: str
    paper_id: Optional[str] = None
    paper_index: Optional[int] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ConditionProgress:
    """Progress tracking for a single condition."""
    specialty: str
    condition: str
    papers_collected: int = 0
    papers_processed: int = 0
    interventions_extracted: int = 0
    duplicates_removed: int = 0
    start_time: Optional[str] = None
    completion_time: Optional[str] = None
    status: str = "pending"  # pending, active, completed, failed
    error_count: int = 0
    last_error: Optional[str] = None

    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate duration in minutes."""
        if self.start_time and self.completion_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.completion_time)
            return (end - start).total_seconds() / 60
        return None

    @property
    def is_complete(self) -> bool:
        """Check if condition processing is complete."""
        return self.status == "completed"


@dataclass
class RotationSession:
    """Complete rotation session state."""
    session_id: str
    start_time: str
    last_update: str

    # Rotation state
    current_specialty_index: int = 0
    current_condition_index: int = 0
    iteration_count: int = 1
    papers_per_condition: int = 10

    # Progress tracking
    total_papers_collected: int = 0
    total_papers_processed: int = 0
    total_interventions_extracted: int = 0
    total_duplicates_removed: int = 0

    # Current state
    current_condition_progress: Optional[ConditionProgress] = None
    interruption_state: Optional[InterruptionState] = None

    # History
    completed_conditions: List[ConditionProgress] = None
    failed_conditions: List[ConditionProgress] = None

    # Session info
    is_active: bool = True
    auto_restart_count: int = 0

    def __post_init__(self):
        if self.completed_conditions is None:
            self.completed_conditions = []
        if self.failed_conditions is None:
            self.failed_conditions = []

    @property
    def current_specialty(self) -> str:
        """Get current specialty name."""
        specialties = list(config.medical_specialties.keys())
        if 0 <= self.current_specialty_index < len(specialties):
            return specialties[self.current_specialty_index]
        return specialties[0]  # Fallback to first specialty

    @property
    def current_condition(self) -> str:
        """Get current condition name."""
        specialty = self.current_specialty
        conditions = config.medical_specialties[specialty]
        if 0 <= self.current_condition_index < len(conditions):
            return conditions[self.current_condition_index]
        return conditions[0]  # Fallback to first condition

    @property
    def total_conditions(self) -> int:
        """Total number of conditions across all specialties."""
        return sum(len(conditions) for conditions in config.medical_specialties.values())

    @property
    def completed_conditions_count(self) -> int:
        """Number of conditions completed in current iteration."""
        return len(self.completed_conditions)

    @property
    def progress_percent(self) -> float:
        """Overall progress percentage for current iteration."""
        if self.total_conditions == 0:
            return 0.0
        return (self.completed_conditions_count / self.total_conditions) * 100

    @property
    def estimated_completion_time(self) -> Optional[str]:
        """Estimate completion time based on current progress."""
        if len(self.completed_conditions) < 2:
            return None

        # Calculate average time per condition
        durations = [c.duration_minutes for c in self.completed_conditions if c.duration_minutes]
        if not durations:
            return None

        avg_duration_minutes = sum(durations) / len(durations)
        remaining_conditions = self.total_conditions - self.completed_conditions_count
        remaining_minutes = remaining_conditions * avg_duration_minutes

        completion_time = datetime.now() + timedelta(minutes=remaining_minutes)
        return completion_time.isoformat()


class RotationSessionManager:
    """Manages session persistence and state for the medical rotation pipeline."""

    def __init__(self, session_file: Path = None):
        """Initialize session manager."""
        self.session_file = session_file or (config.processed_data / "rotation_session.json")
        self.session: Optional[RotationSession] = None
        self.save_lock = threading.Lock()

        # Ensure session directory exists
        self.session_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Session manager initialized with file: {self.session_file}")

    def create_new_session(self, papers_per_condition: int = 10) -> RotationSession:
        """Create a new rotation session."""
        session_id = f"rotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = RotationSession(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
            papers_per_condition=papers_per_condition
        )

        self.session = session
        self.save_session()

        logger.info(f"Created new rotation session: {session_id}")
        logger.info(f"Target: {papers_per_condition} papers per condition")
        logger.info(f"Total conditions: {session.total_conditions}")

        return session

    def load_existing_session(self) -> Optional[RotationSession]:
        """Load existing session from file."""
        if not self.session_file.exists():
            logger.info("No existing session file found")
            return None

        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct ConditionProgress objects
            completed_conditions = []
            for condition_data in data.get('completed_conditions', []):
                completed_conditions.append(ConditionProgress(**condition_data))

            failed_conditions = []
            for condition_data in data.get('failed_conditions', []):
                failed_conditions.append(ConditionProgress(**condition_data))

            # Reconstruct current condition progress
            current_condition_progress = None
            if data.get('current_condition_progress'):
                current_condition_progress = ConditionProgress(**data['current_condition_progress'])

            # Reconstruct interruption state
            interruption_state = None
            if data.get('interruption_state'):
                interruption_state = InterruptionState(**data['interruption_state'])

            # Create session object
            session_data = data.copy()
            session_data['completed_conditions'] = completed_conditions
            session_data['failed_conditions'] = failed_conditions
            session_data['current_condition_progress'] = current_condition_progress
            session_data['interruption_state'] = interruption_state

            session = RotationSession(**session_data)
            self.session = session

            logger.info(f"Loaded existing session: {session.session_id}")
            logger.info(f"Progress: {session.completed_conditions_count}/{session.total_conditions} conditions")
            logger.info(f"Current: {session.current_specialty} -> {session.current_condition}")

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
                # Update last update time
                self.session.last_update = datetime.now().isoformat()

                # Convert to dict and save
                session_dict = asdict(self.session)

                # Create backup of existing session
                if self.session_file.exists():
                    backup_file = self.session_file.with_suffix('.json.bak')
                    backup_file.write_text(self.session_file.read_text(encoding='utf-8'), encoding='utf-8')

                # Save new session
                with open(self.session_file, 'w', encoding='utf-8') as f:
                    json.dump(session_dict, f, indent=2, ensure_ascii=False)

                logger.debug(f"Session saved: {self.session.session_id}")

            except Exception as e:
                logger.error(f"Failed to save session: {e}")

    def advance_to_next_condition(self) -> Tuple[str, str]:
        """Advance to the next condition in rotation."""
        if not self.session:
            raise RuntimeError("No active session")

        # Complete current condition if active
        if self.session.current_condition_progress:
            if self.session.current_condition_progress.status == "active":
                self.session.current_condition_progress.status = "completed"
                self.session.current_condition_progress.completion_time = datetime.now().isoformat()
                self.session.completed_conditions.append(self.session.current_condition_progress)

        # Clear interruption state
        self.session.interruption_state = None

        # Advance condition index
        specialty = self.session.current_specialty
        conditions = config.medical_specialties[specialty]

        self.session.current_condition_index += 1

        # Check if we need to advance to next specialty
        if self.session.current_condition_index >= len(conditions):
            self.session.current_condition_index = 0
            self.session.current_specialty_index += 1

            # Check if we completed full rotation
            specialties = list(config.medical_specialties.keys())
            if self.session.current_specialty_index >= len(specialties):
                self.session.current_specialty_index = 0
                self.session.iteration_count += 1
                logger.info(f"🔄 Completed rotation iteration {self.session.iteration_count - 1}")
                logger.info(f"Starting iteration {self.session.iteration_count}")

        # Initialize progress for new condition
        new_specialty = self.session.current_specialty
        new_condition = self.session.current_condition

        self.session.current_condition_progress = ConditionProgress(
            specialty=new_specialty,
            condition=new_condition,
            start_time=datetime.now().isoformat(),
            status="active"
        )

        self.save_session()

        logger.info(f"Advanced to: {new_specialty} -> {new_condition}")
        return new_specialty, new_condition

    def mark_condition_failed(self, error_message: str):
        """Mark current condition as failed."""
        if not self.session or not self.session.current_condition_progress:
            return

        self.session.current_condition_progress.status = "failed"
        self.session.current_condition_progress.completion_time = datetime.now().isoformat()
        self.session.current_condition_progress.last_error = error_message
        self.session.current_condition_progress.error_count += 1

        self.session.failed_conditions.append(self.session.current_condition_progress)

        logger.error(f"Condition failed: {self.session.current_condition} - {error_message}")
        self.save_session()

    def set_interruption_state(self, phase: PipelinePhase, paper_id: str = None,
                             paper_index: int = None, error_message: str = None):
        """Set interruption state for resumable execution."""
        if not self.session:
            return

        self.session.interruption_state = InterruptionState(
            phase=phase.value,
            paper_id=paper_id,
            paper_index=paper_index,
            error_message=error_message
        )

        logger.info(f"Interruption state set: {phase.value}")
        if paper_id:
            logger.info(f"Paper ID: {paper_id}")
        if paper_index is not None:
            logger.info(f"Paper index: {paper_index}")

        self.save_session()

    def clear_interruption_state(self):
        """Clear interruption state after successful recovery."""
        if not self.session:
            return

        self.session.interruption_state = None
        self.save_session()
        logger.info("Interruption state cleared")

    def update_progress(self, papers_collected: int = 0, papers_processed: int = 0,
                       interventions_extracted: int = 0, duplicates_removed: int = 0):
        """Update progress counters."""
        if not self.session:
            return

        # Update session totals
        self.session.total_papers_collected += papers_collected
        self.session.total_papers_processed += papers_processed
        self.session.total_interventions_extracted += interventions_extracted
        self.session.total_duplicates_removed += duplicates_removed

        # Update current condition progress
        if self.session.current_condition_progress:
            self.session.current_condition_progress.papers_collected += papers_collected
            self.session.current_condition_progress.papers_processed += papers_processed
            self.session.current_condition_progress.interventions_extracted += interventions_extracted
            self.session.current_condition_progress.duplicates_removed += duplicates_removed

        self.save_session()

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive session status."""
        if not self.session:
            return {"error": "No active session"}

        return {
            "session_id": self.session.session_id,
            "iteration_count": self.session.iteration_count,
            "current_specialty": self.session.current_specialty,
            "current_condition": self.session.current_condition,
            "progress_percent": self.session.progress_percent,
            "completed_conditions": self.session.completed_conditions_count,
            "total_conditions": self.session.total_conditions,
            "failed_conditions": len(self.session.failed_conditions),
            "papers_collected": self.session.total_papers_collected,
            "papers_processed": self.session.total_papers_processed,
            "interventions_extracted": self.session.total_interventions_extracted,
            "duplicates_removed": self.session.total_duplicates_removed,
            "estimated_completion": self.session.estimated_completion_time,
            "has_interruption": self.session.interruption_state is not None,
            "is_active": self.session.is_active
        }

    def export_session_report(self, output_file: Path = None) -> str:
        """Export comprehensive session report."""
        if not self.session:
            return "No active session to export"

        output_file = output_file or (config.processed_data / f"rotation_report_{self.session.session_id}.json")

        report = {
            "session_info": {
                "session_id": self.session.session_id,
                "start_time": self.session.start_time,
                "last_update": self.session.last_update,
                "iteration_count": self.session.iteration_count,
                "is_active": self.session.is_active
            },
            "progress_summary": {
                "completed_conditions": self.session.completed_conditions_count,
                "failed_conditions": len(self.session.failed_conditions),
                "total_conditions": self.session.total_conditions,
                "progress_percent": self.session.progress_percent,
                "papers_collected": self.session.total_papers_collected,
                "papers_processed": self.session.total_papers_processed,
                "interventions_extracted": self.session.total_interventions_extracted,
                "duplicates_removed": self.session.total_duplicates_removed
            },
            "current_state": {
                "specialty": self.session.current_specialty,
                "condition": self.session.current_condition,
                "current_progress": asdict(self.session.current_condition_progress) if self.session.current_condition_progress else None,
                "interruption_state": asdict(self.session.interruption_state) if self.session.interruption_state else None
            },
            "completed_conditions": [asdict(c) for c in self.session.completed_conditions],
            "failed_conditions": [asdict(c) for c in self.session.failed_conditions]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Session report exported to: {output_file}")
        return str(output_file)


# Global session manager instance
session_manager = RotationSessionManager()