"""
Batch pipeline session management.

Handles session state persistence, recovery, and tracking.
"""

import json
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from .batch_config import BatchPhase
from back_end.src.data.config import config, setup_logging

logger = setup_logging(__name__, 'batch_medical_rotation.log')

# Platform-specific file locking
if platform.system() == 'Windows':
    import msvcrt
else:
    import fcntl


@dataclass
class BatchSession:
    """Session state for batch pipeline."""
    session_id: str
    papers_per_condition: int
    current_phase: BatchPhase
    iteration_number: int
    start_time: str

    # Phase completion tracking
    collection_completed: bool = False
    processing_completed: bool = False
    semantic_normalization_completed: bool = False
    group_categorization_completed: bool = False
    mechanism_clustering_completed: bool = False
    data_mining_completed: bool = False
    frontend_export_completed: bool = False

    # Statistics (current iteration)
    total_papers_collected: int = 0
    total_papers_processed: int = 0
    total_interventions_extracted: int = 0
    total_canonical_groups_created: int = 0
    total_groups_categorized: int = 0
    total_interventions_categorized: int = 0
    total_orphans_categorized: int = 0
    total_mechanisms_processed: int = 0
    total_mechanism_clusters: int = 0
    total_knowledge_graph_nodes: int = 0
    total_knowledge_graph_edges: int = 0
    total_bayesian_scores: int = 0
    total_files_exported: int = 0

    # Continuous mode settings
    continuous_mode: bool = False
    max_iterations: Optional[int] = None
    iteration_delay_seconds: float = 60.0

    # Iteration history
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)

    # Phase results
    collection_result: Optional[Dict[str, Any]] = None
    processing_result: Optional[Dict[str, Any]] = None
    semantic_normalization_result: Optional[Dict[str, Any]] = None
    group_categorization_result: Optional[Dict[str, Any]] = None
    mechanism_clustering_result: Optional[Dict[str, Any]] = None
    data_mining_result: Optional[Dict[str, Any]] = None
    frontend_export_result: Optional[Dict[str, Any]] = None

    def is_completed(self) -> bool:
        """Check if entire pipeline is completed."""
        return self.current_phase == BatchPhase.COMPLETED

    def reset_for_next_iteration(self):
        """Reset session state for next iteration in continuous mode."""
        self.iteration_number += 1
        self.current_phase = BatchPhase.COLLECTION

        # Reset completion flags
        self.collection_completed = False
        self.processing_completed = False
        self.semantic_normalization_completed = False
        self.group_categorization_completed = False
        self.mechanism_clustering_completed = False
        self.data_mining_completed = False
        self.frontend_export_completed = False

        # Reset iteration statistics
        self.total_papers_collected = 0
        self.total_papers_processed = 0
        self.total_interventions_extracted = 0
        self.total_canonical_groups_created = 0
        self.total_groups_categorized = 0
        self.total_interventions_categorized = 0
        self.total_orphans_categorized = 0
        self.total_mechanisms_processed = 0
        self.total_mechanism_clusters = 0
        self.total_knowledge_graph_nodes = 0
        self.total_knowledge_graph_edges = 0
        self.total_bayesian_scores = 0
        self.total_files_exported = 0

        # Clear phase results
        self.collection_result = None
        self.processing_result = None
        self.semantic_normalization_result = None
        self.group_categorization_result = None
        self.mechanism_clustering_result = None
        self.data_mining_result = None
        self.frontend_export_result = None

    def save_iteration_summary(self, iteration_time: float):
        """Save current iteration stats to history."""
        iteration_summary = {
            'iteration_number': self.iteration_number,
            'completion_time': datetime.now().isoformat(),
            'iteration_duration_seconds': iteration_time,
            'papers_collected': self.total_papers_collected,
            'papers_processed': self.total_papers_processed,
            'interventions_extracted': self.total_interventions_extracted,
            'canonical_groups_created': self.total_canonical_groups_created,
            'groups_categorized': self.total_groups_categorized,
            'interventions_categorized': self.total_interventions_categorized,
            'orphans_categorized': self.total_orphans_categorized,
            'mechanisms_processed': self.total_mechanisms_processed,
            'mechanism_clusters_created': self.total_mechanism_clusters,
            'knowledge_graph_nodes': self.total_knowledge_graph_nodes,
            'knowledge_graph_edges': self.total_knowledge_graph_edges,
            'bayesian_scores_generated': self.total_bayesian_scores,
            'files_exported': self.total_files_exported
        }
        self.iteration_history.append(iteration_summary)


class SessionManager:
    """Manages session persistence and recovery."""

    def __init__(self, session_file: Optional[Path] = None):
        """
        Initialize session manager.

        Args:
            session_file: Path to session file (defaults to config.data_root / "batch_session.json")
        """
        self.session_file = session_file or Path(config.data_root) / "batch_session.json"

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
                data_mining_completed=data.get('data_mining_completed', False),
                frontend_export_completed=data.get('frontend_export_completed', False),
                total_papers_collected=data.get('total_papers_collected', 0),
                total_papers_processed=data.get('total_papers_processed', 0),
                total_interventions_extracted=data.get('total_interventions_extracted', 0),
                total_canonical_groups_created=data.get('total_canonical_groups_created', 0),
                total_groups_categorized=data.get('total_groups_categorized', 0),
                total_interventions_categorized=data.get('total_interventions_categorized', 0),
                total_orphans_categorized=data.get('total_orphans_categorized', 0),
                total_mechanisms_processed=data.get('total_mechanisms_processed', 0),
                total_mechanism_clusters=data.get('total_mechanism_clusters', 0),
                total_knowledge_graph_nodes=data.get('total_knowledge_graph_nodes', 0),
                total_knowledge_graph_edges=data.get('total_knowledge_graph_edges', 0),
                total_bayesian_scores=data.get('total_bayesian_scores', 0),
                total_files_exported=data.get('total_files_exported', 0),
                continuous_mode=data.get('continuous_mode', False),
                max_iterations=data.get('max_iterations'),
                iteration_delay_seconds=data.get('iteration_delay_seconds', 60.0),
                iteration_history=data.get('iteration_history', []),
                collection_result=data.get('collection_result'),
                processing_result=data.get('processing_result'),
                semantic_normalization_result=data.get('semantic_normalization_result'),
                group_categorization_result=data.get('group_categorization_result'),
                mechanism_clustering_result=data.get('mechanism_clustering_result'),
                data_mining_result=data.get('data_mining_result'),
                frontend_export_result=data.get('frontend_export_result')
            )

            logger.info(f"Loaded existing session: {session.session_id}")
            if session.continuous_mode:
                logger.info(f"Continuous mode: max_iterations={session.max_iterations or 'unlimited'}, iteration={session.iteration_number}")

            return session

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None

    def save_session(self, session: BatchSession):
        """
        Save session to file with platform-specific file locking.

        Uses msvcrt on Windows and fcntl on Unix/Linux to prevent race conditions.
        """
        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert session to dictionary
            data = asdict(session)
            # Convert Enum to value
            data['current_phase'] = session.current_phase.value

            # Write with platform-specific file locking
            with open(self.session_file, 'w') as f:
                try:
                    # Acquire exclusive lock
                    if platform.system() == 'Windows':
                        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    else:
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
