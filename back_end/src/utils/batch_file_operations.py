"""
Batch File Operations Utility for Performance Optimization

Provides batched file operations to reduce filesystem overhead during
high-throughput paper collection and processing.
"""

import os
import time
from pathlib import Path
from typing import List, Set
from threading import Lock
from back_end.src.data.config import config, setup_logging

logger = setup_logging(__name__)


class BatchFileManager:
    """Manages batched file operations for performance optimization."""

    def __init__(self, batch_size: int = 10):
        """
        Initialize batch file manager.

        Args:
            batch_size: Number of files to batch before executing operations
        """
        self.batch_size = batch_size
        self.files_to_delete: Set[Path] = set()
        self.lock = Lock()

    def queue_for_deletion(self, file_path: Path):
        """Queue a file for batch deletion."""
        with self.lock:
            self.files_to_delete.add(Path(file_path))

            # Execute batch deletion when we reach batch size
            if len(self.files_to_delete) >= self.batch_size:
                self._execute_batch_deletion()

    def _execute_batch_deletion(self):
        """Execute batched file deletion."""
        if not self.files_to_delete:
            return

        deleted_count = 0
        failed_count = 0

        # Create a copy to avoid modifying set during iteration
        files_to_process = list(self.files_to_delete)
        self.files_to_delete.clear()

        for file_path in files_to_process:
            try:
                if file_path.exists():
                    file_path.unlink()
                    deleted_count += 1
            except Exception as e:
                failed_count += 1
                if not config.fast_mode:  # Only log errors in non-fast mode
                    logger.error(f"Failed to delete {file_path}: {e}")

        # Only log summary in non-fast mode
        if not config.fast_mode and (deleted_count > 0 or failed_count > 0):
            logger.info(f"Batch deletion: {deleted_count} deleted, {failed_count} failed")

    def flush(self):
        """Force execution of all pending operations."""
        with self.lock:
            if self.files_to_delete:
                self._execute_batch_deletion()

    def cleanup_xml_files_by_pattern(self, pattern: str):
        """Queue XML files matching pattern for deletion."""
        try:
            xml_files = list(config.metadata_dir.glob(pattern))
            for xml_file in xml_files:
                self.queue_for_deletion(xml_file)
        except Exception as e:
            if not config.fast_mode:
                logger.error(f"Error queuing XML files for deletion: {e}")

    def cleanup_old_session_files(self, max_age_days: int = 7):
        """Queue old session files for deletion."""
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

            # Session files in project root
            session_patterns = ["*session*.json", "*.log"]

            for pattern in session_patterns:
                for session_file in config.project_root.glob(pattern):
                    try:
                        if session_file.stat().st_mtime < cutoff_time:
                            self.queue_for_deletion(session_file)
                    except Exception:
                        continue  # Skip files we can't stat

        except Exception as e:
            if not config.fast_mode:
                logger.error(f"Error queuing session files for deletion: {e}")

    def cleanup_old_mapping_files(self, max_age_days: int = 7):
        """Queue old mapping suggestion CSV files for deletion."""
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

            # Mapping suggestion file patterns
            mapping_patterns = [
                "mapping_suggestions_all_*.csv",
                "mapping_suggestions_high_confidence_*.csv",
                "mapping_suggestions_interventions_*.csv",
                "mapping_suggestions_conditions_*.csv",
                "mapping_suggestions_llm_enhanced_*.csv"
            ]

            for pattern in mapping_patterns:
                for mapping_file in config.project_root.glob(pattern):
                    try:
                        if mapping_file.stat().st_mtime < cutoff_time:
                            self.queue_for_deletion(mapping_file)
                    except Exception:
                        continue  # Skip files we can't stat

        except Exception as e:
            if not config.fast_mode:
                logger.error(f"Error queuing mapping files for deletion: {e}")


# Global batch file manager instance
batch_file_manager = BatchFileManager(batch_size=10)


# Convenience functions
def queue_file_for_deletion(file_path: Path):
    """Queue a file for batch deletion."""
    batch_file_manager.queue_for_deletion(file_path)


def cleanup_xml_files_for_papers(pmids: List[str]):
    """Queue XML files for specific papers for deletion."""
    for pmid in pmids:
        pattern = f"*{pmid}*.xml"
        batch_file_manager.cleanup_xml_files_by_pattern(pattern)


def flush_pending_operations():
    """Force execution of all pending file operations."""
    batch_file_manager.flush()


def cleanup_old_mapping_files():
    """Clean up old mapping suggestion CSV files."""
    batch_file_manager.cleanup_old_mapping_files()


def cleanup_old_files():
    """Clean up old session and temporary files."""
    batch_file_manager.cleanup_old_session_files()
    batch_file_manager.cleanup_old_mapping_files()