"""
Phase 5 Base Exporter

Base class for all Phase 5 export operations with shared functionality.
"""

import sqlite3
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from back_end.src.data.config import config, setup_logging
from .phase_5_export_operations import (
    atomic_write_json,
    validate_export_data,
    get_file_size_mb,
    format_timestamp
)

logger = setup_logging(__name__, 'phase_5_base_exporter.log')


class BaseExporter(ABC):
    """
    Base class for all Phase 5 exporters.

    Provides:
    - Database connection management
    - Configuration loading
    - Path resolution
    - Atomic file writes
    - Validation
    - Statistics tracking
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        config_path: Optional[str] = None,
        export_type: str = "unknown"
    ):
        """
        Initialize base exporter.

        Args:
            db_path: Path to intervention_research.db (defaults to config.db_path)
            config_path: Path to phase_5_config.yaml
            export_type: Type of export (e.g., 'table_view', 'network_viz')
        """
        self.db_path = Path(db_path) if db_path else Path(config.db_path)
        self.export_type = export_type
        self.project_root = Path(__file__).parent.parent.parent.parent

        # Load configuration
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path(__file__).parent / "phase_5_config.yaml"

        self.export_config = self._load_config()

        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0.0,
            'records_processed': 0,
            'file_size_mb': 0.0,
            'validation_passed': False
        }

        logger.info(f"Initialized {export_type} exporter")

    def _load_config(self) -> Dict[str, Any]:
        """Load Phase 5 configuration from YAML."""
        try:
            with open(self.config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            logger.debug(f"Loaded config from {self.config_path}")
            return cfg
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            # Return default config
            return {
                'export_options': {
                    'pretty_print': True,
                    'atomic_writes': True,
                    'backup_previous': True
                },
                'validation': {
                    'enabled': True,
                    'fail_on_validation_error': False
                }
            }

    def get_database_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def resolve_output_path(self, config_key: str) -> Path:
        """
        Resolve output path from config relative to project root.

        Args:
            config_key: Key in output_paths config section

        Returns:
            Absolute path to output file
        """
        try:
            relative_path = self.export_config['output_paths'][config_key]
            absolute_path = self.project_root / relative_path
            return absolute_path
        except KeyError:
            logger.error(f"Output path not found in config: {config_key}")
            raise

    @abstractmethod
    def extract_data(self) -> Dict[str, Any]:
        """
        Extract data from database.

        Must be implemented by subclasses.

        Returns:
            Dictionary with extracted data
        """
        pass

    @abstractmethod
    def transform_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw data into export format.

        Must be implemented by subclasses.

        Args:
            raw_data: Raw data from extract_data()

        Returns:
            Transformed data ready for export
        """
        pass

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate exported data.

        Args:
            data: Data to validate

        Returns:
            Validation result dictionary
        """
        if not self.export_config.get('validation', {}).get('enabled', True):
            return {'valid': True, 'warnings': ['Validation disabled']}

        return validate_export_data(data, self.export_type)

    def write_output(self, data: Dict[str, Any], output_path: Path) -> bool:
        """
        Write data to output file using atomic writes.

        Args:
            data: Data to write
            output_path: Target file path

        Returns:
            True if successful
        """
        options = self.export_config.get('export_options', {})

        return atomic_write_json(
            data=data,
            filepath=output_path,
            pretty_print=options.get('pretty_print', True),
            backup_existing=options.get('backup_previous', True)
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute complete export pipeline.

        Returns:
            Dictionary with export results
        """
        self.stats['start_time'] = datetime.now()
        logger.info(f"Starting {self.export_type} export...")

        try:
            # Step 1: Extract data from database
            logger.info("Step 1: Extracting data from database...")
            raw_data = self.extract_data()
            self.stats['records_processed'] = self._count_records(raw_data)

            # Step 2: Transform data for export
            logger.info("Step 2: Transforming data...")
            transformed_data = self.transform_data(raw_data)

            # Step 3: Validate data
            logger.info("Step 3: Validating data...")
            validation_result = self.validate(transformed_data)
            self.stats['validation_passed'] = validation_result['valid']

            if not validation_result['valid']:
                logger.warning(f"Validation failed: {validation_result['errors']}")
                if self.export_config.get('validation', {}).get('fail_on_validation_error', False):
                    raise ValueError(f"Validation failed: {validation_result['errors']}")

            if validation_result.get('warnings'):
                logger.warning(f"Validation warnings: {validation_result['warnings']}")

            # Step 4: Determine output path
            output_path = self._get_output_path()
            logger.info(f"Step 4: Writing to {output_path}...")

            # Step 5: Write output file
            success = self.write_output(transformed_data, output_path)

            if not success:
                raise RuntimeError(f"Failed to write output file: {output_path}")

            # Step 6: Calculate statistics
            self.stats['file_size_mb'] = get_file_size_mb(output_path)
            self.stats['end_time'] = datetime.now()
            self.stats['duration_seconds'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

            logger.info(f"[SUCCESS] {self.export_type} export completed")
            logger.info(f"  Records: {self.stats['records_processed']}")
            logger.info(f"  File size: {self.stats['file_size_mb']:.2f} MB")
            logger.info(f"  Duration: {self.stats['duration_seconds']:.1f}s")

            return {
                'success': True,
                'export_type': self.export_type,
                'output_path': str(output_path),
                'statistics': self.stats,
                'validation': validation_result
            }

        except Exception as e:
            logger.error(f"{self.export_type} export failed: {e}")
            self.stats['end_time'] = datetime.now()
            self.stats['duration_seconds'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

            return {
                'success': False,
                'export_type': self.export_type,
                'error': str(e),
                'statistics': self.stats
            }

    @abstractmethod
    def _get_output_path(self) -> Path:
        """
        Get output path for this exporter.

        Must be implemented by subclasses.

        Returns:
            Path to output file
        """
        pass

    def _count_records(self, data: Dict[str, Any]) -> int:
        """
        Count records in extracted data.

        Override in subclasses for specific counting logic.

        Args:
            data: Raw extracted data

        Returns:
            Number of records processed
        """
        # Default implementation - override in subclasses
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            # Try common keys
            for key in ['interventions', 'nodes', 'clusters', 'items']:
                if key in data and isinstance(data[key], list):
                    return len(data[key])
        return 0
