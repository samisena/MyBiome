"""
Phase 5 Export Operations - Shared Utilities

Reusable functions for safe, atomic file operations with validation.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from back_end.src.data.config import setup_logging

logger = setup_logging(__name__, 'phase_5_export_operations.log')


def atomic_write_json(
    data: Dict[str, Any],
    filepath: Path,
    pretty_print: bool = True,
    backup_existing: bool = True
) -> bool:
    """
    Write JSON with atomic rename to prevent corruption.

    Process:
    1. Write to temporary file (.tmp)
    2. Optionally backup existing file (.bak)
    3. Atomic rename (temp → target)

    Args:
        data: Data to serialize to JSON
        filepath: Target file path
        pretty_print: Format JSON with indentation
        backup_existing: Create .bak backup before overwriting

    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        temp_path = filepath.with_suffix('.tmp')

        # Ensure output directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create backup of existing file
        if backup_existing and filepath.exists():
            backup_path = filepath.with_suffix(filepath.suffix + '.bak')
            try:
                shutil.copy2(filepath, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        # Write to temporary file
        with open(temp_path, 'w', encoding='utf-8') as f:
            if pretty_print:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)

        # Atomic rename (temp → target)
        temp_path.replace(filepath)

        logger.info(f"Successfully wrote {filepath} ({get_file_size_mb(filepath):.2f} MB)")
        return True

    except Exception as e:
        logger.error(f"Failed to write {filepath}: {e}")
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        return False


def backup_existing_file(filepath: Path, timestamped: bool = False) -> Optional[Path]:
    """
    Create backup of existing file.

    Args:
        filepath: File to backup
        timestamped: Add timestamp to backup filename

    Returns:
        Path to backup file if created, None otherwise
    """
    try:
        filepath = Path(filepath)

        if not filepath.exists():
            return None

        if timestamped:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = filepath.with_name(f"{filepath.stem}_{timestamp}{filepath.suffix}.bak")
        else:
            backup_path = filepath.with_suffix(filepath.suffix + '.bak')

        shutil.copy2(filepath, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path

    except Exception as e:
        logger.error(f"Failed to backup {filepath}: {e}")
        return None


def validate_export_data(data: Dict[str, Any], export_type: str) -> Dict[str, Any]:
    """
    Validate exported data structure and content.

    Args:
        data: Exported data dictionary
        export_type: Type of export ('table_view', 'network_viz', 'mechanism_clusters')

    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'valid': True,
        'export_type': export_type,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }

    try:
        if export_type == 'table_view':
            # Validate table view data structure
            if 'interventions' not in data:
                validation_result['valid'] = False
                validation_result['errors'].append("Missing 'interventions' field")
            else:
                intervention_count = len(data['interventions'])
                validation_result['statistics']['intervention_count'] = intervention_count

                if intervention_count < 10:
                    validation_result['warnings'].append(f"Low intervention count: {intervention_count}")

            if 'metadata' not in data:
                validation_result['warnings'].append("Missing metadata field")
            else:
                validation_result['statistics'].update(data['metadata'])

        elif export_type == 'network_viz':
            # Validate network visualization data
            if 'nodes' not in data:
                validation_result['valid'] = False
                validation_result['errors'].append("Missing 'nodes' field")
            else:
                node_count = len(data['nodes'])
                validation_result['statistics']['node_count'] = node_count

                if node_count < 50:
                    validation_result['warnings'].append(f"Low node count: {node_count}")

            if 'links' not in data:
                validation_result['valid'] = False
                validation_result['errors'].append("Missing 'links' field")
            else:
                link_count = len(data['links'])
                validation_result['statistics']['link_count'] = link_count

                if link_count < 20:
                    validation_result['warnings'].append(f"Low link count: {link_count}")

        elif export_type == 'mechanism_clusters':
            # Validate mechanism clusters data
            if 'clusters' not in data:
                validation_result['valid'] = False
                validation_result['errors'].append("Missing 'clusters' field")
            else:
                cluster_count = len(data['clusters'])
                validation_result['statistics']['cluster_count'] = cluster_count

                if cluster_count < 5:
                    validation_result['warnings'].append(f"Low cluster count: {cluster_count}")

        else:
            validation_result['warnings'].append(f"Unknown export type: {export_type}")

    except Exception as e:
        validation_result['valid'] = False
        validation_result['errors'].append(f"Validation exception: {e}")

    return validation_result


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes."""
    try:
        return Path(filepath).stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime for export metadata."""
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def serialize_json_compatible(obj: Any) -> Any:
    """
    Make object JSON-serializable.

    Handles common non-serializable types.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


def check_file_exists(filepath: Path) -> bool:
    """Check if file exists and is readable."""
    try:
        filepath = Path(filepath)
        return filepath.exists() and filepath.is_file()
    except Exception:
        return False


def clean_temp_files(directory: Path, pattern: str = "*.tmp") -> int:
    """
    Clean up temporary files in directory.

    Args:
        directory: Directory to clean
        pattern: Glob pattern for temp files

    Returns:
        Number of files deleted
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            return 0

        deleted_count = 0
        for temp_file in directory.glob(pattern):
            try:
                temp_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} temp files")

        return deleted_count

    except Exception as e:
        logger.error(f"Failed to clean temp files: {e}")
        return 0
