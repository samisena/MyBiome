#!/usr/bin/env python3
"""
Rotation Pipeline LLM Processor

LLM processor for medical rotation pipeline supporting both single-condition
and batch processing modes. Includes sequential dual-model analysis optimized
for 8GB VRAM constraints and comprehensive thermal protection.

Features:
- Batch processing for all unprocessed papers
- Sequential dual-model analysis (gemma2:9b → qwen2.5:14b)
- VRAM-optimized model loading/unloading
- Paper-level interruption recovery
- Integration with rotation session manager
- Advanced thermal protection and monitoring
- Progress tracking and validation
"""

import sys
import time
import traceback
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm

try:
    from ..data.config import config, setup_logging
    from ..data_collection.database_manager import database_manager
    from ..llm_processing.dual_model_analyzer import DualModelAnalyzer
    from ..data.repositories import repository_manager
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.data_collection.database_manager import database_manager
    from back_end.src.llm_processing.dual_model_analyzer import DualModelAnalyzer
    from back_end.src.data.repositories import repository_manager

logger = setup_logging(__name__, 'rotation_llm_processor.log')


@dataclass
class ThermalStatus:
    """Current thermal and system status."""
    gpu_temp: float
    gpu_power: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    cpu_temp: float
    timestamp: float
    is_safe: bool
    cooling_needed: bool


class ThermalMonitor:
    """Enhanced thermal monitoring with predictive cooling."""

    def __init__(self, max_temp: float = 85.0, cooling_temp: float = 75.0):
        self.max_temp = max_temp
        self.cooling_temp = cooling_temp
        self.temp_history = []  # For tracking temperature trends

    def get_system_status(self) -> Optional[ThermalStatus]:
        """Get comprehensive system thermal status."""
        try:
            # GPU stats
            gpu_result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)

            gpu_temp = gpu_power = gpu_mem_used = gpu_mem_total = gpu_util = 0.0

            if gpu_result.returncode == 0:
                values = gpu_result.stdout.strip().split(',')
                gpu_temp = float(values[0]) if values[0] != '[Not Supported]' else 0.0
                gpu_power = float(values[1]) if values[1] != '[Not Supported]' else 0.0
                gpu_mem_used = float(values[2]) if values[2] != '[Not Supported]' else 0.0
                gpu_mem_total = float(values[3]) if values[3] != '[Not Supported]' else 0.0
                gpu_util = float(values[4]) if values[4] != '[Not Supported]' else 0.0

            # Get CPU temperature (basic)
            cpu_temp = 0.0
            try:
                import psutil
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        for name, entries in temps.items():
                            if entries:
                                cpu_temp = max(entry.current for entry in entries)
                                break
            except:
                pass

            # Safety checks
            is_safe = (gpu_temp < self.max_temp and cpu_temp < 80.0)
            cooling_needed = gpu_temp > self.cooling_temp

            status = ThermalStatus(
                gpu_temp=gpu_temp,
                gpu_power=gpu_power,
                gpu_memory_used=gpu_mem_used,
                gpu_memory_total=gpu_mem_total,
                gpu_utilization=gpu_util,
                cpu_temp=cpu_temp,
                timestamp=time.time(),
                is_safe=is_safe,
                cooling_needed=cooling_needed
            )

            # Track temperature history for predictive cooling
            self.temp_history.append((time.time(), gpu_temp))
            # Keep only last 10 minutes
            cutoff = time.time() - 600
            self.temp_history = [(t, temp) for t, temp in self.temp_history if t > cutoff]

            return status

        except Exception as e:
            logger.warning(f"Error getting thermal status: {e}")
            return None

    def is_thermal_safe(self) -> Tuple[bool, Optional[ThermalStatus]]:
        """Check if thermal conditions are safe for processing."""
        status = self.get_system_status()
        if not status:
            return True, None  # Assume safe if can't read

        # Predictive cooling: check if temperature is rising rapidly
        if len(self.temp_history) >= 3:
            recent_temps = [temp for _, temp in self.temp_history[-3:]]
            if len(recent_temps) >= 2:
                temp_rate = (recent_temps[-1] - recent_temps[0]) / len(recent_temps)
                if temp_rate > 2 and status.gpu_temp > (self.max_temp - 5):
                    logger.warning(f"GPU heating rapidly: {temp_rate:.1f}°C/check")

        return status.is_safe, status

    def wait_for_cooling(self) -> None:
        """Wait for system to cool down to safe operating temperature."""
        logger.info(f"🌡️ Thermal protection activated, cooling to {self.cooling_temp}°C...")

        cooling_start = time.time()
        while True:
            is_safe, status = self.is_thermal_safe()

            if not status:
                logger.warning("Cannot read temperature during cooling, resuming")
                break

            if status.gpu_temp <= self.cooling_temp:
                cooling_duration = time.time() - cooling_start
                logger.info(f"❄️ System cooled to {status.gpu_temp:.1f}°C in {cooling_duration:.1f}s")
                break

            logger.info(f"🌡️ Cooling... GPU: {status.gpu_temp:.1f}°C, target: {self.cooling_temp}°C")
            time.sleep(30)


class ProcessingError(Exception):
    """Custom exception for LLM processing errors."""
    pass


class RotationLLMProcessor:
    """
    Simplified LLM processor for rotation pipeline.
    Processes papers for a single condition with robust error handling.
    """

    def __init__(self):
        """Initialize the rotation LLM processor."""
        self.dual_analyzer = DualModelAnalyzer()
        self.max_retries = 2
        self.retry_delays = [30, 60]  # seconds

        # Processing configuration
        self.batch_size = 3  # Small batches for better error recovery

        # Enhanced thermal protection
        self.thermal_monitor = ThermalMonitor(max_temp=85.0, cooling_temp=75.0)

        # Performance tracking
        self.papers_processed = 0
        self.interventions_extracted = 0
        self.processing_errors = 0

    # Old condition-specific processing method removed - replaced by process_all_papers_batch()

    # Old condition-specific _get_unprocessed_papers method removed - replaced by _get_all_unprocessed_papers()

    # Old condition-specific batch processing methods removed - replaced by process_all_papers_batch()

    def _mark_paper_processed(self, paper_id: str, status: str):
        """Mark a paper as processed or failed in the database."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE papers
                    SET processing_status = ?
                    WHERE pmid = ?
                """, (status, paper_id))

                conn.commit()

        except Exception as e:
            logger.error(f"Error marking paper {paper_id} as {status}: {e}")

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on current GPU memory availability."""
        try:
            status = self.thermal_monitor.get_system_status()
            if status and status.gpu_memory_total > 0:
                # Conservative estimate: ~1GB per paper for dual models
                free_memory_gb = (status.gpu_memory_total - status.gpu_memory_used) / 1024
                optimal_size = max(1, int(free_memory_gb * 0.8))  # Use 80% of free memory
                return min(optimal_size, 10)  # Cap at 10 papers per batch
        except Exception as e:
            logger.debug(f"Could not determine optimal batch size: {e}")

        return self.batch_size  # Fallback to default

    def get_thermal_status(self) -> Dict[str, Any]:
        """Get current thermal status for monitoring."""
        status = self.thermal_monitor.get_system_status()
        if not status:
            return {'error': 'Cannot read thermal status'}

        return {
            'gpu_temp': status.gpu_temp,
            'gpu_power': status.gpu_power,
            'gpu_memory_used_mb': status.gpu_memory_used,
            'gpu_memory_total_mb': status.gpu_memory_total,
            'gpu_memory_usage_percent': (status.gpu_memory_used / status.gpu_memory_total * 100) if status.gpu_memory_total > 0 else 0,
            'gpu_utilization_percent': status.gpu_utilization,
            'cpu_temp': status.cpu_temp,
            'is_safe': status.is_safe,
            'cooling_needed': status.cooling_needed,
            'optimal_batch_size': self.get_optimal_batch_size()
        }

    def get_processing_status(self, condition: str) -> Dict[str, Any]:
        """Get processing status for a condition."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get processing statistics
                cursor.execute("""
                    SELECT
                        COUNT(DISTINCT p.id) as total_papers,
                        COUNT(DISTINCT CASE WHEN p.processing_status = 'processed' THEN p.id END) as processed_papers,
                        COUNT(DISTINCT CASE WHEN p.processing_status = 'failed' THEN p.id END) as failed_papers,
                        COUNT(DISTINCT CASE WHEN p.processing_status IS NULL THEN p.id END) as unprocessed_papers,
                        COUNT(i.id) as total_interventions
                    FROM papers p
                    JOIN interventions i ON p.id = i.paper_id
                    WHERE LOWER(i.condition) LIKE LOWER(?)
                """, (f"%{condition}%",))

                stats = cursor.fetchone()

                return {
                    'condition': condition,
                    'total_papers': stats[0] if stats else 0,
                    'processed_papers': stats[1] if stats else 0,
                    'failed_papers': stats[2] if stats else 0,
                    'unprocessed_papers': stats[3] if stats else 0,
                    'total_interventions': stats[4] if stats else 0,
                    'processing_rate': (stats[1] / stats[0] * 100) if stats and stats[0] > 0 else 0
                }

        except Exception as e:
            logger.error(f"Error getting processing status for '{condition}': {e}")
            return {
                'condition': condition,
                'error': str(e)
            }

    def validate_processing_result(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate processing result.

        Returns:
            Tuple of (is_valid, validation_message)
        """
        if not isinstance(result, dict):
            return False, "Result is not a dictionary"

        required_fields = ['success', 'condition', 'papers_processed', 'interventions_extracted']
        missing_fields = [field for field in required_fields if field not in result]

        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"

        if not result['success']:
            error_msg = result.get('error', 'Unknown error')
            return False, f"Processing failed: {error_msg}"

        if result['papers_processed'] < 0:
            return False, f"Invalid papers_processed count: {result['papers_processed']}"

        if result['interventions_extracted'] < 0:
            return False, f"Invalid interventions_extracted count: {result['interventions_extracted']}"

        return True, "Processing result is valid"

    def process_all_papers_batch(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Process ALL unprocessed papers using sequential dual-model approach.

        This method is optimized for 8GB VRAM by loading models sequentially:
        1. Load gemma2:9b → process all papers → unload model
        2. Load qwen2.5:14b → process all papers → unload model
        3. Build consensus from both model results

        Args:
            batch_size: Papers per batch (auto-optimized if None)

        Returns:
            Dictionary with comprehensive processing results
        """
        start_time = datetime.now()
        logger.info("Starting batch processing of all unprocessed papers")

        try:
            # Get all unprocessed papers from database
            unprocessed_papers = self._get_all_unprocessed_papers()

            if not unprocessed_papers:
                logger.info("No unprocessed papers found")
                return {
                    'success': True,
                    'papers_processed': 0,
                    'interventions_extracted': 0,
                    'processing_time_seconds': 0,
                    'status': 'no_papers_to_process'
                }

            logger.info(f"Found {len(unprocessed_papers)} unprocessed papers")

            # Check thermal status before starting
            is_safe, thermal_status = self.thermal_monitor.is_thermal_safe()
            if not is_safe:
                logger.warning("System too hot for processing, waiting for cooling...")
                self.thermal_monitor.wait_for_cooling()

            # Determine optimal batch size
            if batch_size is None:
                batch_size = self.get_optimal_batch_size()

            logger.info(f"Using batch size: {batch_size}")

            # Process papers using dual-model analyzer's batch method
            # The dual_model_analyzer already handles sequential processing
            processing_result = self.dual_analyzer.process_papers_batch(
                papers=unprocessed_papers,
                save_to_db=True,
                batch_size=batch_size
            )

            # Calculate final statistics
            processing_time = (datetime.now() - start_time).total_seconds()

            # Update internal counters
            self.papers_processed += processing_result['successful_papers']
            self.interventions_extracted += processing_result['total_interventions']
            self.processing_errors += len(processing_result['failed_papers'])

            # Compile results
            result = {
                'success': True,
                'total_papers_found': len(unprocessed_papers),
                'papers_processed': processing_result['successful_papers'],
                'papers_failed': len(processing_result['failed_papers']),
                'interventions_extracted': processing_result['total_interventions'],
                'processing_time_seconds': processing_time,
                'model_statistics': processing_result['model_statistics'],
                'interventions_by_category': processing_result['interventions_by_category'],
                'failed_papers': processing_result['failed_papers'],
                'status': 'completed'
            }

            # Success rate calculation
            if result['total_papers_found'] > 0:
                result['success_rate'] = (result['papers_processed'] / result['total_papers_found']) * 100
            else:
                result['success_rate'] = 100.0

            logger.info(f"Batch processing completed in {processing_time:.1f}s")
            logger.info(f"Papers processed: {result['papers_processed']}")
            logger.info(f"Interventions extracted: {result['interventions_extracted']}")
            logger.info(f"Success rate: {result['success_rate']:.1f}%")

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Batch processing failed: {e}")
            logger.error(traceback.format_exc())

            return {
                'success': False,
                'papers_processed': 0,
                'interventions_extracted': 0,
                'processing_time_seconds': processing_time,
                'error': str(e),
                'status': 'failed'
            }

    def _get_all_unprocessed_papers(self) -> List[Dict[str, Any]]:
        """Get all papers that haven't been processed by any model yet."""
        try:
            # Use the dual_model_analyzer's method to get unprocessed papers
            return self.dual_analyzer.get_unprocessed_papers()

        except Exception as e:
            logger.error(f"Error getting unprocessed papers: {e}")
            return []


def process_single_condition(condition: str, max_papers: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to process papers for a single condition.
    NOTE: This now uses the batch processing method which processes ALL papers.

    Args:
        condition: Medical condition to process (ignored - all papers processed)
        max_papers: Maximum number of papers to process (ignored - all papers processed)

    Returns:
        Processing result dictionary
    """
    processor = RotationLLMProcessor()
    return processor.process_all_papers_batch()


if __name__ == "__main__":
    """Test the rotation LLM processor with a single condition."""
    import argparse

    parser = argparse.ArgumentParser(description="Rotation LLM Processor Test")
    parser.add_argument('condition', help='Medical condition to process papers for')
    parser.add_argument('--max-papers', type=int, help='Maximum number of papers to process')
    parser.add_argument('--status-only', action='store_true', help='Show status only, no processing')

    args = parser.parse_args()

    processor = RotationLLMProcessor()

    if args.status_only:
        # Show status only
        status = processor.get_processing_status(args.condition)
        print(f"\nProcessing Status for: {args.condition}")
        print("="*50)
        print(f"Total papers: {status['total_papers']}")
        print(f"Processed: {status['processed_papers']}")
        print(f"Failed: {status['failed_papers']}")
        print(f"Unprocessed: {status['unprocessed_papers']}")
        print(f"Total interventions: {status['total_interventions']}")
        print(f"Processing rate: {status['processing_rate']:.1f}%")
    else:
        # Run processing
        print(f"Processing papers for: {args.condition}")
        if args.max_papers:
            print(f"Max papers: {args.max_papers}")

        result = processor.process_condition_papers(
            condition=args.condition,
            max_papers=args.max_papers
        )

        print("\n" + "="*60)
        print("PROCESSING RESULT")
        print("="*60)
        print(f"Success: {result['success']}")
        print(f"Condition: {result['condition']}")
        print(f"Papers processed: {result['papers_processed']}")
        print(f"Interventions extracted: {result['interventions_extracted']}")
        print(f"Processing time: {result['processing_time_seconds']:.1f} seconds")
        print(f"Status: {result['status']}")

        if 'success_rate' in result:
            print(f"Success rate: {result['success_rate']:.1f}%")

        if 'processing_errors' in result and result['processing_errors'] > 0:
            print(f"Processing errors: {result['processing_errors']}")

        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")

        # Validate result
        is_valid, message = processor.validate_processing_result(result)
        print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
        print(f"Message: {message}")