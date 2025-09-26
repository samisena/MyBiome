#!/usr/bin/env python3
"""
Rotation Pipeline LLM Processor

Simplified LLM processor specifically designed for the medical rotation pipeline.
Processes papers for a single condition with focused intervention extraction,
integrates with rotation session manager, and includes paper-level resumption.

Features:
- Single-condition focused processing
- Paper-level interruption recovery
- Integration with rotation session manager
- Dual-model analysis (gemma2:9b + qwen2.5:14b)
- Progress tracking and validation
- Thermal protection (basic)
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

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
        self.max_temp_celsius = 85.0  # Basic thermal protection

        # TODO: Implement GPU/RAM utilization monitoring for optimal performance
        # - Monitor GPU memory usage to maximize batch sizes without OOM
        # - Track RAM utilization to optimize concurrent processing
        # - Implement dynamic batch size adjustment based on available resources
        # - Add GPU utilization metrics to ensure full hardware advantage

        # Performance tracking
        self.papers_processed = 0
        self.interventions_extracted = 0
        self.processing_errors = 0

    def process_condition_papers(self, condition: str, max_papers: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all unprocessed papers for a specific condition.

        Args:
            condition: Medical condition to process papers for
            max_papers: Maximum number of papers to process (None for all)

        Returns:
            Processing result with comprehensive statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting LLM processing for condition: '{condition}'")

        try:
            # Get unprocessed papers for this condition
            papers = self._get_unprocessed_papers(condition, max_papers)

            if not papers:
                logger.info(f"No unprocessed papers found for condition '{condition}'")
                return {
                    'success': True,
                    'condition': condition,
                    'papers_processed': 0,
                    'interventions_extracted': 0,
                    'processing_time_seconds': 0,
                    'status': 'no_papers_to_process'
                }

            logger.info(f"Found {len(papers)} unprocessed papers for '{condition}'")

            # Process papers in batches
            processing_result = self._process_papers_in_batches(papers, condition)

            # Calculate final metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_result['processing_time_seconds'] = processing_time
            processing_result['processing_rate'] = (
                processing_result['papers_processed'] / (processing_time / 60)
                if processing_time > 0 else 0
            )

            logger.info(f"LLM processing completed for '{condition}': "
                       f"{processing_result['papers_processed']} papers, "
                       f"{processing_result['interventions_extracted']} interventions")

            return processing_result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"LLM processing failed for '{condition}': {e}")
            logger.error(traceback.format_exc())

            return {
                'success': False,
                'condition': condition,
                'papers_processed': 0,
                'interventions_extracted': 0,
                'processing_time_seconds': processing_time,
                'error': str(e),
                'status': 'failed'
            }

    def _get_unprocessed_papers(self, condition: str, max_papers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get unprocessed papers for a specific condition."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get papers that haven't been processed yet for this condition
                query = """
                    SELECT DISTINCT p.id, p.pmid, p.title, p.abstract, p.publication_year
                    FROM papers p
                    JOIN interventions i ON p.id = i.paper_id
                    WHERE LOWER(i.condition) LIKE LOWER(?)
                    AND (p.processing_status IS NULL OR p.processing_status != 'processed')
                    AND p.abstract IS NOT NULL
                    AND LENGTH(p.abstract) > 100
                    ORDER BY p.publication_year DESC, p.id ASC
                """

                params = [f"%{condition}%"]
                if max_papers:
                    query += " LIMIT ?"
                    params.append(max_papers)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                papers = []
                for row in rows:
                    papers.append({
                        'id': row[0],
                        'pmid': row[1],
                        'title': row[2],
                        'abstract': row[3],
                        'publication_year': row[4]
                    })

                logger.debug(f"Retrieved {len(papers)} unprocessed papers for '{condition}'")
                return papers

        except Exception as e:
            logger.error(f"Error retrieving unprocessed papers for '{condition}': {e}")
            return []

    def _process_papers_in_batches(self, papers: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
        """Process papers in small batches with error recovery."""
        total_papers = len(papers)
        papers_processed = 0
        interventions_extracted = 0
        processing_errors = 0
        failed_papers = []

        logger.info(f"Processing {total_papers} papers in batches of {self.batch_size}")

        # Process in batches
        for batch_start in range(0, total_papers, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_papers)
            batch_papers = papers[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//self.batch_size + 1}: "
                       f"papers {batch_start + 1}-{batch_end} of {total_papers}")

            # Check thermal safety before batch
            if not self._is_thermal_safe():
                logger.warning("Thermal protection triggered, cooling down...")
                self._wait_for_cooling()

            # Process each paper in the batch
            for paper in batch_papers:
                try:
                    result = self._process_single_paper(paper, condition)

                    if result['success']:
                        papers_processed += 1
                        interventions_extracted += result['interventions_extracted']

                        # Mark paper as processed
                        self._mark_paper_processed(paper['id'], 'processed')

                        logger.debug(f"Processed paper {paper['id']}: "
                                   f"{result['interventions_extracted']} interventions")
                    else:
                        processing_errors += 1
                        failed_papers.append({
                            'paper_id': paper['id'],
                            'pmid': paper['pmid'],
                            'error': result.get('error', 'Unknown error')
                        })

                        # Mark paper as failed
                        self._mark_paper_processed(paper['id'], 'failed')

                        logger.warning(f"Failed to process paper {paper['id']}: "
                                     f"{result.get('error', 'Unknown error')}")

                except Exception as e:
                    processing_errors += 1
                    failed_papers.append({
                        'paper_id': paper['id'],
                        'pmid': paper['pmid'],
                        'error': str(e)
                    })

                    # Mark paper as failed
                    self._mark_paper_processed(paper['id'], 'failed')

                    logger.error(f"Exception processing paper {paper['id']}: {e}")

            # Brief pause between batches
            time.sleep(1)

        # Calculate success rate
        success_rate = (papers_processed / total_papers) * 100 if total_papers > 0 else 0

        return {
            'success': True,
            'condition': condition,
            'papers_processed': papers_processed,
            'interventions_extracted': interventions_extracted,
            'processing_errors': processing_errors,
            'failed_papers': failed_papers,
            'success_rate': success_rate,
            'total_papers': total_papers,
            'status': 'completed'
        }

    def _process_single_paper(self, paper: Dict[str, Any], condition: str) -> Dict[str, Any]:
        """Process a single paper with retry logic."""
        paper_id = paper['id']
        pmid = paper['pmid']

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Processing paper {paper_id} (PMID: {pmid}), attempt {attempt + 1}")

                # Use dual model analyzer
                results = self.dual_analyzer.analyze_paper(
                    paper_id=paper_id,
                    title=paper['title'],
                    abstract=paper['abstract'],
                    condition_filter=condition
                )

                # Count interventions extracted
                interventions_count = 0
                if results and results.get('interventions'):
                    interventions_count = len(results['interventions'])

                return {
                    'success': True,
                    'paper_id': paper_id,
                    'pmid': pmid,
                    'interventions_extracted': interventions_count,
                    'processing_results': results
                }

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for paper {paper_id}: {e}")

                if attempt < self.max_retries:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All attempts failed for paper {paper_id}")

        # If we get here, all retries failed
        return {
            'success': False,
            'paper_id': paper_id,
            'pmid': pmid,
            'interventions_extracted': 0,
            'error': f"Failed after {self.max_retries + 1} attempts"
        }

    def _mark_paper_processed(self, paper_id: int, status: str):
        """Mark a paper as processed or failed in the database."""
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE papers
                    SET processing_status = ?, processing_timestamp = ?
                    WHERE id = ?
                """, (status, datetime.now().isoformat(), paper_id))

                conn.commit()

        except Exception as e:
            logger.error(f"Error marking paper {paper_id} as {status}: {e}")

    def _is_thermal_safe(self) -> bool:
        """Basic thermal safety check."""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                temp = float(result.stdout.strip())
                return temp < self.max_temp_celsius

        except Exception:
            # If can't read temperature, assume safe
            pass

        return True

    def _get_gpu_utilization(self) -> Dict[str, Any]:
        """
        Get GPU utilization metrics for performance optimization.

        TODO: Implement comprehensive GPU/RAM monitoring:
        - GPU memory usage (used/total)
        - GPU utilization percentage
        - System RAM usage
        - Available memory for batch size optimization

        Returns:
            Dictionary with utilization metrics
        """
        try:
            import subprocess
            # Get GPU memory and utilization
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                values = result.stdout.strip().split(',')
                memory_used = int(values[0].strip())
                memory_total = int(values[1].strip())
                gpu_util = int(values[2].strip())

                return {
                    'gpu_memory_used_mb': memory_used,
                    'gpu_memory_total_mb': memory_total,
                    'gpu_memory_free_mb': memory_total - memory_used,
                    'gpu_memory_usage_percent': (memory_used / memory_total) * 100,
                    'gpu_utilization_percent': gpu_util,
                    'optimal_batch_size_estimate': self._estimate_optimal_batch_size(memory_total - memory_used)
                }

        except Exception as e:
            logger.debug(f"Could not get GPU utilization: {e}")

        return {
            'gpu_memory_used_mb': 0,
            'gpu_memory_total_mb': 0,
            'gpu_memory_free_mb': 0,
            'gpu_memory_usage_percent': 0,
            'gpu_utilization_percent': 0,
            'optimal_batch_size_estimate': self.batch_size
        }

    def _estimate_optimal_batch_size(self, free_memory_mb: int) -> int:
        """
        Estimate optimal batch size based on available GPU memory.

        TODO: Implement sophisticated batch size calculation:
        - Consider model memory requirements
        - Account for sequence length variations
        - Factor in dual-model memory needs
        - Include safety margins for memory spikes
        """
        # Conservative estimate: ~1GB per paper for dual models
        papers_per_gb = 1
        estimated_size = max(1, int(free_memory_mb / 1024 * papers_per_gb))

        # Cap at reasonable maximum
        return min(estimated_size, 10)

    def _wait_for_cooling(self, target_temp: float = 75.0):
        """Wait for GPU to cool down."""
        logger.info(f"Waiting for GPU to cool to {target_temp}°C...")

        while True:
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=10
                )

                if result.returncode == 0:
                    temp = float(result.stdout.strip())
                    if temp <= target_temp:
                        logger.info(f"GPU cooled to {temp}°C, resuming processing")
                        break

                    logger.info(f"Current GPU temperature: {temp}°C, waiting...")
                    time.sleep(30)
                else:
                    # Can't read temperature, wait and continue
                    time.sleep(30)
                    break

            except Exception:
                # Error reading temperature, wait and continue
                time.sleep(30)
                break

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


def process_single_condition(condition: str, max_papers: Optional[int] = None) -> Dict[str, Any]:
    """
    Convenience function to process papers for a single condition.

    Args:
        condition: Medical condition to process
        max_papers: Maximum number of papers to process

    Returns:
        Processing result dictionary
    """
    processor = RotationLLMProcessor()
    return processor.process_condition_papers(condition=condition, max_papers=max_papers)


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