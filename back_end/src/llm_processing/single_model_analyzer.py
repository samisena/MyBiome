"""
Single-model intervention analyzer using qwen3:14b.
Simplified approach that runs one model and stores results directly.
No consensus building needed - 2x faster than dual-model approach.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Graceful degradation for tqdm (Phase 5.1)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: dummy tqdm that does nothing
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get('total', 0)
            self.n = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            self.n += n
        def set_postfix(self, **kwargs):
            pass
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    yield item
                    self.update(1)

try:
    import torch
except ImportError:
    torch = None

from back_end.src.data.config import config, setup_logging
from back_end.src.data.api_clients import get_llm_client
from back_end.src.data.repositories import repository_manager
from back_end.src.data.utils import (parse_json_safely, batch_process)
from back_end.src.data.error_handler import handle_llm_errors
from back_end.src.interventions.category_validators import category_validator, CategoryValidationError
from back_end.src.interventions.taxonomy import InterventionType
from back_end.src.llm_processing.prompt_service import prompt_service

logger = setup_logging(__name__, 'single_model_analyzer.log')


@dataclass
class ModelResult:
    """Result from model extraction."""
    model_name: str
    interventions: List[Dict]
    extraction_time: float
    error: Optional[str] = None


class SingleModelAnalyzer:
    """
    Simple single-model analyzer using qwen2.5:14b.
    Runs one model on each paper and stores valid interventions directly.

    Benefits over dual-model approach:
    - 2x faster processing
    - No consensus complexity
    - Preserves Qwen's superior extraction detail
    - Simpler error handling and debugging
    """

    def __init__(self, repository_mgr=None):
        """
        Initialize with qwen2.5:14b model.

        Args:
            repository_mgr: Repository manager instance (optional, uses global if None)
        """
        self.repository_mgr = repository_mgr or repository_manager

        # Single model configuration
        self.model_name = 'qwen2.5:14b'
        self.model_config = {
            'client': get_llm_client('qwen2.5:14b'),
            'temperature': 0.3,
            'max_tokens': None,  # Will be calculated dynamically
            'max_context': 32768,  # Model's maximum context length
            'recommended_max_output': 16384  # Reasonable upper bound for output
        }

        # Get validator and prompt service
        self.validator = category_validator
        self.prompt_service = prompt_service

        # GPU optimization settings
        self.gpu_optimization = self._initialize_gpu_optimization()

        logger.info("Single-model analyzer initialized with qwen2.5:14b")

    def _initialize_gpu_optimization(self) -> Dict[str, Any]:
        """Initialize GPU optimization settings."""
        try:
            import psutil
        except ImportError:
            return {
                'gpu_available': False,
                'gpu_memory_gb': 0,
                'optimal_batch_size': 5,
                'memory_threshold': 0.8
            }

        gpu_available = torch and torch.cuda.is_available()

        if gpu_available:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Optimal batch size for single model (can be higher than dual-model)
            optimal_batch_size = 12  # Optimized based on Phase 2 batch size experiments
        else:
            gpu_memory_gb = 0
            optimal_batch_size = 12

        try:
            import psutil
            system_ram = psutil.virtual_memory().total / (1024**3)
        except:
            system_ram = 16.0

        optimization_config = {
            'gpu_available': gpu_available,
            'gpu_memory_gb': gpu_memory_gb,
            'system_ram_gb': system_ram,
            'optimal_batch_size': optimal_batch_size,
            'memory_threshold': 0.90,
        }

        return optimization_config

    def _calculate_dynamic_max_tokens(self, prompt: str) -> int:
        """
        Calculate dynamic max_tokens based on input length and model capabilities.

        Args:
            prompt: The input prompt

        Returns:
            Optimal max_tokens for this request
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        estimated_input_tokens = len(prompt) // 4

        # Calculate available context space
        max_context = self.model_config.get('max_context', 32768)
        recommended_max_output = self.model_config.get('recommended_max_output', 16384)

        # Leave buffer for system message and response formatting
        buffer_tokens = 500
        available_for_output = max_context - estimated_input_tokens - buffer_tokens

        # Use the minimum of available space and recommended max output
        dynamic_max_tokens = min(available_for_output, recommended_max_output)

        # Ensure we have at least a reasonable minimum
        dynamic_max_tokens = max(dynamic_max_tokens, 2048)

        return dynamic_max_tokens

    def _monitor_gpu_memory(self) -> Dict[str, float]:
        """Monitor current GPU memory utilization."""
        try:
            if torch and torch.cuda.is_available():
                reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                return {
                    'utilization': (reserved / total) if total > 0 else 0
                }
        except:
            pass
        return {'utilization': 0}

    def _optimize_batch_size_for_memory(self, requested_batch_size: int) -> int:
        """Optimize batch size based on current memory usage."""
        gpu_mem = self._monitor_gpu_memory()
        optimal_batch = self.gpu_optimization.get('optimal_batch_size', 8)

        # If GPU memory is getting tight, reduce batch size
        if gpu_mem['utilization'] > 0.7:
            return min(requested_batch_size, max(1, optimal_batch // 2))

        # If we have plenty of memory, allow larger batches
        if gpu_mem['utilization'] < 0.5:
            return min(requested_batch_size, optimal_batch * 2)

        return min(requested_batch_size, optimal_batch)

    @handle_llm_errors("extract interventions", max_retries=3)
    def extract_interventions(self, paper: Dict) -> Dict[str, Any]:
        """
        Extract interventions from a single paper using qwen3:14b.

        Args:
            paper: Dictionary with 'pmid', 'title', and 'abstract'

        Returns:
            Dictionary with extraction results
        """
        pmid = paper.get('pmid', 'unknown')
        start_time = time.time()

        # Validate input
        if not paper.get('abstract') or not paper.get('title'):
            logger.warning(f"Paper {pmid} missing title or abstract - skipping")
            return {'pmid': pmid, 'total_interventions': 0, 'interventions': []}

        if len(paper['abstract'].strip()) < 100:
            logger.debug(f"Paper {pmid} has very short abstract ({len(paper['abstract'])} chars) - proceeding with caution")

        try:
            # Update paper processing status
            self.repository_mgr.papers.update_processing_status(pmid, 'processing')

            # Extract using qwen3:14b
            client = self.model_config['client']

            # Create prompt using shared service
            prompt = self.prompt_service.create_extraction_prompt(paper)

            # Calculate dynamic max_tokens
            dynamic_max_tokens = self._calculate_dynamic_max_tokens(prompt)

            # Get system message for <think> tag suppression
            system_message = self.prompt_service.create_system_message()

            # Call LLM with separate system and user messages
            response = client.generate(
                prompt=prompt,
                temperature=self.model_config['temperature'],
                max_tokens=dynamic_max_tokens,
                system_message=system_message
            )

            # Extract response
            response_text = response.get('content', '')
            extraction_time = time.time() - start_time

            # Parse JSON response
            interventions = parse_json_safely(response_text, f"{pmid}_{self.model_name}")

            # Validate and enhance interventions
            if interventions:
                validated_interventions = self._validate_and_enhance_interventions(
                    interventions, paper, self.model_name
                )
            else:
                validated_interventions = []

            # Compile results
            result = {
                'pmid': pmid,
                'model': self.model_name,
                'total_interventions': len(validated_interventions),
                'interventions': validated_interventions,
                'extraction_time': extraction_time
            }

            # Log results
            logger.info(f"Paper {pmid} processing complete: {len(validated_interventions)} interventions extracted in {extraction_time:.1f}s")

            # Update processing status
            status = 'processed' if validated_interventions else 'processed'
            self.repository_mgr.papers.update_processing_status(pmid, status)

            return result

        except Exception as e:
            extraction_time = time.time() - start_time
            logger.error(f"Error processing paper {pmid}: {e}")
            self.repository_mgr.papers.update_processing_status(pmid, 'failed')
            return {
                'pmid': pmid,
                'model': self.model_name,
                'total_interventions': 0,
                'interventions': [],
                'error': str(e),
                'extraction_time': extraction_time
            }

    def _validate_and_enhance_interventions(self, interventions: List[Dict],
                                          paper: Dict, model_name: str) -> List[Dict]:
        """
        Validate and enhance intervention data.

        Args:
            interventions: Raw interventions from LLM
            paper: Source paper information
            model_name: Name of the model that extracted these

        Returns:
            List of validated and enhanced interventions
        """
        validated = []

        for i, intervention in enumerate(interventions):
            try:
                # Add metadata
                intervention['paper_id'] = paper['pmid']
                intervention['extraction_model'] = model_name

                # Debug: Check if condition_category is in raw extraction
                if 'condition_category' in intervention:
                    logger.info(f"Raw extraction has condition_category: {intervention['condition_category']}")
                else:
                    logger.warning(f"Raw extraction MISSING condition_category for intervention: {intervention.get('intervention_name', 'unknown')}")

                # Validate using intervention validator
                validated_intervention = self.validator.validate_intervention(intervention)

                # Debug: Check if condition_category survived validation
                if 'condition_category' in validated_intervention:
                    logger.info(f"Validated intervention has condition_category: {validated_intervention['condition_category']}")
                else:
                    logger.warning(f"Validated intervention MISSING condition_category for: {validated_intervention.get('intervention_name', 'unknown')}")

                validated.append(validated_intervention)

            except Exception as e:
                logger.debug(f"Intervention validation failed for paper {paper['pmid']}: {e}")
                continue

        return validated

    def process_papers_batch(self, papers: List[Dict], save_to_db: bool = True,
                           batch_size: int = None) -> Dict[str, Any]:
        """
        Process multiple papers with qwen2.5:14b using GPU-optimized batching.

        Args:
            papers: List of paper dictionaries
            save_to_db: Whether to save results to database
            batch_size: Number of papers to process in each batch (auto-optimized if None)

        Returns:
            Processing results summary
        """
        # Optimize batch size based on GPU memory if not specified
        if batch_size is None:
            batch_size = self.gpu_optimization.get('optimal_batch_size', 8)

        # Further optimize based on current memory usage
        optimized_batch_size = self._optimize_batch_size_for_memory(batch_size)

        # Process in batches for better resource management
        batches = batch_process(papers, optimized_batch_size)

        all_results = []
        failed_papers = []
        total_processed = 0
        total_interventions = 0
        category_counts = {cat.value: 0 for cat in InterventionType}
        category_counts['uncategorized'] = 0  # Track NULL categories separately

        # Process all batches with progress bar
        with tqdm(total=len(papers), desc="Processing papers", unit="paper") as pbar:
            for batch_num, batch in enumerate(batches, 1):
                logger.info(f"Processing batch {batch_num}/{len(batches)} ({len(batch)} papers)")

                for i, paper in enumerate(batch, 1):
                    paper_num = (batch_num - 1) * batch_size + i
                    logger.debug(f"Processing paper {paper_num}/{len(papers)}: {paper['pmid']}")

                    try:
                        results = self.extract_interventions(paper)

                        if results.get('total_interventions', 0) > 0:
                            all_results.append(results)

                            # Count by category
                            for intervention in results['interventions']:
                                category = intervention.get('intervention_category')

                                # Track uncategorized separately (NULL categories)
                                if category is None:
                                    category_counts['uncategorized'] += 1
                                elif category in category_counts:
                                    category_counts[category] += 1

                            # Save interventions to database directly (no consensus needed)
                            if save_to_db:
                                self._save_interventions_batch(results['interventions'])
                                # Mark paper as LLM processed
                                self.repository_mgr.db_manager.mark_paper_llm_processed(paper['pmid'])

                            # Update running total
                            total_interventions += len(results['interventions'])
                            pbar.set_postfix({'interventions': total_interventions, 'failed': len(failed_papers)})
                        else:
                            if results.get('error'):
                                logger.error(f"Error processing paper {paper['pmid']}: {results.get('error')}")
                                failed_papers.append(paper['pmid'])
                                pbar.set_postfix({'interventions': total_interventions, 'failed': len(failed_papers)})

                        total_processed += 1
                        pbar.update(1)

                        # Delay between papers
                        if i < len(batch):
                            time.sleep(0.5)

                    except Exception as e:
                        logger.error(f"Failed to process paper {paper['pmid']}: {e}")
                        failed_papers.append(paper['pmid'])
                        pbar.set_postfix({'interventions': total_interventions, 'failed': len(failed_papers)})
                        pbar.update(1)

                # Delay between batches
                if batch_num < len(batches):
                    time.sleep(1.0)

        # Compile results
        model_stats = {
            self.model_name: {
                'papers': len([r for r in all_results if r.get('total_interventions', 0) > 0]),
                'interventions': total_interventions
            }
        }

        results = {
            'total_papers': len(papers),
            'successful_papers': total_processed - len(failed_papers),
            'failed_papers': failed_papers,
            'total_interventions': total_interventions,
            'interventions_by_category': category_counts,
            'model_statistics': model_stats,
            'paper_results': all_results
        }

        return results

    def _save_interventions_batch(self, interventions: List[Dict]):
        """
        Save a batch of interventions to database.
        """
        for intervention in interventions:
            try:
                # Use standard insertion
                success = self.repository_mgr.interventions.insert_intervention(intervention)
                if success:
                    logger.debug(f"Intervention saved: {intervention.get('intervention_name')}")
                else:
                    logger.error(f"Failed to save intervention: {intervention.get('intervention_name')}")
            except Exception as e:
                logger.error(f"Error saving intervention: {e}")

    def get_unprocessed_papers(self, limit: Optional[int] = None) -> List[Dict]:
        """Get papers that haven't been processed yet (truly unprocessed only)."""
        with self.repository_mgr.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            query = f'''
                SELECT DISTINCT p.*
                FROM papers p
                WHERE p.abstract IS NOT NULL
                  AND p.abstract != ''
                  AND (p.processing_status IS NULL OR p.processing_status = 'pending')
                ORDER BY
                    COALESCE(p.influence_score, 0) DESC,
                    COALESCE(p.citation_count, 0) DESC,
                    p.publication_date DESC
            '''

            if limit:
                query += f' LIMIT {limit}'

            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            papers = [dict(zip(columns, row)) for row in cursor.fetchall()]

            return papers

    def process_unprocessed_papers(self, limit: Optional[int] = None,
                                 batch_size: int = None) -> Dict:
        """Process all unprocessed papers."""
        papers = self.get_unprocessed_papers(limit)

        if not papers:
            logger.info("No unprocessed papers found")
            return {
                'total_papers': 0,
                'successful_papers': 0,
                'failed_papers': [],
                'total_interventions': 0,
                'interventions_by_category': {},
                'model_statistics': {},
                'paper_results': []
            }

        logger.info(f"Found {len(papers)} unprocessed papers for processing")
        return self.process_papers_batch(papers, save_to_db=True, batch_size=batch_size)
