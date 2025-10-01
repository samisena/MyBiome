"""
Dual-model intervention analyzer with gemma2:9b and qwen2.5:14b.
Simplified approach that runs both models and stores all results.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm

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

logger = setup_logging(__name__, 'dual_model_analyzer.log')


@dataclass
class ModelResult:
    """Result from a single model extraction."""
    model_name: str
    interventions: List[Dict]
    extraction_time: float
    error: Optional[str] = None



class DualModelAnalyzer:
    """
    Simple dual-model analyzer using gemma2:9b and qwen2.5:14b.
    Runs both models on each paper and stores all valid interventions.
    """
    
    def __init__(self, repository_mgr=None):
        """
        Initialize with the two specified models.
        
        Args:
            repository_mgr: Repository manager instance (optional, uses global if None)
        """
        self.repository_mgr = repository_mgr or repository_manager
        
        # Enhanced model configuration with dynamic token limits
        self.models = {
            'gemma2:9b': {
                'client': get_llm_client('gemma2:9b'),
                'temperature': 0.3,
                'max_tokens': None,  # No limit - will be calculated dynamically
                'max_context': 32768,  # Model's maximum context length
                'recommended_max_output': 16384  # Reasonable upper bound for output
            },
            'qwen2.5:14b': {
                'client': get_llm_client('qwen2.5:14b'),
                'temperature': 0.3,
                'max_tokens': None,  # No limit - will be calculated dynamically
                'max_context': 32768,  # Model's maximum context length
                'recommended_max_output': 16384  # Reasonable upper bound for output
            }
        }
        
        # Get validator and prompt service
        self.validator = category_validator
        self.prompt_service = prompt_service
        
        
        # GPU optimization settings
        self.gpu_optimization = self._initialize_gpu_optimization()

        logger.info("Dual-model analyzer initialized successfully")

    def _initialize_gpu_optimization(self) -> Dict[str, Any]:
        """Initialize GPU optimization settings."""
        try:
            import psutil
        except ImportError:
            # GPU optimization libraries not available, using conservative settings
            pass
            return {
                'gpu_available': False,
                'gpu_memory_gb': 0,
                'optimal_batch_size': 5,  # Use same optimal batch size even without PyTorch
                'memory_threshold': 0.8
            }

        gpu_available = torch and torch.cuda.is_available()

        if gpu_available:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Hard-coded optimal batch size based on empirical testing
            # Batch size 5 provides optimal performance for 8GB GPU with Gemma2:9b + Qwen2.5:14b
            optimal_batch_size = 5  # Empirically determined optimal value
        else:
            gpu_memory_gb = 0
            optimal_batch_size = 5  # Use same optimal batch size for consistency

        optimization_config = {
            'gpu_available': gpu_available,
            'gpu_memory_gb': gpu_memory_gb,
            'system_ram_gb': psutil.virtual_memory().total / (1024**3),
            'optimal_batch_size': optimal_batch_size,  # Use hard-coded value
            'memory_threshold': 0.85,  # 85% memory usage threshold
        }

        return optimization_config

    def _calculate_dynamic_max_tokens(self, prompt: str, model_config: Dict) -> int:
        """
        Calculate dynamic max_tokens based on input length and model capabilities.

        Args:
            prompt: The input prompt
            model_config: Model configuration dictionary

        Returns:
            Optimal max_tokens for this request
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        estimated_input_tokens = len(prompt) // 4

        # Calculate available context space
        max_context = model_config.get('max_context', 32768)
        recommended_max_output = model_config.get('recommended_max_output', 16384)

        # Leave buffer for system message and response formatting
        buffer_tokens = 500
        available_for_output = max_context - estimated_input_tokens - buffer_tokens

        # Use the minimum of available space and recommended max output
        dynamic_max_tokens = min(available_for_output, recommended_max_output)

        # Ensure we have at least a reasonable minimum
        dynamic_max_tokens = max(dynamic_max_tokens, 2048)

        # Dynamic token calculation complete

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
        optimal_batch = self.gpu_optimization.get('optimal_batch_size', 3)

        # If GPU memory is getting tight, reduce batch size
        if gpu_mem['utilization'] > 0.7:
            return min(requested_batch_size, max(1, optimal_batch // 2))

        # If we have plenty of memory, allow larger batches
        if gpu_mem['utilization'] < 0.5:
            return min(requested_batch_size, optimal_batch * 2)

        return min(requested_batch_size, optimal_batch)

    @handle_llm_errors("extract with single model", max_retries=3)
    def extract_with_single_model(self, paper: Dict, model_name: str) -> ModelResult:
        """
        Extract interventions using a single model.
        
        Args:
            paper: Paper dictionary with 'pmid', 'title', and 'abstract'
            model_name: Name of the model to use
            
        Returns:
            ModelResult with interventions and metadata
        """
        pmid = paper.get('pmid', 'unknown')
        start_time = time.time()
        
        try:
            model_config = self.models[model_name]
            client = model_config['client']
            
            # Create prompt using shared service
            prompt = self.prompt_service.create_extraction_prompt(paper)

            # Calculate dynamic max_tokens
            dynamic_max_tokens = self._calculate_dynamic_max_tokens(prompt, model_config)

            # Model configuration set

            # Combine system message and user prompt
            system_message = self.prompt_service.create_system_message()
            full_prompt = f"{system_message}\n\n{prompt}"

            # Call LLM using the client's generate method
            response = client.generate(
                prompt=full_prompt,
                temperature=model_config['temperature'],
                max_tokens=dynamic_max_tokens
            )

            # Extract response
            response_text = response.get('content', '')
            extraction_time = time.time() - start_time

            
            # Parse JSON response
            interventions = parse_json_safely(response_text, f"{pmid}_{model_name}")
            
            return ModelResult(
                model_name=model_name,
                interventions=interventions,
                extraction_time=extraction_time
            )
            
        except Exception as e:
            extraction_time = time.time() - start_time
            logger.error(f"Error extracting with {model_name} from {pmid}: {e}")
            
            return ModelResult(
                model_name=model_name,
                interventions=[],
                extraction_time=extraction_time,
                error=str(e)
            )
    
    def extract_interventions(self, paper: Dict) -> Dict[str, Any]:
        """
        Extract interventions from a single paper using both models.
        
        Args:
            paper: Dictionary with 'pmid', 'title', and 'abstract'
            
        Returns:
            Dictionary with results from both models
        """
        pmid = paper.get('pmid', 'unknown')
        
        # Validate input
        if not paper.get('abstract') or not paper.get('title'):
            logger.warning(f"Paper {pmid} missing title or abstract - skipping")
            return {'pmid': pmid, 'models': {}, 'total_interventions': 0}
        
        if len(paper['abstract'].strip()) < 100:
            logger.debug(f"Paper {pmid} has very short abstract ({len(paper['abstract'])} chars) - proceeding with caution")
        
        try:
            # Update paper processing status
            self.repository_mgr.papers.update_processing_status(pmid, 'processing')
            
            # Run both models
            model_results = {}
            all_interventions = []
            
            for model_name in self.models.keys():
                logger.debug(f"Processing paper {pmid} with model {model_name}")

                result = self.extract_with_single_model(paper, model_name)
                
                # Validate and enhance interventions
                if result.interventions and not result.error:
                    validated_interventions = self._validate_and_enhance_interventions(
                        result.interventions, paper, model_name
                    )
                    result.interventions = validated_interventions
                    all_interventions.extend(validated_interventions)
                
                model_results[model_name] = result
                
                # Small delay between model calls
                time.sleep(0.5)
            
            # Compile raw results (no consensus building)
            paper_results = {
                'pmid': pmid,
                'models': model_results,
                'total_interventions': len(all_interventions),  # Count all raw extractions
                'interventions': all_interventions,  # Store all raw extractions
                'consensus_processed': False  # Flag for later consensus building
            }

            # Log results
            model_counts = {name: len(result.interventions) for name, result in model_results.items()}
            logger.info(f"Paper {pmid} processing complete: {len(all_interventions)} raw interventions from {model_counts}")
            
            # Update processing status
            status = 'processed' if all_interventions else 'processed'  # Both cases are 'processed'
            self.repository_mgr.papers.update_processing_status(pmid, status)
            
            return paper_results
            
        except Exception as e:
            logger.error(f"Error processing paper {pmid}: {e}")
            self.repository_mgr.papers.update_processing_status(pmid, 'failed')
            return {'pmid': pmid, 'models': {}, 'total_interventions': 0, 'error': str(e)}
    
    def _validate_and_enhance_interventions(self, interventions: List[Dict], 
                                          paper: Dict, model_name: str) -> List[Dict]:
        """
        Validate and enhance intervention data (JSON formatting)
        
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
                
                # Validate using intervention validator
                validated_intervention = self.validator.validate_intervention(intervention)
                
                validated.append(validated_intervention)
                
            except Exception as e:
                logger.debug(f"Intervention validation failed for paper {paper['pmid']}, model {model_name}: {e}")
                continue
        
        return validated
    
    # Removed @log_execution_time - use error_handler.py decorators instead
    def process_papers_batch(self, papers: List[Dict], save_to_db: bool = True,
                           batch_size: int = None) -> Dict[str, Any]:
        """
        Process multiple papers with both models using GPU-optimized batching.

        Args:
            papers: List of paper dictionaries
            save_to_db: Whether to save results to database
            batch_size: Number of papers to process in each batch (auto-optimized if None)

        Returns:
            Processing results summary
        """
        # Optimize batch size based on GPU memory if not specified
        if batch_size is None:
            batch_size = self.gpu_optimization.get('optimal_batch_size', 3)

        # Further optimize based on current memory usage
        optimized_batch_size = self._optimize_batch_size_for_memory(batch_size)

        gpu_mem = self._monitor_gpu_memory()
        # GPU-optimized intervention extraction starting

        # Process in batches for better resource management
        batches = batch_process(papers, optimized_batch_size)
        
        all_results = []
        failed_papers = []
        total_processed = 0
        category_counts = {cat.value: 0 for cat in InterventionType}
        model_stats = {model: {'papers': 0, 'interventions': 0} for model in self.models.keys()}
        
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
                            # Phase 2.3 Optimization: Build consensus BEFORE saving
                            # This eliminates duplicate creation entirely
                            consensus_interventions = self._build_consensus_for_paper(
                                results.get('interventions', []),
                                paper
                            )

                            # Update results with consensus interventions
                            results['consensus_interventions'] = consensus_interventions
                            results['consensus_processed'] = True
                            all_results.append(results)

                            # Update model statistics (count consensus, not raw)
                            for model_name, model_result in results.get('models', {}).items():
                                if hasattr(model_result, 'interventions') and model_result.interventions:
                                    model_stats[model_name]['papers'] += 1
                                    # Count raw for stats, but we save consensus
                                    model_stats[model_name]['interventions'] += len(model_result.interventions)

                            # Count by category (from consensus interventions)
                            for intervention in consensus_interventions:
                                category = intervention.get('intervention_category')
                                if category in category_counts:
                                    category_counts[category] += 1

                            # Save consensus interventions to database (Phase 2.3: NO DUPLICATES)
                            if save_to_db:
                                self._save_interventions_batch(consensus_interventions)
                                # Mark paper as LLM processed (Phase 2.1 optimization)
                                self.repository_mgr.db_manager.mark_paper_llm_processed(paper['pmid'])

                            pbar.set_postfix({'interventions': total_interventions + len(consensus_interventions), 'failed': len(failed_papers)})
                        else:
                            if results.get('error'):
                                logger.error(f"Error processing paper {paper['pmid']}: {results.get('error')}")
                                failed_papers.append(paper['pmid'])
                                pbar.set_postfix({'interventions': total_interventions, 'failed': len(failed_papers)})

                        total_processed += 1
                        pbar.update(1)

                        # Delay between papers
                        if i < len(batch):
                            time.sleep(1.0)  # Longer delay for dual models

                    except Exception as e:
                        logger.error(f"Failed to process paper {paper['pmid']}: {e}")
                        failed_papers.append(paper['pmid'])
                        pbar.set_postfix({'interventions': total_interventions, 'failed': len(failed_papers)})
                        pbar.update(1)

                # Delay between batches
                if batch_num < len(batches):
                    time.sleep(2.0)
        
        # Calculate totals
        total_interventions = sum(len(r.get('interventions', [])) for r in all_results)
        
        # Compile results
        results = {
            'total_papers': len(papers),
            'successful_papers': total_processed - len(failed_papers),
            'failed_papers': failed_papers,
            'total_interventions': total_interventions,
            'interventions_by_category': category_counts,
            'model_statistics': model_stats,
            'paper_results': all_results
        }
        
        # Processing summary calculated (removed verbose logging)
        
        return results
    
    def _build_consensus_for_paper(self, raw_interventions: List[Dict], paper: Dict) -> List[Dict]:
        """
        Build consensus from raw interventions for a single paper.

        Phase 2.3 Optimization: Process → Consensus → Save ONCE (no duplicates).
        Uses batch_entity_processor to merge same-paper duplicates immediately.

        Args:
            raw_interventions: All interventions extracted by all models
            paper: Source paper information

        Returns:
            List of consensus interventions (deduplicated)
        """
        if not raw_interventions:
            return []

        try:
            # Import here to avoid circular dependency
            from back_end.src.llm_processing.batch_entity_processor import create_batch_processor

            # Create processor with database connection
            with self.repository_mgr.db_manager.get_connection() as conn:
                processor = create_batch_processor(db_path=None, llm_model="qwen2.5:14b")
                processor.db = conn  # Use this connection

                # Process consensus batch (same-paper deduplication)
                consensus_interventions = processor.process_consensus_batch(
                    raw_interventions=raw_interventions,
                    paper=paper,
                    confidence_threshold=0.5
                )

                logger.debug(f"Consensus: {len(raw_interventions)} raw -> {len(consensus_interventions)} deduplicated")
                return consensus_interventions

        except Exception as e:
            logger.error(f"Consensus building failed for paper {paper.get('pmid')}: {e}")
            # Fallback: return raw interventions (better than losing data)
            logger.warning(f"Falling back to raw interventions for paper {paper.get('pmid')}")
            return raw_interventions

    def _save_interventions_batch(self, interventions: List[Dict]):
        """
        Save a batch of consensus interventions to database.

        Phase 2.3: These are already deduplicated, so no consensus_processed flag needed.
        """
        for intervention in interventions:
            try:
                # Mark as consensus processed (Phase 2.3)
                intervention['consensus_processed'] = True
                intervention['normalized'] = True  # Already normalized through consensus

                # Use standard insertion
                success = self.repository_mgr.interventions.insert_intervention(intervention)
                if success:
                    logger.debug(f"Consensus intervention saved: {intervention.get('intervention_name')}")
                else:
                    logger.error(f"Failed to save consensus intervention: {intervention.get('intervention_name')}")
            except Exception as e:
                logger.error(f"Error saving consensus intervention: {e}")
    
    
    def get_unprocessed_papers(self, limit: Optional[int] = None) -> List[Dict]:
        """Get papers that haven't been processed by ALL models yet."""
        # Get papers that haven't been processed by ANY of our models
        # This ensures we don't reprocess papers that have already been analyzed
        with self.repository_mgr.db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Create a list of model names for the query
            model_names = list(self.models.keys())
            placeholders = ','.join(['?' for _ in model_names])

            query = f'''
                SELECT DISTINCT p.*
                FROM papers p
                WHERE p.abstract IS NOT NULL
                  AND p.abstract != ''
                  AND (p.processing_status IS NULL OR p.processing_status != 'failed')
                  AND p.pmid NOT IN (
                      SELECT DISTINCT paper_id
                      FROM interventions
                      WHERE extraction_model IN ({placeholders})
                  )
                ORDER BY
                    COALESCE(p.influence_score, 0) DESC,
                    COALESCE(p.citation_count, 0) DESC,
                    p.publication_date DESC
            '''

            if limit:
                query += f' LIMIT {limit}'

            cursor.execute(query, model_names)
            columns = [desc[0] for desc in cursor.description]
            papers = [dict(zip(columns, row)) for row in cursor.fetchall()]

            return papers
    
    def process_unprocessed_papers(self, limit: Optional[int] = None,
                                 batch_size: int = None) -> Dict:
        """Process all unprocessed papers with both models."""
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

    # Consensus building logic removed - now handled by batch_entity_processor