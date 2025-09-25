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

from src.data.config import config, setup_logging
from src.data.api_clients import get_llm_client
from src.data.repositories import repository_manager
from src.data.utils import (parse_json_safely, batch_process)
from src.data.error_handler import handle_llm_errors
from src.interventions.category_validators import category_validator, CategoryValidationError
from src.interventions.taxonomy import InterventionType
from src.llm.prompt_service import prompt_service

logger = setup_logging(__name__, 'dual_model_analyzer.log')


@dataclass
class ModelResult:
    """Result from a single model extraction."""
    model_name: str
    interventions: List[Dict]
    extraction_time: float
    token_usage: Dict
    error: Optional[str] = None


@dataclass
class ConsensusResult:
    """Result from consensus analysis of multiple model extractions."""
    intervention_name: str
    health_condition: str
    intervention_category: str
    correlation_type: str

    # Consensus fields
    consensus_confidence: float
    model_agreement: str  # 'full', 'partial', 'single', 'conflict'
    models_contributing: List[str]

    # Aggregated evidence
    avg_confidence_score: float
    confidence_range: str
    avg_correlation_strength: float
    strength_range: str

    # Best values from contributing models
    final_intervention: Dict  # The consensus intervention data
    raw_extractions: List[Dict]  # All original extractions for debugging


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
        
        # Token tracking per model
        self.token_usage = {model_name: {'input': 0, 'output': 0, 'total': 0} 
                           for model_name in self.models.keys()}
        
        # GPU optimization settings
        self.gpu_optimization = self._initialize_gpu_optimization()

        logger.info(f"Dual-model analyzer initialized with: {list(self.models.keys())}")
        logger.info(f"GPU optimization: {self.gpu_optimization}")

    def _initialize_gpu_optimization(self) -> Dict[str, Any]:
        """Initialize GPU optimization settings."""
        try:
            import psutil
            try:
                import torch
            except ImportError:
                torch = None
        except ImportError:
            logger.warning("GPU optimization libraries not available, using conservative settings")
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
            'enable_sequential_processing': gpu_memory_gb < 12  # Use sequential for smaller GPUs
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

        logger.debug(f"Dynamic token calculation: input_est={estimated_input_tokens}, "
                    f"available={available_for_output}, final={dynamic_max_tokens}")

        return dynamic_max_tokens

    def _monitor_gpu_memory(self) -> Dict[str, float]:
        """Monitor current GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                return {
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'total_gb': total,
                    'utilization': (reserved / total) if total > 0 else 0
                }
        except:
            pass
        return {'allocated_gb': 0, 'reserved_gb': 0, 'total_gb': 0, 'utilization': 0}

    def _should_use_sequential_processing(self, batch_size: int) -> bool:
        """Determine if we should use sequential model processing."""
        gpu_mem = self._monitor_gpu_memory()

        # Use sequential processing if:
        # 1. Explicitly enabled in config
        # 2. GPU memory utilization is high
        # 3. Batch size is large
        return (
            self.gpu_optimization.get('enable_sequential_processing', False) or
            gpu_mem['utilization'] > self.gpu_optimization.get('memory_threshold', 0.85) or
            batch_size > 5
        )

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

            logger.debug(f"Using {model_name} for paper {pmid} with max_tokens={dynamic_max_tokens}")

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

            # Track token usage (if available in response)
            token_usage = {}
            if 'usage' in response:
                usage = response['usage']
                token_usage = {
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0)
                }

                # Update running totals
                self.token_usage[model_name]['input'] += usage.get('prompt_tokens', 0)
                self.token_usage[model_name]['output'] += usage.get('completion_tokens', 0)
                self.token_usage[model_name]['total'] += usage.get('total_tokens', 0)
                
                logger.debug(f"Tokens used for {pmid} with {model_name}: {usage.get('prompt_tokens', 0)} in, {usage.get('completion_tokens', 0)} out")
            
            # Parse JSON response
            interventions = parse_json_safely(response_text, f"{pmid}_{model_name}")
            
            return ModelResult(
                model_name=model_name,
                interventions=interventions,
                extraction_time=extraction_time,
                token_usage=token_usage
            )
            
        except Exception as e:
            extraction_time = time.time() - start_time
            logger.error(f"Error extracting with {model_name} from {pmid}: {e}")
            
            return ModelResult(
                model_name=model_name,
                interventions=[],
                extraction_time=extraction_time,
                token_usage={},
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
            logger.warning(f"Paper {pmid} missing title or abstract")
            return {'pmid': pmid, 'models': {}, 'total_interventions': 0}
        
        if len(paper['abstract'].strip()) < 100:
            logger.warning(f"Paper {pmid} has very short abstract ({len(paper['abstract'])} chars)")
        
        try:
            # Update paper processing status
            self.repository_mgr.papers.update_processing_status(pmid, 'processing')
            
            # Run both models
            model_results = {}
            all_interventions = []
            
            for model_name in self.models.keys():
                logger.info(f"Processing {pmid} with {model_name}")
                
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
            
            # Create consensus interventions from all model results
            consensus_interventions = self._create_consensus_interventions(all_interventions, paper)

            # Compile results
            paper_results = {
                'pmid': pmid,
                'models': model_results,
                'total_interventions': len(consensus_interventions),
                'interventions': consensus_interventions,  # Use consensus for database
                'raw_interventions': all_interventions,   # Keep raw for debugging
                'consensus_summary': self._generate_consensus_summary(consensus_interventions)
            }
            
            # Log results
            model_counts = {name: len(result.interventions) for name, result in model_results.items()}
            logger.info(f"Paper {pmid} processed - Total interventions: {len(all_interventions)}, Model counts: {model_counts}")
            
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
                
                # Validate using intervention validator
                validated_intervention = self.validator.validate_intervention(intervention)
                
                validated.append(validated_intervention)
                
            except Exception as e:
                logger.warning(f"Intervention {i} validation failed for {paper['pmid']} with {model_name}: {e}")
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
        use_sequential = self._should_use_sequential_processing(optimized_batch_size)

        gpu_mem = self._monitor_gpu_memory()
        logger.info(f"Starting GPU-optimized intervention extraction for {len(papers)} papers")
        logger.info(f"Batch size: {optimized_batch_size} (requested: {batch_size})")
        logger.info(f"Sequential processing: {use_sequential}")
        logger.info(f"GPU memory: {gpu_mem['utilization']:.1%} utilized ({gpu_mem['reserved_gb']:.1f}GB/{gpu_mem['total_gb']:.1f}GB)")

        # Process in batches for better resource management
        batches = batch_process(papers, optimized_batch_size)
        
        all_results = []
        failed_papers = []
        total_processed = 0
        category_counts = {cat.value: 0 for cat in InterventionType}
        model_stats = {model: {'papers': 0, 'interventions': 0} for model in self.models.keys()}
        
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} ({len(batch)} papers)")
            
            for i, paper in enumerate(batch, 1):
                paper_num = (batch_num - 1) * batch_size + i
                logger.info(f"Processing paper {paper_num}/{len(papers)}: {paper['pmid']}")
                
                try:
                    results = self.extract_interventions(paper)
                    
                    if results.get('total_interventions', 0) > 0:
                        all_results.append(results)
                        
                        # Update model statistics
                        for model_name, model_result in results.get('models', {}).items():
                            if hasattr(model_result, 'interventions') and model_result.interventions:
                                model_stats[model_name]['papers'] += 1
                                model_stats[model_name]['interventions'] += len(model_result.interventions)
                        
                        # Count by category
                        for intervention in results.get('interventions', []):
                            category = intervention.get('intervention_category')
                            if category in category_counts:
                                category_counts[category] += 1
                        
                        # Save to database if requested
                        if save_to_db:
                            self._save_interventions_batch(results.get('interventions', []))
                    else:
                        if results.get('error'):
                            failed_papers.append(paper['pmid'])
                    
                    total_processed += 1
                    
                    # Delay between papers
                    if i < len(batch):
                        time.sleep(1.0)  # Longer delay for dual models
                        
                except Exception as e:
                    logger.error(f"Failed to process paper {paper['pmid']}: {e}")
                    failed_papers.append(paper['pmid'])
            
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
            'token_usage': dict(self.token_usage),
            'paper_results': all_results
        }
        
        # Log summary
        success_rate = (results['successful_papers'] / results['total_papers']) * 100
        logger.info(f"=== Dual-Model Extraction Summary ===")
        logger.info(f"Papers processed: {results['successful_papers']}/{results['total_papers']} ({success_rate:.1f}%)")
        logger.info(f"Total interventions found: {results['total_interventions']}")
        logger.info("Model statistics:")
        for model, stats in model_stats.items():
            logger.info(f"  {model}: {stats['interventions']} interventions from {stats['papers']} papers")
        logger.info("Interventions by category:")
        for category, count in category_counts.items():
            if count > 0:
                logger.info(f"  {category}: {count}")
        
        return results
    
    def _save_interventions_batch(self, interventions: List[Dict]):
        """Save a batch of interventions to database with normalization."""
        for intervention in interventions:
            try:
                # Use normalized insertion to automatically normalize terms
                success = self.repository_mgr.interventions.insert_intervention_normalized(intervention)
                if success:
                    logger.debug(f"Successfully saved normalized intervention: {intervention.get('intervention_name')} -> {intervention.get('health_condition')}")
                else:
                    logger.warning(f"Failed to save intervention: {intervention.get('intervention_name')}")
            except Exception as e:
                logger.error(f"Error saving intervention: {e}")
                # Fallback to standard insertion
                try:
                    self.repository_mgr.interventions.insert_intervention(intervention)
                    logger.info(f"Fallback insertion succeeded for: {intervention.get('intervention_name')}")
                except Exception as fallback_error:
                    logger.error(f"Both normalized and fallback insertion failed: {fallback_error}")
    
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
        
        logger.info(f"Found {len(papers)} unprocessed papers")
        return self.process_papers_batch(papers, save_to_db=True, batch_size=batch_size)

    def _create_consensus_interventions(self, all_interventions: List[Dict], paper: Dict) -> List[Dict]:
        """
        Create consensus interventions from multiple model extractions.

        Args:
            all_interventions: All interventions from all models
            paper: Source paper information

        Returns:
            List of consensus interventions for database storage
        """
        if not all_interventions:
            return []

        # Group interventions by (intervention_name, health_condition, category)
        grouped_interventions = defaultdict(list)

        for intervention in all_interventions:
            # Create a normalized key for grouping
            key = self._create_intervention_key(intervention)
            grouped_interventions[key].append(intervention)

        # Create consensus for each group
        consensus_interventions = []
        for key, intervention_group in grouped_interventions.items():
            consensus = self._create_consensus_intervention(intervention_group, paper)
            consensus_interventions.append(consensus)

        logger.debug(f"Paper {paper.get('pmid')}: {len(all_interventions)} raw → {len(consensus_interventions)} consensus interventions")
        return consensus_interventions

    def _create_intervention_key(self, intervention: Dict) -> str:
        """Create a normalized key for grouping similar interventions."""
        intervention_name = self._normalize_intervention_name(intervention.get('intervention_name', ''))
        health_condition = self._normalize_condition_name(intervention.get('health_condition', ''))
        category = intervention.get('intervention_category', 'unknown')

        return f"{intervention_name}|{health_condition}|{category}"

    def _normalize_intervention_name(self, name: str) -> str:
        """Normalize intervention names for comparison."""
        if not name:
            return ""

        # Convert to lowercase and remove extra spaces
        normalized = name.lower().strip()

        # Handle common variations
        synonyms = {
            'probiotics': ['probiotic', 'probiotics', 'probiotic supplements'],
            'exercise': ['physical activity', 'exercise', 'physical exercise'],
            'meditation': ['mindfulness', 'meditation', 'mindfulness meditation'],
            'omega-3': ['omega 3', 'omega-3', 'fish oil', 'omega-3 fatty acids'],
            'vitamin d': ['vitamin d3', 'vitamin d', 'cholecalciferol'],
            'magnesium': ['magnesium supplement', 'magnesium', 'mg supplement']
        }

        # Find canonical form
        for canonical, variants in synonyms.items():
            if normalized in [v.lower() for v in variants]:
                return canonical

        return normalized

    def _normalize_condition_name(self, condition: str) -> str:
        """Normalize condition names for comparison."""
        if not condition:
            return ""

        # Convert to lowercase and remove extra spaces
        normalized = condition.lower().strip()

        # Handle common variations
        synonyms = {
            'ibs': ['irritable bowel syndrome', 'ibs', 'irritable bowel'],
            'crohns disease': ['crohn\'s disease', 'crohns disease', 'crohn disease'],
            'depression': ['major depression', 'depression', 'depressive disorder'],
            'anxiety': ['anxiety disorder', 'anxiety', 'generalized anxiety'],
            'diabetes': ['type 2 diabetes', 'diabetes mellitus', 'diabetes']
        }

        # Find canonical form
        for canonical, variants in synonyms.items():
            if normalized in [v.lower() for v in variants]:
                return canonical

        return normalized

    def _create_consensus_intervention(self, intervention_group: List[Dict], paper: Dict) -> Dict:
        """
        Create a single consensus intervention from a group of similar interventions.

        Args:
            intervention_group: List of similar interventions from different models
            paper: Source paper information

        Returns:
            Consensus intervention dictionary for database storage
        """
        if len(intervention_group) == 1:
            # Single model result
            intervention = intervention_group[0].copy()
            intervention['consensus_confidence'] = 0.60
            intervention['model_agreement'] = 'single'
            intervention['models_contributing'] = [intervention.get('extraction_model', 'unknown')]
            return intervention

        # Multiple models found this intervention
        models_contributing = [i.get('extraction_model', 'unknown') for i in intervention_group]

        # Check for full agreement
        if self._check_full_agreement(intervention_group):
            # Both models agree completely
            consensus = intervention_group[0].copy()  # Use first as base
            consensus['consensus_confidence'] = 0.95
            consensus['model_agreement'] = 'full'
            consensus['models_contributing'] = models_contributing

            # Average numerical values
            consensus['confidence_score'] = self._average_scores([i.get('confidence_score') for i in intervention_group])
            consensus['correlation_strength'] = self._average_scores([i.get('correlation_strength') for i in intervention_group])

        else:
            # Partial agreement - merge intelligently
            consensus = self._merge_with_weighted_average(intervention_group)
            consensus['consensus_confidence'] = 0.75
            consensus['model_agreement'] = 'partial'
            consensus['models_contributing'] = models_contributing

        # Add metadata for tracking
        consensus['raw_extraction_count'] = len(intervention_group)
        consensus['models_used'] = ','.join(sorted(models_contributing))

        logger.debug(f"Consensus created: {consensus.get('intervention_name')} -> {consensus.get('health_condition')} "
                    f"(agreement: {consensus['model_agreement']}, confidence: {consensus['consensus_confidence']:.2f})")

        return consensus

    def _check_full_agreement(self, interventions: List[Dict]) -> bool:
        """Check if interventions represent full agreement between models."""
        if len(interventions) < 2:
            return False

        first = interventions[0]

        for intervention in interventions[1:]:
            # Check key fields for agreement
            if (intervention.get('correlation_type') != first.get('correlation_type') or
                abs((intervention.get('confidence_score', 0) or 0) - (first.get('confidence_score', 0) or 0)) > 0.2 or
                abs((intervention.get('correlation_strength', 0) or 0) - (first.get('correlation_strength', 0) or 0)) > 0.2):
                return False

        return True

    def _merge_with_weighted_average(self, interventions: List[Dict]) -> Dict:
        """Merge interventions using weighted averages and best values."""
        # Use the intervention with highest confidence as base
        base_intervention = max(interventions,
                              key=lambda x: x.get('confidence_score', 0) or 0)
        consensus = base_intervention.copy()

        # Average numerical scores
        consensus['confidence_score'] = self._average_scores([i.get('confidence_score') for i in interventions])
        consensus['correlation_strength'] = self._average_scores([i.get('correlation_strength') for i in interventions])

        # Use most common correlation type
        correlation_types = [i.get('correlation_type') for i in interventions if i.get('correlation_type')]
        if correlation_types:
            consensus['correlation_type'] = max(set(correlation_types), key=correlation_types.count)

        # Combine supporting quotes
        quotes = [i.get('supporting_quote', '') for i in interventions if i.get('supporting_quote')]
        if quotes:
            consensus['supporting_quote'] = ' | '.join(quotes)

        return consensus

    def _average_scores(self, scores: List[Optional[float]]) -> Optional[float]:
        """Calculate average of numeric scores, handling None values."""
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            return None
        return sum(valid_scores) / len(valid_scores)

    def _generate_consensus_summary(self, consensus_interventions: List[Dict]) -> Dict:
        """Generate summary statistics for consensus interventions."""
        if not consensus_interventions:
            return {}

        agreement_counts = {}
        model_usage = defaultdict(int)

        for intervention in consensus_interventions:
            agreement = intervention.get('model_agreement', 'unknown')
            agreement_counts[agreement] = agreement_counts.get(agreement, 0) + 1

            for model in intervention.get('models_contributing', []):
                model_usage[model] += 1

        return {
            'total_consensus_interventions': len(consensus_interventions),
            'agreement_breakdown': agreement_counts,
            'model_usage': dict(model_usage),
            'avg_consensus_confidence': self._average_scores([i.get('consensus_confidence') for i in consensus_interventions])
        }