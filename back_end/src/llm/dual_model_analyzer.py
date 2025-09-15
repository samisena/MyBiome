"""
Dual-model intervention analyzer with gemma2:9b and qwen2.5:14b.
Simplified approach that runs both models and stores all results.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

from src.data.config import config, setup_logging
from src.data.api_clients import get_llm_client
from src.data.repositories import repository_manager
from src.data.utils import (parse_json_safely, batch_process)
from src.data.error_handler import handle_llm_errors
from src.interventions.validators import intervention_validator
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
        
        # Fixed model configuration
        self.models = {
            'gemma2:9b': {
                'client': get_llm_client('gemma2:9b'),
                'temperature': 0.3,
                'max_tokens': 4096
            },
            'qwen2.5:14b': {
                'client': get_llm_client('qwen2.5:14b'),
                'temperature': 0.3,
                'max_tokens': 4096
            }
        }
        
        # Get validator and prompt service
        self.validator = intervention_validator
        self.prompt_service = prompt_service
        
        # Token tracking per model
        self.token_usage = {model_name: {'input': 0, 'output': 0, 'total': 0} 
                           for model_name in self.models.keys()}
        
        logger.info(f"Dual-model analyzer initialized with: {list(self.models.keys())}")
    

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
            
            logger.debug(f"Using {model_name} for paper {pmid}")
            
            # Call LLM API
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.prompt_service.create_system_message()
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=model_config['temperature'],
                max_tokens=model_config['max_tokens']
            )
            
            # Extract response
            response_text = response.choices[0].message.content
            extraction_time = time.time() - start_time
            
            # Track token usage
            token_usage = {}
            if hasattr(response, 'usage'):
                token_usage = {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                
                # Update running totals
                self.token_usage[model_name]['input'] += response.usage.prompt_tokens
                self.token_usage[model_name]['output'] += response.usage.completion_tokens
                self.token_usage[model_name]['total'] += response.usage.total_tokens
                
                logger.debug(f"Tokens used for {pmid} with {model_name}: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
            
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
    
    # Removed @log_execution_time - use error_handler.py decorators instead
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
            
            # Compile results
            paper_results = {
                'pmid': pmid,
                'models': model_results,
                'total_interventions': len(all_interventions),
                'interventions': all_interventions
            }
            
            # Log results
            model_counts = {name: len(result.interventions) for name, result in model_results.items()}
            logger.info(f"Paper {pmid} processed - Total interventions: {len(all_interventions)}, Model counts: {model_counts}")
            
            # Update processing status
            status = 'processed' if all_interventions else 'no_interventions'
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
                           batch_size: int = 3) -> Dict[str, Any]:
        """
        Process multiple papers with both models.
        
        Args:
            papers: List of paper dictionaries
            save_to_db: Whether to save results to database
            batch_size: Number of papers to process in each batch (small for dual models)
            
        Returns:
            Processing results summary
        """
        logger.info(f"Starting dual-model intervention extraction for {len(papers)} papers (batch size: {batch_size})")
        
        # Process in batches for better resource management
        batches = batch_process(papers, batch_size)
        
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
        """Save a batch of interventions to database."""
        for intervention in interventions:
            try:
                self.repository_mgr.interventions.insert_intervention(intervention)
            except Exception as e:
                logger.error(f"Error saving intervention: {e}")
    
    def get_unprocessed_papers(self, limit: Optional[int] = None) -> List[Dict]:
        """Get papers that haven't been processed yet."""
        # Since we're using multiple models, we need papers that haven't been processed by ANY model yet
        # For simplicity, we'll check against one of the models
        model_name = list(self.models.keys())[0]  # Use first model as reference
        return self.repository_mgr.papers.get_unprocessed_papers(
            extraction_model=model_name,
            limit=limit
        )
    
    def process_unprocessed_papers(self, limit: Optional[int] = None,
                                 batch_size: int = 3) -> Dict:
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