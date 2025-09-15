"""
Intervention analyzer with improved architecture and multi-category support.
Extracts various types of health interventions from research papers.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

from src.data.config import config, setup_logging, LLMConfig
from src.data.api_clients import get_llm_client
from src.data.repositories import repository_manager
from src.data.utils import (parse_json_safely, batch_process)
from src.data.error_handler import handle_llm_errors
from src.interventions.validators import intervention_validator
from src.llm.prompt_service import prompt_service

logger = setup_logging(__name__, 'intervention_analyzer.log')


class InterventionAnalyzer:
    """
    Multi-intervention analyzer with centralized configuration and improved efficiency.
    Extracts exercise, diet, supplement, medication, therapy, and lifestyle interventions.
    """
    
    def __init__(self, llm_config: Optional[LLMConfig] = None, repository_mgr=None):
        """
        Initialize with dependency injection.
        
        Args:
            llm_config: LLM configuration (optional, uses default if None)
            repository_mgr: Repository manager instance (optional, uses global if None)
        """
        self.config = llm_config or LLMConfig()
        self.repository_mgr = repository_mgr or repository_manager
        
        # Get LLM client from centralized function
        self.client = get_llm_client(self.config.model_name)
        
        # Get validator and prompt service
        self.validator = intervention_validator
        self.prompt_service = prompt_service
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        logger.info(f"Intervention analyzer initialized with model: {self.config.model_name}")

    @handle_llm_errors("extract interventions", max_retries=3)
    def extract_interventions(self, paper: Dict) -> List[Dict]:
        """
        Extract interventions from a single paper using the LLM.
        
        Args:
            paper: Dictionary with 'pmid', 'title', and 'abstract'
            
        Returns:
            List of validated interventions
        """
        pmid = paper.get('pmid', 'unknown')
        
        # Validate input
        if not paper.get('abstract') or not paper.get('title'):
            logger.warning(f"Paper {pmid} missing title or abstract")
            return []
        
        if len(paper['abstract'].strip()) < 100:
            logger.warning(f"Paper {pmid} has very short abstract ({len(paper['abstract'])} chars)")
            return []
        
        try:
            # Update paper processing status
            self.repository_mgr.papers.update_processing_status(pmid, 'processing')
            
            # Create prompt using shared service
            prompt = self.prompt_service.create_extraction_prompt(paper)
            
            logger.debug(f"Using {self.config.model_name} for paper {pmid}")
            
            # Call LLM API
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.prompt_service.create_system_message()
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Extract response
            response_text = response.choices[0].message.content
            
            # Track token usage
            if hasattr(response, 'usage'):
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                logger.debug(f"Tokens used for {pmid}: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
            
            # Parse JSON response
            interventions = parse_json_safely(response_text, pmid)
            
            # Validate and filter interventions
            valid_interventions = []
            for intervention in interventions:
                try:
                    validated = self.validator.validate_intervention(intervention)
                    if validated:
                        validated['paper_id'] = pmid
                        validated['extraction_model'] = self.config.model_name
                        valid_interventions.append(validated)
                except Exception as e:
                    logger.warning(f"Intervention validation failed for {pmid}: {e}")
            
            # Update paper status
            status = 'completed' if valid_interventions else 'no_interventions'
            self.repository_mgr.papers.update_processing_status(pmid, status)
            
            logger.info(f"Extracted {len(valid_interventions)} valid interventions from {pmid}")
            return valid_interventions
            
        except Exception as e:
            logger.error(f"Error extracting interventions from {pmid}: {e}")
            self.repository_mgr.papers.update_processing_status(pmid, 'failed')
            return []
    
    # Removed @log_execution_time - use error_handler.py decorators instead
    def process_papers_batch(self, papers: List[Dict], save_to_db: bool = True,
                           show_progress: bool = True) -> Dict:
        """
        Process a batch of papers for intervention extraction.
        
        Args:
            papers: List of paper dictionaries
            save_to_db: Whether to save results to database
            show_progress: Whether to show processing progress
            
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Processing batch of {len(papers)} papers")
        
        all_interventions = []
        successful_papers = 0
        failed_papers = []
        
        for i, paper in enumerate(papers):
            pmid = paper.get('pmid', f'unknown_{i}')
            
            try:
                if show_progress and i % 5 == 0:
                    logger.info(f"Progress: {i+1}/{len(papers)} papers processed")
                
                # Extract interventions
                interventions = self.extract_interventions(paper)
                
                if interventions:
                    all_interventions.extend(interventions)
                    successful_papers += 1
                    
                    # Save to database if requested
                    if save_to_db:
                        self._save_interventions_batch(interventions)
                else:
                    logger.debug(f"No interventions found in {pmid}")
                    
            except Exception as e:
                logger.error(f"Failed to process paper {pmid}: {e}")
                failed_papers.append({'pmid': pmid, 'error': str(e)})
        
        # Compile results
        results = {
            'total_papers': len(papers),
            'successful_papers': successful_papers,
            'failed_papers': failed_papers,
            'total_interventions': len(all_interventions),
            'interventions': all_interventions,
            'token_usage': {
                'input_tokens': self.total_input_tokens,
                'output_tokens': self.total_output_tokens,
                'total_tokens': self.total_input_tokens + self.total_output_tokens
            }
        }
        
        # Group interventions by category
        categories = {}
        for intervention in all_interventions:
            cat = intervention.get('intervention_category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        results['interventions_by_category'] = categories
        
        logger.info(f"Batch processing complete: {successful_papers}/{len(papers)} papers, {len(all_interventions)} interventions")
        return results
    
    def _save_interventions_batch(self, interventions: List[Dict]):
        """Save a batch of interventions to the database."""
        for intervention in interventions:
            try:
                self.repository_mgr.interventions.insert_intervention(intervention)
            except Exception as e:
                logger.error(f"Error saving intervention: {e}")
    
    def get_unprocessed_papers(self, limit: Optional[int] = None) -> List[Dict]:
        """Get papers that haven't been processed yet."""
        return self.repository_mgr.papers.get_unprocessed_papers(
            extraction_model=self.config.model_name,
            limit=limit
        )
    
    def process_unprocessed_papers(self, limit: Optional[int] = None,
                                 batch_size: int = 10) -> Dict:
        """
        Process all unprocessed papers in batches.
        
        Args:
            limit: Maximum number of papers to process
            batch_size: Number of papers to process in each batch
            
        Returns:
            Dictionary with overall processing results
        """
        papers_to_process = self.get_unprocessed_papers(limit)
        
        if not papers_to_process:
            logger.info("No unprocessed papers found")
            return {
                'total_papers': 0,
                'successful_papers': 0,
                'failed_papers': [],
                'total_interventions': 0,
                'interventions_by_category': {}
            }
        
        logger.info(f"Found {len(papers_to_process)} papers to process")
        
        # Process in batches for better memory management
        all_results = {
            'total_papers': 0,
            'successful_papers': 0,
            'failed_papers': [],
            'total_interventions': 0,
            'interventions_by_category': {}
        }
        
        for batch in batch_process(papers_to_process, batch_size):
            batch_results = self.process_papers_batch(batch, save_to_db=True)
            
            # Merge results
            all_results['total_papers'] += batch_results['total_papers']
            all_results['successful_papers'] += batch_results['successful_papers']
            all_results['failed_papers'].extend(batch_results['failed_papers'])
            all_results['total_interventions'] += batch_results['total_interventions']
            
            # Merge category counts
            for cat, count in batch_results.get('interventions_by_category', {}).items():
                all_results['interventions_by_category'][cat] = \
                    all_results['interventions_by_category'].get(cat, 0) + count
        
        return all_results