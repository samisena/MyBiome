"""
Probiotic analyzer with improved architecture and efficiency.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add the current directory to sys.path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from ..data.config import config, setup_logging, LLMConfig
from ..data.api_clients import client_manager
from ..paper_collection.database_manager import database_manager
from ..data.utils import (log_execution_time, retry_with_backoff, parse_json_safely,
                   validate_correlation_data, ValidationError, batch_process, read_fulltext_content)

logger = setup_logging(__name__, 'probiotic_analyzer.log')


class ProbioticAnalyzer:
    """
    Probiotic analyzer with centralized configuration and improved efficiency.
    """
    
    def __init__(self, llm_config: Optional[LLMConfig] = None, db_manager=None):
        """
        Initialize with dependency injection.
        
        Args:
            llm_config: LLM configuration (optional, uses default if None)
            db_manager: Database manager instance (optional, uses global if None)
        """
        self.config = llm_config or config.get_llm_config()
        self.db_manager = db_manager or database_manager
        
        # Get LLM client from centralized manager
        self.client = client_manager.get_llm_client(self.config)
        
        # Use the same configuration for all papers (Llama 3.1 8b can handle both abstracts and full-text)
        # No need for separate fulltext configuration
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        logger.info(f"Enhanced analyzer initialized with model: {self.config.model_name}")
    
    def create_extraction_prompt(self, paper: Dict) -> str:
        """
        Create an optimized prompt for correlation extraction.
        
        Args:
            paper: Paper dictionary with title, abstract, and optionally full-text
            
        Returns:
            Formatted prompt string
        """
        # Check if full-text is available and prioritize it
        content_sections = []
        content_sections.append(f"Title: {paper['title']}")
        content_sections.append(f"Abstract: {paper['abstract']}")
        
        # Add full-text if available - Llama 3.1 8B handles both abstracts and full-text well
        if paper.get('has_fulltext') and paper.get('fulltext_path'):
            fulltext_content = read_fulltext_content(paper['fulltext_path'])
            if fulltext_content:
                content_sections.append(f"Full Text: {fulltext_content}")
                logger.info(f"Using full-text content for paper {paper.get('pmid', 'unknown')}")
            else:
                logger.warning(f"Could not read full-text for paper {paper.get('pmid', 'unknown')}")
        
        paper_content = "\n\n".join(content_sections)
        
        return f"""You are a biomedical expert. Extract probiotic-health correlations from this paper as a JSON array.

PAPER:
{paper_content}

Return ONLY valid JSON. No extra text. Each correlation needs these fields:
- probiotic_strain: specific strain name (e.g., "Lactobacillus rhamnosus GG") - MUST be a valid strain name, not a placeholder
- health_condition: specific condition name  
- correlation_type: "positive", "negative", "neutral", or "inconclusive"
- correlation_strength: number 0.0-1.0 or null
- confidence_score: number 0.0-1.0 or null  
- study_type: type of study or null
- sample_size: number or null
- study_duration: duration or null
- dosage: dose info or null
- supporting_quote: relevant quote from text

IMPORTANT RULES:
- ONLY extract correlations where you can identify specific probiotic strain names
- DO NOT use placeholders like "...", "various probiotics", "multiple strains", "probiotics", etc.
- If specific strain names are not mentioned, skip that correlation entirely
- Each probiotic_strain must be a real, specific strain (genus + species + strain identifier when available)
- Examples of valid strain names: "Lactobacillus rhamnosus GG", "Bifidobacterium longum BB536", "Lactobacillus acidophilus"
- Examples of INVALID entries to avoid: "...", "probiotics", "various strains", "multiple probiotics"

Example of valid extraction:
[{{"probiotic_strain":"Lactobacillus rhamnosus GG","health_condition":"irritable bowel syndrome","correlation_type":"positive","correlation_strength":0.8,"confidence_score":0.9,"study_type":"RCT","sample_size":100,"study_duration":"8 weeks","dosage":"10^9 CFU/day","supporting_quote":"LGG significantly improved IBS symptoms"}}]

Return [] if no specific strain names are found or no correlations can be extracted."""
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def extract_correlations(self, paper: Dict) -> List[Dict]:
        """
        Extract correlations from a single paper using the LLM.
        
        Args:
            paper: Dictionary with 'pmid', 'title', and 'abstract'
            
        Returns:
            List of validated correlations
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
            self.db_manager.update_paper_processing_status(pmid, 'processing')
            
            # Create prompt
            prompt = self.create_extraction_prompt(paper)
            
            # Use single model (Llama 3.1 8b) for all papers
            logger.debug(f"Using {self.config.model_name} for paper {pmid}")
            
            # Call LLM API
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise biomedical data extraction system. Return only valid JSON arrays with no additional formatting or text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Extract response
            response_text = response.choices[0].message.content
            
            # Check for token limit truncation
            if hasattr(response.choices[0], 'finish_reason'):
                if response.choices[0].finish_reason == 'length':
                    logger.warning(f"Response truncated at max_tokens for {pmid}")
            
            # Track token usage
            if hasattr(response, 'usage'):
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                logger.debug(f"Tokens used for {pmid}: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
            
            # Parse and validate JSON response
            correlations = parse_json_safely(response_text, pmid)
            
            # Validate and enhance correlations
            validated_correlations = self._validate_and_enhance_correlations(correlations, paper, self.config)
            
            if validated_correlations:
                logger.info(f"Extracted {len(validated_correlations)} correlations from {pmid}")
                # Update processing status to processed
                self.db_manager.update_paper_processing_status(pmid, 'processed')
            else:
                logger.info(f"No correlations extracted from {pmid}")
                # Still mark as processed since we attempted extraction
                self.db_manager.update_paper_processing_status(pmid, 'processed')
            
            return validated_correlations
            
        except Exception as e:
            logger.error(f"Error extracting correlations from {pmid}: {e}")
            # Mark as failed
            self.db_manager.update_paper_processing_status(pmid, 'failed')
            return []
    
    def _validate_and_enhance_correlations(self, correlations: List[Dict], 
                                         paper: Dict, config_used=None) -> List[Dict]:
        """
        Validate and enhance correlation data.
        
        Args:
            correlations: Raw correlations from LLM
            paper: Source paper information
            
        Returns:
            List of validated and enhanced correlations
        """
        validated = []
        
        for i, corr in enumerate(correlations):
            try:
                # Add metadata
                corr['paper_id'] = paper['pmid']
                corr['extraction_model'] = (config_used or self.config).model_name
                
                # Validate using utility function
                validated_corr = validate_correlation_data(corr)
                
                # Additional validation
                self._validate_supporting_quote(validated_corr, paper.get('abstract', ''))
                
                validated.append(validated_corr)
                
            except ValidationError as e:
                logger.warning(f"Correlation {i} validation failed for {paper['pmid']}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error validating correlation {i} for {paper['pmid']}: {e}")
                continue
        
        return validated
    
    def _validate_supporting_quote(self, correlation: Dict, abstract: str):
        """Validate that supporting quote exists in abstract."""
        supporting_quote = correlation.get('supporting_quote', '')
        
        if supporting_quote and abstract:
            # Simple check - could be enhanced with fuzzy matching
            if supporting_quote.lower() not in abstract.lower():
                logger.warning(f"Supporting quote may not match abstract: {supporting_quote[:50]}...")
    
    @log_execution_time
    def process_papers_batch(self, papers: List[Dict], save_to_db: bool = True,
                           batch_size: int = 10) -> Dict:
        """
        Process multiple papers with improved batch handling and error recovery.
        
        Args:
            papers: List of paper dictionaries
            save_to_db: Whether to save results to database
            batch_size: Number of papers to process in each batch
            
        Returns:
            Processing results summary
        """
        logger.info(f"Starting extraction for {len(papers)} papers (batch size: {batch_size})")
        
        # Process in batches for better resource management
        batches = batch_process(papers, batch_size)
        
        all_correlations = []
        failed_papers = []
        total_processed = 0
        
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} ({len(batch)} papers)")
            
            batch_correlations = []
            
            for i, paper in enumerate(batch, 1):
                paper_num = (batch_num - 1) * batch_size + i
                logger.info(f"Processing paper {paper_num}/{len(papers)}: {paper['pmid']}")
                
                try:
                    correlations = self.extract_correlations(paper)
                    
                    if correlations:
                        batch_correlations.extend(correlations)
                        
                        # Save to database if requested
                        if save_to_db:
                            self._save_correlations_batch(correlations)
                    else:
                        failed_papers.append(paper['pmid'])
                    
                    total_processed += 1
                    
                    # Small delay between papers to avoid overwhelming API
                    if i < len(batch):
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Failed to process paper {paper['pmid']}: {e}")
                    failed_papers.append(paper['pmid'])
            
            all_correlations.extend(batch_correlations)
            
            # Delay between batches
            if batch_num < len(batches):
                time.sleep(1.0)
        
        # Compile results
        results = {
            'total_papers': len(papers),
            'successful_papers': total_processed - len(failed_papers),
            'failed_papers': failed_papers,
            'total_correlations': len(all_correlations),
            'correlations': all_correlations,
            'token_usage': {
                'input_tokens': self.total_input_tokens,
                'output_tokens': self.total_output_tokens,
                'total_tokens': self.total_input_tokens + self.total_output_tokens
            }
        }
        
        # Log summary
        success_rate = (results['successful_papers'] / results['total_papers']) * 100
        logger.info(f"=== Extraction Summary ===")
        logger.info(f"Papers processed: {results['successful_papers']}/{results['total_papers']} ({success_rate:.1f}%)")
        logger.info(f"Correlations found: {results['total_correlations']}")
        logger.info(f"Token usage: {results['token_usage']['total_tokens']} total")
        
        return results
    
    def _save_correlations_batch(self, correlations: List[Dict]):
        """Save a batch of correlations to database."""
        for corr in correlations:
            try:
                self.db_manager.insert_correlation(corr)
            except Exception as e:
                logger.error(f"Error saving correlation: {e}")
    
    def get_unprocessed_papers(self, limit: Optional[int] = None) -> List[Dict]:
        """Get papers that haven't been processed by this model yet."""
        return self.db_manager.get_papers_for_processing(
            extraction_model=self.config.model_name,
            limit=limit
        )
    
    def process_unprocessed_papers(self, limit: Optional[int] = None,
                                 batch_size: int = 10) -> Dict:
        """Process all unprocessed papers for this model."""
        papers = self.get_unprocessed_papers(limit)
        
        if not papers:
            logger.info("No unprocessed papers found")
            return {
                'total_papers': 0,
                'successful_papers': 0,
                'failed_papers': [],
                'total_correlations': 0,
                'correlations': []
            }
        
        logger.info(f"Found {len(papers)} unprocessed papers")
        return self.process_papers_batch(papers, save_to_db=True, batch_size=batch_size)