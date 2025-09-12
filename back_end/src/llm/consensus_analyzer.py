"""
Multi-LLM consensus-based correlation analyzer.
Processes papers with multiple LLMs and determines consensus correlations.
"""

import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import sys
from pathlib import Path

# Add the current directory to sys.path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from ..data.config import config, setup_logging, MultiLLMConfig, LLMConfig
from ..data.api_clients import client_manager
from ..paper_collection.database_manager import database_manager
from ..data.utils import (log_execution_time, retry_with_backoff, parse_json_safely,
                   validate_correlation_data, ValidationError, batch_process, read_fulltext_content)

logger = setup_logging(__name__, 'consensus_analyzer.log')


@dataclass
class ExtractionResult:
    """Result from a single LLM extraction."""
    model_name: str
    correlations: List[Dict]
    extraction_time: float
    token_usage: Dict
    error: Optional[str] = None


@dataclass 
class ConsensusResult:
    """Result of consensus analysis between multiple LLMs."""
    paper_id: str
    agreed_correlations: List[Dict]  # Correlations all models agreed on
    conflicting_correlations: List[Dict]  # Correlations with disagreements
    model_specific_correlations: Dict[str, List[Dict]]  # Correlations only found by specific models
    consensus_status: str  # 'full_agreement', 'partial_agreement', 'no_agreement'
    needs_review: bool
    review_reasons: List[str]


class MultiLLMConsensusAnalyzer:
    """
    Analyzer that processes papers with multiple LLMs and determines consensus.
    Designed for easy addition/removal of LLMs.
    """
    
    def __init__(self, multi_llm_config: Optional[MultiLLMConfig] = None, db_manager=None):
        """
        Initialize with multi-LLM configuration.
        
        Args:
            multi_llm_config: Multi-LLM configuration (optional, uses default if None)
            db_manager: Database manager instance (optional, uses global if None)
        """
        self.config = multi_llm_config or config.multi_llm
        self.db_manager = db_manager or database_manager
        
        # Initialize LLM clients for each model
        self.llm_clients = {}
        for llm_config in self.config.models:
            client = client_manager.get_llm_client(llm_config)
            self.llm_clients[llm_config.model_name] = {
                'client': client,
                'config': llm_config
            }
        
        # Token tracking per model
        self.token_usage = {model_name: {'input': 0, 'output': 0} 
                           for model_name in self.llm_clients.keys()}
        
        logger.info(f"Consensus analyzer initialized with models: {list(self.llm_clients.keys())}")
    
    def create_extraction_prompt(self, paper: Dict) -> str:
        """
        Create an optimized prompt for correlation extraction.
        Same prompt used for all models to ensure fair comparison.
        """
        # Check if full-text is available and prioritize it
        content_sections = []
        content_sections.append(f"Title: {paper['title']}")
        content_sections.append(f"Abstract: {paper['abstract']}")
        
        # Add full-text if available
        if paper.get('has_fulltext') and paper.get('fulltext_path'):
            fulltext_content = read_fulltext_content(paper['fulltext_path'])
            if fulltext_content:
                content_sections.append(f"Full Text: {fulltext_content}")
                logger.debug(f"Using full-text content for paper {paper.get('pmid', 'unknown')}")
        
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
    def extract_with_single_model(self, paper: Dict, model_name: str) -> ExtractionResult:
        """
        Extract correlations using a single LLM model.
        
        Args:
            paper: Paper dictionary with 'pmid', 'title', and 'abstract'
            model_name: Name of the LLM model to use
            
        Returns:
            ExtractionResult with correlations and metadata
        """
        pmid = paper.get('pmid', 'unknown')
        start_time = time.time()
        
        try:
            client_info = self.llm_clients[model_name]
            client = client_info['client']
            llm_config = client_info['config']
            
            # Create prompt
            prompt = self.create_extraction_prompt(paper)
            
            logger.debug(f"Extracting with {model_name} for paper {pmid}")
            
            # Call LLM API
            response = client.chat.completions.create(
                model=llm_config.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise biomedical data extraction system. Return only valid JSON arrays with no additional formatting or text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
            )
            
            # Extract response
            response_text = response.choices[0].message.content
            extraction_time = time.time() - start_time
            
            # Track token usage
            token_usage = {'input': 0, 'output': 0, 'total': 0}
            if hasattr(response, 'usage'):
                token_usage['input'] = response.usage.prompt_tokens
                token_usage['output'] = response.usage.completion_tokens 
                token_usage['total'] = response.usage.prompt_tokens + response.usage.completion_tokens
                
                # Update global tracking
                self.token_usage[model_name]['input'] += token_usage['input']
                self.token_usage[model_name]['output'] += token_usage['output']
                
                logger.debug(f"Tokens for {model_name} on {pmid}: {token_usage['input']} in, {token_usage['output']} out")
            
            # Parse and validate JSON response
            correlations = parse_json_safely(response_text, f"{pmid}-{model_name}")
            
            # Validate and enhance correlations
            validated_correlations = self._validate_and_enhance_correlations(correlations, paper, model_name)
            
            # Save individual extraction to database
            self._save_extraction(pmid, model_name, validated_correlations, response_text, extraction_time)
            
            return ExtractionResult(
                model_name=model_name,
                correlations=validated_correlations,
                extraction_time=extraction_time,
                token_usage=token_usage,
                error=None
            )
            
        except Exception as e:
            extraction_time = time.time() - start_time
            logger.error(f"Error extracting with {model_name} for {pmid}: {e}")
            
            return ExtractionResult(
                model_name=model_name,
                correlations=[],
                extraction_time=extraction_time,
                token_usage={'input': 0, 'output': 0, 'total': 0},
                error=str(e)
            )
    
    def _validate_and_enhance_correlations(self, correlations: List[Dict], 
                                         paper: Dict, model_name: str) -> List[Dict]:
        """Validate and enhance correlation data."""
        validated = []
        
        for i, corr in enumerate(correlations):
            try:
                # Add metadata
                corr['paper_id'] = paper['pmid']
                corr['extraction_model'] = model_name
                
                # Validate using utility function
                validated_corr = validate_correlation_data(corr)
                validated.append(validated_corr)
                
            except ValidationError as e:
                logger.warning(f"Correlation {i} validation failed for {paper['pmid']} ({model_name}): {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error validating correlation {i} for {paper['pmid']} ({model_name}): {e}")
                continue
        
        return validated
    
    def _save_extraction(self, pmid: str, model_name: str, correlations: List[Dict], 
                        raw_response: str, extraction_time: float):
        """Save individual LLM extraction to database."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                for corr in correlations:
                    cursor.execute('''
                        INSERT OR REPLACE INTO correlation_extractions (
                            paper_id, extraction_model, probiotic_strain, health_condition,
                            correlation_type, correlation_strength, confidence_score,
                            supporting_quote, sample_size, study_duration, study_type, dosage,
                            raw_response
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        pmid, model_name, corr.get('probiotic_strain'), corr.get('health_condition'),
                        corr.get('correlation_type'), corr.get('correlation_strength'), 
                        corr.get('confidence_score'), corr.get('supporting_quote'),
                        corr.get('sample_size'), corr.get('study_duration'),
                        corr.get('study_type'), corr.get('dosage'), raw_response
                    ))
                
                conn.commit()
                logger.debug(f"Saved {len(correlations)} extractions for {pmid} ({model_name})")
                
        except Exception as e:
            logger.error(f"Error saving extraction for {pmid} ({model_name}): {e}")
    
    def _correlations_match(self, corr1: Dict, corr2: Dict) -> bool:
        """
        Determine if two correlations represent the same finding.
        They match if they have the same strain and health condition.
        """
        return (corr1.get('probiotic_strain', '').lower().strip() == 
                corr2.get('probiotic_strain', '').lower().strip() and
                corr1.get('health_condition', '').lower().strip() == 
                corr2.get('health_condition', '').lower().strip())
    
    def _correlations_agree(self, corr1: Dict, corr2: Dict) -> bool:
        """
        Determine if two matching correlations agree on their findings.
        They agree if they have the same correlation_type.
        """
        return corr1.get('correlation_type') == corr2.get('correlation_type')
    
    def _merge_correlations(self, correlations: List[Dict]) -> Dict:
        """
        Merge multiple correlations that represent the same finding.
        Takes average of numerical values and combines text fields.
        """
        if not correlations:
            return {}
        
        if len(correlations) == 1:
            return correlations[0]
        
        # Start with first correlation as base
        merged = correlations[0].copy()
        
        # Average numerical fields
        numerical_fields = ['correlation_strength', 'confidence_score', 'sample_size']
        for field in numerical_fields:
            values = [corr.get(field) for corr in correlations if corr.get(field) is not None]
            if values:
                merged[field] = sum(values) / len(values)
        
        # Combine text fields
        text_fields = ['supporting_quote', 'study_type', 'study_duration', 'dosage']
        for field in text_fields:
            values = [corr.get(field) for corr in correlations if corr.get(field)]
            if values:
                # Take the longest/most detailed value
                merged[field] = max(values, key=len)
        
        return merged
    
    def analyze_consensus(self, extractions: List[ExtractionResult]) -> ConsensusResult:
        """
        Analyze consensus between multiple LLM extractions.
        
        Args:
            extractions: List of ExtractionResult from different models
            
        Returns:
            ConsensusResult with consensus analysis
        """
        if not extractions:
            return ConsensusResult(
                paper_id="unknown",
                agreed_correlations=[],
                conflicting_correlations=[],
                model_specific_correlations={},
                consensus_status='no_extractions',
                needs_review=True,
                review_reasons=['no_successful_extractions']
            )
        
        paper_id = extractions[0].correlations[0]['paper_id'] if extractions[0].correlations else "unknown"
        
        # Get all correlations from all models
        all_correlations = []
        model_correlations = {}
        
        for extraction in extractions:
            if extraction.error is None:
                model_correlations[extraction.model_name] = extraction.correlations
                all_correlations.extend(extraction.correlations)
        
        if not all_correlations:
            # Check if all models successfully ran but found no correlations (good consensus)
            # vs some models failed to extract (bad situation requiring review)
            successful_models = [extraction.model_name for extraction in extractions if extraction.error is None]
            
            if len(successful_models) == len(self.config.models):
                # All models successfully ran and found no correlations - this is perfect consensus
                return ConsensusResult(
                    paper_id=paper_id,
                    agreed_correlations=[],
                    conflicting_correlations=[],
                    model_specific_correlations=model_correlations,
                    consensus_status='no_correlations_consensus',
                    needs_review=False,  # Both models agree: no correlations
                    review_reasons=[]
                )
            else:
                # Some models failed to run - this needs review
                return ConsensusResult(
                    paper_id=paper_id,
                    agreed_correlations=[],
                    conflicting_correlations=[],
                    model_specific_correlations=model_correlations,
                    consensus_status='extraction_failures',
                    needs_review=True,
                    review_reasons=['some_models_failed_to_extract']
                )
        
        # Group correlations by strain-condition pairs
        correlation_groups = {}
        
        for model_name, correlations in model_correlations.items():
            for corr in correlations:
                key = f"{corr.get('probiotic_strain', '').lower().strip()}|{corr.get('health_condition', '').lower().strip()}"
                if key not in correlation_groups:
                    correlation_groups[key] = {}
                correlation_groups[key][model_name] = corr
        
        # Analyze each group for consensus
        agreed_correlations = []
        conflicting_correlations = []
        model_specific = {model: [] for model in model_correlations.keys()}
        review_reasons = []
        
        for key, group in correlation_groups.items():
            models_in_group = list(group.keys())
            correlations_in_group = list(group.values())
            
            if len(models_in_group) == 1:
                # Only one model found this correlation
                model_name = models_in_group[0]
                model_specific[model_name].append(correlations_in_group[0])
                review_reasons.append(f"single_model_finding_{model_name}")
                
            elif len(models_in_group) == len(self.config.models):
                # All models found this correlation - check if they agree
                correlation_types = [corr.get('correlation_type') for corr in correlations_in_group]
                
                if len(set(correlation_types)) == 1:
                    # All models agree on correlation type
                    merged_correlation = self._merge_correlations(correlations_in_group)
                    merged_correlation['agreed_by_models'] = json.dumps(models_in_group)
                    agreed_correlations.append(merged_correlation)
                else:
                    # Models disagree on correlation type
                    conflict_info = {
                        'strain_condition': key,
                        'model_findings': {model: corr.get('correlation_type') for model, corr in group.items()},
                        'correlations': correlations_in_group
                    }
                    conflicting_correlations.append(conflict_info)
                    review_reasons.append(f"type_disagreement_{key}")
            
            else:
                # Partial agreement (some but not all models found this)
                for model_name, corr in group.items():
                    model_specific[model_name].append(corr)
                review_reasons.append(f"partial_agreement_{key}")
        
        # Determine consensus status
        if agreed_correlations and not conflicting_correlations and not any(model_specific.values()):
            consensus_status = 'full_agreement'
            needs_review = False
        elif agreed_correlations:
            consensus_status = 'partial_agreement' 
            needs_review = True
        else:
            consensus_status = 'no_agreement'
            needs_review = True
        
        return ConsensusResult(
            paper_id=paper_id,
            agreed_correlations=agreed_correlations,
            conflicting_correlations=conflicting_correlations,
            model_specific_correlations=model_specific,
            consensus_status=consensus_status,
            needs_review=needs_review,
            review_reasons=review_reasons
        )
    
    def _save_consensus_result(self, result: ConsensusResult):
        """Save consensus results to database."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Save agreed correlations to final correlations table
                for corr in result.agreed_correlations:
                    cursor.execute('''
                        INSERT OR REPLACE INTO correlations (
                            paper_id, probiotic_strain, health_condition, correlation_type,
                            correlation_strength, confidence_score, supporting_quote,
                            sample_size, study_duration, study_type, dosage,
                            extraction_model, validation_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result.paper_id, corr.get('probiotic_strain'), corr.get('health_condition'),
                        corr.get('correlation_type'), corr.get('correlation_strength'),
                        corr.get('confidence_score'), corr.get('supporting_quote'),
                        corr.get('sample_size'), corr.get('study_duration'),
                        corr.get('study_type'), corr.get('dosage'),
                        corr.get('agreed_by_models', 'consensus'), 'verified'
                    ))
                
                # Save consensus metadata for all strain-condition pairs
                all_pairs = set()
                
                # Add agreed pairs
                for corr in result.agreed_correlations:
                    pair = (corr.get('probiotic_strain'), corr.get('health_condition'))
                    all_pairs.add(pair)
                
                # Add conflicting pairs
                for conflict in result.conflicting_correlations:
                    for corr in conflict['correlations']:
                        pair = (corr.get('probiotic_strain'), corr.get('health_condition'))
                        all_pairs.add(pair)
                
                # Add model-specific pairs
                for model_correlations in result.model_specific_correlations.values():
                    for corr in model_correlations:
                        pair = (corr.get('probiotic_strain'), corr.get('health_condition'))
                        all_pairs.add(pair)
                
                # Save consensus tracking records
                for strain, condition in all_pairs:
                    # Determine which category this pair falls into
                    agreed_models = []
                    conflicting_models = []
                    
                    # Check if in agreed correlations
                    in_agreed = any(
                        corr.get('probiotic_strain') == strain and corr.get('health_condition') == condition
                        for corr in result.agreed_correlations
                    )
                    
                    if in_agreed:
                        agreed_models = [model for model in self.llm_clients.keys()]
                        status = 'agreed'
                        final_corr = next(
                            corr for corr in result.agreed_correlations
                            if corr.get('probiotic_strain') == strain and corr.get('health_condition') == condition
                        )
                    else:
                        # Check conflicts and model-specific
                        status = 'conflict'
                        final_corr = {}
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO correlation_consensus (
                            paper_id, probiotic_strain, health_condition, consensus_status,
                            agreed_by_models, conflicting_models, needs_review, review_reason,
                            final_correlation_type, final_correlation_strength, final_confidence_score,
                            final_supporting_quote, final_sample_size, final_study_duration,
                            final_study_type, final_dosage
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        result.paper_id, strain, condition, status,
                        json.dumps(agreed_models), json.dumps(conflicting_models),
                        result.needs_review, '; '.join(result.review_reasons),
                        final_corr.get('correlation_type'), final_corr.get('correlation_strength'),
                        final_corr.get('confidence_score'), final_corr.get('supporting_quote'),
                        final_corr.get('sample_size'), final_corr.get('study_duration'),
                        final_corr.get('study_type'), final_corr.get('dosage')
                    ))
                
                conn.commit()
                logger.info(f"Saved consensus results for paper {result.paper_id}: {len(result.agreed_correlations)} agreed, {len(result.conflicting_correlations)} conflicts")
                
        except Exception as e:
            logger.error(f"Error saving consensus results for {result.paper_id}: {e}")
    
    @log_execution_time
    def process_paper_with_consensus(self, paper: Dict) -> ConsensusResult:
        """
        Process a single paper with all configured LLMs and determine consensus.
        
        Args:
            paper: Paper dictionary with 'pmid', 'title', and 'abstract'
            
        Returns:
            ConsensusResult with consensus analysis
        """
        pmid = paper.get('pmid', 'unknown')
        logger.info(f"Processing paper {pmid} with {len(self.config.models)} models for consensus")
        
        # Validate input
        if not paper.get('abstract') or not paper.get('title'):
            logger.warning(f"Paper {pmid} missing title or abstract")
            return ConsensusResult(
                paper_id=pmid,
                agreed_correlations=[],
                conflicting_correlations=[],
                model_specific_correlations={},
                consensus_status='invalid_paper',
                needs_review=True,
                review_reasons=['missing_title_or_abstract']
            )
        
        # Update paper processing status
        self.db_manager.update_paper_processing_status(pmid, 'processing')
        
        try:
            # Extract correlations with each model
            extractions = []
            for model_config in self.config.models:
                extraction = self.extract_with_single_model(paper, model_config.model_name)
                extractions.append(extraction)
                
                # Small delay between models
                time.sleep(0.5)
            
            # Analyze consensus
            consensus_result = self.analyze_consensus(extractions)
            
            # Save results to database
            self._save_consensus_result(consensus_result)
            
            # Update paper processing status
            if consensus_result.needs_review:
                self.db_manager.update_paper_processing_status(pmid, 'needs_review')
            else:
                self.db_manager.update_paper_processing_status(pmid, 'processed')
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"Error processing paper {pmid}: {e}")
            self.db_manager.update_paper_processing_status(pmid, 'failed')
            
            return ConsensusResult(
                paper_id=pmid,
                agreed_correlations=[],
                conflicting_correlations=[],
                model_specific_correlations={},
                consensus_status='processing_error',
                needs_review=True,
                review_reasons=[f'processing_error: {str(e)}']
            )
    
    def get_unprocessed_papers(self, limit: Optional[int] = None) -> List[Dict]:
        """Get papers that haven't been processed by the consensus system yet."""
        return self.db_manager.get_papers_for_processing(
            extraction_model='consensus_system',  # Use special marker for consensus processing
            limit=limit
        )
    
    @log_execution_time
    def process_papers_batch(self, papers: List[Dict], batch_size: int = 5) -> Dict:
        """
        Process multiple papers with consensus analysis.
        Smaller batch size due to multiple LLM calls per paper.
        
        Args:
            papers: List of paper dictionaries
            batch_size: Number of papers to process in each batch (default smaller for multi-LLM)
            
        Returns:
            Processing results summary
        """
        logger.info(f"Starting consensus processing for {len(papers)} papers (batch size: {batch_size})")
        
        # Process in batches
        batches = batch_process(papers, batch_size)
        
        all_results = []
        failed_papers = []
        total_agreed_correlations = 0
        total_conflicts = 0
        
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Processing consensus batch {batch_num}/{len(batches)} ({len(batch)} papers)")
            
            for i, paper in enumerate(batch, 1):
                paper_num = (batch_num - 1) * batch_size + i
                logger.info(f"Processing paper {paper_num}/{len(papers)}: {paper['pmid']}")
                
                try:
                    result = self.process_paper_with_consensus(paper)
                    all_results.append(result)
                    
                    total_agreed_correlations += len(result.agreed_correlations)
                    total_conflicts += len(result.conflicting_correlations)
                    
                    if result.consensus_status in ['processing_error', 'invalid_paper']:
                        failed_papers.append(paper['pmid'])
                    
                    # Delay between papers (multiple LLM calls per paper)
                    if i < len(batch):
                        time.sleep(1.0)
                        
                except Exception as e:
                    logger.error(f"Failed to process paper {paper['pmid']}: {e}")
                    failed_papers.append(paper['pmid'])
            
            # Longer delay between batches
            if batch_num < len(batches):
                time.sleep(2.0)
        
        # Compile results
        successful_papers = len(papers) - len(failed_papers)
        results = {
            'total_papers': len(papers),
            'successful_papers': successful_papers,
            'failed_papers': failed_papers,
            'total_agreed_correlations': total_agreed_correlations,
            'total_conflicts': total_conflicts,
            'papers_needing_review': sum(1 for r in all_results if r.needs_review),
            'consensus_results': all_results,
            'token_usage': {
                model: {
                    'input_tokens': usage['input'],
                    'output_tokens': usage['output'],
                    'total_tokens': usage['input'] + usage['output']
                }
                for model, usage in self.token_usage.items()
            }
        }
        
        # Log summary
        success_rate = (successful_papers / len(papers)) * 100
        logger.info(f"=== Consensus Processing Summary ===")
        logger.info(f"Papers processed: {successful_papers}/{len(papers)} ({success_rate:.1f}%)")
        logger.info(f"Agreed correlations: {total_agreed_correlations}")
        logger.info(f"Conflicts: {total_conflicts}")
        logger.info(f"Papers needing review: {results['papers_needing_review']}")
        
        for model, usage in results['token_usage'].items():
            logger.info(f"Token usage ({model}): {usage['total_tokens']} total")
        
        return results