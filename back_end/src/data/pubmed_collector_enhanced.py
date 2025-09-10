"""
Enhanced PubMed collector with improved architecture and efficiency.
This replaces the original pubmed_collector.py with better error handling and modularity.
"""

import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add the current directory to sys.path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import config, setup_logging
from api_clients import client_manager
from database_manager_enhanced import database_manager
from paper_parser_enhanced import EnhancedPubmedParser
from fulltext_retriever_enhanced import EnhancedFullTextRetriever
from utils import log_execution_time, batch_process, safe_file_write

logger = setup_logging(__name__, 'pubmed_collector.log')


class EnhancedPubMedCollector:
    """
    Enhanced PubMed collector with centralized configuration and improved efficiency.
    Uses dependency injection for database and API clients.
    """
    
    def __init__(self, db_manager=None, parser=None, fulltext_retriever=None):
        """
        Initialize with dependency injection.
        
        Args:
            db_manager: Database manager instance (optional, uses global if None)
            parser: Paper parser instance (optional, creates new if None)
            fulltext_retriever: Fulltext retriever instance (optional, creates new if None)
        """
        # Use dependency injection or defaults
        self.db_manager = db_manager or database_manager
        self.parser = parser or EnhancedPubmedParser(self.db_manager)
        self.fulltext_retriever = fulltext_retriever or EnhancedFullTextRetriever(self.db_manager)
        
        # Get API client
        self.pubmed_client = client_manager.get_pubmed_client()
        
        # Configuration from central config
        self.metadata_dir = config.paths.metadata_dir
        
        logger.info("Enhanced PubMed collector initialized")
    
    @log_execution_time
    def search_papers(self, query: str, min_year: int = 2000, 
                     max_results: int = 100) -> List[str]:
        """
        Search for papers using the centralized PubMed client.
        
        Args:
            query: Search query string
            min_year: Minimum publication year
            max_results: Maximum number of results
            
        Returns:
            List of PMIDs
        """
        try:
            result = self.pubmed_client.search_papers(query, min_year, max_results)
            return result['pmids']
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    @log_execution_time
    def fetch_papers_metadata(self, pmid_list: List[str]) -> Optional[Path]:
        """
        Fetch paper metadata and save to file.
        
        Args:
            pmid_list: List of PMIDs to fetch
            
        Returns:
            Path to saved metadata file or None if failed
        """
        if not pmid_list:
            logger.warning("No PMIDs provided for fetching")
            return None
        
        try:
            # Fetch metadata using centralized client
            xml_content = self.pubmed_client.fetch_papers(pmid_list)
            
            if not xml_content:
                logger.error("No content received from PubMed API")
                return None
            
            # Save to file with timestamp
            timestamp = int(time.time())
            filename = f'pubmed_batch_{timestamp}.xml'
            metadata_file = self.metadata_dir / filename
            
            if safe_file_write(metadata_file, xml_content):
                logger.info(f"Saved metadata for {len(pmid_list)} papers to {metadata_file}")
                return metadata_file
            else:
                logger.error("Failed to save metadata file")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching papers metadata: {e}")
            return None
    
    @log_execution_time
    def collect_probiotics_by_condition(self, condition: str, min_year: int = 2000, 
                                      max_results: int = 100, 
                                      include_fulltext: bool = True) -> Dict[str, Any]:
        """
        Collect probiotic papers for a health condition with enhanced processing.
        
        Args:
            condition: Health condition to search for
            min_year: Minimum publication year
            max_results: Maximum number of papers
            include_fulltext: Whether to attempt fulltext retrieval
            
        Returns:
            Collection results dictionary
        """
        logger.info(f"Starting collection for condition: {condition}")
        
        # Build enhanced search query
        query = self._build_probiotic_query(condition)
        logger.info(f"Using query: {query}")
        
        try:
            # Step 1: Search for papers
            pmid_list = self.search_papers(query, min_year, max_results)
            
            if not pmid_list:
                return {
                    "condition": condition,
                    "paper_count": 0,
                    "status": "no_results",
                    "message": "No papers found matching criteria"
                }
            
            # Step 2: Fetch metadata
            metadata_file = self.fetch_papers_metadata(pmid_list)
            
            if not metadata_file:
                return {
                    "condition": condition,
                    "paper_count": len(pmid_list),
                    "status": "fetch_failed",
                    "message": "Failed to fetch paper metadata"
                }
            
            # Step 3: Parse and store papers
            papers = self.parser.parse_metadata_file(str(metadata_file))
            
            if not papers:
                return {
                    "condition": condition,
                    "paper_count": 0,
                    "status": "parse_failed",
                    "metadata_file": str(metadata_file),
                    "message": "Failed to parse paper metadata"
                }
            
            # Step 4: Retrieve fulltext if requested
            fulltext_stats = None
            if include_fulltext:
                logger.info(f"Attempting fulltext retrieval for {len(papers)} papers...")
                fulltext_stats = self._process_fulltext_batch(papers)
            
            # Step 5: Build result
            result = {
                "condition": condition,
                "paper_count": len(papers),
                "papers_stored": len(papers),
                "metadata_file": str(metadata_file),
                "status": "success",
                "message": f"Successfully collected {len(papers)} papers"
            }
            
            if fulltext_stats:
                result["fulltext_stats"] = fulltext_stats
            
            logger.info(f"Collection completed successfully: {result['message']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in collection process: {e}")
            return {
                "condition": condition,
                "paper_count": 0,
                "status": "error",
                "message": f"Collection failed: {str(e)}"
            }
    
    def _build_probiotic_query(self, condition: str) -> str:
        """Build an optimized search query for probiotics and health conditions."""
        probiotic_terms = [
            'probiotic*[Title/Abstract]',
            '"Probiotics"[MeSH Terms]',
            'lactobacillus[Title/Abstract]',
            'bifidobacterium[Title/Abstract]',
            '"lactic acid bacteria"[Title/Abstract]',
            'saccharomyces[Title/Abstract]',
            'synbiotic*[Title/Abstract]',
            '"Bacillus subtilis"[Title/Abstract]',
            '"Streptococcus thermophilus"[Title/Abstract]'
        ]
        
        condition_terms = [
            f'"{condition}"[Title/Abstract]',
            f'"{condition}"[MeSH Terms]'
        ]
        
        # Combine terms
        probiotic_query = f"({' OR '.join(probiotic_terms)})"
        condition_query = f"({' OR '.join(condition_terms)})"
        
        return f"{condition_query} AND {probiotic_query}"
    
    def _process_fulltext_batch(self, papers: List[Dict]) -> Dict[str, Any]:
        """Process papers for fulltext retrieval in batches."""
        # Filter papers that have PMC IDs or DOIs
        fulltext_candidates = [
            p for p in papers 
            if p.get('pmc_id') or p.get('doi')
        ]
        
        if not fulltext_candidates:
            logger.info("No papers with PMC IDs or DOIs found")
            return {
                'total_candidates': 0,
                'processed': 0,
                'successful_pmc': 0,
                'successful_unpaywall': 0,
                'failed': 0
            }
        
        logger.info(f"Found {len(fulltext_candidates)} papers with fulltext potential")
        
        # Process in smaller batches to manage resources
        batch_size = 20
        batches = batch_process(fulltext_candidates, batch_size)
        
        total_stats = {
            'total_candidates': len(fulltext_candidates),
            'processed': 0,
            'successful_pmc': 0,
            'successful_unpaywall': 0,
            'failed': 0,
            'errors': []
        }
        
        for i, batch in enumerate(batches, 1):
            logger.info(f"Processing fulltext batch {i}/{len(batches)}")
            
            batch_stats = self.fulltext_retriever.process_papers_batch(batch)
            
            # Accumulate statistics
            total_stats['processed'] += batch_stats['total_papers']
            total_stats['successful_pmc'] += batch_stats['successful_pmc']
            total_stats['successful_unpaywall'] += batch_stats['successful_unpaywall']
            total_stats['failed'] += batch_stats['failed']
            total_stats['errors'].extend(batch_stats.get('errors', []))
        
        logger.info(f"Fulltext processing completed: {total_stats}")
        return total_stats
    
    @log_execution_time
    def bulk_collect_conditions(self, conditions: List[str], 
                              max_results: int = 100,
                              include_fulltext: bool = True,
                              delay_between_conditions: float = 2.0) -> List[Dict[str, Any]]:
        """
        Collect papers for multiple conditions with improved batch processing.
        
        Args:
            conditions: List of health conditions
            max_results: Maximum papers per condition
            include_fulltext: Whether to retrieve fulltext
            delay_between_conditions: Delay between condition queries
            
        Returns:
            List of collection results
        """
        logger.info(f"Starting bulk collection for {len(conditions)} conditions")
        
        results = []
        total_papers = 0
        
        for i, condition in enumerate(conditions, 1):
            logger.info(f"Processing condition {i}/{len(conditions)}: {condition}")
            
            try:
                result = self.collect_probiotics_by_condition(
                    condition=condition,
                    max_results=max_results,
                    include_fulltext=include_fulltext
                )
                
                results.append(result)
                total_papers += result.get('paper_count', 0)
                
                # Rate limiting between conditions
                if i < len(conditions) and delay_between_conditions > 0:
                    time.sleep(delay_between_conditions)
                    
            except Exception as e:
                logger.error(f"Error processing condition {condition}: {e}")
                results.append({
                    "condition": condition,
                    "paper_count": 0,
                    "status": "error",
                    "message": str(e)
                })
        
        # Save bulk results
        timestamp = int(time.time())
        results_file = self.metadata_dir / f"bulk_collection_results_{timestamp}.json"
        
        summary = {
            "total_conditions": len(conditions),
            "total_papers_collected": total_papers,
            "successful_conditions": len([r for r in results if r.get('status') == 'success']),
            "failed_conditions": len([r for r in results if r.get('status') == 'error']),
            "conditions_processed": conditions,
            "detailed_results": results
        }
        
        if safe_file_write(results_file, str(summary)):
            logger.info(f"Bulk collection results saved to {results_file}")
        
        logger.info(f"Bulk collection completed: {total_papers} papers across {len(conditions)} conditions")
        return results