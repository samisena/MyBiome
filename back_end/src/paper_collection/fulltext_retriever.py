"""
Fulltext retriever with improved architecture and centralized API management.
"""

from typing import Dict, Optional, List, Any
from pathlib import Path
import sys

from src.data.config import config, setup_logging
from src.data.api_clients import get_pmc_client, get_unpaywall_client
from src.paper_collection.database_manager import database_manager
from src.data.utils import log_execution_time, batch_process, safe_file_write

logger = setup_logging(__name__, 'fulltext_retriever.log')


class FullTextRetriever:
    """
    Enhanced fulltext retriever using centralized API clients and improved error handling.
    """
    
    def __init__(self, db_manager=None):
        """
        Initialize with dependency injection.
        
        Args:
            db_manager: Database manager instance (optional, uses global if None)
        """
        self.db_manager = db_manager or database_manager
        
        # Get API clients from centralized functions
        self.pmc_client = get_pmc_client()
        self.unpaywall_client = get_unpaywall_client()
        
        # Directory configuration
        self.fulltext_dir = config.paths.fulltext_dir
        self.pmc_dir = config.paths.pmc_dir
        self.pdf_dir = config.paths.pdf_dir
        
        logger.info("Enhanced fulltext retriever initialized")
    
    @log_execution_time
    def check_and_retrieve_fulltext(self, paper: Dict) -> Dict[str, Any]:
        """
        Main method to check for and retrieve full text for a paper.
        
        Args:
            paper: Dictionary containing paper information
            
        Returns:
            Dictionary with retrieval results
        """
        pmid = paper.get('pmid')
        pmc_id = paper.get('pmc_id')
        doi = paper.get('doi')
        title = paper.get('title', 'Unknown Title')
        
        result = {
            'pmid': pmid,
            'success': False,
            'source': None,
            'path': None,
            'error': None
        }
        
        logger.info(f"Processing fulltext for {pmid}: {title[:80]}...")
        
        # Try PMC first if PMC ID is available
        if pmc_id:
            logger.info(f"Attempting PMC retrieval for {pmid} (PMC ID: {pmc_id})")
            
            try:
                pmc_result = self._retrieve_from_pmc(pmc_id, pmid)
                if pmc_result['success']:
                    result.update(pmc_result)
                    self._update_database_fulltext(pmid, True, 'pmc', pmc_result['path'])
                    return result
            except Exception as e:
                logger.error(f"PMC retrieval error for {pmid}: {e}")
        
        # Try Unpaywall if DOI is available and PMC failed
        if doi and not result['success']:
            logger.info(f"Attempting Unpaywall check for {pmid} (DOI: {doi})")
            
            try:
                unpaywall_result = self._retrieve_from_unpaywall(doi, pmid)
                if unpaywall_result['success']:
                    result.update(unpaywall_result)
                    self._update_database_fulltext(pmid, True, 'unpaywall', unpaywall_result['path'])
                    return result
            except Exception as e:
                logger.error(f"Unpaywall retrieval error for {pmid}: {e}")
        
        # No full text found
        if not pmc_id and not doi:
            result['error'] = "No PMC ID or DOI available"
        elif not result['success']:
            result['error'] = "No open access full text found"
        
        # Update database to indicate we tried but failed
        self._update_database_fulltext(pmid, False, None, None)
        
        logger.info(f"No full text retrieved for {pmid}: {result['error']}")
        return result
    
    def _retrieve_from_pmc(self, pmc_id: str, pmid: str) -> Dict[str, Any]:
        """Retrieve fulltext from PMC."""
        result = {'success': False, 'source': 'pmc', 'path': None, 'error': None}
        
        try:
            # Check if fulltext is available
            fulltext_info = self.pmc_client.get_fulltext_info(pmc_id)
            
            if not fulltext_info or not fulltext_info.get('available'):
                result['error'] = "PMC fulltext not available"
                return result
            
            # Download the fulltext
            xml_content = self.pmc_client.download_fulltext(fulltext_info['xml_url'])
            
            if not xml_content:
                result['error'] = "Failed to download PMC fulltext"
                return result
            
            # Save to file
            clean_pmc_id = pmc_id.replace('PMC', '') if pmc_id.startswith('PMC') else pmc_id
            filename = f"PMC{clean_pmc_id}_fulltext.xml"
            file_path = self.pmc_dir / filename
            
            if safe_file_write(file_path, xml_content):
                result['success'] = True
                result['path'] = str(file_path)
                logger.info(f"Successfully retrieved PMC fulltext for {pmid}")
            else:
                result['error'] = "Failed to save PMC fulltext"
            
            return result
            
        except Exception as e:
            result['error'] = f"PMC retrieval error: {str(e)}"
            return result
    
    def _retrieve_from_unpaywall(self, doi: str, pmid: str) -> Dict[str, Any]:
        """Retrieve fulltext from Unpaywall."""
        result = {'success': False, 'source': 'unpaywall', 'path': None, 'error': None}
        
        try:
            # Check if paper is open access
            oa_info = self.unpaywall_client.check_open_access(doi)
            
            if not oa_info or not oa_info.get('is_oa'):
                result['error'] = "Paper not open access according to Unpaywall"
                return result
            
            # Download the PDF
            pdf_content = self.unpaywall_client.download_pdf(oa_info['pdf_url'])
            
            if not pdf_content:
                result['error'] = "Failed to download PDF from Unpaywall"
                return result
            
            # Save to file
            safe_pmid = pmid.replace('/', '_')
            filename = f"{safe_pmid}.pdf"
            file_path = self.pdf_dir / filename
            
            try:
                with open(file_path, 'wb') as f:
                    f.write(pdf_content)
                
                result['success'] = True
                result['path'] = str(file_path)
                logger.info(f"Successfully retrieved Unpaywall PDF for {pmid}")
            except Exception as e:
                result['error'] = f"Failed to save PDF: {str(e)}"
            
            return result
            
        except Exception as e:
            result['error'] = f"Unpaywall retrieval error: {str(e)}"
            return result
    
    def _update_database_fulltext(self, pmid: str, has_fulltext: bool, 
                                source: Optional[str], path: Optional[str]) -> bool:
        """Update database with fulltext information."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE papers 
                    SET has_fulltext = ?, fulltext_source = ?, fulltext_path = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE pmid = ?
                ''', (has_fulltext, source, path, pmid))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating fulltext info for {pmid}: {e}")
            return False
    
    @log_execution_time
    def process_papers_batch(self, papers: List[Dict], 
                           batch_size: int = 25) -> Dict[str, Any]:
        """
        Process a batch of papers for full text retrieval with improved batching.
        
        Args:
            papers: List of paper dictionaries
            batch_size: Size of processing batches
            
        Returns:
            Summary statistics
        """
        logger.info(f"Processing {len(papers)} papers for fulltext retrieval")
        
        # Filter papers that might have fulltext
        fulltext_candidates = [
            p for p in papers 
            if p.get('pmc_id') or p.get('doi')
        ]
        
        if not fulltext_candidates:
            logger.info("No papers with PMC IDs or DOIs found")
            return {
                'total_papers': len(papers),
                'candidates': 0,
                'successful_pmc': 0,
                'successful_unpaywall': 0,
                'failed': len(papers),
                'errors': ['No PMC IDs or DOIs available']
            }
        
        logger.info(f"Found {len(fulltext_candidates)} potential fulltext candidates")
        
        # Process in batches
        batches = batch_process(fulltext_candidates, batch_size)
        
        total_stats = {
            'total_papers': len(papers),
            'candidates': len(fulltext_candidates),
            'successful_pmc': 0,
            'successful_unpaywall': 0,
            'failed': 0,
            'errors': []
        }
        
        for i, batch in enumerate(batches, 1):
            logger.info(f"Processing batch {i}/{len(batches)} ({len(batch)} papers)")
            
            for paper in batch:
                try:
                    result = self.check_and_retrieve_fulltext(paper)
                    
                    if result['success']:
                        if result['source'] == 'pmc':
                            total_stats['successful_pmc'] += 1
                        elif result['source'] == 'unpaywall':
                            total_stats['successful_unpaywall'] += 1
                    else:
                        total_stats['failed'] += 1
                        if result['error']:
                            total_stats['errors'].append(f"{paper['pmid']}: {result['error']}")
                
                except Exception as e:
                    total_stats['failed'] += 1
                    error_msg = f"{paper['pmid']}: Unexpected error - {str(e)}"
                    total_stats['errors'].append(error_msg)
                    logger.error(error_msg)
        
        # Add papers without fulltext potential to failed count
        total_stats['failed'] += (len(papers) - len(fulltext_candidates))
        
        # Log summary
        success_rate = ((total_stats['successful_pmc'] + total_stats['successful_unpaywall']) / 
                       total_stats['total_papers'] * 100)
        
        logger.info("=== Fulltext Retrieval Summary ===")
        logger.info(f"Total papers: {total_stats['total_papers']}")
        logger.info(f"Fulltext candidates: {total_stats['candidates']}")
        logger.info(f"Successful PMC retrievals: {total_stats['successful_pmc']}")
        logger.info(f"Successful Unpaywall retrievals: {total_stats['successful_unpaywall']}")
        logger.info(f"Failed retrievals: {total_stats['failed']}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        return total_stats
    
    def get_fulltext_statistics(self) -> Dict[str, Any]:
        """Get statistics about fulltext availability in the database."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Total papers
                cursor.execute('SELECT COUNT(*) FROM papers')
                stats['total_papers'] = cursor.fetchone()[0]
                
                # Papers with fulltext
                cursor.execute('SELECT COUNT(*) FROM papers WHERE has_fulltext = TRUE')
                stats['papers_with_fulltext'] = cursor.fetchone()[0]
                
                # Breakdown by source
                cursor.execute('''
                    SELECT fulltext_source, COUNT(*) 
                    FROM papers 
                    WHERE has_fulltext = TRUE 
                    GROUP BY fulltext_source
                ''')
                stats['by_source'] = dict(cursor.fetchall())
                
                # Papers with PMC IDs
                cursor.execute('SELECT COUNT(*) FROM papers WHERE pmc_id IS NOT NULL')
                stats['papers_with_pmc_id'] = cursor.fetchone()[0]
                
                # Papers with DOIs
                cursor.execute('SELECT COUNT(*) FROM papers WHERE doi IS NOT NULL')
                stats['papers_with_doi'] = cursor.fetchone()[0]
                
                # Coverage percentage
                if stats['total_papers'] > 0:
                    stats['coverage_percentage'] = (stats['papers_with_fulltext'] / stats['total_papers']) * 100
                else:
                    stats['coverage_percentage'] = 0
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting fulltext statistics: {e}")
            return {}