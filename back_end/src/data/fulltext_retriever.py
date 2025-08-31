"""
Full-text retrieval module for PubMed papers.
This module handles retrieving full-text papers from PMC and Unpaywall APIs.
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, List
import logging
from urllib.parse import urlparse

from src.data.database_manager import DatabaseManager

# Project root path
project_root = Path(__file__).parent.parent.parent

class FullTextRetriever:
    """
    Retrieves full-text papers from various sources:
    1. PMC (PubMed Central) OA Service API for open access articles
    2. Unpaywall API for open access PDF links
    """
    
    def __init__(self, email: str = "your_email@example.com"):
        """
        Initialize the FullTextRetriever.
        
        Args:
            email: Email address for API requests (required by Unpaywall)
        """
        self.email = email
        self.pmc_base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        self.unpaywall_base_url = "https://api.unpaywall.org/v2"
        
        # Set up directories
        self.fulltext_dir = project_root / "data" / "raw" / "fulltext"
        self.pmc_dir = self.fulltext_dir / "pmc"
        self.pdf_dir = self.fulltext_dir / "pdf"
        
        # Set up database manager
        self.db_manager = DatabaseManager()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting settings
        self.pmc_delay = 0.5  # 500ms between PMC requests
        self.unpaywall_delay = 1.0  # 1s between Unpaywall requests
        self.last_pmc_request = 0
        self.last_unpaywall_request = 0
        
    def _rate_limit_pmc(self):
        """Ensure proper rate limiting for PMC API."""
        elapsed = time.time() - self.last_pmc_request
        if elapsed < self.pmc_delay:
            time.sleep(self.pmc_delay - elapsed)
        self.last_pmc_request = time.time()
    
    def _rate_limit_unpaywall(self):
        """Ensure proper rate limiting for Unpaywall API."""
        elapsed = time.time() - self.last_unpaywall_request
        if elapsed < self.unpaywall_delay:
            time.sleep(self.unpaywall_delay - elapsed)
        self.last_unpaywall_request = time.time()
    
    def fetch_pmc_fulltext(self, pmc_id: str) -> Optional[str]:
        """
        Fetch full text from PMC OA Service API.
        
        Args:
            pmc_id: PMC identifier (e.g., 'PMC6987087' or just '6987087')
            
        Returns:
            Path to saved XML file if successful, None if failed
        """
        # Clean PMC ID - remove PMC prefix if present
        clean_pmc_id = pmc_id.replace('PMC', '') if pmc_id.startswith('PMC') else pmc_id
        
        self._rate_limit_pmc()
        
        #* Requests full texts from the PMC API
        try:
            # Make API request
            params = {
                'id': f'PMC{clean_pmc_id}'
            }
            
            response = requests.get(self.pmc_base_url, params=params, timeout=30)
            
            if response.status_code != 200:
                self.logger.warning(f"PMC API returned status {response.status_code} for {pmc_id}")
                return None
            
            # Parse XML response to check if full text is available
            try:
                root = ET.fromstring(response.content)
                
                # Check for error messages
                error = root.find('.//error')
                if error is not None:
                    self.logger.warning(f"PMC API error for {pmc_id}: {error.text}")
                    return None
                
                # Look for download links
                records = root.findall('.//record')
                if not records:
                    self.logger.info(f"No full text available for {pmc_id}")
                    return None
                
                # Find XML download link
                xml_link = None
                for record in records:
                    links = record.findall('.//link')
                    for link in links:
                        format_attr = link.get('format', '')
                        href = link.get('href', '')
                        if 'xml' in format_attr.lower() and href:
                            xml_link = href
                            break
                    if xml_link:
                        break
                
                if not xml_link:
                    self.logger.info(f"No XML download link found for {pmc_id}")
                    return None
                
                # Download the full text XML
                self._rate_limit_pmc()  # Rate limit the actual download too
                
                xml_response = requests.get(xml_link, timeout=60)
                if xml_response.status_code != 200:
                    self.logger.warning(f"Failed to download XML for {pmc_id}")
                    return None
                
                # Save to file
                filename = f"PMC{clean_pmc_id}_fulltext.xml"
                file_path = self.pmc_dir / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(xml_response.text)
                
                self.logger.info(f"Successfully downloaded PMC full text: {file_path}")
                return str(file_path)
                
            except ET.ParseError as e:
                self.logger.error(f"Error parsing PMC response for {pmc_id}: {e}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Request error for PMC {pmc_id}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching PMC {pmc_id}: {e}")
            return None
    
    
    def check_unpaywall(self, doi: str) -> Optional[str]:
        """
        Check Unpaywall API for open access PDF.
        
        Args:
            doi: DOI of the paper
            
        Returns:
            URL of open access PDF if available, None otherwise
        """
        self._rate_limit_unpaywall()
        
        try:
            # Clean DOI
            clean_doi = doi.strip()
            if clean_doi.startswith('http'):
                # Extract DOI from URL
                clean_doi = clean_doi.split('/')[-2] + '/' + clean_doi.split('/')[-1]
            
            url = f"{self.unpaywall_base_url}/{clean_doi}"
            params = {
                'email': self.email
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 404:
                self.logger.info(f"DOI {doi} not found in Unpaywall")
                return None
            elif response.status_code != 200:
                self.logger.warning(f"Unpaywall API returned status {response.status_code} for {doi}")
                return None
            
            data = response.json()
            
            # Check if paper is open access
            if not data.get('is_oa', False):
                self.logger.info(f"Paper {doi} is not open access according to Unpaywall")
                return None
            
            # Look for best OA location
            best_oa_location = data.get('best_oa_location')
            if best_oa_location and best_oa_location.get('url_for_pdf'):
                pdf_url = best_oa_location['url_for_pdf']
                self.logger.info(f"Found open access PDF for {doi}: {pdf_url}")
                return pdf_url
            
            # Check other OA locations
            oa_locations = data.get('oa_locations', [])
            for location in oa_locations:
                if location.get('url_for_pdf'):
                    pdf_url = location['url_for_pdf']
                    self.logger.info(f"Found open access PDF for {doi}: {pdf_url}")
                    return pdf_url
            
            self.logger.info(f"No PDF URL found for open access paper {doi}")
            return None
            
        except requests.RequestException as e:
            self.logger.error(f"Request error for Unpaywall {doi}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error checking Unpaywall {doi}: {e}")
            return None
    
    def download_pdf(self, pdf_url: str, filename: str) -> Optional[str]:
        """
        Download PDF from URL.
        
        Args:
            pdf_url: URL of the PDF
            filename: Filename to save as
            
        Returns:
            Path to saved PDF if successful, None if failed
        """
        try:
            # Add delay for politeness
            time.sleep(1)
            
            response = requests.get(pdf_url, timeout=60, stream=True)
            
            if response.status_code != 200:
                self.logger.warning(f"Failed to download PDF from {pdf_url}")
                return None
            
            # Check if response is actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                # Sometimes PDFs are served with generic content types
                # Check the actual content
                first_bytes = response.content[:10] if hasattr(response, 'content') else b''
                if not first_bytes.startswith(b'%PDF'):
                    self.logger.warning(f"Downloaded content is not a PDF from {pdf_url}")
                    return None
            
            file_path = self.pdf_dir / filename
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Successfully downloaded PDF: {file_path}")
            return str(file_path)
            
        except requests.RequestException as e:
            self.logger.error(f"Request error downloading PDF {pdf_url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error downloading PDF {pdf_url}: {e}")
            return None
    
    def check_and_retrieve_fulltext(self, paper: Dict) -> Dict:
        """
        Main method to check for and retrieve full text for a paper.
        Tries PMC first, then Unpaywall.
        
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
        
        self.logger.info(f"Checking full text for paper {pmid}: {title[:80]}...")
        
        # Try PMC first if PMC ID is available
        if pmc_id:
            self.logger.info(f"Attempting PMC retrieval for {pmid} (PMC ID: {pmc_id})")
            pmc_path = self.fetch_pmc_fulltext(pmc_id)
            if pmc_path:
                result['success'] = True
                result['source'] = 'pmc'
                result['path'] = pmc_path
                
                # Update database
                self.db_manager.update_paper_fulltext(pmid, True, 'pmc', pmc_path)
                
                self.logger.info(f"Successfully retrieved PMC full text for {pmid}")
                return result
        
        # Try Unpaywall if DOI is available and PMC failed
        if doi and not result['success']:
            self.logger.info(f"Attempting Unpaywall check for {pmid} (DOI: {doi})")
            pdf_url = self.check_unpaywall(doi)
            if pdf_url:
                # Generate filename
                safe_pmid = pmid.replace('/', '_')
                filename = f"{safe_pmid}.pdf"
                
                pdf_path = self.download_pdf(pdf_url, filename)
                if pdf_path:
                    result['success'] = True
                    result['source'] = 'unpaywall'
                    result['path'] = pdf_path
                    
                    # Update database
                    self.db_manager.update_paper_fulltext(pmid, True, 'unpaywall', pdf_path)
                    
                    self.logger.info(f"Successfully retrieved Unpaywall PDF for {pmid}")
                    return result
        
        # No full text found
        if not pmc_id and not doi:
            result['error'] = "No PMC ID or DOI available"
        elif not result['success']:
            result['error'] = "No open access full text found"
        
        # Update database to indicate we tried but failed
        self.db_manager.update_paper_fulltext(pmid, False, None, None)
        
        self.logger.info(f"No full text retrieved for {pmid}: {result['error']}")
        return result
    
    def process_papers_batch(self, papers: List[Dict], max_papers: Optional[int] = None) -> Dict:
        """
        Process a batch of papers for full text retrieval.
        
        Args:
            papers: List of paper dictionaries
            max_papers: Maximum number of papers to process
            
        Returns:
            Summary statistics
        """
        papers_to_process = papers[:max_papers] if max_papers else papers
        
        stats = {
            'total_papers': len(papers_to_process),
            'successful_pmc': 0,
            'successful_unpaywall': 0,
            'failed': 0,
            'errors': []
        }
        
        self.logger.info(f"Processing {stats['total_papers']} papers for full text retrieval")
        
        for i, paper in enumerate(papers_to_process, 1):
            self.logger.info(f"Processing paper {i}/{stats['total_papers']}")
            
            try:
                result = self.check_and_retrieve_fulltext(paper)
                
                if result['success']:
                    if result['source'] == 'pmc':
                        stats['successful_pmc'] += 1
                    elif result['source'] == 'unpaywall':
                        stats['successful_unpaywall'] += 1
                else:
                    stats['failed'] += 1
                    if result['error']:
                        stats['errors'].append(f"{paper['pmid']}: {result['error']}")
                        
            except Exception as e:
                stats['failed'] += 1
                error_msg = f"{paper['pmid']}: Unexpected error - {str(e)}"
                stats['errors'].append(error_msg)
                self.logger.error(error_msg)
        
        # Log summary
        self.logger.info("=== Full Text Retrieval Summary ===")
        self.logger.info(f"Total papers processed: {stats['total_papers']}")
        self.logger.info(f"Successful PMC retrievals: {stats['successful_pmc']}")
        self.logger.info(f"Successful Unpaywall retrievals: {stats['successful_unpaywall']}")
        self.logger.info(f"Failed retrievals: {stats['failed']}")
        
        return stats
    
    def create_directories(self):
        """Create necessary directories for storing full text files."""
        self.fulltext_dir.mkdir(parents=True, exist_ok=True)
        self.pmc_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created full text directories at {self.fulltext_dir}")