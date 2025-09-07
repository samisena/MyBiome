"""
Centralized API client management for external services.
This module provides singleton clients with proper configuration and rate limiting.
"""

import time
import requests
from typing import Dict, Optional, Any
from openai import OpenAI

from .config import config, LLMConfig, setup_logging
from .utils import rate_limit, retry_with_backoff

logger = setup_logging(__name__)


class APIClientManager:
    """Singleton manager for API clients."""
    
    _instance = None
    _clients = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_pubmed_client(self) -> 'PubMedAPIClient':
        """Get or create PubMed API client."""
        if 'pubmed' not in self._clients:
            self._clients['pubmed'] = PubMedAPIClient()
        return self._clients['pubmed']
    
    def get_pmc_client(self) -> 'PMCAPIClient':
        """Get or create PMC API client."""
        if 'pmc' not in self._clients:
            self._clients['pmc'] = PMCAPIClient()
        return self._clients['pmc']
    
    def get_unpaywall_client(self) -> 'UnpaywallAPIClient':
        """Get or create Unpaywall API client."""
        if 'unpaywall' not in self._clients:
            self._clients['unpaywall'] = UnpaywallAPIClient()
        return self._clients['unpaywall']
    
    def get_llm_client(self, llm_config: Optional[LLMConfig] = None) -> OpenAI:
        """Get or create LLM client."""
        if llm_config is None:
            llm_config = config.llm
            
        client_key = f"llm_{llm_config.base_url}_{llm_config.model_name}"
        
        if client_key not in self._clients:
            self._clients[client_key] = OpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.base_url
            )
            logger.info(f"Created LLM client for {llm_config.model_name} at {llm_config.base_url}")
        
        return self._clients[client_key]


class PubMedAPIClient:
    """Wrapper for PubMed API with rate limiting and error handling."""
    
    def __init__(self):
        self.base_url = config.api.pubmed_base_url
        self.api_key = config.api.ncbi_api_key
        self.session = requests.Session()
        
        # Set up session headers
        if self.api_key:
            self.session.params = {'api_key': self.api_key}
    
    @rate_limit(0.5)  # 2 requests per second max
    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def search_papers(self, query: str, min_year: int = 2000, 
                     max_results: int = 100) -> Dict[str, Any]:
        """Search for papers using PubMed API."""
        search_url = f"{self.base_url}esearch.fcgi"
        
        formatted_query = f"{query} AND {min_year}[PDAT]:3000[PDAT]"
        
        params = {
            'db': 'pubmed',
            'term': formatted_query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        response = self.session.get(search_url, params=params, 
                                  timeout=config.api.api_timeout)
        response.raise_for_status()
        
        search_results = response.json()
        pmid_list = search_results.get("esearchresult", {}).get("idlist", [])
        
        logger.info(f'Found {len(pmid_list)} papers for query: {query}')
        return {
            'pmids': pmid_list,
            'count': len(pmid_list),
            'query': formatted_query
        }
    
    @rate_limit(0.5)
    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def fetch_papers(self, pmid_list: list) -> str:
        """Fetch paper metadata using PubMed API."""
        if not pmid_list:
            return ""
        
        pmids = ",".join(str(pmid) for pmid in pmid_list)
        fetch_url = f"{self.base_url}efetch.fcgi"
        
        params = {
            'db': 'pubmed',
            'id': pmids,
            'retmode': 'xml'
        }
        
        response = self.session.get(fetch_url, params=params, 
                                  timeout=config.api.api_timeout)
        response.raise_for_status()
        
        logger.info(f"Fetched metadata for {len(pmid_list)} papers")
        return response.text


class PMCAPIClient:
    """Wrapper for PMC API with rate limiting and error handling."""
    
    def __init__(self):
        self.base_url = config.api.pmc_base_url
        self.session = requests.Session()
    
    @rate_limit(0.5)
    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def get_fulltext_info(self, pmc_id: str) -> Optional[Dict[str, Any]]:
        """Get fulltext availability info from PMC."""
        clean_pmc_id = pmc_id.replace('PMC', '') if pmc_id.startswith('PMC') else pmc_id
        
        params = {'id': f'PMC{clean_pmc_id}'}
        
        response = self.session.get(self.base_url, params=params, 
                                  timeout=config.api.api_timeout)
        response.raise_for_status()
        
        # Parse XML and extract download links
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(response.content)
            
            # Check for errors
            error = root.find('.//error')
            if error is not None:
                logger.warning(f"PMC API error for {pmc_id}: {error.text}")
                return None
            
            # Find XML download link
            records = root.findall('.//record')
            for record in records:
                links = record.findall('.//link')
                for link in links:
                    format_attr = link.get('format', '')
                    href = link.get('href', '')
                    if 'xml' in format_attr.lower() and href:
                        return {
                            'pmc_id': pmc_id,
                            'xml_url': href,
                            'available': True
                        }
            
            return None
            
        except ET.ParseError as e:
            logger.error(f"Error parsing PMC response for {pmc_id}: {e}")
            return None
    
    @rate_limit(0.5)
    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def download_fulltext(self, xml_url: str) -> Optional[str]:
        """Download fulltext XML from PMC."""
        response = self.session.get(xml_url, timeout=60)
        response.raise_for_status()
        
        logger.info(f"Downloaded fulltext from {xml_url}")
        return response.text


class UnpaywallAPIClient:
    """Wrapper for Unpaywall API with rate limiting and error handling."""
    
    def __init__(self):
        self.base_url = config.api.unpaywall_base_url
        self.email = config.api.email
        self.session = requests.Session()
    
    @rate_limit(1.0)  # 1 request per second max
    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def check_open_access(self, doi: str) -> Optional[Dict[str, Any]]:
        """Check if paper is open access via Unpaywall."""
        # Clean DOI
        clean_doi = doi.strip()
        if clean_doi.startswith('http'):
            clean_doi = '/'.join(clean_doi.split('/')[-2:])
        
        url = f"{self.base_url}/{clean_doi}"
        params = {'email': self.email}
        
        response = self.session.get(url, params=params, 
                                  timeout=config.api.api_timeout)
        
        if response.status_code == 404:
            logger.info(f"DOI {doi} not found in Unpaywall")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        if not data.get('is_oa', False):
            logger.info(f"Paper {doi} is not open access")
            return None
        
        # Find best PDF URL
        pdf_url = None
        best_oa_location = data.get('best_oa_location')
        if best_oa_location and best_oa_location.get('url_for_pdf'):
            pdf_url = best_oa_location['url_for_pdf']
        else:
            # Check other locations
            for location in data.get('oa_locations', []):
                if location.get('url_for_pdf'):
                    pdf_url = location['url_for_pdf']
                    break
        
        if pdf_url:
            logger.info(f"Found open access PDF for {doi}: {pdf_url}")
            return {
                'doi': doi,
                'is_oa': True,
                'pdf_url': pdf_url,
                'host_type': best_oa_location.get('host_type') if best_oa_location else None
            }
        
        return None
    
    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def download_pdf(self, pdf_url: str) -> Optional[bytes]:
        """Download PDF from URL."""
        time.sleep(1)  # Be polite
        
        response = self.session.get(pdf_url, timeout=60, stream=True)
        response.raise_for_status()
        
        # Check if response is actually a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
            # Check the actual content
            content = response.content
            if not content.startswith(b'%PDF'):
                logger.warning(f"Downloaded content is not a PDF from {pdf_url}")
                return None
        
        logger.info(f"Downloaded PDF from {pdf_url}")
        return response.content


# Global client manager instance
client_manager = APIClientManager()