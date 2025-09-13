"""
API clients with enhanced error handling for external services.
"""

import time
import requests
from typing import Dict, List, Optional
from openai import OpenAI
from src.data.config import config, setup_logging
from src.data.error_handler import handle_api_errors, ErrorContext, error_handler

logger = setup_logging(__name__)


@handle_api_errors("API request", max_retries=3)
def api_request(url: str, params: Dict = None, headers: Dict = None, 
               timeout: int = None) -> requests.Response:
    """
    Enhanced API request with comprehensive error handling.
    """
    timeout = timeout or config.api_timeout
    time.sleep(config.api_delay)  # Rate limiting
    
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


class PubMedAPI:
    """Simple PubMed API client."""
    
    def __init__(self):
        self.base_url = config.pubmed_base_url
        self.api_key = config.ncbi_api_key
        self.email = config.email
    
    @handle_api_errors("PubMed search", max_retries=3)
    def search_papers(self, query: str, min_year: int, max_results: int) -> Dict:
        """Search for papers."""
        params = {
            'db': 'pubmed',
            'term': f"{query} AND {min_year}:3000[pdat]",
            'retmax': max_results,
            'retmode': 'json',
            'email': self.email
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        url = f"{self.base_url}esearch.fcgi"
        response = api_request(url, params)
        data = response.json()
        
        return {
            'pmids': data.get('esearchresult', {}).get('idlist', []),
            'count': int(data.get('esearchresult', {}).get('count', 0))
        }
    
    @handle_api_errors("PubMed fetch", max_retries=3)
    def fetch_papers(self, pmids: List[str]) -> str:
        """Fetch paper metadata."""
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'email': self.email
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        url = f"{self.base_url}efetch.fcgi"
        response = api_request(url, params)
        return response.text


class PMCAPI:
    """Simple PMC API client."""
    
    def __init__(self):
        self.base_url = config.pmc_base_url
    
    def get_fulltext_url(self, pmc_id: str) -> Optional[str]:
        """Get fulltext URL for PMC article."""
        params = {
            'id': pmc_id,
            'format': 'json'
        }
        
        try:
            response = api_request(self.base_url, params)
            data = response.json()
            
            records = data.get('OA', {}).get('records', [])
            if records:
                link = records[0].get('link', [])
                if link:
                    return link[0].get('href')
                    
        except Exception as e:
            logger.error(f"Error getting PMC fulltext URL: {e}")
        
        return None


class UnpaywallAPI:
    """Simple Unpaywall API client."""
    
    def __init__(self):
        self.base_url = config.unpaywall_base_url
        self.email = config.email
    
    def get_paper_info(self, doi: str) -> Optional[Dict]:
        """Get paper info from Unpaywall."""
        if not doi:
            return None
            
        url = f"{self.base_url}/{doi}"
        params = {'email': self.email}
        
        try:
            response = api_request(url, params)
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting Unpaywall info: {e}")
            return None


class LLMClient:
    """Simple LLM client."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.llm_model
        self.base_url = config.llm_base_url
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="not-needed"
        )
    
    def generate(self, prompt: str, temperature: float = None, 
                max_tokens: int = None) -> Dict:
        """Generate text using LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature or config.llm_temperature,
                max_tokens=max_tokens or config.llm_max_tokens
            )
            
            return {
                'content': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


# Simple factory functions instead of singleton manager
def get_pubmed_client() -> PubMedAPI:
    """Get PubMed API client."""
    return PubMedAPI()


def get_pmc_client() -> PMCAPI:
    """Get PMC API client."""
    return PMCAPI()


def get_unpaywall_client() -> UnpaywallAPI:
    """Get Unpaywall API client."""
    return UnpaywallAPI()


def get_llm_client(model_name: str = None) -> LLMClient:
    """Get LLM client."""
    return LLMClient(model_name)


# Backward compatibility
class APIClientManager:
    """Backward compatibility wrapper."""
    
    def get_pubmed_client(self):
        return get_pubmed_client()
    
    def get_pmc_client(self):
        return get_pmc_client()
    
    def get_unpaywall_client(self):
        return get_unpaywall_client()
    
    def get_llm_client(self, llm_config):
        return get_llm_client(llm_config.model_name)


# Global instance for backward compatibility
client_manager = APIClientManager()