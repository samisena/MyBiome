import time
import requests  #API request module
from typing import Dict, List, Optional
from openai import OpenAI
from back_end.src.data.config import config, setup_logging
from back_end.src.data.error_handler import handle_api_errors, handle_llm_errors

logger = setup_logging(__name__)  #logs will show 'api_clients' as the module's name

@handle_api_errors("API request", max_retries=3) #*This decorator adds retry functionality 
                                #* so that our function can handle api errors without crashing 
def api_request(url:str, params: Dict = None, headers: Dict = None, 
                timeout: int = None) -> requests.Response:
    """
    Enhanced API request with comprehensive error handling.
    """
    timeout = timeout or config.api_timeout   #timout either explicitely mentioned or taken
                                            #from the config file
    time.sleep(config.api_delay)  #rate limiting
    response = requests.get(url, params=params, headers=headers, timeout=timeout) #makes
                                                        #the api get request
    response.raise_for_status() 
    return response


class PubMedAPI:
    """ PubMed API client. """
    def __init__(self) -> None:
        self.base_url = config.pubmed_base_url
        self.api_key = config.ncbi_api_key
        self.email = config.email  #pubmed requires an email

    @handle_api_errors("PubMed search", max_retries=3)
    def search_papers(self, query: str, min_year: int, max_results:int) -> Dict:
        """Search for papers given a health condition (query).
        Args:
            query (str): The health condition
            min_year(int): don't fetch papers that came out earlier than this year
            max_results: maximum number of papers to receive the metadapubmed ids of 
        Returns:
            {'pmids': the pubmed ids of the papers matching the query,
            'count': the number of papers found
            }
        """
        params = {
            'db': 'pubmed',  #search pubmed database
            'term': f"{query} AND {min_year}:3000[pdat]",
            'retmax': max_results,
            'retmode': 'json',   #return JSON
            'email': self.email            
        }

        if self.api_key:
            params['api_key'] = self.api_key

        url = f"{self.base_url}esearch.fcgi"  #pubmed API formating
        response = api_request(url, params)  #Using the function we defined above
        data = response.json()  #converts JSON to pyhton Dictionary

        return {
            'pmids': data.get('esearchresult', {}).get('idlist', []),
            'count': int(data.get('esearchresult', {}).get('count', 0))            
        } 
    
    @handle_api_errors("PubMed fetch", max_retries=3)
    def fetch_papers(self, pmids: List[str]) -> str:
        """Fetch paper metadata.
        Args:
            pmids: list of paper ids
        Returns:
            XML file as Python string format (requires parsing to extract data)

        """
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),  #join the list into a comma seperated string
            'retmode': 'xml',  #returns XML
            'email': self.email
        }       

        if self.api_key:
            params['api_key'] = self.api_key

        url = f"{self.base_url}efetch.fcgi"
        response = api_request(url, params)
        return response.text  #? returns raw XML as a python string    
    

class PMCAPI:
    """ PMC API client. """
    def __init__(self) -> None:
        self.base_url = config.pmc_base_url

    @handle_api_errors("PMC fulltext retrieval", max_retries=3)
    def get_fulltext_url(self, pmc_id: str) -> Optional[str]:
        """Get full text for a given PMC article.
        Args:
            pmc_id(str): the PMC id of the paper
        Returns:
            A link to download the full text or None Object.
        """
        params ={
            'id': pmc_id,
            'format': 'json'
        }

        response = api_request(self.base_url, params)
        data = response.json()  #JSON to Python ditionary
        records = data.get('OA', {}).get('records', [])  #gets the keywords
        if records:
            link = records[0].get('link', [])
            if link:
                return link[0].get('href')

        return None
    

def get_pubmed_client() -> PubMedAPI:
    """Get the PubMed API client."""
    return PubMedAPI()

def get_pmc_client() -> PMCAPI:
    """Get the PMC API client."""
    return PMCAPI()

class UnpaywallAPI:
    """ Unpaywall API Client."""
    def __init__(self) -> None:
        self.base_url = config.unpaywall_base_url
        self.email = config.email

    @handle_api_errors("Unpaywall paper retrieval", max_retries=3)
    def get_paper_info(self, doi:str) -> Optional[Dict]:
        """ Get paper info from Unpaywall.
        Args:
            doi(str): digital online ID
        Returns:
            A dictionary of information that includes links to free download.
        """
        if not doi:
            return None

        url = f"{self.base_url}/{doi}"
        params = {'email':self.email}

        response = api_request(url, params)
        return response.json()
        

def get_unpaywall_client() -> UnpaywallAPI:
    """Get Unpaywall API client."""
    return UnpaywallAPI()

class SemanticScholarAPI:
    """ Semantic Scholar API client for paper enrichnment and discovery.
    """

    def __init__(self) -> None:
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {
            'Accept': 'application/json',
            'User-Agent': 'MyBiome/1.0 (research purposes)'
        }
    
    @handle_api_errors("Semantic Scholar batch", max_retries=2)
    def get_papers_batch(self, paper_ids: List[str],
                         fields: List[str]=None) -> List[Dict]:
        """ Get multiple papers by DOI/PMID in batch (up to 500).
        Args:
            paper_ids: list of paper ids
            fields: which data fields we'd like to retrieve
        Returns:
            List of dictionaries where each dictionary represents a paper.
            [
                {
                    'paperId': 'a1b2c3d4e5',
                    'title': 'Machine Learning in Biology',
                    'abstract': 'This paper explores...',
                    'year': 2023,
                    'authors': [
                        {'authorId': '12345', 'name': 'John Doe'},
                        {'authorId': '67890', 'name': 'Jane Smith'}
                    ],
                    'citationCount': 42,
                    'influentialCitationCount': 5,
                    # ... other fields
                },
                # More papers...
            ]
        
        """
        if not paper_ids:
            return []

        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'year', 'authors',
                'journal', 'citationCount', 'influentialCitationCount',
                'fieldsOfStudy', 'tldr', 'externalIds'
            ] 

        url = f"{self.base_url}/paper/batch" #? Semantic Scholar batch API endpoint

        #* Ensures adequate formating of the paper ids
        formatted_ids = []
        for paper_id in paper_ids[:500]:
            if paper_id.startswith('PMID:'):
                formatted_ids.append(paper_id)
            elif paper_id.isdigit():
                formatted_ids.append(f'PMID:{paper_id}')
            else:
                formatted_ids.append(paper_id)  #Assumes DOI format
        
        params = {
            'fields': ','.join(fields)
        }

        data = {'ids': formatted_ids}

        response = requests.post(url, json=data, params=params,
                               headers=self.headers, timeout=None)  # No timeout for comprehensive processing

        if response.status_code != 200:
            logger.error(f"S2 API error: {response.status_code} - {response.text}")

        response.raise_for_status()  #Checks success of the GET request           

        results = response.json()
        return [paper for paper in results if paper is not None]
    
    @handle_api_errors("Semantic scholar similar papers", max_retries=2)
    def get_similar_papers(self, paper_id:str, limit: int=10) -> List[Dict]:
        """ Get similar papers using Semantic Scholar recommendations.
        Args:
            paper_id: the paper id we want to run semantic search on
            limit(int): limit of papers, set to 10 by default
        Returns:
            A list of dictionaries, where each dictionary represents a recommended 
            paper similar to the input paper.
        """
        if paper_id.isdigit():
            formatted_id = f"PMID:{paper_id}"
        else:
            formatted_id = paper_id

        url = f"{self.base_url}/paper/{formatted_id}/recommendations"  #recommendation endpoint
        params = {
            'fields': 'paperId,title,abstract,year,authors,journal,citationCount,tldr',
            'limit': min(limit, 100)  # S2 limit           
        }

        response = api_request(url, params=params, headers=self.headers)
        data = response.json()

        return data.get('data', [])

    

def get_semantic_scholar_client() -> SemanticScholarAPI:
    """Get Semantic Scholar API client."""
    return SemanticScholarAPI()
    

class LLMClient:
    """ OpenAI Client for LLM interactions."""
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or "gemma2:9b"  # Default model
        self.base_url = config.llm_base_url
        self.client = OpenAI(
            base_url=self.base_url,
            api_key='not_needed',
            timeout=None  # No timeout for comprehensive LLM processing
        )

    @handle_llm_errors("LLM text generation", max_retries=2)
    def generate(self, prompt: str, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, system_message: Optional[str] = None) -> Dict:
        """Generate text output from LLM."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or config.llm_temperature,
            max_tokens=max_tokens or config.llm_max_tokens,
            extra_body={
                "num_gpu": -1,  # Use all available GPU layers
                "num_thread": 8   # Optimize CPU threads
            }
        )
        return{
            'content':response.choices[0].message.content,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        }


def get_llm_client(model_name: Optional[str] = None) -> LLMClient:
    """Get LLM client."""
    return LLMClient(model_name)



    

