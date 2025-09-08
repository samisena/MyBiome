import os
import sys
import time
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum
from dotenv import load_dotenv

from src.data.pubmed_collector_enhanced import EnhancedPubMedCollector as PubMedCollector
from src.data.database_manager_enhanced import EnhancedDatabaseManager as DatabaseManager
from src.data.probiotic_analyzer_enhanced import EnhancedProbioticAnalyzer as ProbioticAnalyzer, LLMConfig

# Load environment variables
load_dotenv()

#* Configure logging to save log hisotry to 'llm_pipeline_test.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_pipeline_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


#* Sets up an easy way to reference models using .value attribute
class OllamaModel(Enum):   # The class inherits from the Enum module
    """Enum for available Ollama models."""
    DEEPSEEK_LLM_7B = "deepseek-llm:7b-chat"    # has .value attribute
    LLAMA_3_1_8B = "llama3.1:8b"


class OllamaConfig:
    """ 
    A set up of class methods to assist in configuring our local LLMs
    """

    #* We define the following class variables that are the same for all instances of the class OllamaConfig
    #* These values can be accessed via OllamaConfig.DEFAULT_X 
    DEFAULT_BASE_URL = "http://localhost:11434/v1"  #API endpoint for Ollama local server
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 4096

    @classmethod    #?class methods are called within a main class (OllamaConfig). They use cls to refer to the the class
                    #?itself instead of self that refers to an instance of the class with given attributes
                    #? This will be useful later with class inheritance
    def create_config(cls, model: OllamaModel, base_url: str = DEFAULT_BASE_URL,
                      temperature = DEFAULT_TEMPERATURE, max_tokens = DEFAULT_MAX_TOKENS
                      ) -> LLMConfig:
        """
        Creates an LLMConfig Object given our specific Ollama model

        Args:
            model: OllamaModel enum value
            base_url: Ollama server URL (default: localhost:11434/v1)
            temperature: Model temperature (default: 0.3)
            max_tokens: Maximum tokens (default: 4096)
            
        Returns:
            LLMConfig instance for the specified Ollama model
        """

        return LLMConfig(
            base_url=base_url,
            model_name=model.value,
            temperature=temperature,
            max_tokens=max_tokens
        )
    

    @classmethod
    def check_ollama_server_and_model(cls, model: OllamaModel, base_url: str = DEFAULT_BASE_URL) -> bool:
        """
        Checks if the Ollama server is running and if the specified model is available.

        Args:
            model: OllamaModel enum value 
            base_url: Ollama server url, "http://localhost:11434/v1" by default

        Returns:
            True if the server is accessible and the model is available, False otherwise
        """
        try:
            # Remove /v1 from base URL for model check
            models_url = base_url.replace('/v1', '') + '/api/tags'
            response = requests.get(models_url, timeout=5)
            
            if response.status_code != 200:  #Any response other 200 indicates an error
                logger.error(f"Failed to connect to Ollama at {base_url}: HTTP {response.status_code}")
                return False
                
            # Check if model is available
            # The response should have this format:
            #               {
            #   "models": [
            #        {
            #        "name": "llama2:latest",
            #        "modified_at": "2024-01-15T10:30:00Z",
            #        "size": 3825819519
            #        },
            #        {
            #        "name": "mistral:7b",
            #        "modified_at": "2024-01-14T09:15:00Z", 
            #        "size": 4109856789
            #        }
            #    ]
            #    }

            models_data = response.json()  #Parses json response to python dictionary
            available_models = [m['name'] for m in models_data.get('models', [])]  

            #* Checks if our selected model is available via Ollama server
            if model.value not in available_models:
                logger.error(f"Model {model.value} not available. Available models: {available_models}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {base_url} or check model {model.value}: {e}")
            return False    
        

class LLMPipeline:
    """
    """

    #* We are able to access class values without making instances of the objects 1st because
    #* OllamaModel inherits from Enum module and we defined class level attributes for OllamaConfig 
    def __init__(self, 
                 ollama_base_url: str = OllamaConfig.DEFAULT_BASE_URL,
                 primary_model: OllamaModel = OllamaModel.DEEPSEEK_LLM_7B,
                 secondary_model: OllamaModel = OllamaModel.LLAMA_3_1_8B):
        
        """ 
        Initializes our pipeline

        Args:
            ollama_base_url: Ollama server URL
            primary_model: Primary model for processing
            secondary_model: Secondary model for comparison
        """

        #* We create instances of our custom modules:
        self.pubmed_collector = PubMedCollector()
        self.db_manager = DatabaseManager()
        self.ollama_base_url = ollama_base_url  #class variable from OllamaConfig

        #* We assign our models their names
        self.primary_model = primary_model
        self.secondary_model = secondary_model

        #* We create LLMConfig objects for each model (to be passed to ProbioticAnalyzer object)
        self.primary_config = OllamaConfig.create_config(
            model = primary_model,
            base_url= ollama_base_url
        )

        self.secondary_config = OllamaConfig.create_config(
            model=secondary_model,
            base_url=ollama_base_url
        )

        self.results = {
            'data_collection': {},   #data about the papers collected
            'primary_processing': {},  #1st model outputs
            'secondary_processing': {},  #2nd model outputs
            'comparison': {},  #comapraison between the two outputs
            'model_info': {
                'primary_model': primary_model.value,
                'secondary_model': secondary_model.value,
                'ollama_url': ollama_base_url
            }
        }


    def validate_setup(self) -> bool:
        """
        A series of system checks to make sure everything is ready to start collecting and processing papers.
        """

        logger.info("Validating Ollama setup...")

        #* Pubmed API access check:
        ncbi_key = os.getenv('NCBI_API_KEY')  #We load our pubmed API key
        if not ncbi_key:
            logger.error("Missing NCBI_API_KEY for PubMed access")
            return False
        logger.info("OK NCBI_API_KEY found")

        #* Ollama server and models check:
        if not OllamaConfig.check_ollama_server_and_model(self.primary_model, self.ollama_base_url):
            logger.error(f"Cannot connect to Ollama at {self.ollama_base_url} or model {self.primary_model.value} not available")
            return False
        logger.info(f"OK Ollama connection established and primary model {self.primary_model.value} available") 

        if not OllamaConfig.check_ollama_server_and_model(self.secondary_model, self.ollama_base_url):
            logger.error(f"Model {self.secondary_model.value} not available in Ollama")
            logger.error(f"Pull the model with: ollama pull {self.secondary_model.value}")
            return False
        logger.info(f"OK Secondary model {self.secondary_model.value} available")
        
        logger.info("All setup validation passed!")
        return True       
    

    def switch_models(self, new_primary: OllamaModel, new_secondary: OllamaModel) -> bool:
        """
        """

        logger.info(f"Switching models: {new_primary.value} and {new_secondary.value}")

        #* Check if the models are available via Ollama:
        if not OllamaConfig.check_ollama_server_and_model(new_primary, self.ollama_base_url):
            logger.error(f"New primary model {new_primary.value} not available")
            return False
            
        if not OllamaConfig.check_ollama_server_and_model(new_secondary, self.ollama_base_url):
            logger.error(f"New secondary model {new_secondary.value} not available")
            return False

        #* Update our instance variables with the new models:
        self.primary_model = new_primary
        self.secondary_model = new_secondary

        self.primary_config = OllamaConfig.create_config(
            model=new_primary,
            base_url=self.ollama_base_url
        )
        
        self.secondary_config = OllamaConfig.create_config(
            model=new_secondary, 
            base_url=self.ollama_base_url
        )

        self.results['model_info'].update({
            'primary_model': new_primary.value,
            'secondary_model': new_secondary.value
        })

        logger.info("Model switch succesful")
        return True
    

    def collect_test_data(self, condition, max_papers:int = 5) -> List[Dict]:
        """
        Collect test data from PubMed for a specific condition and stores them in a 'results' dictionary
        to be later saved as JSON in save_results() and displayed. 

        Args:
            condition: the health condition search term
            max_papers: the max number of papers the pipeline can retrieve (defaults to 5)
        """

        logger.info(f"Starting paper collection for condition: {condition}")
        logger.info(f"Maximum papers to collect: {max_papers}")

        try:
            collection_result = self.pubmed_collector.collect_probiotics_by_condition(
                condition=condition,
                max_results=max_papers
            )

            #* Stores our paper collection in the results dictionary
            self.results['data_collection'] = {
                'condition': condition,
                'requested_papers': max_papers,
                'papers_found': collection_result.get('paper_count', 0),
                'status': collection_result.get('status', 'unknown'),
                'metadata_file': collection_result.get('metadata_file', '')}
            
            logger.info(f"Data collection completed:")
            logger.info(f"  - Condition: {condition}")
            logger.info(f"  - Papers found: {collection_result.get('paper_count', 0)}")  #0 if not found
            logger.info(f"  - Status: {collection_result.get('status', 'unknown')}") #unknown if not found

            #? If the paper collection was succesful then, the results would have gotten storred in the database
            #? Therefore we fetch these papers according to the 'condition' from the database.
            if collection_result.get('status') == 'success' and collection_result.get('paper_count', 0) > 0:
                papers = self.db_manager.get_papers_by_condition(condition)
                logger.info(f"Retrieved {len(papers)} papers from database for processing")
                return papers[:max_papers]
            else:
                logger.warning("No papers collected or collection failed")
        
        except Exception as e:
            logger.error(f"Error during paper collection:{e}")
            self.results['data_collection']['error'] = str(e)  # adds the error to our results dictionary
            return []
        

    def process_with_primary_model(self, papers: List[Dict]) -> Dict:
        """
        Processes a set of papers with our model #1
        """

        logger.info(f"Starting {self.primary_model.value} processing...")

        try:
            #* We create a ProbioticAnalyzer instance that can extract correlations from papers
            primary_analyzer = ProbioticAnalyzer(self.primary_config, self.db_manager)

            start_time = time.time()  #time at the start of the extraction process
            results = self._process_papers_with_retry(primary_analyzer, papers, save_to_db=True)
            end_time = time.time()  #time at the end of the process

            processing_time = end_time - start_time

            #* Adds the primary model's extraction data to the results dictionary
            self.results['primary_processing'] = {
                'model': self.primary_config.model_name,
                'papers_processed': results['successful_papers'],
                'total_papers': results['total_papers'],
                'failed_papers': results['failed_papers'],
                'correlations_found': results['total_correlations'],
                'processing_time_seconds': processing_time,
                'input_tokens': primary_analyzer.total_input_tokens,
                'output_tokens': primary_analyzer.total_output_tokens,
                'correlations': results['correlations']
            }

            logger.info(f"{self.primary_model.value} processing completed")
            logger.info(f"  - Papers processed: {results['successful_papers']}/{results['total_papers']}")
            logger.info(f"  - Correlations found: {results['total_correlations']}")
            logger.info(f"  - Processing time: {processing_time:.2f} seconds")
            
            return results
        
        except Exception as e:
            logger.error(f"Error in primary model processing: {e}")
            self.results['primary_processing'] = {
                'error': str(e),
                'model': self.primary_config.model_name,
                'papers_processed': 0,
                'total_papers': len(papers),
                'failed_papers': [p.get('pmid', 'unknown') for p in papers],
                'correlations_found': 0,
                'correlations': []
            }
            return self.results['primary_processing']
        

    def process_with_secondary_model(self, papers: List[Dict]) -> Dict:
        """
        Processes a set of papers with our model #2
        """

        logger.info(f"Starting {self.secondary_model.value} processing...")
        
        try:
            # Create analyzer with error handling
            secondary_analyzer = ProbioticAnalyzer(self.secondary_config, self.db_manager)
            
            # Process papers with retry logic
            start_time = time.time()
            results = self._process_papers_with_retry(secondary_analyzer, papers, save_to_db=True)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            self.results['secondary_processing'] = {
                'model': self.secondary_config.model_name,
                'papers_processed': results['successful_papers'],
                'total_papers': results['total_papers'],
                'failed_papers': results['failed_papers'],
                'correlations_found': results['total_correlations'],
                'processing_time_seconds': processing_time,
                'input_tokens': secondary_analyzer.total_input_tokens,
                'output_tokens': secondary_analyzer.total_output_tokens,
                'correlations': results['correlations']
            }
            
            logger.info(f"{self.secondary_model.value} processing completed:")
            logger.info(f"  - Papers processed: {results['successful_papers']}/{results['total_papers']}")
            logger.info(f"  - Correlations found: {results['total_correlations']}")
            logger.info(f"  - Processing time: {processing_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in secondary model processing: {e}")
            self.results['secondary_processing'] = {
                'error': str(e),
                'model': self.secondary_config.model_name,
                'papers_processed': 0,
                'total_papers': len(papers),
                'failed_papers': [p.get('pmid', 'unknown') for p in papers],
                'correlations_found': 0,
                'correlations': []
            }
            return self.results['secondary_processing']   


    def _process_papers_with_retry(self,  analyzer: ProbioticAnalyzer, 
                                   papers: List[Dict], save_to_db: bool, max_retries: int=2
                                   ) -> Dict:
        """
        Process papers with retry logic for more robustness
        """

        #* Tries to process papers:
        for attempt in range(max_retries + 1):
            try:
                results = analyzer.process_papers(papers, save_to_db)
                return results

            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Processing attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt) #Exponential backoff
                else:
                    logger.error(f"All processing attempts failed: {e}")
                    raise


    def compare_results(self, primary_results: Dict, secondary_results: Dict) -> Dict:
        """
        """

        logger.info(f"""Comparing results between {self.primary_model.value} 
                    and {self.secondary_model.value}...""")
        
        primary_correlations = primary_results.get('correlations', [])   #returns empty if error occurs
        secondary_correlations = secondary_results.get('correlations', [])

        #* Extarcts the unique probiotic strains found by each model 
        #? sets automatically delete duplicates
        primary_strains = set()
        secondary_strains = set()

        for corr in primary_correlations:
            if 'probiotic_strain' in corr:
                primary_strains.add(corr['probiotic_strain'])
        for corr in secondary_correlations:
            if 'probiotic_strain' in corr:
                secondary_strains.add(corr['probiotic_strain'])

        #* Find common strains between models
        common_strains = primary_strains.intersection(secondary_strains)

        #* Find unique strains 
        primary_only = primary_strains - secondary_strains
        secondary_only = secondary_strains - primary_strains

        #* Comparaison between the model's outputs
        comparison = {
            'primary_correlations_count': len(primary_correlations),
            'secondary_correlations_count': len(secondary_correlations),
            'primary_unique_strains': len(primary_strains),
            'secondary_unique_strains': len(secondary_strains),
            'common_strains_count': len(common_strains),
            'common_strains': list(common_strains),
            'primary_only_strains': list(primary_only),
            'secondary_only_strains': list(secondary_only),
            'agreement_percentage': (len(common_strains) / max(len(primary_strains.union(secondary_strains)), 1)) * 100,
            'primary_model': self.primary_model.value,
            'secondary_model': self.secondary_model.value
        }

        #* Adds this comparaison to the results dictionary
        self.results['comparison'] = comparison

        logger.info("Results comparison:")
        logger.info(f"  - {self.primary_model.value} found {len(primary_correlations)} correlations with {len(primary_strains)} unique strains")
        logger.info(f"  - {self.secondary_model.value} found {len(secondary_correlations)} correlations with {len(secondary_strains)} unique strains")
        logger.info(f"  - Common strains: {len(common_strains)}")
        logger.info(f"  - Agreement percentage: {comparison['agreement_percentage']:.1f}%")
        
        return comparison
        

    def save_results(self, output_file:str = None) -> str:
        """
        Saves the results dictionary to a JSON file.
        """

        if output_file is None:
            timestamp = int(time.time())
            output_file = f"ollama_pipeline_test_results_{timestamp}.json"

        #* Saves the JSON in 'src/data/test_files' folder:
        current_dir = Path(__file__).parent
        output_path = current_dir / output_file

                # Add metadata
        self.results['metadata'] = {
            'test_timestamp': time.time(),
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': sys.version,
            'ollama_base_url': self.ollama_base_url,
            'models_tested': [
                self.primary_config.model_name,
                self.secondary_config.model_name
            ]
        }
    
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        return str(output_path)
    
    def print_summary(self):
        """Print a summary of the test results."""
        print("\n" + "="*60)
        print("OLLAMA LLM PIPELINE TEST SUMMARY")
        print("="*60)
        
        # Data Collection Summary
        dc = self.results.get('data_collection', {})
        print(f"\nDATA COLLECTION:")
        print(f"  Condition searched: {dc.get('condition', 'Unknown')}")
        print(f"  Papers found: {dc.get('papers_found', 0)}")
        print(f"  Status: {dc.get('status', 'Unknown')}")
        
        # Primary Model Summary
        pp = self.results.get('primary_processing', {})
        print(f"\nPRIMARY MODEL ({self.primary_model.value.upper()}) PROCESSING:")
        print(f"  Model: {pp.get('model', 'Unknown')}")
        print(f"  Papers processed: {pp.get('papers_processed', 0)}/{pp.get('total_papers', 0)}")
        print(f"  Correlations found: {pp.get('correlations_found', 0)}")
        print(f"  Processing time: {pp.get('processing_time_seconds', 0):.2f}s")
        if 'error' in pp:
            print(f"  Error: {pp['error']}")
        
        # Secondary Model Summary
        sp = self.results.get('secondary_processing', {})
        print(f"\nSECONDARY MODEL ({self.secondary_model.value.upper()}) PROCESSING:")
        print(f"  Model: {sp.get('model', 'Unknown')}")
        print(f"  Papers processed: {sp.get('papers_processed', 0)}/{sp.get('total_papers', 0)}")
        print(f"  Correlations found: {sp.get('correlations_found', 0)}")
        print(f"  Processing time: {sp.get('processing_time_seconds', 0):.2f}s")
        if 'error' in sp:
            print(f"  Error: {sp['error']}")
        
        # Comparison Summary
        comp = self.results.get('comparison', {})
        print(f"\nCOMPARISON:")
        print(f"  Common strains identified: {comp.get('common_strains_count', 0)}")
        print(f"  Agreement percentage: {comp.get('agreement_percentage', 0):.1f}%")
        print(f"  {self.primary_model.value}-only strains: {comp.get('primary_only_strains', [])}")
        print(f"  {self.secondary_model.value}-only strains: {comp.get('secondary_only_strains', [])}")
        
        print("="*60)
        
        
    



        


    

