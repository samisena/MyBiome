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

sys.path.append(str(Path(__file__).parent.parent))

from pubmed_collector import PubMedCollector
from database_manager import DatabaseManager
from probiotic_analyzer import ProbioticAnalyzer, LLMConfig

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
    MISTRAL_7B = "mistral:7b"    # has .value attribute
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
            base_url = base_url,
            model_name= model.value,
            temperature= temperature,
            max_tokens= max_tokens
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
        

class LLMPipelineTester:
    """
    """

    #* We are able to access class values without making instances of the objects 1st because
    #* OllamaModel inherits from Enum module and we defined class level attributes for OllamaConfig 
    def __init__(self, 
                 ollama_base_url: str = OllamaConfig.DEFAULT_BASE_URL,
                 primary_model: OllamaModel = OllamaModel.MISTRAL_7B,
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

            #* Stores our collection output in the results dictionary
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

            #* 
            

            
        except:






        
        
    



        


    

