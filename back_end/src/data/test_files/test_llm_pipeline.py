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

from src.data.pubmed_collector import PubMedCollector
from src.data.database_manager import DatabaseManager
from src.data.probiotic_analyzer import ProbioticAnalyzer, LLMConfig


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


#* Sets up an easy way to reference models
class OllamaModel(Enum):   # The class inherits from the Enum module
    """Enum for available Ollama models."""
    MISTRAL_7B = "mistral:7b"
    LLAMA_3_1_8B = "llama3.1:8b"


class OllamaConfig:
    """ 
    """

    #* We define the endpoint URL for our locally installed models
    DEFAULT_BASE_URL = "http://localhost:11434/v1"  #API endpoint for Ollama local server
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 4096



    @classmethod  #class methods are called within a main class (OllaConfig) and use cls to refer to the subclass
                    #itself of self that refers to the main class (OllamaConfig)
    def create_config(cls):
        """
        """

        return LLMConfig(
            base_url = base_url,
            model_name= model.value,
        

        )
