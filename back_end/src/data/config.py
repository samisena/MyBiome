"""
Unified configuration management for the MyBiome research platform.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

class UnifiedConfig:
    """Unified configuration class that handles all system settings."""

    def __init__(self):
        #* Core data locations
        self.project_root = project_root   #MyBiome directory
        self.data_root = project_root / "data"
        self.raw_data = project_root / "data" / "raw"
        self.processed_data = project_root / "data" / "processed"
        self.logs_dir = project_root / "data" / "logs"
        self.papers_dir = project_root / "data" / "raw" / "papers"
        self.metadata_dir = project_root / "data" / "raw" / "metadata"
        self.fulltext_dir = project_root / "data" / "raw" / "fulltext"
        self.pmc_dir = project_root / "data" / "raw" / "fulltext" / "pmc"  
        self.pdf_dir = project_root / "data" / "raw" / "fulltext" / "pdf" #full papers  

        #* Database configuration
        self.db_name = 'intervention_research.db'
        self.db_path = self.processed_data / self.db_name  #path to database
        self.max_connections = 5  # SQLite connection pooling

        #* API configuration
        self.ncbi_api_key = os.getenv("NCBI_API_KEY")
        self.email = os.getenv("EMAIL", "samisena@outlook.com")
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.pmc_base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        self.unpaywall_base_url = "https://api.unpaywall.org/v2"
        self.api_timeout = None  # Remove timeout for comprehensive LLM processing
        self.max_retries = 3
        self.api_delay = 0.5  # Unified delay for all APIs      

        #* LLM configuration
        self.llm_base_url = "http://localhost:11434/v1"
        self.llm_temperature = 0.3
        self.llm_max_tokens = 4096   
        self.dual_models = ["gemma2:9b", "qwen2.5:14b"]

        #* Data collection
        self.intervention_categories = [
            "exercise", "diet", "supplement", "medication", "therapy", "lifestyle"
        ]
        # Processing batch size: Number of papers processed per LLM batch
        # Small batches (5) provide better error recovery and thermal management
        self.intervention_batch_size = 5
        # Collection target: Maximum papers collected per health condition
        # Higher values (100) ensure comprehensive coverage while managing API limits
        self.max_papers_per_condition = 100

        #* Medical rotation pipeline configuration
        self.rotation_papers_per_condition = 10  # Papers collected per condition in rotation
        self.medical_specialties = {
            "cardiology": [
                "coronary artery disease",
                "heart failure",
                "hypertension",
                "atrial fibrillation",
                "valvular heart disease"
            ],
            "neurology": [
                "stroke",
                "alzheimer disease",
                "parkinson disease",
                "epilepsy",
                "multiple sclerosis"
            ],
            "gastroenterology": [
                "gastroesophageal reflux disease",
                "inflammatory bowel disease",
                "irritable bowel syndrome",
                "cirrhosis",
                "peptic ulcer disease"
            ],
            "pulmonology": [
                "chronic obstructive pulmonary disease",
                "asthma",
                "pneumonia",
                "lung cancer",
                "pulmonary embolism"
            ],
            "endocrinology": [
                "diabetes mellitus",
                "thyroid disorders",
                "obesity",
                "osteoporosis",
                "polycystic ovary syndrome"
            ],
            "nephrology": [
                "chronic kidney disease",
                "acute kidney injury",
                "kidney stones",
                "glomerulonephritis",
                "polycystic kidney disease"
            ],
            "oncology": [
                "lung cancer",
                "breast cancer",
                "colorectal cancer",
                "prostate cancer",
                "leukemia"
            ],
            "rheumatology": [
                "rheumatoid arthritis",
                "osteoarthritis",
                "systemic lupus erythematosus",
                "gout",
                "fibromyalgia"
            ],
            "psychiatry": [
                "major depressive disorder",
                "anxiety disorders",
                "bipolar disorder",
                "schizophrenia",
                "attention deficit hyperactivity disorder"
            ],
            "orthopedics": [
                "fractures",
                "osteoarthritis",
                "back pain",
                "rotator cuff tear",
                "anterior cruciate ligament injury"
            ],
            "dermatology": [
                "acne vulgaris",
                "atopic dermatitis",
                "psoriasis",
                "skin cancer",
                "rosacea"
            ],
            "infectious_disease": [
                "human immunodeficiency virus",
                "tuberculosis",
                "hepatitis b",
                "sepsis",
                "influenza"
            ]
        }

        #* Performance settings - FAST_MODE enabled by default for optimal performance
        self.fast_mode = os.getenv('FAST_MODE', '1').lower() in ('1', 'true', 'yes')

        #* Creates the directories
        self._ensure_directories()

    def _ensure_directories(self):
        """Creates the necessary directories"""
        directories = [
            self.raw_data, self.processed_data, self.logs_dir,
            self.papers_dir, self.metadata_dir, self.fulltext_dir, 
            self.pmc_dir, self.pdf_dir            
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_llm_config(self, model_name: Optional[str]=None) -> Dict[str, Any]:
        """Get LLM configuration for a specific model."""
        return {
            'base_url': self.llm_base_url,
            'model_name': model_name or "gemma2:9b",
            'temperature': self.llm_temperature,
            'max_tokens': self.llm_max_tokens,
            'api_key': "not-needed"  # For local Ollama
        }       
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration (NCBI API key, email and writing permissions)."""
        issues = []
        warnings = []
        
        # Check API key
        if not self.ncbi_api_key:
            warnings.append("Missing NCBI_API_KEY - API rate limits will apply")
        
        # Check email format
        if "@" not in self.email:
            issues.append("Invalid email format")
        
        # Check directory permissions
        try:
            test_file = self.processed_data / ".test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            issues.append(f"Cannot write to processed data directory: {e}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings
        }


def setup_logging(name: str, log_file: Optional[str] = None,
                 level: int = logging.ERROR) -> logging.Logger:
    """Centralized logging setup."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    # FAST_MODE: Disable all logging except CRITICAL errors (enabled by default)
    fast_mode = os.getenv('FAST_MODE', '1').lower() in ('1', 'true', 'yes')
    if fast_mode:
        level = logging.CRITICAL
        # Only console handler for critical errors, no file logging
        logger.setLevel(level)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(console_handler)
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = config.logs_dir / log_file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global unified configuration instance
config = UnifiedConfig()

# Backward compatibility exports
class LLMConfig:
    """Backward compatibility class."""
    def __init__(self, model_name=None, base_url=None, temperature=0.3, max_tokens=4096, api_key="not-needed"):
        self.model_name = model_name or config.dual_models[0]
        self.base_url = base_url or config.llm_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

