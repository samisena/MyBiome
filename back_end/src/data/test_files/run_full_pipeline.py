
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

from src.data.pubmed_collector_enhanced import PubMedCollector
from src.data.database_manager_enhanced import DatabaseManager
from src.data.probiotic_analyzer_enhanced import ProbioticAnalyzer, LLMConfig
from src.data.test_files.llm_pipeline import LLMPipeline, OllamaConfig, OllamaModel

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


def main(condition: str = "IBS", 
         max_papers: int = 5, 
         primary_model: str = "deepseek-llm:7b-chat", 
         secondary_model: str = "llama3.1:8b",
         ollama_url: str = "http://localhost:11434/v1"):
    
    """Main function to run the complete Ollama pipeline test.
    
    Args:
        condition: Health condition to search for
        max_papers: Maximum papers to process
        primary_model: Primary Ollama model name
        secondary_model: Secondary Ollama model name 
        ollama_url: Ollama server URL
    """

    logger.info("Starting Ollama LLM Pipeline Test...")
    
    # Convert string model names to enum values
    try:
        primary_enum = OllamaModel(primary_model)
        secondary_enum = OllamaModel(secondary_model)
    except ValueError as e:
        logger.error(f"Invalid model name: {e}")
        logger.error(f"Available models: {[m.value for m in OllamaModel]}")
        return False
    
    # Initialize tester with specified models
    tester = LLMPipeline(
        ollama_base_url=ollama_url,
        primary_model=primary_enum,
        secondary_model=secondary_enum
    )
    
    # Step 1: Validate setup
    if not tester.validate_setup():
        logger.error("Setup validation failed. Exiting...")
        return False
    
    # Step 2: Collect test data
    papers = tester.collect_test_data(condition=condition, max_papers=max_papers)
    if not papers:
        logger.error("No papers collected. Exiting...")
        return False
    
    # Step 3: Process with primary model
    primary_results = tester.process_with_primary_model(papers)
    
    # Step 4: Process with secondary model
    secondary_results = tester.process_with_secondary_model(papers)
    
    # Step 5: Compare results
    tester.compare_results(primary_results, secondary_results)
    
    # Step 6: Save and display results
    results_file = tester.save_results()
    tester.print_summary()
    
    logger.info(f"Ollama pipeline test completed successfully!")
    logger.info(f"Results saved to: {results_file}")
    
    return True


def run_with_custom_models():
    """Example function showing how to run with different model configurations."""
    print("Available Ollama models:")
    for model in OllamaModel:
        print(f"  - {model.value}")
    
    # Example: Test different model combinations
    print("\n=== Testing DeepSeek vs Llama ===")
    success1 = main(primary_model="deepseek-llm:7b-chat", secondary_model="llama3.1:8b")
    
    if success1:
        print("\n=== Testing Llama vs DeepSeek (reversed) ===")
        success2 = main(primary_model="llama3.1:8b", secondary_model="deepseek-llm:7b-chat")
        return success1 and success2
    
    return success1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)