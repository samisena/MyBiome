#!/usr/bin/env python3
"""
Debug script to test LLM extraction with detailed logging and step-by-step verification.
This will help identify exactly where the extraction is failing.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

# Set up paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
src_dir = project_root / "back_end" / "src"
sys.path.insert(0, str(src_dir))

# Load environment variables
env_path = project_root / '.env'
load_dotenv(env_path)

# Import our custom modules
sys.path.append(str(current_dir.parent))
from database_manager import DatabaseManager
from probiotic_analyzer import ProbioticAnalyzer, LLMConfig

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_api_keys():
    """Test if API keys are properly loaded and accessible."""
    logger.info("=== TESTING API KEYS ===")
    
    keys_to_test = {
        'GEMINI_KEY': os.getenv('GEMINI_KEY'),
        'OPEN_ROUTER_KEY': os.getenv('OPEN_ROUTER_KEY'),
        'DEEPSEEK_KEY': os.getenv('DEEPSEEK_KEY')
    }
    
    for key_name, key_value in keys_to_test.items():
        if key_value:
            logger.info(f"✓ {key_name}: Found (length: {len(key_value)}, first 10 chars: {key_value[:10]}...)")
        else:
            logger.error(f"✗ {key_name}: NOT FOUND")
    
    return all(keys_to_test.values())

def test_database_connection():
    """Test database connection and retrieve test paper."""
    logger.info("=== TESTING DATABASE CONNECTION ===")
    
    try:
        db = DatabaseManager()
        logger.info("✓ Database connection established")
        
        # Get the specific paper that should have correlations (PMID: 37184752)
        paper = db.get_paper_by_pmid("37184752")
        if paper:
            logger.info(f"✓ Found test paper: {paper['pmid']}")
            logger.info(f"Title: {paper['title']}")
            logger.info(f"Abstract length: {len(paper.get('abstract', ''))}")
            logger.info(f"Abstract preview: {paper.get('abstract', '')[:200]}...")
            return paper
        else:
            logger.error("✗ Test paper not found in database")
            return None
            
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return None

def test_llm_configs():
    """Test LLM configurations."""
    logger.info("=== TESTING LLM CONFIGURATIONS ===")
    
    # Gemini configuration - using correct API endpoint
    gemini_config = LLMConfig(
        api_key=os.getenv("GEMINI_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # OpenAI-compatible endpoint
        model_name="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=2048
    )
    
    # DeepSeek configuration via OpenRouter
    deepseek_config = LLMConfig(
        api_key=os.getenv("OPEN_ROUTER_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-chat",
        temperature=0.3,
        max_tokens=2048
    )
    
    logger.info("Gemini config:")
    logger.info(f"  API Key present: {bool(gemini_config.api_key)}")
    logger.info(f"  Base URL: {gemini_config.base_url}")
    logger.info(f"  Model: {gemini_config.model_name}")
    
    logger.info("DeepSeek config:")
    logger.info(f"  API Key present: {bool(deepseek_config.api_key)}")
    logger.info(f"  Base URL: {deepseek_config.base_url}")
    logger.info(f"  Model: {deepseek_config.model_name}")
    
    return gemini_config, deepseek_config

def test_prompt_generation(paper: Dict):
    """Test prompt generation with the specific paper."""
    logger.info("=== TESTING PROMPT GENERATION ===")
    
    db = DatabaseManager()
    config = LLMConfig(
        api_key="dummy",
        base_url="dummy",
        model_name="dummy",
        temperature=0.3,
        max_tokens=2048
    )
    
    analyzer = ProbioticAnalyzer(config, db)
    prompt = analyzer.llm_prompt(paper)
    
    logger.info(f"Generated prompt length: {len(prompt)}")
    logger.info("=== FULL PROMPT ===")
    logger.info(prompt)
    logger.info("=== END PROMPT ===")
    
    # Check if the prompt contains the expected data
    abstract = paper.get('abstract', '')
    if 'Lactobacillus plantarum DSM 9843' in abstract:
        logger.info("✓ Target strain found in original abstract")
        if 'Lactobacillus plantarum DSM 9843' in prompt:
            logger.info("✓ Target strain found in generated prompt")
        else:
            logger.error("✗ Target strain NOT found in generated prompt")
    
    return prompt

def test_llm_api_call(config: LLMConfig, paper: Dict, model_name: str):
    """Test individual LLM API call with detailed logging."""
    logger.info(f"=== TESTING {model_name.upper()} API CALL ===")
    
    try:
        db = DatabaseManager()
        analyzer = ProbioticAnalyzer(config, db)
        
        # Log the exact request being made
        prompt = analyzer.llm_prompt(paper)
        logger.info(f"Paper PMID: {paper['pmid']}")
        logger.info(f"Paper title: {paper['title']}")
        logger.info(f"Abstract length: {len(paper.get('abstract', ''))}")
        
        # Make the API call
        logger.info("Making API call...")
        correlations = analyzer.extract_correlations(paper)
        
        logger.info(f"API call completed. Returned {len(correlations)} correlations")
        if correlations:
            logger.info("Correlations found:")
            for i, corr in enumerate(correlations):
                logger.info(f"  {i+1}. {corr.get('probiotic_strain', 'Unknown strain')} -> {corr.get('health_condition', 'Unknown condition')}")
                logger.info(f"      Type: {corr.get('correlation_type', 'Unknown')}")
                logger.info(f"      Confidence: {corr.get('confidence_score', 'Unknown')}")
        else:
            logger.warning("No correlations extracted")
            
        return correlations
        
    except Exception as e:
        logger.error(f"✗ {model_name} API call failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def test_json_parsing():
    """Test JSON parsing with sample responses."""
    logger.info("=== TESTING JSON PARSING ===")
    
    # Test with expected good response
    good_response = '''[{"probiotic_strain": "Lactobacillus plantarum DSM 9843", "health_condition": "IBS", "correlation_type": "positive", "correlation_strength": 0.7, "confidence_score": 0.8, "supporting_quote": "may be efficacious in some patients with IBS"}]'''
    
    db = DatabaseManager()
    config = LLMConfig(api_key="dummy", base_url="dummy", model_name="test", temperature=0.3, max_tokens=2048)
    analyzer = ProbioticAnalyzer(config, db)
    
    logger.info("Testing with good JSON response:")
    logger.info(good_response)
    
    result = analyzer.parse_json_response(good_response, "test_pmid")
    logger.info(f"Parsed result: {len(result)} correlations")
    
    if result:
        for corr in result:
            logger.info(f"  - {corr['probiotic_strain']} -> {corr['health_condition']} ({corr['correlation_type']})")
    
    # Test with empty response
    logger.info("\nTesting with empty response:")
    empty_result = analyzer.parse_json_response("[]", "test_pmid")
    logger.info(f"Empty response parsed: {len(empty_result)} correlations")

def main():
    """Main debug function."""
    logger.info("Starting comprehensive LLM extraction debugging...")
    
    # Step 1: Test API keys
    if not test_api_keys():
        logger.error("API key test failed. Cannot proceed.")
        return False
    
    # Step 2: Test database connection and get test paper
    paper = test_database_connection()
    if not paper:
        logger.error("Database test failed. Cannot proceed.")
        return False
    
    # Step 3: Test prompt generation
    prompt = test_prompt_generation(paper)
    
    # Step 4: Test JSON parsing
    test_json_parsing()
    
    # Step 5: Test LLM configurations
    gemini_config, deepseek_config = test_llm_configs()
    
    # Step 6: Test individual API calls
    logger.info("\n" + "="*60)
    logger.info("TESTING LIVE API CALLS")
    logger.info("="*60)
    
    # Test Gemini
    gemini_correlations = test_llm_api_call(gemini_config, paper, "Gemini")
    
    # Test DeepSeek
    deepseek_correlations = test_llm_api_call(deepseek_config, paper, "DeepSeek")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Gemini correlations found: {len(gemini_correlations)}")
    logger.info(f"DeepSeek correlations found: {len(deepseek_correlations)}")
    
    if len(gemini_correlations) == 0 and len(deepseek_correlations) == 0:
        logger.error("BOTH MODELS FAILED TO EXTRACT CORRELATIONS!")
        logger.error("This indicates a systematic issue with:")
        logger.error("1. API configuration")
        logger.error("2. Prompt effectiveness") 
        logger.error("3. Model capability")
        logger.error("4. Response parsing")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)