# test_pipeline.py
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add your project root to path
sys.path.append(str(Path(__file__).parent))

from src.data.data_collector import PubMedCollector
from src.data.probiotic_analyzer import CorrelationExtractor, LLMConfig
from src.data.correlation_verfier import CorrelationVerifier
from src.data.database_manager import DatabaseManager


def test_groq_pipeline():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Get Groq API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY not found in .env file")
        return
    
    db_manager = DatabaseManager()
    
    # Configure Groq for extraction with an active, instruction-following model
    extraction_config = LLMConfig(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model_name="llama-3.3-70b-versatile",  # Best current model for complex tasks
        temperature=0.1,  # Low temperature for consistent JSON formatting
        max_tokens=4000,  # Increased to avoid truncation
        cost_per_1k_input_tokens=0,  # Free tier
        cost_per_1k_output_tokens=0
    )
    
    extractor = CorrelationExtractor(extraction_config, db_manager)
    
    logger.info("=" * 50)
    logger.info("EXTRACTING CORRELATIONS WITH GROQ")
    logger.info("=" * 50)
    
    # Get papers that haven't been processed
    unprocessed = db_manager.get_unprocessed_papers(
        extraction_model="llama-3.3-70b-versatile",  # Update this to match your model
        limit=5
    )
    
    logger.info(f"Found {len(unprocessed)} unprocessed papers")
    
    if unprocessed:
        # Process 2 papers to test
        extraction_results = extractor.process_papers(unprocessed[:2])
        logger.info(f"Results: {extraction_results}")
    
    # Check final stats
    stats = db_manager.get_database_stats()
    logger.info(f"\nTotal correlations extracted: {stats['total_correlations']}")
    
    return extraction_results

if __name__ == "__main__":
    # Add to your .env file:
    # GROQ_API_KEY=gsk_your_actual_groq_key_here
    
    results = test_groq_pipeline()
    print("\nPIPELINE TEST COMPLETED")

