#!/usr/bin/env python3
"""
Script to run correlation extraction on papers in the database.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

# Also add the parent directory (back_end) so we can import src
back_end_dir = src_dir.parent
sys.path.insert(0, str(back_end_dir))

try:
    from src.paper_collection.database_manager import database_manager
    from src.llm.probiotic_analyzer import ProbioticAnalyzer
    from src.data.config import config, setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)

def run_correlation_extraction(limit: int = None, batch_size: int = 5):
    """Run correlation extraction on pending papers."""
    try:
        print("=== Running Correlation Extraction ===")
        
        # Initialize analyzer with default LLM configuration
        analyzer = ProbioticAnalyzer()
        
        # Get papers that need processing
        model_name = config.llm.model_name
        papers_to_process = database_manager.get_papers_for_processing(
            extraction_model=model_name,
            limit=limit
        )
        
        print(f"Found {len(papers_to_process)} papers to process")
        
        if not papers_to_process:
            print("No papers need processing!")
            return True
        
        # Process papers in batches
        results = analyzer.process_unprocessed_papers(
            limit=limit,
            batch_size=batch_size
        )
        
        print("\nCorrelation Extraction Results:")
        print(f"  Papers processed: {results.get('successful_papers', 0)}/{results.get('total_papers', 0)}")
        print(f"  Papers failed: {results.get('failed_papers', 0)}")
        print(f"  Total correlations found: {results.get('total_correlations', 0)}")
        print(f"  Processing time: {results.get('total_time', 0):.2f} seconds")
        
        # Show token usage if available
        token_usage = results.get('token_usage', {})
        if token_usage:
            print(f"  Total tokens: {token_usage.get('total_tokens', 0):,}")
            print(f"  Prompt tokens: {token_usage.get('prompt_tokens', 0):,}")
            print(f"  Completion tokens: {token_usage.get('completion_tokens', 0):,}")
        
        return results.get('successful_papers', 0) > 0
        
    except Exception as e:
        print(f"Error during correlation extraction: {e}")
        return False

def show_final_stats():
    """Show final database statistics."""
    print("\n=== Final Database Statistics ===")
    
    stats = database_manager.get_database_stats()
    print(f"Total papers: {stats.get('total_papers', 0)}")
    print(f"Total correlations: {stats.get('total_correlations', 0)}")
    print(f"Papers with fulltext: {stats.get('papers_with_fulltext', 0)}")
    
    print("\nProcessing Status:")
    for status, count in stats.get('processing_status', {}).items():
        print(f"  {status}: {count}")
    
    print("\nValidation Status:")
    for status, count in stats.get('validation_status', {}).items():
        print(f"  {status}: {count}")
    
    # Show some example correlations if any exist
    if stats.get('total_correlations', 0) > 0:
        print("\nSample Correlations:")
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT probiotic_strain, health_condition, correlation_type, 
                       correlation_strength, confidence_score
                FROM correlations 
                ORDER BY confidence_score DESC 
                LIMIT 5
            ''')
            
            for row in cursor.fetchall():
                print(f"  • {row[0]} → {row[1]} ({row[2]}, strength: {row[3]:.2f}, confidence: {row[4]:.2f})")

def main():
    """Main function to run correlation extraction."""
    print("MyBiome Correlation Extraction")
    print("=" * 40)
    
    # Setup logging
    logger = setup_logging(__name__, 'correlation_extraction.log')
    
    try:
        # Run correlation extraction on all pending papers
        success = run_correlation_extraction(
            limit=None,  # Process all papers
            batch_size=5  # Process in small batches to manage memory
        )
        
        if success:
            print("\n✅ Correlation extraction completed successfully!")
        else:
            print("\n❌ Correlation extraction failed or no correlations found")
        
        # Show final statistics
        show_final_stats()
        
        return success
        
    except Exception as e:
        logger.error(f"Correlation extraction failed: {e}")
        print(f"❌ Correlation extraction failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)