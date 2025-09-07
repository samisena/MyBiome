"""
Example usage of the enhanced MyBiome data pipeline.
This demonstrates how to use the new architecture effectively.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.config import config, setup_logging
from src.data.enhanced_pipeline import EnhancedResearchPipeline
from src.data.migrate_to_enhanced import main as run_migration


def example_basic_research():
    """Example: Basic research workflow for a few conditions."""
    print("=== Example: Basic Research Workflow ===")
    
    # Initialize pipeline
    pipeline = EnhancedResearchPipeline()
    
    # Define research conditions
    conditions = [
        "irritable bowel syndrome", 
        "inflammatory bowel disease",
        "anxiety"
    ]
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        conditions=conditions,
        max_papers_per_condition=20,  # Small number for example
        include_fulltext=True,
        analyze_limit=50  # Limit analysis to 50 papers
    )
    
    # Print summary
    pipeline.print_pipeline_summary()
    
    return results


def example_data_collection_only():
    """Example: Just collect papers without analysis."""
    print("\n=== Example: Data Collection Only ===")
    
    from src.data.pubmed_collector_enhanced import EnhancedPubMedCollector
    
    collector = EnhancedPubMedCollector()
    
    conditions = ["constipation", "diarrhea"]
    
    # Collect papers
    results = collector.bulk_collect_conditions(
        conditions=conditions,
        max_results=15,
        include_fulltext=True
    )
    
    print(f"Collected data for {len(conditions)} conditions:")
    for result in results:
        print(f"  {result['condition']}: {result['paper_count']} papers ({result['status']})")
    
    return results


def example_analysis_only():
    """Example: Analyze existing papers in database."""
    print("\n=== Example: Analysis Only ===")
    
    from src.data.probiotic_analyzer_enhanced import EnhancedProbioticAnalyzer
    
    # Create analyzer with custom configuration
    custom_config = config.get_llm_config(
        model_name="llama3.1:8b",  # Use different model
        base_url="http://localhost:11434/v1"
    )
    
    analyzer = EnhancedProbioticAnalyzer(custom_config)
    
    # Process unprocessed papers
    results = analyzer.process_unprocessed_papers(
        limit=30,  # Process up to 30 papers
        batch_size=10
    )
    
    print("Analysis Results:")
    print(f"  Papers processed: {results['successful_papers']}/{results['total_papers']}")
    print(f"  Correlations found: {results['total_correlations']}")
    print(f"  Token usage: {results.get('token_usage', {}).get('total_tokens', 0):,}")
    
    return results


def example_database_exploration():
    """Example: Explore database contents."""
    print("\n=== Example: Database Exploration ===")
    
    from src.data.database_manager_enhanced import database_manager
    
    # Get comprehensive statistics
    stats = database_manager.get_database_stats()
    
    print("Database Statistics:")
    print(f"  Total Papers: {stats.get('total_papers', 0):,}")
    print(f"  Total Correlations: {stats.get('total_correlations', 0):,}")
    print(f"  Papers with Fulltext: {stats.get('papers_with_fulltext', 0):,}")
    
    print("\nProcessing Status:")
    for status, count in stats.get('processing_status', {}).items():
        print(f"  {status}: {count}")
    
    print("\nValidation Status:")
    for status, count in stats.get('validation_status', {}).items():
        print(f"  {status}: {count}")
    
    print("\nTop Extraction Models:")
    for model_info in stats.get('top_extraction_models', [])[:3]:
        print(f"  {model_info['model']}: {model_info['correlations']} correlations")
    
    return stats


def example_configuration_usage():
    """Example: Working with centralized configuration."""
    print("\n=== Example: Configuration Usage ===")
    
    # Validate current configuration
    validation = config.validate()
    
    print("Configuration Status:")
    print(f"  Valid: {validation['valid']}")
    
    if validation['issues']:
        print("  Issues:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    
    if validation['warnings']:
        print("  Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")
    
    # Show key configuration values
    print("\nKey Configuration:")
    print(f"  Database Path: {config.database.path}")
    print(f"  LLM Model: {config.llm.model_name}")
    print(f"  LLM Base URL: {config.llm.base_url}")
    print(f"  Data Directory: {config.paths.data_root}")
    print(f"  Max DB Connections: {config.database.max_connections}")
    
    # Create custom LLM config
    custom_llm = config.get_llm_config(
        model_name="custom-model:latest",
        base_url="http://custom-server:8080/v1"
    )
    
    print(f"\nCustom LLM Config Example:")
    print(f"  Model: {custom_llm.model_name}")
    print(f"  Base URL: {custom_llm.base_url}")
    
    return validation


def main():
    """Main example function that demonstrates different use cases."""
    print("MyBiome Enhanced Architecture Examples")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(__name__, 'examples.log')
    logger.info("Starting examples")
    
    try:
        # Skip migration for basic testing - just ensure database is ready
        print("Skipping migration - testing core functionality...")
        from src.data.database_manager_enhanced import database_manager
        print(f"Database ready: {database_manager.db_path}")
        migration_success = True
        
        # Example 1: Configuration
        example_configuration_usage()
        
        # Example 2: Database exploration  
        example_database_exploration()
        
        # Example 3: Data collection only
        example_data_collection_only()
        
        # Example 4: Analysis only
        example_analysis_only()
        
        # Example 5: Complete pipeline (comment out for faster testing)
        # example_basic_research()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print("\nNext steps:")
        print("1. Uncomment example_basic_research() for full pipeline demo")
        print("2. Modify conditions and parameters for your research")
        print("3. Use the enhanced modules in your own scripts")
        print("4. Check logs in data/logs/ for detailed information")
        
        return True
        
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        print(f"Examples failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)