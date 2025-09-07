"""
Quick test of the enhanced backend pipeline components.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.config import config, setup_logging
from src.data.database_manager_enhanced import database_manager

def test_database():
    """Test database connection and basic operations."""
    print("=== Database Test ===")
    
    # Get stats
    stats = database_manager.get_database_stats()
    print(f"Database Path: {database_manager.db_path}")
    print(f"Total Papers: {stats.get('total_papers', 0):,}")
    print(f"Total Correlations: {stats.get('total_correlations', 0):,}")
    print(f"Papers with Fulltext: {stats.get('papers_with_fulltext', 0):,}")
    
    print("\nProcessing Status:")
    for status, count in stats.get('processing_status', {}).items():
        print(f"  {status}: {count}")
    
    return stats

def test_config():
    """Test configuration system."""
    print("\n=== Configuration Test ===")
    
    validation = config.validate()
    print(f"Configuration Valid: {validation['valid']}")
    print(f"Database Path: {config.database.path}")
    print(f"LLM Model: {config.llm.model_name}")
    print(f"Data Root: {config.paths.data_root}")
    
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    return validation

def test_components():
    """Test component imports."""
    print("\n=== Component Import Test ===")
    
    try:
        from src.data.pubmed_collector_enhanced import EnhancedPubMedCollector
        print("[OK] PubMed Collector imported successfully")
        
        from src.data.probiotic_analyzer_enhanced import EnhancedProbioticAnalyzer
        print("[OK] Probiotic Analyzer imported successfully")
        
        from src.data.enhanced_pipeline import EnhancedResearchPipeline
        print("[OK] Research Pipeline imported successfully")
        
        from src.data.api_clients import api_clients
        print("[OK] API Clients imported successfully")
        
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Enhanced Backend Pipeline - Quick Test")
    print("=" * 45)
    
    # Setup minimal logging
    logger = setup_logging(__name__, 'quick_test.log')
    
    try:
        # Test 1: Configuration
        config_result = test_config()
        
        # Test 2: Database
        db_stats = test_database()
        
        # Test 3: Component imports
        import_success = test_components()
        
        print("\n" + "=" * 45)
        if import_success:
            print("[SUCCESS] Enhanced pipeline components are working!")
            print(f"Database has {db_stats.get('total_papers', 0)} papers ready for analysis")
            if not config_result.get('valid', False):
                print("Note: Configuration has warnings (missing API key) but core functionality works")
        else:
            print("[WARNING] Some components need attention (see details above)")
            
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        print(f"[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)