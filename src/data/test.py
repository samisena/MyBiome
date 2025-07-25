# Import your modules
from src.data.data_collector import PubMedCollector
from src.data.probiotic_analyzer import ProbioticAnalyzer

"""
Test script for PubMed Collector with correct imports
"""

import os
import sys
from pathlib import Path
import time
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path includes: {str(project_root) in sys.path}")

def test_imports():
    """Test that we can import the modules"""
    print("\n" + "=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        # Note: Your file is named data_collector.py not pubmed_collector.py
        from src.data.data_collector import PubMedCollector
        print("✓ Successfully imported PubMedCollector")
        
        # Also check other required imports
        from src.data.database_manager import DatabaseManager
        print("✓ Successfully imported DatabaseManager")
        
        from src.data.paper_parser import PubmedParser
        print("✓ Successfully imported PubmedParser")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        
        # Check if files exist
        print("\nChecking file structure:")
        files_to_check = [
            "src/data/data_collector.py",
            "src/data/database_manager.py", 
            "src/data/paper_parser.py",
            "src/data/__init__.py"
        ]
        
        for file_path in files_to_check:
            full_path = project_root / file_path
            if full_path.exists():
                print(f"  ✓ {file_path} exists")
            else:
                print(f"  ✗ {file_path} NOT FOUND")
                
        return False

def test_environment():
    """Check environment setup"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    
    # Load .env
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    
    if not env_path.exists():
        print(f"✗ .env file not found at: {env_path}")
        return False
    
    load_dotenv(env_path)
    api_key = os.getenv("NCBI_API_KEY")
    
    if not api_key:
        print("✗ NCBI_API_KEY not found in .env file")
        return False
    
    print("✓ .env file loaded")
    print(f"✓ NCBI_API_KEY found (length: {len(api_key)})")
    
    # Check directories
    directories_to_create = [
        project_root / "data" / "raw" / "papers",
        project_root / "data" / "raw" / "metadata",
        project_root / "data" / "processed"
    ]
    
    for dir_path in directories_to_create:
        if not dir_path.exists():
            print(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✓ Directory exists: {dir_path.relative_to(project_root)}")
    
    return True

def test_collector():
    """Test the PubMed collector"""
    print("\n" + "=" * 60)
    print("TESTING PUBMED COLLECTOR")
    print("=" * 60)
    
    try:
        from src.data.data_collector import PubMedCollector
        
        # Initialize collector
        collector = PubMedCollector()
        print("✓ PubMedCollector initialized")
        
        # Test with a simple query
        strain = "Lactobacillus rhamnosus GG"
        condition = "diarrhea"
        
        print(f"\nTesting collection for:")
        print(f"  Strain: {strain}")
        print(f"  Condition: {condition}")
        print(f"  Max papers: 2")
        
        # Run collection
        result = collector.collect_by_strain_and_condition(
            strain=strain,
            condition=condition,
            max_results=2
        )
        
        print(f"\nResults:")
        print(f"  Status: {result['status']}")
        print(f"  Papers found: {result['paper_count']}")
        
        if result['status'] == 'success':
            print(f"  Search ID: {result.get('search_id', 'N/A')}")
            print(f"  Metadata file: {result.get('metadata_file', 'N/A')}")
            print("\n✓ Collection test PASSED!")
            return True
        else:
            print("\n✗ Collection test FAILED")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during collection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analyzer_import():
    """Test if we can import the analyzer"""
    print("\n" + "=" * 60)
    print("TESTING ANALYZER IMPORT")
    print("=" * 60)
    
    try:
        from src.data.probiotic_analyzer import ProbioticAnalyzer
        print("✓ Successfully imported ProbioticAnalyzer")
        
        # Try to initialize (might fail if database doesn't exist yet)
        try:
            analyzer = ProbioticAnalyzer(project_root)
            print("✓ ProbioticAnalyzer initialized")
        except Exception as e:
            print(f"⚠ Analyzer initialization failed (this might be OK if no data yet): {e}")
            
        return True
    except ImportError as e:
        print(f"✗ Failed to import analyzer: {e}")
        return False

def main():
    """Run all tests"""
    print("PUBMED PIPELINE TEST SUITE")
    print("=" * 60)
    
    # Test sequence
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("Collector Test", test_collector),
        ("Analyzer Import Test", test_analyzer_import)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Your pipeline is ready to use.")
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        
        if not results.get("Import Test"):
            print("\nMake sure you have these files in src/data/:")
            print("  - data_collector.py (not pubmed_collector.py)")
            print("  - database_manager.py")
            print("  - paper_parser.py")
            print("  - __init__.py")

if __name__ == "__main__":
    main()