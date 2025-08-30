"""
Test script for the enhanced full text retrieval pipeline.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.database_manager import DatabaseManager
from src.data.paper_parser import PubmedParser
from src.data.fulltext_retriever import FullTextRetriever

def test_database_schema():
    """Test that the database schema has been updated correctly."""
    print("=== Testing Database Schema ===")
    
    db_manager = DatabaseManager()
    
    # Check that new columns exist
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(papers)")
        columns = [row[1] for row in cursor.fetchall()]
        
        required_columns = ['pmc_id', 'has_fulltext', 'fulltext_source', 'fulltext_path']
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            print(f"[FAIL] Missing columns: {missing_columns}")
            return False
        else:
            print("[PASS] All required columns present")
            return True

def test_pmc_extraction():
    """Test PMC ID extraction from existing XML files."""
    print("\n=== Testing PMC ID Extraction ===")
    
    # Get an existing XML file to test with
    metadata_dir = project_root / "data" / "raw" / "metadata"
    xml_files = list(metadata_dir.glob("pubmed_batch_*.xml"))
    
    if not xml_files:
        print("[FAIL] No XML files found to test with")
        return False
    
    # Test with the first XML file
    test_file = xml_files[0]
    print(f"Testing with file: {test_file.name}")
    
    parser = PubmedParser()
    papers = parser.parse_metadata_file(str(test_file))
    
    if not papers:
        print("[FAIL] No papers parsed from XML file")
        return False
    
    # Check if any papers have PMC IDs
    papers_with_pmc = [p for p in papers if p.get('pmc_id')]
    total_papers = len(papers)
    pmc_count = len(papers_with_pmc)
    
    print(f"[PASS] Parsed {total_papers} papers")
    print(f"[PASS] Found {pmc_count} papers with PMC IDs ({pmc_count/total_papers*100:.1f}%)")
    
    if papers_with_pmc:
        sample_paper = papers_with_pmc[0]
        print(f"Sample PMC ID: {sample_paper['pmc_id']}")
        print(f"Sample paper title: {sample_paper['title'][:80]}...")
    
    return True

def test_database_queries():
    """Test the new database query methods."""
    print("\n=== Testing Database Queries ===")
    
    db_manager = DatabaseManager()
    
    try:
        # Test getting papers with PMC IDs
        pmc_papers = db_manager.get_papers_with_pmc_ids(limit=5)
        print(f"✅ Found {len(pmc_papers)} papers with PMC IDs")
        
        # Test getting papers with DOIs
        doi_papers = db_manager.get_papers_with_doi_no_fulltext(limit=5)
        print(f"✅ Found {len(doi_papers)} papers with DOIs but no fulltext")
        
        return True
    except Exception as e:
        print(f"❌ Database query error: {e}")
        return False

def test_fulltext_retriever_init():
    """Test that FullTextRetriever can be initialized."""
    print("\n=== Testing FullTextRetriever Initialization ===")
    
    try:
        retriever = FullTextRetriever()
        retriever.create_directories()
        
        # Check that directories were created
        if retriever.pmc_dir.exists() and retriever.pdf_dir.exists():
            print("✅ FullTextRetriever initialized and directories created")
            return True
        else:
            print("❌ Directories not created properly")
            return False
    except Exception as e:
        print(f"❌ FullTextRetriever initialization error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Enhanced PubMed Pipeline with Full Text Retrieval")
    print("="*60)
    
    tests = [
        test_database_schema,
        test_pmc_extraction,
        test_database_queries,
        test_fulltext_retriever_init
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced pipeline is ready.")
        print("\nNext steps:")
        print("1. Run 'python src/data/retrieve_fulltext.py' to process existing papers")
        print("2. Use the enhanced data collector for new paper collections")
        print("3. Check the 'data/raw/fulltext/' directory for downloaded full texts")
    else:
        print("❌ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()