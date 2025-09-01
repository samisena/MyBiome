"""
Simple test script for the enhanced full text retrieval pipeline.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.database_manager import DatabaseManager
from src.data.paper_parser import PubmedParser
from src.data.fulltext_retriever import FullTextRetriever

def main():
    """Run basic tests."""
    print("Testing Enhanced PubMed Pipeline with Full Text Retrieval")
    print("="*60)
    
    # Test 1: Database schema
    print("\n1. Testing Database Schema...")
    db_manager = DatabaseManager()
    
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(papers)")
        columns = [row[1] for row in cursor.fetchall()]
        
        required_columns = ['pmc_id', 'has_fulltext', 'fulltext_source', 'fulltext_path']
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            print(f"FAIL: Missing columns: {missing_columns}")
        else:
            print("PASS: All required columns present")
    
    # Test 2: PMC extraction
    print("\n2. Testing PMC ID Extraction...")
    metadata_dir = project_root / "data" / "raw" / "metadata"
    xml_files = list(metadata_dir.glob("pubmed_batch_*.xml"))
    
    if not xml_files:
        print("FAIL: No XML files found to test with")
    else:
        test_file = xml_files[0]
        print(f"Testing with file: {test_file.name}")
        
        parser = PubmedParser()
        papers = parser.parse_metadata_file(str(test_file))
        
        if not papers:
            print("FAIL: No papers parsed from XML file")
        else:
            papers_with_pmc = [p for p in papers if p.get('pmc_id')]
            total_papers = len(papers)
            pmc_count = len(papers_with_pmc)
            
            print(f"PASS: Parsed {total_papers} papers")
            print(f"PASS: Found {pmc_count} papers with PMC IDs ({pmc_count/total_papers*100:.1f}%)")
            
            if papers_with_pmc:
                sample_paper = papers_with_pmc[0]
                print(f"Sample PMC ID: {sample_paper['pmc_id']}")
                print(f"Sample paper title: {sample_paper['title'][:80]}...")
    
    # Test 3: Database queries
    print("\n3. Testing Database Queries...")
    try:
        pmc_papers = db_manager.get_papers_with_pmc_ids(limit=5)
        doi_papers = db_manager.get_papers_with_doi_no_fulltext(limit=5)
        
        print(f"PASS: Found {len(pmc_papers)} papers with PMC IDs")
        print(f"PASS: Found {len(doi_papers)} papers with DOIs but no fulltext")
    except Exception as e:
        print(f"FAIL: Database query error: {e}")
    
    # Test 4: FullTextRetriever
    print("\n4. Testing FullTextRetriever...")
    try:
        retriever = FullTextRetriever()
        retriever.create_directories()
        
        if retriever.pmc_dir.exists() and retriever.pdf_dir.exists():
            print("PASS: FullTextRetriever initialized and directories created")
        else:
            print("FAIL: Directories not created properly")
    except Exception as e:
        print(f"FAIL: FullTextRetriever initialization error: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("Next steps:")
    print("1. Run 'python back_end/src/data/retrieve_fulltext.py' to process existing papers")
    print("2. Use the enhanced data collector for new paper collections")
    print("3. Check the 'data/raw/fulltext/' directory for downloaded full texts")

if __name__ == "__main__":
    main()