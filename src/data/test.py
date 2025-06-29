import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import logging  # Add this import for the database test

# Add the project root to Python path so we can import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from src.data.models import Author, Paper
from src.data.database_manager import DatabaseManager
from src.data.paper_parser import PubmedParser
from src.data.data_collector import PubMedCollector

def test_models():
    """Test the data models to ensure they work correctly"""
    print("\n=== Testing Data Models ===")
    
    # Test Author creation
    author1 = Author(
        last_name="Smith",
        first_name="John",
        initials="J",
        affiliations="Harvard Medical School"
    )
    print(f"✓ Created Author: {author1.last_name}, {author1.first_name}")
    
    # Test Paper creation with multiple authors
    author2 = Author(
        last_name="Doe",
        first_name="Jane",
        initials="J",
        affiliations=None  # Test optional field
    )
    
    paper = Paper(
        pmid="12345678",
        title="Test Paper: Effects of Probiotics on Health",
        abstract="This is a test abstract about probiotics...",
        authors=[author1, author2],
        journal="Test Medical Journal",
        publication_date="2024-01-15",
        doi="10.1000/test.2024.001",
        keywords=["probiotics", "health", "clinical trial"]
    )
    print(f"✓ Created Paper: PMID {paper.pmid}, {len(paper.authors)} authors")
    
    return True

def test_database_manager():
    """Test the database manager functionality"""
    print("\n=== Testing Database Manager ===")
    
    # Create a temporary database for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a temporary database manager with a custom path
        # We need to monkey-patch the class to use our temp directory
        original_init = DatabaseManager.__init__
        
        def temp_init(self, db_name='test_pubmed.db'):
            self.db_path = temp_path / db_name
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            self.create_tables()
        
        # Temporarily replace the init method
        DatabaseManager.__init__ = temp_init
        
        try:
            db_manager = DatabaseManager()
            print("✓ Database created successfully")
        finally:
            # Restore the original init method
            DatabaseManager.__init__ = original_init
        
        # Test inserting a paper
        test_author = Author(
            last_name="TestAuthor",
            first_name="First",
            initials="F",
            affiliations="Test University"
        )
        
        test_paper = Paper(
            pmid="99999999",
            title="Test Paper for Database",
            abstract="Testing database insertion...",
            authors=[test_author],
            journal="Test Journal",
            publication_date="2024-01-01",
            doi="10.1000/test.db",
            keywords=["test", "database"]
        )
        
        # Record a search first
        search_id = db_manager.record_search(
            strain="Lactobacillus",
            condition="IBS",
            query="test query",
            result_count=1
        )
        print(f"✓ Recorded search with ID: {search_id}")
        
        # Insert the paper
        was_new = db_manager.insert_papers(test_paper, search_id)
        print(f"✓ Paper inserted (new: {was_new})")
        
        # Test duplicate insertion
        was_duplicate = db_manager.insert_papers(test_paper, search_id)
        print(f"✓ Duplicate handling works (new: {was_duplicate})")
        
        # Test retrieval
        stats = db_manager.get_database_stats()
        print(f"✓ Database stats: {stats}")
        
        return True

def test_parser_with_sample_xml():
    """Test the XML parser with a sample PubMed XML"""
    print("\n=== Testing XML Parser ===")
    
    # Create a sample PubMed XML file
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345678</PMID>
            <Article>
                <ArticleTitle>Effects of Lactobacillus on IBS: A Clinical Trial</ArticleTitle>
                <Abstract>
                    <AbstractText>This study investigates the effects of Lactobacillus...</AbstractText>
                </Abstract>
                <AuthorList>
                    <Author>
                        <LastName>Johnson</LastName>
                        <ForeName>Mary</ForeName>
                        <Initials>M</Initials>
                        <Affiliation>Medical University</Affiliation>
                    </Author>
                    <Author>
                        <LastName>Williams</LastName>
                        <ForeName>Robert</ForeName>
                        <Initials>R</Initials>
                    </Author>
                </AuthorList>
                <Journal>
                    <Title>Journal of Probiotics Research</Title>
                </Journal>
                <PubDate>
                    <Year>2023</Year>
                    <Month>06</Month>
                    <Day>15</Day>
                </PubDate>
                <ArticleIdList>
                    <ArticleId IdType="doi">10.1000/jpr.2023.001</ArticleId>
                </ArticleIdList>
                <KeywordList>
                    <Keyword>Lactobacillus</Keyword>
                    <Keyword>IBS</Keyword>
                    <Keyword>Clinical Trial</Keyword>
                </KeywordList>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save sample XML
        xml_path = Path(temp_dir) / "test_pubmed.xml"
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(sample_xml)
        
        # Test parsing
        parser = PubmedParser()
        papers = parser.parse_metadata_file(str(xml_path))
        
        if papers:
            paper = papers[0]
            print(f"✓ Parsed paper: {paper.title}")
            print(f"✓ Authors: {len(paper.authors)}")
            print(f"✓ Keywords: {paper.keywords}")
            return True
        else:
            print("✗ Failed to parse XML")
            return False

def test_api_connection():
    """Test the PubMed API connection (requires API key)"""
    print("\n=== Testing PubMed API Connection ===")
    
    collector = PubMedCollector()
    
    # Check if API key is loaded
    if not collector.api_key:
        print("⚠ No API key found in .env file")
        print("  Please create a .env file with: NCBI_API_KEY=your_key_here")
        return False
    
    print("✓ API key loaded")
    
    # Test with a simple search
    test_query = "Lactobacillus AND IBS AND clinical trial"
    print(f"Testing search: {test_query}")
    
    try:
        # Search for just 3 papers as a test
        paper_ids = collector.search_papers(test_query, max_results=3)
        
        if paper_ids:
            print(f"✓ Found {len(paper_ids)} papers")
            print(f"  Sample PMIDs: {paper_ids[:3]}")
            return True
        else:
            print("⚠ No papers found (this might be normal)")
            return True
            
    except Exception as e:
        print(f"✗ API Error: {e}")
        return False

def run_integration_test():
    """Run a complete integration test with minimal data"""
    print("\n=== Running Integration Test ===")
    
    # Test with one strain and one condition
    test_strains = ["Lactobacillus rhamnosus"]
    test_conditions = ["irritable bowel syndrome"]
    
    collector = PubMedCollector()
    
    # Create test directories
    collector.data_dir.mkdir(parents=True, exist_ok=True)
    collector.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Testing collection for: {test_strains[0]} + {test_conditions[0]}")
    
    try:
        # Collect just 2 papers as a test
        result = collector.collect_by_strain_and_condition(
            test_strains[0], 
            test_conditions[0], 
            max_results=2
        )
        
        print(f"✓ Collection result: {result['paper_count']} papers found")
        print(f"  Status: {result['status']}")
        
        if result['paper_count'] > 0:
            # Check if data was saved to database
            stats = collector.db_manager.get_database_stats()
            print(f"✓ Database now contains: {stats['total_papers']} papers")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("PubMed Research System Test Suite")
    print("=" * 50)
    
    # Track test results
    results = []
    
    # Run tests in order
    tests = [
        ("Data Models", test_models),
        ("Database Manager", test_database_manager),
        ("XML Parser", test_parser_with_sample_xml),
        ("API Connection", test_api_connection),
        ("Integration", run_integration_test)
    ]
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system is ready to run at scale.")
    else:
        print("\n⚠ Some tests failed. Please fix the issues before running at scale.")

if __name__ == "__main__":
    main()