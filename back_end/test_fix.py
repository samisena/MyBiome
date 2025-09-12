#!/usr/bin/env python3
"""
Test the fix for the 'needs_review' status error.
"""

import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.llm.consensus_analyzer import MultiLLMConsensusAnalyzer
from src.paper_collection.database_manager import database_manager

def test_status_update():
    """Test that 'needs_review' status update works."""
    
    print("Testing 'needs_review' status update...")
    
    try:
        # Test updating a paper to 'needs_review' status
        result = database_manager.update_paper_processing_status('test_pmid', 'needs_review')
        print("SUCCESS: 'needs_review' status is now valid")
        
        # Test all valid statuses
        valid_statuses = ['pending', 'processing', 'processed', 'failed', 'needs_review']
        for status in valid_statuses:
            try:
                database_manager.update_paper_processing_status('test_pmid', status)
                print(f"Status '{status}' works")
            except Exception as e:
                print(f"Status '{status}' failed: {e}")
        
        # Test invalid status
        try:
            database_manager.update_paper_processing_status('test_pmid', 'invalid_status')
            print("Invalid status should have failed but didn't")
        except ValueError as e:
            print(f"Invalid status correctly rejected: {e}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_consensus_analyzer():
    """Test that consensus analyzer can now work without the error."""
    
    print("\nTesting consensus analyzer with the fix...")
    
    try:
        # Get a test paper that previously failed
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT pmid, title, abstract, has_fulltext, fulltext_path
                FROM papers 
                WHERE pmid = "35951774"
                LIMIT 1
            ''')
            
            row = cursor.fetchone()
            if not row:
                print("Test paper 35951774 not found, using any paper...")
                cursor.execute('''
                    SELECT pmid, title, abstract, has_fulltext, fulltext_path
                    FROM papers 
                    WHERE abstract IS NOT NULL 
                    AND LENGTH(abstract) > 200
                    LIMIT 1
                ''')
                row = cursor.fetchone()
            
            if not row:
                print("No suitable test paper found")
                return False
            
            test_paper = {
                'pmid': row[0],
                'title': row[1],
                'abstract': row[2],
                'has_fulltext': bool(row[3]),
                'fulltext_path': row[4]
            }
        
        print(f"Testing with paper: {test_paper['pmid']}")
        
        # Initialize consensus analyzer
        analyzer = MultiLLMConsensusAnalyzer()
        
        # Process the paper that previously failed
        result = analyzer.process_paper_with_consensus(test_paper)
        
        print(f"SUCCESS: Paper processed without status error")
        print(f"  Status: {result.consensus_status}")
        print(f"  Needs review: {result.needs_review}")
        print(f"  Agreed correlations: {len(result.agreed_correlations)}")
        print(f"  Conflicts: {len(result.conflicting_correlations)}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def main():
    """Run the fix test."""
    
    print("=" * 60)
    print("TESTING FIX FOR 'needs_review' STATUS ERROR")
    print("=" * 60)
    
    # Test 1: Status update functionality
    status_test = test_status_update()
    
    # Test 2: Consensus analyzer functionality  
    analyzer_test = test_consensus_analyzer()
    
    print("\n" + "=" * 60)
    if status_test and analyzer_test:
        print("ALL TESTS PASSED - FIX SUCCESSFUL!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if (status_test and analyzer_test) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)