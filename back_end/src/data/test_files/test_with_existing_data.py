"""
Test with the actual existing database and data.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.database_manager import DatabaseManager

def main():
    """Test with existing database."""
    print("Testing with existing database...")
    
    # Use the actual database path
    actual_db_path = project_root / "data" / "processed" / "pubmed_research.db" 
    
    if actual_db_path.exists():
        print(f"Found existing database: {actual_db_path}")
        
        # Create database manager with the right path
        db_manager = DatabaseManager(db_name='pubmed_research.db', project_root=project_root)
        
        # Test basic stats
        stats = db_manager.get_database_stats()
        print(f"Total papers in database: {stats['total_papers']}")
        
        if stats['total_papers'] > 0:
            # Test new queries
            pmc_papers = db_manager.get_papers_with_pmc_ids(limit=10)
            doi_papers = db_manager.get_papers_with_doi_no_fulltext(limit=10)
            
            print(f"Papers with PMC IDs: {len(pmc_papers)}")
            print(f"Papers with DOIs (no fulltext): {len(doi_papers)}")
            
            # Show sample data
            if pmc_papers:
                sample = pmc_papers[0]
                print(f"Sample PMC paper: {sample.get('title', 'No title')[:50]}...")
                print(f"PMC ID: {sample.get('pmc_id')}")
            
            if doi_papers:
                sample = doi_papers[0]
                print(f"Sample DOI paper: {sample.get('title', 'No title')[:50]}...")
                print(f"DOI: {sample.get('doi')}")
        else:
            print("No papers in database to test with.")
            
    else:
        print(f"Database not found at: {actual_db_path}")

if __name__ == "__main__":
    main()