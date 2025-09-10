#!/usr/bin/env python3
"""
Script to clear the pubmed_research.db database and collect new papers.
"""

import sys
import sqlite3
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

# Also add the parent directory (back_end) so we can import src
back_end_dir = src_dir.parent
sys.path.insert(0, str(back_end_dir))

try:
    from src.paper_collection.database_manager import database_manager
    from src.paper_collection.pubmed_collector import EnhancedPubMedCollector
    from src.data.config import config, setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)

def clear_database():
    """Clear all entries from the database."""
    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear correlations first (due to foreign key constraints)
            cursor.execute("DELETE FROM correlations")
            correlations_deleted = cursor.rowcount
            
            # Clear papers
            cursor.execute("DELETE FROM papers")
            papers_deleted = cursor.rowcount
            
            # Reset auto-increment counters
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='correlations'")
            
            conn.commit()
            
            print(f"Database cleared:")
            print(f"  - {papers_deleted} papers deleted")
            print(f"  - {correlations_deleted} correlations deleted")
            
            return True
            
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False

def collect_papers(search_term: str, max_papers: int = 50):
    """Collect papers for a specific search term."""
    try:
        print(f"\nCollecting {max_papers} papers for search term: '{search_term}'")
        
        collector = EnhancedPubMedCollector(database_manager)
        
        # Use the collection method for probiotic studies
        results = collector.collect_probiotics_by_condition(
            condition=search_term,
            min_year=2000,
            max_results=max_papers,
            include_fulltext=True
        )
        
        if results.get('status') == 'success':
            print(f"Successfully collected {results.get('paper_count', 0)} papers")
            return True
        else:
            print(f"Collection failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"Error during collection: {e}")
        return False

def main():
    """Main function to clear database and collect new papers."""
    print("=== MyBiome Database Clear & Collection ===")
    
    # Setup logging
    logger = setup_logging(__name__, 'clear_and_collect.log')
    
    # Step 1: Clear database
    print("\nStep 1: Clearing database...")
    if not clear_database():
        print("Failed to clear database. Exiting.")
        return False
    
    # Step 2: Collect new papers
    print("\nStep 2: Collecting new papers...")
    if not collect_papers("IBS", 50):
        print("Failed to collect papers. Exiting.")
        return False
    
    # Step 3: Show final stats
    print("\nStep 3: Final database statistics...")
    stats = database_manager.get_database_stats()
    print(f"Total papers in database: {stats.get('total_papers', 0)}")
    print(f"Total correlations: {stats.get('total_correlations', 0)}")
    
    print("\n=== Process completed successfully! ===")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)