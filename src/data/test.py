# main.py
"""
Main script showing how to use the PubMed collection system with SQLite database.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.database_manager import DatabaseManager
from src.data.data_collector import PubMedCollector
from src.data.paper_parser import PubmedParser

# Test data
TEST_STRAINS = [
    "Lactobacillus acidophilus",
    "Bifidobacterium bifidum"
]

TEST_CONDITIONS = [
    "irritable bowel syndrome",
    "antibiotic-associated diarrhea"
]

def main():
    """Main execution function"""
    
    # Check for API key
    if not os.getenv("NCBI_API_KEY"):
        print("Error: No NCBI API key found. Make sure you have a .env file with NCBI_API_KEY defined.")
        sys.exit(1)
    
    # Create a single database manager instance to share
    print("Initializing database...")
    db_manager = DatabaseManager()
    
    # Option 1: Collect new papers from PubMed
    print("\n=== COLLECTING NEW PAPERS ===")
    collector = PubMedCollector(db_manager)
    
    print(f"Running collection for {len(TEST_STRAINS)} strains and {len(TEST_CONDITIONS)} conditions...")
    print(f"This will make {len(TEST_STRAINS) * len(TEST_CONDITIONS)} API calls")
    
    results = collector.run_collection_for_list(TEST_STRAINS, TEST_CONDITIONS, results_per_query=5)
    
    # Display collection results
    print("\n=== COLLECTION SUMMARY ===")
    for result in results:
        status = "✓" if result['status'] == 'success' else "✗"
        print(f"{status} {result['strain']} + {result['condition']}: {result['paper_count']} papers")
    
    # Option 2: Parse existing XML files (if you have any)
    print("\n=== PARSING EXISTING XML FILES ===")
    parser = PubmedParser(db_manager)
    parser.parse_all_metadata()
    
    # Show final database statistics
    print("\n=== FINAL DATABASE STATISTICS ===")
    stats = db_manager.get_database_stats()
    print(f"Total papers: {stats['total_papers']}")
    print(f"Total authors: {stats['total_authors']}")
    print(f"Total searches: {stats['total_searches']}")
    
    # Example queries
    print("\n=== EXAMPLE QUERIES ===")
    
    # Example 1: Find papers by strain
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT p.title, p.journal, p.publication_date
            FROM papers p
            JOIN search_results sr ON p.pmid = sr.paper_pmid
            JOIN search_history sh ON sr.search_id = sh.search_id
            WHERE sh.strain = ?
            ORDER BY p.publication_date DESC
            LIMIT 5
        ''', ("Lactobacillus acidophilus",))
        
        print("\nRecent papers on Lactobacillus acidophilus:")
        for row in cursor.fetchall():
            print(f"- {row['title'][:80]}...")
            print(f"  {row['journal']}, {row['publication_date']}")
    
    # Example 2: Find most studied conditions
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT condition, SUM(result_count) as total_papers
            FROM search_history
            GROUP BY condition
            ORDER BY total_papers DESC
        ''')
        
        print("\nPapers per condition:")
        for row in cursor.fetchall():
            print(f"- {row['condition']}: {row['total_papers']} papers")
    
    print("\n✅ Script completed successfully!")
    print(f"Database saved at: {db_manager.db_path}")

if __name__ == "__main__":
    main()