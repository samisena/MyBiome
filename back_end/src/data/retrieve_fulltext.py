"""
Standalone script to retrieve full text for existing papers in the database.
This script can be run separately to process papers that were previously collected
without full text retrieval.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

from src.data.database_manager import DatabaseManager
from src.data.fulltext_retriever import FullTextRetriever

def main():
    """Main function to process existing papers for full text retrieval."""
    
    print("=== PubMed Full Text Retrieval ===")
    
    # Initialize components
    db_manager = DatabaseManager()
    fulltext_retriever = FullTextRetriever(email=os.getenv("EMAIL", "your_email@example.com"))
    
    # Create directories
    fulltext_retriever.create_directories()
    
    # Get papers that might have full text available
    print("\nChecking for papers with PMC IDs...")
    pmc_papers = db_manager.get_papers_with_pmc_ids(limit=20)  # Start with 20 papers
    print(f"Found {len(pmc_papers)} papers with PMC IDs")
    
    print("\nChecking for papers with DOIs (no full text yet)...")
    doi_papers = db_manager.get_papers_with_doi_no_fulltext(limit=30)  # Start with 30 papers
    print(f"Found {len(doi_papers)} papers with DOIs but no full text")
    
    # Combine and process
    all_papers = pmc_papers + doi_papers
    
    if not all_papers:
        print("No papers found that might have full text available.")
        return
    
    print(f"\nProcessing {len(all_papers)} papers for full text retrieval...")
    print("This may take a while due to API rate limiting...")
    
    # Process papers in batches
    batch_size = 25
    total_stats = {
        'total_papers': 0,
        'successful_pmc': 0,
        'successful_unpaywall': 0,
        'failed': 0,
        'errors': []
    }
    
    for i in range(0, len(all_papers), batch_size):
        batch = all_papers[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}: papers {i+1}-{min(i+batch_size, len(all_papers))}")
        
        batch_stats = fulltext_retriever.process_papers_batch(batch)
        
        # Accumulate stats
        total_stats['total_papers'] += batch_stats['total_papers']
        total_stats['successful_pmc'] += batch_stats['successful_pmc']
        total_stats['successful_unpaywall'] += batch_stats['successful_unpaywall']
        total_stats['failed'] += batch_stats['failed']
        total_stats['errors'].extend(batch_stats['errors'])
        
        print(f"Batch results: PMC={batch_stats['successful_pmc']}, Unpaywall={batch_stats['successful_unpaywall']}, Failed={batch_stats['failed']}")
    
    # Print final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Total papers processed: {total_stats['total_papers']}")
    print(f"Successful PMC retrievals: {total_stats['successful_pmc']}")
    print(f"Successful Unpaywall retrievals: {total_stats['successful_unpaywall']}")
    print(f"Failed retrievals: {total_stats['failed']}")
    print(f"Success rate: {((total_stats['successful_pmc'] + total_stats['successful_unpaywall']) / total_stats['total_papers'] * 100):.1f}%")
    
    if total_stats['errors']:
        print(f"\nErrors encountered ({len(total_stats['errors'])}):")
        for error in total_stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(total_stats['errors']) > 10:
            print(f"  ... and {len(total_stats['errors']) - 10} more errors")
    
    # Show database stats
    print("\n" + "="*50)
    print("DATABASE STATISTICS")
    print("="*50)
    db_stats = db_manager.get_database_stats()
    print(f"Total papers in database: {db_stats['total_papers']}")
    
    # Count papers with full text
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers WHERE has_fulltext = TRUE")
        fulltext_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM papers WHERE fulltext_source = 'pmc'")
        pmc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM papers WHERE fulltext_source = 'unpaywall'")
        unpaywall_count = cursor.fetchone()[0]
    
    print(f"Papers with full text: {fulltext_count}")
    print(f"  - From PMC: {pmc_count}")
    print(f"  - From Unpaywall: {unpaywall_count}")
    
    if db_stats['total_papers'] > 0:
        fulltext_percentage = (fulltext_count / db_stats['total_papers']) * 100
        print(f"Full text coverage: {fulltext_percentage:.1f}%")

if __name__ == "__main__":
    main()