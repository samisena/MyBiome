"""
Test with the real existing database.
"""

import sys
import sqlite3
from pathlib import Path

# Add the project root to Python path  
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def main():
    """Test with the real database."""
    print("Testing with real existing database...")
    
    # Connect to the actual database
    actual_db_path = project_root / "data" / "processed" / "pubmed_research.db"
    
    if not actual_db_path.exists():
        print(f"Database not found at: {actual_db_path}")
        return
    
    conn = sqlite3.connect(str(actual_db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check if table exists and has data
    cursor.execute("SELECT COUNT(*) FROM papers")
    total_papers = cursor.fetchone()[0]
    print(f"Total papers in database: {total_papers}")
    
    if total_papers == 0:
        print("No papers in database to test with.")
        conn.close()
        return
    
    # Check existing schema
    cursor.execute("PRAGMA table_info(papers)")
    columns = [row[1] for row in cursor.fetchall()]
    print(f"Existing columns: {columns}")
    
    # Check if new columns exist
    new_columns = ['pmc_id', 'has_fulltext', 'fulltext_source', 'fulltext_path']
    missing_columns = [col for col in new_columns if col not in columns]
    
    if missing_columns:
        print(f"Missing columns (need to be added): {missing_columns}")
        
        # Add missing columns
        for col in missing_columns:
            if col == 'pmc_id':
                cursor.execute("ALTER TABLE papers ADD COLUMN pmc_id TEXT")
            elif col == 'has_fulltext':
                cursor.execute("ALTER TABLE papers ADD COLUMN has_fulltext BOOLEAN DEFAULT FALSE")
            elif col == 'fulltext_source':
                cursor.execute("ALTER TABLE papers ADD COLUMN fulltext_source TEXT")
            elif col == 'fulltext_path':
                cursor.execute("ALTER TABLE papers ADD COLUMN fulltext_path TEXT")
        
        conn.commit()
        print("Added missing columns.")
    else:
        print("All required columns already exist.")
    
    # Sample some papers
    cursor.execute("SELECT pmid, title, doi, pmc_id FROM papers LIMIT 5")
    papers = cursor.fetchall()
    
    print(f"\nSample papers:")
    for paper in papers:
        print(f"PMID: {paper['pmid']}")
        print(f"Title: {paper['title'][:60]}...")
        print(f"DOI: {paper['doi']}")
        print(f"PMC ID: {paper['pmc_id']}")
        print("-" * 40)
    
    # Count papers with DOI and PMC
    cursor.execute("SELECT COUNT(*) FROM papers WHERE doi IS NOT NULL AND doi != ''")
    doi_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM papers WHERE pmc_id IS NOT NULL AND pmc_id != ''")
    pmc_count = cursor.fetchone()[0]
    
    print(f"\nPapers with DOIs: {doi_count}")
    print(f"Papers with PMC IDs: {pmc_count}")
    
    conn.close()
    
    print("\nDatabase is ready for full text retrieval!")

if __name__ == "__main__":
    main()