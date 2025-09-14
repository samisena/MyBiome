#!/usr/bin/env python3
"""
Direct test of Semantic Scholar enrichment to verify functionality.
"""

import sys
import sqlite3
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

back_end_dir = Path(__file__).parent
sys.path.insert(0, str(back_end_dir))

from src.data.api_clients import get_semantic_scholar_client
from src.data.config import setup_logging

def test_s2_direct():
    """Test S2 integration directly."""
    print("Direct S2 Integration Test")
    print("-" * 30)

    # Direct database connection
    db_path = "data/processed/intervention_research.db"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get test papers
        cursor.execute("SELECT pmid, title, doi FROM papers LIMIT 2")
        papers = cursor.fetchall()

        print(f"Test papers: {len(papers)}")
        for pmid, title, doi in papers:
            print(f"  {pmid}: {title[:40]}...")

        if not papers:
            print("No papers found")
            return False

        # Test S2 API
        s2_client = get_semantic_scholar_client()

        # Use the real PMID that we know exists
        test_pmid = '25646566'
        print(f"\nTesting S2 API with PMID: {test_pmid}")

        try:
            results = s2_client.get_papers_batch([test_pmid])
            print(f"S2 API returned {len(results)} results")

            if results and results[0]:
                s2_paper = results[0]
                print(f"Found paper: {s2_paper.get('title', 'No title')}")
                print(f"Citations: {s2_paper.get('citationCount', 0)}")
                print(f"Influence: {s2_paper.get('influentialCitationCount', 0)}")

                # Direct database update
                cursor.execute('''
                    UPDATE papers SET
                        s2_paper_id = ?,
                        citation_count = ?,
                        influence_score = ?,
                        tldr = ?,
                        s2_processed = 1
                    WHERE pmid = ?
                ''', (
                    s2_paper.get('paperId'),
                    s2_paper.get('citationCount', 0),
                    s2_paper.get('influentialCitationCount', 0),
                    s2_paper.get('tldr', {}).get('text') if s2_paper.get('tldr') else None,
                    test_pmid
                ))
                conn.commit()

                print("Successfully updated paper with S2 data")

                # Verify update
                cursor.execute("SELECT citation_count, influence_score FROM papers WHERE pmid = ?", (test_pmid,))
                result = cursor.fetchone()
                if result:
                    print(f"Verified: Citations={result[0]}, Influence={result[1]}")

                return True
            else:
                print("No data returned from S2 API")
                return False

        except Exception as e:
            print(f"S2 API error: {e}")
            return False

def main():
    """Run the test."""
    logger = setup_logging(__name__)

    success = test_s2_direct()

    if success:
        print("\nDirect S2 integration test: PASSED")
        return True
    else:
        print("\nDirect S2 integration test: FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)