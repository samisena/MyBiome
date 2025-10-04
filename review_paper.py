#!/usr/bin/env python3
"""Retrieve and display full paper details for manual review."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from back_end.src.data_collection.database_manager import DatabaseManager

def review_paper(pmid):
    """Display full paper details."""
    db = DatabaseManager()

    paper = db.get_paper_by_pmid(pmid)

    if not paper:
        print(f"Paper with PMID {pmid} not found")
        return

    # Clean text for display (remove problematic unicode)
    def clean_text(text):
        if not text:
            return 'N/A'
        return text.encode('ascii', 'ignore').decode('ascii')

    print("="*80)
    print(f"PAPER REVIEW - PMID: {pmid}")
    print("="*80)
    print(f"\nTITLE:\n{clean_text(paper.get('title', 'N/A'))}")
    print(f"\nJOURNAL: {clean_text(paper.get('journal', 'N/A'))}")
    print(f"DATE: {clean_text(paper.get('publication_date', 'N/A'))}")
    print(f"\nABSTRACT:\n{clean_text(paper.get('abstract', 'N/A'))}")
    print(f"\nKEYWORDS: {clean_text(paper.get('keywords', 'N/A'))}")
    print(f"\nCONDITION SEARCHED: {clean_text(paper.get('condition_searched', 'N/A'))}")

    # Show fulltext info if available
    fulltext = paper.get('fulltext', '')
    if fulltext:
        print(f"\nFULLTEXT LENGTH: {len(fulltext)} characters")
        print(f"FULLTEXT PREVIEW (first 500 chars):\n{fulltext[:500]}...")
    else:
        print("\nFULLTEXT: Not available")

    print("\n" + "="*80)

if __name__ == "__main__":
    pmid = sys.argv[1] if len(sys.argv) > 1 else "41044716"
    review_paper(pmid)
