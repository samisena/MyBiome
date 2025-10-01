#!/usr/bin/env python3
"""Test PubMed collector directly."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.data_collection.pubmed_collector import PubMedCollector
from back_end.src.data_collection.database_manager import database_manager

# Clear database
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM papers")
    conn.commit()
    print("Database cleared")

collector = PubMedCollector()

# Test collection directly
print("Testing PubMed collector directly...")
result = collector.collect_interventions_by_condition(
    condition="diabetes",
    min_year=2020,
    max_year=None,
    max_results=2,
    include_fulltext=False,
    use_interleaved_s2=False
)

print(f"Result: {result}")
print(f"  Status: {result.get('status')}")
print(f"  Papers collected: {result.get('paper_count')}")

# Check database
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    print(f"\nPapers in database: {count}")

    cursor.execute("SELECT pmid, title FROM papers LIMIT 5")
    papers = cursor.fetchall()
    if papers:
        print("Sample papers:")
        for pmid, title in papers:
            print(f"  {pmid}: {title[:50]}...")