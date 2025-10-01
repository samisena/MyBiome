#!/usr/bin/env python3
"""Debug the collection process."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.data_collection.pubmed_collector import PubMedCollector
from back_end.src.data_collection.database_manager import database_manager

# Initialize collector
collector = PubMedCollector()

# Test collection for a single condition
condition = "diabetes"
print(f"Testing collection for condition: {condition}")

# Check initial paper count
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    initial_count = cursor.fetchone()[0]
    print(f"Initial paper count: {initial_count}")

# Collect papers
result = collector.collect_interventions_by_condition(
    condition=condition,
    min_year=2020,
    max_year=None,
    max_results=5,
    include_fulltext=False,
    use_interleaved_s2=False  # Disable S2 to match batch pipeline
)

print(f"\nCollection result:")
print(f"  Status: {result.get('status')}")
print(f"  Papers collected: {result.get('paper_count', 0)}")
print(f"  New papers: {result.get('new_papers_count', 0)}")
if 'message' in result:
    print(f"  Message: {result.get('message')}")

# Check final paper count
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    final_count = cursor.fetchone()[0]
    print(f"\nFinal paper count: {final_count}")
    print(f"Papers added: {final_count - initial_count}")

    # Show recent papers
    cursor.execute("SELECT pmid, title FROM papers ORDER BY id DESC LIMIT 5")
    papers = cursor.fetchall()
    if papers:
        print("\nRecent papers:")
        for pmid, title in papers:
            print(f"  {pmid}: {title[:60]}...")