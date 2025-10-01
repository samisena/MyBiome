#!/usr/bin/env python3
"""Debug parallel collection issue."""

import sys
import logging
import concurrent.futures
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Enable SQL logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('database_manager').setLevel(logging.DEBUG)

from back_end.src.data_collection.pubmed_collector import PubMedCollector
from back_end.src.data_collection.database_manager import database_manager

# Clear database
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM papers")
    conn.commit()
    print("Database cleared\n")

def collect_condition(condition):
    """Collect papers for a condition."""
    print(f"[{condition}] Starting collection...")
    collector = PubMedCollector()  # Create new instance per thread

    result = collector.collect_interventions_by_condition(
        condition=condition,
        min_year=2020,
        max_results=1,
        include_fulltext=False,
        use_interleaved_s2=False
    )

    print(f"[{condition}] Result: {result.get('paper_count')} papers")

    # Check what's in database right after collection
    with database_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers")
        count = cursor.fetchone()[0]
        print(f"[{condition}] Database has {count} papers after my collection")

    return result

# Test parallel collection
conditions = ["diabetes", "hypertension"]

print("Testing parallel collection:\n")
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(collect_condition, c) for c in conditions]
    results = [f.result() for f in futures]

print("\n" + "="*50)
print("Final check:")
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    print(f"Total papers in database: {count}")

    cursor.execute("SELECT pmid, title FROM papers")
    papers = cursor.fetchall()
    for pmid, title in papers:
        print(f"  {pmid}: {title[:50]}...")