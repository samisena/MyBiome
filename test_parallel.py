#!/usr/bin/env python3
"""Test parallel collection."""

import sys
import concurrent.futures
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.orchestration.rotation_paper_collector import RotationPaperCollector
from back_end.src.data_collection.database_manager import database_manager

# Clear database
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM papers")
    conn.commit()
    print("Database cleared")

collector = RotationPaperCollector()
conditions = ["diabetes", "hypertension"]

print(f"Testing parallel collection for {conditions}")

# Test with parallel execution
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = []
    for condition in conditions:
        future = executor.submit(
            collector._collect_single_condition_without_s2,
            condition, 1, 2020, None
        )
        futures.append((condition, future))

    for condition, future in futures:
        result = future.result()
        print(f"{condition}: collected={result['papers_collected']}")

# Check database
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    print(f"\nPapers in database: {count}")

    cursor.execute("SELECT pmid, title FROM papers")
    papers = cursor.fetchall()
    for pmid, title in papers:
        print(f"  {pmid}: {title[:50]}...")