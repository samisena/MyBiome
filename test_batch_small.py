#!/usr/bin/env python3
"""Test batch collection with just 2 conditions."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.orchestration.rotation_paper_collector import RotationPaperCollector
from back_end.src.data_collection.database_manager import database_manager

# Clear database
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM papers")
    cursor.execute("DELETE FROM interventions")
    conn.commit()
    print("Database cleared")

collector = RotationPaperCollector()

# Override to test with just 2 conditions
original_get_all = collector.get_all_conditions
collector.get_all_conditions = lambda: ["diabetes", "hypertension"]

print("Testing batch collection with 2 conditions...")
result = collector.collect_all_conditions_batch(
    papers_per_condition=2,
    min_year=2020
)

print(f"\nResult:")
print(f"  Success: {result.success}")
print(f"  Papers collected: {result.total_papers_collected}")

# Check database
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    print(f"\nPapers in database: {count}")

    cursor.execute("SELECT pmid, title FROM papers LIMIT 5")
    papers = cursor.fetchall()
    if papers:
        print("\nSample papers:")
        for pmid, title in papers:
            print(f"  {pmid}: {title[:50]}...")