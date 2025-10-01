#!/usr/bin/env python3
"""Test batch collection directly."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.orchestration.rotation_paper_collector import RotationPaperCollector
from back_end.src.data_collection.database_manager import database_manager
import sqlite3

# Clear database first
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM papers")
    cursor.execute("DELETE FROM interventions")
    conn.commit()
    print("Database cleared")

collector = RotationPaperCollector()

# Test batch collection with just 1 paper per condition
print("Testing batch collection for all 60 conditions with 1 paper each...")
print(f"Max workers: {collector.max_workers}")

result = collector.collect_all_conditions_batch(
    papers_per_condition=1,
    min_year=2020,
    max_year=None
)

print(f"\nCollection result:")
print(f"  Success: {result.success}")
print(f"  Total conditions: {result.total_conditions}")
print(f"  Successful conditions: {result.successful_conditions}")
print(f"  Failed conditions: {result.failed_conditions}")
print(f"  Papers collected: {result.total_papers_collected}")
print(f"  Time taken: {result.total_collection_time_seconds:.1f} seconds")

if result.error:
    print(f"  Error: {result.error}")

# Check database
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    print(f"\nPapers actually in database: {count}")