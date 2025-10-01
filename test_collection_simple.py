#!/usr/bin/env python3
"""Test collection directly."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.orchestration.rotation_paper_collector import RotationPaperCollector

collector = RotationPaperCollector()

# Test with just 1 condition and 1 paper
print("Testing collection for 1 condition with 1 paper...")

# Get first condition
conditions = collector.get_all_conditions()
test_condition = conditions[0]
print(f"Testing with condition: {test_condition}")

# Test the single condition collection
result = collector._collect_single_condition_without_s2(
    condition=test_condition,
    target_count=1,
    min_year=2020,
    max_year=None
)

print(f"Result: {result}")

# Check database
from back_end.src.data_collection.database_manager import database_manager
import sqlite3

with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    print(f"Papers in database: {count}")