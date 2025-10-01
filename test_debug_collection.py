#!/usr/bin/env python3
"""Debug collection issue."""

import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)

from back_end.src.data_collection.pubmed_collector import PubMedCollector
from back_end.src.data_collection.database_manager import database_manager
from back_end.src.orchestration.rotation_paper_collector import RotationPaperCollector

# Clear database
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM papers")
    cursor.execute("DELETE FROM interventions")
    conn.commit()
    print("Database cleared\n")

# Test 1: Direct PubMed collection
print("=== TEST 1: Direct PubMed Collection ===")
collector = PubMedCollector()
result = collector.collect_interventions_by_condition(
    "diabetes", min_year=2020, max_results=1,
    include_fulltext=False, use_interleaved_s2=False
)
print(f"Direct result: papers_collected={result.get('paper_count')}")

with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    print(f"Database after direct: {count} papers\n")

# Clear again
cursor.execute("DELETE FROM papers")
conn.commit()

# Test 2: Single condition via rotation collector
print("=== TEST 2: Single Condition via Rotation Collector ===")
rotation_collector = RotationPaperCollector()
result = rotation_collector._collect_single_condition_without_s2(
    "diabetes", target_count=1, min_year=2020, max_year=None
)
print(f"Rotation result: papers_collected={result.get('papers_collected')}")

with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    print(f"Database after rotation: {count} papers\n")

# Clear again
cursor.execute("DELETE FROM papers")
conn.commit()

# Test 3: Via _collect_with_retry
print("=== TEST 3: Via _collect_with_retry ===")
result = rotation_collector._collect_with_retry(
    "diabetes", needed_papers=1, min_year=2020, max_year=None
)
print(f"Retry result: papers_collected={result.get('papers_collected')}")

with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    print(f"Database after retry: {count} papers")