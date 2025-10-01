#!/usr/bin/env python3
"""Test database insertion directly."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.data_collection.database_manager import database_manager

# Create a minimal test paper
test_paper = {
    'pmid': '99999999',
    'title': 'Test Paper Title',
    'abstract': 'This is a test abstract for debugging database insertion.',
    'journal': 'Test Journal',
    'publication_date': '2024-01-01',
    'doi': None,
    'pmc_id': None,
    'keywords': ['test', 'debugging'],
    'has_fulltext': False,
    'fulltext_source': None,
    'fulltext_path': None,
    'discovery_source': 'test'
}

print("Testing database insertion...")
print(f"Database path: {database_manager.db_path}")
print(f"Database exists: {Path(database_manager.db_path).exists()}")

# Test insertion
result = database_manager.insert_paper(test_paper)
print(f"Insert result: {result}")

# Check if paper was inserted
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers WHERE pmid = ?", ('99999999',))
    count = cursor.fetchone()[0]
    print(f"Paper found in database: {count > 0}")

    # Check total papers
    cursor.execute("SELECT COUNT(*) FROM papers")
    total = cursor.fetchone()[0]
    print(f"Total papers in database: {total}")

# Clean up test paper
with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM papers WHERE pmid = ?", ('99999999',))
    conn.commit()
    print("Test paper cleaned up")