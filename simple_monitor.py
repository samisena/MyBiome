#!/usr/bin/env python3
"""Simple monitor for pipeline progress."""

import time
import sqlite3
from pathlib import Path
from datetime import datetime

def check_progress():
    """Check pipeline progress."""
    print(f"=== Pipeline Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    db_path = Path("back_end/data/processed/intervention_research.db")
    if not db_path.exists():
        print("Database not found")
        return

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check papers collected in last hour
        cursor.execute("""
            SELECT COUNT(*) as total_papers,
                   MAX(collection_timestamp) as latest_collection,
                   COUNT(DISTINCT condition) as conditions_with_papers
            FROM papers
            WHERE collection_timestamp > datetime('now', '-1 hour')
        """)

        papers_result = cursor.fetchone()

        # Check LLM processing
        cursor.execute("""
            SELECT COUNT(*) as processed_papers,
                   MAX(processing_timestamp) as latest_processing
            FROM interventions
            WHERE processing_timestamp > datetime('now', '-1 hour')
        """)

        llm_result = cursor.fetchone()

        # Check duplicate detection
        cursor.execute("""
            SELECT COUNT(*) as checked_papers,
                   COUNT(CASE WHEN is_duplicate = 1 THEN 1 END) as duplicates_found
            FROM papers
            WHERE dedup_timestamp > datetime('now', '-1 hour')
        """)

        dedup_result = cursor.fetchone()

        conn.close()

        print("\n1. PAPER COLLECTION:")
        print(f"   Papers collected (last hour): {papers_result[0]}")
        print(f"   Latest collection: {papers_result[1] or 'None'}")
        print(f"   Conditions with papers: {papers_result[2]}")

        print("\n2. DUAL LLM ANALYSIS:")
        print(f"   Papers processed (last hour): {llm_result[0]}")
        print(f"   Latest processing: {llm_result[1] or 'None'}")

        print("\n3. DUPLICATE DETECTION:")
        print(f"   Papers checked (last hour): {dedup_result[0]}")
        print(f"   Duplicates found: {dedup_result[1]}")

        # Summary
        active_processes = 0
        if papers_result[0] > 0:
            active_processes += 1
            print("   [ACTIVE] Paper collection")
        if llm_result[0] > 0:
            active_processes += 1
            print("   [ACTIVE] LLM processing")
        if dedup_result[0] > 0:
            active_processes += 1
            print("   [ACTIVE] Duplicate detection")

        print(f"\nSUMMARY: {active_processes}/3 processes active")

    except Exception as e:
        print(f"Database error: {e}")

    print("=" * 50)

if __name__ == "__main__":
    check_progress()
    print("\nTo monitor continuously, run this script repeatedly.")