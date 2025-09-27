#!/usr/bin/env python3
"""Continuous monitoring of pipeline progress."""

import time
import sqlite3
from pathlib import Path
from datetime import datetime

def quick_check():
    """Quick status check."""
    db_path = Path("back_end/data/processed/intervention_research.db")
    if not db_path.exists():
        return "Database not found"

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check recent activity
        cursor.execute("""
            SELECT COUNT(*) FROM papers
            WHERE created_at > datetime('now', '-15 minutes')
        """)
        recent_papers = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM intervention_extractions
            WHERE extraction_timestamp > datetime('now', '-15 minutes')
        """)
        recent_extractions = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM intervention_consensus
            WHERE consensus_timestamp > datetime('now', '-15 minutes')
        """)
        recent_consensus = cursor.fetchone()[0]

        # Check unprocessed papers
        cursor.execute("""
            SELECT COUNT(*) FROM papers p
            LEFT JOIN intervention_extractions ie ON p.pmid = ie.paper_id
            WHERE ie.paper_id IS NULL
            AND p.created_at > datetime('now', '-1 hour')
        """)
        unprocessed_papers = cursor.fetchone()[0]

        conn.close()

        status = f"{datetime.now().strftime('%H:%M:%S')} | "
        status += f"Papers(15m): {recent_papers} | "
        status += f"LLM(15m): {recent_extractions} | "
        status += f"Consensus(15m): {recent_consensus} | "
        status += f"Unprocessed: {unprocessed_papers}"

        # Activity indicators
        indicators = []
        if recent_papers > 0:
            indicators.append("COLLECTING")
        if recent_extractions > 0:
            indicators.append("ANALYZING")
        if recent_consensus > 0:
            indicators.append("DEDUPING")

        if indicators:
            status += f" | ACTIVE: {', '.join(indicators)}"
        else:
            status += " | IDLE"

        return status

    except Exception as e:
        return f"Error: {e}"

def main():
    print("=== Pipeline Continuous Monitor ===")
    print("Watching for LLM analysis and duplicate detection to start...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            status = quick_check()
            print(status)

            # Check if LLM started
            if "ANALYZING" in status:
                print("\n*** LLM ANALYSIS STARTED! ***")
                break

            time.sleep(10)  # Check every 10 seconds

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()