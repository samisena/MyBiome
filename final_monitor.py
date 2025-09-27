#!/usr/bin/env python3
"""Final monitor for pipeline progress with correct schema."""

import sqlite3
from pathlib import Path
from datetime import datetime

def check_pipeline_progress():
    """Check all three key processes."""
    print(f"=== Pipeline Progress Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    db_path = Path("back_end/data/processed/intervention_research.db")
    if not db_path.exists():
        print("ERROR: Database not found")
        return

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # 1. PAPER COLLECTION - Check recent papers
        cursor.execute("""
            SELECT COUNT(*) as total_papers,
                   MAX(created_at) as latest_collection,
                   COUNT(DISTINCT discovery_source) as sources_used
            FROM papers
            WHERE created_at > datetime('now', '-1 hour')
        """)
        papers_result = cursor.fetchone()

        # Also check overall paper count
        cursor.execute("SELECT COUNT(*) FROM papers")
        total_papers = cursor.fetchone()[0]

        print("\n1. PAPER COLLECTION:")
        print(f"   Total papers in database: {total_papers}")
        print(f"   Papers added (last hour): {papers_result[0]}")
        print(f"   Latest collection: {papers_result[1] or 'None'}")
        print(f"   Discovery sources used: {papers_result[2]}")

        # 2. DUAL LLM ANALYSIS - Check intervention extractions
        cursor.execute("""
            SELECT COUNT(*) as total_extractions,
                   COUNT(DISTINCT extraction_model) as models_used,
                   MAX(extraction_timestamp) as latest_extraction
            FROM intervention_extractions
            WHERE extraction_timestamp > datetime('now', '-1 hour')
        """)
        extraction_result = cursor.fetchone()

        # Check total interventions
        cursor.execute("SELECT COUNT(*) FROM interventions")
        total_interventions = cursor.fetchone()[0]

        # Check if dual models are being used
        cursor.execute("""
            SELECT extraction_model, COUNT(*) as extractions
            FROM intervention_extractions
            WHERE extraction_timestamp > datetime('now', '-1 hour')
            GROUP BY extraction_model
        """)
        model_usage = cursor.fetchall()

        print("\n2. DUAL LLM ANALYSIS:")
        print(f"   Total interventions found: {total_interventions}")
        print(f"   Extractions (last hour): {extraction_result[0]}")
        print(f"   Models used: {extraction_result[1]}")
        print(f"   Latest extraction: {extraction_result[2] or 'None'}")
        if model_usage:
            print("   Model breakdown:")
            for model, count in model_usage:
                print(f"     {model}: {count} extractions")

        # 3. DUPLICATE DETECTION - Check for consensus processing
        cursor.execute("""
            SELECT COUNT(*) as consensus_records,
                   MAX(consensus_timestamp) as latest_consensus
            FROM intervention_consensus
            WHERE consensus_timestamp > datetime('now', '-1 hour')
        """)
        consensus_result = cursor.fetchone()

        # Check normalization activity
        cursor.execute("""
            SELECT COUNT(*) as normalized_interventions
            FROM interventions
            WHERE normalized = 1
        """)
        normalized_count = cursor.fetchone()[0]

        print("\n3. DUPLICATE DETECTION & CONSENSUS:")
        print(f"   Consensus records (last hour): {consensus_result[0]}")
        print(f"   Latest consensus: {consensus_result[1] or 'None'}")
        print(f"   Normalized interventions: {normalized_count}")

        # SUMMARY
        active_processes = 0
        processes_status = []

        if papers_result[0] > 0:
            active_processes += 1
            processes_status.append("Paper Collection: ACTIVE")
        else:
            processes_status.append("Paper Collection: IDLE")

        if extraction_result[0] > 0:
            active_processes += 1
            processes_status.append("LLM Analysis: ACTIVE")
        else:
            processes_status.append("LLM Analysis: IDLE")

        if consensus_result[0] > 0:
            active_processes += 1
            processes_status.append("Duplicate Detection: ACTIVE")
        else:
            processes_status.append("Duplicate Detection: IDLE")

        print(f"\n=== SUMMARY ===")
        print(f"Active processes: {active_processes}/3")
        for status in processes_status:
            print(f"  {status}")

        # Check if pipeline is working end-to-end
        if active_processes == 3:
            print("\nSTATUS: All three processes are ACTIVE - Pipeline working correctly!")
        elif active_processes > 0:
            print(f"\nSTATUS: {active_processes} processes active - Pipeline partially working")
        else:
            print("\nSTATUS: No activity detected - Pipeline may be idle or starting up")

        conn.close()

    except Exception as e:
        print(f"Database error: {e}")

    print("=" * 60)

if __name__ == "__main__":
    check_pipeline_progress()