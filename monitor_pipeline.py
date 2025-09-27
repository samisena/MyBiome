#!/usr/bin/env python3
"""Monitor pipeline progress and verify the three key processes."""

import time
import sqlite3
from pathlib import Path
from datetime import datetime

def check_database_activity():
    """Check if papers are being collected by monitoring database updates."""
    db_path = Path("back_end/data/processed/intervention_research.db")
    if not db_path.exists():
        return {"status": "error", "message": "Database not found"}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check recent papers
        cursor.execute("""
            SELECT COUNT(*) as total_papers,
                   MAX(collection_timestamp) as latest_collection,
                   COUNT(DISTINCT condition) as conditions_with_papers
            FROM papers
            WHERE collection_timestamp > datetime('now', '-1 hour')
        """)

        result = cursor.fetchone()
        conn.close()

        return {
            "status": "success",
            "total_papers_last_hour": result[0],
            "latest_collection": result[1],
            "conditions_with_papers": result[2]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def check_llm_processing():
    """Check if dual LLM analysis is working."""
    db_path = Path("back_end/data/processed/intervention_research.db")
    if not db_path.exists():
        return {"status": "error", "message": "Database not found"}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check for LLM processed interventions
        cursor.execute("""
            SELECT COUNT(*) as processed_papers,
                   COUNT(DISTINCT paper_id) as unique_papers,
                   MAX(processing_timestamp) as latest_processing
            FROM interventions
            WHERE processing_timestamp > datetime('now', '-1 hour')
        """)

        result = cursor.fetchone()

        # Check for dual model analysis
        cursor.execute("""
            SELECT COUNT(*) as total_interventions,
                   COUNT(CASE WHEN model_1_analysis IS NOT NULL THEN 1 END) as model_1_count,
                   COUNT(CASE WHEN model_2_analysis IS NOT NULL THEN 1 END) as model_2_count
            FROM interventions
            WHERE processing_timestamp > datetime('now', '-1 hour')
        """)

        dual_result = cursor.fetchone()
        conn.close()

        return {
            "status": "success",
            "processed_papers_last_hour": result[0],
            "unique_papers": result[1],
            "latest_processing": result[2],
            "total_interventions": dual_result[0],
            "model_1_processed": dual_result[1],
            "model_2_processed": dual_result[2],
            "dual_analysis_working": dual_result[1] > 0 and dual_result[2] > 0
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def check_duplicate_detection():
    """Check if duplicate detection is working."""
    db_path = Path("back_end/data/processed/intervention_research.db")
    if not db_path.exists():
        return {"status": "error", "message": "Database not found"}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check for duplicate detection activity
        cursor.execute("""
            SELECT COUNT(*) as total_papers,
                   COUNT(CASE WHEN is_duplicate = 1 THEN 1 END) as duplicates_found,
                   MAX(dedup_timestamp) as latest_dedup
            FROM papers
            WHERE dedup_timestamp > datetime('now', '-1 hour')
        """)

        result = cursor.fetchone()
        conn.close()

        return {
            "status": "success",
            "papers_checked_last_hour": result[0],
            "duplicates_found": result[1],
            "latest_dedup": result[2],
            "duplicate_detection_working": result[0] > 0
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def monitor_pipeline():
    """Monitor all three key processes."""
    print(f"=== Pipeline Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    # 1. Paper Collection
    print("\n📄 Paper Collection Status:")
    collection_status = check_database_activity()
    if collection_status["status"] == "success":
        print(f"  ✅ Papers collected (last hour): {collection_status['total_papers_last_hour']}")
        print(f"  📅 Latest collection: {collection_status['latest_collection']}")
        print(f"  🏷️  Conditions with papers: {collection_status['conditions_with_papers']}")
    else:
        print(f"  ❌ Error: {collection_status['message']}")

    # 2. Dual LLM Analysis
    print("\n🤖 Dual LLM Analysis Status:")
    llm_status = check_llm_processing()
    if llm_status["status"] == "success":
        print(f"  📊 Papers processed (last hour): {llm_status['processed_papers_last_hour']}")
        print(f"  🔬 Total interventions found: {llm_status['total_interventions']}")
        print(f"  🤖 Model 1 analyses: {llm_status['model_1_processed']}")
        print(f"  🤖 Model 2 analyses: {llm_status['model_2_processed']}")
        print(f"  ✅ Dual analysis working: {llm_status['dual_analysis_working']}")
        print(f"  📅 Latest processing: {llm_status['latest_processing']}")
    else:
        print(f"  ❌ Error: {llm_status['message']}")

    # 3. Duplicate Detection
    print("\n🔍 Duplicate Detection Status:")
    dedup_status = check_duplicate_detection()
    if dedup_status["status"] == "success":
        print(f"  📋 Papers checked (last hour): {dedup_status['papers_checked_last_hour']}")
        print(f"  🔄 Duplicates found: {dedup_status['duplicates_found']}")
        print(f"  ✅ Detection working: {dedup_status['duplicate_detection_working']}")
        print(f"  📅 Latest dedup: {dedup_status['latest_dedup']}")
    else:
        print(f"  ❌ Error: {dedup_status['message']}")

    # Summary
    working_processes = 0
    if collection_status["status"] == "success" and collection_status["total_papers_last_hour"] > 0:
        working_processes += 1
    if llm_status["status"] == "success" and llm_status["dual_analysis_working"]:
        working_processes += 1
    if dedup_status["status"] == "success" and dedup_status["duplicate_detection_working"]:
        working_processes += 1

    print(f"\n🎯 Summary: {working_processes}/3 processes active")
    print("=" * 60)

if __name__ == "__main__":
    while True:
        try:
            monitor_pipeline()
            print("\n⏳ Checking again in 30 seconds... (Ctrl+C to stop)")
            time.sleep(30)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped.")
            break
        except Exception as e:
            print(f"\n❌ Monitor error: {e}")
            time.sleep(30)