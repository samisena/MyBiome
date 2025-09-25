#!/usr/bin/env python3
"""
Quick script to check migration status without holding locks.
"""

import sqlite3
import time
from pathlib import Path

def check_migration_status():
    """Check the current state of the migration."""
    db_path = Path("back_end/data/processed/intervention_research.db")

    if not db_path.exists():
        print("[ERROR] Database not found")
        return

    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            # Use a very short timeout to avoid hanging
            conn = sqlite3.connect(str(db_path), timeout=1.0)

            # Quick status queries
            cursor = conn.cursor()

            # Count total papers
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]

            # Count papers with semantic fields (migrated papers)
            cursor.execute("""
                SELECT COUNT(*) FROM interventions
                WHERE canonical_name IS NOT NULL
                AND canonical_name != ''
            """)
            migrated_interventions = cursor.fetchone()[0]

            # Count total interventions
            cursor.execute("SELECT COUNT(*) FROM interventions")
            total_interventions = cursor.fetchone()[0]

            # Estimate migrated papers based on semantic fields in interventions
            cursor.execute("""
                SELECT COUNT(DISTINCT paper_id) FROM interventions
                WHERE canonical_name IS NOT NULL
                AND canonical_name != ''
            """)
            migrated_papers = cursor.fetchone()[0]

            conn.close()

            print("=== Migration Status ===")
            print(f"Total papers: {total_papers}")
            print(f"Migrated papers: {migrated_papers} ({migrated_papers/total_papers*100:.1f}%)")
            print(f"Total interventions: {total_interventions}")
            print(f"Interventions with semantic fields: {migrated_interventions} ({migrated_interventions/total_interventions*100:.1f}%)")

            if migrated_papers == total_papers:
                print("[OK] Migration appears to be COMPLETE!")
            elif migrated_papers > 0:
                print(f"[INFO] Migration in progress: {total_papers - migrated_papers} papers remaining")
            else:
                print("[INFO] Migration not started or no markers found")

            return

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                print(f"[ATTEMPT {attempt + 1}] Database locked, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[ERROR] Database error: {e}")
                return
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            return

    print("[ERROR] Could not access database after multiple attempts - likely locked by another process")

if __name__ == "__main__":
    check_migration_status()