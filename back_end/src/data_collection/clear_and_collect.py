#!/usr/bin/env python3
"""
Script to clear the pubmed_research.db database and collect new papers.
This script now orchestrates the separate clear_database.py and collect_papers.py scripts.
"""

import sys
import subprocess
from pathlib import Path

try:
    from back_end.src.data_collection.database_manager import database_manager
    from back_end.src.data.config import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the MyBiome directory")
    sys.exit(1)


def run_script(script_name: str, args: list = None) -> bool:
    """Run a Python script and return success status."""
    try:
        script_path = back_end_dir / script_name
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Script {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    """Main function to clear database and collect new papers using separate scripts."""
    print("=== MyBiome Database Clear & Collection ===")

    # Setup logging
    logger = setup_logging(__name__, 'clear_and_collect.log')

    # Step 1: Clear database
    print("\nStep 1: Clearing database...")
    if not run_script("clear_database.py"):
        print("Failed to clear database. Exiting.")
        return False

    # Step 2: Collect new papers (default: IBS, 50 papers)
    print("\nStep 2: Collecting new papers...")
    if not run_script("collect_papers.py", ["IBS", "--max-papers", "50"]):
        print("Failed to collect papers. Exiting.")
        return False

    # Step 3: Show final stats
    print("\nStep 3: Final database statistics...")
    stats = database_manager.get_database_stats()
    print(f"Total papers in database: {stats.get('total_papers', 0)}")
    print(f"Total interventions: {stats.get('total_interventions', 0)}")

    # Show S2 stats if available
    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM papers WHERE s2_processed = 1')
            s2_processed = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM papers WHERE s2_paper_id IS NOT NULL')
            s2_enriched = cursor.fetchone()[0]
            print(f"Semantic Scholar: {s2_processed} processed, {s2_enriched} enriched")
    except Exception:
        pass

    print("\n=== Process completed successfully! ===")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)