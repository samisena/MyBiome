#!/usr/bin/env python3
"""
Quick script to check database status.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

# Also add the parent directory (back_end) so we can import src
back_end_dir = src_dir.parent
sys.path.insert(0, str(back_end_dir))

try:
    from src.paper_collection.database_manager import database_manager
except ImportError:
    # Fallback for relative imports
    from paper_collection.database_manager import database_manager

def main():
    """Check database status."""
    print("=== Database Status ===")
    
    stats = database_manager.get_database_stats()
    print(f"Total papers: {stats.get('total_papers', 0)}")
    print(f"Total correlations: {stats.get('total_correlations', 0)}")
    print(f"Papers with fulltext: {stats.get('papers_with_fulltext', 0)}")
    print(f"Date range: {stats.get('date_range', 'N/A')}")
    
    if stats.get('processing_status'):
        print("\nProcessing Status:")
        for status, count in stats.get('processing_status', {}).items():
            print(f"  {status}: {count}")

if __name__ == "__main__":
    main()