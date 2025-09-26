#!/usr/bin/env python3
"""
Quick script to check database status.
"""

import sys
from pathlib import Path

from back_end.src.data_collection.database_manager import database_manager

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