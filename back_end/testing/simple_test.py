#!/usr/bin/env python3
"""
Simple test script for enhanced intervention extraction pipeline.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

try:
    from src.paper_collection.database_manager import database_manager
    from src.export_to_json import export_minimal_dataset
    from src.data.config import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_database_schema():
    """Test that database schema includes new optional fields."""
    print("=== Testing Database Schema ===")

    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check if new columns exist in interventions table
            cursor.execute("PRAGMA table_info(interventions)")
            columns = [row[1] for row in cursor.fetchall()]

            required_new_columns = ['delivery_method', 'severity', 'adverse_effects', 'cost_category']
            missing_columns = [col for col in required_new_columns if col not in columns]

            if missing_columns:
                print(f"[FAIL] Missing columns: {missing_columns}")
                return False
            else:
                print("[PASS] All new columns present in database schema")
                print(f"   Found columns: {required_new_columns}")
                return True

    except Exception as e:
        print(f"[FAIL] Database schema test failed: {e}")
        return False

def test_database_stats():
    """Check current database content."""
    print("\n=== Current Database Stats ===")

    try:
        stats = database_manager.get_database_stats()

        print(f"Total papers: {stats.get('total_papers', 0)}")
        print(f"Total interventions: {stats.get('total_interventions', 0)}")

        # Check processing status
        processing_status = stats.get('processing_status', {})
        print("Processing status:")
        for status, count in processing_status.items():
            print(f"  {status}: {count}")

        return True

    except Exception as e:
        print(f"[FAIL] Database stats test failed: {e}")
        return False

def test_sample_collection():
    """Test minimal paper collection."""
    print("\n=== Testing Sample Collection ===")

    try:
        from src.paper_collection.pubmed_collector import PubMedCollector

        collector = PubMedCollector(database_manager)

        # Test with one small condition
        results = collector.collect_papers_for_condition(
            condition="depression",
            max_results=2,
            include_fulltext=False
        )

        if results.get('paper_count', 0) > 0:
            print(f"[PASS] Collected {results['paper_count']} papers")
            return True
        else:
            print("[INFO] No new papers collected (may already exist)")
            return True

    except Exception as e:
        print(f"[FAIL] Sample collection failed: {e}")
        return False

def test_minimal_export():
    """Test minimal export function."""
    print("\n=== Testing Minimal Export ===")

    try:
        minimal_data = export_minimal_dataset()

        if minimal_data:
            print(f"[PASS] Exported {len(minimal_data)} minimal records")
            if len(minimal_data) > 0:
                sample = minimal_data[0]
                print(f"   Sample: {sample}")
        else:
            print("[INFO] No data to export (database may be empty)")

        return True

    except Exception as e:
        print(f"[FAIL] Minimal export failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("Simple Enhanced Pipeline Test")
    print("=" * 40)

    tests = [
        test_database_schema,
        test_database_stats,
        test_sample_collection,
        test_minimal_export
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"[ERROR] Test crashed: {e}")

    print(f"\n=== Summary: {passed}/{total} tests passed ===")

    if passed == total:
        print("All basic tests passed!")
    else:
        print("Some tests failed - check output above")

if __name__ == "__main__":
    main()