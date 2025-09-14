#!/usr/bin/env python3
"""
Simple test script for Semantic Scholar integration.
Tests basic functionality without Unicode characters.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

back_end_dir = Path(__file__).parent
sys.path.insert(0, str(back_end_dir))

try:
    from src.paper_collection.database_manager import database_manager
    from src.paper_collection.semantic_scholar_enrichment import run_semantic_scholar_enrichment
    from src.data.config import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_s2_integration():
    """Test the S2 integration."""
    print("Testing Semantic Scholar Integration")
    print("-" * 40)

    # Check database setup
    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            paper_count = cursor.fetchone()[0]
            print(f"Papers in database: {paper_count}")

            if paper_count == 0:
                print("No papers found - run clear_and_collect.py first")
                return False

            # Check S2 columns
            cursor.execute("PRAGMA table_info(papers)")
            columns = [row[1] for row in cursor.fetchall()]
            s2_columns = ['s2_paper_id', 'influence_score', 'citation_count', 'tldr']

            missing = [col for col in s2_columns if col not in columns]
            if missing:
                print(f"Missing S2 columns: {missing}")
                return False

            print("Database schema: OK")

    except Exception as e:
        print(f"Database test failed: {e}")
        return False

    # Test S2 API
    try:
        from src.data.api_clients import get_semantic_scholar_client
        s2_client = get_semantic_scholar_client()

        # Test with a small batch
        test_result = s2_client.get_papers_batch(['25646566'])
        if test_result and len(test_result) > 0 and test_result[0]:
            print("S2 API connectivity: OK")
        else:
            print("S2 API test failed - no results")
            return False

    except Exception as e:
        print(f"S2 API test failed: {e}")
        return False

    # Test enrichment pipeline
    try:
        print("Running S2 enrichment (limit=3)...")
        results = run_semantic_scholar_enrichment(limit=3)

        enriched = results['enrichment']['enriched_papers']
        discovered = results['discovery']['new_papers_found']

        print(f"Enrichment results: {enriched} papers enriched, {discovered} discovered")

        if enriched > 0 or discovered > 0:
            print("S2 enrichment: OK")
            return True
        else:
            print("S2 enrichment: No papers processed")
            return False

    except Exception as e:
        print(f"S2 enrichment test failed: {e}")
        return False

def main():
    """Run the test."""
    logger = setup_logging(__name__)

    success = test_s2_integration()

    if success:
        print("\nSemantic Scholar integration test: PASSED")
        return True
    else:
        print("\nSemantic Scholar integration test: FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)