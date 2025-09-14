#!/usr/bin/env python3
"""
Test script for Semantic Scholar integration with the PubMed pipeline.
Tests the complete pipeline: PubMed → S2 enrichment → similar paper discovery.
"""

import sys
import time
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Add back_end directory for imports
back_end_dir = Path(__file__).parent
sys.path.insert(0, str(back_end_dir))

try:
    from src.paper_collection.database_manager import database_manager
    from src.paper_collection.semantic_scholar_enrichment import run_semantic_scholar_enrichment
    from src.data.config import setup_logging
    from src.data.repositories import repository_manager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)


def test_database_setup():
    """Test that database has S2 columns and some papers."""
    print("\n=== Testing Database Setup ===")

    try:
        # Check if S2 columns exist
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Get table schema
            cursor.execute("PRAGMA table_info(papers)")
            columns = [row[1] for row in cursor.fetchall()]

            s2_columns = ['s2_paper_id', 'influence_score', 'citation_count', 'tldr', 's2_embedding', 's2_processed']
            missing_columns = [col for col in s2_columns if col not in columns]

            if missing_columns:
                print(f"X Missing S2 columns: {missing_columns}")
                return False
            else:
                print("+ All Semantic Scholar columns present")

            # Check for papers
            cursor.execute("SELECT COUNT(*) FROM papers")
            paper_count = cursor.fetchone()[0]

            print(f"* Found {paper_count} papers in database")

            if paper_count == 0:
                print("! No papers found. You may want to run clear_and_collect.py first")
                return False

            # Check S2 processing status
            cursor.execute("SELECT COUNT(*) FROM papers WHERE s2_processed = 1")
            processed_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM papers WHERE s2_processed = 0 OR s2_processed IS NULL")
            unprocessed_count = cursor.fetchone()[0]

            print(f"* S2 Status: {processed_count} processed, {unprocessed_count} unprocessed")

            return True

    except Exception as e:
        print(f"❌ Database setup test failed: {e}")
        return False


def test_semantic_scholar_api():
    """Test Semantic Scholar API connectivity."""
    print("\n=== Testing Semantic Scholar API ===")

    try:
        from src.data.api_clients import get_semantic_scholar_client

        s2_client = get_semantic_scholar_client()

        # Test with a known PMID (example: a well-cited health paper)
        test_pmids = ['25646566']  # This is a real PMID for testing

        print(f"🧪 Testing S2 API with PMID: {test_pmids[0]}")

        # Test batch API
        papers = s2_client.get_papers_batch(test_pmids)

        if papers and len(papers) > 0 and papers[0] is not None:
            paper = papers[0]
            print(f"✅ S2 API working - found paper: {paper.get('title', 'No title')[:50]}...")
            print(f"   Citations: {paper.get('citationCount', 0)}")
            print(f"   Influence: {paper.get('influentialCitationCount', 0)}")

            if paper.get('tldr'):
                print(f"   TLDR: {paper['tldr'].get('text', '')[:100]}...")

            return True
        else:
            print("❌ S2 API test failed - no paper data returned")
            return False

    except Exception as e:
        print(f"❌ S2 API test failed: {e}")
        print("   This might be due to network issues or API limits")
        return False


def test_enrichment_pipeline():
    """Test the enrichment pipeline with a small subset."""
    print("\n=== Testing Enrichment Pipeline ===")

    try:
        print("🚀 Running S2 enrichment on up to 5 papers...")

        # Run enrichment with a small limit for testing
        results = run_semantic_scholar_enrichment(limit=5)

        print(f"📊 Pipeline Results:")
        print(f"   Enrichment - Total: {results['enrichment']['total_papers']}, "
              f"Success: {results['enrichment']['enriched_papers']}, "
              f"Failed: {results['enrichment']['failed_papers']}")

        print(f"   Discovery - New papers: {results['discovery']['new_papers_found']}, "
              f"Duplicates: {results['discovery']['duplicate_papers']}")

        print(f"   Pipeline status: {results['pipeline']['status']}")
        print(f"   Total time: {results['pipeline']['total_time_seconds']}s")

        if results['enrichment']['errors']:
            print(f"⚠️  Enrichment errors: {results['enrichment']['errors']}")

        if results['discovery']['errors']:
            print(f"⚠️  Discovery errors: {results['discovery']['errors']}")

        # Check if at least some enrichment happened
        if results['enrichment']['enriched_papers'] > 0:
            print("✅ Enrichment pipeline working")
            return True
        else:
            print("❌ No papers were enriched")
            return False

    except Exception as e:
        print(f"❌ Enrichment pipeline test failed: {e}")
        return False


def test_database_updates():
    """Test that database was properly updated with S2 data."""
    print("\n=== Testing Database Updates ===")

    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check for papers with S2 data
            cursor.execute('''
                SELECT COUNT(*) FROM papers
                WHERE s2_paper_id IS NOT NULL
            ''')
            enriched_count = cursor.fetchone()[0]

            cursor.execute('''
                SELECT pmid, title, citation_count, influence_score, tldr
                FROM papers
                WHERE s2_paper_id IS NOT NULL
                LIMIT 3
            ''')
            sample_papers = cursor.fetchall()

            print(f"📈 Found {enriched_count} papers with S2 data")

            if sample_papers:
                print("📄 Sample enriched papers:")
                for paper in sample_papers:
                    pmid, title, citations, influence, tldr = paper
                    title_short = title[:40] + "..." if len(title) > 40 else title
                    tldr_short = (tldr[:60] + "...") if tldr and len(tldr) > 60 else (tldr or "No TLDR")

                    print(f"   • {pmid}: {title_short}")
                    print(f"     Citations: {citations}, Influence: {influence}")
                    print(f"     TLDR: {tldr_short}")

                print("✅ Database updates working")
                return True
            else:
                print("❌ No papers found with S2 data")
                return False

    except Exception as e:
        print(f"❌ Database update test failed: {e}")
        return False


def show_final_stats():
    """Show final database statistics."""
    print("\n=== Final Database Statistics ===")

    try:
        stats = repository_manager.statistics.get_database_stats()
        processing_stats = repository_manager.statistics.get_processing_stats()

        print(f"📊 Papers: {stats.get('total_papers', 0)} total")
        print(f"📈 Interventions: {stats.get('total_interventions', 0)} total")
        print(f"📝 Processing status: {processing_stats.get('processing_status', {})}")

        # S2 specific stats
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM papers WHERE s2_processed = 1')
            s2_processed = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM papers WHERE s2_paper_id IS NOT NULL')
            s2_enriched = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(citation_count), AVG(influence_score) FROM papers WHERE citation_count > 0')
            result = cursor.fetchone()
            avg_citations, avg_influence = result if result[0] is not None else (0, 0)

            print(f"🔬 Semantic Scholar: {s2_processed} processed, {s2_enriched} enriched")
            print(f"📊 Average citations: {avg_citations:.1f}, Average influence: {avg_influence:.1f}")

    except Exception as e:
        print(f"❌ Stats collection failed: {e}")


def main():
    """Run all tests."""
    print("MyBiome Semantic Scholar Integration Test")
    print("=" * 50)

    # Setup logging
    logger = setup_logging(__name__, 'test_s2.log')

    # Run tests
    tests = [
        ("Database Setup", test_database_setup),
        ("Semantic Scholar API", test_semantic_scholar_api),
        ("Enrichment Pipeline", test_enrichment_pipeline),
        ("Database Updates", test_database_updates)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            time.sleep(1)  # Brief pause between tests
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")

    # Show results
    print(f"\n{'=' * 50}")
    print(f"🎯 Test Results: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Semantic Scholar integration is working.")
    elif passed_tests > 0:
        print("⚠️  Some tests passed. Check the issues above.")
    else:
        print("❌ All tests failed. Check your setup and network connection.")

    show_final_stats()

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)