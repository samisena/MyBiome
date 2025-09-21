#!/usr/bin/env python3
"""
Simple script to run Semantic Scholar enrichment on existing papers.
This can be integrated into your existing pipeline or run standalone.
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Add back_end directory for imports
back_end_dir = Path(__file__).parent
sys.path.insert(0, str(back_end_dir))

try:
    from src.paper_collection.semantic_scholar_enrichment import run_semantic_scholar_enrichment
    from src.data.config import setup_logging
    from src.paper_collection.database_manager import database_manager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)


def main():
    """Main function to run S2 enrichment."""
    parser = argparse.ArgumentParser(
        description="Enrich existing papers with Semantic Scholar data and discover similar papers"
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Maximum number of papers to process (default: all unprocessed papers)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(__name__, 'run_s2_enrichment.log')

    if not args.quiet:
        print("=" * 60)
        print("🧬 MyBiome - Semantic Scholar Enrichment")
        print("=" * 60)

        # Show current database stats
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('SELECT COUNT(*) FROM papers')
                total_papers = cursor.fetchone()[0]

                cursor.execute('SELECT COUNT(*) FROM papers WHERE s2_processed = 1')
                processed_papers = cursor.fetchone()[0]

                cursor.execute('SELECT COUNT(*) FROM papers WHERE s2_processed = 0 OR s2_processed IS NULL')
                unprocessed_papers = cursor.fetchone()[0]

                print(f"📊 Current Status:")
                print(f"   Total papers: {total_papers}")
                print(f"   S2 processed: {processed_papers}")
                print(f"   S2 unprocessed: {unprocessed_papers}")

                if unprocessed_papers == 0:
                    print("\n✅ All papers have already been processed by Semantic Scholar!")
                    print("   Run with fresh papers to see enrichment in action.")
                    return True

        except Exception as e:
            print(f"⚠️  Could not check database status: {e}")

    # Run the enrichment pipeline
    try:
        if not args.quiet:
            print(f"\n🚀 Starting Semantic Scholar enrichment...")
            if args.limit:
                print(f"   Processing up to {args.limit} papers")
            else:
                print("   Processing all unprocessed papers")
            print()

        # Execute the enrichment
        results = run_semantic_scholar_enrichment(limit=args.limit)

        # Extract results
        enrichment = results['enrichment']
        discovery = results['discovery']
        pipeline = results['pipeline']

        # Show results
        if not args.quiet:
            print("\n" + "=" * 60)
            print("📈 ENRICHMENT RESULTS")
            print("=" * 60)

            print(f"🔬 Paper Enrichment:")
            print(f"   Total papers processed: {enrichment['total_papers']}")
            print(f"   Successfully enriched: {enrichment['enriched_papers']}")
            print(f"   Failed to enrich: {enrichment['failed_papers']}")

            if enrichment['enriched_papers'] > 0:
                success_rate = (enrichment['enriched_papers'] / enrichment['total_papers']) * 100
                print(f"   Success rate: {success_rate:.1f}%")

            print(f"\n🔍 Similar Paper Discovery:")
            print(f"   New papers found: {discovery['new_papers_found']}")
            print(f"   Duplicates skipped: {discovery['duplicate_papers']}")

            print(f"\n⏱️  Pipeline Summary:")
            print(f"   Status: {pipeline['status']}")
            print(f"   Total time: {pipeline['total_time_seconds']}s")

            # Show errors if any
            all_errors = enrichment['errors'] + discovery['errors']
            if all_errors:
                print(f"\n⚠️  Errors encountered:")
                for error in all_errors[:5]:  # Show first 5 errors
                    print(f"   • {error}")
                if len(all_errors) > 5:
                    print(f"   ... and {len(all_errors) - 5} more errors")

            # Show sample of enriched papers
            try:
                with database_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT pmid, title, citation_count, influence_score, tldr
                        FROM papers
                        WHERE s2_paper_id IS NOT NULL
                        ORDER BY influence_score DESC
                        LIMIT 3
                    ''')
                    top_papers = cursor.fetchall()

                    if top_papers:
                        print(f"\n🌟 Top Enriched Papers by Influence Score:")
                        for pmid, title, citations, influence, tldr in top_papers:
                            title_short = title[:50] + "..." if len(title) > 50 else title
                            print(f"   • PMID {pmid}: {title_short}")
                            print(f"     Citations: {citations}, Influence Score: {influence}")
                            if tldr:
                                tldr_short = tldr[:80] + "..." if len(tldr) > 80 else tldr
                                print(f"     Summary: {tldr_short}")
                            print()

            except Exception as e:
                if not args.quiet:
                    print(f"⚠️  Could not show sample papers: {e}")

        # Determine success
        success = (
            results['pipeline']['status'] in ['success', 'partial_success'] and
            (enrichment['enriched_papers'] > 0 or discovery['new_papers_found'] > 0)
        )

        if not args.quiet:
            if success:
                print("🎉 Semantic Scholar enrichment completed successfully!")
            else:
                print("❌ Semantic Scholar enrichment completed with issues.")

            print("=" * 60)

        return success

    except Exception as e:
        logger.error(f"Enrichment failed: {e}")
        if not args.quiet:
            print(f"❌ Enrichment failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)