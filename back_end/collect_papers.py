#!/usr/bin/env python3
"""
Script to collect papers from PubMed and optionally run Semantic Scholar enrichment.
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
    from src.paper_collection.database_manager import database_manager
    from src.paper_collection.pubmed_collector import PubMedCollector
    from src.data.config import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)


def collect_papers(search_term: str, max_papers: int = 50, min_year: int = 2010,
                  use_interleaved_s2: bool = True):
    """Collect papers for a specific search term."""
    try:
        if use_interleaved_s2:
            print(f"\n[INTERLEAVED MODE] Collecting {max_papers} PubMed seed papers for '{search_term}' (from {min_year})")
            print("Note: Each PubMed paper will trigger discovery of ~5 similar papers via Semantic Scholar.")
            print(f"      Expected total papers: ~{max_papers * 6} (1 seed + 5 similar per seed)")
        else:
            print(f"\n[TRADITIONAL MODE] Collecting {max_papers} NEW papers for '{search_term}' (from {min_year})")
            print("Note: This will add exactly the target number of NEW papers to the database,")
            print("      excluding any papers that already exist in the database.")

        collector = PubMedCollector(database_manager)

        # Use the collection method for intervention studies
        results = collector.collect_interventions_by_condition(
            condition=search_term,
            min_year=min_year,
            max_results=max_papers,
            include_fulltext=True,
            use_interleaved_s2=use_interleaved_s2
        )

        status = results.get('status')

        if results.get('interleaved_workflow', False):
            # Interleaved workflow results
            pubmed_papers = results.get('pubmed_papers', 0)
            s2_papers = results.get('s2_similar_papers', 0)
            total_papers = results.get('paper_count', 0)
            total_searched = results.get('total_papers_searched', 0)

            if status in ['success', 'partial_success']:
                print(f"[SUCCESS] Interleaved collection completed!")
                print(f"   [PUBMED] {pubmed_papers} seed papers collected")
                print(f"   [S2] {s2_papers} similar papers discovered")
                print(f"   [TOTAL] {total_papers} papers added to database")

                if total_searched > pubmed_papers:
                    print(f"   [INFO] Searched through {total_searched} PubMed results to find {pubmed_papers} new seeds")

                multiplier = total_papers / pubmed_papers if pubmed_papers > 0 else 0
                print(f"   [MULTIPLIER] {multiplier:.1f}x expansion factor (target was ~6x)")

                if status == 'partial_success':
                    print(f"   [WARNING] Only collected {pubmed_papers} seed papers (target was {max_papers})")

                return True
        else:
            # Traditional workflow results
            new_papers = results.get('new_papers_count', 0)
            total_searched = results.get('total_papers_searched', 0)

            if status in ['success', 'partial_success']:
                print(f"[SUCCESS] Successfully collected {new_papers} NEW papers")
                if total_searched > new_papers:
                    print(f"   [INFO] Searched through {total_searched} total papers to find {new_papers} new ones")
                    print(f"   [INFO] Skipped {total_searched - new_papers} papers that already existed in database")

                if status == 'partial_success':
                    print(f"   [WARNING] Only found {new_papers} new papers (target was {max_papers})")
                    print("            Consider broadening search terms or date range for more results")

                return True

        # Handle error cases
        error_msg = results.get('message', 'Unknown error')
        print(f"[ERROR] Collection failed: {error_msg}")
        return False

    except Exception as e:
        print(f"[ERROR] Error during collection: {e}")
        return False


def run_semantic_scholar_enrichment(limit: int = None):
    """Run Semantic Scholar enrichment on collected papers."""
    try:
        print(f"\nRunning Semantic Scholar enrichment...")
        if limit:
            print(f"Processing up to {limit} papers")
        else:
            print("Processing all unprocessed papers")

        from src.paper_collection.semantic_scholar_enrichment import run_semantic_scholar_enrichment
        s2_results = run_semantic_scholar_enrichment(limit=limit)

        enriched = s2_results['enrichment']['enriched_papers']
        discovered = s2_results['discovery']['new_papers_found']
        print(f"S2 enrichment completed: {enriched} papers enriched, {discovered} similar papers discovered")

        return True

    except Exception as e:
        print(f"S2 enrichment failed: {e}")
        return False


def show_database_stats():
    """Show current database statistics."""
    print("\n=== Database Statistics ===")
    stats = database_manager.get_database_stats()
    print(f"Total papers: {stats.get('total_papers', 0)}")
    print(f"Papers with fulltext: {stats.get('papers_with_fulltext', 0)}")
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


def main():
    """Main function to collect papers."""
    parser = argparse.ArgumentParser(
        description="Collect papers from PubMed and optionally run Semantic Scholar enrichment"
    )
    parser.add_argument(
        'search_term',
        help='Search term/condition to collect papers for (e.g., "IBS", "diabetes")'
    )
    parser.add_argument(
        '--max-papers', type=int, default=50,
        help='Number of PubMed seed papers (interleaved mode) or total new papers (traditional mode) (default: 50)'
    )
    parser.add_argument(
        '--min-year', type=int, default=2010,
        help='Minimum publication year (default: 2010)'
    )
    parser.add_argument(
        '--skip-s2', action='store_true',
        help='Skip Semantic Scholar enrichment'
    )
    parser.add_argument(
        '--s2-limit', type=int, default=None,
        help='Limit number of papers for S2 enrichment (default: all)'
    )
    parser.add_argument(
        '--traditional-mode', action='store_true',
        help='Use traditional batch collection instead of interleaved S2 discovery'
    )

    args = parser.parse_args()

    print("=== MyBiome Paper Collection ===")

    # Setup logging
    logger = setup_logging(__name__, 'collect_papers.log')

    # Show initial stats
    print("\nInitial database state:")
    show_database_stats()

    # Step 1: Collect papers
    print(f"\nStep 1: Collecting papers for '{args.search_term}'...")
    use_interleaved = not args.traditional_mode
    if not collect_papers(args.search_term, args.max_papers, args.min_year, use_interleaved):
        print("Failed to collect papers. Exiting.")
        return False

    # Step 2: Optional Semantic Scholar enrichment (only for traditional mode)
    if not args.skip_s2 and args.traditional_mode:
        print("\nStep 2: Running Semantic Scholar enrichment...")
        if not run_semantic_scholar_enrichment(args.s2_limit):
            print("S2 enrichment failed (continuing anyway)")
    elif not args.traditional_mode:
        print("\nStep 2: Skipping separate S2 enrichment (already done in interleaved mode)")

    # Step 3: Show final stats
    print("\nFinal database state:")
    show_database_stats()

    print("\n=== Paper collection completed successfully! ===")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)