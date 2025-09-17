#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LLM Paper Processing Script

This script provides a single entry point for processing research papers with LLMs.
It automatically detects unprocessed papers, handles resumption of interrupted work,
and provides clear progress tracking.

Features:
- Automatic detection of papers needing processing
- Resume interrupted processing sessions
- Dual-model analysis with gemma2:9b and qwen2.5:14b
- Configurable batch sizes and limits
- Clear progress reporting
- Database status management

Usage:
    python run_llm_processing.py [options]

Examples:
    # Process all unprocessed papers in small batches
    python run_llm_processing.py --batch-size 5

    # Process only 20 papers for testing
    python run_llm_processing.py --limit 20

    # Check status without processing
    python run_llm_processing.py --status-only

    # Force reprocess specific papers (use with caution)
    python run_llm_processing.py --force-reprocess --limit 5
"""

import sys
import argparse
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Set UTF-8 encoding for stdout/stderr on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import setup_logging
from src.paper_collection.database_manager import database_manager
from src.llm.pipeline import InterventionResearchPipeline
from src.data.utils import format_duration

logger = setup_logging(__name__, 'unified_llm_processing.log')


class UnifiedLLMProcessor:
    """
    Unified processor for handling LLM-based paper analysis with resumption capabilities.
    """

    def __init__(self):
        """Initialize the unified processor."""
        self.db_manager = database_manager
        self.pipeline = InterventionResearchPipeline()
        logger.info("Unified LLM processor initialized")

    def check_processing_status(self) -> Dict[str, Any]:
        """
        Check the current processing status of papers in the database.

        Returns:
            Dictionary containing processing statistics
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Get overall paper statistics
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_papers,
                        COUNT(CASE WHEN processing_status = 'processed' THEN 1 END) as processed_papers,
                        COUNT(CASE WHEN processing_status = 'pending' THEN 1 END) as pending_papers,
                        COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_papers,
                        COUNT(CASE WHEN processing_status IS NULL THEN 1 END) as uninitialized_papers
                    FROM papers
                """)

                stats = cursor.fetchone()

                # Get papers needing processing (NULL status or 'pending')
                cursor.execute("""
                    SELECT pmid, title, processing_status
                    FROM papers
                    WHERE processing_status IS NULL OR processing_status = 'pending'
                    ORDER BY created_at ASC
                """)

                unprocessed_papers = cursor.fetchall()

                # Get intervention statistics
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_interventions,
                        COUNT(DISTINCT paper_id) as papers_with_interventions,
                        COUNT(DISTINCT extraction_model) as models_used
                    FROM interventions
                """)

                intervention_stats = cursor.fetchone()

                return {
                    'total_papers': stats[0] if stats else 0,
                    'processed_papers': stats[1] if stats else 0,
                    'pending_papers': stats[2] if stats else 0,
                    'failed_papers': stats[3] if stats else 0,
                    'uninitialized_papers': stats[4] if stats else 0,
                    'unprocessed_papers': len(unprocessed_papers),
                    'unprocessed_list': [
                        {'pmid': row[0], 'title': row[1][:80] + '...' if len(row[1]) > 80 else row[1], 'status': row[2]}
                        for row in unprocessed_papers
                    ],
                    'total_interventions': intervention_stats[0] if intervention_stats else 0,
                    'papers_with_interventions': intervention_stats[1] if intervention_stats else 0,
                    'models_used': intervention_stats[2] if intervention_stats else 0
                }

        except Exception as e:
            logger.error(f"Error checking processing status: {e}")
            return {'error': str(e)}

    def normalize_processing_status(self) -> int:
        """
        Normalize processing status by setting NULL values to 'pending'.

        Returns:
            Number of papers updated
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Update NULL processing_status to 'pending'
                cursor.execute("""
                    UPDATE papers
                    SET processing_status = 'pending', updated_at = CURRENT_TIMESTAMP
                    WHERE processing_status IS NULL
                """)

                updated_count = cursor.rowcount
                conn.commit()

                logger.info(f"Normalized {updated_count} papers from NULL to 'pending' status")
                return updated_count

        except Exception as e:
            logger.error(f"Error normalizing processing status: {e}")
            return 0

    def get_papers_for_processing(self, limit: Optional[int] = None,
                                 force_reprocess: bool = False) -> List[Dict]:
        """
        Get papers that need processing.

        Args:
            limit: Maximum number of papers to return
            force_reprocess: If True, include already processed papers

        Returns:
            List of papers to process
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                if force_reprocess:
                    query = """
                        SELECT pmid, title, abstract, journal, publication_date, doi, processing_status
                        FROM papers
                        WHERE abstract IS NOT NULL AND abstract != ''
                        ORDER BY created_at DESC
                    """
                else:
                    query = """
                        SELECT pmid, title, abstract, journal, publication_date, doi, processing_status
                        FROM papers
                        WHERE (processing_status IS NULL OR processing_status = 'pending' OR processing_status = 'failed')
                        AND abstract IS NOT NULL AND abstract != ''
                        ORDER BY created_at ASC
                    """

                if limit:
                    query += f" LIMIT {limit}"

                cursor.execute(query)
                rows = cursor.fetchall()

                papers = []
                for row in rows:
                    papers.append({
                        'pmid': row[0],
                        'title': row[1],
                        'abstract': row[2],
                        'journal': row[3],
                        'publication_date': row[4],
                        'doi': row[5],
                        'processing_status': row[6]
                    })

                return papers

        except Exception as e:
            logger.error(f"Error getting papers for processing: {e}")
            return []

    def update_paper_status(self, pmid: str, status: str, error_message: Optional[str] = None):
        """
        Update the processing status of a paper.

        Args:
            pmid: Paper ID
            status: New status ('processed', 'failed', 'pending')
            error_message: Optional error message for failed papers
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()

                if error_message:
                    cursor.execute("""
                        UPDATE papers
                        SET processing_status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE pmid = ?
                    """, (f"{status}: {error_message}", pmid))
                else:
                    cursor.execute("""
                        UPDATE papers
                        SET processing_status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE pmid = ?
                    """, (status, pmid))

                conn.commit()

        except Exception as e:
            logger.error(f"Error updating paper status for {pmid}: {e}")

    def process_papers_batch(self, papers: List[Dict], batch_size: int = 5) -> Dict[str, Any]:
        """
        Process a batch of papers using the intervention pipeline.

        Args:
            papers: List of papers to process
            batch_size: Number of papers to process at once

        Returns:
            Processing results summary
        """
        if not papers:
            return {
                'papers_processed': 0,
                'papers_failed': 0,
                'interventions_extracted': 0,
                'errors': []
            }

        logger.info(f"Processing batch of {len(papers)} papers with batch size {batch_size}")

        results = {
            'papers_processed': 0,
            'papers_failed': 0,
            'interventions_extracted': 0,
            'errors': []
        }

        try:
            # Process papers using the existing pipeline
            analysis_results = self.pipeline.analyze_interventions(
                limit_papers=len(papers),
                batch_size=batch_size
            )

            if 'error' in analysis_results:
                logger.error(f"Pipeline analysis failed: {analysis_results['error']}")
                results['errors'].append(analysis_results['error'])

                # Mark all papers as failed
                for paper in papers:
                    self.update_paper_status(paper['pmid'], 'failed', analysis_results['error'])
                    results['papers_failed'] += 1
            else:
                # Update successful results
                results['papers_processed'] = analysis_results.get('papers_processed', 0)
                results['interventions_extracted'] = analysis_results.get('interventions_extracted', 0)

                # Mark processed papers as successful (the pipeline should have done this, but let's be sure)
                for paper in papers[:results['papers_processed']]:
                    self.update_paper_status(paper['pmid'], 'processed')

                # Mark any remaining papers as failed
                if results['papers_processed'] < len(papers):
                    failed_count = len(papers) - results['papers_processed']
                    results['papers_failed'] = failed_count

                    for paper in papers[results['papers_processed']:]:
                        self.update_paper_status(paper['pmid'], 'failed', 'Processing incomplete')

            return results

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            results['errors'].append(str(e))

            # Mark all papers as failed
            for paper in papers:
                self.update_paper_status(paper['pmid'], 'failed', str(e))
                results['papers_failed'] += 1

            return results

    def run_processing(self, limit: Optional[int] = None, batch_size: int = 5,
                      force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Run the complete processing workflow.

        Args:
            limit: Maximum number of papers to process
            batch_size: Number of papers to process in each batch
            force_reprocess: If True, reprocess already processed papers

        Returns:
            Complete processing results
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("STARTING UNIFIED LLM PROCESSING")
        logger.info("=" * 60)

        try:
            # Step 1: Normalize processing status
            logger.info("Step 1: Normalizing processing status")
            normalized_count = self.normalize_processing_status()

            # Step 2: Get initial status
            logger.info("Step 2: Checking processing status")
            initial_status = self.check_processing_status()

            if 'error' in initial_status:
                return {
                    'status': 'error',
                    'error': initial_status['error'],
                    'duration': time.time() - start_time
                }

            # Step 3: Get papers to process
            logger.info("Step 3: Getting papers for processing")
            papers_to_process = self.get_papers_for_processing(limit, force_reprocess)

            if not papers_to_process:
                logger.info("No papers need processing")
                return {
                    'status': 'completed',
                    'message': 'No papers need processing',
                    'initial_status': initial_status,
                    'papers_processed': 0,
                    'interventions_extracted': 0,
                    'duration': time.time() - start_time
                }

            logger.info(f"Found {len(papers_to_process)} papers to process")

            # Step 4: Process papers in batches
            logger.info(f"Step 4: Processing papers in batches of {batch_size}")

            total_processed = 0
            total_failed = 0
            total_interventions = 0
            all_errors = []

            # Process in chunks
            for i in range(0, len(papers_to_process), batch_size):
                batch = papers_to_process[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(papers_to_process) + batch_size - 1) // batch_size

                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} papers)")

                batch_results = self.process_papers_batch(batch, batch_size)

                total_processed += batch_results['papers_processed']
                total_failed += batch_results['papers_failed']
                total_interventions += batch_results['interventions_extracted']
                all_errors.extend(batch_results['errors'])

                logger.info(f"Batch {batch_num} completed: {batch_results['papers_processed']} processed, "
                           f"{batch_results['interventions_extracted']} interventions extracted")

                # Small delay between batches to prevent overwhelming the system
                if i + batch_size < len(papers_to_process):
                    time.sleep(1)

            # Step 5: Get final status
            logger.info("Step 5: Getting final processing status")
            final_status = self.check_processing_status()

            total_duration = time.time() - start_time

            results = {
                'status': 'completed',
                'normalized_papers': normalized_count,
                'initial_status': initial_status,
                'final_status': final_status,
                'papers_processed': total_processed,
                'papers_failed': total_failed,
                'total_papers_attempted': len(papers_to_process),
                'interventions_extracted': total_interventions,
                'batch_size_used': batch_size,
                'errors': all_errors,
                'duration': total_duration,
                'formatted_duration': format_duration(total_duration)
            }

            logger.info("=" * 60)
            logger.info("UNIFIED LLM PROCESSING COMPLETED")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"Processing workflow failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'duration': time.time() - start_time
            }

    def print_status_report(self, status: Dict[str, Any]):
        """Print a detailed status report."""
        print("\n" + "=" * 80)
        print("LLM PROCESSING STATUS REPORT")
        print("=" * 80)

        if 'error' in status:
            print(f"Error: {status['error']}")
            return

        # Paper statistics
        total = status.get('total_papers', 0)
        processed = status.get('processed_papers', 0)
        pending = status.get('pending_papers', 0)
        failed = status.get('failed_papers', 0)
        uninitialized = status.get('uninitialized_papers', 0)
        unprocessed = status.get('unprocessed_papers', 0)

        print(f"\nPAPER STATISTICS:")
        print(f"   Total Papers: {total:,}")
        print(f"   Processed: {processed:,}")
        print(f"   Pending: {pending:,}")
        print(f"   Failed: {failed:,}")
        print(f"   Uninitialized: {uninitialized:,}")
        print(f"   Need Processing: {unprocessed:,}")

        if total > 0:
            completion_rate = (processed / total) * 100
            print(f"   Completion Rate: {completion_rate:.1f}%")

        # Intervention statistics
        total_interventions = status.get('total_interventions', 0)
        papers_with_interventions = status.get('papers_with_interventions', 0)
        models_used = status.get('models_used', 0)

        print(f"\nINTERVENTION STATISTICS:")
        print(f"   Total Interventions: {total_interventions:,}")
        print(f"   Papers with Interventions: {papers_with_interventions:,}")
        print(f"   Models Used: {models_used}")

        if papers_with_interventions > 0 and processed > 0:
            extraction_rate = (papers_with_interventions / processed) * 100
            avg_interventions = total_interventions / papers_with_interventions
            print(f"   Extraction Rate: {extraction_rate:.1f}%")
            print(f"   Avg Interventions/Paper: {avg_interventions:.1f}")

        # Show sample of unprocessed papers
        unprocessed_list = status.get('unprocessed_list', [])
        if unprocessed_list:
            print(f"\nPAPERS NEEDING PROCESSING (showing first 10):")
            for i, paper in enumerate(unprocessed_list[:10]):
                status_icon = "?" if paper['status'] is None else "P"
                print(f"   {i+1:2d}. {status_icon} {paper['pmid']}: {paper['title']}")

            if len(unprocessed_list) > 10:
                print(f"   ... and {len(unprocessed_list) - 10} more papers")

        print("\n" + "=" * 80)

    def print_processing_summary(self, results: Dict[str, Any]):
        """Print a processing summary."""
        print("\n" + "=" * 80)
        print("LLM PROCESSING SUMMARY")
        print("=" * 80)

        status = results.get('status', 'unknown')
        duration = results.get('formatted_duration', 'unknown')

        print(f"\nOVERALL RESULTS:")
        print(f"   Status: {status.upper()}")
        print(f"   Duration: {duration}")

        if status == 'error':
            print(f"   Error: {results.get('error', 'Unknown error')}")
            return

        if status == 'completed' and results.get('message'):
            print(f"   {results['message']}")
            return

        # Processing statistics
        attempted = results.get('total_papers_attempted', 0)
        processed = results.get('papers_processed', 0)
        failed = results.get('papers_failed', 0)
        interventions = results.get('interventions_extracted', 0)
        batch_size = results.get('batch_size_used', 0)

        print(f"\nPROCESSING RESULTS:")
        print(f"   Papers Attempted: {attempted:,}")
        print(f"   Successfully Processed: {processed:,}")
        print(f"   Failed: {failed:,}")
        print(f"   Interventions Extracted: {interventions:,}")
        print(f"   Batch Size Used: {batch_size}")

        if attempted > 0:
            success_rate = (processed / attempted) * 100
            print(f"   Success Rate: {success_rate:.1f}%")

        if processed > 0 and interventions > 0:
            avg_interventions = interventions / processed
            print(f"   Avg Interventions/Paper: {avg_interventions:.1f}")

        # Show errors if any
        errors = results.get('errors', [])
        if errors:
            print(f"\nERRORS ENCOUNTERED:")
            for i, error in enumerate(errors[:5]):  # Show first 5 errors
                print(f"   {i+1}. {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")

        # Status comparison
        initial = results.get('initial_status', {})
        final = results.get('final_status', {})

        if initial and final and not results.get('message'):
            print(f"\nBEFORE/AFTER COMPARISON:")
            print(f"   Processed Papers: {initial.get('processed_papers', 0)} -> {final.get('processed_papers', 0)}")
            print(f"   Total Interventions: {initial.get('total_interventions', 0)} -> {final.get('total_interventions', 0)}")
            print(f"   Papers Needing Processing: {initial.get('unprocessed_papers', 0)} -> {final.get('unprocessed_papers', 0)}")

        print("\n" + "=" * 80)


def main():
    """Main entry point for the unified LLM processing script."""
    parser = argparse.ArgumentParser(
        description="Unified LLM Paper Processing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --status-only              # Check status without processing
  %(prog)s --batch-size 3 --limit 10  # Process 10 papers in batches of 3
  %(prog)s --force-reprocess --limit 5 # Reprocess 5 papers (use with caution)
        """
    )

    parser.add_argument('--limit', type=int, help='Maximum number of papers to process')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of papers to process in each batch (default: 5)')
    parser.add_argument('--status-only', action='store_true', help='Only check and display status, do not process')
    parser.add_argument('--force-reprocess', action='store_true', help='Force reprocessing of already processed papers (use with caution)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output, only show final summary')

    args = parser.parse_args()

    print("MyBiome Unified LLM Processing")
    print("=" * 50)

    try:
        processor = UnifiedLLMProcessor()

        # Always check status first
        if not args.quiet:
            print("Checking current processing status...")

        status = processor.check_processing_status()

        if not args.quiet or args.status_only:
            processor.print_status_report(status)

        if args.status_only:
            return

        # Validate arguments
        if args.force_reprocess and not args.limit:
            print("\nWarning: --force-reprocess without --limit will reprocess ALL papers!")
            response = input("Are you sure you want to continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                return

        # Run processing
        if not args.quiet:
            print(f"\nStarting processing with batch size {args.batch_size}")
            if args.limit:
                print(f"Limited to {args.limit} papers")
            if args.force_reprocess:
                print("Warning: Force reprocessing enabled")

        results = processor.run_processing(
            limit=args.limit,
            batch_size=args.batch_size,
            force_reprocess=args.force_reprocess
        )

        # Show results
        processor.print_processing_summary(results)

        # Return appropriate exit code
        if results.get('status') == 'error':
            sys.exit(1)
        elif results.get('papers_failed', 0) > 0:
            sys.exit(2)  # Some papers failed but overall success
        else:
            sys.exit(0)  # Complete success

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()