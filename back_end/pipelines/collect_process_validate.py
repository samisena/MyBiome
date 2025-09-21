#!/usr/bin/env python3
"""
Complete Research Pipeline - End-to-End Biomedical Research Automation

This script orchestrates the entire research pipeline from paper collection
through LLM processing to quality validation in a single command.

Pipeline Steps:
1. Paper Collection (PubMed + Semantic Scholar)
2. LLM Processing (Dual-model intervention extraction)
3. Quality Validation (Consistency checking)
4. Results Summary (Database statistics)

Usage:
    python research_pipeline.py "diabetes" --papers 100 --full-pipeline
    python research_pipeline.py "ibs" --collection-only
    python research_pipeline.py "gerd" --processing-only
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import setup_logging
from src.paper_collection.database_manager import database_manager
from src.paper_collection.pubmed_collector import PubMedCollector
from src.llm.dual_model_analyzer import DualModelAnalyzer

# Import the individual script functions
try:
    from correlation_consistency_checker import CorrelationConsistencyChecker
except ImportError:
    CorrelationConsistencyChecker = None

logger = setup_logging(__name__, 'research_pipeline.log')


class ResearchPipeline:
    """Complete end-to-end research pipeline orchestrator."""

    def __init__(self, use_thermal_protection: bool = True):
        self.collector = PubMedCollector()
        self.analyzer = DualModelAnalyzer()
        self.use_thermal_protection = use_thermal_protection

        # Initialize consistency checker if available
        self.consistency_checker = CorrelationConsistencyChecker() if CorrelationConsistencyChecker else None

    def run_complete_pipeline(self, condition: str, target_papers: int = 100,
                            min_year: int = 2015, batch_size: int = 5,
                            enable_s2: bool = True) -> Dict[str, Any]:
        """Run the complete research pipeline for a condition."""

        pipeline_start = time.time()
        results = {
            'condition': condition,
            'target_papers': target_papers,
            'pipeline_start': datetime.now().isoformat(),
            'steps_completed': [],
            'errors': []
        }

        logger.info(f"🚀 Starting complete research pipeline for: {condition}")
        logger.info(f"📊 Target: {target_papers} papers")

        try:
            # Step 1: Paper Collection
            logger.info("📚 Step 1: Paper Collection")
            collection_results = self._run_collection_step(
                condition, target_papers, min_year, enable_s2
            )
            results['collection'] = collection_results
            results['steps_completed'].append('collection')

            # Step 2: LLM Processing
            logger.info("🤖 Step 2: LLM Processing")
            processing_results = self._run_processing_step(
                condition, batch_size
            )
            results['processing'] = processing_results
            results['steps_completed'].append('processing')

            # Step 3: Quality Validation
            logger.info("✅ Step 3: Quality Validation")
            validation_results = self._run_validation_step()
            results['validation'] = validation_results
            results['steps_completed'].append('validation')

            # Step 4: Final Summary
            logger.info("📊 Step 4: Results Summary")
            summary_results = self._generate_final_summary(condition)
            results['summary'] = summary_results
            results['steps_completed'].append('summary')

            # Pipeline completion
            results['pipeline_duration'] = time.time() - pipeline_start
            results['status'] = 'completed'
            results['completion_time'] = datetime.now().isoformat()

            logger.info(f"🎉 Pipeline completed for '{condition}' in {results['pipeline_duration']:.1f}s")
            return results

        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            results['pipeline_duration'] = time.time() - pipeline_start
            logger.error(f"💥 Pipeline failed for '{condition}': {e}")
            raise

    def _run_collection_step(self, condition: str, target_papers: int,
                           min_year: int, enable_s2: bool) -> Dict[str, Any]:
        """Execute paper collection step."""
        step_start = time.time()

        try:
            # Get initial paper count
            initial_stats = database_manager.get_database_stats()
            initial_count = initial_stats.get('total_papers', 0)

            # Run collection
            collection_result = self.collector.collect_interventions_by_condition(
                condition=condition,
                min_year=min_year,
                max_results=target_papers,
                include_fulltext=True,
                use_interleaved_s2=enable_s2
            )

            # Get final paper count
            final_stats = database_manager.get_database_stats()
            final_count = final_stats.get('total_papers', 0)
            papers_added = final_count - initial_count

            result = {
                'papers_collected': collection_result.get('paper_count', 0),
                'papers_added_to_db': papers_added,
                'pubmed_papers': collection_result.get('pubmed_papers', 0),
                's2_papers': collection_result.get('s2_similar_papers', 0),
                'collection_time': time.time() - step_start,
                'status': collection_result.get('status', 'unknown')
            }

            logger.info(f"📚 Collection completed: {result['papers_collected']} papers ({result['collection_time']:.1f}s)")
            return result

        except Exception as e:
            logger.error(f"📚 Collection failed: {e}")
            raise

    def _run_processing_step(self, condition: str, batch_size: int) -> Dict[str, Any]:
        """Execute LLM processing step."""
        step_start = time.time()

        try:
            # Get unprocessed papers
            unprocessed_papers = self.analyzer.get_unprocessed_papers()

            if not unprocessed_papers:
                logger.info("🤖 No papers need processing")
                return {
                    'papers_processed': 0,
                    'interventions_extracted': 0,
                    'processing_time': 0,
                    'status': 'no_work_needed'
                }

            logger.info(f"🤖 Processing {len(unprocessed_papers)} papers...")

            # Run processing
            processing_result = self.analyzer.process_papers_batch(
                papers=unprocessed_papers,
                save_to_db=True,
                batch_size=batch_size
            )

            result = {
                'papers_processed': processing_result.get('successful_papers', 0),
                'failed_papers': len(processing_result.get('failed_papers', [])),
                'interventions_extracted': processing_result.get('total_interventions', 0),
                'processing_time': time.time() - step_start,
                'model_statistics': processing_result.get('model_statistics', {}),
                'token_usage': processing_result.get('token_usage', {}),
                'status': 'completed'
            }

            logger.info(f"🤖 Processing completed: {result['papers_processed']} papers, {result['interventions_extracted']} interventions ({result['processing_time']:.1f}s)")
            return result

        except Exception as e:
            logger.error(f"🤖 Processing failed: {e}")
            raise

    def _run_validation_step(self) -> Dict[str, Any]:
        """Execute quality validation step."""
        step_start = time.time()

        try:
            if not self.consistency_checker:
                logger.warning("✅ Consistency checker not available, skipping validation")
                return {
                    'status': 'skipped',
                    'reason': 'consistency_checker_not_available'
                }

            # Run consistency check
            inconsistencies = self.consistency_checker.check_all_inconsistencies()

            result = {
                'total_inconsistencies': len(inconsistencies),
                'inconsistency_types': {},
                'validation_time': time.time() - step_start,
                'status': 'completed'
            }

            # Categorize inconsistencies
            for inconsistency in inconsistencies:
                issue_type = inconsistency.get('type', 'unknown')
                result['inconsistency_types'][issue_type] = result['inconsistency_types'].get(issue_type, 0) + 1

            logger.info(f"✅ Validation completed: {result['total_inconsistencies']} inconsistencies found ({result['validation_time']:.1f}s)")
            return result

        except Exception as e:
            logger.error(f"✅ Validation failed: {e}")
            # Don't fail the entire pipeline for validation issues
            return {
                'status': 'failed',
                'error': str(e),
                'validation_time': time.time() - step_start
            }

    def _generate_final_summary(self, condition: str) -> Dict[str, Any]:
        """Generate final pipeline summary."""
        try:
            # Get database statistics
            db_stats = database_manager.get_database_stats()

            # Get condition-specific interventions
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Count interventions for this condition
                cursor.execute('''
                    SELECT COUNT(*) FROM interventions
                    WHERE LOWER(health_condition) LIKE LOWER(?)
                ''', (f'%{condition}%',))
                condition_interventions = cursor.fetchone()[0]

                # Get intervention categories for this condition
                cursor.execute('''
                    SELECT intervention_category, COUNT(*)
                    FROM interventions
                    WHERE LOWER(health_condition) LIKE LOWER(?)
                    GROUP BY intervention_category
                    ORDER BY COUNT(*) DESC
                ''', (f'%{condition}%',))
                category_breakdown = dict(cursor.fetchall())

            summary = {
                'condition': condition,
                'total_papers_in_db': db_stats.get('total_papers', 0),
                'total_interventions_in_db': db_stats.get('total_interventions', 0),
                'condition_interventions': condition_interventions,
                'condition_categories': category_breakdown,
                'database_stats': db_stats
            }

            logger.info(f"📊 Summary: {condition_interventions} interventions found for '{condition}'")
            return summary

        except Exception as e:
            logger.error(f"📊 Summary generation failed: {e}")
            return {'error': str(e)}

    def run_collection_only(self, condition: str, target_papers: int = 100,
                          min_year: int = 2015, enable_s2: bool = True) -> Dict[str, Any]:
        """Run only the paper collection step."""
        logger.info(f"📚 Running collection-only pipeline for: {condition}")
        return self._run_collection_step(condition, target_papers, min_year, enable_s2)

    def run_processing_only(self, batch_size: int = 5) -> Dict[str, Any]:
        """Run only the LLM processing step."""
        logger.info("🤖 Running processing-only pipeline")
        return self._run_processing_step("all_conditions", batch_size)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and database overview."""
        try:
            db_stats = database_manager.get_database_stats()

            # Get unprocessed paper count
            unprocessed_papers = self.analyzer.get_unprocessed_papers()

            status = {
                'database_stats': db_stats,
                'unprocessed_papers': len(unprocessed_papers),
                'ready_for_processing': len(unprocessed_papers) > 0,
                'timestamp': datetime.now().isoformat()
            }

            return status

        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}


def main():
    """Main entry point for research pipeline."""
    parser = argparse.ArgumentParser(
        description='Complete Research Pipeline - End-to-End Automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Complete pipeline for diabetes research
    python research_pipeline.py "diabetes" --papers 100

    # Collection only
    python research_pipeline.py "ibs" --collection-only --papers 50

    # Processing only (all unprocessed papers)
    python research_pipeline.py --processing-only --batch-size 5

    # Status check
    python research_pipeline.py --status-only
        """
    )

    parser.add_argument('condition', nargs='?', help='Health condition to research')
    parser.add_argument('--papers', type=int, default=100, help='Number of papers to collect (default: 100)')
    parser.add_argument('--min-year', type=int, default=2015, help='Minimum publication year (default: 2015)')
    parser.add_argument('--batch-size', type=int, default=5, help='LLM processing batch size (default: 5)')
    parser.add_argument('--collection-only', action='store_true', help='Run only paper collection step')
    parser.add_argument('--processing-only', action='store_true', help='Run only LLM processing step')
    parser.add_argument('--status-only', action='store_true', help='Show pipeline status and exit')
    parser.add_argument('--no-s2', action='store_true', help='Disable Semantic Scholar enrichment')
    parser.add_argument('--thermal-protection', action='store_true', help='Enable thermal protection (experimental)')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ResearchPipeline(use_thermal_protection=args.thermal_protection)

    try:
        if args.status_only:
            status = pipeline.get_pipeline_status()
            print("📊 Pipeline Status:")
            print(f"Total papers: {status.get('database_stats', {}).get('total_papers', 0)}")
            print(f"Total interventions: {status.get('database_stats', {}).get('total_interventions', 0)}")
            print(f"Unprocessed papers: {status.get('unprocessed_papers', 0)}")
            print(f"Ready for processing: {status.get('ready_for_processing', False)}")
            return

        if not args.condition and not args.processing_only:
            print("Error: Must provide a condition or use --processing-only")
            parser.print_help()
            sys.exit(1)

        if args.collection_only:
            # Collection only
            results = pipeline.run_collection_only(
                condition=args.condition,
                target_papers=args.papers,
                min_year=args.min_year,
                enable_s2=not args.no_s2
            )
            print(f"📚 Collection completed: {results.get('papers_collected', 0)} papers")

        elif args.processing_only:
            # Processing only
            results = pipeline.run_processing_only(batch_size=args.batch_size)
            print(f"🤖 Processing completed: {results.get('papers_processed', 0)} papers")

        else:
            # Complete pipeline
            results = pipeline.run_complete_pipeline(
                condition=args.condition,
                target_papers=args.papers,
                min_year=args.min_year,
                batch_size=args.batch_size,
                enable_s2=not args.no_s2
            )

            print("\n🎉 Pipeline Results:")
            print(f"Condition: {results['condition']}")
            print(f"Papers collected: {results.get('collection', {}).get('papers_collected', 0)}")
            print(f"Papers processed: {results.get('processing', {}).get('papers_processed', 0)}")
            print(f"Interventions extracted: {results.get('processing', {}).get('interventions_extracted', 0)}")
            print(f"Quality issues: {results.get('validation', {}).get('total_inconsistencies', 0)}")
            print(f"Total duration: {results.get('pipeline_duration', 0):.1f}s")

    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
    except Exception as e:
        print(f"💥 Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()