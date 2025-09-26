"""
Simplified intervention research pipeline with dual-model analysis.
"""

import time
from typing import List, Dict, Optional, Any

from back_end.src.data.config import setup_logging
from back_end.src.data_collection.database_manager import database_manager
from back_end.src.data_collection.pubmed_collector import PubMedCollector
from back_end.src.llm_processing.dual_model_analyzer import DualModelAnalyzer
from back_end.src.llm_processing.pipeline_analyzer import PipelineAnalyzer
from back_end.src.data.utils import format_duration, calculate_success_rate

logger = setup_logging(__name__, 'intervention_pipeline.log')


class InterventionResearchPipeline:
    """
    Simplified intervention research pipeline with dual-model analysis.
    Extracts health interventions from research papers using gemma2:9b and qwen2.5:14b models.
    """

    def __init__(self):
        """
        Initialize the intervention pipeline components.
        """
        self.db_manager = database_manager
        self.collector = PubMedCollector(self.db_manager)
        self.analyzer = DualModelAnalyzer()
        self.pipeline_analyzer = PipelineAnalyzer()

        # Pipeline initialized (logging removed for performance)

    def collect_research_data(self, conditions: List[str],
                            max_papers_per_condition: int = 50,
                            include_fulltext: bool = True) -> Dict[str, Any]:
        """
        Collect research papers for health conditions.

        Args:
            conditions: List of health conditions to research
            max_papers_per_condition: Maximum papers per condition
            include_fulltext: Whether to attempt fulltext retrieval

        Returns:
            Collection results summary
        """
        # Collecting papers (logging removed for performance)
        start_time = time.time()

        try:
            collection_results = self.collector.bulk_collect_conditions(
                conditions=conditions,
                max_results=max_papers_per_condition,
                include_fulltext=include_fulltext,
                delay_between_conditions=1.5
            )

            # Calculate summary statistics
            successful_conditions = [r for r in collection_results if r.get('status') == 'success']
            total_papers = sum(r.get('paper_count', 0) for r in collection_results)

            results = {
                'conditions_processed': len(conditions),
                'successful_conditions': len(successful_conditions),
                'total_papers_collected': total_papers,
                'duration': time.time() - start_time
            }

            # Papers collected (logging removed for performance)
            return results

        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return {
                'error': str(e),
                'conditions_processed': 0,
                'successful_conditions': 0,
                'total_papers_collected': 0,
                'duration': time.time() - start_time
            }

    def analyze_interventions(self, limit_papers: Optional[int] = None,
                             batch_size: int = None) -> Dict[str, Any]:
        """
        Extract interventions from papers using dual-model approach.

        Args:
            limit_papers: Optional limit on number of papers to process
            batch_size: Number of papers to process in each batch (auto-optimized if None)

        Returns:
            Analysis results summary
        """
        logger.info("Starting dual-model intervention analysis")
        start_time = time.time()

        try:
            # Get unprocessed papers
            unprocessed_papers = self.analyzer.get_unprocessed_papers(limit_papers)

            if not unprocessed_papers:
                # No unprocessed papers found (logging removed for performance)
                return {
                    'papers_processed': 0,
                    'interventions_extracted': 0,
                    'success_rate': 100.0,
                    'duration': time.time() - start_time
                }

            # Processing papers (logging removed for performance)

            # Process papers with dual-model analyzer
            analysis_results = self.analyzer.process_papers_batch(
                papers=unprocessed_papers,
                save_to_db=True,
                batch_size=batch_size
            )

            # Calculate success rate
            success_rate = calculate_success_rate(
                analysis_results['successful_papers'],
                analysis_results['total_papers']
            )

            results = {
                'papers_processed': analysis_results['successful_papers'],
                'interventions_extracted': analysis_results['total_interventions'],
                'interventions_by_category': analysis_results['interventions_by_category'],
                'model_statistics': analysis_results['model_statistics'],
                'success_rate': success_rate,
                'duration': time.time() - start_time
            }

            # Analysis completed (logging removed for performance)
            return results

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'error': str(e),
                'papers_processed': 0,
                'interventions_extracted': 0,
                'success_rate': 0.0,
                'duration': time.time() - start_time
            }

    def generate_research_insights(self) -> Dict[str, Any]:
        """
        Generate research insights from intervention data.

        Returns:
            Research insights summary
        """
        # Generating insights (logging removed for performance)
        return self.pipeline_analyzer.generate_research_insights()

    def run_complete_pipeline(self, conditions: List[str],
                            max_papers_per_condition: int = 50,
                            include_fulltext: bool = True,
                            analyze_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete research pipeline from data collection to analysis.

        Args:
            conditions: List of health conditions to research
            max_papers_per_condition: Maximum papers per condition
            include_fulltext: Whether to retrieve fulltext
            analyze_limit: Optional limit on papers to analyze

        Returns:
            Complete pipeline results
        """
        # Starting pipeline (logging removed for performance)

        pipeline_start = time.time()

        try:
            # Stage 1: Data Collection (logging removed for performance)
            collection_results = self.collect_research_data(
                conditions=conditions,
                max_papers_per_condition=max_papers_per_condition,
                include_fulltext=include_fulltext
            )

            if collection_results.get('total_papers_collected', 0) == 0:
                logger.warning("No papers collected, skipping analysis")
                return {
                    'status': 'no_data',
                    'collection_results': collection_results,
                    'total_duration': time.time() - pipeline_start
                }

            # Stage 2: Intervention Analysis (logging removed for performance)
            analysis_results = self.analyze_interventions(
                limit_papers=analyze_limit,
                batch_size=None  # Auto-optimize based on GPU
            )

            # Stage 3: Research Insights (logging removed for performance)
            insights = self.generate_research_insights()

            # Compile final results
            total_duration = time.time() - pipeline_start

            final_results = {
                'status': 'completed',
                'total_duration': total_duration,
                'formatted_duration': format_duration(total_duration),
                'collection_results': collection_results,
                'analysis_results': analysis_results,
                'research_insights': insights
            }

            # Pipeline completed (logging removed for performance)

            return final_results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'total_duration': time.time() - pipeline_start
            }

    def print_pipeline_summary(self, results: Dict[str, Any]):
        """
        Print a comprehensive pipeline summary.

        Args:
            results: Pipeline results from run_complete_pipeline
        """
        print("\n" + "=" * 80)
        print("INTERVENTION RESEARCH PIPELINE SUMMARY")
        print("=" * 80)

        # Overall status
        status = results.get('status', 'unknown')
        duration = results.get('formatted_duration', 'unknown')
        print(f"\nStatus: {status.upper()}")
        print(f"Total Duration: {duration}")

        if status == 'error':
            print(f"Error: {results.get('error', 'Unknown error')}")
            return

        # Collection results
        collection = results.get('collection_results', {})
        print(f"\nDATA COLLECTION:")
        print(f"  Conditions Processed: {collection.get('successful_conditions', 0)}/{collection.get('conditions_processed', 0)}")
        print(f"  Papers Collected: {collection.get('total_papers_collected', 0)}")

        # Analysis results
        analysis = results.get('analysis_results', {})
        print(f"\nINTERVENTION ANALYSIS:")
        print(f"  Papers Processed: {analysis.get('papers_processed', 0)}")
        print(f"  Interventions Extracted: {analysis.get('interventions_extracted', 0)}")
        print(f"  Success Rate: {analysis.get('success_rate', 0):.1f}%")

        # Model statistics
        model_stats = analysis.get('model_statistics', {})
        if model_stats:
            print(f"  Model Statistics:")
            for model, stats in model_stats.items():
                print(f"    {model}: {stats.get('interventions', 0)} interventions from {stats.get('papers', 0)} papers")

        # Category breakdown
        categories = analysis.get('interventions_by_category', {})
        if categories:
            print(f"  Interventions by Category:")
            for category, count in categories.items():
                if count > 0:
                    print(f"    {category}: {count}")

        # Research insights
        insights = results.get('research_insights', {})
        if insights and 'database_overview' in insights:
            overview = insights['database_overview']
            print(f"\nDATABASE OVERVIEW:")
            print(f"  Total Papers: {overview.get('total_papers', 0):,}")
            print(f"  Total Interventions: {overview.get('total_interventions', 0):,}")
            print(f"  Papers with Interventions: {overview.get('papers_with_interventions', 0):,}")

        print("\n" + "=" * 80)

    def get_research_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a comprehensive research summary as structured data.

        Args:
            results: Pipeline results from run_complete_pipeline

        Returns:
            Structured summary data
        """
        if results.get('status') == 'error':
            return {'error': results.get('error')}

        collection = results.get('collection_results', {})
        analysis = results.get('analysis_results', {})
        insights = results.get('research_insights', {})

        return {
            'pipeline_status': results.get('status'),
            'total_duration': results.get('total_duration'),
            'collection_summary': {
                'conditions_processed': collection.get('conditions_processed', 0),
                'successful_conditions': collection.get('successful_conditions', 0),
                'total_papers_collected': collection.get('total_papers_collected', 0)
            },
            'analysis_summary': {
                'papers_processed': analysis.get('papers_processed', 0),
                'interventions_extracted': analysis.get('interventions_extracted', 0),
                'success_rate': analysis.get('success_rate', 0),
                'interventions_by_category': analysis.get('interventions_by_category', {}),
                'model_statistics': analysis.get('model_statistics', {})
            },
            'database_overview': insights.get('database_overview', {}),
            'top_findings': insights.get('top_findings', {})
        }


# Backward compatibility alias
ResearchPipeline = InterventionResearchPipeline