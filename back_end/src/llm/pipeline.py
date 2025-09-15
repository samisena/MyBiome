"""
Main pipeline that demonstrates the improved architecture.
"""

import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import sys

from src.data.config import config, setup_logging
from src.paper_collection.database_manager import database_manager
from src.paper_collection.pubmed_collector import PubMedCollector  
from src.llm.dual_model_analyzer import DualModelAnalyzer
from src.data.utils import format_duration, calculate_success_rate

logger = setup_logging(__name__, 'enhanced_pipeline.log')


class InterventionResearchPipeline:
    """
    Intervention research pipeline with dual-model analysis.
    Supports all intervention types: exercise, diet, supplements, medication, therapy, lifestyle.
    """
    
    def __init__(self):
        """
        Initialize the intervention pipeline with dual-model analyzer.
        """
        self.db_manager = database_manager
        self.collector = PubMedCollector(self.db_manager)
        
        # Initialize dual-model analyzer (gemma2:9b + qwen2.5:14b)
        self.analyzer = DualModelAnalyzer()
        
        self.results = {
            'pipeline_start_time': time.time(),
            'stages': {},
            'final_stats': {}
        }
        
        logger.info("Intervention research pipeline initialized with dual-model analysis (gemma2:9b + qwen2.5:14b)")
    
    # Removed @log_execution_time - use error_handler.py decorators instead
    def collect_research_data(self, conditions: List[str], 
                            max_papers_per_condition: int = 50,
                            include_fulltext: bool = True) -> Dict[str, Any]:
        """
        Collect intervention research papers for multiple health conditions.
        
        Args:
            conditions: List of health conditions to research
            max_papers_per_condition: Maximum papers per condition
            include_fulltext: Whether to attempt fulltext retrieval
            
        Returns:
            Collection results summary
        """
        logger.info(f"Starting intervention data collection for {len(conditions)} conditions")
        stage_start = time.time()
        
        try:
            # Collect papers using intervention-focused collector
            collection_results = self.collector.bulk_collect_conditions(
                conditions=conditions,
                max_results=max_papers_per_condition,
                include_fulltext=include_fulltext,
                delay_between_conditions=1.5
            )
            
            # Analyze results
            successful_conditions = [r for r in collection_results if r.get('status') == 'success']
            total_papers = sum(r.get('paper_count', 0) for r in collection_results)
            
            stage_results = {
                'conditions_processed': len(conditions),
                'successful_conditions': len(successful_conditions),
                'total_papers_collected': total_papers,
                'collection_details': collection_results,
                'stage_duration': time.time() - stage_start
            }
            
            self.results['stages']['data_collection'] = stage_results
            
            logger.info(f"Collection completed: {total_papers} papers from {len(successful_conditions)}/{len(conditions)} conditions")
            return stage_results
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            error_results = {
                'error': str(e),
                'conditions_processed': 0,
                'successful_conditions': 0,
                'total_papers_collected': 0,
                'stage_duration': time.time() - stage_start
            }
            self.results['stages']['data_collection'] = error_results
            return error_results
    
    # Removed @log_execution_time - use error_handler.py decorators instead
    def analyze_interventions(self, limit_papers: Optional[int] = None,
                             batch_size: int = 3) -> Dict[str, Any]:
        """
        Analyze papers for health interventions using dual-model approach.
        
        Args:
            limit_papers: Optional limit on number of papers to process
            batch_size: Number of papers to process in each batch (small for dual models)
            
        Returns:
            Analysis results summary
        """
        logger.info(f"Starting intervention analysis with dual-model approach")
        stage_start = time.time()
        
        try:
            # Get unprocessed papers
            unprocessed_papers = self.analyzer.get_unprocessed_papers(limit_papers)
            
            if not unprocessed_papers:
                logger.info("No unprocessed papers found for analysis")
                stage_results = {
                    'papers_available': 0,
                    'papers_processed': 0,
                    'interventions_extracted': 0,
                    'success_rate': 100.0,
                    'stage_duration': time.time() - stage_start
                }
                self.results['stages']['intervention_analysis'] = stage_results
                return stage_results
            
            logger.info(f"Found {len(unprocessed_papers)} unprocessed papers")
            
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
            
            # Build stage results
            stage_results = {
                'papers_available': len(unprocessed_papers),
                'papers_processed': analysis_results['successful_papers'],
                'failed_papers': len(analysis_results['failed_papers']),
                'interventions_extracted': analysis_results['total_interventions'],
                'interventions_by_category': analysis_results['interventions_by_category'],
                'model_statistics': analysis_results['model_statistics'],
                'success_rate': success_rate,
                'token_usage': analysis_results['token_usage'],
                'stage_duration': time.time() - stage_start,
                'analysis_type': 'dual_model'
            }
            
            logger.info(f"Dual-model analysis completed: {analysis_results['total_interventions']} interventions from {analysis_results['successful_papers']} papers")
            
            # Log model statistics
            for model, stats in analysis_results['model_statistics'].items():
                logger.info(f"  {model}: {stats['interventions']} interventions from {stats['papers']} papers")
            
            self.results['stages']['intervention_analysis'] = stage_results
            return stage_results
            
        except Exception as e:
            logger.error(f"Intervention analysis failed: {e}")
            error_results = {
                'error': str(e),
                'papers_processed': 0,
                'interventions_extracted': 0,
                'success_rate': 0.0,
                'stage_duration': time.time() - stage_start,
                'analysis_type': 'dual_model'
            }
            self.results['stages']['intervention_analysis'] = error_results
            return error_results
    
    def generate_research_insights(self) -> Dict[str, Any]:
        """
        Generate research insights from the collected and analyzed data.
        
        Returns:
            Research insights summary
        """
        logger.info("Generating research insights")
        
        try:
            # Get comprehensive database statistics
            db_stats = self.db_manager.get_database_stats()
            
            insights = {
                'database_overview': {
                    'total_papers': db_stats.get('total_papers', 0),
                    'total_correlations': db_stats.get('total_correlations', 0),
                    'papers_with_fulltext': db_stats.get('papers_with_fulltext', 0),
                    'processing_status': db_stats.get('processing_status', {}),
                    'validation_status': db_stats.get('validation_status', {})
                },
                'research_coverage': self._analyze_research_coverage(),
                'top_findings': self._get_top_findings(),
                'data_quality': self._assess_data_quality()
            }
            
            self.results['research_insights'] = insights
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {'error': str(e)}
    
    def _analyze_research_coverage(self) -> Dict[str, Any]:
        """Analyze research coverage across strains and conditions."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Top studied strains
                cursor.execute('''
                    SELECT probiotic_strain, COUNT(DISTINCT paper_id) as paper_count,
                           COUNT(*) as correlation_count
                    FROM correlations 
                    GROUP BY probiotic_strain
                    ORDER BY paper_count DESC, correlation_count DESC
                    LIMIT 10
                ''')
                top_strains = [
                    {'strain': row[0], 'papers': row[1], 'correlations': row[2]}
                    for row in cursor.fetchall()
                ]
                
                # Top studied conditions
                cursor.execute('''
                    SELECT health_condition, COUNT(DISTINCT paper_id) as paper_count,
                           COUNT(*) as correlation_count
                    FROM correlations 
                    GROUP BY health_condition
                    ORDER BY paper_count DESC, correlation_count DESC
                    LIMIT 10
                ''')
                top_conditions = [
                    {'condition': row[0], 'papers': row[1], 'correlations': row[2]}
                    for row in cursor.fetchall()
                ]
                
                return {
                    'top_strains': top_strains,
                    'top_conditions': top_conditions
                }
                
        except Exception as e:
            logger.error(f"Error analyzing research coverage: {e}")
            return {}
    
    def _get_top_findings(self) -> Dict[str, Any]:
        """Get top research findings with strongest evidence."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Strongest positive correlations
                cursor.execute('''
                    SELECT probiotic_strain, health_condition,
                           COUNT(DISTINCT paper_id) as study_count,
                           AVG(correlation_strength) as avg_strength,
                           AVG(confidence_score) as avg_confidence
                    FROM correlations 
                    WHERE correlation_type = 'positive' 
                      AND correlation_strength IS NOT NULL
                      AND confidence_score IS NOT NULL
                    GROUP BY probiotic_strain, health_condition
                    HAVING study_count >= 2
                    ORDER BY avg_strength DESC, avg_confidence DESC, study_count DESC
                    LIMIT 10
                ''')
                
                top_positive = [
                    {
                        'strain': row[0],
                        'condition': row[1], 
                        'studies': row[2],
                        'avg_strength': round(row[3], 3) if row[3] else None,
                        'avg_confidence': round(row[4], 3) if row[4] else None
                    }
                    for row in cursor.fetchall()
                ]
                
                return {'strongest_positive_correlations': top_positive}
                
        except Exception as e:
            logger.error(f"Error getting top findings: {e}")
            return {}
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess overall data quality metrics."""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Papers with abstracts
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_papers,
                        COUNT(CASE WHEN abstract IS NOT NULL AND abstract != '' THEN 1 END) as papers_with_abstracts,
                        COUNT(CASE WHEN has_fulltext = TRUE THEN 1 END) as papers_with_fulltext
                    FROM papers
                ''')
                paper_stats = cursor.fetchone()
                
                # Correlation completeness
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_correlations,
                        COUNT(CASE WHEN correlation_strength IS NOT NULL THEN 1 END) as with_strength,
                        COUNT(CASE WHEN confidence_score IS NOT NULL THEN 1 END) as with_confidence,
                        COUNT(CASE WHEN supporting_quote IS NOT NULL AND supporting_quote != '' THEN 1 END) as with_quotes
                    FROM correlations
                ''')
                correlation_stats = cursor.fetchone()
                
                quality_metrics = {
                    'abstract_coverage': (paper_stats[1] / paper_stats[0] * 100) if paper_stats[0] > 0 else 0,
                    'fulltext_coverage': (paper_stats[2] / paper_stats[0] * 100) if paper_stats[0] > 0 else 0,
                    'correlation_completeness': {
                        'with_strength': (correlation_stats[1] / correlation_stats[0] * 100) if correlation_stats[0] > 0 else 0,
                        'with_confidence': (correlation_stats[2] / correlation_stats[0] * 100) if correlation_stats[0] > 0 else 0,
                        'with_supporting_quotes': (correlation_stats[3] / correlation_stats[0] * 100) if correlation_stats[0] > 0 else 0
                    }
                }
                
                return quality_metrics
                
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {}
    
    # Removed @log_execution_time - use error_handler.py decorators instead
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
        logger.info("=" * 60)
        logger.info("STARTING ENHANCED RESEARCH PIPELINE")
        logger.info("=" * 60)
        
        pipeline_start = time.time()
        
        try:
            # Stage 1: Data Collection
            logger.info("Stage 1: Data Collection")
            collection_results = self.collect_research_data(
                conditions=conditions,
                max_papers_per_condition=max_papers_per_condition,
                include_fulltext=include_fulltext
            )
            
            if collection_results.get('total_papers_collected', 0) == 0:
                logger.warning("No papers collected, skipping analysis")
                return self._compile_final_results(pipeline_start)
            
            # Stage 2: Intervention Analysis
            logger.info("Stage 2: Intervention Analysis")
            analysis_results = self.analyze_interventions(
                limit_papers=analyze_limit,
                batch_size=3  # Small batch size for dual models
            )
            
            # Stage 3: Research Insights
            logger.info("Stage 3: Research Insights")
            insights = self.generate_research_insights()
            
            # Compile final results
            final_results = self._compile_final_results(pipeline_start)
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return self._compile_final_results(pipeline_start, error=str(e))
    
    def _compile_final_results(self, pipeline_start: float, error: Optional[str] = None) -> Dict[str, Any]:
        """Compile final pipeline results."""
        total_duration = time.time() - pipeline_start
        
        final_results = {
            'pipeline_status': 'error' if error else 'completed',
            'total_duration': total_duration,
            'formatted_duration': format_duration(total_duration),
            'stages': self.results.get('stages', {}),
            'research_insights': self.results.get('research_insights', {}),
            'timestamp': time.time()
        }
        
        if error:
            final_results['error'] = error
        
        # Add summary statistics
        collection = final_results['stages'].get('data_collection', {})
        analysis = final_results['stages'].get('intervention_analysis', {})
        
        # Handle dual-model analysis results
        interventions_metric = analysis.get('interventions_extracted', 0)
        
        final_results['summary'] = {
            'conditions_processed': collection.get('successful_conditions', 0),
            'papers_collected': collection.get('total_papers_collected', 0),
            'papers_analyzed': analysis.get('papers_processed', 0),
            'interventions_found': interventions_metric,
            'interventions_by_category': analysis.get('interventions_by_category', {}),
            'model_statistics': analysis.get('model_statistics', {}),
            'analysis_type': analysis.get('analysis_type', 'dual_model'),
            'overall_success': not error and collection.get('total_papers_collected', 0) > 0
        }
        
        self.results['final_results'] = final_results
        return final_results
    
    def print_pipeline_summary(self):
        """Print a comprehensive pipeline summary."""
        final_results = self.results.get('final_results', {})
        
        print("\n" + "=" * 80)
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get a comprehensive research summary."""
        print("ENHANCED RESEARCH PIPELINE SUMMARY")
        print("=" * 80)
        
        # Overall status
        status = final_results.get('pipeline_status', 'unknown')
        duration = final_results.get('formatted_duration', 'unknown')
        print(f"\nStatus: {status.upper()}")
        print(f"Total Duration: {duration}")
        
        # Summary statistics  
        summary = final_results.get('summary', {})
        analysis_type = summary.get('analysis_type', 'unknown')
        print(f"\nSUMMARY STATISTICS ({analysis_type.upper()}):")
        print(f"  Conditions Processed: {summary.get('conditions_processed', 0)}")
        print(f"  Papers Collected: {summary.get('papers_collected', 0)}")
        print(f"  Papers Analyzed: {summary.get('papers_analyzed', 0)}")
        
        if analysis_type == 'consensus':
            print(f"  Agreed Correlations: {summary.get('correlations_found', 0)}")
            print(f"  Conflicts: {summary.get('conflicts', 0)}")
            print(f"  Papers Needing Review: {summary.get('papers_needing_review', 0)}")
        else:
            print(f"  Correlations Found: {summary.get('correlations_found', 0)}")
        
        # Stage details
        stages = final_results.get('stages', {})
        
        if 'data_collection' in stages:
            dc = stages['data_collection']
            print(f"\nDATA COLLECTION:")
            print(f"  Duration: {format_duration(dc.get('stage_duration', 0))}")
            print(f"  Success Rate: {dc.get('successful_conditions', 0)}/{dc.get('conditions_processed', 0)} conditions")
        
        if 'intervention_analysis' in stages:
            ia = stages['intervention_analysis'] 
            analysis_type = ia.get('analysis_type', 'unknown')
            print(f"\nINTERVENTION ANALYSIS ({analysis_type.upper()}):")
            print(f"  Duration: {format_duration(ia.get('stage_duration', 0))}")
            print(f"  Success Rate: {ia.get('success_rate', 0):.1f}%")
            
            # Show model statistics
            model_stats = ia.get('model_statistics', {})
            if model_stats:
                print(f"  Model Statistics:")
                for model, stats in model_stats.items():
                    print(f"    {model}: {stats.get('interventions', 0)} interventions from {stats.get('papers', 0)} papers")
            
            # Show intervention categories
            categories = ia.get('interventions_by_category', {})
            if categories:
                print(f"  Interventions by Category:")
                for category, count in categories.items():
                    if count > 0:
                        print(f"    {category}: {count}")
            
            token_usage = ia.get('token_usage', {})
            if token_usage:
                print(f"  Token Usage by Model:")
                for model, usage in token_usage.items():
                    if isinstance(usage, dict):
                        print(f"    {model}: {usage.get('total', 0):,}")
                    else:
                        print(f"    {model}: {usage:,}")
        
        # Research insights
        insights = final_results.get('research_insights', {})
        if insights and 'database_overview' in insights:
            overview = insights['database_overview']
            print(f"\nDATABASE OVERVIEW:")
            print(f"  Total Papers: {overview.get('total_papers', 0):,}")
            print(f"  Total Interventions: {overview.get('total_interventions', 0):,}")
            print(f"  Papers with Fulltext: {overview.get('papers_with_fulltext', 0):,}")
        
        print("\n" + "=" * 80)


# Backward compatibility alias
ResearchPipeline = InterventionResearchPipeline