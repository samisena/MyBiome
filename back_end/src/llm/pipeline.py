"""
Enhanced main pipeline that demonstrates the improved architecture.
This replaces the scattered pipeline logic with a centralized, efficient approach.
"""

import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import sys

# Add the current directory to sys.path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from ..data.config import config, setup_logging
from ..paper_collection.database_manager import database_manager
from ..paper_collection.pubmed_collector import EnhancedPubMedCollector  
from .probiotic_analyzer import EnhancedProbioticAnalyzer
from ..data.utils import log_execution_time, format_duration, calculate_success_rate

logger = setup_logging(__name__, 'enhanced_pipeline.log')


class EnhancedResearchPipeline:
    """
    Centralized research pipeline with improved architecture and efficiency.
    Demonstrates the benefits of the enhanced system design.
    """
    
    def __init__(self, llm_config=None):
        """
        Initialize the enhanced pipeline.
        
        Args:
            llm_config: Optional LLM configuration override
        """
        self.db_manager = database_manager
        self.collector = EnhancedPubMedCollector(self.db_manager)
        self.analyzer = EnhancedProbioticAnalyzer(llm_config, self.db_manager)
        
        self.results = {
            'pipeline_start_time': time.time(),
            'stages': {},
            'final_stats': {}
        }
        
        logger.info("Enhanced research pipeline initialized")
    
    @log_execution_time
    def collect_research_data(self, conditions: List[str], 
                            max_papers_per_condition: int = 50,
                            include_fulltext: bool = True) -> Dict[str, Any]:
        """
        Collect research papers for multiple health conditions.
        
        Args:
            conditions: List of health conditions to research
            max_papers_per_condition: Maximum papers per condition
            include_fulltext: Whether to attempt fulltext retrieval
            
        Returns:
            Collection results summary
        """
        logger.info(f"Starting data collection for {len(conditions)} conditions")
        stage_start = time.time()
        
        try:
            # Collect papers using enhanced collector
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
    
    @log_execution_time
    def analyze_correlations(self, limit_papers: Optional[int] = None,
                           batch_size: int = 15) -> Dict[str, Any]:
        """
        Analyze papers for probiotic-health correlations.
        
        Args:
            limit_papers: Optional limit on number of papers to process
            batch_size: Number of papers to process in each batch
            
        Returns:
            Analysis results summary
        """
        logger.info("Starting correlation analysis")
        stage_start = time.time()
        
        try:
            # Get unprocessed papers
            unprocessed_papers = self.analyzer.get_unprocessed_papers(limit_papers)
            
            if not unprocessed_papers:
                logger.info("No unprocessed papers found for analysis")
                stage_results = {
                    'papers_available': 0,
                    'papers_processed': 0,
                    'correlations_extracted': 0,
                    'success_rate': 100.0,
                    'stage_duration': time.time() - stage_start
                }
                self.results['stages']['correlation_analysis'] = stage_results
                return stage_results
            
            logger.info(f"Found {len(unprocessed_papers)} unprocessed papers")
            
            # Process papers with enhanced analyzer
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
            
            stage_results = {
                'papers_available': len(unprocessed_papers),
                'papers_processed': analysis_results['successful_papers'],
                'failed_papers': len(analysis_results['failed_papers']),
                'correlations_extracted': analysis_results['total_correlations'],
                'success_rate': success_rate,
                'token_usage': analysis_results['token_usage'],
                'stage_duration': time.time() - stage_start
            }
            
            self.results['stages']['correlation_analysis'] = stage_results
            
            logger.info(f"Analysis completed: {analysis_results['total_correlations']} correlations from {analysis_results['successful_papers']} papers")
            return stage_results
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            error_results = {
                'error': str(e),
                'papers_processed': 0,
                'correlations_extracted': 0,
                'success_rate': 0.0,
                'stage_duration': time.time() - stage_start
            }
            self.results['stages']['correlation_analysis'] = error_results
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
    
    @log_execution_time
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
            
            # Stage 2: Correlation Analysis
            logger.info("Stage 2: Correlation Analysis")
            analysis_results = self.analyze_correlations(
                limit_papers=analyze_limit,
                batch_size=15
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
        analysis = final_results['stages'].get('correlation_analysis', {})
        
        final_results['summary'] = {
            'conditions_processed': collection.get('successful_conditions', 0),
            'papers_collected': collection.get('total_papers_collected', 0),
            'papers_analyzed': analysis.get('papers_processed', 0),
            'correlations_found': analysis.get('correlations_extracted', 0),
            'overall_success': not error and collection.get('total_papers_collected', 0) > 0
        }
        
        self.results['final_results'] = final_results
        return final_results
    
    def print_pipeline_summary(self):
        """Print a comprehensive pipeline summary."""
        final_results = self.results.get('final_results', {})
        
        print("\n" + "=" * 80)
        print("ENHANCED RESEARCH PIPELINE SUMMARY")
        print("=" * 80)
        
        # Overall status
        status = final_results.get('pipeline_status', 'unknown')
        duration = final_results.get('formatted_duration', 'unknown')
        print(f"\nStatus: {status.upper()}")
        print(f"Total Duration: {duration}")
        
        # Summary statistics  
        summary = final_results.get('summary', {})
        print(f"\nSUMMARY STATISTICS:")
        print(f"  Conditions Processed: {summary.get('conditions_processed', 0)}")
        print(f"  Papers Collected: {summary.get('papers_collected', 0)}")
        print(f"  Papers Analyzed: {summary.get('papers_analyzed', 0)}")
        print(f"  Correlations Found: {summary.get('correlations_found', 0)}")
        
        # Stage details
        stages = final_results.get('stages', {})
        
        if 'data_collection' in stages:
            dc = stages['data_collection']
            print(f"\nDATA COLLECTION:")
            print(f"  Duration: {format_duration(dc.get('stage_duration', 0))}")
            print(f"  Success Rate: {dc.get('successful_conditions', 0)}/{dc.get('conditions_processed', 0)} conditions")
        
        if 'correlation_analysis' in stages:
            ca = stages['correlation_analysis'] 
            print(f"\nCORRELATION ANALYSIS:")
            print(f"  Duration: {format_duration(ca.get('stage_duration', 0))}")
            print(f"  Success Rate: {ca.get('success_rate', 0):.1f}%")
            
            token_usage = ca.get('token_usage', {})
            if token_usage:
                print(f"  Token Usage: {token_usage.get('total_tokens', 0):,}")
        
        # Research insights
        insights = final_results.get('research_insights', {})
        if insights and 'database_overview' in insights:
            overview = insights['database_overview']
            print(f"\nDATABASE OVERVIEW:")
            print(f"  Total Papers: {overview.get('total_papers', 0):,}")
            print(f"  Total Correlations: {overview.get('total_correlations', 0):,}")
            print(f"  Papers with Fulltext: {overview.get('papers_with_fulltext', 0):,}")
        
        print("\n" + "=" * 80)