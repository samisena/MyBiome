"""
Pipeline analysis service for generating research insights.
Extracts complex analysis logic from the main pipeline.
"""

from typing import Dict, List, Any
from back_end.src.data.config import setup_logging
from back_end.src.data.repositories import repository_manager
from back_end.src.llm_processing.emerging_category_analyzer import emerging_category_analyzer

logger = setup_logging(__name__, 'pipeline_analyzer.log')


class PipelineAnalyzer:
    """
    Service for analyzing pipeline results and generating research insights.
    Separates analysis logic from pipeline orchestration.
    """

    def __init__(self, repository_mgr=None):
        """
        Initialize analyzer with repository manager.

        Args:
            repository_mgr: Repository manager instance (optional, uses global if None)
        """
        self.repository_mgr = repository_mgr or repository_manager

    def generate_research_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive research insights from interventions data.

        Returns:
            Research insights summary
        """
        # Generating insights (logging removed for performance)

        try:
            # Get database statistics
            db_stats = self._get_database_stats()

            insights = {
                'database_overview': db_stats,
                'intervention_coverage': self._analyze_intervention_coverage(),
                'top_findings': self._get_top_intervention_findings(),
                'data_quality': self._assess_intervention_data_quality(),
                'emerging_categories': self._analyze_emerging_categories()
            }

            return insights

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {'error': str(e)}

    def _get_database_stats(self) -> Dict[str, Any]:
        """Get basic database statistics."""
        try:
            with self.repository_mgr.get_connection() as conn:
                cursor = conn.cursor()

                # Paper statistics
                cursor.execute('''
                    SELECT
                        COUNT(*) as total_papers,
                        COUNT(CASE WHEN abstract IS NOT NULL AND abstract != '' THEN 1 END) as papers_with_abstracts,
                        COUNT(CASE WHEN has_fulltext = TRUE THEN 1 END) as papers_with_fulltext
                    FROM papers
                ''')
                paper_stats = cursor.fetchone()

                # Intervention statistics
                cursor.execute('''
                    SELECT
                        COUNT(*) as total_interventions,
                        COUNT(DISTINCT intervention_category) as unique_categories,
                        COUNT(DISTINCT health_condition) as unique_conditions,
                        COUNT(DISTINCT paper_id) as papers_with_interventions
                    FROM interventions
                ''')
                intervention_stats = cursor.fetchone()

                return {
                    'total_papers': paper_stats[0] if paper_stats else 0,
                    'papers_with_abstracts': paper_stats[1] if paper_stats else 0,
                    'papers_with_fulltext': paper_stats[2] if paper_stats else 0,
                    'total_interventions': intervention_stats[0] if intervention_stats else 0,
                    'unique_categories': intervention_stats[1] if intervention_stats else 0,
                    'unique_conditions': intervention_stats[2] if intervention_stats else 0,
                    'papers_with_interventions': intervention_stats[3] if intervention_stats else 0
                }

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

    def _analyze_intervention_coverage(self) -> Dict[str, Any]:
        """Analyze intervention coverage across categories and conditions."""
        try:
            with self.repository_mgr.get_connection() as conn:
                cursor = conn.cursor()

                # Top intervention categories
                cursor.execute('''
                    SELECT intervention_category, COUNT(DISTINCT paper_id) as paper_count,
                           COUNT(*) as intervention_count
                    FROM interventions
                    GROUP BY intervention_category
                    ORDER BY paper_count DESC, intervention_count DESC
                    LIMIT 10
                ''')
                top_categories = [
                    {'category': row[0], 'papers': row[1], 'interventions': row[2]}
                    for row in cursor.fetchall()
                ]

                # Top studied conditions
                cursor.execute('''
                    SELECT health_condition, COUNT(DISTINCT paper_id) as paper_count,
                           COUNT(*) as intervention_count
                    FROM interventions
                    GROUP BY health_condition
                    ORDER BY paper_count DESC, intervention_count DESC
                    LIMIT 10
                ''')
                top_conditions = [
                    {'condition': row[0], 'papers': row[1], 'interventions': row[2]}
                    for row in cursor.fetchall()
                ]

                return {
                    'top_categories': top_categories,
                    'top_conditions': top_conditions
                }

        except Exception as e:
            logger.error(f"Error analyzing intervention coverage: {e}")
            return {}

    def _get_top_intervention_findings(self) -> Dict[str, Any]:
        """Get top intervention findings with strongest evidence."""
        try:
            with self.repository_mgr.get_connection() as conn:
                cursor = conn.cursor()

                # Strongest positive interventions
                cursor.execute('''
                    SELECT intervention_name, health_condition,
                           COUNT(DISTINCT paper_id) as study_count,
                           AVG(correlation_strength) as avg_strength,
                           AVG(confidence_score) as avg_confidence
                    FROM interventions
                    WHERE correlation_type = 'positive'
                      AND correlation_strength IS NOT NULL
                      AND confidence_score IS NOT NULL
                    GROUP BY intervention_name, health_condition
                    HAVING study_count >= 2
                    ORDER BY avg_strength DESC, avg_confidence DESC, study_count DESC
                    LIMIT 10
                ''')

                top_positive = [
                    {
                        'intervention': row[0],
                        'condition': row[1],
                        'studies': row[2],
                        'avg_strength': round(row[3], 3) if row[3] else None,
                        'avg_confidence': round(row[4], 3) if row[4] else None
                    }
                    for row in cursor.fetchall()
                ]

                return {'strongest_positive_interventions': top_positive}

        except Exception as e:
            logger.error(f"Error getting top findings: {e}")
            return {}

    def _assess_intervention_data_quality(self) -> Dict[str, Any]:
        """Assess overall intervention data quality metrics."""
        try:
            with self.repository_mgr.get_connection() as conn:
                cursor = conn.cursor()

                # Intervention completeness
                cursor.execute('''
                    SELECT
                        COUNT(*) as total_interventions,
                        COUNT(CASE WHEN correlation_strength IS NOT NULL THEN 1 END) as with_strength,
                        COUNT(CASE WHEN confidence_score IS NOT NULL THEN 1 END) as with_confidence,
                        COUNT(CASE WHEN supporting_quote IS NOT NULL AND supporting_quote != '' THEN 1 END) as with_quotes,
                        COUNT(CASE WHEN intervention_details IS NOT NULL THEN 1 END) as with_details
                    FROM interventions
                ''')
                intervention_stats = cursor.fetchone()

                if intervention_stats and intervention_stats[0] > 0:
                    total = intervention_stats[0]
                    quality_metrics = {
                        'intervention_completeness': {
                            'with_strength': (intervention_stats[1] / total * 100),
                            'with_confidence': (intervention_stats[2] / total * 100),
                            'with_supporting_quotes': (intervention_stats[3] / total * 100),
                            'with_details': (intervention_stats[4] / total * 100)
                        }
                    }
                else:
                    quality_metrics = {
                        'intervention_completeness': {
                            'with_strength': 0,
                            'with_confidence': 0,
                            'with_supporting_quotes': 0,
                            'with_details': 0
                        }
                    }

                return quality_metrics

        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return {}

    def _analyze_emerging_categories(self) -> Dict[str, Any]:
        """Analyze emerging interventions for new category patterns."""
        try:
            # Analyzing emerging categories (logging removed for performance)

            # Run emerging category analysis
            candidates = emerging_category_analyzer.analyze_emerging_interventions(
                min_intervention_count=2,  # Lower threshold for exploration
                min_unique_papers=2
            )

            # Generate analysis report
            report = emerging_category_analyzer.generate_analysis_report(candidates)

            return {
                'analysis_completed': True,
                'candidates_found': len(candidates),
                'high_confidence_candidates': report.get('high_confidence_candidates', 0),
                'recommendations': report.get('recommendations', []),
                'top_candidates': report.get('detailed_candidates', [])[:5]  # Top 5 only
            }

        except Exception as e:
            logger.error(f"Error analyzing emerging categories: {e}")
            return {
                'analysis_completed': False,
                'error': str(e),
                'candidates_found': 0
            }