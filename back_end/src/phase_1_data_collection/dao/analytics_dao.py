"""
AnalyticsDAO - Data access object for analytics and statistics.

Handles data mining queries, statistics, and complex analytical operations.
"""

from typing import Dict, Any
from back_end.src.data.config import setup_logging
from .base_dao import BaseDAO

logger = setup_logging(__name__, 'database.log')


class AnalyticsDAO(BaseDAO):
    """Data Access Object for analytics and statistics."""

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics for intervention-focused system."""
        stats = {}

        # Basic counts
        stats['total_papers'] = self.execute_single('SELECT COUNT(*) as count FROM papers')['count']
        stats['total_interventions'] = self.execute_single('SELECT COUNT(*) as count FROM interventions')['count']

        # Processing status breakdown
        processing_status_rows = self.execute_query('''
            SELECT processing_status, COUNT(*) as count
            FROM papers
            GROUP BY processing_status
        ''')
        stats['processing_status'] = {row['processing_status']: row['count'] for row in processing_status_rows}

        # Validation status breakdown for interventions (if column exists)
        if self.column_exists('interventions', 'validation_status'):
            validation_status_rows = self.execute_query('''
                SELECT validation_status, COUNT(*) as count
                FROM interventions
                GROUP BY validation_status
            ''')
            stats['validation_status'] = {row['validation_status']: row['count'] for row in validation_status_rows}

        # Intervention category breakdown
        category_rows = self.execute_query('''
            SELECT intervention_category, COUNT(*) as count
            FROM interventions
            GROUP BY intervention_category
            ORDER BY count DESC
        ''')
        stats['intervention_categories'] = {row['intervention_category']: row['count'] for row in category_rows}

        # Date range
        date_result = self.execute_single('''
            SELECT MIN(publication_date) as min_date, MAX(publication_date) as max_date
            FROM papers
            WHERE publication_date IS NOT NULL AND publication_date != ''
        ''')
        stats['date_range'] = f"{date_result['min_date']} to {date_result['max_date']}" if date_result and date_result['min_date'] else "No papers yet"

        # Fulltext availability
        stats['papers_with_fulltext'] = self.execute_single('SELECT COUNT(*) as count FROM papers WHERE has_fulltext = TRUE')['count']

        # Top extraction models
        model_rows = self.execute_query('''
            SELECT extraction_model, COUNT(*) as count
            FROM interventions
            GROUP BY extraction_model
            ORDER BY count DESC
            LIMIT 5
        ''')
        stats['top_extraction_models'] = [
            {'model': row['extraction_model'], 'interventions': row['count']}
            for row in model_rows
        ]

        # Top health conditions
        condition_rows = self.execute_query('''
            SELECT health_condition, COUNT(*) as count
            FROM interventions
            GROUP BY health_condition
            ORDER BY count DESC
            LIMIT 10
        ''')
        stats['top_health_conditions'] = [
            {'condition': row['health_condition'], 'interventions': row['count']}
            for row in condition_rows
        ]

        # Data mining stats (if tables exist)
        if self.table_exists('knowledge_graph_nodes'):
            try:
                stats['knowledge_graph_nodes'] = self.execute_single('SELECT COUNT(*) as count FROM knowledge_graph_nodes')['count']
                stats['knowledge_graph_edges'] = self.execute_single('SELECT COUNT(*) as count FROM knowledge_graph_edges')['count']
                stats['bayesian_analyses'] = self.execute_single('SELECT COUNT(*) as count FROM bayesian_scores')['count']
                stats['treatment_recommendations'] = self.execute_single('SELECT COUNT(*) as count FROM treatment_recommendations')['count']
                stats['research_gaps'] = self.execute_single('SELECT COUNT(*) as count FROM research_gaps')['count']
                stats['data_mining_sessions'] = self.execute_single('SELECT COUNT(*) as count FROM data_mining_sessions')['count']
            except Exception as e:
                logger.warning(f"Error getting data mining stats: {e}")
                stats['data_mining_error'] = str(e)
        else:
            stats['data_mining_tables'] = 'not_available'

        return stats
