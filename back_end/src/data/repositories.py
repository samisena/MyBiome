"""
Repository classes for standardized database interactions.
Implements repository pattern to abstract database operations.
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import sys
from pathlib import Path

from src.data.config import setup_logging
from src.paper_collection.database_manager import database_manager

logger = setup_logging(__name__, 'repositories.log')


class BaseRepository(ABC):
    """Abstract base class for repository pattern."""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager or database_manager
    
    @abstractmethod
    def get_by_id(self, entity_id: str) -> Optional[Dict]:
        """Get entity by ID."""
        pass


class PaperRepository(BaseRepository):
    """Repository for paper-related database operations."""
    
    def get_by_id(self, pmid: str) -> Optional[Dict]:
        """Get paper by PMID."""
        return self.db_manager.get_paper_by_pmid(pmid)
    
    def get_unprocessed_papers(self, extraction_model: str, limit: Optional[int] = None) -> List[Dict]:
        """Get papers that need processing."""
        return self.db_manager.get_papers_for_processing(extraction_model, limit)
    
    def get_all_papers(self, limit: Optional[int] = None, 
                      processing_status: Optional[str] = None) -> List[Dict]:
        """Get all papers with optional filtering."""
        return self.db_manager.get_all_papers(limit, processing_status)
    
    def insert_paper(self, paper: Dict) -> bool:
        """Insert a new paper."""
        return self.db_manager.insert_paper(paper)
    
    def insert_papers_batch(self, papers: List[Dict]) -> tuple[int, int]:
        """Insert multiple papers in batch."""
        return self.db_manager.insert_papers_batch(papers)
    
    def update_processing_status(self, pmid: str, status: str) -> bool:
        """Update paper processing status."""
        return self.db_manager.update_paper_processing_status(pmid, status)
    
    def update_fulltext_status(self, pmid: str, has_fulltext: bool, 
                              fulltext_path: Optional[str] = None) -> bool:
        """Update paper fulltext availability."""
        return self.db_manager.update_paper_fulltext_status(pmid, has_fulltext, fulltext_path)


class InterventionRepository(BaseRepository):
    """Repository for intervention-related database operations."""
    
    def get_by_id(self, intervention_id: str) -> Optional[Dict]:
        """Get intervention by ID."""
        # Note: interventions don't have single ID field, using paper_id + intervention_name
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM interventions 
                WHERE id = ?
            ''', (intervention_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_by_paper(self, pmid: str) -> List[Dict]:
        """Get all interventions for a paper."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM interventions WHERE paper_id = ?', (pmid,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_by_category(self, category: str, limit: Optional[int] = None) -> List[Dict]:
        """Get interventions by category."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM interventions WHERE intervention_category = ?'
            params = [category]
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
                
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def insert_intervention(self, intervention: Dict) -> bool:
        """Insert a new intervention."""
        return self.db_manager.insert_intervention(intervention)
    
    def insert_interventions_batch(self, interventions: List[Dict]) -> int:
        """Insert multiple interventions in batch."""
        successful = 0
        for intervention in interventions:
            if self.insert_intervention(intervention):
                successful += 1
        return successful
    
    def get_interventions_by_condition(self, condition: str, limit: Optional[int] = None) -> List[Dict]:
        """Get interventions by health condition."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM interventions WHERE health_condition LIKE ?'
            params = [f'%{condition}%']
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
                
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


class StatisticsRepository(BaseRepository):
    """Repository for database statistics and analytics."""
    
    def get_by_id(self, stat_id: str) -> Optional[Dict]:
        """Not applicable for statistics."""
        return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        return self.db_manager.get_database_stats()
    
    def get_intervention_stats(self) -> Dict[str, Any]:
        """Get intervention-specific statistics."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count by category
            cursor.execute('''
                SELECT intervention_category, COUNT(*) as count
                FROM interventions 
                GROUP BY intervention_category
                ORDER BY count DESC
            ''')
            categories = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Count by correlation type
            cursor.execute('''
                SELECT correlation_type, COUNT(*) as count
                FROM interventions 
                WHERE correlation_type IS NOT NULL
                GROUP BY correlation_type
                ORDER BY count DESC
            ''')
            correlations = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Count by model
            cursor.execute('''
                SELECT extraction_model, COUNT(*) as count
                FROM interventions 
                GROUP BY extraction_model
                ORDER BY count DESC
            ''')
            models = {row[0]: row[1] for row in cursor.fetchall()}
            
            return {
                'categories': categories,
                'correlation_types': correlations,
                'extraction_models': models,
                'total_interventions': sum(categories.values())
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get paper processing statistics."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Processing status counts
            cursor.execute('''
                SELECT processing_status, COUNT(*) as count
                FROM papers 
                GROUP BY processing_status
                ORDER BY count DESC
            ''')
            processing = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Papers with fulltext
            cursor.execute('SELECT COUNT(*) FROM papers WHERE has_fulltext = 1')
            fulltext_count = cursor.fetchone()[0]
            
            # Total papers
            cursor.execute('SELECT COUNT(*) FROM papers')
            total_papers = cursor.fetchone()[0]
            
            return {
                'processing_status': processing,
                'total_papers': total_papers,
                'papers_with_fulltext': fulltext_count,
                'fulltext_percentage': (fulltext_count / total_papers * 100) if total_papers > 0 else 0
            }


class RepositoryManager:
    """Manager class that provides access to all repositories."""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager or database_manager
        self._papers = None
        self._interventions = None
        self._statistics = None
    
    @property
    def papers(self) -> PaperRepository:
        """Get paper repository."""
        if self._papers is None:
            self._papers = PaperRepository(self.db_manager)
        return self._papers
    
    @property
    def interventions(self) -> InterventionRepository:
        """Get intervention repository."""
        if self._interventions is None:
            self._interventions = InterventionRepository(self.db_manager)
        return self._interventions
    
    @property
    def statistics(self) -> StatisticsRepository:
        """Get statistics repository."""
        if self._statistics is None:
            self._statistics = StatisticsRepository(self.db_manager)
        return self._statistics
    
    def get_connection(self):
        """Get database connection (for custom queries)."""
        return self.db_manager.get_connection()


# Global repository manager instance
repository_manager = RepositoryManager()