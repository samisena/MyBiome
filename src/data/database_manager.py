"""
Database management for the PubMed collection system.
This module handles all SQLite operations.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager  #? context for "with" statements
import logging

# Import the shared data models
from src.data.models import Author, Paper

# Get project root 
project_root = Path(__file__).parent.parent.parent

class DatabaseManager:
    """Manages all database operations for the PubMed research system.
    This class handles creating tables, inserting papers,
    and querying the stored data.
    """
    
    def __init__(self, db_name: str = 'pubmed_research.db'):
        """Initiates the database connection and creates tables if needed.
        
        Args:
            db_name (str, optional): Database filename. Defaults to 'pubmed_research.db'.
        """
        
        # Setting up the database path
        self.db_path = project_root / "data" / "processed" / db_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
                                                
        # Setting up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create all necessary tables                          
        self.create_tables()
        
    @contextmanager  
    def get_connection(self):
        """Context manager that handles database connections safely.
        Ensures connections are properly closed even if errors occur."""
        
        # Connect to the SQLite database
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enables column access by name
        try:
            yield conn
        finally:
            conn.close()
            
    def create_tables(self):
        """Creates all necessary database tables."""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create the Papers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    pmid TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    journal TEXT,
                    publication_date TEXT,
                    doi TEXT,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for faster querying
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_papers_date 
                ON papers(publication_date)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_papers_journal
                ON papers(journal)
            ''')
            
            conn.commit()
            self.logger.info(f"Database tables created at {self.db_path}")
    
    def insert_paper(self, paper: 'Paper') -> bool:
        """Inserts a paper into the database.
        
        Args:
            paper: Paper object containing all paper details
            
        Returns:
            True if paper was newly inserted, False if it already existed
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Insert the paper (or ignore if it already exists)
                cursor.execute('''
                    INSERT OR IGNORE INTO papers
                    (pmid, title, abstract, journal, publication_date, doi, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    paper.pmid,
                    paper.title,
                    paper.abstract,
                    paper.journal,
                    paper.publication_date,
                    paper.doi,
                    json.dumps(paper.keywords) if paper.keywords else None
                ))
                
                was_new_paper = cursor.rowcount > 0
                
                conn.commit()
                
                if was_new_paper:
                    self.logger.info(f"Inserted new paper: {paper.title}")
                else:
                    self.logger.debug(f"Paper already exists: {paper.title}")
                    
                return was_new_paper
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error inserting paper {paper.pmid}: {e}")
                raise
    
    def get_papers_by_keyword(self, keyword: str) -> List[Dict]:
        """Retrieves all papers containing a keyword in title or abstract.
        
        Args:
            keyword: The keyword to search for
            
        Returns:
            List of dictionaries containing paper details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM papers
                WHERE title LIKE ? OR abstract LIKE ?
                ORDER BY publication_date DESC
            ''', (f'%{keyword}%', f'%{keyword}%'))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_papers_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """Retrieves papers published within a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of dictionaries containing paper details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM papers
                WHERE publication_date >= ? AND publication_date <= ?
                ORDER BY publication_date DESC
            ''', (start_date, end_date))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_paper_by_pmid(self, pmid: str) -> Optional[Dict]:
        """Retrieves a specific paper by its PMID.
        
        Args:
            pmid: PubMed ID of the paper
            
        Returns:
            Dictionary containing paper details or None if not found
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM papers WHERE pmid = ?', (pmid,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_papers(self, limit: Optional[int] = None) -> List[Dict]:
        """Retrieves all papers from the database.
        
        Args:
            limit: Optional limit on number of papers to return
            
        Returns:
            List of dictionaries containing paper details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if limit:
                cursor.execute('SELECT * FROM papers ORDER BY publication_date DESC LIMIT ?', (limit,))
            else:
                cursor.execute('SELECT * FROM papers ORDER BY publication_date DESC')
            return [dict(row) for row in cursor.fetchall()]
    
    def get_database_stats(self) -> Dict:
        """Retrieves statistics about the database contents.
        
        Returns:
            Dictionary containing counts and date ranges
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count total papers
            cursor.execute('SELECT COUNT(*) FROM papers')
            stats['total_papers'] = cursor.fetchone()[0]
            
            # Get date range of publications
            cursor.execute('''
                SELECT MIN(publication_date), MAX(publication_date)
                FROM papers
            ''')
            min_date, max_date = cursor.fetchone()
            stats['date_range'] = f"{min_date} to {max_date}" if min_date else "No papers yet"
            
            # Count papers by journal
            cursor.execute('''
                SELECT journal, COUNT(*) as count
                FROM papers
                GROUP BY journal
                ORDER BY count DESC
                LIMIT 10
            ''')
            stats['top_journals'] = [(row[0], row[1]) for row in cursor.fetchall()]
            
            # Get recent papers count
            cursor.execute('''
                SELECT COUNT(*) FROM papers
                WHERE datetime(created_at) >= datetime('now', '-7 days')
            ''')
            stats['papers_added_last_week'] = cursor.fetchone()[0]
            
            return stats
    
    def search_papers(self, query: str) -> List[Dict]:
        """Full-text search across titles and abstracts.
        
        Args:
            query: Search query string
            
        Returns:
            List of dictionaries containing matching papers
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Simple LIKE search - could be enhanced with FTS5 for better performance
            cursor.execute('''
                SELECT * FROM papers
                WHERE title LIKE ? OR abstract LIKE ?
                ORDER BY 
                    CASE 
                        WHEN title LIKE ? THEN 0 
                        ELSE 1 
                    END,
                    publication_date DESC
            ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_paper(self, pmid: str) -> bool:
        """Deletes a paper from the database.
        
        Args:
            pmid: PubMed ID of the paper to delete
            
        Returns:
            True if paper was deleted, False if it didn't exist
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM papers WHERE pmid = ?', (pmid,))
            conn.commit()
            return cursor.rowcount > 0
            
                
                    
                    
