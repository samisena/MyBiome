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
    This class handles creating tables, inserting papers and authors,
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
        """Creates all necessary database tables with proper relationships."""
        
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
            
            # Create the Authors table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS authors (
                    author_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    last_name TEXT NOT NULL,
                    first_name TEXT,
                    initials TEXT,
                    affiliations TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(last_name, first_name, initials, affiliations)
                )
            ''')
            
            # Junction table for papers-authors relationship
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers_authors (
                    paper_pmid TEXT,
                    author_id INTEGER,
                    author_order INTEGER,
                    PRIMARY KEY (paper_pmid, author_id),
                    FOREIGN KEY (paper_pmid) REFERENCES papers(pmid),
                    FOREIGN KEY (author_id) REFERENCES authors(author_id)
                )
            ''')
            
            # Search history table 
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    search_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strain TEXT,
                    condition TEXT,
                    query TEXT,
                    result_count INTEGER,
                    search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Search results junction table (links searches to papers)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_results (
                    search_id INTEGER,
                    paper_pmid TEXT,
                    PRIMARY KEY (search_id, paper_pmid),
                    FOREIGN KEY (search_id) REFERENCES search_history(search_id),
                    FOREIGN KEY (paper_pmid) REFERENCES papers(pmid)
                )
            ''')
            
            # Create indexes for faster querying
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_papers_date 
                ON papers(publication_date)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_authors_name
                ON authors(last_name, first_name)
            ''')
            
            conn.commit()
            self.logger.info(f"Database tables created at {self.db_path}")
    
    def insert_papers(self, paper: Paper, search_id: Optional[int] = None) -> bool:
        """Inserts a paper and its authors into the database.
        
        Args:
            paper: Paper object containing all paper details
            search_id: Optional ID linking this paper to a search
            
        Returns:
            True if paper was newly inserted, False if it already existed
        """
        
        with self.get_connection() as conn:  # Fixed: added parentheses
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
                
                # Add each author
                for order, author in enumerate(paper.authors):
                    # Insert author (or ignore if exists)
                    cursor.execute('''
                        INSERT OR IGNORE INTO authors
                        (last_name, first_name, initials, affiliations)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        author.last_name,
                        author.first_name,
                        author.initials,  # Fixed: was 'intials'
                        author.affiliations
                    ))
                    
                    # Get the author's ID
                    cursor.execute('''
                        SELECT author_id FROM authors
                        WHERE last_name = ? AND first_name = ?
                        AND initials = ? AND affiliations IS ?
                    ''', (
                        author.last_name,
                        author.first_name,
                        author.initials,  # Fixed: was 'intials'
                        author.affiliations
                    ))
                    
                    author_id = cursor.fetchone()[0]  # Fixed: was 'fetchtone'
                    
                    # Link paper to author
                    cursor.execute('''
                        INSERT OR IGNORE INTO papers_authors
                        (paper_pmid, author_id, author_order)
                        VALUES (?, ?, ?)
                    ''', (paper.pmid, author_id, order))
                
                # Link to search if search_id provided
                if search_id is not None:
                    cursor.execute('''
                        INSERT OR IGNORE INTO search_results
                        (search_id, paper_pmid)
                        VALUES (?, ?)
                    ''', (search_id, paper.pmid))
                
                conn.commit()
                
                if was_new_paper:  # Fixed: removed parentheses
                    self.logger.info(f"Inserted new paper: {paper.title}")
                else:
                    self.logger.debug(f"Paper already exists: {paper.title}")
                    
                return was_new_paper
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error inserting paper {paper.pmid}: {e}")
                raise
    
    def record_search(self, strain: str, condition: str, query: str, 
                      result_count: int) -> int:
        """Records a search in the search_history table.
        
        Args:
            strain: Probiotic strain searched for
            condition: Health condition searched for
            query: Full search query used
            result_count: Number of papers found
            
        Returns:
            The search_id of the recorded search
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO search_history (strain, condition, query, result_count)
                VALUES (?, ?, ?, ?)
            ''', (strain, condition, query, result_count))
            conn.commit()
            return cursor.lastrowid
    
    def get_papers_by_strain(self, strain: str) -> List[Dict]:
        """Retrieves all papers related to a specific strain.
        
        Args:
            strain: The probiotic strain to search for
            
        Returns:
            List of dictionaries containing paper details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT p.* FROM papers p
                JOIN search_results sr ON p.pmid = sr.paper_pmid
                JOIN search_history sh ON sr.search_id = sh.search_id
                WHERE sh.strain = ?
                ORDER BY p.publication_date DESC
            ''', (strain,))  # Fixed: added comma to make it a tuple
            return [dict(row) for row in cursor.fetchall()]
    
    def get_papers_by_author(self, last_name: str, 
                           first_name: Optional[str] = None) -> List[Dict]:
        """Finds all papers by a specific author.
        
        Args:
            last_name: Author's last name
            first_name: Optional first name for more specific search
            
        Returns:
            List of dictionaries containing paper details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if first_name:
                query = '''
                    SELECT p.* FROM papers p
                    JOIN papers_authors pa ON p.pmid = pa.paper_pmid
                    JOIN authors a ON pa.author_id = a.author_id
                    WHERE a.last_name = ? AND a.first_name = ?
                    ORDER BY p.publication_date DESC
                '''
                params = (last_name, first_name)
            else:
                query = '''
                    SELECT p.* FROM papers p
                    JOIN papers_authors pa ON p.pmid = pa.paper_pmid
                    JOIN authors a ON pa.author_id = a.author_id
                    WHERE a.last_name = ?
                    ORDER BY p.publication_date DESC
                '''
                params = (last_name,)
                
            cursor.execute(query, params)
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
            stats['total_papers'] = cursor.fetchone()[0]  # Fixed: was conn.fetchone()
            
            # Count total authors
            cursor.execute('SELECT COUNT(*) FROM authors')
            stats['total_authors'] = cursor.fetchone()[0]  # Fixed: was conn.fetchone()
            
            # Count total searches
            cursor.execute('SELECT COUNT(*) FROM search_history')
            stats['total_searches'] = cursor.fetchone()[0]  # Fixed: was conn.fetchone()
            
            # Get date range of publications
            cursor.execute('''
                SELECT MIN(publication_date), MAX(publication_date)
                FROM papers
            ''')
            min_date, max_date = cursor.fetchone()
            stats['date_range'] = f"{min_date} to {max_date}" if min_date else "No papers yet"
            
            return stats
            
                
                    
                    
