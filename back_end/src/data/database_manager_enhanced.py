"""
Enhanced database manager with connection pooling and dependency injection support.
This replaces the original database_manager.py with improved efficiency and architecture.
"""

import sqlite3
import json
import threading
from contextlib import contextmanager
from typing import List, Dict, Optional, Any
from datetime import datetime
from queue import Queue, Empty
from dataclasses import dataclass
import sys
from pathlib import Path

# Add the current directory to sys.path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from config import config, setup_logging
from utils import validate_paper_data, validate_correlation_data, ValidationError

logger = setup_logging(__name__, 'database.log')


@dataclass
class ConnectionPool:
    """Simple connection pool for SQLite database."""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._pool = Queue(maxsize=max_connections)
        self._lock = threading.Lock()
        self._total_connections = 0
        
        # Pre-populate pool with initial connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool with some connections."""
        for _ in range(min(3, self.max_connections)):  # Start with 3 connections
            conn = self._create_connection()
            self._pool.put(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key constraints
        conn.execute('PRAGMA journal_mode = WAL')  # Enable WAL mode for better concurrency
        self._total_connections += 1
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = None
        try:
            # Try to get existing connection from pool
            try:
                conn = self._pool.get_nowait()
            except Empty:
                # If pool is empty and we haven't reached max connections, create new one
                with self._lock:
                    if self._total_connections < self.max_connections:
                        conn = self._create_connection()
                    else:
                        # Wait for a connection to be available
                        conn = self._pool.get(timeout=30)
            
            yield conn
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                try:
                    # Return connection to pool
                    self._pool.put_nowait(conn)
                except:
                    # Pool is full, close the connection
                    conn.close()
                    with self._lock:
                        self._total_connections -= 1
    
    def close_all(self):
        """Close all connections in the pool."""
        while True:
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
        
        with self._lock:
            self._total_connections = 0


class EnhancedDatabaseManager:
    """
    Enhanced database manager with connection pooling, better error handling,
    and validation. Designed to be used as a singleton via dependency injection.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_config=None):
        """Singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_config=None):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
            
        self.db_config = db_config or config.database
        self.db_path = self.db_config.path
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection pool
        self.pool = ConnectionPool(str(self.db_path), self.db_config.max_connections)
        
        # Create tables
        self.create_tables()
        
        logger.info(f"Enhanced database manager initialized at {self.db_path}")
        self._initialized = True
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        with self.pool.get_connection() as conn:
            yield conn
    
    def create_tables(self):
        """Create all necessary database tables with optimized schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Papers table with enhanced schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    pmid TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    abstract TEXT,
                    journal TEXT,
                    publication_date TEXT,
                    doi TEXT,
                    pmc_id TEXT,
                    keywords TEXT,  -- JSON array
                    has_fulltext BOOLEAN DEFAULT FALSE,
                    fulltext_source TEXT,
                    fulltext_path TEXT,
                    processing_status TEXT DEFAULT 'pending',  -- pending, processed, failed
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Correlations table with enhanced validation tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS correlations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    probiotic_strain TEXT NOT NULL,
                    health_condition TEXT NOT NULL,
                    correlation_type TEXT CHECK(correlation_type IN ('positive', 'negative', 'neutral', 'inconclusive')),
                    correlation_strength REAL CHECK(correlation_strength >= 0 AND correlation_strength <= 1),
                    effect_size TEXT,
                    sample_size INTEGER,
                    study_duration TEXT,
                    study_type TEXT,
                    dosage TEXT,
                    population_details TEXT,
                    confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
                    supporting_quote TEXT,
                    
                    -- Extraction tracking
                    extraction_model TEXT NOT NULL,
                    extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Validation tracking
                    validation_status TEXT DEFAULT 'pending' CHECK(validation_status IN ('pending', 'verified', 'conflicted', 'failed')),
                    validation_issues TEXT,
                    verification_model TEXT,
                    verification_timestamp TIMESTAMP,
                    verification_confidence REAL CHECK(verification_confidence >= 0 AND verification_confidence <= 1),
                    human_reviewed BOOLEAN DEFAULT FALSE,
                    human_reviewer TEXT,
                    review_timestamp TIMESTAMP,
                    review_notes TEXT,
                    
                    FOREIGN KEY (paper_id) REFERENCES papers(pmid) ON DELETE CASCADE,
                    UNIQUE(paper_id, probiotic_strain, health_condition, extraction_model)
                )
            ''')
            
            # Create optimized indexes
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(publication_date)',
                'CREATE INDEX IF NOT EXISTS idx_papers_journal ON papers(journal)',
                'CREATE INDEX IF NOT EXISTS idx_papers_fulltext ON papers(has_fulltext)',
                'CREATE INDEX IF NOT EXISTS idx_papers_processing ON papers(processing_status)',
                
                'CREATE INDEX IF NOT EXISTS idx_correlations_paper ON correlations(paper_id)',
                'CREATE INDEX IF NOT EXISTS idx_correlations_strain ON correlations(probiotic_strain)',
                'CREATE INDEX IF NOT EXISTS idx_correlations_condition ON correlations(health_condition)',
                'CREATE INDEX IF NOT EXISTS idx_correlations_type ON correlations(correlation_type)',
                'CREATE INDEX IF NOT EXISTS idx_correlations_validation ON correlations(validation_status)',
                'CREATE INDEX IF NOT EXISTS idx_correlations_model ON correlations(extraction_model)',
                
                # Composite indexes for common queries
                'CREATE INDEX IF NOT EXISTS idx_strain_condition ON correlations(probiotic_strain, health_condition)',
                'CREATE INDEX IF NOT EXISTS idx_paper_model ON correlations(paper_id, extraction_model)',
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except sqlite3.OperationalError as e:
                    if "no such column" in str(e):
                        logger.warning(f"Skipping index creation due to missing column: {e}")
                        # This means we need to add the missing column first
                        if "processing_status" in index_sql:
                            # Add the processing_status column if it's missing
                            try:
                                cursor.execute('ALTER TABLE papers ADD COLUMN processing_status TEXT DEFAULT "pending"')
                                logger.info("Added missing processing_status column to papers table")
                                # Now retry the index creation
                                cursor.execute(index_sql)
                            except sqlite3.OperationalError as e2:
                                if "duplicate column name" not in str(e2):
                                    logger.error(f"Failed to add processing_status column: {e2}")
                    else:
                        raise
            
            # Add triggers for updated_at timestamps
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS update_papers_timestamp 
                AFTER UPDATE ON papers
                BEGIN
                    UPDATE papers SET updated_at = CURRENT_TIMESTAMP WHERE pmid = NEW.pmid;
                END
            ''')
            
            conn.commit()
            logger.info("Database tables and indexes created successfully")
    
    def insert_paper(self, paper: Dict) -> bool:
        """Insert a paper with validation and enhanced error handling."""
        try:
            # Validate paper data
            validated_paper = validate_paper_data(paper)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR IGNORE INTO papers
                    (pmid, title, abstract, journal, publication_date, doi, pmc_id, 
                     keywords, has_fulltext, fulltext_source, fulltext_path, processing_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validated_paper['pmid'],
                    validated_paper['title'],
                    validated_paper['abstract'],
                    validated_paper['journal'],
                    validated_paper['publication_date'],
                    validated_paper['doi'],
                    validated_paper['pmc_id'],
                    json.dumps(validated_paper['keywords']) if validated_paper['keywords'] else None,
                    validated_paper['has_fulltext'],
                    validated_paper['fulltext_source'],
                    validated_paper['fulltext_path'],
                    'pending'
                ))
                
                was_new = cursor.rowcount > 0
                conn.commit()
                
                if was_new:
                    logger.info(f"Inserted new paper: {validated_paper['pmid']}")
                else:
                    logger.debug(f"Paper already exists: {validated_paper['pmid']}")
                
                return was_new
                
        except ValidationError as e:
            logger.error(f"Validation error for paper {paper.get('pmid', 'unknown')}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inserting paper {paper.get('pmid', 'unknown')}: {e}")
            return False
    
    def insert_correlation(self, correlation: Dict) -> bool:
        """Insert a correlation with validation."""
        try:
            # Validate correlation data
            validated_corr = validate_correlation_data(correlation)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO correlations
                    (paper_id, probiotic_strain, health_condition, correlation_type,
                     correlation_strength, effect_size, sample_size, study_duration,
                     study_type, dosage, population_details, confidence_score,
                     supporting_quote, extraction_model, validation_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validated_corr['paper_id'],
                    validated_corr['probiotic_strain'],
                    validated_corr['health_condition'],
                    validated_corr['correlation_type'],
                    validated_corr.get('correlation_strength'),
                    validated_corr.get('effect_size'),
                    validated_corr.get('sample_size'),
                    validated_corr.get('study_duration'),
                    validated_corr.get('study_type'),
                    validated_corr.get('dosage'),
                    validated_corr.get('population_details'),
                    validated_corr.get('confidence_score'),
                    validated_corr.get('supporting_quote'),
                    validated_corr['extraction_model'],
                    'pending'
                ))
                
                was_new = cursor.rowcount > 0
                conn.commit()
                
                if was_new:
                    logger.info(f"Inserted correlation: {validated_corr['probiotic_strain']} - {validated_corr['health_condition']}")
                
                return was_new
                
        except ValidationError as e:
            logger.error(f"Validation error for correlation: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inserting correlation: {e}")
            return False
    
    def get_paper_by_pmid(self, pmid: str) -> Optional[Dict]:
        """Get a paper by PMID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM papers WHERE pmid = ?', (pmid,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_papers(self, limit: Optional[int] = None, 
                      processing_status: Optional[str] = None) -> List[Dict]:
        """Get all papers with optional filtering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM papers'
            params = []
            
            if processing_status:
                query += ' WHERE processing_status = ?'
                params.append(processing_status)
            
            query += ' ORDER BY publication_date DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_papers_for_processing(self, extraction_model: str, 
                                limit: Optional[int] = None) -> List[Dict]:
        """Get papers that need processing by a specific model."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT p.*
                FROM papers p
                LEFT JOIN correlations c ON p.pmid = c.paper_id AND c.extraction_model = ?
                WHERE c.id IS NULL 
                  AND p.abstract IS NOT NULL 
                  AND p.abstract != ''
                  AND p.processing_status != 'failed'
                ORDER BY p.publication_date DESC
            '''
            
            params = [extraction_model]
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def update_paper_processing_status(self, pmid: str, status: str) -> bool:
        """Update paper processing status."""
        valid_statuses = ['pending', 'processing', 'processed', 'failed']
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE papers 
                SET processing_status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE pmid = ?
            ''', (status, pmid))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_papers_by_condition(self, condition: str, limit: Optional[int] = None) -> List[Dict]:
        """Get papers related to a specific health condition."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Search for papers containing the condition in title, abstract, or keywords
            query = '''
                SELECT * FROM papers 
                WHERE (LOWER(title) LIKE LOWER(?) 
                       OR LOWER(abstract) LIKE LOWER(?) 
                       OR LOWER(keywords) LIKE LOWER(?))
                  AND abstract IS NOT NULL 
                  AND abstract != ''
                ORDER BY publication_date DESC
            '''
            
            condition_pattern = f"%{condition}%"
            params = [condition_pattern, condition_pattern, condition_pattern]
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Basic counts
            cursor.execute('SELECT COUNT(*) FROM papers')
            stats['total_papers'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM correlations')
            stats['total_correlations'] = cursor.fetchone()[0]
            
            # Processing status breakdown
            cursor.execute('''
                SELECT processing_status, COUNT(*) 
                FROM papers 
                GROUP BY processing_status
            ''')
            stats['processing_status'] = dict(cursor.fetchall())
            
            # Validation status breakdown
            cursor.execute('''
                SELECT validation_status, COUNT(*) 
                FROM correlations 
                GROUP BY validation_status
            ''')
            stats['validation_status'] = dict(cursor.fetchall())
            
            # Date range
            cursor.execute('''
                SELECT MIN(publication_date), MAX(publication_date)
                FROM papers
                WHERE publication_date IS NOT NULL AND publication_date != ''
            ''')
            date_result = cursor.fetchone()
            stats['date_range'] = f"{date_result[0]} to {date_result[1]}" if date_result[0] else "No papers yet"
            
            # Fulltext availability
            cursor.execute('SELECT COUNT(*) FROM papers WHERE has_fulltext = TRUE')
            stats['papers_with_fulltext'] = cursor.fetchone()[0]
            
            # Top extraction models
            cursor.execute('''
                SELECT extraction_model, COUNT(*) as count
                FROM correlations
                GROUP BY extraction_model
                ORDER BY count DESC
                LIMIT 5
            ''')
            stats['top_extraction_models'] = [
                {'model': row[0], 'correlations': row[1]} 
                for row in cursor.fetchall()
            ]
            
            return stats
    
    def close(self):
        """Close all database connections."""
        if hasattr(self, 'pool'):
            self.pool.close_all()
            logger.info("Database connections closed")


# Global instance for dependency injection
database_manager = EnhancedDatabaseManager()