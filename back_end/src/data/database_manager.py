import sqlite3
import json
import logging
from pathlib import Path
from contextlib import contextmanager #? context for "with" statements
from typing import List, Dict, Optional
from datetime import datetime

#* Get project root 
project_root = Path(__file__).parent.parent.parent

class DatabaseManager:
    """Manages all database operations for the PubMed research system.
    This class handles creating tables, inserting papers and correlations,
    and querying the stored data.
    """
    
    def __init__(self, db_name: str = 'pubmed_research.db', project_root: Path = None):
        """
        Initiates the database connection and creates tables if needed.
        
        Args:
            db_name (str, optional): Database filename. Defaults to 'pubmed_research.db'.
            project_root (Path, optional): Project root directory.
        """
        
        #* Setting up the database path
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent  
        self.db_path = project_root / "data" / "processed" / db_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)  #creates the directory if it doesn't exist
                                                
        #* Setting up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        #* Create all necessary tables                          
        self.create_tables()
        

    @contextmanager  
    def get_connection(self):
        """
        Context manager that handles database connections safely.
        Ensures connections are properly closed even if errors occur.
        Important for SQLite databases.
        """
        
        #* Connect to the SQLite database
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enables column access by name
        try:
            yield conn
        finally:
            conn.close()
            

    def create_tables(self):
        """Creates all necessary database tables including the papers table,
        correlations table, and indexes."""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            #* Creates the Papers table (main table)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    pmid TEXT PRIMARY KEY,  
                    title TEXT NOT NULL,
                    abstract TEXT,
                    journal TEXT,
                    publication_date TEXT,
                    doi TEXT,
                    pmc_id TEXT,
                    keywords TEXT,
                    has_fulltext BOOLEAN DEFAULT FALSE,
                    fulltext_source TEXT,
                    fulltext_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            #* Creates the Correlations table 
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
                    verification_confidence REAL,
                    human_reviewed BOOLEAN DEFAULT FALSE,
                    
                    FOREIGN KEY (paper_id) REFERENCES papers(pmid),
                    UNIQUE(paper_id, probiotic_strain, health_condition, extraction_model)
                )
            ''')
            
            #* Create indexes for faster querying 
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_papers_date 
                ON papers(publication_date)
            ''')
            
            #* Create indexes for correlation table
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_correlations_paper
                ON correlations(paper_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_correlations_strain
                ON correlations(probiotic_strain)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_correlations_condition
                ON correlations(health_condition)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_correlations_type
                ON correlations(correlation_type)
            ''')
            
            conn.commit()
            
            #* Add new columns to existing databases if they don't exist
            # self._add_missing_columns(cursor)  # Method not implemented
            
            self.logger.info(f"Database tables created ans saved at {self.db_path}")

        
    def insert_paper(self, paper: Dict) -> bool:
        """
        Adds a paper into the database, in the papers table.
        
        Args:
            paper: Paper object containing all paper details
            
        Returns:
            True if paper was newly inserted, False if it already existed
        """
        
        with self.get_connection() as conn:  #connects to the database
            cursor = conn.cursor()
            
            #* Insert the paper (or ignore if it already exists)
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO papers
                    (pmid, title, abstract, journal, publication_date, doi, pmc_id, keywords, 
                     has_fulltext, fulltext_source, fulltext_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    paper['pmid'],
                    paper['title'],
                    paper.get('abstract', ''),  # Use .get() for optional fields to avoid errors
                    paper.get('journal', 'Unknown journal'),
                    paper.get('publication_date', ''),
                    paper.get('doi'),
                    paper.get('pmc_id'),
                    json.dumps(paper.get('keywords')) if paper.get('keywords') else None,
                    paper.get('has_fulltext', False),
                    paper.get('fulltext_source'),
                    paper.get('fulltext_path')
                ))
                
                #* Check if the paper was added
                was_new_paper = cursor.rowcount > 0  # the number of rows that were actually inserted,
                                                    # not the total number of rows in the database.
                conn.commit()  #applies changes
                
                if was_new_paper:
                    self.logger.info(f"Inserted new paper: {paper['title']}")
                else:
                    self.logger.debug(f"Paper already exists: {paper['title']}")
                return was_new_paper
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error inserting paper {paper['pmid']}: {e}")
                raise   

    
    def get_paper_by_pmid(self, pmid: str) -> Optional[Dict]:
        """
        Retrieves a specific paper by its PMID.
        
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
        """
        Retrieves all papers from the database.
        
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
    

    def search_papers(self, query: str) -> List[Dict]:
        """
        Full-text search for a specific stringacross titles and abstracts.
        
        Args:
            query: Search query string
            
        Returns:
            List of dictionaries containing matching papers
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
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
    

    def update_paper_fulltext(self, pmid: str, has_fulltext: bool, 
                            fulltext_source: str = None, fulltext_path: str = None) -> bool:
        """
        Updates fulltext information for an existing paper.
        
        Args:
            pmid: PubMed ID of the paper
            has_fulltext: Whether fulltext is available
            fulltext_source: Source of fulltext ('pmc', 'unpaywall', etc.)
            fulltext_path: Path where fulltext is stored
            
        Returns:
            True if paper was updated, False if it didn't exist
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE papers 
                SET has_fulltext = ?, fulltext_source = ?, fulltext_path = ?
                WHERE pmid = ?
            ''', (has_fulltext, fulltext_source, fulltext_path, pmid))
            conn.commit()
            return cursor.rowcount > 0
    

    def get_papers_with_pmc_ids(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieves papers that have PMC IDs but no fulltext yet.
        
        Args:
            limit: Optional limit on number of papers to return
            
        Returns:
            List of dictionaries containing paper details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT * FROM papers 
                WHERE pmc_id IS NOT NULL AND pmc_id != '' 
                AND (has_fulltext IS NULL OR has_fulltext = FALSE)
                ORDER BY publication_date DESC
            '''
            if limit:
                query += ' LIMIT ?'
                cursor.execute(query, (limit,))
            else:
                cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    

    def get_papers_with_doi_no_fulltext(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieves papers that have DOIs but no fulltext yet (for Unpaywall check).
        
        Args:
            limit: Optional limit on number of papers to return
            
        Returns:
            List of dictionaries containing paper details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT * FROM papers 
                WHERE doi IS NOT NULL AND doi != '' 
                AND (has_fulltext IS NULL OR has_fulltext = FALSE)
                ORDER BY publication_date DESC
            '''
            if limit:
                query += ' LIMIT ?'
                cursor.execute(query, (limit,))
            else:
                cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_papers_by_condition(self, condition: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieves papers that mention a specific health condition.
        
        Args:
            condition: Health condition to search for
            limit: Optional limit on number of papers to return
            
        Returns:
            List of dictionaries containing paper details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT * FROM papers
                WHERE title LIKE ? OR abstract LIKE ?
                ORDER BY publication_date DESC
            '''
            if limit:
                query += ' LIMIT ?'
                cursor.execute(query, (f'%{condition}%', f'%{condition}%', limit))
            else:
                cursor.execute(query, (f'%{condition}%', f'%{condition}%'))
            return [dict(row) for row in cursor.fetchall()]


    def delete_paper(self, pmid: str) -> bool:
        """
        Deletes a paper from the database.
        
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
    
    
    def insert_correlation(self, correlation: Dict) -> bool:
        """
        Inserts a correlation into the database.
        
        Args:
            correlation: Dictionary containing correlation details with keys:
                - paper_id (str): PMID of the paper
                - probiotic_strain (str): Name of the probiotic strain
                - health_condition (str): Health condition being studied
                - correlation_type (str): 'positive', 'negative', 'neutral', or 'inconclusive'
                - correlation_strength (float): 0.0 to 1.0
                - confidence_score (float): 0.0 to 1.0
                - extraction_model (str): Model used for extraction
                - Optional: effect_size, sample_size, study_duration, study_type,
                           dosage, population_details, supporting_quote
        Returns:
            True if correlation was newly inserted, False if it already existed
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO correlations
                    (paper_id, probiotic_strain, health_condition, correlation_type,
                    correlation_strength, effect_size, sample_size, study_duration,
                    study_type, dosage, population_details, confidence_score,
                    supporting_quote, extraction_model, validation_status, 
                    validation_issues, verification_model, human_reviewed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',(
                    correlation['paper_id'],
                    correlation['probiotic_strain'],
                    correlation['health_condition'],
                    correlation['correlation_type'],
                    correlation.get('correlation_strength'),
                    correlation.get('effect_size'),
                    correlation.get('sample_size'),
                    correlation.get('study_duration'),
                    correlation.get('study_type'),
                    correlation.get('dosage'),
                    correlation.get('population_details'),
                    correlation.get('confidence_score'),
                    correlation.get('supporting_quote'),
                    correlation['extraction_model'],
                    correlation.get('validation_status', 'validated'),
                    correlation.get('validation_issues'),
                    correlation.get('verification_model'),
                    correlation.get('human_reviewed', False)
                ))
                
                conn.commit()
                was_new = cursor.rowcount > 0
                
                if was_new:
                    self.logger.info(f"Inserted correlation: {correlation['probiotic_strain']} - {correlation['health_condition']}")
                
                return was_new
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error inserting correlation: {e}")
                raise
    

    def get_correlations_by_strain(self, strain: str) -> List[Dict]:
        """
        Retrieves all correlations for a specific probiotic strain.
        
        Args:
            strain: Name of the probiotic strain
            
        Returns:
            List of dictionaries containing correlation details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, p.title, p.publication_date, p.journal
                FROM correlations c
                JOIN papers p ON c.paper_id = p.pmid
                WHERE c.probiotic_strain LIKE ?
                ORDER BY c.confidence_score DESC, p.publication_date DESC
            ''', (f'%{strain}%',))
            return [dict(row) for row in cursor.fetchall()]
    
    
    def get_correlations_by_condition(self, condition: str) -> List[Dict]:
        """
        Retrieves all correlations for a specific health condition.
        
        Args:
            condition: Name of the health condition
            
        Returns:
            List of dictionaries containing correlation details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, p.title, p.publication_date, p.journal
                FROM correlations c
                JOIN papers p ON c.paper_id = p.pmid
                WHERE c.health_condition LIKE ?
                ORDER BY c.confidence_score DESC, p.publication_date DESC
            ''', (f'%{condition}%',))
            return [dict(row) for row in cursor.fetchall()]
    

    def get_correlations_by_paper(self, paper_id: str) -> List[Dict]:
        """
        Retrieves all correlations extracted from a specific paper.
        
        Args:
            paper_id: PMID of the paper
            
        Returns:
            List of dictionaries containing correlation details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM correlations
                WHERE paper_id = ?
                ORDER BY confidence_score DESC
            ''', (paper_id,))
            return [dict(row) for row in cursor.fetchall()]
    

    def get_unprocessed_papers(self, extraction_model: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieves papers that haven't been processed by a specific extraction model.
        
        Args:
            extraction_model: Name of the extraction model
            limit: Optional limit on number of papers to return
            
        Returns:
            List of dictionaries containing paper details
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT p.*
                FROM papers p
                LEFT JOIN correlations c ON p.pmid = c.paper_id AND c.extraction_model = ?
                WHERE c.id IS NULL AND p.abstract IS NOT NULL AND p.abstract != ''
                ORDER BY p.publication_date DESC
            '''
            
            if limit:
                query += ' LIMIT ?'
                cursor.execute(query, (extraction_model, limit))
            else:
                cursor.execute(query, (extraction_model,))
                
            return [dict(row) for row in cursor.fetchall()]
    

    def aggregate_correlations(self, min_papers: int = 2) -> List[Dict]:
        """
        Aggregates correlations across multiple papers for the same strain-condition pair.
        
        Args:
            min_papers: Minimum number of papers required to include in results
            
        Returns:
            List of aggregated correlation data
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    probiotic_strain,
                    health_condition,
                    COUNT(DISTINCT paper_id) as paper_count,
                    AVG(correlation_strength) as avg_strength,
                    AVG(confidence_score) as avg_confidence,
                    GROUP_CONCAT(DISTINCT correlation_type) as correlation_types,
                    GROUP_CONCAT(DISTINCT study_type) as study_types,
                    MIN(sample_size) as min_sample_size,
                    MAX(sample_size) as max_sample_size,
                    SUM(CASE WHEN correlation_type = 'positive' THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN correlation_type = 'negative' THEN 1 ELSE 0 END) as negative_count,
                    SUM(CASE WHEN correlation_type = 'neutral' THEN 1 ELSE 0 END) as neutral_count
                FROM correlations
                GROUP BY probiotic_strain, health_condition
                HAVING paper_count >= ?
                ORDER BY paper_count DESC, avg_confidence DESC
            ''', (min_papers,))
            
            return [dict(row) for row in cursor.fetchall()]
            
                
    def get_correlations_for_verification(self, limit: int = 10) -> List[Dict]:
        """Get correlations that need verification (pending status)."""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT c.*, p.title, p.abstract
                FROM correlations c
                JOIN papers p ON c.paper_id = p.pmid
                WHERE c.validation_status = 'pending'
                ORDER BY c.extraction_timestamp
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]


    def update_correlation_verification(self, correlation_id: int, 
                                    verification_data: Dict) -> bool:
        """Update correlation with verification results."""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    UPDATE correlations
                    SET validation_status = ?,
                        validation_issues = ?,
                        verification_model = ?,
                        verification_timestamp = CURRENT_TIMESTAMP,
                        verification_confidence = ?
                    WHERE id = ?
                ''', (
                    verification_data['validation_status'],
                    verification_data.get('validation_issues'),
                    verification_data['verification_model'],
                    verification_data.get('verification_confidence'),
                    correlation_id
                ))
                
                conn.commit()
                return cursor.rowcount > 0
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error updating verification: {e}")
                return False


    def update_human_review(self, correlation_id: int, 
                        reviewer: str, notes: str = None) -> bool:
        """Mark correlation as human-reviewed."""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                    UPDATE correlations
                    SET human_reviewed = TRUE,
                        human_reviewer = ?,
                        review_timestamp = CURRENT_TIMESTAMP,
                        review_notes = ?
                    WHERE id = ?
                ''', (reviewer, notes, correlation_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error updating human review: {e}")
                return False          


    #* ============= DATABASE STATISTICS METHOD ============= 
    def get_database_stats(self) -> Dict:
        """
        Retrieves statistics about the database contents including correlations.
        
        Returns:
            Dictionary containing counts and date ranges
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            #* Count total papers
            cursor.execute('SELECT COUNT(*) FROM papers')
            stats['total_papers'] = cursor.fetchone()[0]
            
            #* Count total correlations
            cursor.execute('SELECT COUNT(*) FROM correlations')
            stats['total_correlations'] = cursor.fetchone()[0]
            
            #* Count unique strains
            cursor.execute('SELECT COUNT(DISTINCT probiotic_strain) FROM correlations')
            stats['unique_strains'] = cursor.fetchone()[0]
            
            #* Count unique conditions
            cursor.execute('SELECT COUNT(DISTINCT health_condition) FROM correlations')
            stats['unique_conditions'] = cursor.fetchone()[0]
            
            #* Get date range of publications
            cursor.execute('''
                SELECT MIN(publication_date), MAX(publication_date)
                FROM papers
            ''')
            min_date, max_date = cursor.fetchone()
            stats['date_range'] = f"{min_date} to {max_date}" if min_date else "No papers yet"
            
            #* Count papers by journal
            cursor.execute('''
                SELECT journal, COUNT(*) as count
                FROM papers
                GROUP BY journal
                ORDER BY count DESC
                LIMIT 10
            ''')
            stats['top_journals'] = [(row[0], row[1]) for row in cursor.fetchall()]
            
            #* Get recent papers count
            cursor.execute('''
                SELECT COUNT(*) FROM papers
                WHERE datetime(created_at) >= datetime('now', '-7 days')
            ''')
            stats['papers_added_last_week'] = cursor.fetchone()[0]
            
            #* Top studied strain-condition pairs
            cursor.execute('''
                SELECT probiotic_strain, health_condition, COUNT(DISTINCT paper_id) as count
                FROM correlations
                GROUP BY probiotic_strain, health_condition
                ORDER BY count DESC
                LIMIT 10
            ''')
            stats['top_strain_condition_pairs'] = [
                {'strain': row[0], 'condition': row[1], 'papers': row[2]} 
                for row in cursor.fetchall()
            ]
            
            #* Correlation type distribution
            cursor.execute('''
                SELECT correlation_type, COUNT(*) as count
                FROM correlations
                GROUP BY correlation_type
            ''')
            stats['correlation_types'] = dict(cursor.fetchall())
            
            return stats             
