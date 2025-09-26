"""
Database manager with connection pooling and improved efficiency.
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

from back_end.src.data.config import config, setup_logging
from back_end.src.data.validators import validation_manager
from back_end.src.interventions.category_validators import category_validator

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


class DatabaseManager:
    """
    Database manager with connection pooling and validation.
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
            
        self.db_config = db_config or type('DatabaseConfig', (), {
            'name': config.db_name,
            'path': config.db_path,
            'max_connections': config.max_connections
        })()
        self.db_path = self.db_config.path
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection pool
        self.pool = ConnectionPool(str(self.db_path), self.db_config.max_connections)
        
        # Create tables
        self.create_tables()
        
        # Set up intervention categories
        self.setup_intervention_categories()
        
        logger.info(f"Enhanced database manager initialized at {self.db_path}")
        self._initialized = True
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        with self.pool.get_connection() as conn:
            yield conn
    
    def create_tables(self):
        """Create all necessary database tables with intervention-focused schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Papers table with Semantic Scholar fields
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
                    discovery_source TEXT DEFAULT 'pubmed',  -- pubmed, semantic_scholar, reference_following

                    -- Semantic Scholar fields
                    s2_paper_id TEXT,  -- Semantic Scholar paper ID
                    influence_score REAL,  -- Semantic Scholar influentialCitationCount
                    citation_count INTEGER,  -- Total citations from S2
                    tldr TEXT,  -- AI-generated summary from Semantic Scholar
                    s2_embedding TEXT,  -- JSON array of embedding vector
                    s2_processed BOOLEAN DEFAULT FALSE,  -- Track S2 enrichment status

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # NEW: Intervention categories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intervention_categories (
                    category TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    validation_schema TEXT,  -- JSON schema for validation
                    search_terms TEXT,       -- JSON array of search terms
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # NEW: Main interventions table (replaces correlations)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interventions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    intervention_category TEXT NOT NULL,
                    intervention_name TEXT NOT NULL,
                    intervention_details TEXT,  -- JSON object with category-specific fields
                    health_condition TEXT NOT NULL,
                    correlation_type TEXT CHECK(correlation_type IN ('positive', 'negative', 'neutral', 'inconclusive')),
                    correlation_strength REAL CHECK(correlation_strength >= 0 AND correlation_strength <= 1),
                    confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
                    
                    -- Study details
                    sample_size INTEGER,
                    study_duration TEXT,
                    study_type TEXT,
                    population_details TEXT,
                    supporting_quote TEXT,

                    -- Additional optional fields for enhanced dataset
                    delivery_method TEXT,
                    severity TEXT CHECK(severity IN ('mild', 'moderate', 'severe')),
                    adverse_effects TEXT,
                    cost_category TEXT CHECK(cost_category IN ('low', 'medium', 'high')),
                    
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
                    FOREIGN KEY (intervention_category) REFERENCES intervention_categories(category) ON DELETE RESTRICT,
                    UNIQUE(paper_id, intervention_category, intervention_name, health_condition)  -- Consensus: one record per intervention-condition pair per paper
                )
            ''')
            
            
            # Create optimized indexes
            indexes = [
                # Papers indexes
                'CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(publication_date)',
                'CREATE INDEX IF NOT EXISTS idx_papers_journal ON papers(journal)',
                'CREATE INDEX IF NOT EXISTS idx_papers_fulltext ON papers(has_fulltext)',
                'CREATE INDEX IF NOT EXISTS idx_papers_processing ON papers(processing_status)',
                
                # Interventions indexes  
                'CREATE INDEX IF NOT EXISTS idx_interventions_paper ON interventions(paper_id)',
                'CREATE INDEX IF NOT EXISTS idx_interventions_category ON interventions(intervention_category)',
                'CREATE INDEX IF NOT EXISTS idx_interventions_name ON interventions(intervention_name)',
                'CREATE INDEX IF NOT EXISTS idx_interventions_condition ON interventions(health_condition)',
                'CREATE INDEX IF NOT EXISTS idx_interventions_type ON interventions(correlation_type)',
                'CREATE INDEX IF NOT EXISTS idx_interventions_validation ON interventions(validation_status)',
                'CREATE INDEX IF NOT EXISTS idx_interventions_model ON interventions(extraction_model)',
                
                # Composite indexes for common queries
                'CREATE INDEX IF NOT EXISTS idx_category_condition ON interventions(intervention_category, health_condition)',
                'CREATE INDEX IF NOT EXISTS idx_intervention_condition ON interventions(intervention_name, health_condition)',
                'CREATE INDEX IF NOT EXISTS idx_paper_category ON interventions(paper_id, intervention_category)',
                'CREATE INDEX IF NOT EXISTS idx_paper_model ON interventions(paper_id, extraction_model)',
                
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

            # Add Semantic Scholar columns if they don't exist
            self._add_semantic_scholar_columns(cursor)

            # Add new optional fields to interventions table if they don't exist
            self._add_optional_intervention_columns(cursor)
            
            # Add triggers for updated_at timestamps
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS update_papers_timestamp
                AFTER UPDATE ON papers
                BEGIN
                    UPDATE papers SET updated_at = CURRENT_TIMESTAMP WHERE pmid = NEW.pmid;
                END
            ''')

            # Check if we need to create data mining tables
            self._create_data_mining_tables_if_needed(cursor)

            conn.commit()
            logger.info("Database tables and indexes created successfully")

    def _add_semantic_scholar_columns(self, cursor):
        """Add Semantic Scholar columns to existing papers table if they don't exist."""
        s2_columns = [
            ('s2_paper_id', 'TEXT'),
            ('influence_score', 'REAL'),
            ('citation_count', 'INTEGER'),
            ('tldr', 'TEXT'),
            ('s2_embedding', 'TEXT'),
            ('s2_processed', 'BOOLEAN DEFAULT FALSE'),
            ('discovery_source', 'TEXT DEFAULT \'pubmed\'')
        ]

        for column_name, column_type in s2_columns:
            try:
                cursor.execute(f'ALTER TABLE papers ADD COLUMN {column_name} {column_type}')
                logger.info(f"Added Semantic Scholar column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    # Column already exists, which is fine
                    continue
                else:
                    logger.error(f"Failed to add S2 column {column_name}: {e}")
                    raise

    def _add_optional_intervention_columns(self, cursor):
        """Add optional intervention columns to existing interventions table if they don't exist."""
        optional_columns = [
            ('delivery_method', 'TEXT'),
            ('severity', 'TEXT CHECK(severity IN (\'mild\', \'moderate\', \'severe\'))'),
            ('adverse_effects', 'TEXT'),
            ('cost_category', 'TEXT CHECK(cost_category IN (\'low\', \'medium\', \'high\'))'),

            # Consensus tracking fields
            ('consensus_confidence', 'REAL CHECK(consensus_confidence >= 0 AND consensus_confidence <= 1)'),
            ('model_agreement', 'TEXT CHECK(model_agreement IN (\'full\', \'partial\', \'single\', \'conflict\'))'),
            ('models_used', 'TEXT'),  # Comma-separated list of contributing models
            ('raw_extraction_count', 'INTEGER DEFAULT 1'),  # Number of original extractions
            ('models_contributing', 'TEXT')  # JSON array of contributing model details
        ]

        for column_name, column_type in optional_columns:
            try:
                cursor.execute(f'ALTER TABLE interventions ADD COLUMN {column_name} {column_type}')
                logger.info(f"Added optional intervention column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    # Column already exists, which is fine
                    continue
                else:
                    logger.error(f"Failed to add intervention column {column_name}: {e}")
                    raise

    def _create_data_mining_tables_if_needed(self, cursor):
        """Create data mining tables if they don't exist."""
        try:
            # Check if data mining tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_graph_nodes'
            """)

            if cursor.fetchone():
                logger.debug("Data mining tables already exist")
                return

            # Read and execute the enhanced schema
            schema_path = Path(__file__).parent / "enhanced_database_schema.sql"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()

                # Execute schema in parts to handle any errors
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                for statement in statements:
                    if statement:
                        try:
                            cursor.execute(statement)
                        except sqlite3.OperationalError as e:
                            if "table" in str(e) and "already exists" in str(e):
                                continue  # Table already exists
                            else:
                                logger.warning(f"Error executing schema statement: {e}")
                                continue

                logger.info("Data mining tables created successfully")
            else:
                logger.warning("Enhanced database schema file not found")

        except Exception as e:
            logger.error(f"Error creating data mining tables: {e}")

    def get_data_mining_connection(self):
        """Get a connection specifically for data mining operations."""
        return self.get_connection()

    def check_data_mining_tables_exist(self) -> bool:
        """Check if data mining tables exist in the database."""
        required_tables = [
            'knowledge_graph_nodes', 'knowledge_graph_edges', 'bayesian_scores',
            'treatment_recommendations', 'research_gaps', 'innovation_tracking',
            'biological_patterns', 'condition_similarities', 'intervention_combinations',
            'failed_interventions', 'data_mining_sessions'
        ]

        with self.get_connection() as conn:
            cursor = conn.cursor()

            for table in required_tables:
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name=?
                """, (table,))

                if not cursor.fetchone():
                    return False

            return True

    def initialize_data_mining_schema(self) -> bool:
        """Initialize data mining schema if not already present."""
        if self.check_data_mining_tables_exist():
            logger.info("Data mining tables already exist")
            return True

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                self._create_data_mining_tables_if_needed(cursor)
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to initialize data mining schema: {e}")
            return False
    
    def insert_paper(self, paper: Dict) -> bool:
        """Insert a paper with validation and enhanced error handling."""
        try:
            # Validate paper data
            # Validate paper data
            validation_result = validation_manager.validate_paper(paper)
            if not validation_result.is_valid:
                error_messages = [issue.message for issue in validation_result.errors]
                raise ValueError(f"Paper validation failed: {'; '.join(error_messages)}")
            validated_paper = validation_result.cleaned_data
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR IGNORE INTO papers
                    (pmid, title, abstract, journal, publication_date, doi, pmc_id,
                     keywords, has_fulltext, fulltext_source, fulltext_path, processing_status, discovery_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    'pending',
                    validated_paper.get('discovery_source', 'pubmed')
                ))
                
                was_new = cursor.rowcount > 0
                conn.commit()
                
                if was_new:
                    # Paper successfully inserted - perform cleanup
                    self._cleanup_paper_files(validated_paper['pmid'])
                else:
                    # Paper already existed, no cleanup needed
                    pass

                return was_new
                
        except ValueError as e:
            logger.error(f"Validation error for paper {paper.get('pmid', 'unknown')}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inserting paper {paper.get('pmid', 'unknown')}: {e}")
            return False

    def _cleanup_paper_files(self, pmid: str):
        """Clean up temporary files for a successfully processed paper."""
        try:
            # Import here to avoid circular imports
            from back_end.src.utils.batch_file_operations import cleanup_xml_files_for_papers

            # In FAST_MODE, queue files for batch deletion
            # In normal mode, still use batching but smaller batches
            cleanup_xml_files_for_papers([pmid])

        except Exception as e:
            logger.error(f"Error during cleanup for paper {pmid}: {e}")

    def insert_papers_batch(self, papers: List[Dict]) -> tuple[int, int]:
        """
        Insert multiple papers efficiently.

        Args:
            papers: List of paper dictionaries

        Returns:
            Tuple of (inserted_count, failed_count)
        """
        inserted_count = 0
        failed_count = 0

        for paper in papers:
            try:
                if self.insert_paper(paper):
                    inserted_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error inserting paper {paper.get('pmid', 'unknown')}: {e}")
                failed_count += 1

        # Flush any pending file operations after batch completion
        try:
            from back_end.src.utils.batch_file_operations import flush_pending_operations
            flush_pending_operations()
        except Exception as e:
            logger.error(f"Error flushing batch file operations: {e}")

        return inserted_count, failed_count
    
    def insert_intervention(self, intervention: Dict) -> bool:
        """Insert an intervention with validation."""
        try:
            # Validate intervention data
            validated_intervention = category_validator.validate_intervention(intervention)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO interventions
                    (paper_id, intervention_category, intervention_name, intervention_details,
                     health_condition, correlation_type, correlation_strength, confidence_score,
                     sample_size, study_duration, study_type, population_details,
                     supporting_quote, delivery_method, severity, adverse_effects, cost_category,
                     extraction_model, validation_status, consensus_confidence, model_agreement,
                     models_used, raw_extraction_count, models_contributing)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validated_intervention['paper_id'] if 'paper_id' in validated_intervention else validated_intervention.get('pmid'),
                    validated_intervention['intervention_category'],
                    validated_intervention['intervention_name'],
                    json.dumps(validated_intervention.get('intervention_details', {})),
                    validated_intervention['health_condition'],
                    validated_intervention['correlation_type'],
                    validated_intervention.get('correlation_strength'),
                    validated_intervention.get('confidence_score'),
                    validated_intervention.get('sample_size'),
                    validated_intervention.get('study_duration'),
                    validated_intervention.get('study_type'),
                    validated_intervention.get('population_details'),
                    validated_intervention.get('supporting_quote'),
                    validated_intervention.get('delivery_method'),
                    validated_intervention.get('severity'),
                    validated_intervention.get('adverse_effects'),
                    validated_intervention.get('cost_category'),
                    validated_intervention.get('extraction_model', 'consensus'),
                    'pending',
                    validated_intervention.get('consensus_confidence'),
                    validated_intervention.get('model_agreement'),
                    validated_intervention.get('models_used'),
                    validated_intervention.get('raw_extraction_count', 1),
                    json.dumps(validated_intervention.get('models_contributing', []))
                ))
                
                was_new = cursor.rowcount > 0
                conn.commit()
                
                if was_new:
                    logger.info(f"Inserted intervention: {validated_intervention['intervention_category']} - {validated_intervention['intervention_name']} - {validated_intervention['health_condition']}")
                
                return was_new
                
        except Exception as e:
            logger.error(f"Validation error for intervention: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inserting intervention: {e}")
            return False
    
    def setup_intervention_categories(self):
        """Set up the intervention categories table with taxonomy data."""
        from ..interventions.taxonomy import intervention_taxonomy
        from ..interventions.search_terms import search_terms
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                for category_type, category_def in intervention_taxonomy.get_all_categories().items():
                    # Get search terms for this category
                    category_search_terms = search_terms.get_terms_for_category(category_type)
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO intervention_categories
                        (category, display_name, description, search_terms)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        category_type.value,
                        category_def.display_name,
                        category_def.description,
                        json.dumps(category_search_terms)
                    ))
                
                conn.commit()
                logger.info(f"Set up {len(intervention_taxonomy.get_all_categories())} intervention categories")
                return True
                
        except Exception as e:
            logger.error(f"Error setting up intervention categories: {e}")
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
        """Get papers that need processing by a specific model (intervention extraction).
        Papers are prioritized by influence score, then citation count, then publication date."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT p.*
                FROM papers p
                LEFT JOIN interventions i ON p.pmid = i.paper_id AND i.extraction_model = ?
                WHERE i.id IS NULL
                  AND p.abstract IS NOT NULL
                  AND p.abstract != ''
                  AND (p.processing_status IS NULL OR p.processing_status != 'failed')
                ORDER BY
                    COALESCE(p.influence_score, 0) DESC,
                    COALESCE(p.citation_count, 0) DESC,
                    p.publication_date DESC
            '''
            
            params = [extraction_model]
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def update_paper_processing_status(self, pmid: str, status: str) -> bool:
        """Update paper processing status."""
        valid_statuses = ['pending', 'processing', 'processed', 'failed', 'needs_review']
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
        """Get comprehensive database statistics for intervention-focused system."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Basic counts
            cursor.execute('SELECT COUNT(*) FROM papers')
            stats['total_papers'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM interventions')
            stats['total_interventions'] = cursor.fetchone()[0]
            
            # Processing status breakdown
            cursor.execute('''
                SELECT processing_status, COUNT(*) 
                FROM papers 
                GROUP BY processing_status
            ''')
            stats['processing_status'] = dict(cursor.fetchall())
            
            # Validation status breakdown for interventions
            cursor.execute('''
                SELECT validation_status, COUNT(*) 
                FROM interventions 
                GROUP BY validation_status
            ''')
            stats['validation_status'] = dict(cursor.fetchall())
            
            # Intervention category breakdown
            cursor.execute('''
                SELECT intervention_category, COUNT(*) 
                FROM interventions 
                GROUP BY intervention_category
                ORDER BY COUNT(*) DESC
            ''')
            stats['intervention_categories'] = dict(cursor.fetchall())
            
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
                FROM interventions
                GROUP BY extraction_model
                ORDER BY count DESC
                LIMIT 5
            ''')
            stats['top_extraction_models'] = [
                {'model': row[0], 'interventions': row[1]} 
                for row in cursor.fetchall()
            ]
            
            # Top health conditions
            cursor.execute('''
                SELECT health_condition, COUNT(*) as count
                FROM interventions
                GROUP BY health_condition
                ORDER BY count DESC
                LIMIT 10
            ''')
            stats['top_health_conditions'] = [
                {'condition': row[0], 'interventions': row[1]}
                for row in cursor.fetchall()
            ]

            # Data mining stats (if tables exist)
            if self.check_data_mining_tables_exist():
                try:
                    # Knowledge graph stats
                    cursor.execute('SELECT COUNT(*) FROM knowledge_graph_nodes')
                    stats['knowledge_graph_nodes'] = cursor.fetchone()[0]

                    cursor.execute('SELECT COUNT(*) FROM knowledge_graph_edges')
                    stats['knowledge_graph_edges'] = cursor.fetchone()[0]

                    # Bayesian analysis stats
                    cursor.execute('SELECT COUNT(*) FROM bayesian_scores')
                    stats['bayesian_analyses'] = cursor.fetchone()[0]

                    # Treatment recommendations
                    cursor.execute('SELECT COUNT(*) FROM treatment_recommendations')
                    stats['treatment_recommendations'] = cursor.fetchone()[0]

                    # Research gaps
                    cursor.execute('SELECT COUNT(*) FROM research_gaps')
                    stats['research_gaps'] = cursor.fetchone()[0]

                    # Data mining sessions
                    cursor.execute('SELECT COUNT(*) FROM data_mining_sessions')
                    stats['data_mining_sessions'] = cursor.fetchone()[0]

                except Exception as e:
                    logger.warning(f"Error getting data mining stats: {e}")
                    stats['data_mining_error'] = str(e)
            else:
                stats['data_mining_tables'] = 'not_available'

            return stats
    
    def clean_placeholder_interventions(self) -> Dict[str, int]:
        """
        Remove interventions with placeholder names from the database.
        
        Returns:
            Dictionary with count of removed entries
        """
        placeholder_patterns = ['...', 'N/A', 'n/a', 'NA', 'na', 'null', 'NULL', 
                               'unknown', 'Unknown', 'UNKNOWN', 'placeholder', 
                               'Placeholder', 'PLACEHOLDER', 'TBD', 'tbd', 'TODO', 'todo',
                               'intervention', 'treatment', 'therapy',
                               'various', 'multiple', 'several', 'different']
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Count entries to be removed
                placeholders = "', '".join(placeholder_patterns)
                count_query = f"""
                    SELECT COUNT(*) FROM interventions 
                    WHERE intervention_name IN ('{placeholders}')
                    OR LENGTH(TRIM(intervention_name)) < 3
                    OR LOWER(intervention_name) LIKE 'unknown%'
                    OR LOWER(intervention_name) LIKE 'placeholder%'
                    OR LOWER(intervention_name) LIKE 'various%'
                    OR LOWER(intervention_name) LIKE 'multiple%'
                    OR LOWER(health_condition) IN ('{placeholders.lower()}')
                    OR LENGTH(TRIM(health_condition)) < 3
                """
                
                cursor.execute(count_query)
                count_to_remove = cursor.fetchone()[0]
                
                if count_to_remove == 0:
                    logger.info("No placeholder interventions found")
                    return {'removed_count': 0}
                
                # Remove placeholder entries
                delete_query = f"""
                    DELETE FROM interventions 
                    WHERE intervention_name IN ('{placeholders}')
                    OR LENGTH(TRIM(intervention_name)) < 3
                    OR LOWER(intervention_name) LIKE 'unknown%'
                    OR LOWER(intervention_name) LIKE 'placeholder%'
                    OR LOWER(intervention_name) LIKE 'various%'
                    OR LOWER(intervention_name) LIKE 'multiple%'
                    OR LOWER(health_condition) IN ('{placeholders.lower()}')
                    OR LENGTH(TRIM(health_condition)) < 3
                """
                
                cursor.execute(delete_query)
                removed_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Removed {removed_count} placeholder interventions from database")
                return {'removed_count': removed_count}
                
        except Exception as e:
            logger.error(f"Error cleaning placeholder interventions: {e}")
            return {'removed_count': 0, 'error': str(e)}
    
    def close(self):
        """Close all database connections."""
        if hasattr(self, 'pool'):
            self.pool.close_all()
            logger.info("Database connections closed")


# Global instance for dependency injection
database_manager = DatabaseManager()