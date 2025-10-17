"""
SchemaDAO - Database schema creation and migrations.

Handles table creation, schema migrations, and DDL operations.
"""

import sqlite3
import json
from pathlib import Path
from back_end.src.data.config import setup_logging
from .base_dao import BaseDAO

logger = setup_logging(__name__, 'database.log')


class SchemaDAO(BaseDAO):
    """Handles database schema creation and migrations."""

    def create_all_tables(self):
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

                    -- LLM processing tracking (Phase 2 optimization)
                    llm_processed BOOLEAN DEFAULT FALSE,  -- Fast indexed flag for LLM processing status

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

            # Intervention categories table
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

            # Condition categories table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS condition_categories (
                    category TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    description TEXT,
                    validation_schema TEXT,  -- JSON schema for validation
                    search_terms TEXT,       -- JSON array of search terms
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Main interventions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interventions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    intervention_category TEXT,
                    intervention_name TEXT NOT NULL,
                    intervention_details TEXT,  -- JSON object with category-specific fields
                    health_condition TEXT NOT NULL,
                    mechanism TEXT,  -- Biological/behavioral mechanism of action
                    outcome_type TEXT CHECK(outcome_type IN ('improves', 'worsens', 'no_effect', 'inconclusive')),

                    -- Study confidence metric
                    study_confidence REAL CHECK(study_confidence >= 0 AND study_confidence <= 1),

                    -- Study details
                    sample_size INTEGER,
                    study_duration TEXT,
                    study_type TEXT,
                    population_details TEXT,
                    supporting_quote TEXT,

                    -- Additional optional fields
                    delivery_method TEXT,
                    severity TEXT CHECK(severity IN ('mild', 'moderate', 'severe')),
                    adverse_effects TEXT,
                    cost_category TEXT CHECK(cost_category IN ('low', 'medium', 'high')),

                    -- Extraction tracking
                    extraction_model TEXT NOT NULL,
                    extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Validation tracking
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
                    UNIQUE(paper_id, intervention_category, intervention_name, health_condition)
                )
            ''')

            # Create indexes
            self._create_indexes(cursor)

            # Add triggers
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS update_papers_timestamp
                AFTER UPDATE ON papers
                BEGIN
                    UPDATE papers SET updated_at = CURRENT_TIMESTAMP WHERE pmid = NEW.pmid;
                END
            ''')

            logger.info("Database tables and indexes created successfully")

    def _create_indexes(self, cursor):
        """Create optimized indexes for common queries."""
        indexes = [
            # Papers indexes
            'CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(publication_date)',
            'CREATE INDEX IF NOT EXISTS idx_papers_journal ON papers(journal)',
            'CREATE INDEX IF NOT EXISTS idx_papers_fulltext ON papers(has_fulltext)',
            'CREATE INDEX IF NOT EXISTS idx_papers_processing ON papers(processing_status)',
            'CREATE INDEX IF NOT EXISTS idx_papers_llm_processed ON papers(llm_processed)',

            # Interventions indexes
            'CREATE INDEX IF NOT EXISTS idx_interventions_paper ON interventions(paper_id)',
            'CREATE INDEX IF NOT EXISTS idx_interventions_category ON interventions(intervention_category)',
            'CREATE INDEX IF NOT EXISTS idx_interventions_name ON interventions(intervention_name)',
            'CREATE INDEX IF NOT EXISTS idx_interventions_condition ON interventions(health_condition)',
            'CREATE INDEX IF NOT EXISTS idx_interventions_outcome ON interventions(outcome_type)',
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
                else:
                    raise

    def migrate_llm_processed_flag(self):
        """Add llm_processed flag for Phase 2 optimization."""
        if self.column_exists('papers', 'llm_processed'):
            logger.debug("llm_processed column already exists")
            return

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                logger.info("Adding llm_processed column to papers table")
                cursor.execute("ALTER TABLE papers ADD COLUMN llm_processed BOOLEAN DEFAULT FALSE")

                # Create index
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_papers_llm_processed
                    ON papers(llm_processed)
                ''')

                # Mark existing papers with interventions as processed
                cursor.execute("""
                    UPDATE papers
                    SET llm_processed = TRUE
                    WHERE pmid IN (
                        SELECT DISTINCT paper_id FROM interventions
                    )
                """)

                rows_updated = cursor.rowcount
                logger.info(f"Marked {rows_updated} existing papers as llm_processed")

        except Exception as e:
            logger.error(f"Failed to migrate to llm_processed flag: {e}")
            raise

    def migrate_study_confidence(self):
        """Add study_confidence column for quality metrics."""
        if self.column_exists('interventions', 'study_confidence'):
            logger.debug("study_confidence column already exists")
            return

        try:
            logger.info("Adding study_confidence column to interventions table")
            self.add_column_if_missing(
                'interventions',
                'study_confidence',
                'REAL CHECK(study_confidence >= 0 AND study_confidence <= 1)'
            )
            logger.info("Successfully migrated database to study_confidence system")

        except Exception as e:
            logger.error(f"Failed to migrate database: {e}")
            raise

    def add_optional_intervention_columns(self):
        """Add optional intervention columns to existing interventions table if they don't exist."""
        optional_columns = [
            ('delivery_method', 'TEXT'),
            ('severity', 'TEXT CHECK(severity IN (\'mild\', \'moderate\', \'severe\'))'),
            ('adverse_effects', 'TEXT'),
            ('cost_category', 'TEXT CHECK(cost_category IN (\'low\', \'medium\', \'high\'))'),
            ('condition_category', 'TEXT'),

            # Hierarchical prompt fields
            ('study_focus', 'TEXT'),
            ('measured_metrics', 'TEXT'),
            ('findings', 'TEXT'),
            ('study_location', 'TEXT'),
            ('publisher', 'TEXT'),

            # Consensus tracking fields
            ('consensus_confidence', 'REAL CHECK(consensus_confidence >= 0 AND consensus_confidence <= 1)'),
            ('model_agreement', 'TEXT CHECK(model_agreement IN (\'full\', \'partial\', \'single\', \'conflict\'))'),
            ('models_used', 'TEXT'),
            ('raw_extraction_count', 'INTEGER DEFAULT 1'),
            ('models_contributing', 'TEXT'),

            # Consensus wording selection
            ('condition_wording_source', 'TEXT'),
            ('condition_wording_confidence', 'REAL CHECK(condition_wording_confidence >= 0 AND condition_wording_confidence <= 1)'),
            ('original_condition_wordings', 'TEXT'),

            # Deduplication and canonical entity tracking
            ('intervention_canonical_id', 'INTEGER'),
            ('condition_canonical_id', 'INTEGER'),
            ('normalized', 'BOOLEAN DEFAULT 0')
        ]

        for column_name, column_type in optional_columns:
            self.add_column_if_missing('interventions', column_name, column_type)

    def add_semantic_scholar_columns(self):
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
            self.add_column_if_missing('papers', column_name, column_type)

    def create_data_mining_tables(self):
        """Create data mining tables if they don't exist."""
        try:
            # Check if data mining tables exist
            if self.table_exists('knowledge_graph_nodes'):
                logger.debug("Data mining tables already exist")
                self.create_frontend_export_session_table()
                return

            # Read and execute the enhanced schema
            schema_path = Path(__file__).parent.parent / "enhanced_database_schema.sql"
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()

                # Execute schema in parts to handle any errors
                statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    for statement in statements:
                        if statement:
                            try:
                                cursor.execute(statement)
                            except sqlite3.OperationalError as e:
                                if "table" in str(e) and "already exists" in str(e):
                                    continue
                                else:
                                    logger.warning(f"Error executing schema statement: {e}")
                                    continue

                logger.info("Data mining tables created successfully")
            else:
                logger.warning("Enhanced database schema file not found")

            # Create frontend export sessions table (Phase 5)
            self.create_frontend_export_session_table()

        except Exception as e:
            logger.error(f"Error creating data mining tables: {e}")

    def create_frontend_export_session_table(self):
        """Create frontend_export_sessions table for Phase 5 tracking."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS frontend_export_sessions (
                        session_id TEXT PRIMARY KEY,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        status TEXT CHECK(status IN ('running', 'completed', 'failed')),
                        files_exported INTEGER DEFAULT 0,
                        table_view_size_kb INTEGER,
                        network_viz_size_kb INTEGER,
                        validation_passed BOOLEAN DEFAULT TRUE,
                        error_message TEXT
                    )
                ''')
                logger.debug("Created/verified frontend_export_sessions table")
        except Exception as e:
            logger.error(f"Error creating frontend_export_sessions table: {e}")

    def check_data_mining_tables_exist(self) -> bool:
        """Check if data mining tables exist in the database."""
        required_tables = [
            'knowledge_graph_nodes', 'knowledge_graph_edges', 'bayesian_scores',
            'treatment_recommendations', 'research_gaps', 'innovation_tracking',
            'biological_patterns', 'condition_similarities', 'intervention_combinations',
            'failed_interventions', 'data_mining_sessions'
        ]

        for table in required_tables:
            if not self.table_exists(table):
                return False

        return True
