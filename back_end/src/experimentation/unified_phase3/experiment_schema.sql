-- Experiment Database Schema for Unified Phase 3 Pipeline
-- =========================================================
-- Tracks experiments, results, and comparisons across different configurations

-- ==============================================================================
-- TABLE 1: experiments
-- ==============================================================================
-- Tracks each experiment run with configuration and metadata

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL UNIQUE,
    description TEXT,

    -- Configuration
    config_path TEXT NOT NULL,
    embedding_model TEXT NOT NULL,          -- nomic-embed-text, mxbai-embed-large
    clustering_algorithm TEXT NOT NULL,     -- hdbscan, hierarchical
    naming_temperature REAL NOT NULL,       -- 0.0, 0.2, 0.3, 0.4

    -- Hyperparameters (JSON)
    embedding_hyperparameters TEXT,         -- JSON of embedding settings
    clustering_hyperparameters TEXT,        -- JSON of clustering settings
    naming_hyperparameters TEXT,            -- JSON of naming settings

    -- Execution metadata
    status TEXT DEFAULT 'pending',          -- pending, running, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,

    -- Tags for filtering
    tags TEXT,                              -- JSON array of tags

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(experiment_name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_temperature ON experiments(naming_temperature);


-- ==============================================================================
-- TABLE 2: experiment_results
-- ==============================================================================
-- Stores detailed results for each experiment

CREATE TABLE IF NOT EXISTS experiment_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL,              -- intervention, condition, mechanism

    -- Phase 3a: Embedding results
    embedding_duration_seconds REAL,
    embeddings_generated INTEGER,
    embedding_cache_hit_rate REAL,

    -- Phase 3b: Clustering results
    clustering_duration_seconds REAL,
    num_clusters INTEGER,
    num_natural_clusters INTEGER,          -- HDBSCAN multi-member clusters
    num_singleton_clusters INTEGER,         -- Singleton clusters
    num_noise_points INTEGER,               -- Pre-singleton-handler
    assignment_rate REAL,                   -- Should be 1.0 (100%)
    silhouette_score REAL,
    davies_bouldin_score REAL,

    -- Cluster size statistics
    min_cluster_size INTEGER,
    max_cluster_size INTEGER,
    mean_cluster_size REAL,
    median_cluster_size REAL,

    -- Phase 3c: Naming results
    naming_duration_seconds REAL,
    names_generated INTEGER,
    naming_failures INTEGER,
    naming_cache_hit_rate REAL,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_experiment ON experiment_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_results_entity_type ON experiment_results(entity_type);


-- ==============================================================================
-- TABLE 3: cluster_details
-- ==============================================================================
-- Stores individual cluster information for each experiment

CREATE TABLE IF NOT EXISTS cluster_details (
    cluster_detail_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,

    -- Cluster properties
    canonical_name TEXT,
    category TEXT,
    parent_cluster TEXT,
    member_count INTEGER,
    is_singleton BOOLEAN,

    -- Member entities (JSON array)
    member_entities TEXT,                   -- JSON array of member names
    member_frequencies TEXT,                -- JSON array of frequencies

    -- Quality metrics
    silhouette_score REAL,
    confidence REAL,

    -- Provenance
    naming_method TEXT,                     -- llm, fallback
    naming_model TEXT,
    naming_temperature REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cluster_details_experiment ON cluster_details(experiment_id);
CREATE INDEX IF NOT EXISTS idx_cluster_details_entity_type ON cluster_details(entity_type);
CREATE INDEX IF NOT EXISTS idx_cluster_details_cluster_id ON cluster_details(cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_details_singleton ON cluster_details(is_singleton);


-- ==============================================================================
-- TABLE 4: naming_comparisons
-- ==============================================================================
-- Compares naming results for same cluster across different temperatures

CREATE TABLE IF NOT EXISTS naming_comparisons (
    comparison_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Source experiments (same cluster, different temperatures)
    experiment_id_1 INTEGER NOT NULL,
    experiment_id_2 INTEGER NOT NULL,

    entity_type TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,

    -- Names from each experiment
    canonical_name_1 TEXT,
    canonical_name_2 TEXT,

    -- Comparison metrics
    levenshtein_distance INTEGER,          -- Edit distance between names
    semantic_similarity REAL,              -- Cosine similarity of name embeddings
    category_match BOOLEAN,                 -- Same category?

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id_1) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (experiment_id_2) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_naming_comp_exp1 ON naming_comparisons(experiment_id_1);
CREATE INDEX IF NOT EXISTS idx_naming_comp_exp2 ON naming_comparisons(experiment_id_2);
CREATE INDEX IF NOT EXISTS idx_naming_comp_entity_type ON naming_comparisons(entity_type);


-- ==============================================================================
-- TABLE 5: temperature_analysis
-- ==============================================================================
-- Aggregated analysis of temperature effects on naming quality

CREATE TABLE IF NOT EXISTS temperature_analysis (
    analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
    temperature REAL NOT NULL,
    entity_type TEXT NOT NULL,

    -- Consistency metrics (across multiple runs at same temperature)
    consistency_score REAL,                 -- 0.0-1.0 (higher = more consistent)
    avg_levenshtein_distance REAL,

    -- Quality metrics (manual review or heuristic)
    naming_quality_score REAL,              -- 0.0-1.0 (higher = better names)
    category_accuracy REAL,                 -- % correct categories

    -- Stability metrics
    json_parsing_success_rate REAL,         -- % successful JSON parses
    failure_rate REAL,                      -- % failed LLM calls

    -- Sample size
    num_experiments INTEGER,
    num_clusters_analyzed INTEGER,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(temperature, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_temp_analysis_temperature ON temperature_analysis(temperature);
CREATE INDEX IF NOT EXISTS idx_temp_analysis_entity_type ON temperature_analysis(entity_type);


-- ==============================================================================
-- TABLE 6: experiment_logs
-- ==============================================================================
-- Detailed logs for debugging and auditing

CREATE TABLE IF NOT EXISTS experiment_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    log_level TEXT NOT NULL,                -- DEBUG, INFO, WARNING, ERROR
    phase TEXT NOT NULL,                     -- embedding, clustering, naming
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_logs_experiment ON experiment_logs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_logs_level ON experiment_logs(log_level);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON experiment_logs(timestamp DESC);


-- ==============================================================================
-- VIEWS: Convenient queries for analysis
-- ==============================================================================

-- View 1: Experiment summary
CREATE VIEW IF NOT EXISTS v_experiment_summary AS
SELECT
    e.experiment_id,
    e.experiment_name,
    e.embedding_model,
    e.clustering_algorithm,
    e.naming_temperature,
    e.status,
    e.duration_seconds,
    COUNT(DISTINCT er.entity_type) AS entity_types_processed,
    SUM(er.num_clusters) AS total_clusters,
    SUM(er.num_singleton_clusters) AS total_singletons,
    AVG(er.silhouette_score) AS avg_silhouette,
    AVG(er.naming_cache_hit_rate) AS avg_naming_cache_hit_rate
FROM experiments e
LEFT JOIN experiment_results er ON e.experiment_id = er.experiment_id
GROUP BY e.experiment_id
ORDER BY e.created_at DESC;


-- View 2: Temperature comparison
CREATE VIEW IF NOT EXISTS v_temperature_comparison AS
SELECT
    e.naming_temperature,
    COUNT(*) AS num_experiments,
    AVG(e.duration_seconds) AS avg_duration_seconds,
    AVG(er.num_clusters) AS avg_clusters_per_entity_type,
    AVG(er.silhouette_score) AS avg_silhouette,
    AVG(er.naming_failures) AS avg_naming_failures,
    AVG(er.naming_cache_hit_rate) AS avg_naming_cache_hit_rate
FROM experiments e
JOIN experiment_results er ON e.experiment_id = er.experiment_id
WHERE e.status = 'completed'
GROUP BY e.naming_temperature
ORDER BY e.naming_temperature;


-- View 3: Cluster size distribution
CREATE VIEW IF NOT EXISTS v_cluster_size_distribution AS
SELECT
    e.experiment_name,
    cd.entity_type,
    cd.is_singleton,
    COUNT(*) AS num_clusters,
    AVG(cd.member_count) AS avg_member_count,
    MIN(cd.member_count) AS min_member_count,
    MAX(cd.member_count) AS max_member_count
FROM experiments e
JOIN cluster_details cd ON e.experiment_id = cd.experiment_id
GROUP BY e.experiment_name, cd.entity_type, cd.is_singleton
ORDER BY e.created_at DESC, cd.entity_type;


-- ==============================================================================
-- SAMPLE QUERIES
-- ==============================================================================

-- Query 1: Find best-performing temperature
-- SELECT * FROM v_temperature_comparison ORDER BY avg_silhouette DESC LIMIT 1;

-- Query 2: Compare two specific experiments
-- SELECT * FROM naming_comparisons WHERE experiment_id_1 = 1 AND experiment_id_2 = 2;

-- Query 3: Get all clusters for an experiment
-- SELECT * FROM cluster_details WHERE experiment_id = 1 ORDER BY member_count DESC;

-- Query 4: Find failed experiments
-- SELECT * FROM experiments WHERE status = 'failed' ORDER BY created_at DESC;

-- Query 5: Get experiment logs for debugging
-- SELECT * FROM experiment_logs WHERE experiment_id = 1 AND log_level IN ('ERROR', 'WARNING') ORDER BY timestamp DESC;
