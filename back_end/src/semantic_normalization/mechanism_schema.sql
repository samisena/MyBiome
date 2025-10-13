--
-- Database Schema for Mechanism Semantic Normalization
--
-- Creates 4 new tables for mechanism clustering and cross-entity tracking:
-- 1. mechanism_clusters - Cluster metadata and canonical names
-- 2. mechanism_cluster_membership - Many-to-many mapping (mechanism → cluster)
-- 3. intervention_mechanisms - Junction table (intervention → mechanism → cluster → condition)
-- 4. mechanism_condition_associations - Analytics (mechanism cluster → condition stats)
--
-- Also extends existing interventions table with embedding_vector column
--

-- ==============================================================================
-- TABLE 1: mechanism_clusters
-- ==============================================================================
-- Stores cluster metadata, canonical names, and hierarchy information

CREATE TABLE IF NOT EXISTS mechanism_clusters (
    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    description TEXT,

    -- Hierarchy (2-level: parent mechanisms → sub-mechanisms)
    parent_cluster_id INTEGER,
    hierarchy_level INTEGER DEFAULT 0,  -- 0 = root, 1 = child, 2+ = reserved for future

    -- Cluster statistics
    member_count INTEGER DEFAULT 0,
    avg_silhouette REAL,  -- Average silhouette score for cluster members

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key constraint for hierarchy
    FOREIGN KEY (parent_cluster_id) REFERENCES mechanism_clusters(cluster_id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_mechanism_clusters_canonical ON mechanism_clusters(canonical_name);
CREATE INDEX IF NOT EXISTS idx_mechanism_clusters_parent ON mechanism_clusters(parent_cluster_id);
CREATE INDEX IF NOT EXISTS idx_mechanism_clusters_hierarchy ON mechanism_clusters(hierarchy_level);


-- ==============================================================================
-- TABLE 2: mechanism_cluster_membership
-- ==============================================================================
-- Many-to-many mapping: mechanism_text → cluster_id
-- Supports multi-label clustering (one mechanism can belong to multiple clusters)

CREATE TABLE IF NOT EXISTS mechanism_cluster_membership (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_text TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,

    -- Assignment details
    assignment_type TEXT NOT NULL,  -- 'primary', 'multi_label', 'singleton'
    similarity_score REAL,  -- Similarity to cluster centroid (0.0-1.0)

    -- Embeddings
    embedding_vector BLOB,  -- 768-dim nomic-embed-text embedding (numpy array as bytes)
    embedding_model TEXT DEFAULT 'nomic-embed-text',

    -- Metadata
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    iteration_number INTEGER DEFAULT 1,  -- Tracks when mechanism was assigned

    -- Foreign key constraint
    FOREIGN KEY (cluster_id) REFERENCES mechanism_clusters(cluster_id) ON DELETE CASCADE,

    -- Unique constraint: (mechanism_text, cluster_id) to prevent duplicates
    UNIQUE(mechanism_text, cluster_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_membership_mechanism ON mechanism_cluster_membership(mechanism_text);
CREATE INDEX IF NOT EXISTS idx_membership_cluster ON mechanism_cluster_membership(cluster_id);
CREATE INDEX IF NOT EXISTS idx_membership_assignment_type ON mechanism_cluster_membership(assignment_type);
CREATE INDEX IF NOT EXISTS idx_membership_iteration ON mechanism_cluster_membership(iteration_number);


-- ==============================================================================
-- TABLE 3: intervention_mechanisms
-- ==============================================================================
-- Junction table: intervention_id → mechanism_text → cluster_id → condition
-- Enables cross-entity tracking and analytics

CREATE TABLE IF NOT EXISTS intervention_mechanisms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_id INTEGER NOT NULL,
    mechanism_text TEXT NOT NULL,
    cluster_id INTEGER,  -- NULL if mechanism not yet clustered

    -- Context
    health_condition TEXT NOT NULL,  -- Track which condition this mechanism applies to
    intervention_name TEXT,  -- Denormalized for faster queries

    -- Effectiveness
    correlation_strength REAL,  -- From interventions.correlation_strength
    correlation_type TEXT,  -- From interventions.correlation_type

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign keys
    FOREIGN KEY (intervention_id) REFERENCES interventions(id) ON DELETE CASCADE,
    FOREIGN KEY (cluster_id) REFERENCES mechanism_clusters(cluster_id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_intervention_mechanisms_intervention ON intervention_mechanisms(intervention_id);
CREATE INDEX IF NOT EXISTS idx_intervention_mechanisms_cluster ON intervention_mechanisms(cluster_id);
CREATE INDEX IF NOT EXISTS idx_intervention_mechanisms_condition ON intervention_mechanisms(health_condition);
CREATE INDEX IF NOT EXISTS idx_intervention_mechanisms_mechanism_text ON intervention_mechanisms(mechanism_text);


-- ==============================================================================
-- TABLE 4: mechanism_condition_associations
-- ==============================================================================
-- Analytics table: mechanism cluster → condition statistics
-- Pre-computed aggregations for fast queries

CREATE TABLE IF NOT EXISTS mechanism_condition_associations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    health_condition TEXT NOT NULL,

    -- Statistics
    intervention_count INTEGER DEFAULT 0,  -- How many interventions use this mechanism for this condition
    avg_correlation_strength REAL,  -- Average effectiveness across interventions

    -- Metadata
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key
    FOREIGN KEY (cluster_id) REFERENCES mechanism_clusters(cluster_id) ON DELETE CASCADE,

    -- Unique constraint: (cluster_id, health_condition)
    UNIQUE(cluster_id, health_condition)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_mech_cond_assoc_cluster ON mechanism_condition_associations(cluster_id);
CREATE INDEX IF NOT EXISTS idx_mech_cond_assoc_condition ON mechanism_condition_associations(health_condition);
CREATE INDEX IF NOT EXISTS idx_mech_cond_assoc_intervention_count ON mechanism_condition_associations(intervention_count DESC);


-- ==============================================================================
-- TABLE 5: mechanism_cluster_history
-- ==============================================================================
-- Cluster evolution tracking over iterations

CREATE TABLE IF NOT EXISTS mechanism_cluster_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration_number INTEGER NOT NULL,
    cluster_id INTEGER NOT NULL,
    canonical_name TEXT NOT NULL,
    member_count INTEGER DEFAULT 0,
    avg_silhouette REAL,

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Foreign key
    FOREIGN KEY (cluster_id) REFERENCES mechanism_clusters(cluster_id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_cluster_history_iteration ON mechanism_cluster_history(iteration_number);
CREATE INDEX IF NOT EXISTS idx_cluster_history_cluster ON mechanism_cluster_history(cluster_id);


-- ==============================================================================
-- EXTENSION: Add embedding_vector to interventions table
-- ==============================================================================
-- This allows storing intervention-level embeddings for future semantic queries

-- Check if column already exists (SQLite doesn't support IF NOT EXISTS for columns)
-- Run this manually if interventions table already exists:
-- ALTER TABLE interventions ADD COLUMN embedding_vector BLOB;
-- ALTER TABLE interventions ADD COLUMN embedding_model TEXT DEFAULT 'nomic-embed-text';


-- ==============================================================================
-- VIEWS: Convenient queries for common use cases
-- ==============================================================================

-- View 1: Mechanisms with cluster info
CREATE VIEW IF NOT EXISTS v_mechanisms_with_clusters AS
SELECT
    mcm.mechanism_text,
    mcm.cluster_id,
    mc.canonical_name,
    mc.parent_cluster_id,
    mc.hierarchy_level,
    mcm.assignment_type,
    mcm.similarity_score,
    mcm.iteration_number
FROM mechanism_cluster_membership mcm
LEFT JOIN mechanism_clusters mc ON mcm.cluster_id = mc.cluster_id
ORDER BY mc.canonical_name, mcm.mechanism_text;


-- View 2: Top mechanisms for each condition
CREATE VIEW IF NOT EXISTS v_top_mechanisms_by_condition AS
SELECT
    mca.health_condition,
    mc.canonical_name AS mechanism_cluster,
    mca.intervention_count,
    mca.avg_correlation_strength,
    mc.member_count AS cluster_size
FROM mechanism_condition_associations mca
JOIN mechanism_clusters mc ON mca.cluster_id = mc.cluster_id
ORDER BY mca.health_condition, mca.intervention_count DESC;


-- View 3: Conditions most treated by mechanism
CREATE VIEW IF NOT EXISTS v_conditions_by_mechanism AS
SELECT
    mc.canonical_name AS mechanism_cluster,
    mca.health_condition,
    mca.intervention_count,
    mca.avg_correlation_strength
FROM mechanism_condition_associations mca
JOIN mechanism_clusters mc ON mca.cluster_id = mc.cluster_id
ORDER BY mc.canonical_name, mca.intervention_count DESC;


-- View 4: Intervention-mechanism-condition relationships
CREATE VIEW IF NOT EXISTS v_intervention_mechanism_details AS
SELECT
    im.intervention_name,
    im.health_condition,
    im.mechanism_text,
    mc.canonical_name AS mechanism_cluster,
    mc.hierarchy_level,
    im.correlation_strength,
    im.correlation_type
FROM intervention_mechanisms im
LEFT JOIN mechanism_clusters mc ON im.cluster_id = mc.cluster_id
ORDER BY im.intervention_name, mc.canonical_name;


-- ==============================================================================
-- TRIGGERS: Automatic timestamp updates
-- ==============================================================================

-- Trigger: Update mechanism_clusters.updated_at on changes
CREATE TRIGGER IF NOT EXISTS trg_mechanism_clusters_updated_at
AFTER UPDATE ON mechanism_clusters
FOR EACH ROW
BEGIN
    UPDATE mechanism_clusters
    SET updated_at = CURRENT_TIMESTAMP
    WHERE cluster_id = NEW.cluster_id;
END;


-- Trigger: Update mechanism_condition_associations.updated_at on changes
CREATE TRIGGER IF NOT EXISTS trg_mech_cond_assoc_updated_at
AFTER UPDATE ON mechanism_condition_associations
FOR EACH ROW
BEGIN
    UPDATE mechanism_condition_associations
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;


-- ==============================================================================
-- SAMPLE QUERIES
-- ==============================================================================

-- Query 1: Most common mechanisms for a specific condition
-- SELECT * FROM v_top_mechanisms_by_condition WHERE health_condition = 'irritable bowel syndrome' LIMIT 10;

-- Query 2: Conditions most treated by a specific mechanism
-- SELECT * FROM v_conditions_by_mechanism WHERE mechanism_cluster = 'gut microbiome modulation' LIMIT 10;

-- Query 3: Interventions using a specific mechanism for a specific condition
-- SELECT intervention_name, mechanism_text, correlation_strength
-- FROM intervention_mechanisms im
-- JOIN mechanism_clusters mc ON im.cluster_id = mc.cluster_id
-- WHERE mc.canonical_name = 'gut-brain axis' AND im.health_condition = 'depression'
-- ORDER BY correlation_strength DESC;

-- Query 4: Cluster evolution over iterations
-- SELECT iteration_number, COUNT(*) AS cluster_count, AVG(member_count) AS avg_size
-- FROM mechanism_cluster_history
-- GROUP BY iteration_number
-- ORDER BY iteration_number;

-- Query 5: Singletons (unclustered mechanisms)
-- SELECT mechanism_text, similarity_score
-- FROM mechanism_cluster_membership
-- WHERE assignment_type = 'singleton'
-- ORDER BY similarity_score DESC;
