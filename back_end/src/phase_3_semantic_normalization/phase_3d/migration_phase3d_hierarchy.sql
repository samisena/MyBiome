-- ============================================================================
-- Phase 3d Hierarchical Cluster Merging - Database Migration
-- ============================================================================
--
-- This migration adds tables for storing hierarchical cluster relationships
-- created by Phase 3d merging pipeline.
--
-- Run this migration before enabling Phase 3d in the pipeline.
-- ============================================================================

-- Table 1: cluster_hierarchy
-- Stores parent-child-grandparent relationships for all entity types
CREATE TABLE IF NOT EXISTS cluster_hierarchy (
    hierarchy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('intervention', 'condition', 'mechanism')),

    -- Cluster IDs at each level
    child_cluster_id INTEGER NOT NULL,
    parent_cluster_id INTEGER,
    grandparent_cluster_id INTEGER,
    great_grandparent_cluster_id INTEGER,

    -- Metadata
    hierarchy_level INTEGER NOT NULL CHECK(hierarchy_level BETWEEN 0 AND 4),
    -- Level 0 = great-grandparent (top)
    -- Level 1 = grandparent
    -- Level 2 = parent
    -- Level 3 = child (base clusters from Phase 3b)
    -- Level 4 = individual entities (not used for hierarchy, just reference)

    merge_confidence REAL CHECK(merge_confidence BETWEEN 0.0 AND 1.0),
    merge_validation TEXT,  -- LLM reasoning for merge decision

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique child entries per entity type
    UNIQUE(entity_type, child_cluster_id)
);

CREATE INDEX idx_cluster_hierarchy_entity ON cluster_hierarchy(entity_type);
CREATE INDEX idx_cluster_hierarchy_child ON cluster_hierarchy(child_cluster_id);
CREATE INDEX idx_cluster_hierarchy_parent ON cluster_hierarchy(parent_cluster_id);
CREATE INDEX idx_cluster_hierarchy_level ON cluster_hierarchy(hierarchy_level);


-- Table 2: cluster_merges
-- Tracks individual merge operations (A + B → Parent)
CREATE TABLE IF NOT EXISTS cluster_merges (
    merge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('intervention', 'condition', 'mechanism')),

    -- Merge participants
    cluster_a_id INTEGER NOT NULL,
    cluster_a_name TEXT NOT NULL,
    cluster_b_id INTEGER NOT NULL,
    cluster_b_name TEXT NOT NULL,

    -- Result
    parent_cluster_id INTEGER NOT NULL,
    parent_canonical_name TEXT NOT NULL,
    relationship_type TEXT NOT NULL CHECK(relationship_type IN ('MERGE_IDENTICAL', 'CREATE_PARENT')),

    -- Quality metrics
    similarity_score REAL CHECK(similarity_score BETWEEN 0.0 AND 1.0),
    llm_confidence TEXT CHECK(llm_confidence IN ('HIGH', 'MEDIUM', 'LOW')),
    llm_validation TEXT,  -- LLM reasoning
    name_quality_score INTEGER CHECK(name_quality_score BETWEEN 0 AND 100),
    diversity_severity TEXT CHECK(diversity_severity IN ('NONE', 'MODERATE', 'SEVERE')),
    auto_approved BOOLEAN DEFAULT 0,

    -- Provenance
    phase3d_run_id INTEGER,  -- Links to experiment tracking
    merge_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Child name refinements (optional)
    child_a_refined_name TEXT,
    child_b_refined_name TEXT
);

CREATE INDEX idx_cluster_merges_entity ON cluster_merges(entity_type);
CREATE INDEX idx_cluster_merges_parent ON cluster_merges(parent_cluster_id);
CREATE INDEX idx_cluster_merges_timestamp ON cluster_merges(merge_timestamp);


-- Table 3: functional_categories
-- Multi-category support (PRIMARY, FUNCTIONAL, THERAPEUTIC, etc.)
CREATE TABLE IF NOT EXISTS functional_categories (
    functional_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('intervention', 'condition', 'mechanism')),

    -- Cluster reference
    cluster_id INTEGER NOT NULL,
    cluster_name TEXT NOT NULL,

    -- Category types (can have multiple)
    primary_category TEXT,          -- Original category from Phase 3c
    functional_category TEXT,       -- Functional grouping (e.g., "gut_flora_modulator")
    therapeutic_category TEXT,      -- Therapeutic use (e.g., "ibs_treatment")
    system_category TEXT,           -- Body system (e.g., "cardiovascular")
    pathway_category TEXT,          -- Biological pathway (e.g., "inflammation_pathway")
    target_category TEXT,           -- Target (e.g., "cox_enzyme_inhibitor")
    comorbidity_category TEXT,      -- Related conditions (for condition entities)

    -- Cross-category merge indicator
    is_cross_category BOOLEAN DEFAULT 0,
    llm_suggestion TEXT,            -- LLM-suggested functional category
    confidence TEXT CHECK(confidence IN ('HIGH', 'MEDIUM', 'LOW')),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Allow multiple category assignments per cluster
    UNIQUE(entity_type, cluster_id, functional_category)
);

CREATE INDEX idx_functional_categories_entity ON functional_categories(entity_type);
CREATE INDEX idx_functional_categories_cluster ON functional_categories(cluster_id);
CREATE INDEX idx_functional_categories_primary ON functional_categories(primary_category);
CREATE INDEX idx_functional_categories_functional ON functional_categories(functional_category);
CREATE INDEX idx_functional_categories_therapeutic ON functional_categories(therapeutic_category);
CREATE INDEX idx_functional_categories_cross ON functional_categories(is_cross_category);


-- ============================================================================
-- VIEWS: Convenient queries for hierarchy navigation
-- ============================================================================

-- View 1: Full hierarchy path for each cluster
CREATE VIEW IF NOT EXISTS v_cluster_hierarchy_paths AS
SELECT
    h.entity_type,
    h.child_cluster_id,
    h.parent_cluster_id,
    h.grandparent_cluster_id,
    h.great_grandparent_cluster_id,
    h.hierarchy_level,

    -- Count ancestors
    CASE
        WHEN h.great_grandparent_cluster_id IS NOT NULL THEN 3
        WHEN h.grandparent_cluster_id IS NOT NULL THEN 2
        WHEN h.parent_cluster_id IS NOT NULL THEN 1
        ELSE 0
    END as ancestor_count,

    -- Top-level parent (highest in hierarchy)
    COALESCE(
        h.great_grandparent_cluster_id,
        h.grandparent_cluster_id,
        h.parent_cluster_id,
        h.child_cluster_id
    ) as root_cluster_id,

    h.merge_confidence,
    h.created_at
FROM cluster_hierarchy h;


-- View 2: Merge summary statistics
CREATE VIEW IF NOT EXISTS v_merge_statistics AS
SELECT
    entity_type,
    COUNT(*) as total_merges,
    SUM(CASE WHEN relationship_type = 'MERGE_IDENTICAL' THEN 1 ELSE 0 END) as merge_identical_count,
    SUM(CASE WHEN relationship_type = 'CREATE_PARENT' THEN 1 ELSE 0 END) as create_parent_count,
    SUM(CASE WHEN auto_approved = 1 THEN 1 ELSE 0 END) as auto_approved_count,
    AVG(similarity_score) as avg_similarity,
    AVG(name_quality_score) as avg_name_quality,
    MIN(merge_timestamp) as first_merge,
    MAX(merge_timestamp) as last_merge
FROM cluster_merges
GROUP BY entity_type;


-- View 3: Multi-category cluster summary
CREATE VIEW IF NOT EXISTS v_multi_category_clusters AS
SELECT
    entity_type,
    cluster_id,
    cluster_name,
    COUNT(*) as category_count,
    GROUP_CONCAT(functional_category, ', ') as all_categories,
    MAX(CASE WHEN is_cross_category = 1 THEN 1 ELSE 0 END) as is_cross_category
FROM functional_categories
WHERE functional_category IS NOT NULL
GROUP BY entity_type, cluster_id, cluster_name;


-- ============================================================================
-- TRIGGERS: Maintain data integrity
-- ============================================================================

-- Trigger 1: Update timestamp on hierarchy changes
CREATE TRIGGER IF NOT EXISTS trg_cluster_hierarchy_update
AFTER UPDATE ON cluster_hierarchy
FOR EACH ROW
BEGIN
    UPDATE cluster_hierarchy
    SET updated_at = CURRENT_TIMESTAMP
    WHERE hierarchy_id = NEW.hierarchy_id;
END;


-- Trigger 2: Validate hierarchy level consistency
CREATE TRIGGER IF NOT EXISTS trg_validate_hierarchy_level
BEFORE INSERT ON cluster_hierarchy
FOR EACH ROW
BEGIN
    -- Ensure child level > parent level
    SELECT CASE
        WHEN NEW.parent_cluster_id IS NOT NULL
             AND NEW.hierarchy_level <= (
                 SELECT hierarchy_level
                 FROM cluster_hierarchy
                 WHERE child_cluster_id = NEW.parent_cluster_id
                 AND entity_type = NEW.entity_type
             )
        THEN RAISE(ABORT, 'Child hierarchy level must be greater than parent level')
    END;
END;


-- ============================================================================
-- SAMPLE QUERIES (commented out - for reference)
-- ============================================================================

-- Query 1: Get all children of a parent cluster
-- SELECT * FROM cluster_hierarchy
-- WHERE entity_type = 'intervention' AND parent_cluster_id = 42;

-- Query 2: Get full hierarchy path for a cluster
-- SELECT * FROM v_cluster_hierarchy_paths
-- WHERE entity_type = 'intervention' AND child_cluster_id = 123;

-- Query 3: Find clusters with multiple categories
-- SELECT * FROM v_multi_category_clusters
-- WHERE category_count > 1;

-- Query 4: Get merge history for a cluster
-- SELECT * FROM cluster_merges
-- WHERE entity_type = 'intervention'
--   AND (cluster_a_id = 123 OR cluster_b_id = 123 OR parent_cluster_id = 123)
-- ORDER BY merge_timestamp DESC;

-- Query 5: Find cross-category functional groups
-- SELECT * FROM functional_categories
-- WHERE is_cross_category = 1
-- ORDER BY entity_type, functional_category;


-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
-- Run status check:
-- SELECT name, type FROM sqlite_master
-- WHERE type IN ('table', 'view', 'trigger')
-- AND name LIKE '%cluster%' OR name LIKE '%merge%' OR name LIKE '%functional%';
-- ============================================================================
