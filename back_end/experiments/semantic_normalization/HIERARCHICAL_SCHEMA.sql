-- Hierarchical Semantic Normalization Database Schema
-- Purpose: Support multi-layer intervention/condition normalization with semantic embeddings

-- =============================================================================
-- Main Hierarchical Entity Table
-- =============================================================================

CREATE TABLE semantic_hierarchy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Entity Information
    entity_name TEXT NOT NULL,                  -- Original name from interventions table
    entity_type TEXT NOT NULL CHECK(entity_type IN ('intervention', 'condition')),

    -- Hierarchical Layers
    layer_0_category TEXT,                      -- From existing taxonomy (13 intervention categories / 18 condition categories)
    layer_1_canonical TEXT,                     -- Semantic group (e.g., "probiotics", "statins", "IBS")
    layer_2_variant TEXT,                       -- Specific entity (e.g., "L. reuteri", "atorvastatin", "IBS-D")
    layer_3_detail TEXT,                        -- Dosage/admin details (e.g., "L. reuteri 10^9 CFU")

    -- Parent-Child Relationships
    parent_id INTEGER,                          -- Points to parent entity in hierarchy
    relationship_type TEXT,                     -- 'EXACT_MATCH', 'VARIANT', 'SUBTYPE', 'SAME_CATEGORY', 'DOSAGE_VARIANT'
    aggregation_rule TEXT,                      -- How to aggregate at different layers

    -- Semantic Embedding (for similarity matching)
    embedding_vector BLOB,                      -- Binary serialized embedding vector
    embedding_model TEXT,                       -- Model used to generate embedding
    embedding_dimension INTEGER,                -- Dimension of embedding vector

    -- Metadata
    source_table TEXT,                          -- 'interventions' or 'health_conditions'
    source_ids TEXT,                            -- JSON array of intervention_ids that map to this entity
    occurrence_count INTEGER DEFAULT 1,         -- Number of papers mentioning this entity

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    FOREIGN KEY (parent_id) REFERENCES semantic_hierarchy(id) ON DELETE CASCADE,
    UNIQUE(entity_name, entity_type, layer_2_variant)
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- Index for hierarchical queries
CREATE INDEX idx_hierarchy_layers ON semantic_hierarchy(
    entity_type,
    layer_1_canonical,
    layer_2_variant
);

-- Index for parent-child traversal
CREATE INDEX idx_hierarchy_parent ON semantic_hierarchy(parent_id);

-- Index for entity lookup
CREATE INDEX idx_hierarchy_entity ON semantic_hierarchy(entity_name, entity_type);

-- Index for category filtering
CREATE INDEX idx_hierarchy_category ON semantic_hierarchy(layer_0_category);

-- =============================================================================
-- Entity Relationships Table (explicit relationship tracking)
-- =============================================================================

CREATE TABLE entity_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Entity Pair
    entity_1_id INTEGER NOT NULL,
    entity_2_id INTEGER NOT NULL,

    -- Relationship Details
    relationship_type TEXT NOT NULL,            -- 'EXACT_MATCH', 'VARIANT', 'SUBTYPE', etc.
    relationship_confidence REAL CHECK(relationship_confidence >= 0 AND relationship_confidence <= 1),

    -- Source of Relationship
    source TEXT NOT NULL,                       -- 'manual_labeling', 'llm_inference', 'embedding_similarity'
    labeled_by TEXT,                            -- User or model that labeled this relationship

    -- Hierarchical Aggregation Rules
    share_layer_1 BOOLEAN DEFAULT FALSE,        -- Should they share Layer 1 canonical?
    share_layer_2 BOOLEAN DEFAULT FALSE,        -- Should they share Layer 2 variant?

    -- Metadata
    similarity_score REAL,                      -- Embedding similarity score (if applicable)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    FOREIGN KEY (entity_1_id) REFERENCES semantic_hierarchy(id) ON DELETE CASCADE,
    FOREIGN KEY (entity_2_id) REFERENCES semantic_hierarchy(id) ON DELETE CASCADE,
    CHECK(entity_1_id < entity_2_id),           -- Prevent duplicate pairs (canonical ordering)
    UNIQUE(entity_1_id, entity_2_id)
);

-- Index for relationship queries
CREATE INDEX idx_relationships_entities ON entity_relationships(entity_1_id, entity_2_id);
CREATE INDEX idx_relationships_type ON entity_relationships(relationship_type);

-- =============================================================================
-- Canonical Groups Table (Layer 1 entities)
-- =============================================================================

CREATE TABLE canonical_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Group Information
    canonical_name TEXT NOT NULL UNIQUE,        -- e.g., "probiotics", "statins", "IBS"
    display_name TEXT,                          -- User-friendly display name
    entity_type TEXT NOT NULL CHECK(entity_type IN ('intervention', 'condition')),

    -- Category
    layer_0_category TEXT,                      -- From taxonomy

    -- Description
    description TEXT,                           -- What this group represents

    -- Aggregation Metadata
    member_count INTEGER DEFAULT 0,             -- Number of Layer 2 variants in this group
    total_paper_count INTEGER DEFAULT 0,        -- Total papers across all variants

    -- Semantic Embedding
    group_embedding BLOB,                       -- Average/centroid embedding for the group

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for canonical group lookup
CREATE INDEX idx_canonical_name ON canonical_groups(canonical_name);
CREATE INDEX idx_canonical_category ON canonical_groups(layer_0_category);

-- =============================================================================
-- Views for Common Queries
-- =============================================================================

-- View: All interventions with full hierarchical context
CREATE VIEW v_intervention_hierarchy AS
SELECT
    sh.id,
    sh.entity_name,
    sh.layer_0_category,
    sh.layer_1_canonical,
    sh.layer_2_variant,
    sh.layer_3_detail,
    sh.relationship_type,
    sh.occurrence_count,
    cg.display_name AS canonical_display_name,
    cg.description AS canonical_description
FROM semantic_hierarchy sh
LEFT JOIN canonical_groups cg ON sh.layer_1_canonical = cg.canonical_name
WHERE sh.entity_type = 'intervention';

-- View: Intervention aggregation by Layer 1 (canonical group)
CREATE VIEW v_intervention_by_canonical AS
SELECT
    layer_1_canonical,
    layer_0_category,
    COUNT(DISTINCT layer_2_variant) AS variant_count,
    SUM(occurrence_count) AS total_occurrences,
    GROUP_CONCAT(DISTINCT layer_2_variant, ', ') AS variants
FROM semantic_hierarchy
WHERE entity_type = 'intervention'
AND layer_1_canonical IS NOT NULL
GROUP BY layer_1_canonical, layer_0_category;

-- View: Intervention aggregation by Layer 2 (specific variant)
CREATE VIEW v_intervention_by_variant AS
SELECT
    layer_2_variant,
    layer_1_canonical,
    layer_0_category,
    COUNT(*) AS entity_count,
    SUM(occurrence_count) AS total_occurrences,
    GROUP_CONCAT(DISTINCT entity_name, ', ') AS entity_names
FROM semantic_hierarchy
WHERE entity_type = 'intervention'
AND layer_2_variant IS NOT NULL
GROUP BY layer_2_variant, layer_1_canonical, layer_0_category;

-- =============================================================================
-- Example Usage Queries
-- =============================================================================

-- Query 1: Find all probiotics and their variants
-- SELECT * FROM v_intervention_hierarchy WHERE layer_1_canonical = 'probiotics';

-- Query 2: Aggregate all evidence for "statins" (Layer 1)
-- SELECT * FROM v_intervention_by_canonical WHERE layer_1_canonical = 'statins';

-- Query 3: Compare specific probiotic strains (Layer 2)
-- SELECT * FROM v_intervention_by_variant
-- WHERE layer_1_canonical = 'probiotics'
-- AND layer_2_variant IN ('L. reuteri', 'S. boulardii');

-- Query 4: Find all VARIANT relationships (same concept, different formulation)
-- SELECT sh1.entity_name AS entity_1, sh2.entity_name AS entity_2, er.relationship_type
-- FROM entity_relationships er
-- JOIN semantic_hierarchy sh1 ON er.entity_1_id = sh1.id
-- JOIN semantic_hierarchy sh2 ON er.entity_2_id = sh2.id
-- WHERE er.relationship_type = 'VARIANT';

-- =============================================================================
-- Migration Notes
-- =============================================================================

-- This schema will replace the existing canonical_entities and entity_mappings tables
-- Migration steps:
-- 1. Create new tables (semantic_hierarchy, entity_relationships, canonical_groups)
-- 2. Migrate existing interventions to semantic_hierarchy with NULL hierarchical layers
-- 3. Run Phase 2 embedding system to populate layers and relationships
-- 4. Deprecate old canonical_entities and entity_mappings tables
-- 5. Update all queries to use new views
