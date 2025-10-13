-- Phase 3.6: Mechanism Clustering Database Indexes
-- Performance optimization for mechanism cluster queries
-- Run after Phase 3.6 completes to improve query performance

-- Index on mechanism_clusters for fast lookups
CREATE INDEX IF NOT EXISTS idx_mechanism_clusters_member_count
ON mechanism_clusters(member_count DESC);

CREATE INDEX IF NOT EXISTS idx_mechanism_clusters_hierarchy
ON mechanism_clusters(hierarchy_level, parent_cluster_id);

-- Index on mechanism_cluster_membership for fast joins
CREATE INDEX IF NOT EXISTS idx_mechanism_membership_cluster
ON mechanism_cluster_membership(cluster_id);

CREATE INDEX IF NOT EXISTS idx_mechanism_membership_text
ON mechanism_cluster_membership(mechanism_text);

-- Note: intervention_mechanisms and mechanism_condition_associations indexes
-- will be added when those tables are populated in future data mining phases
