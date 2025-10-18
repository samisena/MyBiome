"""
Phase 3 Database Persistence Layer

Saves Phase 3 results (embeddings, clusters, canonical names) to database tables:
- semantic_hierarchy: Intervention and condition hierarchies with embeddings
- canonical_groups: Canonical group metadata
- mechanism_clusters: Mechanism cluster metadata
- mechanism_cluster_membership: Mechanism-to-cluster assignments
- intervention_mechanisms: Intervention-to-mechanism links

Author: Claude Code
Created: 2025-10-18
"""

import json
import logging
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Phase3DatabaseSaver:
    """
    Saves Phase 3 semantic normalization results to database.

    Handles all database persistence for interventions, conditions, and mechanisms.
    """

    def __init__(self, db_path: str):
        """
        Initialize database saver.

        Args:
            db_path: Path to intervention_research.db
        """
        self.db_path = Path(db_path)

    def save_all(
        self,
        entity_type: str,
        entity_names: List[str],
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        naming_results: Dict,
        embedding_model: str = "mxbai-embed-large"
    ) -> Dict[str, int]:
        """
        Save all Phase 3 results for an entity type.

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'
            entity_names: List of entity names
            embeddings: Embedding vectors (N x D)
            cluster_labels: Cluster assignments (N,)
            naming_results: Dict of cluster_id -> NamingResult
            embedding_model: Model name used for embeddings

        Returns:
            Dict with save statistics
        """
        logger.info(f"Saving {entity_type} Phase 3 results to database...")

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            # Route to appropriate save method
            if entity_type == 'intervention':
                stats = self._save_intervention_hierarchy(
                    conn, cursor, entity_names, embeddings,
                    cluster_labels, naming_results, embedding_model
                )
            elif entity_type == 'condition':
                stats = self._save_condition_hierarchy(
                    conn, cursor, entity_names, embeddings,
                    cluster_labels, naming_results, embedding_model
                )
            elif entity_type == 'mechanism':
                stats = self._save_mechanism_clusters(
                    conn, cursor, entity_names, embeddings,
                    cluster_labels, naming_results, embedding_model
                )
            else:
                raise ValueError(f"Invalid entity_type: {entity_type}")

            conn.commit()
            logger.info(f"Saved {entity_type} data: {stats}")

            return stats

        except Exception as e:
            conn.rollback()
            logger.error(f"Database save failed for {entity_type}s: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def _save_intervention_hierarchy(
        self,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        entity_names: List[str],
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        naming_results: Dict,
        embedding_model: str
    ) -> Dict[str, int]:
        """Save intervention semantic hierarchy to database."""

        # Clear existing intervention hierarchy
        cursor.execute("DELETE FROM semantic_hierarchy WHERE entity_type = 'intervention'")
        deleted = cursor.rowcount
        logger.info(f"Cleared {deleted} existing intervention hierarchy entries")

        # Build cluster_id -> naming_result mapping
        cluster_names = {}
        cluster_categories = {}
        for cluster_id, naming_result in naming_results.items():
            cluster_names[cluster_id] = naming_result.canonical_name
            cluster_categories[cluster_id] = naming_result.category

        # Insert each intervention with its cluster assignment
        inserted = 0
        for i, (entity_name, embedding, cluster_id) in enumerate(zip(entity_names, embeddings, cluster_labels)):
            # Get canonical name and category from cluster
            canonical_name = cluster_names.get(cluster_id, entity_name)
            category = cluster_categories.get(cluster_id, None)

            # Serialize embedding
            embedding_blob = pickle.dumps(embedding)

            cursor.execute("""
                INSERT INTO semantic_hierarchy (
                    entity_name,
                    entity_type,
                    layer_0_category,
                    layer_1_canonical,
                    layer_2_variant,
                    layer_3_detail,
                    embedding_vector,
                    embedding_model,
                    embedding_dimension,
                    source_table,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity_name,                    # entity_name
                'intervention',                 # entity_type
                category,                       # layer_0_category
                canonical_name,                 # layer_1_canonical (from Phase 3c)
                entity_name,                    # layer_2_variant (raw name)
                None,                          # layer_3_detail (dosage - not extracted yet)
                embedding_blob,                # embedding_vector
                embedding_model,               # embedding_model
                len(embedding),                # embedding_dimension
                'interventions',               # source_table
                datetime.now().isoformat(),    # created_at
                datetime.now().isoformat()     # updated_at
            ))
            inserted += 1

        logger.info(f"Inserted {inserted} intervention hierarchy entries")

        # Save canonical groups
        groups_inserted = self._save_canonical_groups(
            cursor, naming_results, 'intervention', embeddings, cluster_labels, entity_names
        )

        return {
            'hierarchy_entries': inserted,
            'canonical_groups': groups_inserted,
            'deleted_old': deleted
        }

    def _save_condition_hierarchy(
        self,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        entity_names: List[str],
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        naming_results: Dict,
        embedding_model: str
    ) -> Dict[str, int]:
        """Save condition semantic hierarchy to database."""

        # Clear existing condition hierarchy
        cursor.execute("DELETE FROM semantic_hierarchy WHERE entity_type = 'condition'")
        deleted = cursor.rowcount
        logger.info(f"Cleared {deleted} existing condition hierarchy entries")

        # Build cluster_id -> naming_result mapping
        cluster_names = {}
        cluster_categories = {}
        for cluster_id, naming_result in naming_results.items():
            cluster_names[cluster_id] = naming_result.canonical_name
            cluster_categories[cluster_id] = naming_result.category

        # Insert each condition with its cluster assignment
        inserted = 0
        for i, (entity_name, embedding, cluster_id) in enumerate(zip(entity_names, embeddings, cluster_labels)):
            # Get canonical name and category from cluster
            canonical_name = cluster_names.get(cluster_id, entity_name)
            category = cluster_categories.get(cluster_id, None)

            # Serialize embedding
            embedding_blob = pickle.dumps(embedding)

            cursor.execute("""
                INSERT INTO semantic_hierarchy (
                    entity_name,
                    entity_type,
                    layer_0_category,
                    layer_1_canonical,
                    layer_2_variant,
                    layer_3_detail,
                    embedding_vector,
                    embedding_model,
                    embedding_dimension,
                    source_table,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity_name,                    # entity_name
                'condition',                    # entity_type
                category,                       # layer_0_category
                canonical_name,                 # layer_1_canonical (from Phase 3c)
                entity_name,                    # layer_2_variant (raw name)
                None,                          # layer_3_detail (severity - not extracted yet)
                embedding_blob,                # embedding_vector
                embedding_model,               # embedding_model
                len(embedding),                # embedding_dimension
                'interventions',               # source_table (conditions come from interventions table)
                datetime.now().isoformat(),    # created_at
                datetime.now().isoformat()     # updated_at
            ))
            inserted += 1

        logger.info(f"Inserted {inserted} condition hierarchy entries")

        # Save canonical groups
        groups_inserted = self._save_canonical_groups(
            cursor, naming_results, 'condition', embeddings, cluster_labels, entity_names
        )

        return {
            'hierarchy_entries': inserted,
            'canonical_groups': groups_inserted,
            'deleted_old': deleted
        }

    def _save_mechanism_clusters(
        self,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        entity_names: List[str],
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        naming_results: Dict,
        embedding_model: str
    ) -> Dict[str, int]:
        """Save mechanism clusters and memberships to database."""

        logger.info(f"[MECH DEBUG] Starting mechanism cluster save")
        logger.info(f"[MECH DEBUG] Entity names: {len(entity_names)}, Embeddings shape: {embeddings.shape}, Naming results: {len(naming_results)}")

        # Clear existing mechanism clusters and memberships
        cursor.execute("DELETE FROM mechanism_cluster_membership")
        deleted_memberships = cursor.rowcount
        cursor.execute("DELETE FROM mechanism_clusters")
        deleted_clusters = cursor.rowcount
        logger.info(f"Cleared {deleted_clusters} mechanism clusters, {deleted_memberships} memberships")

        # Insert mechanism clusters
        logger.info(f"[MECH DEBUG] Inserting {len(naming_results)} mechanism clusters...")
        clusters_inserted = 0
        for cluster_id, naming_result in naming_results.items():
            # Count members in this cluster
            member_count = int(np.sum(cluster_labels == cluster_id))

            # Calculate average silhouette score (placeholder - would need actual calculation)
            avg_silhouette = 0.5  # Default value

            cursor.execute("""
                INSERT INTO mechanism_clusters (
                    cluster_id,
                    canonical_name,
                    parent_cluster_id,
                    hierarchy_level,
                    member_count,
                    avg_silhouette,
                    creation_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(cluster_id),                # cluster_id
                naming_result.canonical_name,   # canonical_name
                None,                           # parent_cluster_id (Phase 3d would set this)
                0,                              # hierarchy_level (Phase 3d would set this)
                member_count,                   # member_count
                avg_silhouette,                 # avg_silhouette
                datetime.now().isoformat()      # creation_timestamp
            ))
            clusters_inserted += 1

        logger.info(f"Inserted {clusters_inserted} mechanism clusters")

        # Insert mechanism cluster memberships
        memberships_inserted = 0
        for i, (mechanism_name, cluster_id, embedding) in enumerate(zip(entity_names, cluster_labels, embeddings)):
            # Determine assignment type (singleton vs cluster member)
            is_singleton = int(np.sum(cluster_labels == cluster_id)) == 1
            assignment_type = 'singleton' if is_singleton else 'hdbscan'

            # Serialize embedding
            embedding_blob = pickle.dumps(embedding)

            cursor.execute("""
                INSERT INTO mechanism_cluster_membership (
                    mechanism_text,
                    cluster_id,
                    assignment_type,
                    similarity_score,
                    embedding_vector
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                mechanism_name,                     # mechanism_text
                int(cluster_id),                    # cluster_id
                assignment_type,                    # assignment_type ('singleton' or 'hdbscan')
                1.0,                                # similarity_score (placeholder)
                embedding_blob                      # embedding_vector
            ))
            memberships_inserted += 1

        logger.info(f"Inserted {memberships_inserted} mechanism cluster memberships")

        # Link interventions to mechanism clusters
        # This requires querying the interventions table to match mechanisms
        links_inserted = self._link_interventions_to_mechanisms(cursor, entity_names, cluster_labels)

        return {
            'clusters': clusters_inserted,
            'memberships': memberships_inserted,
            'intervention_links': links_inserted,
            'deleted_clusters': deleted_clusters,
            'deleted_memberships': deleted_memberships
        }

    def _save_canonical_groups(
        self,
        cursor: sqlite3.Cursor,
        naming_results: Dict,
        entity_type: str,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        entity_names: List[str]
    ) -> int:
        """Save canonical groups to database."""

        # Clear existing canonical groups for this entity type
        cursor.execute("DELETE FROM canonical_groups WHERE entity_type = ?", (entity_type,))
        deleted = cursor.rowcount
        logger.info(f"Cleared {deleted} existing {entity_type} canonical groups")

        inserted = 0
        for cluster_id, naming_result in naming_results.items():
            # Get all members of this cluster
            member_indices = np.where(cluster_labels == cluster_id)[0]
            member_count = len(member_indices)

            # Calculate group embedding (mean of member embeddings)
            if len(member_indices) > 0:
                group_embedding = np.mean(embeddings[member_indices], axis=0)
                group_embedding_blob = pickle.dumps(group_embedding)
            else:
                group_embedding_blob = None

            cursor.execute("""
                INSERT OR REPLACE INTO canonical_groups (
                    canonical_name,
                    display_name,
                    entity_type,
                    layer_0_category,
                    description,
                    member_count,
                    total_paper_count,
                    group_embedding,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                naming_result.canonical_name,       # canonical_name
                naming_result.canonical_name,       # display_name
                entity_type,                        # entity_type
                naming_result.category,             # layer_0_category
                None,                               # description (could extract from LLM)
                member_count,                       # member_count
                0,                                  # total_paper_count (would need to count from interventions)
                group_embedding_blob,               # group_embedding
                datetime.now().isoformat(),         # created_at
                datetime.now().isoformat()          # updated_at
            ))
            inserted += 1

        logger.info(f"Inserted {inserted} {entity_type} canonical groups")
        return inserted

    def _link_interventions_to_mechanisms(
        self,
        cursor: sqlite3.Cursor,
        mechanism_names: List[str],
        cluster_labels: np.ndarray
    ) -> int:
        """Link interventions to mechanism clusters via intervention_mechanisms table."""

        logger.info(f"[MECH LINK] Starting intervention-mechanism linking")

        # Clear existing intervention-mechanism links
        cursor.execute("DELETE FROM intervention_mechanisms")
        deleted = cursor.rowcount
        logger.info(f"[MECH LINK] Cleared {deleted} existing intervention-mechanism links")

        # Build mechanism -> cluster_id mapping with NORMALIZED keys for better matching
        logger.info(f"[MECH LINK] Building mechanism -> cluster mapping from {len(mechanism_names)} entities")
        mechanism_to_cluster = {}
        for i, (mech_name, cluster_id) in enumerate(zip(mechanism_names, cluster_labels)):
            # Store both raw and normalized versions
            mechanism_to_cluster[mech_name] = int(cluster_id)
            normalized_key = mech_name.lower().strip()
            mechanism_to_cluster[normalized_key] = int(cluster_id)

        logger.info(f"[MECH LINK] Created mapping with {len(mechanism_to_cluster)} keys (raw + normalized)")

        # Query interventions with mechanisms
        cursor.execute("""
            SELECT id, mechanism
            FROM interventions
            WHERE mechanism IS NOT NULL AND mechanism != ''
        """)
        intervention_rows = cursor.fetchall()
        logger.info(f"[MECH LINK] Found {len(intervention_rows)} interventions with mechanisms")

        inserted = 0
        not_found = 0
        duplicates = 0

        for row in intervention_rows:
            intervention_id = row[0]
            mechanism_text = row[1]

            # Try exact match first
            cluster_id = mechanism_to_cluster.get(mechanism_text)

            # Try normalized match if exact fails
            if cluster_id is None:
                normalized_text = mechanism_text.lower().strip()
                cluster_id = mechanism_to_cluster.get(normalized_text)

            if cluster_id is not None:
                # Insert link (schema: intervention_id, mechanism_text, cluster_id)
                try:
                    cursor.execute("""
                        INSERT INTO intervention_mechanisms (
                            intervention_id,
                            mechanism_text,
                            cluster_id
                        ) VALUES (?, ?, ?)
                    """, (
                        intervention_id,                # intervention_id
                        mechanism_text,                 # mechanism_text
                        cluster_id                      # cluster_id
                    ))
                    inserted += 1
                except sqlite3.IntegrityError:
                    duplicates += 1
                    logger.warning(f"[MECH LINK] Duplicate link: intervention_id={intervention_id}, cluster_id={cluster_id}")
            else:
                not_found += 1
                if not_found <= 5:  # Only log first 5 to avoid spam
                    logger.warning(f"[MECH LINK] No cluster found for mechanism: '{mechanism_text[:80]}...'")

        logger.info(f"[MECH LINK] Results: {inserted} links inserted, {not_found} mechanisms not found, {duplicates} duplicates skipped")
        return inserted
