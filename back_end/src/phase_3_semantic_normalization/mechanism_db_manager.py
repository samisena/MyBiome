"""
Database Manager for Mechanism Semantic Normalization

Handles all database operations for mechanism clustering:
- Schema initialization
- CRUD operations for mechanism clusters and memberships
- Cross-entity association queries
- Analytics and reporting

Integrates with existing HierarchyManager for consistency.
"""

import os
import sqlite3
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MechanismCluster:
    """Dataclass for mechanism cluster metadata."""
    cluster_id: int
    canonical_name: str
    description: Optional[str]
    parent_cluster_id: Optional[int]
    hierarchy_level: int
    member_count: int
    avg_silhouette: Optional[float]
    created_at: str
    updated_at: str


@dataclass
class MechanismMembership:
    """Dataclass for mechanism cluster membership."""
    id: int
    mechanism_text: str
    cluster_id: int
    assignment_type: str
    similarity_score: Optional[float]
    embedding_vector: Optional[np.ndarray]
    embedding_model: str
    assigned_at: str
    iteration_number: int


class MechanismDatabaseManager:
    """
    Database manager for mechanism clustering operations.

    Provides high-level interface for:
    - Schema initialization
    - Cluster CRUD operations
    - Membership tracking
    - Cross-entity analytics
    """

    def __init__(self, db_path: str):
        """
        Initialize database manager.

        Args:
            db_path: Path to intervention_research.db
        """
        self.db_path = db_path
        self.schema_file = Path(__file__).parent / "mechanism_schema.sql"

        logger.info(f"MechanismDatabaseManager initialized: {db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self, force: bool = False) -> bool:
        """
        Initialize database schema from SQL file.

        Args:
            force: If True, drop existing tables before creating

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read schema file
            if not self.schema_file.exists():
                logger.error(f"Schema file not found: {self.schema_file}")
                return False

            with open(self.schema_file, 'r') as f:
                schema_sql = f.read()

            conn = self._get_connection()
            cursor = conn.cursor()

            # Drop tables if force=True
            if force:
                logger.warning("Dropping existing mechanism tables (force=True)")
                cursor.execute("DROP TABLE IF EXISTS mechanism_cluster_history")
                cursor.execute("DROP TABLE IF EXISTS mechanism_condition_associations")
                cursor.execute("DROP TABLE IF EXISTS intervention_mechanisms")
                cursor.execute("DROP TABLE IF EXISTS mechanism_cluster_membership")
                cursor.execute("DROP TABLE IF EXISTS mechanism_clusters")

            # Execute schema SQL
            cursor.executescript(schema_sql)
            conn.commit()
            conn.close()

            logger.info("Database schema initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            return False

    def create_cluster(
        self,
        canonical_name: str,
        description: Optional[str] = None,
        parent_cluster_id: Optional[int] = None,
        hierarchy_level: int = 0,
        avg_silhouette: Optional[float] = None
    ) -> Optional[int]:
        """
        Create a new mechanism cluster.

        Args:
            canonical_name: Canonical name for cluster
            description: Optional description
            parent_cluster_id: Parent cluster ID for hierarchy
            hierarchy_level: Hierarchy level (0=root, 1=child)
            avg_silhouette: Average silhouette score

        Returns:
            Cluster ID if successful, None otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO mechanism_clusters (
                    canonical_name, description, parent_cluster_id,
                    hierarchy_level, avg_silhouette
                )
                VALUES (?, ?, ?, ?, ?)
            """, (canonical_name, description, parent_cluster_id, hierarchy_level, avg_silhouette))

            cluster_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.debug(f"Created cluster: {canonical_name} (ID: {cluster_id})")
            return cluster_id

        except sqlite3.IntegrityError:
            logger.warning(f"Cluster already exists: {canonical_name}")
            return self.get_cluster_id_by_name(canonical_name)
        except Exception as e:
            logger.error(f"Failed to create cluster: {e}")
            return None

    def get_cluster_id_by_name(self, canonical_name: str) -> Optional[int]:
        """Get cluster ID by canonical name."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT cluster_id
                FROM mechanism_clusters
                WHERE canonical_name = ?
            """, (canonical_name,))

            row = cursor.fetchone()
            conn.close()

            return row['cluster_id'] if row else None

        except Exception as e:
            logger.error(f"Failed to get cluster ID: {e}")
            return None

    def add_membership(
        self,
        mechanism_text: str,
        cluster_id: int,
        assignment_type: str,
        similarity_score: Optional[float] = None,
        embedding_vector: Optional[np.ndarray] = None,
        iteration_number: int = 1
    ) -> Optional[int]:
        """
        Add mechanism to cluster membership.

        Args:
            mechanism_text: Mechanism text
            cluster_id: Target cluster ID
            assignment_type: 'primary', 'multi_label', or 'singleton'
            similarity_score: Similarity to cluster centroid
            embedding_vector: Optional embedding vector (768-dim)
            iteration_number: Iteration when assigned

        Returns:
            Membership ID if successful, None otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Convert embedding to bytes if provided
            embedding_bytes = embedding_vector.tobytes() if embedding_vector is not None else None

            cursor.execute("""
                INSERT INTO mechanism_cluster_membership (
                    mechanism_text, cluster_id, assignment_type,
                    similarity_score, embedding_vector, iteration_number
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(mechanism_text, cluster_id) DO UPDATE SET
                    assignment_type = excluded.assignment_type,
                    similarity_score = excluded.similarity_score,
                    iteration_number = excluded.iteration_number
            """, (mechanism_text, cluster_id, assignment_type, similarity_score, embedding_bytes, iteration_number))

            membership_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.debug(f"Added membership: {mechanism_text[:50]}... → cluster {cluster_id}")
            return membership_id

        except Exception as e:
            logger.error(f"Failed to add membership: {e}")
            return None

    def update_cluster_stats(self, cluster_id: int) -> bool:
        """
        Update cluster statistics (member_count).

        Args:
            cluster_id: Cluster ID to update

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Count members
            cursor.execute("""
                SELECT COUNT(*) AS member_count
                FROM mechanism_cluster_membership
                WHERE cluster_id = ?
            """, (cluster_id,))

            member_count = cursor.fetchone()['member_count']

            # Update cluster
            cursor.execute("""
                UPDATE mechanism_clusters
                SET member_count = ?
                WHERE cluster_id = ?
            """, (member_count, cluster_id))

            conn.commit()
            conn.close()

            logger.debug(f"Updated cluster {cluster_id} stats: {member_count} members")
            return True

        except Exception as e:
            logger.error(f"Failed to update cluster stats: {e}")
            return False

    def populate_intervention_mechanisms(self) -> int:
        """
        Populate intervention_mechanisms junction table from interventions table.

        Returns:
            Number of rows inserted
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Check if interventions table has mechanism column
            cursor.execute("PRAGMA table_info(interventions)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'mechanism' not in columns:
                logger.warning("interventions.mechanism column does not exist")
                conn.close()
                return 0

            # Insert intervention-mechanism mappings
            cursor.execute("""
                INSERT INTO intervention_mechanisms (
                    intervention_id, mechanism_text, health_condition,
                    intervention_name, correlation_strength, correlation_type
                )
                SELECT
                    i.id,
                    i.mechanism,
                    i.health_condition,
                    i.intervention_name,
                    i.correlation_strength,
                    i.correlation_type
                FROM interventions i
                WHERE i.mechanism IS NOT NULL
                  AND i.mechanism != ''
                ON CONFLICT DO NOTHING
            """)

            rows_inserted = cursor.rowcount
            conn.commit()

            # Update cluster_id for mechanisms that have been clustered
            cursor.execute("""
                UPDATE intervention_mechanisms
                SET cluster_id = (
                    SELECT mcm.cluster_id
                    FROM mechanism_cluster_membership mcm
                    WHERE mcm.mechanism_text = intervention_mechanisms.mechanism_text
                    LIMIT 1
                )
                WHERE cluster_id IS NULL
                  AND mechanism_text IN (SELECT mechanism_text FROM mechanism_cluster_membership)
            """)

            rows_updated = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Populated intervention_mechanisms: {rows_inserted} inserted, {rows_updated} updated")
            return rows_inserted

        except Exception as e:
            logger.error(f"Failed to populate intervention_mechanisms: {e}")
            return 0

    def build_mechanism_condition_associations(self) -> int:
        """
        Build mechanism_condition_associations analytics table.

        Returns:
            Number of associations created
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Aggregate intervention counts and avg correlation strength
            cursor.execute("""
                INSERT INTO mechanism_condition_associations (
                    cluster_id, health_condition, intervention_count, avg_correlation_strength
                )
                SELECT
                    im.cluster_id,
                    im.health_condition,
                    COUNT(*) AS intervention_count,
                    AVG(im.correlation_strength) AS avg_correlation_strength
                FROM intervention_mechanisms im
                WHERE im.cluster_id IS NOT NULL
                GROUP BY im.cluster_id, im.health_condition
                ON CONFLICT(cluster_id, health_condition) DO UPDATE SET
                    intervention_count = excluded.intervention_count,
                    avg_correlation_strength = excluded.avg_correlation_strength,
                    updated_at = CURRENT_TIMESTAMP
            """)

            rows_inserted = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Built mechanism_condition_associations: {rows_inserted} associations")
            return rows_inserted

        except Exception as e:
            logger.error(f"Failed to build associations: {e}")
            return 0

    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get overall clustering statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Count clusters
            cursor.execute("SELECT COUNT(*) AS count FROM mechanism_clusters")
            total_clusters = cursor.fetchone()['count']

            # Count memberships
            cursor.execute("SELECT COUNT(*) AS count FROM mechanism_cluster_membership")
            total_memberships = cursor.fetchone()['count']

            # Count singletons
            cursor.execute("""
                SELECT COUNT(*) AS count
                FROM mechanism_cluster_membership
                WHERE assignment_type = 'singleton'
            """)
            singleton_count = cursor.fetchone()['count']

            # Count multi-label
            cursor.execute("""
                SELECT COUNT(*) AS count
                FROM mechanism_cluster_membership
                WHERE assignment_type = 'multi_label'
            """)
            multi_label_count = cursor.fetchone()['count']

            # Count associations
            cursor.execute("SELECT COUNT(*) AS count FROM mechanism_condition_associations")
            total_associations = cursor.fetchone()['count']

            # Average cluster size
            cursor.execute("""
                SELECT AVG(member_count) AS avg_size
                FROM mechanism_clusters
                WHERE member_count > 0
            """)
            avg_cluster_size = cursor.fetchone()['avg_size'] or 0.0

            conn.close()

            return {
                'total_clusters': total_clusters,
                'total_memberships': total_memberships,
                'singleton_count': singleton_count,
                'singleton_percentage': singleton_count / total_memberships if total_memberships > 0 else 0.0,
                'multi_label_count': multi_label_count,
                'total_associations': total_associations,
                'avg_cluster_size': avg_cluster_size
            }

        except Exception as e:
            logger.error(f"Failed to get cluster stats: {e}")
            return {}

    def save_cluster_history(self, iteration_number: int) -> bool:
        """
        Save current cluster state to history table.

        Args:
            iteration_number: Current iteration number

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO mechanism_cluster_history (
                    iteration_number, cluster_id, canonical_name,
                    member_count, avg_silhouette
                )
                SELECT
                    ?, cluster_id, canonical_name,
                    member_count, avg_silhouette
                FROM mechanism_clusters
            """, (iteration_number,))

            rows_inserted = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Saved cluster history for iteration {iteration_number}: {rows_inserted} clusters")
            return True

        except Exception as e:
            logger.error(f"Failed to save cluster history: {e}")
            return False

    def get_cluster_evolution_report(self) -> List[Dict[str, Any]]:
        """
        Get cluster evolution report across iterations.

        Returns:
            List of iteration summaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    iteration_number,
                    COUNT(*) AS cluster_count,
                    AVG(member_count) AS avg_size,
                    AVG(avg_silhouette) AS avg_silhouette
                FROM mechanism_cluster_history
                GROUP BY iteration_number
                ORDER BY iteration_number
            """)

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get evolution report: {e}")
            return []


def initialize_mechanism_schema(db_path: str, force: bool = False) -> bool:
    """
    Convenience function to initialize schema.

    Args:
        db_path: Path to intervention_research.db
        force: Drop existing tables before creating

    Returns:
        True if successful, False otherwise
    """
    manager = MechanismDatabaseManager(db_path)
    return manager.initialize_schema(force=force)


def main():
    """Command-line interface for schema management."""
    import argparse

    parser = argparse.ArgumentParser(description="Mechanism Database Schema Manager")
    parser.add_argument('--db-path', required=True, help='Path to intervention_research.db')
    parser.add_argument('--init', action='store_true', help='Initialize schema')
    parser.add_argument('--force', action='store_true', help='Force drop existing tables')
    parser.add_argument('--stats', action='store_true', help='Show cluster statistics')

    args = parser.parse_args()

    manager = MechanismDatabaseManager(args.db_path)

    if args.init:
        success = manager.initialize_schema(force=args.force)
        print(f"Schema initialization: {'SUCCESS' if success else 'FAILED'}")

    if args.stats:
        stats = manager.get_cluster_stats()
        print("\nCluster Statistics:")
        print(f"  Total clusters: {stats.get('total_clusters', 0)}")
        print(f"  Total memberships: {stats.get('total_memberships', 0)}")
        print(f"  Singletons: {stats.get('singleton_count', 0)} ({stats.get('singleton_percentage', 0):.1%})")
        print(f"  Multi-label: {stats.get('multi_label_count', 0)}")
        print(f"  Associations: {stats.get('total_associations', 0)}")
        print(f"  Avg cluster size: {stats.get('avg_cluster_size', 0):.1f}")


if __name__ == "__main__":
    main()
