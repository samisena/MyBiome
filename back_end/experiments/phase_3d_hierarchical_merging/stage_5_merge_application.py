"""
Stage 5: Merge Application

Apply approved merges to database with validation.
Updates parent-child relationships and junction tables.
"""

import sqlite3
import logging
import json
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from stage_3_llm_validation import LLMValidationResult
from validation_metrics import Cluster

logger = logging.getLogger(__name__)


@dataclass
class MergeApplicationResult:
    """Result of applying merges to database."""
    success: bool
    parents_created: int
    children_updated: int
    merges_identical: int
    merges_parent_child: int
    junction_tables_updated: int
    validation_passed: bool
    errors: List[str]


class MergeApplicator:
    """
    Applies approved merges to database.

    Handles both MERGE_IDENTICAL and CREATE_PARENT relationship types.
    """

    def __init__(self, db_path: str, entity_type: str = 'mechanism'):
        """
        Initialize merge applicator.

        Args:
            db_path: Path to database
            entity_type: 'mechanism', 'intervention', or 'condition'
        """
        self.db_path = db_path
        self.entity_type = entity_type

        logger.info(f"Initialized MergeApplicator for {entity_type}")

    def apply_merges(
        self,
        approved_merges: List[LLMValidationResult],
        target_level: int = 1,
        create_backup: bool = True
    ) -> MergeApplicationResult:
        """
        Apply all approved merges to database.

        Args:
            approved_merges: List of auto-approved merges
            target_level: Target hierarchy level for parents
            create_backup: Whether to backup database first

        Returns:
            MergeApplicationResult with statistics
        """
        logger.info(f"Applying {len(approved_merges)} merges to database...")

        if create_backup:
            self._create_backup()

        conn = sqlite3.connect(self.db_path)
        conn.execute("BEGIN TRANSACTION")

        parents_created = 0
        children_updated = 0
        merges_identical = 0
        merges_parent_child = 0
        errors = []

        try:
            for merge in approved_merges:
                if merge.relationship_type == 'MERGE_IDENTICAL':
                    # Merge into single cluster
                    self._apply_merge_identical(conn, merge, target_level)
                    merges_identical += 1
                    parents_created += 0  # No new parent, just reassignment
                    children_updated += 2

                elif merge.relationship_type == 'CREATE_PARENT':
                    # Create parent, preserve children
                    parent_id = self._apply_create_parent(conn, merge, target_level)
                    if parent_id:
                        parents_created += 1
                        children_updated += 2

            # Update junction tables
            junction_updates = self._update_junction_tables(conn)

            # Validate hierarchy
            validation_passed, validation_errors = self._validate_hierarchy(conn)
            errors.extend(validation_errors)

            if validation_passed:
                conn.commit()
                logger.info("Merges applied successfully")
            else:
                conn.rollback()
                logger.error("Validation failed, rolling back")
                return MergeApplicationResult(
                    success=False,
                    parents_created=0,
                    children_updated=0,
                    merges_identical=0,
                    merges_parent_child=0,
                    junction_tables_updated=0,
                    validation_passed=False,
                    errors=errors
                )

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to apply merges: {e}")
            errors.append(str(e))
            return MergeApplicationResult(
                success=False,
                parents_created=0,
                children_updated=0,
                merges_identical=0,
                merges_parent_child=0,
                junction_tables_updated=0,
                validation_passed=False,
                errors=errors
            )

        finally:
            conn.close()

        return MergeApplicationResult(
            success=True,
            parents_created=parents_created,
            children_updated=children_updated,
            merges_identical=merges_identical,
            merges_parent_child=merges_parent_child,
            junction_tables_updated=junction_updates,
            validation_passed=validation_passed,
            errors=errors
        )

    def _apply_merge_identical(
        self,
        conn: sqlite3.Connection,
        merge: LLMValidationResult,
        target_level: int
    ):
        """
        Apply MERGE_IDENTICAL: Reassign all members to one cluster.

        Args:
            conn: Database connection
            merge: LLMValidationResult object
            target_level: Target hierarchy level
        """
        candidate = merge.candidate
        keep_id = candidate.cluster_a_id  # Keep cluster A
        merge_id = candidate.cluster_b_id  # Merge B into A

        cursor = conn.cursor()

        if self.entity_type == 'mechanism':
            # Reassign all members from B to A
            cursor.execute("""
                UPDATE mechanism_cluster_membership
                SET cluster_id = ?
                WHERE cluster_id = ?
            """, (keep_id, merge_id))

            # Update cluster A metadata
            cursor.execute("""
                UPDATE mechanism_clusters
                SET member_count = (
                    SELECT COUNT(*) FROM mechanism_cluster_membership WHERE cluster_id = ?
                ),
                canonical_name = ?
                WHERE cluster_id = ?
            """, (keep_id, merge.suggested_parent_name or candidate.cluster_a.canonical_name, keep_id))

            # Delete cluster B
            cursor.execute("DELETE FROM mechanism_clusters WHERE cluster_id = ?", (merge_id,))

        logger.debug(f"  MERGE_IDENTICAL: {merge_id} → {keep_id}")

    def _apply_create_parent(
        self,
        conn: sqlite3.Connection,
        merge: LLMValidationResult,
        target_level: int
    ) -> Optional[int]:
        """
        Apply CREATE_PARENT: Create new parent cluster, update children.

        Args:
            conn: Database connection
            merge: LLMValidationResult object
            target_level: Target hierarchy level for parent

        Returns:
            Parent cluster ID if successful, None otherwise
        """
        candidate = merge.candidate
        cursor = conn.cursor()

        # Get next available cluster ID
        if self.entity_type == 'mechanism':
            cursor.execute("SELECT MAX(cluster_id) FROM mechanism_clusters")
            max_id = cursor.fetchone()[0]
            parent_id = (max_id or 0) + 1

            # Calculate total member count
            total_members = len(candidate.cluster_a.members) + len(candidate.cluster_b.members)

            # Create parent cluster
            cursor.execute("""
                INSERT INTO mechanism_clusters (
                    cluster_id, canonical_name, parent_cluster_id,
                    hierarchy_level, member_count
                )
                VALUES (?, ?, NULL, ?, ?)
            """, (
                parent_id,
                merge.suggested_parent_name or f"Parent_{parent_id}",
                target_level,
                total_members
            ))

            # Update children to point to parent
            for child_id, refined_name in [
                (candidate.cluster_a_id, merge.child_a_refined_name),
                (candidate.cluster_b_id, merge.child_b_refined_name)
            ]:
                update_fields = ["parent_cluster_id = ?", "hierarchy_level = ?"]
                update_values = [parent_id, target_level + 1]

                if refined_name:
                    update_fields.append("canonical_name = ?")
                    update_values.append(refined_name)

                update_values.append(child_id)

                cursor.execute(f"""
                    UPDATE mechanism_clusters
                    SET {', '.join(update_fields)}
                    WHERE cluster_id = ?
                """, update_values)

            logger.debug(f"  CREATE_PARENT: {parent_id} <- [{candidate.cluster_a_id}, {candidate.cluster_b_id}]")

            return parent_id

        return None

    def _update_junction_tables(self, conn: sqlite3.Connection) -> int:
        """
        Update junction tables after merges.

        For mechanisms: update intervention_mechanisms to reflect new cluster IDs.

        Args:
            conn: Database connection

        Returns:
            Number of rows updated
        """
        if self.entity_type != 'mechanism':
            return 0

        cursor = conn.cursor()

        # Rebuild intervention_mechanisms junction table
        cursor.execute("DELETE FROM intervention_mechanisms")

        cursor.execute("""
            INSERT OR REPLACE INTO intervention_mechanisms (
                intervention_id, mechanism_text, cluster_id,
                health_condition, correlation_strength
            )
            SELECT
                i.id,
                i.mechanism,
                mcm.cluster_id,
                i.health_condition,
                i.correlation_strength
            FROM interventions i
            INNER JOIN mechanism_cluster_membership mcm ON i.mechanism = mcm.mechanism_text
            WHERE i.mechanism IS NOT NULL
              AND i.mechanism != ''
              AND i.mechanism != 'N/A'
        """)

        rows_updated = cursor.rowcount
        logger.debug(f"  Updated junction tables: {rows_updated} rows")

        return rows_updated

    def _validate_hierarchy(self, conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
        """
        Validate hierarchy structure after merges.

        Checks:
        - No circular dependencies
        - All parents exist
        - No orphaned children

        Args:
            conn: Database connection

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        cursor = conn.cursor()

        if self.entity_type == 'mechanism':
            # Check for circular dependencies
            cursor.execute("""
                WITH RECURSIVE parent_chain AS (
                    SELECT cluster_id, parent_cluster_id, 1 as depth
                    FROM mechanism_clusters
                    WHERE parent_cluster_id IS NOT NULL

                    UNION ALL

                    SELECT pc.cluster_id, mc.parent_cluster_id, pc.depth + 1
                    FROM parent_chain pc
                    INNER JOIN mechanism_clusters mc ON pc.parent_cluster_id = mc.cluster_id
                    WHERE mc.parent_cluster_id IS NOT NULL AND pc.depth < 10
                )
                SELECT cluster_id, COUNT(*) as chain_length
                FROM parent_chain
                GROUP BY cluster_id
                HAVING chain_length > 5
            """)

            cycles = cursor.fetchall()
            if cycles:
                errors.append(f"Potential circular dependencies detected: {len(cycles)} clusters")

            # Check all parent IDs exist
            cursor.execute("""
                SELECT COUNT(*)
                FROM mechanism_clusters c1
                WHERE c1.parent_cluster_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM mechanism_clusters c2
                      WHERE c2.cluster_id = c1.parent_cluster_id
                  )
            """)

            orphans = cursor.fetchone()[0]
            if orphans > 0:
                errors.append(f"Orphaned children detected: {orphans} clusters reference non-existent parents")

            # Check hierarchy level consistency
            cursor.execute("""
                SELECT c.cluster_id, c.hierarchy_level, p.hierarchy_level
                FROM mechanism_clusters c
                INNER JOIN mechanism_clusters p ON c.parent_cluster_id = p.cluster_id
                WHERE c.hierarchy_level <= p.hierarchy_level
            """)

            level_errors = cursor.fetchall()
            if level_errors:
                errors.append(f"Hierarchy level inconsistencies: {len(level_errors)} clusters")

        is_valid = len(errors) == 0

        if not is_valid:
            for error in errors:
                logger.error(f"  Validation error: {error}")

        return is_valid, errors

    def _create_backup(self):
        """Create database backup before applying merges."""
        import shutil

        db_path = Path(self.db_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = db_path.parent / f"{db_path.stem}_backup_phase3d_{timestamp}.db"

        try:
            shutil.copy2(db_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    def rollback_merges(self, backup_path: str):
        """
        Rollback to backup database.

        Args:
            backup_path: Path to backup database
        """
        import shutil

        try:
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"Rolled back to backup: {backup_path}")
        except Exception as e:
            logger.error(f"Failed to rollback: {e}")


def print_application_summary(result: MergeApplicationResult):
    """
    Print summary of merge application.

    Args:
        result: MergeApplicationResult object
    """
    print("\n" + "="*80)
    print("MERGE APPLICATION SUMMARY")
    print("="*80)

    if result.success:
        print("\nStatus: SUCCESS")
        print(f"\nMerges Applied:")
        print(f"  MERGE_IDENTICAL: {result.merges_identical}")
        print(f"  CREATE_PARENT: {result.merges_parent_child}")
        print(f"\nDatabase Updates:")
        print(f"  Parents created: {result.parents_created}")
        print(f"  Children updated: {result.children_updated}")
        print(f"  Junction tables updated: {result.junction_tables_updated} rows")
        print(f"\nValidation: {'PASSED' if result.validation_passed else 'FAILED'}")
    else:
        print("\nStatus: FAILED")
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")

    print("="*80)


if __name__ == "__main__":
    # Test merge applicator
    logging.basicConfig(level=logging.INFO)

    print("Merge applicator module ready.")
    print("Use apply_merges() with approved LLMValidationResults to update database.")
