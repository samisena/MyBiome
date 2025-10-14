"""
Test Suite for Stage 5: Merge Application

Tests database merge operations with in-memory test database.
"""

import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from stage_5_merge_application import MergeApplicator, MergeApplicationResult
from stage_3_llm_validation import LLMValidationResult, NameQualityScore, DiversityCheck
from stage_2_candidate_generation import MergeCandidate
from validation_metrics import Cluster


def create_test_database() -> str:
    """Create in-memory test database with mechanism clusters."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create mechanism_clusters table
    cursor.execute("""
        CREATE TABLE mechanism_clusters (
            cluster_id INTEGER PRIMARY KEY,
            canonical_name TEXT NOT NULL UNIQUE,
            parent_cluster_id INTEGER,
            hierarchy_level INTEGER DEFAULT 0,
            member_count INTEGER,
            avg_silhouette REAL,
            creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create mechanism_cluster_membership table
    cursor.execute("""
        CREATE TABLE mechanism_cluster_membership (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mechanism_text TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            assignment_type TEXT CHECK(assignment_type IN ('hdbscan', 'singleton')),
            similarity_score REAL,
            embedding_vector BLOB,
            UNIQUE(mechanism_text, cluster_id)
        )
    """)

    # Create intervention_mechanisms junction table
    cursor.execute("""
        CREATE TABLE intervention_mechanisms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            intervention_id INTEGER NOT NULL,
            mechanism_text TEXT NOT NULL,
            cluster_id INTEGER,
            health_condition TEXT,
            correlation_strength REAL
        )
    """)

    # Insert test data
    cursor.execute("""
        INSERT INTO mechanism_clusters (cluster_id, canonical_name, hierarchy_level, member_count)
        VALUES (1, 'Cluster A', 0, 2), (2, 'Cluster B', 0, 2)
    """)

    cursor.execute("""
        INSERT INTO mechanism_cluster_membership (mechanism_text, cluster_id, assignment_type)
        VALUES
            ('mechanism_1', 1, 'hdbscan'),
            ('mechanism_2', 1, 'hdbscan'),
            ('mechanism_3', 2, 'hdbscan'),
            ('mechanism_4', 2, 'hdbscan')
    """)

    conn.commit()
    conn.close()

    return ':memory:'


class TestMergeIdentical:
    """Test MERGE_IDENTICAL operations."""

    def test_merge_identical_basic(self):
        """Test merging two clusters into one."""
        # Create test database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        # Setup schema
        cursor.execute("""
            CREATE TABLE mechanism_clusters (
                cluster_id INTEGER PRIMARY KEY,
                canonical_name TEXT,
                parent_cluster_id INTEGER,
                hierarchy_level INTEGER,
                member_count INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE mechanism_cluster_membership (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mechanism_text TEXT,
                cluster_id INTEGER,
                assignment_type TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE intervention_mechanisms (
                id INTEGER PRIMARY KEY,
                intervention_id INTEGER,
                mechanism_text TEXT,
                cluster_id INTEGER,
                health_condition TEXT,
                correlation_strength REAL
            )
        """)

        # Insert test data
        cursor.execute("INSERT INTO mechanism_clusters VALUES (1, 'Cluster A', NULL, 0, 2)")
        cursor.execute("INSERT INTO mechanism_clusters VALUES (2, 'Cluster B', NULL, 0, 2)")
        cursor.execute("INSERT INTO mechanism_cluster_membership VALUES (1, 'm1', 1, 'hdbscan')")
        cursor.execute("INSERT INTO mechanism_cluster_membership VALUES (2, 'm2', 1, 'hdbscan')")
        cursor.execute("INSERT INTO mechanism_cluster_membership VALUES (3, 'm3', 2, 'hdbscan')")
        cursor.execute("INSERT INTO mechanism_cluster_membership VALUES (4, 'm4', 2, 'hdbscan')")

        conn.commit()

        # Create mock merge
        cluster_a = Cluster(1, "Cluster A", ["m1", "m2"], None, 0)
        cluster_b = Cluster(2, "Cluster B", ["m3", "m4"], None, 0)

        candidate = MergeCandidate(1, 2, cluster_a, cluster_b, 0.95, 'HIGH')

        merge = LLMValidationResult(
            candidate=candidate,
            relationship_type='MERGE_IDENTICAL',
            llm_confidence='HIGH',
            suggested_parent_name='Merged Cluster',
            child_a_refined_name=None,
            child_b_refined_name=None,
            llm_reasoning='Identical',
            name_quality=NameQualityScore(80, [], True),
            diversity_check=DiversityCheck(False, 'NONE', 0.9, 'Good'),
            auto_approved=True,
            flagged_reason=None
        )

        # Apply merge
        applicator = MergeApplicator(':memory:', 'mechanism')
        applicator.db_path = ':memory:'  # Override for test
        applicator._apply_merge_identical(conn, merge, target_level=0)

        # Verify: all members should be in cluster 1, cluster 2 should be deleted
        cursor.execute("SELECT COUNT(*) FROM mechanism_cluster_membership WHERE cluster_id = 1")
        assert cursor.fetchone()[0] == 4  # All 4 members

        cursor.execute("SELECT COUNT(*) FROM mechanism_cluster_membership WHERE cluster_id = 2")
        assert cursor.fetchone()[0] == 0  # No members

        cursor.execute("SELECT COUNT(*) FROM mechanism_clusters WHERE cluster_id = 2")
        assert cursor.fetchone()[0] == 0  # Cluster deleted

        conn.close()


class TestCreateParent:
    """Test CREATE_PARENT operations."""

    def test_create_parent_basic(self):
        """Test creating parent cluster with children."""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        # Setup schema
        cursor.execute("""
            CREATE TABLE mechanism_clusters (
                cluster_id INTEGER PRIMARY KEY,
                canonical_name TEXT,
                parent_cluster_id INTEGER,
                hierarchy_level INTEGER,
                member_count INTEGER
            )
        """)

        cursor.execute("INSERT INTO mechanism_clusters VALUES (1, 'Child A', NULL, 0, 2)")
        cursor.execute("INSERT INTO mechanism_clusters VALUES (2, 'Child B', NULL, 0, 3)")

        conn.commit()

        # Create mock merge
        cluster_a = Cluster(1, "Child A", ["m1", "m2"], None, 0)
        cluster_b = Cluster(2, "Child B", ["m3", "m4", "m5"], None, 0)

        candidate = MergeCandidate(1, 2, cluster_a, cluster_b, 0.88, 'MEDIUM')

        merge = LLMValidationResult(
            candidate=candidate,
            relationship_type='CREATE_PARENT',
            llm_confidence='HIGH',
            suggested_parent_name='Parent Cluster',
            child_a_refined_name='Refined Child A',
            child_b_refined_name='Refined Child B',
            llm_reasoning='Related',
            name_quality=NameQualityScore(75, [], True),
            diversity_check=DiversityCheck(False, 'NONE', 0.7, 'Good'),
            auto_approved=True,
            flagged_reason=None
        )

        # Apply merge
        applicator = MergeApplicator(':memory:', 'mechanism')
        parent_id = applicator._apply_create_parent(conn, merge, target_level=1)

        # Verify parent created
        assert parent_id == 3  # Next available ID

        cursor.execute("SELECT canonical_name, hierarchy_level, member_count FROM mechanism_clusters WHERE cluster_id = 3")
        row = cursor.fetchone()
        assert row[0] == 'Parent Cluster'
        assert row[1] == 1  # hierarchy_level
        assert row[2] == 5  # total members (2+3)

        # Verify children updated
        cursor.execute("SELECT parent_cluster_id, hierarchy_level, canonical_name FROM mechanism_clusters WHERE cluster_id = 1")
        row = cursor.fetchone()
        assert row[0] == 3  # Points to parent
        assert row[1] == 2  # Level 2 (child of level 1)
        assert row[2] == 'Refined Child A'

        conn.close()


class TestValidation:
    """Test hierarchy validation."""

    def test_validation_no_errors(self):
        """Test validation with valid hierarchy."""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE mechanism_clusters (
                cluster_id INTEGER PRIMARY KEY,
                canonical_name TEXT,
                parent_cluster_id INTEGER,
                hierarchy_level INTEGER,
                member_count INTEGER
            )
        """)

        # Valid hierarchy: parent (3) with two children (1, 2)
        cursor.execute("INSERT INTO mechanism_clusters VALUES (1, 'Child A', 3, 1, 2)")
        cursor.execute("INSERT INTO mechanism_clusters VALUES (2, 'Child B', 3, 1, 2)")
        cursor.execute("INSERT INTO mechanism_clusters VALUES (3, 'Parent', NULL, 0, 4)")

        conn.commit()

        applicator = MergeApplicator(':memory:', 'mechanism')
        is_valid, errors = applicator._validate_hierarchy(conn)

        assert is_valid == True
        assert len(errors) == 0

        conn.close()

    def test_validation_orphaned_children(self):
        """Test validation catches orphaned children."""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE mechanism_clusters (
                cluster_id INTEGER PRIMARY KEY,
                canonical_name TEXT,
                parent_cluster_id INTEGER,
                hierarchy_level INTEGER,
                member_count INTEGER
            )
        """)

        # Orphaned child: references non-existent parent
        cursor.execute("INSERT INTO mechanism_clusters VALUES (1, 'Child', 999, 1, 2)")

        conn.commit()

        applicator = MergeApplicator(':memory:', 'mechanism')
        is_valid, errors = applicator._validate_hierarchy(conn)

        assert is_valid == False
        assert len(errors) > 0
        assert any('orphan' in e.lower() for e in errors)

        conn.close()

    def test_validation_level_inconsistency(self):
        """Test validation catches hierarchy level errors."""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE mechanism_clusters (
                cluster_id INTEGER PRIMARY KEY,
                canonical_name TEXT,
                parent_cluster_id INTEGER,
                hierarchy_level INTEGER,
                member_count INTEGER
            )
        """)

        # Invalid: child has same or higher level than parent
        cursor.execute("INSERT INTO mechanism_clusters VALUES (1, 'Child', 2, 0, 2)")  # Level 0
        cursor.execute("INSERT INTO mechanism_clusters VALUES (2, 'Parent', NULL, 1, 4)")  # Level 1

        conn.commit()

        applicator = MergeApplicator(':memory:', 'mechanism')
        is_valid, errors = applicator._validate_hierarchy(conn)

        assert is_valid == False
        assert any('level' in e.lower() for e in errors)

        conn.close()


class TestJunctionTables:
    """Test junction table updates."""

    def test_junction_table_update(self):
        """Test updating intervention_mechanisms junction table."""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()

        # Setup schema
        cursor.execute("""
            CREATE TABLE interventions (
                id INTEGER PRIMARY KEY,
                mechanism TEXT,
                health_condition TEXT,
                correlation_strength REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE mechanism_cluster_membership (
                mechanism_text TEXT,
                cluster_id INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE intervention_mechanisms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intervention_id INTEGER,
                mechanism_text TEXT,
                cluster_id INTEGER,
                health_condition TEXT,
                correlation_strength REAL
            )
        """)

        # Insert test data
        cursor.execute("INSERT INTO interventions VALUES (1, 'mech1', 'diabetes', 0.8)")
        cursor.execute("INSERT INTO mechanism_cluster_membership VALUES ('mech1', 10)")

        conn.commit()

        # Update junction tables
        applicator = MergeApplicator(':memory:', 'mechanism')
        rows_updated = applicator._update_junction_tables(conn)

        # Verify junction table populated
        cursor.execute("SELECT COUNT(*) FROM intervention_mechanisms")
        count = cursor.fetchone()[0]
        assert count == 1

        cursor.execute("SELECT cluster_id FROM intervention_mechanisms WHERE mechanism_text = 'mech1'")
        cluster_id = cursor.fetchone()[0]
        assert cluster_id == 10

        conn.close()


def run_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING STAGE 5 TESTS")
    print("="*60)

    test_classes = [
        TestMergeIdentical(),
        TestCreateParent(),
        TestValidation(),
        TestJunctionTables()
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")

        test_methods = [m for m in dir(test_class) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"  [PASS] {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  [FAIL] {method_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*60)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("="*60)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
