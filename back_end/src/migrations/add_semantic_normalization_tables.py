"""
Database Migration: Add Semantic Normalization Tables

Creates hierarchical semantic normalization schema:
- semantic_hierarchy (main 4-layer hierarchical structure)
- canonical_groups (Layer 1 aggregation entities)

Note: entity_relationships table removed - relationship analysis moved to Phase 3d (cluster-level).
Plus indexes and views for efficient querying.

Usage:
    python -m back_end.src.migrations.add_semantic_normalization_tables [--drop-old]

Options:
    --drop-old: Drop old canonical_entities and entity_mappings tables (DESTRUCTIVE)
"""

import sqlite3
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path
root_dir = Path(__file__).parent.parent.parent.parent  # MyBiome root
sys.path.insert(0, str(root_dir))

try:
    from back_end.src.data.config import config
except ImportError:
    # Fallback: use relative imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.config import config


def create_semantic_hierarchy_table(conn: sqlite3.Connection):
    """Create semantic_hierarchy table with 4-layer structure."""
    print("Creating semantic_hierarchy table...")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_hierarchy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Entity Information
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL CHECK(entity_type IN ('intervention', 'condition')),

            -- Hierarchical Layers
            layer_0_category TEXT,                      -- From existing taxonomy
            layer_1_canonical TEXT,                     -- Semantic group (e.g., "probiotics", "statins")
            layer_2_variant TEXT,                       -- Specific entity (e.g., "L. reuteri", "atorvastatin")
            layer_3_detail TEXT,                        -- Dosage/details (e.g., "atorvastatin 20mg")

            -- Parent-Child Relationships
            parent_id INTEGER,
            relationship_type TEXT,                     -- EXACT_MATCH, VARIANT, SUBTYPE, etc.
            aggregation_rule TEXT,

            -- Semantic Embedding (for similarity matching)
            embedding_vector BLOB,                      -- Binary serialized embedding
            embedding_model TEXT,                       -- Model name (e.g., "nomic-embed-text")
            embedding_dimension INTEGER,                -- Dimension (e.g., 768)

            -- Metadata
            source_table TEXT,                          -- 'interventions' or 'health_conditions'
            source_ids TEXT,                            -- JSON array of source IDs
            occurrence_count INTEGER DEFAULT 1,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Constraints
            FOREIGN KEY (parent_id) REFERENCES semantic_hierarchy(id) ON DELETE CASCADE,
            UNIQUE(entity_name, entity_type, layer_2_variant)
        )
    """)

    # Create indexes
    print("Creating indexes for semantic_hierarchy...")

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hierarchy_layers ON semantic_hierarchy(
            entity_type, layer_1_canonical, layer_2_variant
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hierarchy_parent ON semantic_hierarchy(parent_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hierarchy_entity ON semantic_hierarchy(entity_name, entity_type)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hierarchy_category ON semantic_hierarchy(layer_0_category)
    """)

    print("[OK] semantic_hierarchy table created")




def create_canonical_groups_table(conn: sqlite3.Connection):
    """Create canonical_groups table for Layer 1 aggregation."""
    print("Creating canonical_groups table...")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS canonical_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            -- Group Information
            canonical_name TEXT NOT NULL UNIQUE,
            display_name TEXT,
            entity_type TEXT NOT NULL CHECK(entity_type IN ('intervention', 'condition')),

            -- Category
            layer_0_category TEXT,

            -- Description
            description TEXT,

            -- Aggregation Metadata
            member_count INTEGER DEFAULT 0,
            total_paper_count INTEGER DEFAULT 0,

            -- Semantic Embedding
            group_embedding BLOB,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes
    print("Creating indexes for canonical_groups...")

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_canonical_name ON canonical_groups(canonical_name)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_canonical_category ON canonical_groups(layer_0_category)
    """)

    print("[OK] canonical_groups table created")


def create_views(conn: sqlite3.Connection):
    """Create helpful views for common queries."""
    print("Creating views...")

    # View: All interventions with full hierarchical context
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_intervention_hierarchy AS
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
        WHERE sh.entity_type = 'intervention'
    """)

    # View: Intervention aggregation by Layer 1 (canonical group)
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_intervention_by_canonical AS
        SELECT
            layer_1_canonical,
            layer_0_category,
            COUNT(DISTINCT layer_2_variant) AS variant_count,
            SUM(occurrence_count) AS total_occurrences,
            GROUP_CONCAT(DISTINCT layer_2_variant, ', ') AS variants
        FROM semantic_hierarchy
        WHERE entity_type = 'intervention'
        AND layer_1_canonical IS NOT NULL
        GROUP BY layer_1_canonical, layer_0_category
    """)

    # View: Intervention aggregation by Layer 2 (specific variant)
    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_intervention_by_variant AS
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
        GROUP BY layer_2_variant, layer_1_canonical, layer_0_category
    """)

    print("[OK] Views created")


def check_old_tables(conn: sqlite3.Connection) -> dict:
    """Check if old canonical tables exist and get their row counts."""
    cursor = conn.cursor()

    old_tables = {}

    # Check canonical_entities
    try:
        cursor.execute("SELECT COUNT(*) FROM canonical_entities")
        old_tables['canonical_entities'] = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        old_tables['canonical_entities'] = None

    # Check entity_mappings
    try:
        cursor.execute("SELECT COUNT(*) FROM entity_mappings")
        old_tables['entity_mappings'] = cursor.fetchone()[0]
    except sqlite3.OperationalError:
        old_tables['entity_mappings'] = None

    return old_tables


def drop_old_tables(conn: sqlite3.Connection):
    """Drop old canonical_entities and entity_mappings tables."""
    print("\n[WARNING]  DROPPING OLD TABLES (this is DESTRUCTIVE)...")

    old_tables = check_old_tables(conn)

    if old_tables['canonical_entities'] is not None:
        print(f"Dropping canonical_entities ({old_tables['canonical_entities']} rows)...")
        conn.execute("DROP TABLE IF EXISTS canonical_entities")

    if old_tables['entity_mappings'] is not None:
        print(f"Dropping entity_mappings ({old_tables['entity_mappings']} rows)...")
        conn.execute("DROP TABLE IF EXISTS entity_mappings")

    print("[OK] Old tables dropped")


def check_new_tables(conn: sqlite3.Connection) -> dict:
    """Check if new tables already exist."""
    cursor = conn.cursor()

    new_tables = {}

    for table in ['semantic_hierarchy', 'canonical_groups']:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            new_tables[table] = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            new_tables[table] = None

    return new_tables


def main():
    """Run migration to add semantic normalization tables."""
    parser = argparse.ArgumentParser(description="Add semantic normalization tables to database")
    parser.add_argument('--drop-old', action='store_true',
                       help='Drop old canonical_entities and entity_mappings tables (DESTRUCTIVE)')
    parser.add_argument('--db-path', type=str, default=None,
                       help='Override database path from config')

    args = parser.parse_args()

    # Get database path
    db_path = args.db_path or config.db_path

    print("=" * 80)
    print("SEMANTIC NORMALIZATION DATABASE MIGRATION")
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints

    try:
        # Check existing tables
        print("Checking existing tables...")
        old_tables = check_old_tables(conn)
        new_tables = check_new_tables(conn)

        print("\nOld tables:")
        for table, count in old_tables.items():
            if count is not None:
                print(f"  - {table}: {count} rows")
            else:
                print(f"  - {table}: NOT FOUND")

        print("\nNew tables:")
        for table, count in new_tables.items():
            if count is not None:
                print(f"  - {table}: {count} rows (ALREADY EXISTS)")
            else:
                print(f"  - {table}: NOT FOUND (will create)")

        print()

        # Create new tables
        if new_tables['semantic_hierarchy'] is None:
            create_semantic_hierarchy_table(conn)
        else:
            print("[WARNING]  semantic_hierarchy already exists, skipping creation")

        if new_tables['canonical_groups'] is None:
            create_canonical_groups_table(conn)
        else:
            print("[WARNING]  canonical_groups already exists, skipping creation")

        # Create views
        create_views(conn)

        # Drop old tables if requested
        if args.drop_old:
            if old_tables['canonical_entities'] or old_tables['entity_mappings']:
                response = input("\n[WARNING]  Are you sure you want to drop old tables? (yes/no): ")
                if response.lower() == 'yes':
                    drop_old_tables(conn)
                else:
                    print("Skipping drop of old tables")
            else:
                print("\nNo old tables to drop")

        # Commit changes
        conn.commit()

        print("\n" + "=" * 80)
        print("MIGRATION COMPLETE")
        print("=" * 80)
        print("\nNew tables created:")
        print("  - semantic_hierarchy (4-layer hierarchical structure)")
        print("  - canonical_groups (Layer 1 aggregation entities)")
        print("\nViews created:")
        print("  - v_intervention_hierarchy")
        print("  - v_intervention_by_canonical")
        print("  - v_intervention_by_variant")
        print("\nNote: entity_relationships table removed - relationships handled at cluster-level in Phase 3d")

        if not args.drop_old and (old_tables['canonical_entities'] or old_tables['entity_mappings']):
            print("\n[WARNING]  Old tables still exist:")
            if old_tables['canonical_entities']:
                print(f"  - canonical_entities ({old_tables['canonical_entities']} rows)")
            if old_tables['entity_mappings']:
                print(f"  - entity_mappings ({old_tables['entity_mappings']} rows)")
            print("\nRun with --drop-old to remove them (after verifying migration)")

        print("\nNext steps:")
        print("  1. Run semantic normalization to populate tables:")
        print("     python -m back_end.src.orchestration.rotation_semantic_normalizer")
        print("  2. Verify data integrity")
        print("  3. Update data mining queries to use new schema")
        print("=" * 80 + "\n")

    except Exception as e:
        conn.rollback()
        print(f"\n[ERROR] Error during migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
