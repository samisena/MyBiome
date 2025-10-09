"""
Database Migration: Create v_interventions_categorized VIEW (Option B)

Creates a VIEW that dynamically computes intervention categories from canonical groups,
establishing single source of truth in canonical_groups.layer_0_category.

This is Phase 3 of the group-based categorization migration:
- Phase 1: Experiment (complete)
- Phase 2: Integration with Option A - persist in interventions table (complete)
- Phase 3: Migrate to Option B - VIEW-based architecture (this script)

Benefits of Option B:
- Single source of truth (categories stored ONCE in canonical_groups)
- Auto-sync: group category changes propagate automatically
- Cleaner architecture: true hierarchical inheritance
- Future-proof: easier to add sub-categories, multi-level hierarchies

Usage:
    python -m back_end.src.migrations.create_interventions_view_option_b

    # Check VIEW exists
    python -m back_end.src.migrations.create_interventions_view_option_b --status

    # Drop VIEW (for rollback)
    python -m back_end.src.migrations.create_interventions_view_option_b --drop
"""

import sqlite3
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path
root_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_dir))

try:
    from back_end.src.data.config import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.config import config


def create_view(conn: sqlite3.Connection):
    """
    Create v_interventions_categorized VIEW.

    This VIEW dynamically computes intervention_category by:
    1. Joining with semantic_hierarchy to get canonical group
    2. Joining with canonical_groups to get group's category
    3. Falling back to interventions.intervention_category for orphans
    """
    print("Creating v_interventions_categorized VIEW...")

    conn.execute("""
        CREATE VIEW IF NOT EXISTS v_interventions_categorized AS
        SELECT
            -- All original intervention fields
            i.id,
            i.paper_id,
            i.intervention_name,
            i.intervention_details,
            i.health_condition,
            i.mechanism,
            i.correlation_type,
            i.correlation_strength,
            i.extraction_confidence,
            i.study_confidence,
            i.sample_size,
            i.study_duration,
            i.study_type,
            i.population_details,
            i.delivery_method,
            i.severity,
            i.adverse_effects,
            i.cost_category,
            i.supporting_quote,
            i.extraction_model,
            i.extraction_timestamp,

            -- Dynamic category (from group or fallback)
            COALESCE(
                cg.layer_0_category,            -- Inherit from canonical group (primary)
                i.intervention_category         -- Fallback for orphans
            ) AS intervention_category,

            -- Condition category (no change)
            i.condition_category,

            -- Expose semantic hierarchy info for context
            sh.layer_1_canonical AS canonical_group_name,
            sh.layer_2_variant AS variant_name,
            sh.layer_3_detail AS detail_name,
            sh.relationship_type,

            -- Expose canonical group metadata
            cg.member_count AS group_member_count,
            cg.total_paper_count AS group_paper_count,

            -- Flag: is this from group or orphan fallback?
            CASE
                WHEN cg.layer_0_category IS NOT NULL THEN 'group'
                WHEN i.intervention_category IS NOT NULL THEN 'orphan'
                ELSE 'uncategorized'
            END AS category_source

        FROM interventions i

        -- LEFT JOIN to semantic_hierarchy (many-to-one: intervention → group)
        LEFT JOIN semantic_hierarchy sh
            ON i.intervention_name = sh.entity_name
            AND sh.entity_type = 'intervention'

        -- LEFT JOIN to canonical_groups (one-to-one: group → category)
        LEFT JOIN canonical_groups cg
            ON sh.layer_1_canonical = cg.canonical_name
            AND cg.entity_type = 'intervention'
    """)

    print("[OK] v_interventions_categorized VIEW created")


def drop_view(conn: sqlite3.Connection):
    """Drop v_interventions_categorized VIEW."""
    print("Dropping v_interventions_categorized VIEW...")

    conn.execute("DROP VIEW IF EXISTS v_interventions_categorized")

    print("[OK] VIEW dropped")


def check_view_status(conn: sqlite3.Connection) -> dict:
    """Check if VIEW exists and get sample data."""
    cursor = conn.cursor()

    # Check if VIEW exists
    cursor.execute("""
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type='view' AND name='v_interventions_categorized'
    """)
    view_exists = cursor.fetchone()[0] > 0

    if not view_exists:
        return {'exists': False}

    # Get row count
    cursor.execute("SELECT COUNT(*) FROM v_interventions_categorized")
    total_rows = cursor.fetchone()[0]

    # Get categorization breakdown
    cursor.execute("""
        SELECT
            category_source,
            COUNT(*) as count
        FROM v_interventions_categorized
        GROUP BY category_source
    """)
    breakdown = {row[0]: row[1] for row in cursor.fetchall()}

    # Sample data
    cursor.execute("""
        SELECT
            intervention_name,
            intervention_category,
            canonical_group_name,
            category_source
        FROM v_interventions_categorized
        LIMIT 5
    """)
    sample = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]

    return {
        'exists': True,
        'total_rows': total_rows,
        'breakdown': breakdown,
        'sample': sample
    }


def validate_view(conn: sqlite3.Connection) -> dict:
    """Validate VIEW correctness."""
    cursor = conn.cursor()

    results = {}

    # 1. Check: All interventions have categories
    cursor.execute("""
        SELECT COUNT(*)
        FROM v_interventions_categorized
        WHERE intervention_category IS NULL OR intervention_category = ''
    """)
    uncategorized = cursor.fetchone()[0]
    results['all_categorized'] = uncategorized == 0
    results['uncategorized_count'] = uncategorized

    # 2. Check: Category values are valid
    valid_categories = [
        'exercise', 'diet', 'supplement', 'medication', 'therapy',
        'lifestyle', 'surgery', 'test', 'device', 'procedure',
        'biologics', 'gene_therapy', 'emerging'
    ]

    cursor.execute(f"""
        SELECT COUNT(*)
        FROM v_interventions_categorized
        WHERE intervention_category NOT IN ({','.join(['?'] * len(valid_categories))})
        AND intervention_category IS NOT NULL
    """, valid_categories)
    invalid_categories = cursor.fetchone()[0]
    results['all_valid_categories'] = invalid_categories == 0
    results['invalid_count'] = invalid_categories

    # 3. Check: Group-based categories match group definitions
    cursor.execute("""
        SELECT COUNT(*)
        FROM v_interventions_categorized v
        JOIN semantic_hierarchy sh ON v.intervention_name = sh.entity_name
        JOIN canonical_groups cg ON sh.layer_1_canonical = cg.canonical_name
        WHERE v.category_source = 'group'
        AND v.intervention_category != cg.layer_0_category
    """)
    mismatched = cursor.fetchone()[0]
    results['group_categories_match'] = mismatched == 0
    results['mismatched_count'] = mismatched

    # 4. Overall validation
    results['valid'] = all([
        results['all_categorized'],
        results['all_valid_categories'],
        results['group_categories_match']
    ])

    return results


def main():
    """Run migration to create v_interventions_categorized VIEW."""
    parser = argparse.ArgumentParser(description="Create v_interventions_categorized VIEW (Option B)")
    parser.add_argument('--drop', action='store_true', help='Drop VIEW (for rollback)')
    parser.add_argument('--status', action='store_true', help='Check VIEW status')
    parser.add_argument('--validate', action='store_true', help='Validate VIEW correctness')
    parser.add_argument('--db-path', type=str, default=None, help='Override database path')

    args = parser.parse_args()

    # Get database path
    db_path = args.db_path or config.db_path

    print("=" * 80)
    print("VIEW MIGRATION: v_interventions_categorized (Option B)")
    print("=" * 80)
    print(f"Database: {db_path}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Connect to database
    conn = sqlite3.connect(db_path)

    try:
        if args.status:
            # Check status
            status = check_view_status(conn)

            if status['exists']:
                print("[OK] VIEW exists")
                print(f"Total rows: {status['total_rows']}")
                print(f"\nCategorization breakdown:")
                for source, count in status['breakdown'].items():
                    print(f"  {source}: {count} ({count/status['total_rows']*100:.1f}%)")

                print(f"\nSample data (first 5 rows):")
                for row in status['sample']:
                    print(f"  {row['intervention_name']}: {row['intervention_category']} (source: {row['category_source']})")
            else:
                print("[NOT FOUND] VIEW does not exist")

        elif args.validate:
            # Validate VIEW
            validation = validate_view(conn)

            print("Validation results:")
            if validation['all_categorized']:
                print("  All categorized: [OK] PASSED")
            else:
                print(f"  All categorized: [FAIL] ({validation['uncategorized_count']} uncategorized)")

            if validation['all_valid_categories']:
                print("  All valid categories: [OK] PASSED")
            else:
                print(f"  All valid categories: [FAIL] ({validation['invalid_count']} invalid)")

            if validation['group_categories_match']:
                print("  Group categories match: [OK] PASSED")
            else:
                print(f"  Group categories match: [FAIL] ({validation['mismatched_count']} mismatched)")

            print(f"\nOverall: {'[VALID]' if validation['valid'] else '[INVALID]'}")

        elif args.drop:
            # Drop VIEW
            drop_view(conn)

        else:
            # Create VIEW
            create_view(conn)

            # Commit
            conn.commit()

            # Show status
            status = check_view_status(conn)
            print(f"\nVIEW created successfully:")
            print(f"  Total rows: {status['total_rows']}")
            print(f"  Categorization breakdown:")
            for source, count in status['breakdown'].items():
                print(f"    {source}: {count} ({count/status['total_rows']*100:.1f}%)")

            # Auto-validate
            print("\nRunning validation...")
            validation = validate_view(conn)
            if validation['valid']:
                print("[OK] Validation PASSED")
            else:
                print("[FAIL] Validation FAILED - see details above")

        print("\n" + "=" * 80)
        print("MIGRATION COMPLETE")
        print("=" * 80)

    except Exception as e:
        conn.rollback()
        print(f"\n[ERROR] Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
