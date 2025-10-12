"""
Database Migration Script: Old 6-Type to New 5-Type Layer-Based Taxonomy

Migrates existing relationship types from the old taxonomy to the new layer-based taxonomy:

OLD TAXONOMY (6 types):                NEW TAXONOMY (5 types):
- EXACT_MATCH        →                 - EXACT_MATCH
- VARIANT            →                 - SAME_CATEGORY_TYPE_VARIANT
- SUBTYPE            →                 - SAME_CATEGORY_TYPE_VARIANT
- SAME_CATEGORY      →                 - SAME_CATEGORY (or SAME_CATEGORY_TYPE_VARIANT if share Layer 1)
- DOSAGE_VARIANT     →                 - DOSAGE_VARIANT
- DIFFERENT          →                 - DIFFERENT

Usage:
    python migrate_relationship_taxonomy.py --db path/to/intervention_research.db [--dry-run]
"""

import sqlite3
import argparse
import sys
from pathlib import Path
from datetime import datetime

def check_relationship_table_exists(conn):
    """Check if entity_relationships table exists."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='entity_relationships'
    """)
    return cursor.fetchone() is not None

def get_relationship_counts(conn):
    """Get current counts of relationship types."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT relationship_type, COUNT(*) as count
        FROM entity_relationships
        GROUP BY relationship_type
        ORDER BY count DESC
    """)
    return {row[0]: row[1] for row in cursor.fetchall()}

def migrate_relationships(conn, dry_run=False):
    """Migrate old relationship types to new taxonomy."""
    cursor = conn.cursor()

    # Migration mapping
    migrations = []

    # 1. VARIANT → SAME_CATEGORY_TYPE_VARIANT (Layer 2 differs)
    migrations.append({
        'old_type': 'VARIANT',
        'new_type': 'SAME_CATEGORY_TYPE_VARIANT',
        'description': 'Same canonical group, different formulation/biosimilar (Layer 2 differs)'
    })

    # 2. SUBTYPE → SAME_CATEGORY_TYPE_VARIANT (Layer 2 differs)
    migrations.append({
        'old_type': 'SUBTYPE',
        'new_type': 'SAME_CATEGORY_TYPE_VARIANT',
        'description': 'Same canonical group, clinically distinct subtypes (Layer 2 differs)'
    })

    # 3. SAME_CATEGORY - Need to check if share Layer 1
    # If share_layer_1 = TRUE, migrate to SAME_CATEGORY_TYPE_VARIANT
    # If share_layer_1 = FALSE, keep as SAME_CATEGORY
    cursor.execute("""
        SELECT COUNT(*)
        FROM entity_relationships
        WHERE relationship_type = 'SAME_CATEGORY' AND share_layer_1 = 1
    """)
    same_cat_with_shared_layer1 = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM entity_relationships
        WHERE relationship_type = 'SAME_CATEGORY' AND (share_layer_1 = 0 OR share_layer_1 IS NULL)
    """)
    same_cat_without_shared_layer1 = cursor.fetchone()[0]

    if same_cat_with_shared_layer1 > 0:
        migrations.append({
            'old_type': 'SAME_CATEGORY (share_layer_1=TRUE)',
            'new_type': 'SAME_CATEGORY_TYPE_VARIANT',
            'description': f'Same canonical group, different entities ({same_cat_with_shared_layer1} records)',
            'sql': """
                UPDATE entity_relationships
                SET relationship_type = 'SAME_CATEGORY_TYPE_VARIANT'
                WHERE relationship_type = 'SAME_CATEGORY' AND share_layer_1 = 1
            """
        })

    if same_cat_without_shared_layer1 > 0:
        migrations.append({
            'old_type': 'SAME_CATEGORY (share_layer_1=FALSE)',
            'new_type': 'SAME_CATEGORY',
            'description': f'Different canonical groups, same taxonomy (Layer 1 differs) ({same_cat_without_shared_layer1} records)',
            'sql': None  # No change needed
        })

    # Print migration plan
    print("\n" + "="*80)
    print("RELATIONSHIP TAXONOMY MIGRATION PLAN")
    print("="*80)
    print(f"\nDatabase: {conn.execute('PRAGMA database_list').fetchone()[2]}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'DRY RUN (no changes will be made)' if dry_run else 'LIVE MIGRATION'}")
    print("\n" + "-"*80)

    total_affected = 0

    for migration in migrations:
        old_type = migration['old_type']
        new_type = migration['new_type']
        desc = migration['description']

        # Count affected rows
        if 'sql' in migration and migration['sql']:
            # Custom query for SAME_CATEGORY split
            cursor.execute(migration['sql'].replace('UPDATE', 'SELECT COUNT(*) FROM').split('SET')[0])
            count = cursor.fetchone()[0]
        elif '(' in old_type:
            # Already counted above for SAME_CATEGORY cases
            count = same_cat_with_shared_layer1 if 'TRUE' in old_type else same_cat_without_shared_layer1
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM entity_relationships
                WHERE relationship_type = ?
            """, (old_type,))
            count = cursor.fetchone()[0]

        if count > 0:
            print(f"\n{old_type:30s} → {new_type:30s}")
            print(f"  Description: {desc}")
            print(f"  Affected rows: {count}")
            total_affected += count

    print("\n" + "-"*80)
    print(f"TOTAL ROWS TO MIGRATE: {total_affected}")
    print("="*80 + "\n")

    if total_affected == 0:
        print("No relationships need migration. Database is already using new taxonomy.")
        return 0

    if dry_run:
        print("DRY RUN: No changes made to database.")
        print("Run without --dry-run flag to apply migrations.")
        return 0

    # Execute migrations
    print("Executing migrations...")

    # 1. Migrate VARIANT → SAME_CATEGORY_TYPE_VARIANT
    cursor.execute("""
        UPDATE entity_relationships
        SET relationship_type = 'SAME_CATEGORY_TYPE_VARIANT'
        WHERE relationship_type = 'VARIANT'
    """)
    variant_count = cursor.rowcount
    print(f"  ✓ Migrated {variant_count} VARIANT relationships")

    # 2. Migrate SUBTYPE → SAME_CATEGORY_TYPE_VARIANT
    cursor.execute("""
        UPDATE entity_relationships
        SET relationship_type = 'SAME_CATEGORY_TYPE_VARIANT'
        WHERE relationship_type = 'SUBTYPE'
    """)
    subtype_count = cursor.rowcount
    print(f"  ✓ Migrated {subtype_count} SUBTYPE relationships")

    # 3. Migrate SAME_CATEGORY (with share_layer_1=TRUE) → SAME_CATEGORY_TYPE_VARIANT
    cursor.execute("""
        UPDATE entity_relationships
        SET relationship_type = 'SAME_CATEGORY_TYPE_VARIANT'
        WHERE relationship_type = 'SAME_CATEGORY' AND share_layer_1 = 1
    """)
    same_cat_migrated = cursor.rowcount
    print(f"  ✓ Migrated {same_cat_migrated} SAME_CATEGORY (with Layer 1 shared) relationships")

    # 4. SAME_CATEGORY (without share_layer_1) remains unchanged
    cursor.execute("""
        SELECT COUNT(*) FROM entity_relationships
        WHERE relationship_type = 'SAME_CATEGORY' AND (share_layer_1 = 0 OR share_layer_1 IS NULL)
    """)
    same_cat_unchanged = cursor.fetchone()[0]
    print(f"  ✓ Kept {same_cat_unchanged} SAME_CATEGORY (Layer 1 differs) relationships unchanged")

    conn.commit()

    print(f"\n✓ Migration complete! {total_affected} relationships updated.")

    # Show new distribution
    print("\n" + "="*80)
    print("NEW RELATIONSHIP TYPE DISTRIBUTION")
    print("="*80)
    new_counts = get_relationship_counts(conn)
    for rel_type, count in new_counts.items():
        print(f"  {rel_type:30s}: {count:5d}")
    print("="*80 + "\n")

    return total_affected

def main():
    parser = argparse.ArgumentParser(
        description="Migrate relationship taxonomy from 6-type to 5-type layer-based taxonomy"
    )
    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='Path to intervention_research.db'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show migration plan without making changes'
    )

    args = parser.parse_args()

    # Validate database path
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        sys.exit(1)

    # Connect to database
    try:
        conn = sqlite3.connect(str(db_path))
        print(f"\n✓ Connected to database: {db_path}")
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}")
        sys.exit(1)

    # Check if relationship table exists
    if not check_relationship_table_exists(conn):
        print("\nINFO: entity_relationships table does not exist.")
        print("No migration needed. This is expected if Phase 3 hasn't been run yet.")
        conn.close()
        sys.exit(0)

    # Get current relationship counts
    print("\nCURRENT RELATIONSHIP TYPE DISTRIBUTION:")
    print("-"*80)
    current_counts = get_relationship_counts(conn)
    if not current_counts:
        print("  No relationships found in database.")
        conn.close()
        sys.exit(0)

    for rel_type, count in current_counts.items():
        print(f"  {rel_type:30s}: {count:5d}")
    print("-"*80)

    # Run migration
    try:
        migrated_count = migrate_relationships(conn, dry_run=args.dry_run)
        conn.close()

        if migrated_count > 0 and not args.dry_run:
            print(f"\n✓ SUCCESS: Migration complete!")
            print(f"  Backup recommendation: Create a backup of {db_path.name} before running Phase 3 again.")

        sys.exit(0)

    except Exception as e:
        print(f"\nERROR: Migration failed: {e}")
        conn.rollback()
        conn.close()
        sys.exit(1)

if __name__ == "__main__":
    main()
