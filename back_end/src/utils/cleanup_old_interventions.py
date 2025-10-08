"""
Clean up interventions extracted before mechanism field was added.

Deletes:
- 579 interventions extracted before Oct 5, 2025 (no mechanism data)
- 91 papers that only have old interventions
- Semantic hierarchy entries for deleted interventions (if not in kept interventions)
- Entity relationships referencing deleted entries
- Empty canonical groups

Keeps:
- 792 interventions extracted Oct 5+ (all have mechanisms - 100% coverage)
- 197 papers with both old and new interventions (keeps papers, removes old interventions)
- Semantic hierarchy for intervention names that exist in kept interventions
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

class DatabaseCleaner:
    """Clean up pre-mechanism interventions from database."""

    def __init__(self, db_path: str = None):
        """Initialize cleaner."""
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "processed" / "intervention_research.db"

        self.db_path = Path(db_path)
        self.cutoff_date = '2025-10-05'

    def get_cleanup_stats(self) -> dict:
        """Get statistics before cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        stats = {}

        # Interventions to delete
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM interventions
            WHERE extraction_timestamp < ?
        ''', (self.cutoff_date,))
        stats['interventions_to_delete'] = cursor.fetchone()['count']

        # Interventions to keep
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM interventions
            WHERE extraction_timestamp >= ?
        ''', (self.cutoff_date,))
        stats['interventions_to_keep'] = cursor.fetchone()['count']

        # Papers with only old interventions
        cursor.execute('''
            SELECT COUNT(DISTINCT paper_id) as count
            FROM interventions
            WHERE extraction_timestamp < ?
              AND paper_id NOT IN (
                SELECT DISTINCT paper_id FROM interventions WHERE extraction_timestamp >= ?
              )
        ''', (self.cutoff_date, self.cutoff_date))
        stats['papers_to_delete'] = cursor.fetchone()['count']

        # Papers to keep
        cursor.execute('''
            SELECT COUNT(DISTINCT paper_id) as count
            FROM interventions
            WHERE extraction_timestamp >= ?
        ''', (self.cutoff_date,))
        stats['papers_to_keep'] = cursor.fetchone()['count']

        # Semantic hierarchy entries
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM semantic_hierarchy
            WHERE entity_type = 'intervention'
        ''')
        stats['semantic_entries_total'] = cursor.fetchone()['count']

        conn.close()
        return stats

    def get_papers_to_delete(self) -> List[str]:
        """Get list of paper PMIDs that only have old interventions."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT paper_id
            FROM interventions
            WHERE extraction_timestamp < ?
              AND paper_id NOT IN (
                SELECT DISTINCT paper_id FROM interventions WHERE extraction_timestamp >= ?
              )
        ''', (self.cutoff_date, self.cutoff_date))

        pmids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return pmids

    def get_intervention_names_to_delete(self) -> List[str]:
        """
        Get intervention names that should be deleted from semantic_hierarchy.
        Only returns names that:
        1. Appear in old interventions (before Oct 5)
        2. Do NOT appear in kept interventions (Oct 5+)
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT intervention_name
            FROM interventions
            WHERE extraction_timestamp < ?
              AND intervention_name NOT IN (
                SELECT DISTINCT intervention_name FROM interventions WHERE extraction_timestamp >= ?
              )
        ''', (self.cutoff_date, self.cutoff_date))

        names = [row[0] for row in cursor.fetchall()]
        conn.close()
        return names

    def delete_old_interventions(self) -> int:
        """Delete interventions extracted before cutoff date."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            DELETE FROM interventions
            WHERE extraction_timestamp < ?
        ''', (self.cutoff_date,))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted

    def delete_papers(self, pmids: List[str]) -> int:
        """Delete papers by PMID list."""
        if not pmids:
            return 0

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Use parameterized query with placeholders
        placeholders = ','.join('?' * len(pmids))
        cursor.execute(f'''
            DELETE FROM papers
            WHERE pmid IN ({placeholders})
        ''', pmids)

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted

    def cleanup_semantic_hierarchy(self, intervention_names: List[str]) -> Tuple[int, int, int]:
        """
        Clean up semantic hierarchy for deleted interventions.

        Returns:
            (deleted_entries, deleted_relationships, updated_groups)
        """
        if not intervention_names:
            return 0, 0, 0

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get IDs of semantic entries to delete
        placeholders = ','.join('?' * len(intervention_names))
        cursor.execute(f'''
            SELECT id FROM semantic_hierarchy
            WHERE entity_type = 'intervention'
              AND entity_name IN ({placeholders})
        ''', intervention_names)

        entry_ids = [row[0] for row in cursor.fetchall()]

        if not entry_ids:
            conn.close()
            return 0, 0, 0

        # Delete entity relationships referencing these entries
        id_placeholders = ','.join('?' * len(entry_ids))
        cursor.execute(f'''
            DELETE FROM entity_relationships
            WHERE entity_1_id IN ({id_placeholders})
               OR entity_2_id IN ({id_placeholders})
        ''', entry_ids + entry_ids)

        deleted_relationships = cursor.rowcount

        # Delete semantic hierarchy entries
        cursor.execute(f'''
            DELETE FROM semantic_hierarchy
            WHERE id IN ({id_placeholders})
        ''', entry_ids)

        deleted_entries = cursor.rowcount

        # Update canonical groups member counts
        cursor.execute('''
            UPDATE canonical_groups
            SET member_count = (
                SELECT COUNT(*)
                FROM semantic_hierarchy
                WHERE layer_1_canonical = canonical_groups.canonical_name
                  AND entity_type = canonical_groups.entity_type
            )
        ''')

        # Delete empty canonical groups
        cursor.execute('''
            DELETE FROM canonical_groups
            WHERE member_count = 0
        ''')

        updated_groups = cursor.rowcount

        conn.commit()
        conn.close()

        return deleted_entries, deleted_relationships, updated_groups

    def verify_cleanup(self) -> dict:
        """Verify cleanup results."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        verification = {}

        # Check remaining interventions
        cursor.execute('SELECT COUNT(*) as count FROM interventions')
        verification['total_interventions'] = cursor.fetchone()['count']

        # Check mechanism coverage
        cursor.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN mechanism IS NOT NULL THEN 1 ELSE 0 END) as has_mechanism
            FROM interventions
        ''')
        row = cursor.fetchone()
        verification['interventions_with_mechanism'] = row['has_mechanism']
        verification['mechanism_coverage_pct'] = (row['has_mechanism'] / row['total'] * 100) if row['total'] > 0 else 0

        # Check remaining papers
        cursor.execute('SELECT COUNT(*) as count FROM papers')
        verification['total_papers'] = cursor.fetchone()['count']

        # Check semantic hierarchy
        cursor.execute('''
            SELECT COUNT(*) as count
            FROM semantic_hierarchy
            WHERE entity_type = 'intervention'
        ''')
        verification['semantic_intervention_entries'] = cursor.fetchone()['count']

        cursor.execute('SELECT COUNT(*) as count FROM entity_relationships')
        verification['entity_relationships'] = cursor.fetchone()['count']

        cursor.execute('SELECT COUNT(*) as count FROM canonical_groups')
        verification['canonical_groups'] = cursor.fetchone()['count']

        conn.close()
        return verification

    def run_cleanup(self, dry_run: bool = False):
        """
        Run the complete cleanup process.

        Args:
            dry_run: If True, only shows what would be deleted without making changes
        """
        print('='*70)
        print('DATABASE CLEANUP: Remove Pre-Mechanism Interventions')
        print('='*70)

        # Get initial stats
        print('\n1. ANALYZING DATABASE...')
        stats = self.get_cleanup_stats()

        print(f'\nCurrent State:')
        print(f'  Interventions to delete: {stats["interventions_to_delete"]:,}')
        print(f'  Interventions to keep: {stats["interventions_to_keep"]:,}')
        print(f'  Papers to delete: {stats["papers_to_delete"]:,}')
        print(f'  Papers to keep: {stats["papers_to_keep"]:,}')
        print(f'  Semantic entries: {stats["semantic_entries_total"]:,}')

        if dry_run:
            print('\n[DRY RUN] No changes will be made.')
            return

        # Get deletion lists
        print('\n2. PREPARING DELETION LISTS...')
        papers_to_delete = self.get_papers_to_delete()
        interventions_to_delete = self.get_intervention_names_to_delete()

        print(f'  Papers to delete: {len(papers_to_delete)}')
        print(f'  Intervention names to remove from semantic_hierarchy: {len(interventions_to_delete)}')

        # Execute deletions
        print('\n3. DELETING OLD INTERVENTIONS...')
        deleted_interventions = self.delete_old_interventions()
        print(f'  Deleted {deleted_interventions:,} interventions')

        print('\n4. DELETING PAPERS WITH ONLY OLD INTERVENTIONS...')
        deleted_papers = self.delete_papers(papers_to_delete)
        print(f'  Deleted {deleted_papers:,} papers')

        print('\n5. CLEANING UP SEMANTIC HIERARCHY...')
        deleted_entries, deleted_rels, updated_groups = self.cleanup_semantic_hierarchy(interventions_to_delete)
        print(f'  Deleted {deleted_entries:,} semantic entries')
        print(f'  Deleted {deleted_rels:,} entity relationships')
        print(f'  Updated/deleted {updated_groups:,} canonical groups')

        # Verify results
        print('\n6. VERIFYING RESULTS...')
        verification = self.verify_cleanup()

        print('\n' + '='*70)
        print('CLEANUP COMPLETE')
        print('='*70)
        print(f'\nFinal State:')
        print(f'  Total interventions: {verification["total_interventions"]:,}')
        print(f'  With mechanisms: {verification["interventions_with_mechanism"]:,} ({verification["mechanism_coverage_pct"]:.1f}%)')
        print(f'  Total papers: {verification["total_papers"]:,}')
        print(f'  Semantic intervention entries: {verification["semantic_intervention_entries"]:,}')
        print(f'  Entity relationships: {verification["entity_relationships"]:,}')
        print(f'  Canonical groups: {verification["canonical_groups"]:,}')
        print('='*70)


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Clean up pre-mechanism interventions')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without making changes')

    args = parser.parse_args()

    cleaner = DatabaseCleaner()
    cleaner.run_cleanup(dry_run=args.dry_run)


if __name__ == "__main__":
    main()