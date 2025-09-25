#!/usr/bin/env python3
"""
Normalization Admin CLI Tool

Simple command-line interface for managing canonical entities and mappings
without requiring SQL knowledge.

Features:
1. View canonical entities and their mappings
2. Manually merge two canonical entities (when duplicates found)
3. Review and approve medium-confidence mappings
4. Add new term mappings manually
5. View statistics dashboard

Usage:
    python admin_cli.py
"""

import sqlite3
import sys
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


class NormalizationAdmin:
    """Admin interface for normalization system management"""

    def __init__(self, db_path: str = "data/processed/intervention_research.db"):
        self.db_path = db_path
        self.ensure_connection()

    def ensure_connection(self):
        """Ensure database connection works"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
            print(f"[OK] Connected to database: {self.db_path}")
        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}")
            sys.exit(1)

    def show_main_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("NORMALIZATION ADMIN TOOL")
        print("="*60)
        print("1. View Statistics Dashboard")
        print("2. View Canonical Entities")
        print("3. View Entity Mappings")
        print("4. Add New Term Mapping")
        print("5. Merge Canonical Entities")
        print("6. Review Pending Mappings")
        print("7. Search Entities/Mappings")
        print("8. Export Data")
        print("0. Exit")
        print("-"*60)

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            stats = {}

            # Canonical entities stats
            cursor.execute("""
                SELECT entity_type, COUNT(*) as count
                FROM canonical_entities
                GROUP BY entity_type
            """)

            canonical_stats = {}
            for row in cursor.fetchall():
                canonical_stats[row['entity_type']] = row['count']

            # Entity mappings stats
            cursor.execute("""
                SELECT entity_type, COUNT(*) as count
                FROM entity_mappings
                GROUP BY entity_type
            """)

            mapping_stats = {}
            for row in cursor.fetchall():
                mapping_stats[row['entity_type']] = row['count']

            # Confidence distribution
            cursor.execute("""
                SELECT
                    CASE
                        WHEN confidence_score >= 0.9 THEN 'high'
                        WHEN confidence_score >= 0.7 THEN 'medium'
                        ELSE 'low'
                    END as confidence_level,
                    COUNT(*) as count
                FROM entity_mappings
                GROUP BY confidence_level
            """)

            confidence_stats = {}
            for row in cursor.fetchall():
                confidence_stats[row['confidence_level']] = row['count']

            # Intervention table stats
            cursor.execute("SELECT COUNT(*) FROM interventions")
            total_interventions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM interventions WHERE normalized = 1")
            normalized_interventions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM interventions WHERE intervention_canonical_id IS NOT NULL")
            intervention_mapped = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM interventions WHERE condition_canonical_id IS NOT NULL")
            condition_mapped = cursor.fetchone()[0]

            return {
                'canonical_entities': canonical_stats,
                'entity_mappings': mapping_stats,
                'confidence_distribution': confidence_stats,
                'intervention_stats': {
                    'total': total_interventions,
                    'normalized': normalized_interventions,
                    'intervention_mapped': intervention_mapped,
                    'condition_mapped': condition_mapped,
                    'normalization_percentage': round((normalized_interventions / max(total_interventions, 1)) * 100, 1)
                }
            }

    def display_statistics(self):
        """Display statistics dashboard"""
        print("\n" + "="*60)
        print("STATISTICS DASHBOARD")
        print("="*60)

        stats = self.get_statistics()

        print("\n[CANONICAL ENTITIES]")
        for entity_type, count in stats['canonical_entities'].items():
            print(f"  {entity_type.capitalize()}: {count}")

        print("\n[ENTITY MAPPINGS]")
        for entity_type, count in stats['entity_mappings'].items():
            print(f"  {entity_type.capitalize()}: {count}")

        print("\n[MAPPING CONFIDENCE]")
        for level, count in stats['confidence_distribution'].items():
            print(f"  {level.capitalize()}: {count}")

        print("\n[INTERVENTION RECORDS]")
        intervention_stats = stats['intervention_stats']
        print(f"  Total records: {intervention_stats['total']}")
        print(f"  Normalized: {intervention_stats['normalized']} ({intervention_stats['normalization_percentage']}%)")
        print(f"  With intervention mapping: {intervention_stats['intervention_mapped']}")
        print(f"  With condition mapping: {intervention_stats['condition_mapped']}")

        input("\nPress Enter to continue...")

    def view_canonical_entities(self, entity_type: Optional[str] = None, limit: int = 20):
        """View canonical entities with their mapping counts"""
        print("\n" + "="*60)
        print("CANONICAL ENTITIES")
        print("="*60)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if entity_type:
                where_clause = "WHERE ce.entity_type = ?"
                params = [entity_type, limit]
            else:
                where_clause = ""
                params = [limit]

            query = f"""
                SELECT ce.id, ce.canonical_name, ce.entity_type,
                       ce.confidence_score, ce.created_timestamp,
                       COUNT(em.id) as mapping_count
                FROM canonical_entities ce
                LEFT JOIN entity_mappings em ON ce.id = em.canonical_id
                {where_clause}
                GROUP BY ce.id, ce.canonical_name, ce.entity_type
                ORDER BY mapping_count DESC, ce.canonical_name
                LIMIT ?
            """

            cursor.execute(query, params)
            results = cursor.fetchall()

            if not results:
                print("No canonical entities found.")
                return

            print(f"\n{'ID':<5} {'Name':<30} {'Type':<12} {'Mappings':<8} {'Confidence':<10} {'Created'}")
            print("-" * 80)

            for row in results:
                created = row['created_timestamp'][:10] if row['created_timestamp'] else 'N/A'
                print(f"{row['id']:<5} {row['canonical_name'][:29]:<30} {row['entity_type']:<12} {row['mapping_count']:<8} {row['confidence_score']:<10.2f} {created}")

            print(f"\nShowing {len(results)} entities")

        # Sub-menu
        print("\nOptions:")
        print("1. View mappings for an entity")
        print("2. Filter by type (intervention/condition)")
        print("3. Show more entities")
        print("0. Back to main menu")

        choice = input("Choice: ").strip()

        if choice == "1":
            entity_id = input("Enter canonical entity ID: ").strip()
            try:
                self.view_entity_mappings(int(entity_id))
            except ValueError:
                print("Invalid ID format")
        elif choice == "2":
            entity_type = input("Enter type (intervention/condition): ").strip().lower()
            if entity_type in ['intervention', 'condition']:
                self.view_canonical_entities(entity_type, limit)
        elif choice == "3":
            self.view_canonical_entities(entity_type, limit + 20)

    def view_entity_mappings(self, canonical_id: int):
        """View all mappings for a canonical entity"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get canonical entity info
            cursor.execute("""
                SELECT canonical_name, entity_type
                FROM canonical_entities
                WHERE id = ?
            """, (canonical_id,))

            canonical = cursor.fetchone()
            if not canonical:
                print("Canonical entity not found.")
                return

            print(f"\n" + "="*60)
            print(f"MAPPINGS FOR: {canonical['canonical_name']} ({canonical['entity_type']})")
            print("="*60)

            # Get mappings
            cursor.execute("""
                SELECT id, raw_text, mapping_method, confidence_score,
                       created_timestamp, created_by
                FROM entity_mappings
                WHERE canonical_id = ?
                ORDER BY confidence_score DESC, raw_text
            """, (canonical_id,))

            mappings = cursor.fetchall()

            if not mappings:
                print("No mappings found for this entity.")
                return

            print(f"\n{'ID':<5} {'Raw Text':<35} {'Method':<12} {'Confidence':<10} {'Created'}")
            print("-" * 80)

            for mapping in mappings:
                created = mapping['created_timestamp'][:10] if mapping['created_timestamp'] else 'N/A'
                print(f"{mapping['id']:<5} {mapping['raw_text'][:34]:<35} {mapping['mapping_method']:<12} {mapping['confidence_score']:<10.2f} {created}")

            print(f"\nTotal mappings: {len(mappings)}")

            # Count usage in interventions
            cursor.execute(f"""
                SELECT COUNT(*) FROM interventions
                WHERE {canonical['entity_type']}_canonical_id = ?
            """, (canonical_id,))

            usage_count = cursor.fetchone()[0]
            print(f"Used in {usage_count} intervention records")

        input("\nPress Enter to continue...")

    def add_new_mapping(self):
        """Add a new term mapping manually"""
        print("\n" + "="*60)
        print("ADD NEW TERM MAPPING")
        print("="*60)

        # Get input
        raw_text = input("Enter raw text to map: ").strip()
        if not raw_text:
            print("Raw text cannot be empty.")
            return

        entity_type = input("Enter entity type (intervention/condition): ").strip().lower()
        if entity_type not in ['intervention', 'condition']:
            print("Entity type must be 'intervention' or 'condition'.")
            return

        print(f"\nFor '{raw_text}' ({entity_type}):")
        print("1. Map to existing canonical entity")
        print("2. Create new canonical entity")

        choice = input("Choice: ").strip()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if choice == "1":
                # Show existing canonicals
                cursor.execute("""
                    SELECT id, canonical_name
                    FROM canonical_entities
                    WHERE entity_type = ?
                    ORDER BY canonical_name
                    LIMIT 50
                """, (entity_type,))

                canonicals = cursor.fetchall()

                print(f"\nExisting {entity_type} canonicals:")
                for canonical in canonicals[:20]:  # Show first 20
                    print(f"  {canonical[0]}: {canonical[1]}")

                if len(canonicals) > 20:
                    print(f"  ... and {len(canonicals) - 20} more")

                canonical_id = input(f"\nEnter canonical ID to map to: ").strip()
                try:
                    canonical_id = int(canonical_id)
                except ValueError:
                    print("Invalid ID format.")
                    return

                # Verify canonical exists
                cursor.execute("SELECT canonical_name FROM canonical_entities WHERE id = ?", (canonical_id,))
                canonical_name = cursor.fetchone()
                if not canonical_name:
                    print("Canonical entity not found.")
                    return

                canonical_name = canonical_name[0]

            elif choice == "2":
                # Create new canonical
                canonical_name = input("Enter canonical name: ").strip()
                if not canonical_name:
                    print("Canonical name cannot be empty.")
                    return

                description = input("Enter description (optional): ").strip() or None

                # Create canonical
                cursor.execute("""
                    INSERT INTO canonical_entities (canonical_name, entity_type, description)
                    VALUES (?, ?, ?)
                """, (canonical_name, entity_type, description))

                canonical_id = cursor.lastrowid
                print(f"Created new canonical entity: '{canonical_name}' (ID: {canonical_id})")

            else:
                print("Invalid choice.")
                return

            # Check if mapping already exists
            cursor.execute("""
                SELECT id FROM entity_mappings
                WHERE raw_text = ? AND entity_type = ?
            """, (raw_text, entity_type))

            if cursor.fetchone():
                print(f"Mapping for '{raw_text}' already exists.")
                return

            # Add mapping
            confidence = input("Enter confidence score (0.0-1.0, default 1.0): ").strip()
            try:
                confidence = float(confidence) if confidence else 1.0
                if not 0 <= confidence <= 1:
                    raise ValueError()
            except ValueError:
                print("Invalid confidence score. Using 1.0.")
                confidence = 1.0

            cursor.execute("""
                INSERT INTO entity_mappings
                (raw_text, canonical_id, entity_type, mapping_method, confidence_score, created_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (raw_text, canonical_id, entity_type, 'manual', confidence, 'admin'))

            conn.commit()

            print(f"\n[SUCCESS] Added mapping:")
            print(f"  '{raw_text}' -> '{canonical_name}' (confidence: {confidence})")

        input("\nPress Enter to continue...")

    def merge_canonical_entities(self):
        """Merge two canonical entities (handle duplicates)"""
        print("\n" + "="*60)
        print("MERGE CANONICAL ENTITIES")
        print("="*60)

        print("This will merge two canonical entities by:")
        print("1. Moving all mappings from source to target")
        print("2. Updating intervention records")
        print("3. Deleting the source canonical")
        print("\nWarning: This action cannot be undone!")

        entity_type = input("\nEnter entity type (intervention/condition): ").strip().lower()
        if entity_type not in ['intervention', 'condition']:
            print("Entity type must be 'intervention' or 'condition'.")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Show canonicals
            cursor.execute("""
                SELECT ce.id, ce.canonical_name, COUNT(em.id) as mapping_count
                FROM canonical_entities ce
                LEFT JOIN entity_mappings em ON ce.id = em.canonical_id
                WHERE ce.entity_type = ?
                GROUP BY ce.id, ce.canonical_name
                ORDER BY ce.canonical_name
            """, (entity_type,))

            canonicals = cursor.fetchall()

            print(f"\n{entity_type.capitalize()} canonicals:")
            for canonical in canonicals:
                print(f"  {canonical['id']}: {canonical['canonical_name']} ({canonical['mapping_count']} mappings)")

            # Get source and target IDs
            try:
                source_id = int(input(f"\nEnter SOURCE canonical ID (will be deleted): ").strip())
                target_id = int(input(f"Enter TARGET canonical ID (will receive mappings): ").strip())
            except ValueError:
                print("Invalid ID format.")
                return

            if source_id == target_id:
                print("Source and target cannot be the same.")
                return

            # Verify entities exist
            cursor.execute("SELECT canonical_name FROM canonical_entities WHERE id IN (?, ?)", (source_id, target_id))
            entities = cursor.fetchall()

            if len(entities) != 2:
                print("One or both entities not found.")
                return

            source_name = next(e['canonical_name'] for e in entities if cursor.lastrowid != e['canonical_name'])
            target_name = next(e['canonical_name'] for e in entities if cursor.lastrowid != e['canonical_name'])

            print(f"\nMerging '{source_name}' -> '{target_name}'")
            confirm = input("Are you sure? (yes/no): ").strip().lower()

            if confirm != 'yes':
                print("Merge cancelled.")
                return

            try:
                # Start transaction
                cursor.execute("BEGIN TRANSACTION")

                # Move mappings
                cursor.execute("""
                    UPDATE entity_mappings
                    SET canonical_id = ?, created_by = 'admin_merge'
                    WHERE canonical_id = ?
                """, (target_id, source_id))

                moved_mappings = cursor.rowcount

                # Update intervention records
                field_name = f"{entity_type}_canonical_id"
                cursor.execute(f"""
                    UPDATE interventions
                    SET {field_name} = ?
                    WHERE {field_name} = ?
                """, (target_id, source_id))

                updated_interventions = cursor.rowcount

                # Delete source canonical
                cursor.execute("DELETE FROM canonical_entities WHERE id = ?", (source_id,))

                # Commit transaction
                cursor.execute("COMMIT")

                print(f"\n[SUCCESS] Merge completed:")
                print(f"  Moved {moved_mappings} mappings")
                print(f"  Updated {updated_interventions} intervention records")
                print(f"  Deleted source canonical '{source_name}'")

            except Exception as e:
                cursor.execute("ROLLBACK")
                print(f"[ERROR] Merge failed: {e}")

        input("\nPress Enter to continue...")

    def review_pending_mappings(self):
        """Review and approve medium-confidence mappings"""
        print("\n" + "="*60)
        print("REVIEW PENDING MAPPINGS")
        print("="*60)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get medium confidence mappings (0.5 - 0.8)
            cursor.execute("""
                SELECT em.id, em.raw_text, em.canonical_id, em.confidence_score,
                       em.mapping_method, ce.canonical_name, ce.entity_type
                FROM entity_mappings em
                JOIN canonical_entities ce ON em.canonical_id = ce.id
                WHERE em.confidence_score BETWEEN 0.5 AND 0.8
                ORDER BY em.confidence_score ASC, em.raw_text
                LIMIT 50
            """)

            mappings = cursor.fetchall()

            if not mappings:
                print("No pending mappings found.")
                input("\nPress Enter to continue...")
                return

            print(f"Found {len(mappings)} pending mappings to review:\n")

            for i, mapping in enumerate(mappings, 1):
                print(f"\n[{i}/{len(mappings)}] Review mapping:")
                print(f"  Raw text: '{mapping['raw_text']}'")
                print(f"  Canonical: '{mapping['canonical_name']}' ({mapping['entity_type']})")
                print(f"  Method: {mapping['mapping_method']}")
                print(f"  Confidence: {mapping['confidence_score']:.3f}")

                print("\nActions:")
                print("1. Approve (set confidence to 1.0)")
                print("2. Reject (delete mapping)")
                print("3. Edit confidence score")
                print("4. Skip to next")
                print("5. Exit review")

                choice = input("Choice: ").strip()

                if choice == "1":
                    # Approve
                    cursor.execute("""
                        UPDATE entity_mappings
                        SET confidence_score = 1.0, created_by = 'admin_approved'
                        WHERE id = ?
                    """, (mapping['id'],))
                    print("✓ Approved")

                elif choice == "2":
                    # Reject
                    confirm = input("Confirm deletion (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        cursor.execute("DELETE FROM entity_mappings WHERE id = ?", (mapping['id'],))
                        print("✓ Deleted")
                    else:
                        print("Deletion cancelled")

                elif choice == "3":
                    # Edit confidence
                    try:
                        new_confidence = float(input("Enter new confidence (0.0-1.0): ").strip())
                        if 0 <= new_confidence <= 1:
                            cursor.execute("""
                                UPDATE entity_mappings
                                SET confidence_score = ?, created_by = 'admin_edited'
                                WHERE id = ?
                            """, (new_confidence, mapping['id']))
                            print(f"✓ Updated confidence to {new_confidence}")
                        else:
                            print("Invalid confidence score")
                    except ValueError:
                        print("Invalid number format")

                elif choice == "4":
                    # Skip
                    continue

                elif choice == "5":
                    # Exit
                    break

                else:
                    print("Invalid choice")

                conn.commit()

        input("\nPress Enter to continue...")

    def search_entities(self):
        """Search entities and mappings"""
        print("\n" + "="*60)
        print("SEARCH ENTITIES AND MAPPINGS")
        print("="*60)

        search_term = input("Enter search term: ").strip()
        if not search_term:
            print("Search term cannot be empty.")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Search canonical entities
            print(f"\n[CANONICAL ENTITIES matching '{search_term}']")
            cursor.execute("""
                SELECT id, canonical_name, entity_type, confidence_score
                FROM canonical_entities
                WHERE canonical_name LIKE ?
                ORDER BY canonical_name
                LIMIT 20
            """, (f"%{search_term}%",))

            canonicals = cursor.fetchall()
            if canonicals:
                for canonical in canonicals:
                    print(f"  {canonical['id']}: {canonical['canonical_name']} ({canonical['entity_type']})")
            else:
                print("  No matches found")

            # Search entity mappings
            print(f"\n[ENTITY MAPPINGS matching '{search_term}']")
            cursor.execute("""
                SELECT em.id, em.raw_text, ce.canonical_name, em.entity_type, em.confidence_score
                FROM entity_mappings em
                JOIN canonical_entities ce ON em.canonical_id = ce.id
                WHERE em.raw_text LIKE ?
                ORDER BY em.raw_text
                LIMIT 20
            """, (f"%{search_term}%",))

            mappings = cursor.fetchall()
            if mappings:
                for mapping in mappings:
                    print(f"  '{mapping['raw_text']}' -> '{mapping['canonical_name']}' ({mapping['entity_type']}, conf: {mapping['confidence_score']:.2f})")
            else:
                print("  No matches found")

        input("\nPress Enter to continue...")

    def export_data(self):
        """Export normalization data"""
        print("\n" + "="*60)
        print("EXPORT DATA")
        print("="*60)

        print("1. Export canonical entities")
        print("2. Export entity mappings")
        print("3. Export statistics")
        print("4. Export all data")

        choice = input("Choice: ").strip()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if choice in ['1', '4']:
                # Export canonicals
                cursor.execute("SELECT * FROM canonical_entities ORDER BY entity_type, canonical_name")
                canonicals = [dict(row) for row in cursor.fetchall()]

                filename = f"data/exports/canonical_entities_{timestamp}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(canonicals, f, indent=2, ensure_ascii=False)
                print(f"✓ Exported canonical entities to {filename}")

            if choice in ['2', '4']:
                # Export mappings
                cursor.execute("""
                    SELECT em.*, ce.canonical_name
                    FROM entity_mappings em
                    JOIN canonical_entities ce ON em.canonical_id = ce.id
                    ORDER BY em.entity_type, em.raw_text
                """)
                mappings = [dict(row) for row in cursor.fetchall()]

                filename = f"data/exports/entity_mappings_{timestamp}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(mappings, f, indent=2, ensure_ascii=False)
                print(f"✓ Exported entity mappings to {filename}")

            if choice in ['3', '4']:
                # Export statistics
                stats = self.get_statistics()

                filename = f"data/exports/normalization_stats_{timestamp}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2)
                print(f"✓ Exported statistics to {filename}")

        input("\nPress Enter to continue...")

    def run(self):
        """Run the admin CLI"""
        print("Welcome to Normalization Admin Tool")

        while True:
            try:
                self.show_main_menu()
                choice = input("Enter choice: ").strip()

                if choice == "0":
                    print("Goodbye!")
                    break
                elif choice == "1":
                    self.display_statistics()
                elif choice == "2":
                    self.view_canonical_entities()
                elif choice == "3":
                    entity_type = input("Filter by type (intervention/condition, or Enter for all): ").strip().lower()
                    entity_type = entity_type if entity_type in ['intervention', 'condition'] else None
                    print("\nNot implemented - showing canonicals instead")
                    self.view_canonical_entities(entity_type)
                elif choice == "4":
                    self.add_new_mapping()
                elif choice == "5":
                    self.merge_canonical_entities()
                elif choice == "6":
                    self.review_pending_mappings()
                elif choice == "7":
                    self.search_entities()
                elif choice == "8":
                    self.export_data()
                else:
                    print("Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                input("Press Enter to continue...")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Normalization Admin CLI Tool")
    parser.add_argument("--db", default="data/processed/intervention_research.db",
                       help="Database path")

    args = parser.parse_args()

    admin = NormalizationAdmin(args.db)
    admin.run()


if __name__ == "__main__":
    main()