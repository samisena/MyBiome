#!/usr/bin/env python3
"""
Simple deduplication script that works directly on the database without locks.
Uses bulk operations and focuses on the most obvious duplicates.
"""

import sqlite3
import json
import re
from pathlib import Path
from typing import Dict, List, Set

def normalize_intervention_name(name: str) -> str:
    """Normalize intervention name for comparison."""
    name = name.lower().strip()

    # Remove parenthetical information
    name = re.sub(r'\([^)]*\)', '', name).strip()

    # Common replacements
    replacements = {
        'proton pump inhibitors': 'ppi',
        'proton pump inhibitor': 'ppi',
        'ppis': 'ppi',
        'low fodmap diet': 'low-fodmap-diet',
        'fodmap diet': 'low-fodmap-diet',
        'fodmap elimination diet': 'low-fodmap-diet',
        'anti-reflux mucosal ablation': 'arma',
        'fecal microbiota transplantation': 'fmt',
        'faecal microbiota transplantation': 'fmt',
        'cognitive behavioral therapy': 'cbt',
        'cognitive behaviour therapy': 'cbt',
        'high-intensity interval training': 'hiit'
    }

    for original, normalized in replacements.items():
        if original in name:
            name = normalized
            break

    return name

def get_duplicate_groups(interventions: List[Dict]) -> Dict[str, List[Dict]]:
    """Group interventions by normalized names."""
    groups = {}

    for intervention in interventions:
        normalized = normalize_intervention_name(intervention['intervention_name'])

        if normalized not in groups:
            groups[normalized] = []
        groups[normalized].append(intervention)

    # Only return groups with duplicates
    return {k: v for k, v in groups.items() if len(v) > 1}

def main():
    """Main deduplication function."""
    db_path = Path("back_end/data/processed/intervention_research.db")

    if not db_path.exists():
        print("[ERROR] Database not found")
        return

    print("=== Simple Deduplication ===")

    # Connect with minimal timeout
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()

        # Get all interventions
        print("Loading interventions...")
        cursor.execute("""
            SELECT id, intervention_name, health_condition, confidence_score, paper_id
            FROM interventions
            ORDER BY confidence_score DESC
        """)

        interventions = [dict(row) for row in cursor.fetchall()]
        print(f"Found {len(interventions)} interventions")

        # Group duplicates
        duplicate_groups = get_duplicate_groups(interventions)
        print(f"Found {len(duplicate_groups)} groups with duplicates")

        total_removed = 0

        for normalized_name, group in duplicate_groups.items():
            if len(group) <= 1:
                continue

            print(f"\nProcessing group '{normalized_name}' ({len(group)} items):")

            # Sort by confidence score (desc) and keep the best one - handle None values
            group.sort(key=lambda x: x['confidence_score'] if x['confidence_score'] is not None else 0, reverse=True)
            best = group[0]
            duplicates = group[1:]

            try:
                print(f"  Keeping: {best['intervention_name']} (confidence: {best['confidence_score']})")

                # Remove duplicates
                for dup in duplicates:
                    print(f"  Removing: {dup['intervention_name']} (confidence: {dup['confidence_score']})")
                    cursor.execute("DELETE FROM interventions WHERE id = ?", (dup['id'],))
                    total_removed += 1
            except UnicodeEncodeError:
                # Handle Unicode encoding issues in console output
                print(f"  Keeping intervention with ID {best['id']} (confidence: {best['confidence_score']})")
                for dup in duplicates:
                    print(f"  Removing intervention with ID {dup['id']} (confidence: {dup['confidence_score']})")
                    cursor.execute("DELETE FROM interventions WHERE id = ?", (dup['id'],))
                    total_removed += 1

        # Commit changes
        conn.commit()
        print(f"\n[SUCCESS] Removed {total_removed} duplicate interventions")

        # Final count
        cursor.execute("SELECT COUNT(*) FROM interventions")
        final_count = cursor.fetchone()[0]
        print(f"Final intervention count: {final_count}")

    except Exception as e:
        print(f"[ERROR] {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    main()