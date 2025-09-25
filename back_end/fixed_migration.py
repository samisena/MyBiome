#!/usr/bin/env python3
"""
Fixed migration script with proper SQLite connection handling.
Avoids connection pooling and uses WAL mode for better concurrency.
"""

import sys
import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import SemanticMerger, InterventionExtraction

def setup_database_for_migration(db_path: str) -> None:
    """Configure database for migration with WAL mode and proper settings."""
    conn = sqlite3.connect(db_path, timeout=60.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=268435456")  # 256MB
    conn.commit()
    conn.close()
    print("Database configured for migration (WAL mode, optimized settings)")

def get_interventions_needing_migration(db_path: str) -> List[Dict[str, Any]]:
    """Get all interventions that need semantic processing."""
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, intervention_name, health_condition, intervention_category,
               correlation_type, confidence_score, correlation_strength,
               supporting_quote, paper_id
        FROM interventions
        WHERE (canonical_name IS NULL OR canonical_name = '')
        ORDER BY confidence_score DESC NULLS LAST
    """)

    interventions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return interventions

def process_interventions_semantic(interventions: List[Dict[str, Any]],
                                 merger: SemanticMerger) -> List[Dict[str, Any]]:
    """Process interventions through semantic merger without database calls."""
    print(f"Processing {len(interventions)} interventions with semantic merger...")

    # Convert to InterventionExtraction objects
    extractions = []
    for intervention in interventions:
        extraction = InterventionExtraction(
            model_name='migration',
            intervention_name=intervention['intervention_name'] or '',
            health_condition=intervention['health_condition'] or '',
            intervention_category=intervention['intervention_category'] or 'unknown',
            correlation_type=intervention['correlation_type'] or 'unknown',
            confidence_score=intervention['confidence_score'] or 0.0,
            correlation_strength=intervention['correlation_strength'] or 0.0,
            supporting_quote=intervention['supporting_quote'] or '',
            raw_data={'intervention_id': intervention['id']}
        )
        extractions.append(extraction)

    # Group by health condition for processing
    condition_groups = {}
    for extraction in extractions:
        condition = extraction.health_condition.lower().strip()
        if condition not in condition_groups:
            condition_groups[condition] = []
        condition_groups[condition].append(extraction)

    # Process semantic merging within each condition
    processed_interventions = []
    total_duplicates_found = 0

    for condition, group_extractions in condition_groups.items():
        print(f"  Processing {condition}: {len(group_extractions)} interventions")

        if len(group_extractions) == 1:
            # Single intervention, just add semantic fields
            extraction = group_extractions[0]
            canonical_name = extraction.intervention_name
            alternative_names = [canonical_name]
            search_terms = [canonical_name.lower()]
            semantic_group_id = f"sem_{hash(canonical_name.lower())}"[:12]

            processed_interventions.append({
                'intervention_id': extraction.raw_data['intervention_id'],
                'canonical_name': canonical_name,
                'alternative_names': json.dumps(alternative_names),
                'search_terms': json.dumps(search_terms),
                'semantic_group_id': semantic_group_id,
                'semantic_confidence': 1.0,
                'merge_source': 'rule-based',
                'consensus_confidence': extraction.confidence_score,
                'model_agreement': 'single',
                'models_used': 'migration',
                'raw_extraction_count': 1
            })
        else:
            # Multiple interventions, use semantic merger
            seen_groups = {}

            for extraction in group_extractions:
                # Use simple rule-based grouping for migration
                normalized_name = normalize_intervention_name(extraction.intervention_name)

                if normalized_name not in seen_groups:
                    seen_groups[normalized_name] = {
                        'canonical_name': extraction.intervention_name,
                        'extractions': [extraction],
                        'best_confidence': extraction.confidence_score or 0.0
                    }
                else:
                    seen_groups[normalized_name]['extractions'].append(extraction)
                    if (extraction.confidence_score or 0.0) > seen_groups[normalized_name]['best_confidence']:
                        seen_groups[normalized_name]['canonical_name'] = extraction.intervention_name
                        seen_groups[normalized_name]['best_confidence'] = extraction.confidence_score or 0.0

            # Process each semantic group
            for normalized_name, group_info in seen_groups.items():
                group_extractions = group_info['extractions']
                canonical_name = group_info['canonical_name']

                # Collect alternative names
                alternative_names = list(set(e.intervention_name for e in group_extractions))
                search_terms = [normalized_name] + [name.lower() for name in alternative_names]
                search_terms = list(set(search_terms))

                semantic_group_id = f"sem_{hash(normalized_name)}"[:12]

                # Keep the best extraction as primary
                best_extraction = max(group_extractions, key=lambda e: e.confidence_score or 0.0)

                processed_interventions.append({
                    'intervention_id': best_extraction.raw_data['intervention_id'],
                    'canonical_name': canonical_name,
                    'alternative_names': json.dumps(alternative_names),
                    'search_terms': json.dumps(search_terms),
                    'semantic_group_id': semantic_group_id,
                    'semantic_confidence': 0.95,  # High confidence for rule-based
                    'merge_source': 'rule-based',
                    'consensus_confidence': best_extraction.confidence_score or 0.0,
                    'model_agreement': 'partial' if len(group_extractions) > 1 else 'single',
                    'models_used': 'migration',
                    'raw_extraction_count': len(group_extractions)
                })

                # Mark duplicates for removal
                for extraction in group_extractions[1:]:  # Skip the best one
                    processed_interventions.append({
                        'intervention_id': extraction.raw_data['intervention_id'],
                        'to_remove': True
                    })
                    total_duplicates_found += 1

    print(f"Semantic processing complete: {len(processed_interventions)} records, {total_duplicates_found} duplicates found")
    return processed_interventions

def normalize_intervention_name(name: str) -> str:
    """Normalize intervention name for comparison."""
    if not name:
        return 'unknown'

    name = name.lower().strip()

    # Remove parenthetical information
    import re
    name = re.sub(r'\([^)]*\)', '', name).strip()

    # Common replacements for migration
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
            return normalized

    return name

def update_database_with_semantics(db_path: str, processed_interventions: List[Dict[str, Any]]) -> None:
    """Update database with semantic fields and remove duplicates."""
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("BEGIN TRANSACTION")

    try:
        updates_count = 0
        removals_count = 0

        for item in processed_interventions:
            if item.get('to_remove'):
                # Remove duplicate
                conn.execute("DELETE FROM interventions WHERE id = ?", (item['intervention_id'],))
                removals_count += 1
            else:
                # Update with semantic fields
                conn.execute("""
                    UPDATE interventions
                    SET canonical_name = ?,
                        alternative_names = ?,
                        search_terms = ?,
                        semantic_group_id = ?,
                        semantic_confidence = ?,
                        merge_source = ?,
                        consensus_confidence = ?,
                        model_agreement = ?,
                        models_used = ?,
                        raw_extraction_count = ?
                    WHERE id = ?
                """, (
                    item['canonical_name'],
                    item['alternative_names'],
                    item['search_terms'],
                    item['semantic_group_id'],
                    item['semantic_confidence'],
                    item['merge_source'],
                    item['consensus_confidence'],
                    item['model_agreement'],
                    item['models_used'],
                    item['raw_extraction_count'],
                    item['intervention_id']
                ))
                updates_count += 1

        conn.execute("COMMIT")
        print(f"Database updated: {updates_count} interventions enhanced, {removals_count} duplicates removed")

    except Exception as e:
        conn.execute("ROLLBACK")
        print(f"Database update failed: {e}")
        raise
    finally:
        conn.close()

def main():
    """Main migration function."""
    db_path = "data/processed/intervention_research.db"

    if not Path(db_path).exists():
        print(f"[ERROR] Database not found: {db_path}")
        return

    print("=== FIXED SEMANTIC MIGRATION ===")
    print(f"Starting migration at {datetime.now()}")

    # Step 1: Configure database for migration
    setup_database_for_migration(db_path)

    # Step 2: Get interventions needing migration
    print("Loading interventions needing semantic processing...")
    interventions = get_interventions_needing_migration(db_path)
    print(f"Found {len(interventions)} interventions to process")

    if not interventions:
        print("No interventions need migration. Exiting.")
        return

    # Step 3: Initialize semantic merger (without LLM calls for migration)
    print("Initializing semantic merger...")
    merger = SemanticMerger()  # We'll use rule-based processing for migration

    # Step 4: Process interventions semantically
    processed_interventions = process_interventions_semantic(interventions, merger)

    # Step 5: Update database with semantic fields
    print("Updating database with semantic fields...")
    update_database_with_semantics(db_path, processed_interventions)

    print(f"Migration completed successfully at {datetime.now()}")

if __name__ == "__main__":
    main()