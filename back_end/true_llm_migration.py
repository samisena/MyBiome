#!/usr/bin/env python3
"""
True LLM-based migration using the actual SemanticMerger with condition-specific processing.
This uses the real LLM models to identify semantic duplicates, not rule-based logic.
"""

import sys
import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import SemanticMerger, InterventionExtraction

def setup_database_for_migration(db_path: str) -> None:
    """Configure database for migration with WAL mode."""
    conn = sqlite3.connect(db_path, timeout=60.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.commit()
    conn.close()
    print("Database configured for LLM migration (WAL mode)")

def get_interventions_by_condition(db_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get all interventions grouped by health condition for condition-specific processing."""
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, intervention_name, health_condition, intervention_category,
               correlation_type, confidence_score, correlation_strength,
               supporting_quote, paper_id
        FROM interventions
        WHERE (canonical_name IS NULL OR canonical_name = '')
        ORDER BY health_condition, confidence_score DESC NULLS LAST
    """)

    all_interventions = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Group by health condition (normalized)
    by_condition = {}
    for intervention in all_interventions:
        condition = (intervention['health_condition'] or 'unknown').lower().strip()
        if condition not in by_condition:
            by_condition[condition] = []
        by_condition[condition].append(intervention)

    return by_condition

def process_condition_with_llm(condition: str, interventions: List[Dict[str, Any]],
                              merger: SemanticMerger) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process interventions for a single condition using the LLM semantic merger.
    Returns (processed_interventions, duplicates_found)
    """
    if len(interventions) <= 1:
        # Single intervention, just add basic semantic fields
        if interventions:
            intervention = interventions[0]
            return [{
                'intervention_id': intervention['id'],
                'canonical_name': intervention['intervention_name'],
                'alternative_names': json.dumps([intervention['intervention_name']]),
                'search_terms': json.dumps([intervention['intervention_name'].lower()]),
                'semantic_group_id': f"sem_{hash(intervention['intervention_name'].lower())}"[:12],
                'semantic_confidence': 1.0,
                'merge_source': 'single',
                'consensus_confidence': intervention['confidence_score'] or 0.0,
                'model_agreement': 'single',
                'models_used': 'migration',
                'raw_extraction_count': 1
            }], 0
        else:
            return [], 0

    print(f"  Processing {len(interventions)} interventions for '{condition}' with LLM...")

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
            raw_data={'intervention_id': intervention['id'], 'paper_id': intervention['paper_id']}
        )
        extractions.append(extraction)

    # Use LLM semantic merger to find duplicates
    processed_interventions = []
    duplicates_found = 0
    used_indices = set()

    for i, extraction1 in enumerate(extractions):
        if i in used_indices:
            continue

        # Find all semantically similar interventions using LLM
        similar_extractions = [extraction1]
        similar_indices = [i]

        for j, extraction2 in enumerate(extractions[i+1:], i+1):
            if j in used_indices:
                continue

            try:
                # Use actual LLM comparison
                merge_decision = merger.compare_interventions(extraction1, extraction2)

                if merge_decision.is_duplicate:
                    # Validate the merge decision with second LLM
                    validation = merger.validate_merge_decision(merge_decision, extraction1, extraction2)

                    if validation.agrees_with_merge and validation.confidence > 0.7:
                        similar_extractions.append(extraction2)
                        similar_indices.append(j)
                        duplicates_found += 1

            except Exception as e:
                print(f"    LLM comparison error: {e}")
                continue

        # Mark all similar extractions as used
        for idx in similar_indices:
            used_indices.add(idx)

        if len(similar_extractions) > 1:
            # Create merged intervention using LLM decision
            try:
                # Get the merge decision for the group
                merge_decision = merger.compare_interventions(similar_extractions[0], similar_extractions[1])
                merged = merger.create_merged_intervention(similar_extractions, merge_decision)

                # Keep the best extraction as primary
                best_extraction = max(similar_extractions, key=lambda e: e.confidence_score or 0.0)

                processed_interventions.append({
                    'intervention_id': best_extraction.raw_data['intervention_id'],
                    'canonical_name': merged['canonical_name'],
                    'alternative_names': json.dumps(merged['alternative_names']),
                    'search_terms': json.dumps(merged['search_terms']),
                    'semantic_group_id': merged['semantic_group_id'],
                    'semantic_confidence': merged['semantic_confidence'],
                    'merge_source': merged['merge_source'],
                    'consensus_confidence': merged['consensus_confidence'],
                    'model_agreement': merged['model_agreement'],
                    'models_used': merged['models_used'],
                    'raw_extraction_count': len(similar_extractions)
                })

                # Mark duplicates for removal
                for extraction in similar_extractions[1:]:
                    processed_interventions.append({
                        'intervention_id': extraction.raw_data['intervention_id'],
                        'to_remove': True
                    })

            except Exception as e:
                print(f"    Error creating merged intervention: {e}")
                # Fallback: treat as separate interventions
                for extraction in similar_extractions:
                    processed_interventions.append({
                        'intervention_id': extraction.raw_data['intervention_id'],
                        'canonical_name': extraction.intervention_name,
                        'alternative_names': json.dumps([extraction.intervention_name]),
                        'search_terms': json.dumps([extraction.intervention_name.lower()]),
                        'semantic_group_id': f"sem_{hash(extraction.intervention_name.lower())}"[:12],
                        'semantic_confidence': 0.8,
                        'merge_source': 'fallback',
                        'consensus_confidence': extraction.confidence_score or 0.0,
                        'model_agreement': 'single',
                        'models_used': 'migration',
                        'raw_extraction_count': 1
                    })
        else:
            # Single intervention
            extraction = similar_extractions[0]
            processed_interventions.append({
                'intervention_id': extraction.raw_data['intervention_id'],
                'canonical_name': extraction.intervention_name,
                'alternative_names': json.dumps([extraction.intervention_name]),
                'search_terms': json.dumps([extraction.intervention_name.lower()]),
                'semantic_group_id': f"sem_{hash(extraction.intervention_name.lower())}"[:12],
                'semantic_confidence': 1.0,
                'merge_source': 'single',
                'consensus_confidence': extraction.confidence_score or 0.0,
                'model_agreement': 'single',
                'models_used': 'migration',
                'raw_extraction_count': 1
            })

    return processed_interventions, duplicates_found

def update_database_with_llm_results(db_path: str, processed_interventions: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Update database with LLM processing results."""
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
                # Update with LLM-generated semantic fields
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
        return updates_count, removals_count

    except Exception as e:
        conn.execute("ROLLBACK")
        raise e
    finally:
        conn.close()

def main():
    """Main LLM migration function."""
    db_path = "data/processed/intervention_research.db"

    if not Path(db_path).exists():
        print(f"[ERROR] Database not found: {db_path}")
        return

    print("=== TRUE LLM SEMANTIC MIGRATION ===")
    print(f"Starting LLM migration at {datetime.now()}")
    print("Using models: qwen2.5:14b (primary), gemma2:9b (validator)")
    print()

    # Step 1: Configure database
    setup_database_for_migration(db_path)

    # Step 2: Get interventions grouped by condition
    print("Loading interventions by health condition...")
    interventions_by_condition = get_interventions_by_condition(db_path)

    total_conditions = len(interventions_by_condition)
    total_interventions = sum(len(interventions) for interventions in interventions_by_condition.values())

    print(f"Found {total_interventions} interventions across {total_conditions} conditions")
    print()

    # Step 3: Initialize semantic merger with actual LLM models
    print("Initializing LLM semantic merger...")
    merger = SemanticMerger(
        primary_model='qwen2.5:14b',
        validator_model='gemma2:9b'
    )

    # Step 4: Process each condition with LLM
    all_processed = []
    total_duplicates = 0

    # Start with GERD/reflux conditions for testing
    priority_conditions = [k for k in interventions_by_condition.keys() if 'gerd' in k or 'reflux' in k]

    print("Processing priority conditions (GERD/reflux) first...")
    for condition in priority_conditions:
        interventions = interventions_by_condition[condition]
        processed, duplicates = process_condition_with_llm(condition, interventions, merger)
        all_processed.extend(processed)
        total_duplicates += duplicates
        print(f"    {condition}: {len(interventions)} -> {len([p for p in processed if not p.get('to_remove', False)])} interventions ({duplicates} duplicates)")

    # Process remaining conditions in smaller batches
    remaining_conditions = [k for k in interventions_by_condition.keys() if k not in priority_conditions]

    print(f"\nProcessing {len(remaining_conditions)} remaining conditions...")
    for i, condition in enumerate(remaining_conditions, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(remaining_conditions)} conditions")

        interventions = interventions_by_condition[condition]
        processed, duplicates = process_condition_with_llm(condition, interventions, merger)
        all_processed.extend(processed)
        total_duplicates += duplicates

    # Step 5: Update database
    print(f"\nUpdating database with LLM results...")
    print(f"Total duplicates found: {total_duplicates}")

    updates, removals = update_database_with_llm_results(db_path, all_processed)

    print(f"Database updated: {updates} interventions enhanced, {removals} duplicates removed")

    # Step 6: Print LLM statistics
    print("\n=== LLM Processing Statistics ===")
    stats = merger.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nLLM migration completed successfully at {datetime.now()}")

if __name__ == "__main__":
    main()