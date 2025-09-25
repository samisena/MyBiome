#!/usr/bin/env python3
"""
GERD-focused LLM semantic migration to test the system with a smaller dataset.
This will process only GERD-related interventions to verify the LLM semantic merger works.
"""

import sys
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import SemanticMerger, InterventionExtraction

def get_gerd_interventions(db_path: str) -> List[Dict[str, Any]]:
    """Get all GERD-related interventions for testing."""
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, intervention_name, health_condition, intervention_category,
               correlation_type, confidence_score, correlation_strength,
               supporting_quote, paper_id
        FROM interventions
        WHERE (canonical_name IS NULL OR canonical_name = '')
        AND (health_condition LIKE '%GERD%' OR health_condition LIKE '%reflux%'
             OR health_condition LIKE '%esophag%')
        ORDER BY confidence_score DESC NULLS LAST
    """)

    interventions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return interventions

def process_gerd_interventions(interventions: List[Dict[str, Any]],
                              merger: SemanticMerger) -> List[Dict[str, Any]]:
    """Process GERD interventions using LLM semantic merger."""
    if len(interventions) <= 1:
        if interventions:
            intervention = interventions[0]
            return [{
                'intervention_id': intervention['id'],
                'canonical_name': intervention['intervention_name'],
                'alternative_names': json.dumps([intervention['intervention_name']]),
                'search_terms': json.dumps([intervention['intervention_name'].lower()]),
                'semantic_group_id': f"sem_{hash(intervention['intervention_name'].lower())}",
                'semantic_confidence': 1.0,
                'merge_source': 'single',
                'consensus_confidence': intervention['confidence_score'] or 0.0,
                'model_agreement': 'single',
                'models_used': 'migration',
                'raw_extraction_count': 1
            }]
        else:
            return []

    print(f"Processing {len(interventions)} GERD interventions with LLM...")

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

        print(f"  Analyzing: {extraction1.intervention_name}")

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
                    print(f"    -> Found duplicate: {extraction2.intervention_name} (confidence: {merge_decision.semantic_confidence})")

                    # Validate the merge decision with second LLM
                    validation = merger.validate_merge_decision(merge_decision, extraction1, extraction2)

                    if validation.agrees_with_merge and validation.confidence > 0.7:
                        similar_extractions.append(extraction2)
                        similar_indices.append(j)
                        duplicates_found += 1
                        print(f"      -> Validation passed (confidence: {validation.confidence})")
                    else:
                        print(f"      -> Validation failed (confidence: {validation.confidence})")

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

                print(f"    -> Created merged intervention: {merged['canonical_name']}")
                print(f"       Alternatives: {merged['alternative_names']}")
                print(f"       Group ID: {merged['semantic_group_id']}")

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
                        'semantic_group_id': f"sem_{hash(extraction.intervention_name.lower())}",
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
                'semantic_group_id': f"sem_{hash(extraction.intervention_name.lower())}",
                'semantic_confidence': 1.0,
                'merge_source': 'single',
                'consensus_confidence': extraction.confidence_score or 0.0,
                'model_agreement': 'single',
                'models_used': 'migration',
                'raw_extraction_count': 1
            })
            print(f"    -> Single intervention: {extraction.intervention_name}")

    print(f"GERD processing complete: {duplicates_found} duplicates found")
    return processed_interventions

def update_database_with_results(db_path: str, processed_interventions: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Update database with LLM processing results."""
    conn = sqlite3.connect(db_path, timeout=60.0)

    try:
        conn.execute("BEGIN TRANSACTION")

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
    """Main GERD-focused migration function."""
    db_path = "data/processed/intervention_research.db"

    if not Path(db_path).exists():
        print(f"[ERROR] Database not found: {db_path}")
        return

    print("=== GERD-FOCUSED LLM SEMANTIC MIGRATION ===")
    print(f"Starting GERD migration at {datetime.now()}")
    print("Using models: qwen2.5:14b (primary), gemma2:9b (validator)")
    print()

    # Step 1: Get GERD interventions
    print("Loading GERD-related interventions...")
    gerd_interventions = get_gerd_interventions(db_path)
    print(f"Found {len(gerd_interventions)} GERD interventions")

    if not gerd_interventions:
        print("No GERD interventions found for processing.")
        return

    # Print the interventions we found
    print("\nGERD interventions to process:")
    for i, intervention in enumerate(gerd_interventions[:10], 1):  # Show first 10
        print(f"  {i}. {intervention['intervention_name']} (condition: {intervention['health_condition']})")
    if len(gerd_interventions) > 10:
        print(f"  ... and {len(gerd_interventions) - 10} more")
    print()

    # Step 2: Initialize semantic merger
    print("Initializing LLM semantic merger...")
    merger = SemanticMerger(
        primary_model='qwen2.5:14b',
        validator_model='gemma2:9b'
    )

    # Step 3: Process GERD interventions with LLM
    processed = process_gerd_interventions(gerd_interventions, merger)

    # Step 4: Update database
    print(f"\nUpdating database with LLM results...")
    updates, removals = update_database_with_results(db_path, processed)

    print(f"Database updated: {updates} interventions enhanced, {removals} duplicates removed")

    # Step 5: Print LLM statistics
    print("\n=== LLM Processing Statistics ===")
    stats = merger.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nGERD migration completed successfully at {datetime.now()}")

if __name__ == "__main__":
    main()