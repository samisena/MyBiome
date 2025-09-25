#!/usr/bin/env python3
"""
Optimized semantic migration with batch processing and improved error handling.
Addresses JSON parsing errors and reduces exponential complexity.
"""

import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import SemanticMerger, InterventionExtraction, MergeDecision

def get_interventions_by_condition_limited(db_path: str, max_per_condition: int = 15) -> dict:
    """Get interventions grouped by condition, limiting large conditions to avoid exponential complexity.
    IBS subtypes (IBS-D, IBS-C, IBS-M) are treated as separate conditions."""
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

    # Group by condition (normalized) with size limits
    by_condition = defaultdict(list)
    for intervention in all_interventions:
        condition = intervention['health_condition'].lower().strip()
        if len(by_condition[condition]) < max_per_condition:
            by_condition[condition].append(intervention)

    return dict(by_condition)

def build_semantic_groups_batch(interventions: list, condition: str, merger: SemanticMerger, batch_size: int = 10) -> list:
    """Build semantic groups using smaller batches to avoid exponential complexity."""
    if len(interventions) <= 1:
        if interventions:
            return [{
                'canonical_name': interventions[0]['intervention_name'],
                'alternative_names': [interventions[0]['intervention_name']],
                'interventions': interventions,
                'size': 1
            }]
        return []

    print(f"  Processing {len(interventions)} interventions in batches of {batch_size}...")

    # Process in smaller batches to reduce pairwise comparisons
    all_groups = []
    processed_interventions = set()

    # Sort by confidence to process highest quality first
    sorted_interventions = sorted(interventions, key=lambda x: x.get('confidence_score', 0), reverse=True)

    for i in range(0, len(sorted_interventions), batch_size):
        batch = sorted_interventions[i:i + batch_size]
        batch_unprocessed = [interv for interv in batch if interv['id'] not in processed_interventions]

        if not batch_unprocessed:
            continue

        print(f"    Processing batch {i//batch_size + 1}: {len(batch_unprocessed)} interventions")

        try:
            batch_groups = build_semantic_groups_for_batch(batch_unprocessed, condition, merger)
            all_groups.extend(batch_groups)

            # Mark all interventions in this batch as processed
            for group in batch_groups:
                for intervention in group['interventions']:
                    processed_interventions.add(intervention['id'])

        except Exception as e:
            print(f"    Error processing batch: {e}")
            # Add remaining interventions as individual groups
            for interv in batch_unprocessed:
                if interv['id'] not in processed_interventions:
                    all_groups.append({
                        'canonical_name': interv['intervention_name'],
                        'alternative_names': [interv['intervention_name']],
                        'interventions': [interv],
                        'size': 1
                    })
                    processed_interventions.add(interv['id'])

    print(f"  Created {len(all_groups)} semantic groups for {condition}")
    return all_groups

def build_semantic_groups_for_batch(interventions: list, condition: str, merger: SemanticMerger) -> list:
    """Build semantic groups for a small batch of interventions."""
    if len(interventions) <= 1:
        if interventions:
            return [{
                'canonical_name': interventions[0]['intervention_name'],
                'alternative_names': [interventions[0]['intervention_name']],
                'interventions': interventions,
                'size': 1
            }]
        return []

    # Convert to InterventionExtraction objects
    extractions = []
    for interv in interventions:
        extraction = InterventionExtraction(
            model_name='migration_tool',
            intervention_name=interv['intervention_name'],
            health_condition=interv['health_condition'],
            intervention_category=interv.get('intervention_category', 'Unknown'),
            correlation_type=interv.get('correlation_type', 'Unknown'),
            confidence_score=interv.get('confidence_score', 0.5),
            correlation_strength=interv.get('correlation_strength', 0.5),
            supporting_quote=interv.get('supporting_quote', ''),
            raw_data={'id': interv['id'], 'paper_id': interv.get('paper_id', '')}
        )
        extractions.append(extraction)

    # Find semantic duplicates using pairwise comparison (limited scope)
    groups = []
    already_grouped = set()

    for i, extract1 in enumerate(extractions):
        if i in already_grouped:
            continue

        # Start a new group with this intervention
        current_group = [i]
        group_interventions = [extract1]

        # Compare with remaining interventions in this batch only
        for j, extract2 in enumerate(extractions[i+1:], i+1):
            if j in already_grouped:
                continue

            try:
                decision = merger.compare_interventions(extract1, extract2)

                if decision.is_duplicate:
                    print(f"      MATCH: {extract1.intervention_name} <-> {extract2.intervention_name}")
                    current_group.append(j)
                    group_interventions.append(extract2)
                    already_grouped.add(j)

                # Small delay to prevent overwhelming LLM
                time.sleep(0.1)  # Reduced delay since we're using smaller batches

            except Exception as e:
                print(f"      Error comparing interventions: {e}")
                continue

        # Mark this intervention as grouped
        already_grouped.add(i)

        # Create group result
        if len(group_interventions) > 1:
            canonical_name = group_interventions[0].intervention_name
            alternative_names = [gi.intervention_name for gi in group_interventions]
            print(f"      GROUP: {canonical_name} ({len(group_interventions)} members)")
        else:
            canonical_name = group_interventions[0].intervention_name
            alternative_names = [canonical_name]

        group_info = {
            'canonical_name': canonical_name,
            'alternative_names': alternative_names,
            'interventions': [interventions[idx] for idx in current_group],
            'size': len(group_interventions)
        }
        groups.append(group_info)

    return groups

def update_database_with_groups(db_path: str, condition: str, groups: list) -> int:
    """Update database with semantic group results."""
    updated_count = 0
    condition_safe = condition.replace(' ', '_').replace('(', '').replace(')', '')

    for group_idx, group in enumerate(groups, 1):
        base_semantic_id = f"{condition_safe}_group_{group_idx}"

        for intervention in group['interventions']:
            semantic_updates = {
                'canonical_name': group['canonical_name'],
                'alternative_names': json.dumps(group['alternative_names']),
                'semantic_group_id': base_semantic_id,
                'semantic_confidence': 0.9 if group['size'] > 1 else 1.0,
                'merge_source': 'qwen2.5:14b',
                'validator_agreement': True,
                'merge_decision_log': json.dumps({
                    'condition': condition,
                    'group_size': group['size'],
                    'processed_at': datetime.now().isoformat(),
                    'method': 'optimized_batch_processing'
                })
            }

            try:
                conn = sqlite3.connect(db_path, timeout=60.0)
                cursor = conn.cursor()

                set_clauses = []
                values = []

                for field, value in semantic_updates.items():
                    set_clauses.append(f"{field} = ?")
                    values.append(value)

                values.append(intervention['id'])

                query = f"""
                    UPDATE interventions
                    SET {', '.join(set_clauses)}
                    WHERE id = ?
                """

                cursor.execute(query, values)
                conn.commit()
                conn.close()
                updated_count += 1

            except Exception as e:
                print(f"    ERROR updating {intervention['intervention_name']}: {e}")

    return updated_count

def main():
    """Run optimized semantic migration."""
    db_path = "data/processed/intervention_research.db"

    print(f"Starting optimized semantic migration at {datetime.now()}")

    # Initialize semantic merger
    try:
        merger = SemanticMerger(
            primary_model="qwen2.5:14b",
            validator_model="gemma2:9b"
        )
        print("SUCCESS: SemanticMerger initialized with improved error handling")
    except Exception as e:
        print(f"ERROR: Failed to initialize SemanticMerger: {e}")
        return False

    # Get interventions by condition with limits
    try:
        interventions_by_condition = get_interventions_by_condition_limited(db_path, max_per_condition=15)
        total_conditions = len(interventions_by_condition)
        total_interventions = sum(len(interventions) for interventions in interventions_by_condition.values())

        print(f"FOUND: {total_interventions} interventions across {total_conditions} conditions")
        print("LIMITED: Large conditions capped at 15 interventions to prevent exponential complexity")

        # Show top conditions by intervention count
        condition_counts = [(condition, len(interventions))
                          for condition, interventions in interventions_by_condition.items()]
        condition_counts.sort(key=lambda x: x[1], reverse=True)

        print("Top conditions by intervention count:")
        for condition, count in condition_counts[:10]:
            print(f"  {condition}: {count} interventions")

    except Exception as e:
        print(f"ERROR: Failed to load interventions: {e}")
        return False

    # Process each condition
    processed_conditions = 0
    total_updated = 0

    for condition, interventions in interventions_by_condition.items():
        if len(interventions) == 1:
            print(f"SKIP: {condition} (only 1 intervention)")
            # Still process single interventions to mark them as completed
            try:
                groups = [{
                    'canonical_name': interventions[0]['intervention_name'],
                    'alternative_names': [interventions[0]['intervention_name']],
                    'interventions': interventions,
                    'size': 1
                }]
                updated_count = update_database_with_groups(db_path, condition, groups)
                total_updated += updated_count
            except Exception as e:
                print(f"ERROR processing single intervention for {condition}: {e}")

            processed_conditions += 1
            continue

        print(f"\nPROCESSING: {condition} ({len(interventions)} interventions)")

        try:
            groups = build_semantic_groups_batch(interventions, condition, merger, batch_size=8)
            updated_count = update_database_with_groups(db_path, condition, groups)
            total_updated += updated_count

            print(f"COMPLETED: {condition} - {updated_count} interventions updated")

            # Print statistics
            stats = merger.get_statistics()
            print(f"  Stats: {stats['total_comparisons']} comparisons, {stats['duplicates_found']} duplicates, {stats['json_parse_errors']} JSON errors, {stats['llm_errors']} LLM errors")

        except Exception as e:
            print(f"ERROR processing {condition}: {e}")

        processed_conditions += 1

        # Progress update
        print(f"PROGRESS: {processed_conditions}/{total_conditions} conditions, {total_updated} interventions updated")

        # Short delay between conditions
        time.sleep(1)

    print(f"\nOPTIMIZED MIGRATION COMPLETE!")
    print(f"Final stats: {total_updated} interventions updated across {processed_conditions} conditions")

    # Print final statistics
    final_stats = merger.get_statistics()
    print(f"Migration Statistics:")
    print(f"  Total comparisons: {final_stats['total_comparisons']}")
    print(f"  Duplicates found: {final_stats['duplicates_found']}")
    print(f"  JSON parse errors: {final_stats['json_parse_errors']}")
    print(f"  LLM errors: {final_stats['llm_errors']}")

    return True

if __name__ == "__main__":
    main()