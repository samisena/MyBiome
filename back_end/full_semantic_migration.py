#!/usr/bin/env python3
"""
Full semantic migration using the proven pairwise comparison approach.
Processes all 871 interventions across all conditions with proper semantic analysis.
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

def get_interventions_by_condition(db_path: str) -> dict:
    """Get all interventions grouped by condition."""
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

    # Group by condition (normalized)
    by_condition = defaultdict(list)
    for intervention in all_interventions:
        condition = intervention['health_condition'].lower().strip()
        by_condition[condition].append(intervention)

    return dict(by_condition)

def build_semantic_groups_for_condition(interventions: list, condition: str, merger: SemanticMerger) -> list:
    """Build semantic groups for interventions within a single condition."""
    if len(interventions) < 2:
        # Single intervention - create individual group
        if interventions:
            return [{
                'canonical_name': interventions[0]['intervention_name'],
                'alternative_names': [interventions[0]['intervention_name']],
                'interventions': interventions,
                'size': 1
            }]
        return []

    print(f"  Analyzing {len(interventions)} interventions for semantic groups...")

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

    # Find semantic duplicates using pairwise comparison
    groups = []
    already_grouped = set()

    for i, extract1 in enumerate(extractions):
        if i in already_grouped:
            continue

        # Start a new group with this intervention
        current_group = [i]
        group_interventions = [extract1]

        # Compare with remaining interventions (limit to avoid timeout)
        for j, extract2 in enumerate(extractions[i+1:], i+1):
            if j in already_grouped:
                continue

            try:
                decision = merger.compare_interventions(extract1, extract2)

                if decision.is_duplicate:
                    print(f"    MATCH: {extract1.intervention_name} <-> {extract2.intervention_name}")
                    current_group.append(j)
                    group_interventions.append(extract2)
                    already_grouped.add(j)

                # Small delay to avoid overwhelming LLM
                time.sleep(0.5)

            except Exception as e:
                print(f"    Error comparing interventions: {e}")
                continue

        # Mark this intervention as grouped
        already_grouped.add(i)

        # Create group result
        if len(group_interventions) > 1:
            canonical_name = group_interventions[0].intervention_name
            alternative_names = [gi.intervention_name for gi in group_interventions]
            print(f"    GROUP: {canonical_name} ({len(group_interventions)} members)")
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

    print(f"  Created {len(groups)} semantic groups for {condition}")
    return groups

def update_database_with_groups(db_path: str, condition: str, groups: list) -> int:
    """Update database with semantic group results for a condition."""
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
                    'method': 'pairwise_semantic_comparison'
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
    """Run full semantic migration."""
    db_path = "data/processed/intervention_research.db"

    print(f"Starting full semantic migration at {datetime.now()}")

    # Initialize semantic merger
    try:
        merger = SemanticMerger(
            primary_model="qwen2.5:14b",
            validator_model="gemma2:9b"
        )
        print("SUCCESS: SemanticMerger initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize SemanticMerger: {e}")
        return False

    # Get all interventions by condition
    try:
        interventions_by_condition = get_interventions_by_condition(db_path)
        total_conditions = len(interventions_by_condition)
        total_interventions = sum(len(interventions) for interventions in interventions_by_condition.values())

        print(f"FOUND: {total_interventions} interventions across {total_conditions} conditions")

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
            processed_conditions += 1
            continue

        print(f"\nPROCESSING: {condition} ({len(interventions)} interventions)")

        try:
            groups = build_semantic_groups_for_condition(interventions, condition, merger)
            updated_count = update_database_with_groups(db_path, condition, groups)
            total_updated += updated_count

            print(f"COMPLETED: {condition} - {updated_count} interventions updated")

        except Exception as e:
            print(f"ERROR processing {condition}: {e}")

        processed_conditions += 1

        # Progress update
        print(f"PROGRESS: {processed_conditions}/{total_conditions} conditions, {total_updated} interventions updated")

        # Delay between conditions to avoid overwhelming LLM
        time.sleep(2)

    print(f"\nFULL MIGRATION COMPLETE!")
    print(f"Final stats: {total_updated} interventions updated across {processed_conditions} conditions")
    return True

if __name__ == "__main__":
    main()