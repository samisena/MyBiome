#!/usr/bin/env python3
"""
Direct LLM migration that bypasses database manager initialization issues.
Directly connects to the database and runs semantic merger on interventions.
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
        condition = intervention['health_condition'].lower().strip()
        if condition not in by_condition:
            by_condition[condition] = []
        by_condition[condition].append(intervention)

    return by_condition

def update_intervention_semantics(db_path: str, intervention_id: int, updates: Dict[str, Any]) -> bool:
    """Update intervention with semantic analysis results."""
    try:
        conn = sqlite3.connect(db_path, timeout=60.0)
        cursor = conn.cursor()

        # Build update query dynamically
        set_clauses = []
        values = []

        for field, value in updates.items():
            set_clauses.append(f"{field} = ?")
            if isinstance(value, (dict, list)):
                values.append(json.dumps(value))
            else:
                values.append(value)

        values.append(intervention_id)  # For WHERE clause

        query = f"""
            UPDATE interventions
            SET {', '.join(set_clauses)}
            WHERE id = ?
        """

        cursor.execute(query, values)
        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"Error updating intervention {intervention_id}: {e}")
        return False

def main():
    """Run the direct LLM migration."""
    db_path = "data/processed/intervention_research.db"

    print(f"Starting direct LLM migration at {datetime.now()}")

    # Initialize semantic merger
    try:
        merger = SemanticMerger(
            primary_model="qwen2.5:14b",
            validator_model="gemma2:9b"
        )
        print("SUCCESS: SemanticMerger initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize SemanticMerger: {e}")
        return False

    # Get interventions by condition
    try:
        interventions_by_condition = get_interventions_by_condition(db_path)
        total_conditions = len(interventions_by_condition)
        total_interventions = sum(len(interventions) for interventions in interventions_by_condition.values())

        print(f"FOUND: {total_interventions} interventions across {total_conditions} conditions")
    except Exception as e:
        print(f"ERROR: Failed to load interventions: {e}")
        return False

    # Process each condition
    processed_conditions = 0
    processed_interventions = 0
    updated_interventions = 0

    for condition, interventions in interventions_by_condition.items():
        if len(interventions) < 2:
            print(f"SKIP: {condition} (only {len(interventions)} intervention)")
            processed_conditions += 1
            processed_interventions += len(interventions)
            continue

        print(f"PROCESSING: {condition} ({len(interventions)} interventions)...")

        try:
            # Convert to InterventionExtraction objects
            extraction_data = [
                InterventionExtraction(
                    intervention=interv['intervention_name'],
                    category=interv.get('intervention_category', 'Unknown'),
                    correlation_type=interv.get('correlation_type', 'Unknown'),
                    confidence=interv.get('confidence_score', 0.5),
                    supporting_quote=interv.get('supporting_quote', ''),
                    paper_id=interv.get('paper_id', '')
                ) for interv in interventions
            ]

            # Use semantic merger to analyze this condition's interventions
            merged_data = merger.merge_interventions_for_condition(
                condition=condition,
                interventions=extraction_data
            )

            if merged_data and hasattr(merged_data, 'groups'):
                # Update database with semantic analysis results
                group_id_counter = 1
                for group in merged_data.groups:
                    base_semantic_id = f"{condition.replace(' ', '_')}_{group_id_counter}"

                    # Update all interventions in this group
                    for i, member in enumerate(group.members):
                        # Find the corresponding database intervention
                        db_intervention = None
                        for interv in interventions:
                            if interv['intervention_name'] == member.intervention:
                                db_intervention = interv
                                break

                        if db_intervention:
                            semantic_updates = {
                                'canonical_name': group.canonical_name,
                                'alternative_names': [m.intervention for m in group.members],
                                'semantic_group_id': base_semantic_id,
                                'semantic_confidence': group.confidence,
                                'merge_source': 'qwen2.5:14b',
                                'validator_agreement': group.validator_agreed,
                                'merge_decision_log': {
                                    'reasoning': group.reasoning,
                                    'processed_at': datetime.now().isoformat(),
                                    'group_size': len(group.members)
                                }
                            }

                            if update_intervention_semantics(db_path, db_intervention['id'], semantic_updates):
                                updated_interventions += 1
                                print(f"  SUCCESS: Updated intervention: {member.intervention} -> {group.canonical_name}")
                            else:
                                print(f"  ERROR: Failed to update: {member.intervention}")

                    group_id_counter += 1

            print(f"COMPLETED: {condition}")

        except Exception as e:
            print(f"ERROR: Error processing {condition}: {e}")

        processed_conditions += 1
        processed_interventions += len(interventions)

        # Progress update
        print(f"PROGRESS: {processed_conditions}/{total_conditions} conditions, {processed_interventions}/{total_interventions} interventions, {updated_interventions} updated")

        # Small delay to prevent overwhelming the LLM
        time.sleep(1)

    print(f"\nMIGRATION COMPLETE!")
    print(f"FINAL STATS: {updated_interventions} interventions updated across {processed_conditions} conditions")
    return True

if __name__ == "__main__":
    main()