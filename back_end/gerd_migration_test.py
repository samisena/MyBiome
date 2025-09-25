#!/usr/bin/env python3
"""
Test migration focusing specifically on GERD interventions to validate the LLM flow.
"""

import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import SemanticMerger, InterventionExtraction

def main():
    """Test LLM migration with GERD interventions only."""
    db_path = "data/processed/intervention_research.db"

    print(f"Starting GERD-focused migration test at {datetime.now()}")

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

    # Get GERD interventions specifically
    try:
        conn = sqlite3.connect(db_path, timeout=60.0)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, intervention_name, health_condition, intervention_category,
                   correlation_type, confidence_score, correlation_strength,
                   supporting_quote, paper_id
            FROM interventions
            WHERE LOWER(health_condition) LIKE '%ibs%'
            AND (canonical_name IS NULL OR canonical_name = '')
            ORDER BY confidence_score DESC NULLS LAST
            LIMIT 20
        """)

        gerd_interventions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        print(f"FOUND: {len(gerd_interventions)} IBS interventions needing processing")

        for i, interv in enumerate(gerd_interventions):
            print(f"  {i+1}. {interv['intervention_name']} (confidence: {interv.get('confidence_score', 'N/A')})")

    except Exception as e:
        print(f"ERROR: Failed to load GERD interventions: {e}")
        return False

    if not gerd_interventions:
        print("NO GERD interventions found that need processing")
        return True

    # Process GERD interventions
    try:
        print(f"\nPROCESSING: IBS interventions...")

        # Convert to InterventionExtraction objects
        extraction_data = [
            InterventionExtraction(
                model_name='manual_migration',
                intervention_name=interv['intervention_name'],
                health_condition=interv['health_condition'],
                intervention_category=interv.get('intervention_category', 'Unknown'),
                correlation_type=interv.get('correlation_type', 'Unknown'),
                confidence_score=interv.get('confidence_score', 0.5),
                correlation_strength=interv.get('correlation_strength', 0.5),
                supporting_quote=interv.get('supporting_quote', ''),
                raw_data={'id': interv['id'], 'paper_id': interv.get('paper_id', '')}
            ) for interv in gerd_interventions
        ]

        print(f"CALLING: merge_interventions_for_condition with {len(extraction_data)} interventions")

        # Use semantic merger to analyze IBS interventions
        merged_data = merger.merge_interventions_for_condition(
            condition="IBS",
            interventions=extraction_data
        )

        print(f"RETURNED: from merge_interventions_for_condition")

        if merged_data and hasattr(merged_data, 'groups'):
            print(f"FOUND: {len(merged_data.groups)} semantic groups")

            # Update database with semantic analysis results
            updated_count = 0

            for group_idx, group in enumerate(merged_data.groups, 1):
                base_semantic_id = f"gerd_{group_idx}"

                print(f"\nGROUP {group_idx}: '{group.canonical_name}' (confidence: {group.confidence})")
                print(f"  Members: {[m.intervention for m in group.members]}")
                print(f"  Reasoning: {group.reasoning}")

                # Update all interventions in this group
                for member in group.members:
                    # Find the corresponding database intervention
                    db_intervention = None
                    for interv in gerd_interventions:
                        if interv['intervention_name'] == member.intervention:
                            db_intervention = interv
                            break

                    if db_intervention:
                        semantic_updates = {
                            'canonical_name': group.canonical_name,
                            'alternative_names': json.dumps([m.intervention for m in group.members]),
                            'semantic_group_id': base_semantic_id,
                            'semantic_confidence': group.confidence,
                            'merge_source': 'qwen2.5:14b',
                            'validator_agreement': group.validator_agreed,
                            'merge_decision_log': json.dumps({
                                'reasoning': group.reasoning,
                                'processed_at': datetime.now().isoformat(),
                                'group_size': len(group.members)
                            })
                        }

                        # Update database
                        try:
                            conn = sqlite3.connect(db_path, timeout=60.0)
                            cursor = conn.cursor()

                            set_clauses = []
                            values = []

                            for field, value in semantic_updates.items():
                                set_clauses.append(f"{field} = ?")
                                values.append(value)

                            values.append(db_intervention['id'])

                            query = f"""
                                UPDATE interventions
                                SET {', '.join(set_clauses)}
                                WHERE id = ?
                            """

                            cursor.execute(query, values)
                            conn.commit()
                            conn.close()

                            updated_count += 1
                            print(f"    SUCCESS: Updated {member.intervention}")

                        except Exception as e:
                            print(f"    ERROR: Failed to update {member.intervention}: {e}")

            print(f"\nCOMPLETED: GERD migration - {updated_count} interventions updated")

        else:
            print("WARNING: No merged data returned or no groups found")

    except Exception as e:
        print(f"ERROR: Processing GERD interventions failed: {e}")
        return False

    print(f"\nGERD MIGRATION TEST COMPLETE!")
    return True

if __name__ == "__main__":
    main()