#!/usr/bin/env python3
"""
LLM-Based Canonical Entity Deduplication
"""

import sqlite3
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Set up logging for deduplication"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'data/logs/llm_deduplication_{timestamp}.log'

    os.makedirs('data/logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def get_canonical_entities_by_type(cursor, entity_type):
    """Get all canonical entities of a specific type with usage counts"""
    cursor.execute("""
        SELECT ce.id, ce.canonical_name,
               COALESCE(intervention_count, 0) as intervention_usage,
               COALESCE(condition_count, 0) as condition_usage,
               (COALESCE(intervention_count, 0) + COALESCE(condition_count, 0)) as total_usage
        FROM canonical_entities ce
        LEFT JOIN (
            SELECT intervention_canonical_id, COUNT(*) as intervention_count
            FROM interventions
            WHERE intervention_canonical_id IS NOT NULL
            GROUP BY intervention_canonical_id
        ) i ON i.intervention_canonical_id = ce.id
        LEFT JOIN (
            SELECT condition_canonical_id, COUNT(*) as condition_count
            FROM interventions
            WHERE condition_canonical_id IS NOT NULL
            GROUP BY condition_canonical_id
        ) c ON c.condition_canonical_id = ce.id
        WHERE ce.entity_type = ?
        ORDER BY total_usage DESC
    """, (entity_type,))

    return cursor.fetchall()

def get_llm_deduplication(terms: List[str]) -> Dict[str, Any]:
    """Get LLM analysis of duplicate terms"""

    # Import here to avoid circular imports
    from data.api_clients import get_llm_client

    client = get_llm_client()

    terms_list = "\n".join([f"- {term}" for term in terms])

    prompt = f"""Analyze these medical terms and identify which ones refer to the same concept.

Terms to analyze:
{terms_list}

Return ONLY valid JSON in this format:
{{
  "duplicate_groups": [
    {{
      "canonical_name": "most formal scientific name",
      "synonyms": ["term1", "term2", "term3"],
      "confidence": 0.95
    }}
  ]
}}

Rules:
- Each group must have confidence 0.0-1.0
- Use the most formal medical/scientific name as canonical_name
- Only group terms that are definitely the same concept"""

    try:
        response = client.generate(prompt, temperature=0.1)
        response_text = response['content'].strip()

        # Clean response - remove markdown formatting if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        result = json.loads(response_text.strip())
        return result
    except Exception as e:
        logging.error(f"LLM deduplication failed: {str(e)}")
        return {"duplicate_groups": []}

def merge_canonical_entities(cursor, canonical_name: str, synonyms: List[str], entity_type: str):
    """Merge duplicate canonical entities into one"""

    # Find the canonical entity to keep (prefer existing one with canonical_name)
    cursor.execute("""
        SELECT id, canonical_name
        FROM canonical_entities
        WHERE canonical_name = ? AND entity_type = ?
    """, (canonical_name, entity_type))

    target_entity = cursor.fetchone()

    if not target_entity:
        # Create new canonical entity
        cursor.execute("""
            INSERT INTO canonical_entities (canonical_name, entity_type, confidence_score)
            VALUES (?, ?, ?)
        """, (canonical_name, entity_type, 1.0))
        target_id = cursor.lastrowid
        logging.info(f"Created new canonical: '{canonical_name}' -> ID {target_id}")
    else:
        target_id = target_entity[0]
        logging.info(f"Using existing canonical: '{canonical_name}' -> ID {target_id}")

    # Get all entities to merge
    synonym_placeholders = ','.join(['?' for _ in synonyms])
    cursor.execute(f"""
        SELECT id, canonical_name
        FROM canonical_entities
        WHERE canonical_name IN ({synonym_placeholders}) AND entity_type = ? AND id != ?
    """, synonyms + [entity_type, target_id])

    entities_to_merge = cursor.fetchall()

    if not entities_to_merge:
        logging.info(f"No entities to merge for {canonical_name}")
        return 0

    merged_count = 0

    for entity_id, entity_name in entities_to_merge:
        logging.info(f"Merging '{entity_name}' (ID {entity_id}) into '{canonical_name}' (ID {target_id})")

        # Update intervention records
        if entity_type == 'intervention':
            cursor.execute("""
                UPDATE interventions
                SET intervention_canonical_id = ?
                WHERE intervention_canonical_id = ?
            """, (target_id, entity_id))

            updated_interventions = cursor.rowcount
            logging.info(f"Updated {updated_interventions} intervention records")

        elif entity_type == 'condition':
            cursor.execute("""
                UPDATE interventions
                SET condition_canonical_id = ?
                WHERE condition_canonical_id = ?
            """, (target_id, entity_id))

            updated_conditions = cursor.rowcount
            logging.info(f"Updated {updated_conditions} condition records")

        # Update entity mappings
        cursor.execute("""
            UPDATE entity_mappings
            SET canonical_id = ?
            WHERE canonical_id = ?
        """, (target_id, entity_id))

        updated_mappings = cursor.rowcount
        logging.info(f"Updated {updated_mappings} entity mappings")

        # Add synonym to entity_mappings if not exists
        cursor.execute("""
            INSERT OR IGNORE INTO entity_mappings
            (raw_text, canonical_id, entity_type, confidence_score)
            VALUES (?, ?, ?, ?)
        """, (entity_name, target_id, entity_type, 0.95))

        # Delete the old canonical entity
        cursor.execute("""
            DELETE FROM canonical_entities
            WHERE id = ?
        """, (entity_id,))

        merged_count += 1

    return merged_count

def run_deduplication():
    """Run LLM-based deduplication process"""

    db_path = 'data/processed/intervention_research.db'

    # Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{db_path.replace('.db', '')}_deduplication_backup_{timestamp}.db"

    import shutil
    shutil.copy2(db_path, backup_path)
    logging.info(f"Created backup: {backup_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        total_merged = 0

        # Process each entity type
        for entity_type in ['intervention', 'condition']:
            logging.info(f"Processing {entity_type} entities...")

            entities = get_canonical_entities_by_type(cursor, entity_type)

            if len(entities) < 2:
                logging.info(f"Not enough {entity_type} entities to deduplicate")
                continue

            # Get terms with usage > 0
            used_entities = [e for e in entities if e[4] > 0]  # total_usage > 0

            if len(used_entities) < 2:
                logging.info(f"Not enough used {entity_type} entities to deduplicate")
                continue

            # Extract just the canonical names for LLM analysis
            term_names = [entity[1] for entity in used_entities[:50]]  # Limit to top 50 most used

            logging.info(f"Analyzing {len(term_names)} {entity_type} terms for duplicates...")

            # Get LLM deduplication results
            llm_result = get_llm_deduplication(term_names)

            duplicate_groups = llm_result.get('duplicate_groups', [])
            logging.info(f"Found {len(duplicate_groups)} potential duplicate groups")

            # Process each duplicate group
            for group in duplicate_groups:
                canonical_name = group.get('canonical_name', '')
                synonyms = group.get('synonyms', [])
                confidence = group.get('confidence', 0.0)

                if confidence > 0.8 and len(synonyms) > 1:
                    logging.info(f"Merging group: '{canonical_name}' <- {synonyms} (confidence: {confidence})")

                    merged = merge_canonical_entities(cursor, canonical_name, synonyms, entity_type)
                    total_merged += merged

                    if merged > 0:
                        conn.commit()
                else:
                    logging.info(f"Skipping group with low confidence: {confidence} or insufficient synonyms")

        logging.info(f"""
================================================================================
DEDUPLICATION COMPLETE
================================================================================
Total entities merged: {total_merged}
Backup created: {backup_path}
================================================================================
        """)

if __name__ == "__main__":
    log_file = setup_logging()

    try:
        run_deduplication()
        print(f"\nDeduplication complete! Check log file: {log_file}")
    except Exception as e:
        logging.error(f"Deduplication failed: {str(e)}")
        raise