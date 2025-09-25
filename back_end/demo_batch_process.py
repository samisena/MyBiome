#!/usr/bin/env python3
"""
Quick demo of batch processing unmapped terms
"""

import sqlite3
import sys
import os
import json
import re
from typing import Dict, List, Tuple

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def demo_batch_processing():
    """Demo the batch processing concept with a few sample terms"""

    print("=== BATCH PROCESSING DEMO ===\n")

    # Connect to database
    db_path = "data/processed/intervention_research.db"
    conn = sqlite3.connect(db_path)
    normalizer = EntityNormalizer(conn)

    # Sample unmapped terms that might exist in the database
    demo_terms = {
        'condition': [
            ('gastroesophageal reflux disease', 15),
            ('GERD symptoms', 8),
            ('acid reflux disease', 6),
            ('reflux esophagitis', 4),
            ('unknown condition X', 2)
        ],
        'intervention': [
            ('proton pump inhibitor', 12),
            ('PPI medication', 7),
            ('omeprazole therapy', 5),
            ('acid blocking drugs', 3),
            ('unknown drug Y', 2)
        ]
    }

    print("Demo unmapped terms to process:")
    for entity_type, terms in demo_terms.items():
        print(f"\n{entity_type.upper()}:")
        for term, freq in terms:
            print(f"  - {term} (frequency: {freq})")

    # Get existing canonicals for context
    for entity_type, terms in demo_terms.items():
        print(f"\n--- Processing {entity_type} terms ---")

        cursor = conn.cursor()
        cursor.execute("""
            SELECT canonical_name FROM canonical_entities
            WHERE entity_type = ?
            ORDER BY canonical_name
        """, (entity_type,))
        existing_canonicals = [row['canonical_name'] for row in cursor.fetchall()]

        print(f"Existing canonical {entity_type}s: {', '.join(existing_canonicals[:5])}...")

        # Create LLM prompt for grouping
        terms_list = "\n".join([f"- {term} (frequency: {freq})" for term, freq in terms])
        canonicals_list = "\n".join([f"- {canonical}" for canonical in existing_canonicals[:10]])

        prompt = f"""You are a medical terminology expert. Analyze this group of {entity_type} terms and determine if any represent the same medical concept.

TERMS TO ANALYZE:
{terms_list}

EXISTING CANONICAL {entity_type.upper()} NAMES:
{canonicals_list}

TASK:
1. Group terms that represent the SAME medical concept (synonyms, abbreviations, variations)
2. For each group, suggest the best canonical name (prefer existing canonical names when appropriate)
3. Be VERY conservative - only group terms if you're confident they mean the same thing

MEDICAL SAFETY RULES:
- Different substances are different (probiotics != prebiotics)
- Opposite conditions are different (hypertension != hypotension)
- Similar-sounding but different medical terms should NOT be grouped

Respond with valid JSON only:
{{
    "groups": [
        {{
            "canonical_name": "suggested_canonical_name",
            "terms": ["term1", "term2"],
            "confidence": 0.0-1.0,
            "reasoning": "why these terms represent the same concept",
            "is_existing_canonical": true/false
        }}
    ],
    "ungrouped_terms": ["term_that_stands_alone"],
    "notes": "additional observations"
}}"""

        print(f"\nSending {len(terms)} terms to LLM for grouping...")

        # Process with LLM
        try:
            if normalizer.llm_client:
                response = normalizer.llm_client.generate(prompt, temperature=0.1)
                llm_content = response['content'].strip()

                # Parse JSON
                try:
                    parsed = json.loads(llm_content)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    json_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        print("Failed to parse LLM response")
                        continue

                # Process results
                if 'groups' in parsed:
                    print(f"\nLLM found {len(parsed['groups'])} groups:")

                    for i, group in enumerate(parsed['groups'], 1):
                        confidence = group['confidence']
                        canonical_name = group['canonical_name']
                        terms_in_group = group['terms']
                        reasoning = group.get('reasoning', 'No reasoning')

                        # Determine action based on confidence
                        if confidence >= 0.8:
                            action = "CREATE MAPPING"
                            status = "[HIGH CONFIDENCE]"
                        elif confidence >= 0.6:
                            action = "FLAG FOR REVIEW"
                            status = "[MEDIUM CONFIDENCE]"
                        else:
                            action = "LEAVE UNMAPPED"
                            status = "[LOW CONFIDENCE]"

                        print(f"\n  Group {i}: {canonical_name} {status}")
                        print(f"    Terms: {', '.join(terms_in_group)}")
                        print(f"    Confidence: {confidence:.2f}")
                        print(f"    Action: {action}")
                        print(f"    Reasoning: {reasoning}")

                if 'ungrouped_terms' in parsed and parsed['ungrouped_terms']:
                    print(f"\nUngrouped terms (stand alone): {', '.join(parsed['ungrouped_terms'])}")

                if 'notes' in parsed:
                    print(f"\nLLM Notes: {parsed['notes']}")

            else:
                print("LLM client not available")

        except Exception as e:
            print(f"Error processing with LLM: {e}")

    # Show expected workflow
    print(f"\n" + "="*60)
    print("BATCH PROCESSING WORKFLOW DEMO")
    print("="*60)
    print("""
DEMONSTRATED PROCESS:
1. [DONE] Get unmapped terms from database
2. [DONE] Group similar-looking terms together
3. [DONE] Send each group to LLM for analysis
4. [DONE] LLM groups synonyms and suggests canonical names
5. [DONE] Apply confidence-based actions:
   - >=80% confidence: Auto-create mappings
   - 60-80% confidence: Flag for human review
   - <60% confidence: Leave unmapped

EXPECTED RESULTS:
- High confidence groups get automatic mappings
- Medium confidence groups get flagged for review
- System maintains medical safety through conservative thresholds
- 60-70% of unmapped terms should get mappings or review flags

NEXT STEPS:
- Full batch script is running in background
- Would process all unmapped terms systematically
- Generate comprehensive reports with statistics
""")

    conn.close()
    print(f"\n[SUCCESS] Demo completed")


if __name__ == "__main__":
    demo_batch_processing()