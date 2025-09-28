#!/usr/bin/env python3
"""
Batch process unmapped terms with LLM-based grouping and mapping
"""

import sqlite3
import sys
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import csv

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ..llm_processing.batch_entity_processor import BatchEntityProcessor as EntityNormalizer


def get_unmapped_terms(conn: sqlite3.Connection, entity_type: str, min_frequency: int = 2) -> List[Tuple[str, int]]:
    """Get terms that have no mappings (safe or otherwise)"""

    cursor = conn.cursor()

    if entity_type == 'intervention':
        column = 'intervention_name'
    else:
        column = 'health_condition'

    cursor.execute(f"""
        SELECT
            i.{column} as term,
            COUNT(*) as frequency
        FROM interventions i
        LEFT JOIN entity_mappings em ON i.{column} = em.raw_text AND em.entity_type = ?
        WHERE em.raw_text IS NULL
        AND i.{column} IS NOT NULL
        AND TRIM(i.{column}) != ''
        GROUP BY i.{column}
        HAVING frequency >= ?
        ORDER BY frequency DESC
    """, (entity_type, min_frequency))

    return [(row['term'], row['frequency']) for row in cursor.fetchall()]


def normalize_for_grouping(term: str) -> str:
    """Normalize term for similarity grouping"""
    # Convert to lowercase, remove punctuation, normalize spacing
    normalized = re.sub(r'[^\w\s]', '', term.lower())
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def group_similar_terms(terms: List[Tuple[str, int]], max_group_size: int = 8) -> List[List[Tuple[str, int]]]:
    """Group terms that look similar for batch LLM processing"""

    # Create groups based on normalized forms and common patterns
    groups = defaultdict(list)

    for term, freq in terms:
        normalized = normalize_for_grouping(term)

        # Group by key characteristics
        key_parts = []

        # Extract key medical words (longer than 3 chars)
        words = [w for w in normalized.split() if len(w) > 3]
        if words:
            # Sort words to group anagrams and similar terms
            key_parts.extend(sorted(words))

        # Group by length category
        length_category = len(normalized) // 10  # Group by length deciles
        key_parts.append(f"len_{length_category}")

        # Create grouping key
        group_key = "_".join(key_parts[:3])  # Use first 3 components to avoid over-fragmenting
        groups[group_key].append((term, freq))

    # Convert to list and split large groups
    final_groups = []
    for group_terms in groups.values():
        # Sort by frequency within group
        group_terms.sort(key=lambda x: x[1], reverse=True)

        # Split large groups
        while group_terms:
            chunk = group_terms[:max_group_size]
            group_terms = group_terms[max_group_size:]
            final_groups.append(chunk)

    # Sort groups by total frequency
    final_groups.sort(key=lambda g: sum(freq for _, freq in g), reverse=True)

    return final_groups


def build_group_llm_prompt(terms: List[Tuple[str, int]], entity_type: str, existing_canonicals: List[str]) -> str:
    """Build LLM prompt for processing a group of similar terms"""

    terms_list = "\n".join([f"- {term} (frequency: {freq})" for term, freq in terms])
    canonicals_list = "\n".join([f"- {canonical}" for canonical in existing_canonicals[:20]])  # Limit to top 20

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
- Different dosages/formulations may be different concepts
- Conservative is better than wrong

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

    return prompt


def process_group_with_llm(normalizer: EntityNormalizer, terms: List[Tuple[str, int]],
                          entity_type: str, existing_canonicals: List[str]) -> Dict:
    """Process a group of terms with LLM"""

    if not normalizer.llm_client:
        return {"groups": [], "ungrouped_terms": [term for term, _ in terms], "notes": "LLM not available"}

    prompt = build_group_llm_prompt(terms, entity_type, existing_canonicals)

    try:
        response = normalizer.llm_client.generate(prompt, temperature=0.1)
        llm_content = response['content'].strip()

        # Parse JSON response (using the same logic as in EntityNormalizer)
        try:
            parsed = json.loads(llm_content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                return {"groups": [], "ungrouped_terms": [term for term, _ in terms], "notes": "JSON parsing failed"}

        return parsed

    except Exception as e:
        print(f"Error processing group with LLM: {e}")
        return {"groups": [], "ungrouped_terms": [term for term, _ in terms], "notes": f"LLM error: {e}"}


def create_mapping_from_group(normalizer: EntityNormalizer, group: Dict, entity_type: str) -> Dict:
    """Create mapping from a group suggestion"""

    canonical_name = group['canonical_name']
    terms = group['terms']
    confidence = group['confidence']
    reasoning = group.get('reasoning', '')
    is_existing = group.get('is_existing_canonical', False)

    result = {
        'canonical_name': canonical_name,
        'terms': terms,
        'confidence': confidence,
        'reasoning': reasoning,
        'action': 'none',
        'canonical_id': None,
        'error': None
    }

    try:
        # Check if canonical entity exists or needs to be created
        canonical_id = None

        if is_existing:
            # Try to find existing canonical
            cursor = normalizer.db.cursor()
            cursor.execute("""
                SELECT id FROM canonical_entities
                WHERE canonical_name = ? AND entity_type = ?
            """, (canonical_name, entity_type))
            row = cursor.fetchone()
            if row:
                canonical_id = row['id']

        if not canonical_id:
            # Create new canonical entity
            canonical_id = normalizer.create_canonical_entity(canonical_name, entity_type)
            result['created_canonical'] = True
        else:
            result['created_canonical'] = False

        result['canonical_id'] = canonical_id

        # Determine action based on confidence
        if confidence >= 0.8:
            # High confidence - create mappings automatically
            for term in terms:
                try:
                    normalizer.add_term_mapping(term, canonical_id, confidence, "llm_batch_high")
                    result['action'] = 'created'
                except Exception as e:
                    # May already exist
                    if 'UNIQUE constraint failed' in str(e):
                        result['action'] = 'already_exists'
                    else:
                        raise e

        elif confidence >= 0.6:
            # Medium confidence - flag for human review
            result['action'] = 'review_needed'

        else:
            # Low confidence - leave unmapped
            result['action'] = 'low_confidence'

    except Exception as e:
        result['error'] = str(e)
        result['action'] = 'error'

    return result


def process_unmapped_terms_batch(conn: sqlite3.Connection, entity_type: str, max_terms: int = 100) -> Dict:
    """Process unmapped terms in batches with LLM grouping"""

    print(f"\n=== Processing unmapped {entity_type} terms ===")

    normalizer = EntityNormalizer(conn)

    # Get unmapped terms
    unmapped_terms = get_unmapped_terms(conn, entity_type, min_frequency=2)

    if not unmapped_terms:
        print(f"No unmapped {entity_type} terms found.")
        return {
            'entity_type': entity_type,
            'total_unmapped': 0,
            'processed': 0,
            'results': []
        }

    print(f"Found {len(unmapped_terms)} unmapped {entity_type} terms")

    # Limit processing for demo
    if len(unmapped_terms) > max_terms:
        print(f"Limiting to first {max_terms} terms for processing")
        unmapped_terms = unmapped_terms[:max_terms]

    # Get existing canonical names for context
    cursor = conn.cursor()
    cursor.execute("""
        SELECT canonical_name FROM canonical_entities
        WHERE entity_type = ?
        ORDER BY canonical_name
    """, (entity_type,))
    existing_canonicals = [row['canonical_name'] for row in cursor.fetchall()]

    # Group similar terms
    print("Grouping similar terms...")
    groups = group_similar_terms(unmapped_terms, max_group_size=6)
    print(f"Created {len(groups)} groups")

    # Process each group
    results = []
    processed_terms = set()

    for i, group_terms in enumerate(groups, 1):
        print(f"\nProcessing group {i}/{len(groups)}: {len(group_terms)} terms")
        group_term_names = [term for term, _ in group_terms]
        print(f"  Terms: {', '.join(group_term_names[:3])}{'...' if len(group_term_names) > 3 else ''}")

        # Process with LLM
        llm_result = process_group_with_llm(normalizer, group_terms, entity_type, existing_canonicals)

        # Handle LLM response
        if llm_result.get('groups'):
            for group_suggestion in llm_result['groups']:
                mapping_result = create_mapping_from_group(normalizer, group_suggestion, entity_type)
                mapping_result['group_id'] = i
                results.append(mapping_result)

                # Track processed terms
                for term in group_suggestion['terms']:
                    processed_terms.add(term)

                print(f"    -> {group_suggestion['canonical_name']} ({mapping_result['action']}, {group_suggestion['confidence']:.2f})")

        # Handle ungrouped terms
        if llm_result.get('ungrouped_terms'):
            for term in llm_result['ungrouped_terms']:
                if term not in processed_terms:
                    results.append({
                        'canonical_name': None,
                        'terms': [term],
                        'confidence': 0.0,
                        'reasoning': 'LLM determined this term stands alone',
                        'action': 'ungrouped',
                        'canonical_id': None,
                        'group_id': i,
                        'error': None
                    })
                    processed_terms.add(term)

    # Handle any unprocessed terms
    all_unmapped_term_names = {term for term, _ in unmapped_terms}
    unprocessed_terms = all_unmapped_term_names - processed_terms

    for term in unprocessed_terms:
        results.append({
            'canonical_name': None,
            'terms': [term],
            'confidence': 0.0,
            'reasoning': 'Not processed',
            'action': 'unprocessed',
            'canonical_id': None,
            'group_id': None,
            'error': None
        })

    return {
        'entity_type': entity_type,
        'total_unmapped': len(unmapped_terms),
        'processed': len(processed_terms),
        'results': results
    }


def generate_batch_report(condition_results: Dict, intervention_results: Dict) -> str:
    """Generate comprehensive batch processing report"""

    all_results = condition_results['results'] + intervention_results['results']

    # Categorize results
    high_confidence = [r for r in all_results if r['action'] == 'created']
    review_needed = [r for r in all_results if r['action'] == 'review_needed']
    low_confidence = [r for r in all_results if r['confidence'] > 0 and r['confidence'] < 0.6]
    ungrouped = [r for r in all_results if r['action'] == 'ungrouped']
    errors = [r for r in all_results if r['error']]

    report = []
    report.append("=" * 80)
    report.append("BATCH PROCESSING REPORT - UNMAPPED TERMS")
    report.append("=" * 80)

    # Summary
    total_unmapped = condition_results['total_unmapped'] + intervention_results['total_unmapped']
    total_processed = condition_results['processed'] + intervention_results['processed']

    report.append(f"\nSUMMARY:")
    report.append(f"  Total unmapped terms: {total_unmapped}")
    report.append(f"  Terms processed: {total_processed}")
    report.append(f"  Processing coverage: {(total_processed/total_unmapped)*100 if total_unmapped > 0 else 0:.1f}%")

    # Breakdown by entity type
    report.append(f"\nBREAKDOWN BY TYPE:")
    report.append(f"  Conditions: {condition_results['total_unmapped']} unmapped → {condition_results['processed']} processed")
    report.append(f"  Interventions: {intervention_results['total_unmapped']} unmapped → {intervention_results['processed']} processed")

    # Results breakdown
    report.append(f"\nRESULTS:")
    report.append(f"  ✅ New mappings created (≥80% confidence): {len(high_confidence)}")
    report.append(f"  ⚠️  Mappings needing review (60-80% confidence): {len(review_needed)}")
    report.append(f"  ❌ Left unmapped (<60% confidence): {len(low_confidence) + len(ungrouped)}")
    if errors:
        report.append(f"  🔥 Errors: {len(errors)}")

    # Success metric
    mapped_or_flagged = len(high_confidence) + len(review_needed)
    success_rate = (mapped_or_flagged / total_processed) * 100 if total_processed > 0 else 0
    report.append(f"\nSUCCESS METRIC:")
    report.append(f"  Terms with mappings or review flags: {mapped_or_flagged}/{total_processed} ({success_rate:.1f}%)")

    target_met = "✅ TARGET MET" if success_rate >= 60 else "❌ TARGET NOT MET"
    report.append(f"  Target (60-70%): {target_met}")

    # Detailed sections
    if high_confidence:
        report.append(f"\n--- NEW MAPPINGS CREATED (HIGH CONFIDENCE ≥80%) ---")
        for result in high_confidence:
            terms_str = ", ".join(result['terms'])
            report.append(f"  {result['canonical_name']}: {terms_str} ({result['confidence']:.2f})")
            report.append(f"    Reasoning: {result['reasoning']}")

    if review_needed:
        report.append(f"\n--- MAPPINGS NEEDING HUMAN REVIEW (60-80% CONFIDENCE) ---")
        for result in review_needed:
            terms_str = ", ".join(result['terms'])
            report.append(f"  {result['canonical_name']}: {terms_str} ({result['confidence']:.2f})")
            report.append(f"    Reasoning: {result['reasoning']}")

    if low_confidence or ungrouped:
        report.append(f"\n--- TERMS LEFT UNMAPPED ---")
        unmapped_results = low_confidence + ungrouped
        for result in unmapped_results[:20]:  # Show first 20
            terms_str = ", ".join(result['terms'])
            report.append(f"  {terms_str}: {result['reasoning'] if result['reasoning'] else 'Low confidence/ungrouped'}")

        if len(unmapped_results) > 20:
            report.append(f"  ... and {len(unmapped_results) - 20} more terms")

    if errors:
        report.append(f"\n--- ERRORS ---")
        for result in errors:
            terms_str = ", ".join(result['terms'])
            report.append(f"  {terms_str}: {result['error']}")

    return "\n".join(report)


def save_detailed_results(condition_results: Dict, intervention_results: Dict, timestamp: str):
    """Save detailed results to CSV files"""

    # Combine all results
    all_results = []

    for result in condition_results['results']:
        for term in result['terms']:
            all_results.append({
                'entity_type': 'condition',
                'original_term': term,
                'canonical_name': result['canonical_name'],
                'confidence': result['confidence'],
                'action': result['action'],
                'canonical_id': result['canonical_id'],
                'reasoning': result['reasoning'],
                'error': result['error'],
                'group_id': result.get('group_id')
            })

    for result in intervention_results['results']:
        for term in result['terms']:
            all_results.append({
                'entity_type': 'intervention',
                'original_term': term,
                'canonical_name': result['canonical_name'],
                'confidence': result['confidence'],
                'action': result['action'],
                'canonical_id': result['canonical_id'],
                'reasoning': result['reasoning'],
                'error': result['error'],
                'group_id': result.get('group_id')
            })

    # Save to CSV
    csv_path = f"batch_processing_results_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['entity_type', 'original_term', 'canonical_name', 'confidence',
                     'action', 'canonical_id', 'reasoning', 'error', 'group_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    return csv_path


def main():
    """Main batch processing function"""

    print("=== BATCH PROCESSING UNMAPPED TERMS ===\n")

    # Connect to database
    db_path = "data/processed/intervention_research.db"
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Process both entity types
        condition_results = process_unmapped_terms_batch(conn, 'condition', max_terms=50)
        intervention_results = process_unmapped_terms_batch(conn, 'intervention', max_terms=50)

        # Generate report
        report = generate_batch_report(condition_results, intervention_results)
        print("\n" + report)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed CSV
        csv_path = save_detailed_results(condition_results, intervention_results, timestamp)

        # Save report
        report_path = f"batch_processing_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n=== FILES CREATED ===")
        print(f"Detailed results: {csv_path}")
        print(f"Summary report: {report_path}")

        # Show cache statistics
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM llm_normalization_cache")
        cache_count = cursor.fetchone()[0]
        print(f"LLM cache entries: {cache_count}")

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Process stopped by user")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()