#!/usr/bin/env python3
"""
Generate mapping suggestions with LLM enhancement
Enhanced version of the mapping suggestion script that uses LLM for semantic matching
"""

import sqlite3
import sys
import os
from datetime import datetime
import csv
from typing import Dict, List, Tuple, Optional

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def analyze_existing_mappings(conn: sqlite3.Connection) -> Dict[str, int]:
    """Analyze current mapping coverage"""

    cursor = conn.cursor()

    # Count total unique terms
    cursor.execute("""
        SELECT
            'intervention' as entity_type,
            COUNT(DISTINCT intervention_name) as total_terms,
            COUNT(DISTINCT CASE WHEN em.raw_text IS NOT NULL THEN intervention_name END) as mapped_terms
        FROM interventions i
        LEFT JOIN entity_mappings em ON i.intervention_name = em.raw_text AND em.entity_type = 'intervention'

        UNION ALL

        SELECT
            'condition' as entity_type,
            COUNT(DISTINCT health_condition) as total_terms,
            COUNT(DISTINCT CASE WHEN em.raw_text IS NOT NULL THEN health_condition END) as mapped_terms
        FROM interventions i
        LEFT JOIN entity_mappings em ON i.health_condition = em.raw_text AND em.entity_type = 'condition'
    """)

    results = cursor.fetchall()
    coverage = {}

    for row in results:
        entity_type = row['entity_type']
        total = row['total_terms']
        mapped = row['mapped_terms']
        coverage[entity_type] = {
            'total': total,
            'mapped': mapped,
            'coverage_percent': (mapped / total * 100) if total > 0 else 0
        }

    return coverage


def get_unmapped_terms_with_frequency(conn: sqlite3.Connection, entity_type: str, min_frequency: int = 1) -> List[Tuple[str, int]]:
    """Get unmapped terms with their frequency counts"""

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


def generate_llm_enhanced_suggestions(normalizer: EntityNormalizer, unmapped_terms: List[Tuple[str, int]],
                                    entity_type: str, batch_size: int = 20) -> List[Dict]:
    """Generate mapping suggestions using safe methods + LLM enhancement"""

    suggestions = []

    print(f"\n=== Processing {len(unmapped_terms)} unmapped {entity_type} terms ===")

    # Process in batches for better performance
    for i in range(0, len(unmapped_terms), batch_size):
        batch = unmapped_terms[i:i + batch_size]
        batch_terms = [term for term, freq in batch]
        freq_map = {term: freq for term, freq in batch}

        print(f"\nProcessing batch {i//batch_size + 1}: {len(batch_terms)} terms...")

        for term in batch_terms:
            frequency = freq_map[term]

            # Try safe methods first
            safe_matches = normalizer.find_safe_matches_only(term, entity_type)

            if safe_matches:
                # Use best safe match
                best_match = safe_matches[0]
                suggestions.append({
                    'entity_type': entity_type,
                    'original_term': term,
                    'frequency': frequency,
                    'suggested_canonical': best_match['canonical_name'],
                    'confidence': best_match['confidence'],
                    'method': best_match['match_method'],
                    'canonical_id': best_match['id'],
                    'notes': 'Safe pattern/exact matching'
                })
                print(f"  [SAFE] {term} -> {best_match['canonical_name']} ({best_match['match_method']})")

            else:
                # Try LLM semantic matching
                llm_match = normalizer.find_by_llm(term, entity_type)

                if llm_match and llm_match['confidence'] >= 0.7:  # High confidence threshold
                    suggestions.append({
                        'entity_type': entity_type,
                        'original_term': term,
                        'frequency': frequency,
                        'suggested_canonical': llm_match['canonical_name'],
                        'confidence': llm_match['confidence'],
                        'method': 'llm_semantic',
                        'canonical_id': llm_match['id'],
                        'notes': f"LLM match: {llm_match.get('reasoning', 'No reasoning')}"
                    })
                    print(f"  [LLM-HIGH] {term} -> {llm_match['canonical_name']} ({llm_match['confidence']:.2f})")

                elif llm_match and llm_match['confidence'] >= 0.5:  # Medium confidence
                    suggestions.append({
                        'entity_type': entity_type,
                        'original_term': term,
                        'frequency': frequency,
                        'suggested_canonical': llm_match['canonical_name'],
                        'confidence': llm_match['confidence'],
                        'method': 'llm_semantic_review',
                        'canonical_id': llm_match['id'],
                        'notes': f"LLM match - REVIEW NEEDED: {llm_match.get('reasoning', 'No reasoning')}"
                    })
                    print(f"  [LLM-MED] {term} -> {llm_match['canonical_name']} ({llm_match['confidence']:.2f}) - REVIEW")

                else:
                    # No good match found
                    suggestions.append({
                        'entity_type': entity_type,
                        'original_term': term,
                        'frequency': frequency,
                        'suggested_canonical': None,
                        'confidence': 0.0,
                        'method': 'no_match',
                        'canonical_id': None,
                        'notes': 'No safe or confident LLM match found - manual review needed'
                    })
                    print(f"  [NO-MATCH] {term} - no confident match")

    return suggestions


def save_suggestions_to_csv(suggestions: List[Dict], output_path: str):
    """Save suggestions to CSV file"""

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['entity_type', 'original_term', 'frequency', 'suggested_canonical',
                     'confidence', 'method', 'canonical_id', 'notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for suggestion in suggestions:
            writer.writerow(suggestion)


def generate_summary_report(suggestions: List[Dict], coverage: Dict) -> str:
    """Generate a summary report of the mapping suggestions"""

    # Categorize suggestions by method and confidence
    categories = {
        'safe_matches': [],
        'llm_high_confidence': [],
        'llm_medium_confidence': [],
        'no_matches': []
    }

    for suggestion in suggestions:
        method = suggestion['method']
        confidence = suggestion['confidence']

        if method in ['existing_mapping', 'exact_normalized', 'safe_pattern']:
            categories['safe_matches'].append(suggestion)
        elif method == 'llm_semantic' and confidence >= 0.7:
            categories['llm_high_confidence'].append(suggestion)
        elif method in ['llm_semantic', 'llm_semantic_review'] and confidence >= 0.5:
            categories['llm_medium_confidence'].append(suggestion)
        else:
            categories['no_matches'].append(suggestion)

    # Generate report
    report = []
    report.append("=" * 60)
    report.append("LLM-ENHANCED MAPPING SUGGESTIONS REPORT")
    report.append("=" * 60)

    # Current coverage
    report.append("\nCURRENT MAPPING COVERAGE:")
    for entity_type, stats in coverage.items():
        report.append(f"  {entity_type}: {stats['mapped']}/{stats['total']} ({stats['coverage_percent']:.1f}%)")

    # New suggestions breakdown
    report.append(f"\nNEW MAPPING SUGGESTIONS:")
    report.append(f"  Total terms analyzed: {len(suggestions)}")
    report.append(f"  Safe matches (ready to apply): {len(categories['safe_matches'])}")
    report.append(f"  LLM high confidence (≥70%): {len(categories['llm_high_confidence'])}")
    report.append(f"  LLM medium confidence (50-69%): {len(categories['llm_medium_confidence'])}")
    report.append(f"  No confident matches: {len(categories['no_matches'])}")

    # Action recommendations
    report.append("\nRECOMMENDED ACTIONS:")
    if categories['safe_matches']:
        report.append(f"1. AUTO-APPLY {len(categories['safe_matches'])} safe matches")
    if categories['llm_high_confidence']:
        report.append(f"2. REVIEW & APPLY {len(categories['llm_high_confidence'])} high-confidence LLM matches")
    if categories['llm_medium_confidence']:
        report.append(f"3. MANUAL REVIEW {len(categories['llm_medium_confidence'])} medium-confidence LLM matches")
    if categories['no_matches']:
        report.append(f"4. RESEARCH {len(categories['no_matches'])} unmatched terms (create new canonical entities?)")

    # Top frequency unmatched terms
    no_matches_by_freq = sorted(categories['no_matches'], key=lambda x: x['frequency'], reverse=True)
    if no_matches_by_freq:
        report.append("\nTOP UNMATCHED TERMS (by frequency):")
        for suggestion in no_matches_by_freq[:10]:
            report.append(f"  {suggestion['original_term']} (freq: {suggestion['frequency']})")

    return "\n".join(report)


def main():
    """Main function to generate LLM-enhanced mapping suggestions"""

    print("=== LLM-Enhanced Mapping Suggestion Generator ===\n")

    # Connect to database
    db_path = "data/processed/intervention_research.db"
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Initialize entity normalizer with LLM
    normalizer = EntityNormalizer(conn, llm_model="gemma2:9b")

    # Check if LLM is available
    if not normalizer.llm_client:
        print("Warning: LLM client not available. Will only use safe matching methods.")

    # Analyze current coverage
    print("Analyzing current mapping coverage...")
    coverage = analyze_existing_mappings(conn)

    for entity_type, stats in coverage.items():
        print(f"{entity_type}: {stats['mapped']}/{stats['total']} mapped ({stats['coverage_percent']:.1f}%)")

    # Generate suggestions for both entity types
    all_suggestions = []

    for entity_type in ['condition', 'intervention']:
        print(f"\n--- Processing {entity_type} terms ---")

        # Get unmapped terms (frequency >= 2 to focus on common terms)
        unmapped_terms = get_unmapped_terms_with_frequency(conn, entity_type, min_frequency=2)

        if unmapped_terms:
            # Generate suggestions with LLM enhancement
            suggestions = generate_llm_enhanced_suggestions(normalizer, unmapped_terms, entity_type)
            all_suggestions.extend(suggestions)
        else:
            print(f"No unmapped {entity_type} terms found.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"mapping_suggestions_llm_enhanced_{timestamp}.csv"
    save_suggestions_to_csv(all_suggestions, csv_path)

    # Generate and display summary report
    report = generate_summary_report(all_suggestions, coverage)
    print("\n" + report)

    # Save report to file
    report_path = f"mapping_suggestions_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n=== Results Saved ===")
    print(f"Suggestions CSV: {csv_path}")
    print(f"Summary report: {report_path}")
    print(f"Total suggestions: {len(all_suggestions)}")

    # Show cache statistics
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM llm_normalization_cache")
    cache_count = cursor.fetchone()[0]
    print(f"LLM cache entries: {cache_count}")

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