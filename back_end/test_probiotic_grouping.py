#!/usr/bin/env python3
"""
Test probiotic grouping for success check
"""

import sqlite3
import sys
import os
from collections import defaultdict

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from entity_normalizer import EntityNormalizer


def test_probiotic_grouping():
    """Test grouping of probiotic-related interventions"""

    print('=== TESTING SUCCESS CHECK: PROBIOTIC GROUPING ===')

    db_path = 'data/processed/intervention_research.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    normalizer = EntityNormalizer(conn)

    # Find probiotic-related interventions
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT intervention_name
        FROM interventions
        WHERE intervention_name LIKE '%probiotic%'
        ORDER BY intervention_name
    """)

    probiotic_interventions = [row['intervention_name'] for row in cursor.fetchall()]

    print(f'Found {len(probiotic_interventions)} probiotic-related interventions:')
    for intervention in probiotic_interventions:
        print(f'  - {intervention}')

    print()

    # Test grouping by canonical names
    groups = defaultdict(list)

    for intervention in probiotic_interventions:
        try:
            canonical = normalizer.get_canonical_name(intervention, 'intervention')
            groups[canonical].append(intervention)
        except:
            groups[intervention].append(intervention)

    print('GROUPING RESULTS:')
    success_found = False
    for canonical, terms in groups.items():
        if len(terms) > 1:
            print(f'  {canonical} [GROUPED] - {len(terms)} terms:')
            for term in terms:
                print(f'    - {term}')
            if 'probiotic' in canonical.lower():
                success_found = True
        else:
            print(f'  {canonical} - single term')

    if success_found:
        print()
        print('[SUCCESS CHECK MET] Probiotic variants are being grouped!')
    else:
        print()
        print('[INFO] Need more probiotic variant mappings for success check')

    # Test top interventions for IBS with grouping
    print()
    print('=== TESTING TOP INTERVENTIONS WITH GROUPING ===')

    cursor.execute("""
        SELECT DISTINCT
            i.intervention_name,
            i.correlation_type,
            i.correlation_strength,
            i.confidence_score,
            i.sample_size
        FROM interventions i
        WHERE i.health_condition LIKE '%irritable%'
    """)

    rows = cursor.fetchall()

    # Group by canonical names
    grouped_data = defaultdict(lambda: {
        'studies': [],
        'original_terms': set()
    })

    for row in rows:
        try:
            canonical = normalizer.get_canonical_name(row['intervention_name'], 'intervention')
            grouped_data[canonical]['studies'].append({
                'correlation_type': row['correlation_type'],
                'correlation_strength': row['correlation_strength'],
                'confidence_score': row['confidence_score'],
                'sample_size': row['sample_size']
            })
            grouped_data[canonical]['original_terms'].add(row['intervention_name'])
        except:
            grouped_data[row['intervention_name']]['studies'].append({
                'correlation_type': row['correlation_type'],
                'correlation_strength': row['correlation_strength'],
                'confidence_score': row['confidence_score'],
                'sample_size': row['sample_size']
            })
            grouped_data[row['intervention_name']]['original_terms'].add(row['intervention_name'])

    # Process results
    results = []
    for canonical_name, data in grouped_data.items():
        studies = data['studies']
        study_count = len(studies)
        positive_studies = sum(1 for s in studies if s['correlation_type'] == 'positive')

        result = {
            'intervention': canonical_name,
            'study_count': study_count,
            'positive_studies': positive_studies,
            'original_terms': sorted(list(data['original_terms'])),
            'is_grouped': len(data['original_terms']) > 1
        }
        results.append(result)

    # Sort and show top results
    results.sort(key=lambda x: (x['positive_studies'], x['study_count']), reverse=True)

    print(f'Top interventions for IBS (grouped by canonical names):')
    success_check_met = False

    for i, result in enumerate(results[:10], 1):
        grouped_indicator = " [GROUPED]" if result['is_grouped'] else ""
        print(f"{i:2d}. {result['intervention']}{grouped_indicator}")
        print(f"    Studies: {result['study_count']} (+{result['positive_studies']})")

        if result['is_grouped']:
            terms = ', '.join(result['original_terms'][:3])
            extra = '...' if len(result['original_terms']) > 3 else ''
            print(f"    Original terms: {terms}{extra}")

            # Check for probiotic success criteria
            probiotic_terms = [t for t in result['original_terms'] if 'probiotic' in t.lower()]
            if len(probiotic_terms) > 1:
                success_check_met = True
                print(f"    [SUCCESS CHECK MET] Probiotic variants grouped: {', '.join(probiotic_terms)}")
        print()

    if success_check_met:
        print('[SUCCESS] Success check achieved - probiotic variants grouped in top interventions!')
    else:
        print('[INFO] Success check criteria may need more data or mappings')

    conn.close()


if __name__ == "__main__":
    test_probiotic_grouping()