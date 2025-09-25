#!/usr/bin/env python3
"""
Quick analysis of mapping suggestions to understand quality and coverage
"""

import csv
import sys
from collections import defaultdict, Counter


def analyze_suggestions(csv_file):
    """Analyze the mapping suggestions CSV"""

    suggestions = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        suggestions = list(reader)

    print(f"=== ANALYSIS OF {csv_file} ===")
    print(f"Total suggestions: {len(suggestions)}")

    # Basic stats
    interventions = [s for s in suggestions if s['entity_type'] == 'intervention']
    conditions = [s for s in suggestions if s['entity_type'] == 'condition']

    print(f"Interventions: {len(interventions)}")
    print(f"Conditions: {len(conditions)}")

    # Method distribution
    method_counts = Counter(s['method'] for s in suggestions)
    print(f"\nMethod distribution:")
    for method, count in method_counts.most_common():
        print(f"  {method}: {count}")

    # Confidence distribution
    confidence_ranges = {'high (>=0.8)': 0, 'medium (0.6-0.8)': 0, 'low (0.1-0.6)': 0, 'none (0)': 0}
    for s in suggestions:
        conf = float(s['confidence'])
        if conf >= 0.8:
            confidence_ranges['high (>=0.8)'] += 1
        elif conf >= 0.6:
            confidence_ranges['medium (0.6-0.8)'] += 1
        elif conf > 0:
            confidence_ranges['low (0.1-0.6)'] += 1
        else:
            confidence_ranges['none (0)'] += 1

    print(f"\nConfidence distribution:")
    for range_name, count in confidence_ranges.items():
        print(f"  {range_name}: {count}")

    # High-value mappings (high frequency + high confidence)
    high_value = []
    for s in suggestions:
        if float(s['confidence']) >= 0.8 and int(s['frequency']) >= 5:
            high_value.append(s)

    high_value.sort(key=lambda x: int(x['frequency']), reverse=True)

    print(f"\nHigh-value mappings (confidence >=0.8, frequency >=5): {len(high_value)}")
    print("Top 10:")
    for i, s in enumerate(high_value[:10]):
        print(f"  {i+1}. {s['original_term']} ({s['entity_type']}) -> {s['suggested_canonical']} "
              f"(freq: {s['frequency']}, conf: {s['confidence']}, method: {s['method']})")

    # Interesting cases (non-existing mappings with good confidence)
    interesting = []
    for s in suggestions:
        if s['method'] != 'existing_mapping' and float(s['confidence']) >= 0.8:
            interesting.append(s)

    if interesting:
        interesting.sort(key=lambda x: float(x['confidence']), reverse=True)
        print(f"\nNew mapping opportunities (non-existing, confidence >=0.8): {len(interesting)}")
        print("Top 5:")
        for i, s in enumerate(interesting[:5]):
            print(f"  {i+1}. {s['original_term']} -> {s['suggested_canonical']} "
                  f"(conf: {s['confidence']}, method: {s['method']}, notes: {s['notes']})")

    # Potential issues to review
    potential_issues = []
    for s in suggestions:
        # Look for similarity matches that might be questionable
        if 'similarity' in s['method'] and float(s['confidence']) < 0.9:
            potential_issues.append(s)

    if potential_issues:
        print(f"\nPotential issues to review (similarity matches < 0.9): {len(potential_issues)}")
        for i, s in enumerate(potential_issues[:3]):
            print(f"  {i+1}. {s['original_term']} -> {s['suggested_canonical']} "
                  f"(conf: {s['confidence']}, notes: {s['notes']})")

    return suggestions


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_mapping_suggestions.py <csv_file>")
        print("Example: python analyze_mapping_suggestions.py mapping_suggestions_all_20250925_175029.csv")
        sys.exit(1)

    csv_file = sys.argv[1]
    try:
        suggestions = analyze_suggestions(csv_file)
        print(f"\n[SUCCESS] Analysis complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()