"""
Quick Threshold Comparison Test for Phase 3 Semantic Normalization

Compares three similarity thresholds (0.70, 0.65, 0.60) to analyze:
1. Number of candidate pairs generated
2. Relationship type distribution
3. Precision/recall against ground truth (if available)
4. Performance metrics (LLM calls, processing time)
5. Category-specific performance

Usage:
    python -m back_end.scripts.test_threshold_comparison
"""

import sys
import os
import sqlite3
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from back_end.src.phase_3_semantic_normalization.phase_3_embedding_engine import EmbeddingEngine
from back_end.src.phase_3_semantic_normalization.phase_3_llm_classifier import LLMClassifier
from back_end.src.data.config import config


# ==============================================================================
# CONFIGURATION
# ==============================================================================

THRESHOLDS_TO_TEST = [0.70, 0.65, 0.60]
TOP_K_SIMILAR = 5
SAMPLE_SIZE = 3  # Test on 3 interventions for quick test (72s per LLM call = ~3-5 min per intervention)
CACHE_PATH = "c:/Users/samis/Desktop/MyBiome/back_end/data/cache/embeddings.pkl"
OUTPUT_DIR = "c:/Users/samis/Desktop/MyBiome/back_end/data/threshold_analysis"


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_sample_interventions(db_path: str, sample_size: int = 50) -> List[Dict]:
    """
    Load stratified sample of interventions across all categories.

    Args:
        db_path: Path to SQLite database
        sample_size: Number of interventions to sample

    Returns:
        List of intervention dicts with name and category
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get category distribution
    cursor.execute("""
        SELECT intervention_category, COUNT(*) as count
        FROM interventions
        WHERE intervention_category IS NOT NULL
        GROUP BY intervention_category
        ORDER BY count DESC
    """)
    category_counts = dict(cursor.fetchall())

    # Calculate stratified sample sizes
    total_interventions = sum(category_counts.values())
    samples = []

    for category, count in category_counts.items():
        # Proportional sampling
        category_sample_size = max(1, int((count / total_interventions) * sample_size))

        cursor.execute("""
            SELECT DISTINCT intervention_name, intervention_category
            FROM interventions
            WHERE intervention_category = ?
            ORDER BY RANDOM()
            LIMIT ?
        """, (category, category_sample_size))

        for row in cursor.fetchall():
            samples.append({
                'name': row['intervention_name'],
                'category': row['intervention_category']
            })

    conn.close()

    # If we're under sample_size, add more random samples
    if len(samples) < sample_size:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        existing_names = {s['name'] for s in samples}
        needed = sample_size - len(samples)

        cursor.execute("""
            SELECT DISTINCT intervention_name, intervention_category
            FROM interventions
            WHERE intervention_category IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """, (needed * 2,))  # Get extras in case of duplicates

        for row in cursor.fetchall():
            if row['intervention_name'] not in existing_names:
                samples.append({
                    'name': row['intervention_name'],
                    'category': row['intervention_category']
                })
                existing_names.add(row['intervention_name'])
                if len(samples) >= sample_size:
                    break

        conn.close()

    return samples[:sample_size]


def load_all_intervention_names(db_path: str) -> List[str]:
    """Load all unique intervention names from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT intervention_name
        FROM interventions
        ORDER BY intervention_name
    """)

    names = [row[0] for row in cursor.fetchall()]
    conn.close()

    return names


def load_ground_truth_labels() -> Optional[Dict[Tuple[str, str], str]]:
    """
    Load ground truth labeled pairs if available.

    Returns:
        Dict mapping (entity1, entity2) -> relationship_type, or None if not found
    """
    gt_path = Path("back_end/src/semantic_normalization/ground_truth/data/labeling_session.json")

    if not gt_path.exists():
        return None

    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            session = json.load(f)

        labels = {}
        for pair in session.get('labeled_pairs', []):
            entity1 = pair['intervention_1']
            entity2 = pair['intervention_2']
            rel_type = pair['relationship_type']

            # Store both orderings
            labels[(entity1, entity2)] = rel_type
            labels[(entity2, entity1)] = rel_type

        return labels
    except Exception as e:
        print(f"Warning: Could not load ground truth labels: {e}")
        return None


# ==============================================================================
# THRESHOLD TESTING
# ==============================================================================

def test_single_threshold(
    threshold: float,
    sample_interventions: List[Dict],
    all_intervention_names: List[str],
    embedding_engine: EmbeddingEngine,
    llm_classifier: LLMClassifier,
    ground_truth_labels: Optional[Dict] = None
) -> Dict:
    """
    Test a single similarity threshold and collect metrics.

    Returns:
        Dict with test results and metrics
    """
    print(f"\n{'='*80}")
    print(f"Testing threshold: {threshold}")
    print(f"{'='*80}")

    results = {
        'threshold': threshold,
        'sample_size': len(sample_interventions),
        'total_candidates_found': 0,
        'total_llm_calls': 0,
        'relationship_types': Counter(),
        'category_performance': defaultdict(lambda: {
            'candidates': 0,
            'relationships': Counter()
        }),
        'processing_time_seconds': 0,
        'ground_truth_accuracy': None,
        'examples': []
    }

    start_time = time.time()

    for i, intervention in enumerate(sample_interventions, 1):
        name = intervention['name']
        category = intervention['category']

        print(f"\n[{i}/{len(sample_interventions)}] Processing: {name} ({category})", flush=True)

        # Find similar interventions
        similar = embedding_engine.find_similar(
            query_text=name,
            candidate_texts=all_intervention_names,
            top_k=TOP_K_SIMILAR,
            min_similarity=threshold
        )

        num_candidates = len(similar)
        results['total_candidates_found'] += num_candidates
        results['category_performance'][category]['candidates'] += num_candidates

        print(f"  Found {num_candidates} candidates above threshold {threshold}", flush=True)

        # Classify relationships with LLM
        for similar_name, similarity in similar:
            # LLM classification
            rel_result = llm_classifier.classify_relationship(
                name,
                similar_name,
                similarity
            )

            rel_type = rel_result['relationship_type']
            results['total_llm_calls'] += 1
            results['relationship_types'][rel_type] += 1
            results['category_performance'][category]['relationships'][rel_type] += 1

            print(f"    - {similar_name} (sim={similarity:.3f}): {rel_type}", flush=True)

            # Store first 3 examples for each threshold
            if len(results['examples']) < 3:
                results['examples'].append({
                    'query': name,
                    'match': similar_name,
                    'similarity': similarity,
                    'relationship': rel_type,
                    'category': category
                })

    results['processing_time_seconds'] = time.time() - start_time

    # Calculate ground truth accuracy if available
    if ground_truth_labels:
        results['ground_truth_accuracy'] = calculate_accuracy_vs_ground_truth(
            results,
            ground_truth_labels
        )

    return results


def calculate_accuracy_vs_ground_truth(
    results: Dict,
    ground_truth_labels: Dict[Tuple[str, str], str]
) -> Dict:
    """
    Calculate accuracy metrics against ground truth labeled pairs.

    Returns:
        Dict with accuracy metrics
    """
    # This is a simplified version - would need to track individual predictions
    # For full implementation, modify test_single_threshold to store predictions

    print("\n  Checking against ground truth labels...")

    return {
        'note': 'Full ground truth comparison requires storing all predictions',
        'ground_truth_pairs_available': len(ground_truth_labels) // 2
    }


# ==============================================================================
# ANALYSIS & REPORTING
# ==============================================================================

def compare_results(all_results: List[Dict]) -> Dict:
    """
    Compare results across all thresholds.

    Returns:
        Comparison analysis dict
    """
    comparison = {
        'summary': {},
        'relationship_distribution': {},
        'category_analysis': {},
        'performance_metrics': {},
        'recommendations': []
    }

    # Summary statistics
    for result in all_results:
        threshold = result['threshold']
        comparison['summary'][threshold] = {
            'total_candidates': result['total_candidates_found'],
            'avg_candidates_per_intervention': result['total_candidates_found'] / result['sample_size'],
            'total_llm_calls': result['total_llm_calls'],
            'processing_time': result['processing_time_seconds'],
            'llm_calls_per_second': result['total_llm_calls'] / result['processing_time_seconds']
        }

        # Relationship type distribution
        comparison['relationship_distribution'][threshold] = dict(result['relationship_types'])

    # Category-specific analysis
    categories = set()
    for result in all_results:
        categories.update(result['category_performance'].keys())

    for category in categories:
        comparison['category_analysis'][category] = {}
        for result in all_results:
            threshold = result['threshold']
            cat_data = result['category_performance'].get(category, {})
            comparison['category_analysis'][category][threshold] = {
                'candidates': cat_data.get('candidates', 0),
                'relationships': dict(cat_data.get('relationships', Counter()))
            }

    # Performance metrics
    thresholds = [r['threshold'] for r in all_results]
    candidates = [r['total_candidates_found'] for r in all_results]
    llm_calls = [r['total_llm_calls'] for r in all_results]
    times = [r['processing_time_seconds'] for r in all_results]

    comparison['performance_metrics'] = {
        'candidate_increase_0.65_vs_0.70': ((candidates[1] - candidates[0]) / candidates[0] * 100) if len(candidates) > 1 else None,
        'candidate_increase_0.60_vs_0.70': ((candidates[2] - candidates[0]) / candidates[0] * 100) if len(candidates) > 2 else None,
        'llm_call_increase_0.65_vs_0.70': ((llm_calls[1] - llm_calls[0]) / llm_calls[0] * 100) if len(llm_calls) > 1 else None,
        'llm_call_increase_0.60_vs_0.70': ((llm_calls[2] - llm_calls[0]) / llm_calls[0] * 100) if len(llm_calls) > 2 else None,
    }

    # Recommendations
    comparison['recommendations'] = generate_recommendations(all_results, comparison)

    return comparison


def generate_recommendations(all_results: List[Dict], comparison: Dict) -> List[str]:
    """Generate threshold recommendations based on analysis."""
    recommendations = []

    # Get metrics for 0.70 and 0.65
    if len(all_results) >= 2:
        candidates_070 = all_results[0]['total_candidates_found']
        candidates_065 = all_results[1]['total_candidates_found']

        increase_pct = ((candidates_065 - candidates_070) / candidates_070 * 100)

        if increase_pct > 50:
            recommendations.append(
                f"CAUTION: Lowering to 0.65 increases candidates by {increase_pct:.1f}%. "
                "This may generate too many false positives and increase LLM costs."
            )
        elif increase_pct > 25:
            recommendations.append(
                f"MODERATE: 0.65 increases candidates by {increase_pct:.1f}%. "
                "Consider testing with ground truth to validate precision."
            )
        else:
            recommendations.append(
                f"LOW IMPACT: 0.65 only increases candidates by {increase_pct:.1f}%. "
                "May be safe to lower threshold for better recall."
            )

    # Category-specific recommendations
    for category, data in comparison['category_analysis'].items():
        if len(data) >= 2:
            candidates_070 = data.get(0.70, {}).get('candidates', 0)
            candidates_065 = data.get(0.65, {}).get('candidates', 0)

            if candidates_070 > 0:
                cat_increase = ((candidates_065 - candidates_070) / candidates_070 * 100)

                if cat_increase > 100:
                    recommendations.append(
                        f"CATEGORY '{category}': High sensitivity to threshold change (+{cat_increase:.1f}%). "
                        "Consider adaptive threshold for this category."
                    )

    return recommendations


def save_results(all_results: List[Dict], comparison: Dict, output_dir: str):
    """Save test results to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual results
    results_file = os.path.join(output_dir, f"threshold_test_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'configuration': {
                'thresholds': THRESHOLDS_TO_TEST,
                'sample_size': SAMPLE_SIZE,
                'top_k': TOP_K_SIMILAR
            },
            'results': all_results,
            'comparison': comparison
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Save summary report
    report_file = os.path.join(output_dir, f"threshold_comparison_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("THRESHOLD COMPARISON TEST REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Sample Size: {SAMPLE_SIZE} interventions\n")
        f.write(f"Thresholds Tested: {', '.join(map(str, THRESHOLDS_TO_TEST))}\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-"*80 + "\n")
        for threshold, stats in comparison['summary'].items():
            f.write(f"\nThreshold {threshold}:\n")
            f.write(f"  Total Candidates: {stats['total_candidates']}\n")
            f.write(f"  Avg Candidates/Intervention: {stats['avg_candidates_per_intervention']:.2f}\n")
            f.write(f"  Total LLM Calls: {stats['total_llm_calls']}\n")
            f.write(f"  Processing Time: {stats['processing_time']:.1f}s\n")

        f.write("\n\nRELATIONSHIP TYPE DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        for threshold, rel_types in comparison['relationship_distribution'].items():
            f.write(f"\nThreshold {threshold}:\n")
            for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {rel_type}: {count}\n")

        f.write("\n\nRECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        for rec in comparison['recommendations']:
            f.write(f"- {rec}\n")

    print(f"Report saved to: {report_file}")


def print_results_summary(all_results: List[Dict], comparison: Dict):
    """Print formatted results summary to console."""
    print("\n" + "="*80)
    print("THRESHOLD COMPARISON TEST - SUMMARY")
    print("="*80)

    # Summary table
    print("\nCANDIDATE GENERATION:")
    print(f"{'Threshold':<12} {'Candidates':<12} {'Avg/Intervention':<18} {'LLM Calls':<12}")
    print("-"*80)
    for result in all_results:
        threshold = result['threshold']
        stats = comparison['summary'][threshold]
        print(f"{threshold:<12.2f} {stats['total_candidates']:<12} "
              f"{stats['avg_candidates_per_intervention']:<18.2f} {stats['total_llm_calls']:<12}")

    # Relationship distribution
    print("\nRELATIONSHIP TYPE DISTRIBUTION:")
    for result in all_results:
        threshold = result['threshold']
        print(f"\n  Threshold {threshold}:")
        for rel_type, count in result['relationship_types'].most_common():
            percentage = (count / result['total_llm_calls'] * 100) if result['total_llm_calls'] > 0 else 0
            print(f"    {rel_type:<20} {count:>4} ({percentage:>5.1f}%)")

    # Performance impact
    print("\nPERFORMANCE IMPACT:")
    metrics = comparison['performance_metrics']
    if metrics.get('candidate_increase_0.65_vs_0.70') is not None:
        print(f"  0.65 vs 0.70: +{metrics['candidate_increase_0.65_vs_0.70']:.1f}% candidates, "
              f"+{metrics['llm_call_increase_0.65_vs_0.70']:.1f}% LLM calls")
    if metrics.get('candidate_increase_0.60_vs_0.70') is not None:
        print(f"  0.60 vs 0.70: +{metrics['candidate_increase_0.60_vs_0.70']:.1f}% candidates, "
              f"+{metrics['llm_call_increase_0.60_vs_0.70']:.1f}% LLM calls")

    # Top recommendations
    print("\nTOP RECOMMENDATIONS:")
    for i, rec in enumerate(comparison['recommendations'][:3], 1):
        print(f"  {i}. {rec}")

    print("\n" + "="*80)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run threshold comparison test."""
    print("="*80)
    print("THRESHOLD COMPARISON TEST FOR PHASE 3 SEMANTIC NORMALIZATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Thresholds: {THRESHOLDS_TO_TEST}")
    print(f"  Sample size: {SAMPLE_SIZE} interventions")
    print(f"  Top-K similar: {TOP_K_SIMILAR}")
    print(f"  Database: {config.db_path}")

    # Initialize components
    print("\nInitializing embedding engine and LLM classifier...")
    embedding_engine = EmbeddingEngine(cache_path=CACHE_PATH)
    llm_classifier = LLMClassifier()

    # Load data
    print(f"\nLoading sample interventions (stratified by category)...")
    sample_interventions = load_sample_interventions(str(config.db_path), SAMPLE_SIZE)
    print(f"  Loaded {len(sample_interventions)} interventions")

    # Count by category
    category_counts = Counter(i['category'] for i in sample_interventions)
    print(f"  Category distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {category}: {count}")

    print(f"\nLoading all intervention names for similarity search...")
    all_intervention_names = load_all_intervention_names(str(config.db_path))
    print(f"  Loaded {len(all_intervention_names)} unique interventions")

    print(f"\nLoading ground truth labels (if available)...")
    ground_truth_labels = load_ground_truth_labels()
    if ground_truth_labels:
        print(f"  Loaded {len(ground_truth_labels) // 2} labeled pairs")
    else:
        print(f"  No ground truth labels found")

    # Test each threshold
    all_results = []
    for threshold in THRESHOLDS_TO_TEST:
        result = test_single_threshold(
            threshold=threshold,
            sample_interventions=sample_interventions,
            all_intervention_names=all_intervention_names,
            embedding_engine=embedding_engine,
            llm_classifier=llm_classifier,
            ground_truth_labels=ground_truth_labels
        )
        all_results.append(result)

    # Compare results
    print("\n" + "="*80)
    print("ANALYZING RESULTS...")
    print("="*80)
    comparison = compare_results(all_results)

    # Print summary
    print_results_summary(all_results, comparison)

    # Save results
    print("\nSaving results to disk...")
    save_results(all_results, comparison, OUTPUT_DIR)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()