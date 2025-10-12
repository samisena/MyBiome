"""
Compare Thresholds - Similarity Only (No LLM)

Shows what candidates each threshold picks up based purely on embedding similarity.
No LLM calls needed - instant results!
"""

import sys
import sqlite3
from pathlib import Path
from collections import defaultdict, Counter

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from back_end.src.semantic_normalization.embedding_engine import EmbeddingEngine
from back_end.src.data.config import config

# ==============================================================================
# CONFIGURATION
# ==============================================================================

THRESHOLDS = [0.70, 0.65, 0.60]
TOP_K = 10  # Look at top 10 candidates
SAMPLE_SIZE = 5  # Test 5 interventions
CACHE_PATH = "c:/Users/samis/Desktop/MyBiome/back_end/data/cache/embeddings.pkl"


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_sample_interventions(db_path: str, sample_size: int = 5):
    """Load stratified sample of interventions."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT intervention_name, intervention_category
        FROM interventions
        WHERE intervention_category IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (sample_size,))

    samples = []
    for row in cursor.fetchall():
        samples.append({
            'name': row['intervention_name'],
            'category': row['intervention_category']
        })

    conn.close()
    return samples


def load_all_intervention_names(db_path: str):
    """Load all unique intervention names."""
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


# ==============================================================================
# THRESHOLD COMPARISON
# ==============================================================================

def compare_thresholds():
    """Compare what each threshold picks up."""

    print("="*80)
    print("THRESHOLD COMPARISON - SIMILARITY ONLY (NO LLM CALLS)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Sample size: {SAMPLE_SIZE} interventions")
    print(f"  Top-K candidates: {TOP_K}")
    print(f"  Database: {config.db_path}")

    # Initialize embedding engine
    print(f"\nInitializing embedding engine...")
    embedding_engine = EmbeddingEngine(cache_path=CACHE_PATH)

    # Load data
    print(f"\nLoading data...")
    sample_interventions = load_sample_interventions(str(config.db_path), SAMPLE_SIZE)
    all_intervention_names = load_all_intervention_names(str(config.db_path))

    print(f"  Loaded {len(sample_interventions)} test interventions")
    print(f"  Loaded {len(all_intervention_names)} total interventions")

    # Statistics per threshold
    threshold_stats = {t: {'total_candidates': 0, 'candidates_per_intervention': []} for t in THRESHOLDS}

    # Test each intervention
    for i, intervention in enumerate(sample_interventions, 1):
        name = intervention['name']
        category = intervention['category']

        print(f"\n{'='*80}")
        print(f"INTERVENTION {i}/{len(sample_interventions)}: {name} ({category})")
        print(f"{'='*80}")

        # Get candidates at different thresholds
        candidates_by_threshold = {}

        for threshold in THRESHOLDS:
            similar = embedding_engine.find_similar(
                query_text=name,
                candidate_texts=all_intervention_names,
                top_k=TOP_K,
                min_similarity=threshold
            )
            candidates_by_threshold[threshold] = similar

            # Update stats
            threshold_stats[threshold]['total_candidates'] += len(similar)
            threshold_stats[threshold]['candidates_per_intervention'].append(len(similar))

        # Show what each threshold picked up
        for threshold in THRESHOLDS:
            candidates = candidates_by_threshold[threshold]
            print(f"\nThreshold {threshold} ({len(candidates)} candidates):")

            if len(candidates) == 0:
                print(f"  (No candidates above threshold)")
            else:
                for candidate_name, similarity in candidates[:5]:  # Show top 5
                    print(f"  {similarity:.3f} - {candidate_name}")

                if len(candidates) > 5:
                    print(f"  ... and {len(candidates) - 5} more")

        # Show differences between thresholds
        print(f"\n{'-'*80}")
        print("THRESHOLD COMPARISON:")

        candidates_070 = set(c[0] for c in candidates_by_threshold[0.70])
        candidates_065 = set(c[0] for c in candidates_by_threshold[0.65])
        candidates_060 = set(c[0] for c in candidates_by_threshold[0.60])

        # What does 0.65 add over 0.70?
        new_at_065 = candidates_065 - candidates_070
        if new_at_065:
            print(f"\nNEW at 0.65 (not in 0.70): {len(new_at_065)} candidates")
            for candidate_name in list(new_at_065)[:3]:
                sim = [s for c, s in candidates_by_threshold[0.65] if c == candidate_name][0]
                print(f"  {sim:.3f} - {candidate_name}")
            if len(new_at_065) > 3:
                print(f"  ... and {len(new_at_065) - 3} more")
        else:
            print(f"\nNEW at 0.65: None (same as 0.70)")

        # What does 0.60 add over 0.65?
        new_at_060 = candidates_060 - candidates_065
        if new_at_060:
            print(f"\nNEW at 0.60 (not in 0.65): {len(new_at_060)} candidates")
            for candidate_name in list(new_at_060)[:3]:
                sim = [s for c, s in candidates_by_threshold[0.60] if c == candidate_name][0]
                print(f"  {sim:.3f} - {candidate_name}")
            if len(new_at_060) > 3:
                print(f"  ... and {len(new_at_060) - 3} more")
        else:
            print(f"\nNEW at 0.60: None (same as 0.65)")

    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")

    for threshold in THRESHOLDS:
        stats = threshold_stats[threshold]
        total = stats['total_candidates']
        avg = sum(stats['candidates_per_intervention']) / len(stats['candidates_per_intervention'])
        min_c = min(stats['candidates_per_intervention'])
        max_c = max(stats['candidates_per_intervention'])

        print(f"\nThreshold {threshold}:")
        print(f"  Total candidates: {total}")
        print(f"  Average per intervention: {avg:.1f}")
        print(f"  Min/Max: {min_c}/{max_c}")

    # Comparison
    print(f"\n{'-'*80}")
    print("THRESHOLD IMPACT:")

    total_070 = threshold_stats[0.70]['total_candidates']
    total_065 = threshold_stats[0.65]['total_candidates']
    total_060 = threshold_stats[0.60]['total_candidates']

    if total_070 > 0:
        increase_065 = ((total_065 - total_070) / total_070 * 100)
        print(f"\n0.65 vs 0.70: +{increase_065:.1f}% candidates ({total_070} -> {total_065})")

    if total_065 > 0:
        increase_060 = ((total_060 - total_065) / total_065 * 100)
        print(f"0.60 vs 0.65: +{increase_060:.1f}% candidates ({total_065} -> {total_060})")

    if total_070 > 0:
        total_increase = ((total_060 - total_070) / total_070 * 100)
        print(f"0.60 vs 0.70: +{total_increase:.1f}% candidates ({total_070} -> {total_060})")

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    if total_070 > 0:
        increase_065 = ((total_065 - total_070) / total_070 * 100)

        if increase_065 < 20:
            print(f"\n✓ SAFE to lower to 0.65 (only +{increase_065:.1f}% candidates)")
            print(f"  Impact: Minimal increase in LLM calls, may catch useful variants")

        elif increase_065 < 40:
            print(f"\n⚠ MODERATE impact at 0.65 (+{increase_065:.1f}% candidates)")
            print(f"  Impact: Noticeable increase in LLM calls, validate quality")

        else:
            print(f"\n✗ HIGH impact at 0.65 (+{increase_065:.1f}% candidates)")
            print(f"  Impact: Significant increase in LLM calls, risk of false positives")

        if total_065 > 0:
            increase_060 = ((total_060 - total_065) / total_065 * 100)

            if increase_060 < 20:
                print(f"\n✓ 0.60 is close to 0.65 (+{increase_060:.1f}%)")
            elif increase_060 < 40:
                print(f"\n⚠ 0.60 adds moderately more (+{increase_060:.1f}%)")
            else:
                print(f"\n✗ 0.60 adds significantly more (+{increase_060:.1f}%)")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    try:
        compare_thresholds()
        print("\nTest completed successfully")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user (Ctrl+C)")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
