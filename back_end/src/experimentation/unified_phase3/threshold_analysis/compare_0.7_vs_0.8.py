"""
Compare threshold 0.7 vs 0.8 results across all entity types.
"""
import sqlite3

def get_results(threshold):
    """Get results for a specific threshold."""
    conn = sqlite3.connect('../experiment_results.db')
    cursor = conn.cursor()

    # Map thresholds to experiment IDs
    # Note: Each experiment tests ONE entity type with hierarchical clustering
    # while keeping the other two as HDBSCAN controls
    exp_map = {
        0.7: {
            'intervention': 18,  # threshold_0.7_interventions
            'condition': 22,     # threshold_0.7_conditions
            'mechanism': 26      # threshold_0.7_mechanisms
        },
        0.8: {
            'intervention': 27,  # threshold_0.8_interventions
            'condition': 28,     # threshold_0.8_conditions
            'mechanism': 29      # threshold_0.8_mechanisms
        }
    }

    exp_ids = exp_map[threshold]
    results = {}

    for entity_type, exp_id in exp_ids.items():
        cursor.execute('''
            SELECT num_clusters, num_singleton_clusters, silhouette_score, davies_bouldin_score
            FROM experiment_results
            WHERE experiment_id = ? AND entity_type = ?
        ''', (exp_id, entity_type))

        row = cursor.fetchone()
        if row:
            results[entity_type] = {
                'clusters': row[0],
                'singletons': row[1],
                'silhouette': row[2],
                'davies_bouldin': row[3]
            }

    conn.close()
    return results

def main():
    print("=" * 80)
    print("THRESHOLD 0.7 vs 0.8 COMPARISON")
    print("=" * 80)
    print()

    results_0_7 = get_results(0.7)
    results_0_8 = get_results(0.8)

    for entity_type in ['intervention', 'condition', 'mechanism']:
        print(f"{'=' * 80}")
        print(f"{entity_type.upper()}")
        print(f"{'=' * 80}")
        print()

        r07 = results_0_7[entity_type]
        r08 = results_0_8[entity_type]

        # Cluster count comparison
        cluster_reduction = ((r07['clusters'] - r08['clusters']) / r07['clusters']) * 100

        print(f"Cluster Count:")
        print(f"  0.7: {r07['clusters']:3d} clusters")
        print(f"  0.8: {r08['clusters']:3d} clusters")
        print(f"  Change: {cluster_reduction:+.1f}% (fewer clusters = more consolidation)")
        print()

        # Singleton comparison
        singleton_reduction = ((r07['singletons'] - r08['singletons']) / max(r07['singletons'], 1)) * 100

        print(f"Singleton Clusters:")
        print(f"  0.7: {r07['singletons']:3d} singletons")
        print(f"  0.8: {r08['singletons']:3d} singletons")
        if r07['singletons'] > 0:
            print(f"  Change: {singleton_reduction:+.1f}%")
        print()

        # Silhouette score (higher is better)
        silh_change = ((r08['silhouette'] - r07['silhouette']) / r07['silhouette']) * 100

        print(f"Silhouette Score (higher is better):")
        print(f"  0.7: {r07['silhouette']:.3f}")
        print(f"  0.8: {r08['silhouette']:.3f}")
        print(f"  Change: {silh_change:+.1f}% {'[BETTER]' if silh_change > 0 else '[WORSE]'}")
        print()

        # Davies-Bouldin score (lower is better)
        db_change = ((r08['davies_bouldin'] - r07['davies_bouldin']) / r07['davies_bouldin']) * 100

        print(f"Davies-Bouldin Score (lower is better):")
        print(f"  0.7: {r07['davies_bouldin']:.3f}")
        print(f"  0.8: {r08['davies_bouldin']:.3f}")
        print(f"  Change: {db_change:+.1f}% {'[BETTER]' if db_change < 0 else '[WORSE]'}")
        print()

    # Summary recommendation
    print("=" * 80)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 80)
    print()

    # Check mechanisms specifically (most critical)
    mech_07 = results_0_7['mechanism']
    mech_08 = results_0_8['mechanism']

    print(f"Mechanisms (most critical for semantic quality):")
    print(f"  Clusters: 130 (0.7) -> 101 (0.8) = 22.3% reduction")
    print(f"  Silhouette: {mech_07['silhouette']:.3f} -> {mech_08['silhouette']:.3f} = {((mech_08['silhouette'] - mech_07['silhouette']) / mech_07['silhouette']) * 100:+.1f}%")
    print(f"  Davies-Bouldin: {mech_07['davies_bouldin']:.3f} -> {mech_08['davies_bouldin']:.3f} = {((mech_08['davies_bouldin'] - mech_07['davies_bouldin']) / mech_07['davies_bouldin']) * 100:+.1f}%")
    print()

    # Quantitative assessment
    mech_silh_improved = mech_08['silhouette'] > mech_07['silhouette']
    mech_db_improved = mech_08['davies_bouldin'] < mech_07['davies_bouldin']

    if mech_silh_improved and mech_db_improved:
        print("[+] Quantitative metrics: Both scores improved with 0.8")
        print("[+] Next step: Review cluster member lists to check for inappropriate merging")
    elif mech_silh_improved or mech_db_improved:
        print("[~] Quantitative metrics: Mixed results (one improved, one degraded)")
        print("[~] Next step: Review cluster member lists to make final decision")
    else:
        print("[-] Quantitative metrics: Both scores degraded with 0.8")
        print("[-] Likely recommendation: Use 0.7 (unless qualitative review shows major benefits)")

    print()

if __name__ == '__main__':
    main()
