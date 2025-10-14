"""
Distance Threshold Results Analyzer

Analyzes threshold experiment results and generates:
1. Quantitative comparison tables (metrics)
2. Detailed cluster member lists for manual quality inspection

Usage:
    python analyze_threshold_results.py \\
        --exp-db experiment_results.db \\
        --output-dir threshold_analysis \\
        --include-member-lists
"""

import argparse
import json
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_threshold_experiments(db_path: str) -> Dict[str, List[Dict]]:
    """
    Load all threshold experiments from database.

    Returns:
        Dict mapping entity_type to list of experiment results
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query experiments tagged as threshold_experiment
    cursor.execute("""
        SELECT
            e.experiment_id,
            e.experiment_name,
            e.config_data,
            e.status,
            e.duration_seconds,
            er.entity_type,
            er.num_entities,
            er.num_clusters,
            er.num_natural_clusters,
            er.num_singleton_clusters,
            er.silhouette_score,
            er.davies_bouldin_score,
            er.cluster_size_min,
            er.cluster_size_max,
            er.cluster_size_mean,
            er.cluster_size_median
        FROM experiments e
        JOIN experiment_results er ON e.experiment_id = er.experiment_id
        WHERE e.status = 'completed'
          AND e.experiment_name LIKE 'threshold_%'
        ORDER BY er.entity_type, e.experiment_name
    """)

    rows = cursor.fetchall()
    conn.close()

    # Organize by entity type
    results = defaultdict(list)

    for row in rows:
        exp_id, exp_name, config_data, status, duration, entity_type, num_entities, \\
        num_clusters, num_natural, num_singletons, silhouette, davies_bouldin, \\
        size_min, size_max, size_mean, size_median = row

        # Extract threshold from experiment name
        # Format: threshold_0.5_interventions
        parts = exp_name.split('_')
        threshold = float(parts[1]) if len(parts) >= 2 else None

        results[entity_type].append({
            'experiment_id': exp_id,
            'experiment_name': exp_name,
            'threshold': threshold,
            'num_entities': num_entities,
            'num_clusters': num_clusters,
            'num_natural': num_natural,
            'num_singletons': num_singletons,
            'singleton_ratio': num_singletons / num_clusters if num_clusters > 0 else 0,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'cluster_size_min': size_min,
            'cluster_size_max': size_max,
            'cluster_size_mean': size_mean,
            'cluster_size_median': size_median,
            'duration_seconds': duration
        })

    # Sort each entity type by threshold
    for entity_type in results:
        results[entity_type].sort(key=lambda x: x['threshold'] or 0)

    return dict(results)


def load_cluster_members(db_path: str, experiment_id: int, entity_type: str) -> List[Dict]:
    """
    Load cluster members for an experiment.

    Returns:
        List of clusters with member lists
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            cluster_id,
            canonical_name,
            category,
            member_entities,
            num_members
        FROM cluster_details
        WHERE experiment_id = ? AND entity_type = ?
        ORDER BY num_members DESC, cluster_id
    """, (experiment_id, entity_type))

    rows = cursor.fetchall()
    conn.close()

    clusters = []
    for cluster_id, canonical_name, category, members_json, num_members in rows:
        members = json.loads(members_json) if members_json else []

        clusters.append({
            'cluster_id': cluster_id,
            'canonical_name': canonical_name,
            'category': category,
            'members': members,
            'num_members': num_members
        })

    return clusters


def generate_quantitative_report(
    results: Dict[str, List[Dict]],
    output_path: Path
):
    """Generate quantitative comparison table."""
    logger.info(f"Generating quantitative report: {output_path}")

    with open(output_path, 'w') as f:
        f.write("# Distance Threshold Experiment Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        f.write("="*80 + "\\n\\n")

        for entity_type, experiments in results.items():
            f.write(f"## {entity_type.title()} - Distance Threshold Comparison\\n\\n")

            # Table header
            f.write("| Threshold | Clusters | Singletons (%) | Silhouette | Davies-Bouldin | Avg Size | Min/Max Size | Duration |\\n")
            f.write("|-----------|----------|----------------|------------|----------------|----------|--------------|----------|\\n")

            # Table rows
            for exp in experiments:
                threshold = exp['threshold']
                clusters = exp['num_clusters']
                singletons = exp['num_singletons']
                singleton_pct = exp['singleton_ratio'] * 100
                silhouette = exp['silhouette_score'] or 0
                davies_bouldin = exp['davies_bouldin_score'] or 0
                avg_size = exp['cluster_size_mean'] or 0
                min_size = exp['cluster_size_min'] or 0
                max_size = exp['cluster_size_max'] or 0
                duration = exp['duration_seconds']

                # Mark baseline (0.5)
                marker = " ← Current" if threshold == 0.5 else ""

                f.write(f"| {threshold:.1f} | {clusters} | {singletons} ({singleton_pct:.1f}%) | "
                       f"{silhouette:.3f} | {davies_bouldin:.3f} | {avg_size:.1f} | "
                       f"{min_size}/{max_size} | {duration:.1f}s{marker} |\\n")

            f.write("\\n")

            # Recommendations
            f.write("### Recommendations\\n\\n")

            # Best silhouette
            best_silhouette = max(experiments, key=lambda x: x['silhouette_score'] or -999)
            f.write(f"**Best Silhouette Score**: Threshold {best_silhouette['threshold']:.1f} "
                   f"(score: {best_silhouette['silhouette_score']:.3f})\\n\\n")

            # Best Davies-Bouldin
            best_davies = min(experiments, key=lambda x: x['davies_bouldin_score'] or 999)
            f.write(f"**Best Davies-Bouldin Score**: Threshold {best_davies['threshold']:.1f} "
                   f"(score: {best_davies['davies_bouldin_score']:.3f})\\n\\n")

            # Lowest singleton ratio
            best_singleton = min(experiments, key=lambda x: x['singleton_ratio'])
            f.write(f"**Lowest Singleton Ratio**: Threshold {best_singleton['threshold']:.1f} "
                   f"({best_singleton['singleton_ratio']*100:.1f}%)\\n\\n")

            f.write("**Manual Review Required**: Check cluster member lists to verify quality.\\n\\n")
            f.write("-"*80 + "\\n\\n")


def generate_member_lists(
    db_path: str,
    results: Dict[str, List[Dict]],
    output_dir: Path
):
    """Generate detailed cluster member lists for manual inspection."""
    logger.info(f"Generating cluster member lists in: {output_dir}")

    for entity_type, experiments in results.items():
        for exp in experiments:
            threshold = exp['threshold']
            exp_id = exp['experiment_id']

            # Load cluster members
            clusters = load_cluster_members(db_path, exp_id, entity_type)

            if not clusters:
                logger.warning(f"No clusters found for {entity_type} threshold {threshold}")
                continue

            # Generate markdown report
            output_file = output_dir / f"{entity_type}_clusters_threshold_{threshold:.1f}.md"

            with open(output_file, 'w') as f:
                f.write(f"# {entity_type.title()} - Distance Threshold {threshold:.1f}\\n\\n")
                f.write(f"Experiment: {exp['experiment_name']}\\n")
                f.write(f"Total clusters: {exp['num_clusters']}\\n")
                f.write(f"Singleton clusters: {exp['num_singletons']} ({exp['singleton_ratio']*100:.1f}%)\\n")
                f.write(f"Silhouette score: {exp['silhouette_score']:.3f}\\n")
                f.write(f"Davies-Bouldin score: {exp['davies_bouldin_score']:.3f}\\n\\n")
                f.write("="*80 + "\\n\\n")

                # Show top 30 clusters (most members)
                for i, cluster in enumerate(clusters[:30], 1):
                    f.write(f"## Cluster {cluster['cluster_id']}: {cluster['canonical_name']}\\n\\n")
                    f.write(f"**Category**: {cluster['category']}\\n")
                    f.write(f"**Members**: {cluster['num_members']}\\n\\n")

                    # List all members
                    for member in cluster['members']:
                        f.write(f"- {member}\\n")

                    f.write("\\n")
                    f.write("**Quality Assessment**: [ ] GOOD / [ ] QUESTIONABLE / [ ] BAD\\n")
                    f.write("**Notes**: \\n\\n")
                    f.write("-"*80 + "\\n\\n")

                # Summary of remaining clusters
                if len(clusters) > 30:
                    f.write(f"## Remaining Clusters\\n\\n")
                    f.write(f"Showing top 30 of {len(clusters)} total clusters. ")
                    f.write(f"Remaining {len(clusters) - 30} clusters not shown (smaller clusters).\\n\\n")

            logger.info(f"Created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze distance threshold experiment results'
    )
    parser.add_argument(
        '--exp-db',
        required=True,
        help='Path to experiment database'
    )
    parser.add_argument(
        '--output-dir',
        default='threshold_analysis',
        help='Output directory for reports (default: threshold_analysis)'
    )
    parser.add_argument(
        '--include-member-lists',
        action='store_true',
        help='Generate detailed cluster member lists (default: False)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment results
    logger.info(f"Loading threshold experiments from: {args.exp_db}")
    results = load_threshold_experiments(args.exp_db)

    if not results:
        logger.error("No threshold experiments found in database!")
        return 1

    logger.info(f"Found experiments for: {list(results.keys())}")

    # Generate quantitative report
    quant_report = output_dir / "quantitative_comparison.md"
    generate_quantitative_report(results, quant_report)

    # Generate member lists if requested
    if args.include_member_lists:
        generate_member_lists(args.exp_db, results, output_dir)
    else:
        logger.info("Skipping member lists (use --include-member-lists to generate)")

    logger.info("")
    logger.info("="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Reports saved to: {output_dir.absolute()}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review quantitative_comparison.md for metric comparisons")
    logger.info("  2. Review cluster member lists for quality assessment")
    logger.info("  3. Mark clusters as GOOD/QUESTIONABLE/BAD")
    logger.info("  4. Select optimal threshold per entity type")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
