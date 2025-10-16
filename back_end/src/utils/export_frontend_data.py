"""
Export intervention research data to JSON for frontend display.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def get_database_path() -> Path:
    """Get the path to the intervention research database."""
    return Path(__file__).parent.parent.parent / "data" / "processed" / "intervention_research.db"

def get_frontend_data_path() -> Path:
    """Get the path to the frontend data directory."""
    return Path(__file__).parent.parent.parent.parent / "frontend" / "data"

def get_entity_categories(cursor, entity_type: str, entity_id: Any) -> Dict[str, List[str]]:
    """
    Get all categories for an entity organized by type.

    Args:
        cursor: Database cursor
        entity_type: 'intervention' or 'condition'
        entity_id: intervention.id or condition name

    Returns:
        Dict with category types as keys, lists of category names as values
    """
    try:
        if entity_type == 'intervention':
            cursor.execute("""
                SELECT category_type, category_name, confidence
                FROM intervention_category_mapping
                WHERE intervention_id = ?
                ORDER BY category_type, category_name
            """, (entity_id,))
        elif entity_type == 'condition':
            cursor.execute("""
                SELECT category_type, category_name, confidence
                FROM condition_category_mapping
                WHERE condition_name = ?
                ORDER BY category_type, category_name
            """, (entity_id,))
        else:
            return {}

        rows = cursor.fetchall()

        # Organize by category type
        categories_by_type = {}
        for row in rows:
            cat_type = row['category_type']
            cat_name = row['category_name']

            if cat_type not in categories_by_type:
                categories_by_type[cat_type] = []
            categories_by_type[cat_type].append(cat_name)

        return categories_by_type
    except sqlite3.OperationalError:
        # Table doesn't exist yet - return empty dict
        return {}

def export_interventions_data() -> Dict[str, Any]:
    """
    Export interventions data with papers and canonical entity information.

    Returns:
        Dictionary with interventions data and metadata
    """
    db_path = get_database_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Query to get interventions with paper details and hierarchical semantic data (Phase 3.5) + Bayesian scores (Phase 4b)
    query = """
    SELECT
        i.id,
        i.intervention_name,
        i.intervention_category,
        i.intervention_details,
        i.health_condition,
        i.condition_category,
        i.mechanism,
        i.correlation_type,
        i.correlation_strength,
        i.extraction_confidence,
        i.study_confidence,
        i.sample_size,
        i.study_duration,
        i.study_type,
        i.population_details,
        i.delivery_method,
        i.severity,
        i.adverse_effects,
        i.cost_category,
        i.supporting_quote,
        i.study_focus,
        i.measured_metrics,
        i.findings,
        i.study_location,
        i.publisher,
        p.title as paper_title,
        p.journal as paper_journal,
        p.publication_date,
        p.pmid as pubmed_id,
        p.doi,
        sh_i.layer_1_canonical as intervention_canonical_name,
        sh_c.layer_1_canonical as condition_canonical_name,
        sh_i.layer_0_category as intervention_l0_category,
        sh_i.layer_1_canonical as intervention_l1_canonical,
        sh_i.layer_2_variant as intervention_l2_variant,
        sh_i.layer_3_detail as intervention_l3_detail,
        sh_c.layer_0_category as condition_l0_category,
        sh_c.layer_1_canonical as condition_l1_canonical,
        sh_c.layer_2_variant as condition_l2_variant,
        sh_c.layer_3_detail as condition_l3_detail,
        bs.posterior_mean as bayesian_score,
        bs.confidence_adjusted_score as bayesian_conservative_score,
        bs.positive_evidence_count,
        bs.negative_evidence_count,
        bs.neutral_evidence_count,
        bs.total_studies as bayesian_total_studies,
        bs.bayes_factor,
        i.extraction_model,
        i.extraction_timestamp
    FROM interventions i
    LEFT JOIN papers p ON i.paper_id = p.pmid
    LEFT JOIN semantic_hierarchy sh_i ON i.intervention_name = sh_i.entity_name AND sh_i.entity_type = 'intervention'
    LEFT JOIN semantic_hierarchy sh_c ON i.health_condition = sh_c.entity_name AND sh_c.entity_type = 'condition'
    LEFT JOIN bayesian_scores bs ON sh_i.layer_1_canonical = bs.intervention_name AND sh_c.layer_1_canonical = bs.condition_name
    ORDER BY bs.posterior_mean DESC NULLS LAST, i.extraction_confidence DESC
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    # Helper function to get mechanism canonical names for an intervention
    def get_mechanism_canonical_names(intervention_id):
        """Get all mechanism canonical names for an intervention."""
        cursor.execute("""
            SELECT DISTINCT mc.canonical_name
            FROM intervention_mechanisms im
            JOIN mechanism_clusters mc ON im.cluster_id = mc.cluster_id
            WHERE im.intervention_id = ?
            ORDER BY mc.canonical_name
        """, (intervention_id,))
        results = cursor.fetchall()
        return [row[0] for row in results] if results else []

    interventions = []
    for row in rows:
        # Get multi-category data for intervention and condition
        intervention_categories = get_entity_categories(cursor, 'intervention', row['id'])
        condition_categories = get_entity_categories(cursor, 'condition', row['health_condition'])

        # Get mechanism canonical names (Phase 3c)
        mechanism_canonical_names = get_mechanism_canonical_names(row['id'])

        intervention = {
            'id': row['id'],
            'intervention': {
                'name': row['intervention_name'],
                'canonical_name': row['intervention_canonical_name'],
                'category': row['intervention_category'],  # Legacy single category
                'categories': intervention_categories,  # NEW: Multi-category support
                'details': row['intervention_details'],
                'delivery_method': row['delivery_method'],
                'hierarchy': {
                    'layer_0_category': row['intervention_l0_category'],
                    'layer_1_canonical': row['intervention_l1_canonical'],
                    'layer_2_variant': row['intervention_l2_variant'],
                    'layer_3_detail': row['intervention_l3_detail']
                }
            },
            'condition': {
                'name': row['health_condition'],
                'canonical_name': row['condition_canonical_name'],
                'category': row['condition_category'],  # Legacy single category
                'categories': condition_categories,  # NEW: Multi-category support
                'severity': row['severity'],
                'hierarchy': {
                    'layer_0_category': row['condition_l0_category'],
                    'layer_1_canonical': row['condition_l1_canonical'],
                    'layer_2_variant': row['condition_l2_variant'],
                    'layer_3_detail': row['condition_l3_detail']
                }
            },
            'mechanism': row['mechanism'],
            'mechanism_canonical_names': mechanism_canonical_names,  # NEW: Mechanism canonical names from Phase 3c
            'correlation': {
                'type': row['correlation_type'],
                'strength': row['correlation_strength'],
                'extraction_confidence': row['extraction_confidence'],
                'study_confidence': row['study_confidence']
            },
            'bayesian_scoring': {
                'score': row['bayesian_score'],  # Posterior mean (0-1)
                'conservative_score': row['bayesian_conservative_score'],  # 10th percentile
                'positive_evidence': row['positive_evidence_count'],
                'negative_evidence': row['negative_evidence_count'],
                'neutral_evidence': row['neutral_evidence_count'],
                'total_studies': row['bayesian_total_studies'],
                'bayes_factor': row['bayes_factor']
            } if row['bayesian_score'] is not None else None,
            'study': {
                'type': row['study_type'],
                'sample_size': row['sample_size'],
                'duration': row['study_duration'],
                'population': row['population_details'],
                'adverse_effects': row['adverse_effects'],
                'cost_category': row['cost_category'],
                'study_focus': json.loads(row['study_focus']) if row['study_focus'] else None,
                'measured_metrics': json.loads(row['measured_metrics']) if row['measured_metrics'] else None,
                'findings': json.loads(row['findings']) if row['findings'] else None,
                'study_location': row['study_location'],
                'publisher': row['publisher']
            },
            'paper': {
                'title': row['paper_title'],
                'journal': row['paper_journal'],
                'publication_date': row['publication_date'],
                'pubmed_id': row['pubmed_id'],
                'doi': row['doi']
            },
            'supporting_quote': row['supporting_quote'],
            'extraction_model': row['extraction_model'],
            'extraction_timestamp': row['extraction_timestamp']
        }
        interventions.append(intervention)

    # Get summary statistics
    cursor.execute("SELECT COUNT(*) FROM interventions")
    total_interventions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT intervention_name) FROM interventions")
    unique_interventions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT health_condition) FROM interventions")
    unique_conditions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT paper_id) FROM interventions")
    unique_papers = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM interventions WHERE correlation_type = 'positive'")
    positive_correlations = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM interventions WHERE correlation_type = 'negative'")
    negative_correlations = cursor.fetchone()[0]

    # Get semantic hierarchy statistics (Phase 3a - canonical groups only)
    cursor.execute("SELECT COUNT(*) FROM semantic_hierarchy WHERE entity_type = 'intervention'")
    semantic_interventions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT layer_1_canonical) FROM semantic_hierarchy WHERE entity_type = 'intervention' AND layer_1_canonical IS NOT NULL")
    canonical_groups = cursor.fetchone()[0]

    # Get top interventions by frequency (using Phase 3.5 hierarchical canonical names)
    cursor.execute("""
        SELECT
            COALESCE(sh.layer_1_canonical, i.intervention_name) as name,
            i.intervention_category,
            COUNT(*) as count,
            AVG(i.correlation_strength) as avg_strength,
            COUNT(DISTINCT i.paper_id) as paper_count
        FROM interventions i
        LEFT JOIN semantic_hierarchy sh ON i.intervention_name = sh.entity_name AND sh.entity_type = 'intervention'
        WHERE i.correlation_type = 'positive'
        GROUP BY COALESCE(sh.layer_1_canonical, i.intervention_name), i.intervention_category
        ORDER BY count DESC
        LIMIT 10
    """)
    top_interventions = [dict(row) for row in cursor.fetchall()]

    # Get top conditions by frequency (using Phase 3.5 hierarchical canonical names)
    cursor.execute("""
        SELECT
            COALESCE(sh.layer_1_canonical, i.health_condition) as name,
            i.condition_category,
            COUNT(*) as count,
            COUNT(DISTINCT i.intervention_name) as intervention_count,
            COUNT(DISTINCT i.paper_id) as paper_count
        FROM interventions i
        LEFT JOIN semantic_hierarchy sh ON i.health_condition = sh.entity_name AND sh.entity_type = 'condition'
        GROUP BY COALESCE(sh.layer_1_canonical, i.health_condition), i.condition_category
        ORDER BY count DESC
        LIMIT 10
    """)
    top_conditions = [dict(row) for row in cursor.fetchall()]

    # Get category breakdown
    cursor.execute("""
        SELECT intervention_category, COUNT(*) as count
        FROM interventions
        WHERE intervention_category IS NOT NULL
        GROUP BY intervention_category
        ORDER BY count DESC
    """)
    intervention_categories = {row['intervention_category']: row['count'] for row in cursor.fetchall()}

    cursor.execute("""
        SELECT condition_category, COUNT(*) as count
        FROM interventions
        WHERE condition_category IS NOT NULL
        GROUP BY condition_category
        ORDER BY count DESC
    """)
    condition_categories = {row['condition_category']: row['count'] for row in cursor.fetchall()}

    # NEW: Get multi-category statistics
    multi_category_stats = {}
    multi_category_interventions = 0
    try:
        cursor.execute("""
            SELECT category_type, category_name, COUNT(*) as count
            FROM intervention_category_mapping
            GROUP BY category_type, category_name
            ORDER BY category_type, count DESC
        """)
        for row in cursor.fetchall():
            cat_type = row['category_type']
            if cat_type not in multi_category_stats:
                multi_category_stats[cat_type] = {}
            multi_category_stats[cat_type][row['category_name']] = row['count']

        # Count interventions with multiple categories
        cursor.execute("""
            SELECT COUNT(DISTINCT intervention_id) as count
            FROM intervention_category_mapping
            GROUP BY intervention_id
            HAVING COUNT(*) > 1
        """)
        multi_category_interventions = len(cursor.fetchall())
    except sqlite3.OperationalError:
        # Tables don't exist yet - skip multi-category stats
        pass

    # Get Bayesian score statistics (Phase 4b)
    cursor.execute("SELECT COUNT(*) FROM bayesian_scores")
    bayesian_scores_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM bayesian_scores WHERE posterior_mean > 0.7")
    high_bayesian_scores = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM bayesian_scores WHERE posterior_mean > 0.5")
    medium_bayesian_scores = cursor.fetchone()[0]

    # Get total relationships (sum across canonical groups)
    cursor.execute("SELECT COUNT(DISTINCT intervention_name || '::' || condition_name) FROM bayesian_scores")
    total_relationships = cursor.fetchone()[0]

    conn.close()

    # Compile the complete dataset
    data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_interventions': total_interventions,
            'unique_interventions': unique_interventions,
            'unique_conditions': unique_conditions,
            'unique_papers': unique_papers,
            'semantic_interventions': semantic_interventions,
            'canonical_groups': canonical_groups,
            'positive_correlations': positive_correlations,
            'negative_correlations': negative_correlations,
            'intervention_categories': intervention_categories,
            'condition_categories': condition_categories,
            'multi_category_stats': multi_category_stats,  # NEW: Multi-category statistics
            'multi_category_interventions': multi_category_interventions,  # NEW: Count of interventions with >1 category
            'bayesian_scores_available': bayesian_scores_count > 0,  # Phase 4b integration
            'total_relationships': total_relationships if bayesian_scores_count > 0 else canonical_groups,  # For backward compatibility
            'high_scoring_interventions': high_bayesian_scores,  # Bayesian score > 0.7
            'medium_scoring_interventions': medium_bayesian_scores  # Bayesian score > 0.5
        },
        'top_performers': {
            'interventions': top_interventions,
            'conditions': top_conditions
        },
        'interventions': interventions
    }

    return data

def export_mechanism_clusters_data() -> Dict[str, Any]:
    """
    Export mechanism cluster data for frontend display.

    Returns:
        Dictionary with mechanism clusters, membership, and analytics
    """
    db_path = get_database_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all mechanism clusters with their members
    query = """
    SELECT
        mc.cluster_id,
        mc.canonical_name,
        mc.member_count,
        mc.hierarchy_level,
        mc.avg_silhouette,
        GROUP_CONCAT(mcm.mechanism_text, '|||') as member_mechanisms
    FROM mechanism_clusters mc
    LEFT JOIN mechanism_cluster_membership mcm ON mc.cluster_id = mcm.cluster_id
    GROUP BY mc.cluster_id
    ORDER BY mc.member_count DESC, mc.canonical_name
    """

    cursor.execute(query)
    cluster_rows = cursor.fetchall()

    clusters = []
    for row in cluster_rows:
        cluster = {
            'cluster_id': row['cluster_id'],
            'canonical_name': row['canonical_name'],
            'member_count': row['member_count'],
            'cluster_type': 'singleton' if row['member_count'] == 1 else 'hdbscan',
            'hierarchy_level': row['hierarchy_level'],
            'avg_silhouette': row['avg_silhouette'],
            'members': row['member_mechanisms'].split('|||') if row['member_mechanisms'] else []
        }
        clusters.append(cluster)

    # Get mechanism-condition associations (if table exists)
    try:
        cursor.execute("""
            SELECT
                cluster_id,
                condition_name,
                intervention_count,
                avg_correlation_strength
            FROM mechanism_condition_associations
            ORDER BY cluster_id, intervention_count DESC
        """)
        associations = [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        # Table doesn't exist yet - expected if associations not built
        associations = []

    # Get summary statistics
    cursor.execute("SELECT COUNT(*) FROM mechanism_clusters")
    total_clusters = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM mechanism_cluster_membership")
    total_mechanisms = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM mechanism_clusters WHERE member_count > 1")
    natural_clusters = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM mechanism_clusters WHERE member_count = 1")
    singleton_clusters = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(member_count) FROM mechanism_clusters")
    avg_cluster_size = cursor.fetchone()[0] or 0.0

    # Get top mechanisms by member count (since intervention_mechanisms might not exist yet)
    cursor.execute("""
        SELECT
            canonical_name,
            member_count,
            cluster_id
        FROM mechanism_clusters
        ORDER BY member_count DESC
        LIMIT 10
    """)
    top_mechanisms = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return {
        'metadata': {
            'total_clusters': total_clusters,
            'total_mechanisms': total_mechanisms,
            'natural_clusters': natural_clusters,
            'singleton_clusters': singleton_clusters,
            'avg_cluster_size': round(avg_cluster_size, 2),
            'assignment_rate': 1.0  # 100% by design
        },
        'clusters': clusters,
        'associations': associations,
        'top_mechanisms': top_mechanisms
    }

def main():
    """Export data to JSON file for frontend."""
    print("Exporting intervention research data to JSON...")

    # Get interventions data
    data = export_interventions_data()

    # Ensure frontend data directory exists
    frontend_data_path = get_frontend_data_path()
    frontend_data_path.mkdir(parents=True, exist_ok=True)

    # Write interventions to JSON file
    output_file = frontend_data_path / "interventions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Interventions data exported successfully to {output_file}")
    print(f"Total interventions: {data['metadata']['total_interventions']}")
    print(f"Unique interventions: {data['metadata']['unique_interventions']}")
    print(f"Unique conditions: {data['metadata']['unique_conditions']}")
    print(f"Papers referenced: {data['metadata']['unique_papers']}")
    print(f"Canonical groups: {data['metadata']['canonical_groups']}")

    # Export mechanism clusters data (Phase 3.6)
    print("\nExporting mechanism clusters data...")
    try:
        mechanism_data = export_mechanism_clusters_data()

        # Write mechanisms to separate JSON file
        mechanism_output_file = frontend_data_path / "mechanism_clusters.json"
        with open(mechanism_output_file, 'w', encoding='utf-8') as f:
            json.dump(mechanism_data, f, indent=2, ensure_ascii=False)

        print(f"Mechanism clusters data exported successfully to {mechanism_output_file}")
        print(f"Total mechanism clusters: {mechanism_data['metadata']['total_clusters']}")
        print(f"  - Natural clusters: {mechanism_data['metadata']['natural_clusters']}")
        print(f"  - Singleton clusters: {mechanism_data['metadata']['singleton_clusters']}")
        print(f"Total mechanisms: {mechanism_data['metadata']['total_mechanisms']}")
        print(f"Average cluster size: {mechanism_data['metadata']['avg_cluster_size']}")
        print(f"Assignment rate: {mechanism_data['metadata']['assignment_rate']:.1%}")
    except Exception as e:
        print(f"Warning: Failed to export mechanism clusters data: {e}")
        print("This is expected if Phase 3.6 hasn't been run yet.")

if __name__ == "__main__":
    main()
