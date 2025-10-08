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

    # Query to get interventions with paper details and hierarchical semantic data (Phase 3.5)
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
        i.extraction_model,
        i.extraction_timestamp
    FROM interventions i
    LEFT JOIN papers p ON i.paper_id = p.pmid
    LEFT JOIN semantic_hierarchy sh_i ON i.intervention_name = sh_i.entity_name AND sh_i.entity_type = 'intervention'
    LEFT JOIN semantic_hierarchy sh_c ON i.health_condition = sh_c.entity_name AND sh_c.entity_type = 'condition'
    ORDER BY i.extraction_timestamp DESC
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    interventions = []
    for row in rows:
        intervention = {
            'id': row['id'],
            'intervention': {
                'name': row['intervention_name'],
                'canonical_name': row['intervention_canonical_name'],
                'category': row['intervention_category'],
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
                'category': row['condition_category'],
                'severity': row['severity'],
                'hierarchy': {
                    'layer_0_category': row['condition_l0_category'],
                    'layer_1_canonical': row['condition_l1_canonical'],
                    'layer_2_variant': row['condition_l2_variant'],
                    'layer_3_detail': row['condition_l3_detail']
                }
            },
            'mechanism': row['mechanism'],
            'correlation': {
                'type': row['correlation_type'],
                'strength': row['correlation_strength'],
                'extraction_confidence': row['extraction_confidence'],
                'study_confidence': row['study_confidence']
            },
            'study': {
                'type': row['study_type'],
                'sample_size': row['sample_size'],
                'duration': row['study_duration'],
                'population': row['population_details'],
                'adverse_effects': row['adverse_effects'],
                'cost_category': row['cost_category']
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

    # Get semantic hierarchy statistics (Phase 3.5 - current system)
    cursor.execute("SELECT COUNT(*) FROM semantic_hierarchy WHERE entity_type = 'intervention'")
    semantic_interventions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT layer_1_canonical) FROM semantic_hierarchy WHERE entity_type = 'intervention' AND layer_1_canonical IS NOT NULL")
    canonical_groups = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM entity_relationships")
    total_relationships = cursor.fetchone()[0]

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
            'total_relationships': total_relationships,
            'positive_correlations': positive_correlations,
            'negative_correlations': negative_correlations,
            'intervention_categories': intervention_categories,
            'condition_categories': condition_categories
        },
        'top_performers': {
            'interventions': top_interventions,
            'conditions': top_conditions
        },
        'interventions': interventions
    }

    return data

def main():
    """Export data to JSON file for frontend."""
    print("Exporting intervention research data to JSON...")

    # Get data
    data = export_interventions_data()

    # Ensure frontend data directory exists
    frontend_data_path = get_frontend_data_path()
    frontend_data_path.mkdir(parents=True, exist_ok=True)

    # Write to JSON file
    output_file = frontend_data_path / "interventions.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Data exported successfully to {output_file}")
    print(f"Total interventions: {data['metadata']['total_interventions']}")
    print(f"Unique interventions: {data['metadata']['unique_interventions']}")
    print(f"Unique conditions: {data['metadata']['unique_conditions']}")
    print(f"Papers referenced: {data['metadata']['unique_papers']}")
    print(f"Canonical groups: {data['metadata']['canonical_groups']}")
    print(f"Semantic relationships: {data['metadata']['total_relationships']}")

if __name__ == "__main__":
    main()
