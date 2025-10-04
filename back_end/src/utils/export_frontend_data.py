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

    # Query to get interventions with paper details and canonical entities
    query = """
    SELECT
        i.id,
        i.intervention_name,
        i.intervention_category,
        i.intervention_details,
        i.health_condition,
        i.condition_category,
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
        ie.canonical_name as intervention_canonical_name,
        ce.canonical_name as condition_canonical_name,
        i.extraction_model,
        i.extraction_timestamp
    FROM interventions i
    LEFT JOIN papers p ON i.paper_id = p.pmid
    LEFT JOIN canonical_entities ie ON i.intervention_canonical_id = ie.id
    LEFT JOIN canonical_entities ce ON i.condition_canonical_id = ce.id
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
            },
            'condition': {
                'name': row['health_condition'],
                'canonical_name': row['condition_canonical_name'],
                'category': row['condition_category'],
                'severity': row['severity']
            },
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

    cursor.execute("SELECT COUNT(*) FROM canonical_entities")
    total_canonical_entities = cursor.fetchone()[0]

    # Get top interventions by frequency
    cursor.execute("""
        SELECT
            COALESCE(ce.canonical_name, i.intervention_name) as name,
            i.intervention_category,
            COUNT(*) as count,
            AVG(i.correlation_strength) as avg_strength,
            COUNT(DISTINCT i.paper_id) as paper_count
        FROM interventions i
        LEFT JOIN canonical_entities ce ON i.intervention_canonical_id = ce.id
        WHERE i.correlation_type = 'positive'
        GROUP BY COALESCE(ce.canonical_name, i.intervention_name), i.intervention_category
        ORDER BY count DESC
        LIMIT 10
    """)
    top_interventions = [dict(row) for row in cursor.fetchall()]

    # Get top conditions by frequency
    cursor.execute("""
        SELECT
            COALESCE(ce.canonical_name, i.health_condition) as name,
            i.condition_category,
            COUNT(*) as count,
            COUNT(DISTINCT i.intervention_name) as intervention_count,
            COUNT(DISTINCT i.paper_id) as paper_count
        FROM interventions i
        LEFT JOIN canonical_entities ce ON i.condition_canonical_id = ce.id
        GROUP BY COALESCE(ce.canonical_name, i.health_condition), i.condition_category
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
            'canonical_entities': total_canonical_entities,
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
    print(f"Canonical entities: {data['metadata']['canonical_entities']}")

if __name__ == "__main__":
    main()
