#!/usr/bin/env python3
"""
Export SQLite data to JSON for static website frontend.
Exports probiotic-health correlations data for DataTables.js visualization.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, List, Any

def get_database_path() -> str:
    """Get the path to the SQLite database."""
    from back_end.src.data.config import config
    return str(config.db_path)

def get_output_path() -> str:
    """Get the output path for JSON file."""
    script_dir = Path(__file__).parent
    output_path = script_dir.parent.parent / "frontend" / "data" / "correlations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)

def export_correlations_data() -> Dict[str, Any]:
    """Export interventions data with paper information (enhanced with all optional fields)."""
    db_path = get_database_path()

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        # Query interventions with paper details (enhanced query)
        query = """
        SELECT
            i.id,
            i.intervention_name,
            i.intervention_category,
            i.health_condition,
            i.correlation_type,
            i.correlation_strength,
            i.confidence_score,
            i.sample_size,
            i.study_type,
            i.study_duration,
            i.population_details,
            i.supporting_quote,
            i.delivery_method,
            i.severity,
            i.adverse_effects,
            i.cost_category,
            i.validation_status,
            p.pmid,
            p.title,
            p.journal,
            p.publication_date,
            p.doi
        FROM interventions i
        LEFT JOIN papers p ON i.paper_id = p.pmid
        ORDER BY i.confidence_score DESC, i.correlation_strength DESC
        """
        
        cursor = conn.execute(query)
        interventions = []

        for row in cursor.fetchall():
            # Extract publication year for tier 1
            publication_year = None
            if row['publication_date']:
                try:
                    publication_year = int(row['publication_date'][:4])
                except (ValueError, TypeError):
                    publication_year = None

            # Build intervention record with all tiers
            intervention = {
                # Core required fields
                'condition': row['health_condition'],
                'intervention': row['intervention_name'],
                'correlation': 'unsure' if row['correlation_type'] == 'inconclusive' else row['correlation_type'],

                # Tier 1 optional
                'study_size': row['sample_size'],
                'publication_year': publication_year,
                'confidence_score': row['confidence_score'],

                # Tier 2 optional
                'duration': row['study_duration'] or None,
                'demographic': row['population_details'] or None,
                'delivery_method': row['delivery_method'] or None,
                'severity': row['severity'] or None,

                # Tier 3 optional
                'study_type': row['study_type'] or None,
                'journal': row['journal'] or None,
                'adverse_effects': row['adverse_effects'] or None,
                'cost_category': row['cost_category'] or None,

                # Additional metadata
                'id': row['id'],
                'intervention_category': row['intervention_category'],
                'correlation_strength': row['correlation_strength'],
                'supporting_quote': row['supporting_quote'],
                'validation_status': row['validation_status'],
                'paper': {
                    'pmid': row['pmid'],
                    'title': row['title'],
                    'journal': row['journal'],
                    'publication_date': row['publication_date'],
                    'doi': row['doi'],
                    'pubmed_url': f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/" if row['pmid'] else None
                }
            }
            interventions.append(intervention)
        
        # Get summary statistics
        stats_query = """
        SELECT
            COUNT(*) as total_interventions,
            COUNT(DISTINCT intervention_name) as unique_interventions,
            COUNT(DISTINCT health_condition) as unique_conditions,
            COUNT(DISTINCT paper_id) as unique_papers,
            COUNT(CASE WHEN correlation_type = 'positive' THEN 1 END) as positive_correlations,
            COUNT(CASE WHEN correlation_type = 'negative' THEN 1 END) as negative_correlations,
            COUNT(CASE WHEN correlation_type = 'neutral' THEN 1 END) as neutral_correlations,
            COUNT(CASE WHEN correlation_type = 'inconclusive' THEN 1 END) as inconclusive_correlations,
            AVG(confidence_score) as avg_confidence,
            AVG(correlation_strength) as avg_correlation_strength
        FROM interventions
        """
        
        stats_cursor = conn.execute(stats_query)
        stats_row = stats_cursor.fetchone()
        
        summary_stats = {
            'total_interventions': stats_row['total_interventions'],
            'unique_interventions': stats_row['unique_interventions'],
            'unique_conditions': stats_row['unique_conditions'],
            'unique_papers': stats_row['unique_papers'],
            'positive_correlations': stats_row['positive_correlations'] or 0,
            'negative_correlations': stats_row['negative_correlations'] or 0,
            'neutral_correlations': stats_row['neutral_correlations'] or 0,
            'inconclusive_correlations': stats_row['inconclusive_correlations'] or 0,
            'avg_confidence': round(stats_row['avg_confidence'] or 0, 3),
            'avg_correlation_strength': round(stats_row['avg_correlation_strength'] or 0, 3)
        }
        
        # Get top interventions and conditions
        top_interventions_query = """
        SELECT intervention_name, COUNT(*) as count
        FROM interventions
        GROUP BY intervention_name
        ORDER BY count DESC
        LIMIT 10
        """

        top_conditions_query = """
        SELECT health_condition, COUNT(*) as count
        FROM interventions
        GROUP BY health_condition
        ORDER BY count DESC
        LIMIT 10
        """

        top_interventions = [dict(row) for row in conn.execute(top_interventions_query).fetchall()]
        top_conditions = [dict(row) for row in conn.execute(top_conditions_query).fetchall()]

        return {
            'interventions': interventions,
            'summary_stats': summary_stats,
            'top_interventions': top_interventions,
            'top_conditions': top_conditions,
            'export_timestamp': __import__('datetime').datetime.now().isoformat()
        }


def export_minimal_dataset() -> List[Dict[str, str]]:
    """Export minimal viable dataset format."""
    db_path = get_database_path()

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT
                health_condition as condition,
                intervention_name as intervention,
                CASE
                    WHEN correlation_type = 'inconclusive' THEN 'unsure'
                    ELSE correlation_type
                END as correlation
            FROM interventions
            WHERE validation_status != 'failed'
            AND health_condition IS NOT NULL
            AND intervention_name IS NOT NULL
            AND correlation_type IS NOT NULL
        ''')
        return [{'condition': row[0], 'intervention': row[1], 'correlation': row[2]}
                for row in cursor.fetchall()]

def main():
    """Main export function."""
    try:
        print("Exporting enhanced interventions data to JSON...")

        data = export_correlations_data()
        output_path = get_output_path()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print("Export completed successfully!")
        print(f"Exported {len(data['interventions'])} interventions with all optional tiers")
        print(f"Output: {output_path}")
        print("Summary stats:")
        print(f"   - {data['summary_stats']['unique_interventions']} unique interventions")
        print(f"   - {data['summary_stats']['unique_conditions']} unique conditions")
        print(f"   - {data['summary_stats']['positive_correlations']} positive correlations")
        print(f"   - {data['summary_stats']['negative_correlations']} negative correlations")
        
    except Exception as e:
        print(f"Export failed: {e}")
        raise

if __name__ == "__main__":
    main()