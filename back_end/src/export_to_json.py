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
    script_dir = Path(__file__).parent
    db_path = script_dir.parent / "data" / "processed" / "pubmed_research.db"
    return str(db_path)

def get_output_path() -> str:
    """Get the output path for JSON file."""
    script_dir = Path(__file__).parent
    output_path = script_dir.parent.parent / "frontend" / "data" / "correlations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)

def export_correlations_data() -> Dict[str, Any]:
    """Export correlations data with paper information."""
    db_path = get_database_path()
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # Query correlations with paper details
        query = """
        SELECT 
            c.id,
            c.probiotic_strain,
            c.health_condition,
            c.correlation_type,
            c.correlation_strength,
            c.confidence_score,
            c.sample_size,
            c.study_type,
            c.effect_size,
            c.dosage,
            c.study_duration,
            c.population_details,
            c.supporting_quote,
            c.validation_status,
            p.pmid,
            p.title,
            p.journal,
            p.publication_date,
            p.doi
        FROM correlations c
        LEFT JOIN papers p ON c.paper_id = p.pmid
        ORDER BY c.confidence_score DESC, c.correlation_strength DESC
        """
        
        cursor = conn.execute(query)
        correlations = []
        
        for row in cursor.fetchall():
            correlation = {
                'id': row['id'],
                'probiotic_strain': row['probiotic_strain'],
                'health_condition': row['health_condition'],
                'correlation_type': row['correlation_type'],
                'correlation_strength': row['correlation_strength'],
                'confidence_score': row['confidence_score'],
                'sample_size': row['sample_size'],
                'study_type': row['study_type'] or 'Not specified',
                'effect_size': row['effect_size'] or 'Not specified',
                'dosage': row['dosage'] or 'Not specified',
                'study_duration': row['study_duration'] or 'Not specified',
                'population_details': row['population_details'] or 'Not specified',
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
            correlations.append(correlation)
        
        # Get summary statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_correlations,
            COUNT(DISTINCT probiotic_strain) as unique_strains,
            COUNT(DISTINCT health_condition) as unique_conditions,
            COUNT(DISTINCT paper_id) as unique_papers,
            COUNT(CASE WHEN correlation_type = 'positive' THEN 1 END) as positive_correlations,
            COUNT(CASE WHEN correlation_type = 'negative' THEN 1 END) as negative_correlations,
            COUNT(CASE WHEN correlation_type = 'neutral' THEN 1 END) as neutral_correlations,
            COUNT(CASE WHEN correlation_type = 'inconclusive' THEN 1 END) as inconclusive_correlations,
            AVG(confidence_score) as avg_confidence,
            AVG(correlation_strength) as avg_correlation_strength
        FROM correlations
        """
        
        stats_cursor = conn.execute(stats_query)
        stats_row = stats_cursor.fetchone()
        
        summary_stats = {
            'total_correlations': stats_row['total_correlations'],
            'unique_strains': stats_row['unique_strains'],
            'unique_conditions': stats_row['unique_conditions'],
            'unique_papers': stats_row['unique_papers'],
            'positive_correlations': stats_row['positive_correlations'] or 0,
            'negative_correlations': stats_row['negative_correlations'] or 0,
            'neutral_correlations': stats_row['neutral_correlations'] or 0,
            'inconclusive_correlations': stats_row['inconclusive_correlations'] or 0,
            'avg_confidence': round(stats_row['avg_confidence'] or 0, 3),
            'avg_correlation_strength': round(stats_row['avg_correlation_strength'] or 0, 3)
        }
        
        # Get top strains and conditions
        top_strains_query = """
        SELECT probiotic_strain, COUNT(*) as count
        FROM correlations
        GROUP BY probiotic_strain
        ORDER BY count DESC
        LIMIT 10
        """
        
        top_conditions_query = """
        SELECT health_condition, COUNT(*) as count
        FROM correlations
        GROUP BY health_condition
        ORDER BY count DESC
        LIMIT 10
        """
        
        top_strains = [dict(row) for row in conn.execute(top_strains_query).fetchall()]
        top_conditions = [dict(row) for row in conn.execute(top_conditions_query).fetchall()]
        
        return {
            'correlations': correlations,
            'summary_stats': summary_stats,
            'top_strains': top_strains,
            'top_conditions': top_conditions,
            'export_timestamp': __import__('datetime').datetime.now().isoformat()
        }

def main():
    """Main export function."""
    try:
        print("Exporting correlations data to JSON...")
        
        data = export_correlations_data()
        output_path = get_output_path()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print("Export completed successfully!")
        print(f"Exported {len(data['correlations'])} correlations")
        print(f"Output: {output_path}")
        print("Summary stats:")
        print(f"   - {data['summary_stats']['unique_strains']} unique strains")
        print(f"   - {data['summary_stats']['unique_conditions']} unique conditions")
        print(f"   - {data['summary_stats']['positive_correlations']} positive correlations")
        print(f"   - {data['summary_stats']['negative_correlations']} negative correlations")
        
    except Exception as e:
        print(f"Export failed: {e}")
        raise

if __name__ == "__main__":
    main()