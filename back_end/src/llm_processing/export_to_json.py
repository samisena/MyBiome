#!/usr/bin/env python3
"""
Unified Data Export for JSON Frontend
Exports intervention-health correlations data with optional entity normalization.
Consolidates both basic and enhanced export functionality.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime
from back_end.src.data_collection.database_manager import database_manager

# Optional import for entity normalization - graceful fallback if not available
try:
    from .batch_entity_processor import BatchEntityProcessor as EntityNormalizer
    NORMALIZATION_AVAILABLE = True
except ImportError:
    NORMALIZATION_AVAILABLE = False

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


class UnifiedDataExporter:
    """
    Unified data exporter that combines basic and enhanced export functionality.
    Supports optional entity normalization while maintaining backward compatibility.
    """

    def __init__(self, use_normalization: bool = False):
        """
        Initialize the unified exporter.

        Args:
            use_normalization: Enable entity normalization features if available
        """
        self.use_normalization = use_normalization and NORMALIZATION_AVAILABLE
        if use_normalization and not NORMALIZATION_AVAILABLE:
            print("Warning: Normalization requested but not available - running without normalization")

    def get_display_info(self, term: str, entity_type: str) -> Dict[str, Any]:
        """
        Get display information for a term including canonical name and alternatives.

        Args:
            term: Original term
            entity_type: 'intervention' or 'condition'

        Returns:
            Dictionary with canonical_name, original_term, alternative_names
        """
        if not self.use_normalization:
            return {
                'canonical_name': term,
                'original_term': term,
                'alternative_names': [],
                'normalized': False
            }

        try:
            with database_manager.get_connection() as conn:
                normalizer = EntityNormalizer(conn)
                mapping = normalizer.find_mapping(term, entity_type)

                if mapping:
                    # Get all alternative names for this canonical entity
                    alternatives = normalizer.get_alternative_names(mapping['canonical_id'])
                    return {
                        'canonical_name': mapping['canonical_name'],
                        'original_term': term,
                        'alternative_names': [alt for alt in alternatives if alt != term],
                        'normalized': True,
                        'confidence': mapping.get('confidence', 1.0)
                    }
                else:
                    return {
                        'canonical_name': term,
                        'original_term': term,
                        'alternative_names': [],
                        'normalized': False
                    }

        except Exception as e:
            print(f"Error getting display info for {term}: {e}")
            return {
                'canonical_name': term,
                'original_term': term,
                'alternative_names': [],
                'normalized': False
            }

    def export_interventions_enhanced(self, include_normalization: bool = None) -> Dict[str, Any]:
        """
        Export interventions with enhanced features and optional normalization.

        Args:
            include_normalization: Override instance setting for normalization

        Returns:
            Comprehensive interventions dataset
        """
        use_norm = include_normalization if include_normalization is not None else self.use_normalization

        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Enhanced query with normalization fields if available
            base_query = """
                SELECT
                    i.id,
                    i.intervention_category,
                    i.intervention_name,
                    i.health_condition,
                    i.correlation_type,
                    i.correlation_strength,
                    i.confidence_score,
                    i.sample_size,
                    i.study_duration,
                    i.study_type,
                    i.delivery_method,
                    i.severity,
                    i.adverse_effects,
                    i.cost_category,
                    i.supporting_quote,
                    i.extraction_model,
                    p.title as paper_title,
                    p.authors,
                    p.publication_date,
                    p.pmid,
                    p.doi"""

            if use_norm:
                base_query += """,
                    i.intervention_canonical_id,
                    i.condition_canonical_id,
                    i.normalized"""

            query = base_query + """
                FROM interventions i
                LEFT JOIN papers p ON i.paper_id = p.id
                WHERE i.intervention_name IS NOT NULL
                AND i.health_condition IS NOT NULL
                ORDER BY i.confidence_score DESC NULLS LAST,
                         i.correlation_strength DESC NULLS LAST
            """

            cursor.execute(query)
            interventions = []

            for row in cursor.fetchall():
                intervention = {
                    'id': row[0],
                    'intervention_category': row[1],
                    'intervention_name': row[2],
                    'health_condition': row[3],
                    'correlation_type': row[4],
                    'correlation_strength': row[5],
                    'confidence_score': row[6],
                    'sample_size': row[7],
                    'study_duration': row[8],
                    'study_type': row[9],
                    'delivery_method': row[10],
                    'severity': row[11],
                    'adverse_effects': row[12],
                    'cost_category': row[13],
                    'supporting_quote': row[14],
                    'extraction_model': row[15],
                    'paper': {
                        'title': row[16],
                        'authors': row[17],
                        'publication_date': row[18],
                        'pmid': row[19],
                        'doi': row[20]
                    }
                }

                # Add normalization info if available
                if use_norm and len(row) > 21:
                    intervention.update({
                        'intervention_canonical_id': row[21],
                        'condition_canonical_id': row[22],
                        'normalized': row[23],
                        'intervention_display': self.get_display_info(row[2], 'intervention'),
                        'condition_display': self.get_display_info(row[3], 'condition')
                    })

                interventions.append(intervention)

        return {
            'interventions': interventions,
            'total_count': len(interventions),
            'export_metadata': {
                'export_date': datetime.now().isoformat(),
                'normalization_enabled': use_norm,
                'data_source': 'intervention_research_database'
            }
        }

    def export_summary_statistics(self) -> Dict[str, Any]:
        """Export summary statistics for the dataset."""
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM interventions")
            stats['total_interventions'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM papers")
            stats['total_papers'] = cursor.fetchone()[0]

            # Intervention categories
            cursor.execute("""
                SELECT intervention_category, COUNT(*) as count
                FROM interventions
                WHERE intervention_category IS NOT NULL
                GROUP BY intervention_category
                ORDER BY count DESC
            """)
            stats['intervention_categories'] = dict(cursor.fetchall())

            # Top conditions
            cursor.execute("""
                SELECT health_condition, COUNT(*) as count
                FROM interventions
                WHERE health_condition IS NOT NULL
                GROUP BY health_condition
                ORDER BY count DESC
                LIMIT 20
            """)
            stats['top_conditions'] = dict(cursor.fetchall())

            # Correlation types
            cursor.execute("""
                SELECT correlation_type, COUNT(*) as count
                FROM interventions
                WHERE correlation_type IS NOT NULL
                GROUP BY correlation_type
            """)
            stats['correlation_types'] = dict(cursor.fetchall())

            # Normalization statistics if available
            if self.use_normalization:
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(CASE WHEN normalized = 1 THEN 1 END) as normalized_count
                    FROM interventions
                """)
                norm_stats = cursor.fetchone()
                stats['normalization'] = {
                    'total_interventions': norm_stats[0],
                    'normalized_count': norm_stats[1],
                    'normalization_rate': (norm_stats[1] / norm_stats[0] * 100) if norm_stats[0] > 0 else 0
                }

        return stats

def export_correlations_data() -> Dict[str, Any]:
    """Export interventions data with paper information (enhanced with all optional fields)."""
    db_path = get_database_path()

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    with database_manager.get_connection() as conn:

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

    with database_manager.get_connection() as conn:
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
    """Export data with unified exporter and command-line options."""
    import argparse

    parser = argparse.ArgumentParser(description="Export intervention data to JSON")
    parser.add_argument('--normalization', action='store_true',
                       help='Enable entity normalization in export')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory')
    parser.add_argument('--format', choices=['enhanced', 'basic', 'minimal', 'all'],
                       default='basic', help='Export format')

    args = parser.parse_args()

    try:
        # Create unified exporter
        exporter = UnifiedDataExporter(use_normalization=args.normalization)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(get_output_path()).parent

        if args.format in ['enhanced', 'all']:
            # Export enhanced data
            enhanced_path = output_dir / 'interventions_enhanced.json'
            enhanced_data = exporter.export_interventions_enhanced()

            with open(enhanced_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Exported {enhanced_data['total_count']} enhanced interventions to {enhanced_path}")

        if args.format in ['basic', 'all']:
            # Export basic correlations data (backward compatibility)
            basic_path = output_dir / 'correlations.json'
            basic_data = export_correlations_data()

            with open(basic_path, 'w', encoding='utf-8') as f:
                json.dump(basic_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Exported {len(basic_data['interventions'])} correlations to {basic_path}")

        if args.format in ['minimal', 'all']:
            # Export minimal dataset
            minimal_path = output_dir / 'minimal_dataset.json'
            minimal_data = export_minimal_dataset()

            with open(minimal_path, 'w', encoding='utf-8') as f:
                json.dump(minimal_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Exported {len(minimal_data)} minimal records to {minimal_path}")

        if args.format in ['all']:
            # Export summary statistics
            stats_path = output_dir / 'summary_statistics.json'
            stats_data = exporter.export_summary_statistics()

            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, indent=2, ensure_ascii=False)

            print(f"✓ Exported summary statistics to {stats_path}")

        print(f"Export completed successfully! (normalization: {exporter.use_normalization})")

    except Exception as e:
        print(f"✗ Export failed: {e}")
        raise

if __name__ == "__main__":
    main()