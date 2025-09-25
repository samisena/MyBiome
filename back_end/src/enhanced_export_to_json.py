#!/usr/bin/env python3
"""
Enhanced export with entity normalization integration.
Safely integrates normalized entities while maintaining backward compatibility.
"""

import sqlite3
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from entity_normalizer import EntityNormalizer


class EnhancedDataExporter:
    """Enhanced data exporter with entity normalization integration"""

    def __init__(self, db_path: str, use_normalization: bool = True):
        """
        Initialize the enhanced exporter

        Args:
            db_path: Path to SQLite database
            use_normalization: Feature flag to enable/disable normalization
        """
        self.db_path = db_path
        self.use_normalization = use_normalization
        self.normalizer = None

        # We'll create EntityNormalizer instances as needed to avoid connection issues

    def get_display_info(self, term: str, entity_type: str) -> Dict[str, Any]:
        """
        Get display information for a term including canonical name and alternatives

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
                'canonical_id': None,
                'is_normalized': False
            }

        # Create normalizer connection for this operation
        try:
            conn = sqlite3.connect(self.db_path)
            normalizer = EntityNormalizer(conn)

            # Get canonical mapping
            canonical_name = normalizer.get_canonical_name(term, entity_type)
            canonical_id = normalizer.find_canonical_id(term, entity_type)

            # If we have a mapping, get all alternative names
            alternative_names = []
            if canonical_id:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT raw_text FROM entity_mappings
                    WHERE canonical_id = ? AND raw_text != ?
                    ORDER BY confidence_score DESC
                """, (canonical_id, term))

                alternative_names = [row['raw_text'] for row in cursor.fetchall()]

            conn.close()

            return {
                'canonical_name': canonical_name,
                'original_term': term,
                'alternative_names': alternative_names,
                'canonical_id': canonical_id,
                'is_normalized': canonical_name != term
            }

        except Exception as e:
            print(f"Warning: Could not get display info for {term}: {e}")
            return {
                'canonical_name': term,
                'original_term': term,
                'alternative_names': [],
                'canonical_id': None,
                'is_normalized': False
            }

    def export_interventions_normalized(self) -> Dict[str, Any]:
        """Export interventions data with entity normalization"""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if self.use_normalization:
                # Enhanced query with entity mappings JOIN
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
                    p.doi,
                    -- Normalized intervention info
                    ei.canonical_name as intervention_canonical,
                    ei.id as intervention_canonical_id,
                    -- Normalized condition info
                    ec.canonical_name as condition_canonical,
                    ec.id as condition_canonical_id
                FROM interventions i
                LEFT JOIN papers p ON i.paper_id = p.pmid
                LEFT JOIN entity_mappings emi ON i.intervention_name = emi.raw_text AND emi.entity_type = 'intervention'
                LEFT JOIN canonical_entities ei ON emi.canonical_id = ei.id
                LEFT JOIN entity_mappings emc ON i.health_condition = emc.raw_text AND emc.entity_type = 'condition'
                LEFT JOIN canonical_entities ec ON emc.canonical_id = ec.id
                ORDER BY i.confidence_score DESC, i.correlation_strength DESC
                """
            else:
                # Standard query without normalization
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
                # Get display info for intervention and condition
                intervention_info = self.get_display_info(row['intervention_name'], 'intervention')
                condition_info = self.get_display_info(row['health_condition'], 'condition')

                # Extract publication year
                publication_year = None
                if row['publication_date']:
                    try:
                        publication_year = int(row['publication_date'][:4])
                    except (ValueError, TypeError):
                        publication_year = None

                # Build intervention record
                intervention = {
                    # Core fields (using canonical names for grouping)
                    'condition': condition_info['canonical_name'],
                    'intervention': intervention_info['canonical_name'],
                    'correlation': 'unsure' if row['correlation_type'] == 'inconclusive' else row['correlation_type'],

                    # Display information
                    'display_info': {
                        'intervention': intervention_info,
                        'condition': condition_info
                    },

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

                # Add normalization metadata if available
                if self.use_normalization and hasattr(row, 'intervention_canonical'):
                    intervention['normalization_info'] = {
                        'intervention_canonical_id': row.get('intervention_canonical_id'),
                        'condition_canonical_id': row.get('condition_canonical_id'),
                        'is_intervention_normalized': intervention_info['is_normalized'],
                        'is_condition_normalized': condition_info['is_normalized']
                    }

                interventions.append(intervention)

            return {
                'interventions': interventions,
                'metadata': {
                    'normalization_enabled': self.use_normalization,
                    'total_records': len(interventions)
                }
            }

    def get_top_interventions_for_condition(self, condition: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top interventions for a condition, grouped by canonical names

        Args:
            condition: Health condition to search for
            limit: Maximum number of results

        Returns:
            List of interventions grouped by canonical names
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if self.use_normalization:
                # First, get the canonical condition name
                condition_info = self.get_display_info(condition, 'condition')
                search_condition = condition_info['canonical_name']

                # Simplified approach: Get interventions and group them in Python
                # This avoids complex JOINs that might cause database issues
                query = """
                SELECT DISTINCT
                    i.intervention_name,
                    i.health_condition,
                    i.correlation_type,
                    i.correlation_strength,
                    i.confidence_score,
                    i.sample_size,
                    i.paper_id
                FROM interventions i
                WHERE i.health_condition = ? OR i.health_condition = ?
                """

                cursor = conn.execute(query, (condition, search_condition))
                rows = cursor.fetchall()

                # Group by canonical intervention names in Python
                grouped_data = defaultdict(lambda: {
                    'studies': [],
                    'original_terms': set(),
                    'canonical_id': None
                })

                for row in rows:
                    intervention_info = self.get_display_info(row['intervention_name'], 'intervention')
                    canonical_name = intervention_info['canonical_name']
                    canonical_id = intervention_info['canonical_id']

                    grouped_data[canonical_name]['studies'].append({
                        'correlation_type': row['correlation_type'],
                        'correlation_strength': row['correlation_strength'],
                        'confidence_score': row['confidence_score'],
                        'sample_size': row['sample_size'],
                        'paper_id': row['paper_id']
                    })
                    grouped_data[canonical_name]['original_terms'].add(row['intervention_name'])
                    if canonical_id:
                        grouped_data[canonical_name]['canonical_id'] = canonical_id

                # Process grouped data into results
                results = []
                for canonical_name, data in grouped_data.items():
                    studies = data['studies']
                    study_count = len(studies)
                    positive_studies = sum(1 for s in studies if s['correlation_type'] == 'positive')
                    negative_studies = sum(1 for s in studies if s['correlation_type'] == 'negative')
                    neutral_studies = sum(1 for s in studies if s['correlation_type'] == 'neutral')
                    avg_correlation_strength = sum(s['correlation_strength'] for s in studies if s['correlation_strength']) / max(study_count, 1)
                    avg_confidence_score = sum(s['confidence_score'] for s in studies if s['confidence_score']) / max(study_count, 1)
                    max_sample_size = max((s['sample_size'] for s in studies if s['sample_size']), default=0)
                    unique_papers = len(set(s['paper_id'] for s in studies if s['paper_id']))

                    result = {
                        'intervention': canonical_name,
                        'canonical_id': data['canonical_id'],
                        'study_count': study_count,
                        'positive_studies': positive_studies,
                        'negative_studies': negative_studies,
                        'neutral_studies': neutral_studies,
                        'avg_correlation_strength': round(avg_correlation_strength, 3),
                        'avg_confidence_score': round(avg_confidence_score, 3),
                        'max_sample_size': max_sample_size,
                        'unique_papers': unique_papers,
                        'original_terms': sorted(list(data['original_terms'])),
                        'is_grouped': len(data['original_terms']) > 1
                    }
                    results.append(result)

                # Sort results
                results.sort(key=lambda x: (x['positive_studies'], x['avg_correlation_strength'], x['study_count']), reverse=True)
                return results[:limit]

            else:
                # Standard query without normalization
                query = """
                SELECT
                    i.intervention_name as intervention_canonical,
                    NULL as canonical_id,
                    COUNT(*) as study_count,
                    AVG(i.correlation_strength) as avg_correlation_strength,
                    AVG(i.confidence_score) as avg_confidence_score,
                    SUM(CASE WHEN i.correlation_type = 'positive' THEN 1 ELSE 0 END) as positive_studies,
                    SUM(CASE WHEN i.correlation_type = 'negative' THEN 1 ELSE 0 END) as negative_studies,
                    SUM(CASE WHEN i.correlation_type = 'neutral' THEN 1 ELSE 0 END) as neutral_studies,
                    i.intervention_name as original_terms,
                    MAX(i.sample_size) as max_sample_size,
                    COUNT(DISTINCT i.paper_id) as unique_papers
                FROM interventions i
                WHERE i.health_condition = ?
                GROUP BY i.intervention_name
                ORDER BY positive_studies DESC, avg_correlation_strength DESC, study_count DESC
                LIMIT ?
                """

                cursor = conn.execute(query, (condition, limit))

            results = []
            for row in cursor.fetchall():
                result = {
                    'intervention': row['intervention_canonical'],
                    'canonical_id': row['canonical_id'],
                    'study_count': row['study_count'],
                    'positive_studies': row['positive_studies'],
                    'negative_studies': row['negative_studies'],
                    'neutral_studies': row['neutral_studies'],
                    'avg_correlation_strength': round(row['avg_correlation_strength'] or 0, 3),
                    'avg_confidence_score': round(row['avg_confidence_score'] or 0, 3),
                    'max_sample_size': row['max_sample_size'],
                    'unique_papers': row['unique_papers'],
                    'original_terms': row['original_terms'].split(',') if row['original_terms'] else [],
                    'is_grouped': self.use_normalization and ',' in (row['original_terms'] or '')
                }
                results.append(result)

            return results

    def export_summary_statistics(self) -> Dict[str, Any]:
        """Export summary statistics with normalization info"""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Basic statistics
            stats_query = """
            SELECT
                COUNT(*) as total_interventions,
                COUNT(DISTINCT intervention_name) as unique_intervention_names,
                COUNT(DISTINCT health_condition) as unique_condition_names,
                COUNT(DISTINCT paper_id) as unique_papers,
                COUNT(CASE WHEN correlation_type = 'positive' THEN 1 END) as positive_correlations,
                COUNT(CASE WHEN correlation_type = 'negative' THEN 1 END) as negative_correlations,
                COUNT(CASE WHEN correlation_type = 'neutral' THEN 1 END) as neutral_correlations,
                COUNT(CASE WHEN correlation_type = 'inconclusive' THEN 1 END) as inconclusive_correlations,
                AVG(confidence_score) as avg_confidence,
                AVG(correlation_strength) as avg_correlation_strength
            FROM interventions
            """

            stats_result = conn.execute(stats_query).fetchone()
            stats = dict(stats_result)

            # Add normalization statistics if enabled
            if self.use_normalization:
                # Canonical entity counts
                canonical_stats_query = """
                SELECT
                    entity_type,
                    COUNT(*) as canonical_count
                FROM canonical_entities
                GROUP BY entity_type
                """

                canonical_stats = {}
                for row in conn.execute(canonical_stats_query):
                    canonical_stats[row['entity_type']] = row['canonical_count']

                # Mapping statistics
                mapping_stats_query = """
                SELECT
                    entity_type,
                    COUNT(*) as mapped_terms,
                    COUNT(DISTINCT canonical_id) as unique_canonicals
                FROM entity_mappings
                GROUP BY entity_type
                """

                mapping_stats = {}
                for row in conn.execute(mapping_stats_query):
                    mapping_stats[row['entity_type']] = {
                        'mapped_terms': row['mapped_terms'],
                        'unique_canonicals': row['unique_canonicals']
                    }

                stats['normalization'] = {
                    'enabled': True,
                    'canonical_entities': canonical_stats,
                    'mapping_coverage': mapping_stats,
                    'intervention_reduction': {
                        'original_unique': stats['unique_intervention_names'],
                        'canonical_unique': canonical_stats.get('intervention', 0),
                        'reduction_percent': round(
                            (1 - canonical_stats.get('intervention', stats['unique_intervention_names']) /
                             max(stats['unique_intervention_names'], 1)) * 100, 1
                        )
                    },
                    'condition_reduction': {
                        'original_unique': stats['unique_condition_names'],
                        'canonical_unique': canonical_stats.get('condition', 0),
                        'reduction_percent': round(
                            (1 - canonical_stats.get('condition', stats['unique_condition_names']) /
                             max(stats['unique_condition_names'], 1)) * 100, 1
                        )
                    }
                }
            else:
                stats['normalization'] = {'enabled': False}

            return stats


def get_database_path() -> str:
    """Get the path to the SQLite database."""
    return "data/processed/intervention_research.db"


def get_output_path() -> str:
    """Get the output path for JSON file."""
    script_dir = Path(__file__).parent
    output_path = script_dir.parent.parent / "frontend" / "data" / "correlations_enhanced.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def main():
    """Main export function with normalization integration"""

    db_path = get_database_path()

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    # Export with normalization enabled (feature flag)
    exporter = EnhancedDataExporter(db_path, use_normalization=True)

    print("Exporting enhanced correlations data with entity normalization...")

    # Export main data
    data = exporter.export_interventions_normalized()

    # Add summary statistics
    data['summary_stats'] = exporter.export_summary_statistics()

    # Test the grouping functionality
    print("\nTesting intervention grouping for 'irritable bowel syndrome':")
    top_interventions = exporter.get_top_interventions_for_condition('irritable bowel syndrome', limit=10)

    for i, intervention in enumerate(top_interventions, 1):
        grouped_indicator = " [GROUPED]" if intervention['is_grouped'] else ""
        print(f"  {i}. {intervention['intervention']}{grouped_indicator}")
        print(f"     Studies: {intervention['study_count']} (+{intervention['positive_studies']}, -{intervention['negative_studies']})")
        if intervention['original_terms']:
            print(f"     Original terms: {', '.join(intervention['original_terms'])}")
        print()

    data['sample_analysis'] = {
        'condition': 'irritable bowel syndrome',
        'top_interventions': top_interventions
    }

    # Save to file
    output_path = get_output_path()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nExport completed successfully!")
    print(f"Enhanced data exported to: {output_path}")
    print(f"Total interventions: {len(data['interventions'])}")

    # Show normalization benefits
    if data['summary_stats']['normalization']['enabled']:
        norm_stats = data['summary_stats']['normalization']
        print(f"\nNormalization Benefits:")
        print(f"  Intervention reduction: {norm_stats['intervention_reduction']['reduction_percent']:.1f}%")
        print(f"  Condition reduction: {norm_stats['condition_reduction']['reduction_percent']:.1f}%")
        print(f"  Original intervention names: {norm_stats['intervention_reduction']['original_unique']}")
        print(f"  Canonical interventions: {norm_stats['intervention_reduction']['canonical_unique']}")


if __name__ == "__main__":
    main()