"""
Phase 5 Table View Exporter

Exports intervention research data to JSON for frontend DataTables display.
Refactored from back_end/src/utils/export_frontend_data.py
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from back_end.src.data.config import setup_logging
from .phase_5_base_exporter import BaseExporter

logger = setup_logging(__name__, 'phase_5_table_view.log')


class TableViewExporter(BaseExporter):
    """
    Export interventions data for frontend table view.

    Joins:
    - interventions
    - papers
    - semantic_hierarchy
    - canonical_groups
    - bayesian_scores
    - intervention_category_mapping
    - condition_category_mapping
    - mechanism_clusters
    """

    def __init__(self, db_path: str = None, config_path: str = None):
        super().__init__(db_path=db_path, config_path=config_path, export_type="table_view")

    def extract_data(self) -> Dict[str, Any]:
        """Extract interventions data with all relationships."""
        conn = self.get_database_connection()
        cursor = conn.cursor()

        try:
            # Main query: interventions with all joins
            query = """
            SELECT
                i.id,
                i.intervention_name,
                i.intervention_category,
                i.intervention_details,
                i.health_condition,
                i.condition_category,
                i.mechanism,
                i.outcome_type,
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
            ORDER BY bs.posterior_mean DESC NULLS LAST, i.study_confidence DESC NULLS LAST
            """

            cursor.execute(query)
            intervention_rows = cursor.fetchall()

            # Get summary statistics
            cursor.execute("SELECT COUNT(*) FROM interventions")
            total_interventions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT intervention_name) FROM interventions")
            unique_interventions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT health_condition) FROM interventions")
            unique_conditions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT paper_id) FROM interventions")
            unique_papers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM interventions WHERE outcome_type = 'positive'")
            positive_correlations = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM interventions WHERE outcome_type = 'negative'")
            negative_correlations = cursor.fetchone()[0]

            # Semantic hierarchy stats
            cursor.execute("SELECT COUNT(*) FROM semantic_hierarchy WHERE entity_type = 'intervention'")
            semantic_interventions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT layer_1_canonical) FROM semantic_hierarchy WHERE entity_type = 'intervention' AND layer_1_canonical IS NOT NULL")
            canonical_groups = cursor.fetchone()[0]

            # Top interventions by frequency
            cursor.execute("""
                SELECT
                    COALESCE(sh.layer_1_canonical, i.intervention_name) as name,
                    i.intervention_category,
                    COUNT(*) as count,
                    COUNT(DISTINCT i.paper_id) as paper_count
                FROM interventions i
                LEFT JOIN semantic_hierarchy sh ON i.intervention_name = sh.entity_name AND sh.entity_type = 'intervention'
                WHERE i.outcome_type = 'positive'
                GROUP BY COALESCE(sh.layer_1_canonical, i.intervention_name), i.intervention_category
                ORDER BY count DESC
                LIMIT 10
            """)
            top_interventions = [dict(row) for row in cursor.fetchall()]

            # Top conditions by frequency
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

            # Category breakdown
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

            # Multi-category statistics
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

                cursor.execute("""
                    SELECT COUNT(DISTINCT intervention_id) as count
                    FROM intervention_category_mapping
                    GROUP BY intervention_id
                    HAVING COUNT(*) > 1
                """)
                multi_category_interventions = len(cursor.fetchall())
            except sqlite3.OperationalError:
                pass

            # Bayesian score statistics
            cursor.execute("SELECT COUNT(*) FROM bayesian_scores")
            bayesian_scores_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM bayesian_scores WHERE posterior_mean > 0.7")
            high_bayesian_scores = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM bayesian_scores WHERE posterior_mean > 0.5")
            medium_bayesian_scores = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT intervention_name || '::' || condition_name) FROM bayesian_scores")
            total_relationships = cursor.fetchone()[0]

            return {
                'intervention_rows': intervention_rows,
                'statistics': {
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
                    'multi_category_stats': multi_category_stats,
                    'multi_category_interventions': multi_category_interventions,
                    'bayesian_scores_available': bayesian_scores_count > 0,
                    'total_relationships': total_relationships if bayesian_scores_count > 0 else canonical_groups,
                    'high_scoring_interventions': high_bayesian_scores,
                    'medium_scoring_interventions': medium_bayesian_scores
                },
                'top_performers': {
                    'interventions': top_interventions,
                    'conditions': top_conditions
                }
            }

        finally:
            conn.close()

    def transform_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw database rows into frontend JSON format."""
        interventions = []
        conn = self.get_database_connection()
        cursor = conn.cursor()

        try:
            for row in raw_data['intervention_rows']:
                # Get multi-category data
                intervention_categories = self._get_entity_categories(cursor, 'intervention', row['id'])
                condition_categories = self._get_entity_categories(cursor, 'condition', row['health_condition'])

                # Get mechanism canonical names
                mechanism_canonical_names = self._get_mechanism_canonical_names(cursor, row['id'])

                intervention = {
                    'id': row['id'],
                    'intervention': {
                        'name': row['intervention_name'],
                        'canonical_name': row['intervention_canonical_name'],
                        'category': row['intervention_category'],
                        'categories': intervention_categories,
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
                        'categories': condition_categories,
                        'severity': row['severity'],
                        'hierarchy': {
                            'layer_0_category': row['condition_l0_category'],
                            'layer_1_canonical': row['condition_l1_canonical'],
                            'layer_2_variant': row['condition_l2_variant'],
                            'layer_3_detail': row['condition_l3_detail']
                        }
                    },
                    'mechanism': row['mechanism'],
                    'mechanism_canonical_names': mechanism_canonical_names,
                    'correlation': {
                        'type': row['outcome_type'],
                        'study_confidence': row['study_confidence']
                    },
                    'bayesian_scoring': {
                        'score': row['bayesian_score'],
                        'conservative_score': row['bayesian_conservative_score'],
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

            return {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    **raw_data['statistics']
                },
                'top_performers': raw_data['top_performers'],
                'interventions': interventions
            }

        finally:
            conn.close()

    def _get_entity_categories(self, cursor, entity_type: str, entity_id: Any) -> Dict[str, List[str]]:
        """Get all categories for an entity organized by type."""
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
            categories_by_type = {}
            for row in rows:
                cat_type = row['category_type']
                cat_name = row['category_name']
                if cat_type not in categories_by_type:
                    categories_by_type[cat_type] = []
                categories_by_type[cat_type].append(cat_name)

            return categories_by_type
        except sqlite3.OperationalError:
            return {}

    def _get_mechanism_canonical_names(self, cursor, intervention_id: int) -> List[str]:
        """Get mechanism canonical names for an intervention."""
        try:
            cursor.execute("""
                SELECT DISTINCT mc.canonical_name
                FROM intervention_mechanisms im
                JOIN mechanism_clusters mc ON im.cluster_id = mc.cluster_id
                WHERE im.intervention_id = ?
                ORDER BY mc.canonical_name
            """, (intervention_id,))
            results = cursor.fetchall()
            return [row[0] for row in results] if results else []
        except sqlite3.OperationalError:
            return []

    def _get_output_path(self) -> Path:
        """Get output path for table view JSON."""
        return self.resolve_output_path('table_view')

    def _count_records(self, data: Dict[str, Any]) -> int:
        """Count intervention records."""
        return len(data.get('intervention_rows', []))
