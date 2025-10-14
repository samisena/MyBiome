"""
Data Mining Repository - Database access layer for data mining results

Provides high-level interfaces for storing and retrieving data mining insights
including knowledge graphs, Bayesian scores, recommendations, and analytics.

Features:
- Type-safe data models
- Efficient batch operations
- Relationship management
- Query optimization
- Data validation
"""

import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, date
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging

from back_end.src.data.config import setup_logging
from back_end.src.phase_1_data_collection.database_manager import database_manager

logger = setup_logging(__name__, 'data_mining_repository.log')


@dataclass
class KnowledgeGraphNode:
    """Represents a node in the medical knowledge graph."""
    node_id: str
    node_type: str  # 'intervention', 'condition', 'study'
    node_name: str
    node_data: Dict[str, Any] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.node_data is None:
            self.node_data = {}


@dataclass
class KnowledgeGraphEdge:
    """Represents an edge in the medical knowledge graph."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str  # 'treats', 'causes', 'correlates', 'prevents'
    edge_weight: float = 1.0
    evidence_type: str = 'neutral'  # 'positive', 'negative', 'neutral', 'unsure'
    confidence: float = 0.5
    study_id: Optional[str] = None
    study_title: Optional[str] = None
    sample_size: Optional[int] = None
    study_design: Optional[str] = None
    publication_year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    effect_size: Optional[float] = None
    p_value: Optional[float] = None
    generation_model: Optional[str] = None
    generation_version: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class BayesianScore:
    """Bayesian analysis results for intervention-condition pairs."""
    intervention_name: str
    condition_name: str
    posterior_mean: float
    posterior_variance: float
    credible_interval_lower: float
    credible_interval_upper: float
    bayes_factor: float
    positive_evidence_count: int = 0
    negative_evidence_count: int = 0
    neutral_evidence_count: int = 0
    total_studies: int = 0
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    innovation_penalty_adjusted: bool = False
    confidence_adjusted_score: Optional[float] = None
    analysis_model: Optional[str] = None
    data_snapshot_id: Optional[str] = None
    id: Optional[int] = None
    analysis_timestamp: Optional[datetime] = None


@dataclass
class TreatmentRecommendation:
    """AI-generated treatment recommendation."""
    condition_name: str
    recommended_intervention: str
    recommendation_rank: int
    confidence_score: float
    evidence_strength: str
    supporting_studies_count: int
    average_effect_size: Optional[float] = None
    consistency_score: Optional[float] = None
    recommendation_rationale: Optional[str] = None
    contraindications: Optional[str] = None
    optimal_dosage: Optional[str] = None
    duration_recommendation: Optional[str] = None
    safety_profile: Optional[str] = None
    side_effects: Optional[List[str]] = None
    interaction_warnings: Optional[str] = None
    population_specificity: Optional[str] = None
    age_restrictions: Optional[str] = None
    comorbidity_considerations: Optional[str] = None
    generation_model: Optional[str] = None
    model_version: Optional[str] = None
    data_version: Optional[str] = None
    human_reviewed: bool = False
    reviewer_notes: Optional[str] = None
    id: Optional[int] = None
    generated_at: Optional[datetime] = None


@dataclass
class ResearchGap:
    """Identified research gap or opportunity."""
    gap_type: str  # 'intervention_gap', 'condition_gap', 'population_gap', 'methodology_gap'
    gap_name: str
    gap_description: str
    intervention_name: Optional[str] = None
    condition_name: Optional[str] = None
    population_demographic: Optional[str] = None
    priority_score: float = 5.0
    urgency_level: str = 'medium'
    current_study_count: int = 0
    ideal_study_count: Optional[int] = None
    evidence_quality_score: Optional[float] = None
    suggested_study_design: Optional[str] = None
    suggested_sample_size: Optional[int] = None
    estimated_cost_category: Optional[str] = None
    potential_impact_score: Optional[float] = None
    innovation_opportunity: Optional[str] = None
    technology_requirements: Optional[str] = None
    collaboration_opportunities: Optional[str] = None
    identification_method: Optional[str] = None
    analysis_model: Optional[str] = None
    validation_status: str = 'pending'
    id: Optional[int] = None
    identified_at: Optional[datetime] = None


class DataMiningRepository:
    """High-level repository for data mining results."""

    def __init__(self):
        self.db_manager = database_manager

    @contextmanager
    def get_connection(self):
        """Get database connection."""
        with self.db_manager.get_connection() as conn:
            yield conn

    # =================================================================
    # KNOWLEDGE GRAPH OPERATIONS
    # =================================================================

    def save_knowledge_graph_node(self, node: KnowledgeGraphNode) -> int:
        """Save a knowledge graph node."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_graph_nodes
                (node_id, node_type, node_name, node_data)
                VALUES (?, ?, ?, ?)
            """, (
                node.node_id,
                node.node_type,
                node.node_name,
                json.dumps(node.node_data) if node.node_data else None
            ))

            conn.commit()
            return cursor.lastrowid

    def save_knowledge_graph_edge(self, edge: KnowledgeGraphEdge) -> int:
        """Save a knowledge graph edge."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_graph_edges
                (edge_id, source_node_id, target_node_id, edge_type, edge_weight,
                 evidence_type, confidence, study_id, study_title, sample_size,
                 study_design, publication_year, journal, doi, effect_size,
                 p_value, generation_model, generation_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.edge_id, edge.source_node_id, edge.target_node_id,
                edge.edge_type, edge.edge_weight, edge.evidence_type,
                edge.confidence, edge.study_id, edge.study_title,
                edge.sample_size, edge.study_design, edge.publication_year,
                edge.journal, edge.doi, edge.effect_size, edge.p_value,
                edge.generation_model, edge.generation_version
            ))

            conn.commit()
            return cursor.lastrowid

    def get_knowledge_graph_nodes(self, node_type: Optional[str] = None) -> List[KnowledgeGraphNode]:
        """Get knowledge graph nodes, optionally filtered by type."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if node_type:
                cursor.execute("""
                    SELECT * FROM knowledge_graph_nodes
                    WHERE node_type = ?
                    ORDER BY node_name
                """, (node_type,))
            else:
                cursor.execute("""
                    SELECT * FROM knowledge_graph_nodes
                    ORDER BY node_type, node_name
                """)

            nodes = []
            for row in cursor.fetchall():
                node_data = json.loads(row['node_data']) if row['node_data'] else {}
                nodes.append(KnowledgeGraphNode(
                    id=row['id'],
                    node_id=row['node_id'],
                    node_type=row['node_type'],
                    node_name=row['node_name'],
                    node_data=node_data,
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
                ))

            return nodes

    def get_knowledge_graph_edges(self, source_node_id: Optional[str] = None,
                                 target_node_id: Optional[str] = None,
                                 edge_type: Optional[str] = None) -> List[KnowledgeGraphEdge]:
        """Get knowledge graph edges with optional filtering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            conditions = []
            params = []

            if source_node_id:
                conditions.append("source_node_id = ?")
                params.append(source_node_id)

            if target_node_id:
                conditions.append("target_node_id = ?")
                params.append(target_node_id)

            if edge_type:
                conditions.append("edge_type = ?")
                params.append(edge_type)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(f"""
                SELECT * FROM knowledge_graph_edges
                {where_clause}
                ORDER BY edge_weight DESC, confidence DESC
            """, params)

            edges = []
            for row in cursor.fetchall():
                edges.append(KnowledgeGraphEdge(
                    id=row['id'],
                    edge_id=row['edge_id'],
                    source_node_id=row['source_node_id'],
                    target_node_id=row['target_node_id'],
                    edge_type=row['edge_type'],
                    edge_weight=row['edge_weight'],
                    evidence_type=row['evidence_type'],
                    confidence=row['confidence'],
                    study_id=row['study_id'],
                    study_title=row['study_title'],
                    sample_size=row['sample_size'],
                    study_design=row['study_design'],
                    publication_year=row['publication_year'],
                    journal=row['journal'],
                    doi=row['doi'],
                    effect_size=row['effect_size'],
                    p_value=row['p_value'],
                    generation_model=row['generation_model'],
                    generation_version=row['generation_version'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
                ))

            return edges

    # =================================================================
    # BAYESIAN SCORING OPERATIONS
    # =================================================================

    def save_bayesian_score(self, score: BayesianScore) -> int:
        """Save Bayesian analysis results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO bayesian_scores
                (intervention_name, condition_name, posterior_mean, posterior_variance,
                 credible_interval_lower, credible_interval_upper, bayes_factor,
                 positive_evidence_count, negative_evidence_count, neutral_evidence_count,
                 total_studies, alpha_prior, beta_prior, innovation_penalty_adjusted,
                 confidence_adjusted_score, analysis_model, data_snapshot_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                score.intervention_name, score.condition_name,
                score.posterior_mean, score.posterior_variance,
                score.credible_interval_lower, score.credible_interval_upper,
                score.bayes_factor, score.positive_evidence_count,
                score.negative_evidence_count, score.neutral_evidence_count,
                score.total_studies, score.alpha_prior, score.beta_prior,
                score.innovation_penalty_adjusted, score.confidence_adjusted_score,
                score.analysis_model, score.data_snapshot_id
            ))

            conn.commit()
            return cursor.lastrowid

    def get_bayesian_scores(self, intervention_name: Optional[str] = None,
                          condition_name: Optional[str] = None,
                          min_bayes_factor: Optional[float] = None) -> List[BayesianScore]:
        """Get Bayesian scores with optional filtering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            conditions = []
            params = []

            if intervention_name:
                conditions.append("intervention_name = ?")
                params.append(intervention_name)

            if condition_name:
                conditions.append("condition_name = ?")
                params.append(condition_name)

            if min_bayes_factor is not None:
                conditions.append("bayes_factor >= ?")
                params.append(min_bayes_factor)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(f"""
                SELECT * FROM bayesian_scores
                {where_clause}
                ORDER BY bayes_factor DESC, posterior_mean DESC
            """, params)

            scores = []
            for row in cursor.fetchall():
                scores.append(BayesianScore(
                    id=row['id'],
                    intervention_name=row['intervention_name'],
                    condition_name=row['condition_name'],
                    posterior_mean=row['posterior_mean'],
                    posterior_variance=row['posterior_variance'],
                    credible_interval_lower=row['credible_interval_lower'],
                    credible_interval_upper=row['credible_interval_upper'],
                    bayes_factor=row['bayes_factor'],
                    positive_evidence_count=row['positive_evidence_count'],
                    negative_evidence_count=row['negative_evidence_count'],
                    neutral_evidence_count=row['neutral_evidence_count'],
                    total_studies=row['total_studies'],
                    alpha_prior=row['alpha_prior'],
                    beta_prior=row['beta_prior'],
                    innovation_penalty_adjusted=row['innovation_penalty_adjusted'],
                    confidence_adjusted_score=row['confidence_adjusted_score'],
                    analysis_model=row['analysis_model'],
                    data_snapshot_id=row['data_snapshot_id'],
                    analysis_timestamp=datetime.fromisoformat(row['analysis_timestamp']) if row['analysis_timestamp'] else None
                ))

            return scores

    # =================================================================
    # TREATMENT RECOMMENDATIONS OPERATIONS
    # =================================================================

    def save_treatment_recommendation(self, recommendation: TreatmentRecommendation) -> int:
        """Save a treatment recommendation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            side_effects_json = json.dumps(recommendation.side_effects) if recommendation.side_effects else None

            cursor.execute("""
                INSERT OR REPLACE INTO treatment_recommendations
                (condition_name, recommended_intervention, recommendation_rank,
                 confidence_score, evidence_strength, supporting_studies_count,
                 average_effect_size, consistency_score, recommendation_rationale,
                 contraindications, optimal_dosage, duration_recommendation,
                 safety_profile, side_effects, interaction_warnings,
                 population_specificity, age_restrictions, comorbidity_considerations,
                 generation_model, model_version, data_version, human_reviewed,
                 reviewer_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recommendation.condition_name, recommendation.recommended_intervention,
                recommendation.recommendation_rank, recommendation.confidence_score,
                recommendation.evidence_strength, recommendation.supporting_studies_count,
                recommendation.average_effect_size, recommendation.consistency_score,
                recommendation.recommendation_rationale, recommendation.contraindications,
                recommendation.optimal_dosage, recommendation.duration_recommendation,
                recommendation.safety_profile, side_effects_json,
                recommendation.interaction_warnings, recommendation.population_specificity,
                recommendation.age_restrictions, recommendation.comorbidity_considerations,
                recommendation.generation_model, recommendation.model_version,
                recommendation.data_version, recommendation.human_reviewed,
                recommendation.reviewer_notes
            ))

            conn.commit()
            return cursor.lastrowid

    def get_treatment_recommendations(self, condition_name: Optional[str] = None,
                                    min_confidence: Optional[float] = None,
                                    evidence_strength: Optional[str] = None) -> List[TreatmentRecommendation]:
        """Get treatment recommendations with optional filtering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            conditions = []
            params = []

            if condition_name:
                conditions.append("condition_name = ?")
                params.append(condition_name)

            if min_confidence is not None:
                conditions.append("confidence_score >= ?")
                params.append(min_confidence)

            if evidence_strength:
                conditions.append("evidence_strength = ?")
                params.append(evidence_strength)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(f"""
                SELECT * FROM treatment_recommendations
                {where_clause}
                ORDER BY condition_name, recommendation_rank
            """, params)

            recommendations = []
            for row in cursor.fetchall():
                side_effects = json.loads(row['side_effects']) if row['side_effects'] else None

                recommendations.append(TreatmentRecommendation(
                    id=row['id'],
                    condition_name=row['condition_name'],
                    recommended_intervention=row['recommended_intervention'],
                    recommendation_rank=row['recommendation_rank'],
                    confidence_score=row['confidence_score'],
                    evidence_strength=row['evidence_strength'],
                    supporting_studies_count=row['supporting_studies_count'],
                    average_effect_size=row['average_effect_size'],
                    consistency_score=row['consistency_score'],
                    recommendation_rationale=row['recommendation_rationale'],
                    contraindications=row['contraindications'],
                    optimal_dosage=row['optimal_dosage'],
                    duration_recommendation=row['duration_recommendation'],
                    safety_profile=row['safety_profile'],
                    side_effects=side_effects,
                    interaction_warnings=row['interaction_warnings'],
                    population_specificity=row['population_specificity'],
                    age_restrictions=row['age_restrictions'],
                    comorbidity_considerations=row['comorbidity_considerations'],
                    generation_model=row['generation_model'],
                    model_version=row['model_version'],
                    data_version=row['data_version'],
                    human_reviewed=row['human_reviewed'],
                    reviewer_notes=row['reviewer_notes'],
                    generated_at=datetime.fromisoformat(row['generated_at']) if row['generated_at'] else None
                ))

            return recommendations

    # =================================================================
    # RESEARCH GAPS OPERATIONS
    # =================================================================

    def save_research_gap(self, gap: ResearchGap) -> int:
        """Save a research gap."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO research_gaps
                (gap_type, gap_name, gap_description, intervention_name,
                 condition_name, population_demographic, priority_score,
                 urgency_level, current_study_count, ideal_study_count,
                 evidence_quality_score, suggested_study_design,
                 suggested_sample_size, estimated_cost_category,
                 potential_impact_score, innovation_opportunity,
                 technology_requirements, collaboration_opportunities,
                 identification_method, analysis_model, validation_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gap.gap_type, gap.gap_name, gap.gap_description,
                gap.intervention_name, gap.condition_name, gap.population_demographic,
                gap.priority_score, gap.urgency_level, gap.current_study_count,
                gap.ideal_study_count, gap.evidence_quality_score,
                gap.suggested_study_design, gap.suggested_sample_size,
                gap.estimated_cost_category, gap.potential_impact_score,
                gap.innovation_opportunity, gap.technology_requirements,
                gap.collaboration_opportunities, gap.identification_method,
                gap.analysis_model, gap.validation_status
            ))

            conn.commit()
            return cursor.lastrowid

    def get_research_gaps(self, gap_type: Optional[str] = None,
                         min_priority: Optional[float] = None,
                         urgency_level: Optional[str] = None) -> List[ResearchGap]:
        """Get research gaps with optional filtering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            conditions = []
            params = []

            if gap_type:
                conditions.append("gap_type = ?")
                params.append(gap_type)

            if min_priority is not None:
                conditions.append("priority_score >= ?")
                params.append(min_priority)

            if urgency_level:
                conditions.append("urgency_level = ?")
                params.append(urgency_level)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(f"""
                SELECT * FROM research_gaps
                {where_clause}
                ORDER BY priority_score DESC, potential_impact_score DESC
            """, params)

            gaps = []
            for row in cursor.fetchall():
                gaps.append(ResearchGap(
                    id=row['id'],
                    gap_type=row['gap_type'],
                    gap_name=row['gap_name'],
                    gap_description=row['gap_description'],
                    intervention_name=row['intervention_name'],
                    condition_name=row['condition_name'],
                    population_demographic=row['population_demographic'],
                    priority_score=row['priority_score'],
                    urgency_level=row['urgency_level'],
                    current_study_count=row['current_study_count'],
                    ideal_study_count=row['ideal_study_count'],
                    evidence_quality_score=row['evidence_quality_score'],
                    suggested_study_design=row['suggested_study_design'],
                    suggested_sample_size=row['suggested_sample_size'],
                    estimated_cost_category=row['estimated_cost_category'],
                    potential_impact_score=row['potential_impact_score'],
                    innovation_opportunity=row['innovation_opportunity'],
                    technology_requirements=row['technology_requirements'],
                    collaboration_opportunities=row['collaboration_opportunities'],
                    identification_method=row['identification_method'],
                    analysis_model=row['analysis_model'],
                    validation_status=row['validation_status'],
                    identified_at=datetime.fromisoformat(row['identified_at']) if row['identified_at'] else None
                ))

            return gaps

    # =================================================================
    # ANALYTICS AND REPORTING
    # =================================================================

    def get_data_mining_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for data mining results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Knowledge graph stats
            cursor.execute("SELECT COUNT(*) FROM knowledge_graph_nodes")
            stats['knowledge_graph_nodes'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM knowledge_graph_edges")
            stats['knowledge_graph_edges'] = cursor.fetchone()[0]

            # Bayesian scores stats
            cursor.execute("SELECT COUNT(*) FROM bayesian_scores")
            stats['bayesian_scores'] = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(bayes_factor) FROM bayesian_scores")
            avg_bayes = cursor.fetchone()[0]
            stats['average_bayes_factor'] = avg_bayes if avg_bayes else 0.0

            # Treatment recommendations stats
            cursor.execute("SELECT COUNT(*) FROM treatment_recommendations")
            stats['treatment_recommendations'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT condition_name) FROM treatment_recommendations")
            stats['conditions_with_recommendations'] = cursor.fetchone()[0]

            # Research gaps stats
            cursor.execute("SELECT COUNT(*) FROM research_gaps")
            stats['research_gaps'] = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(priority_score) FROM research_gaps")
            avg_priority = cursor.fetchone()[0]
            stats['average_gap_priority'] = avg_priority if avg_priority else 0.0

            # Top conditions by activity
            cursor.execute("""
                SELECT condition_name, COUNT(*) as activity_count
                FROM (
                    SELECT condition_name FROM bayesian_scores
                    UNION ALL
                    SELECT condition_name FROM treatment_recommendations
                    UNION ALL
                    SELECT condition_name FROM research_gaps WHERE condition_name IS NOT NULL
                ) combined
                GROUP BY condition_name
                ORDER BY activity_count DESC
                LIMIT 10
            """)
            stats['top_conditions'] = [
                {'condition': row[0], 'activity_count': row[1]}
                for row in cursor.fetchall()
            ]

            return stats

    def get_intervention_insights(self, intervention_name: str) -> Dict[str, Any]:
        """Get comprehensive insights for a specific intervention."""
        insights = {
            'intervention_name': intervention_name,
            'bayesian_scores': [],
            'recommendations': [],
            'research_gaps': [],
            'knowledge_graph_connections': []
        }

        # Get Bayesian scores
        insights['bayesian_scores'] = self.get_bayesian_scores(intervention_name=intervention_name)

        # Get recommendations
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM treatment_recommendations
                WHERE recommended_intervention = ?
                ORDER BY confidence_score DESC
            """, (intervention_name,))

            for row in cursor.fetchall():
                insights['recommendations'].append({
                    'condition': row['condition_name'],
                    'rank': row['recommendation_rank'],
                    'confidence': row['confidence_score'],
                    'evidence_strength': row['evidence_strength']
                })

        # Get related research gaps
        insights['research_gaps'] = self.get_research_gaps()
        insights['research_gaps'] = [
            gap for gap in insights['research_gaps']
            if gap.intervention_name == intervention_name
        ]

        # Get knowledge graph connections
        insights['knowledge_graph_connections'] = self.get_knowledge_graph_edges(
            source_node_id=intervention_name
        ) + self.get_knowledge_graph_edges(target_node_id=intervention_name)

        return insights

    def save_data_mining_session(self, session_id: str, session_type: str,
                                config_snapshot: Dict[str, Any],
                                start_time: datetime,
                                data_version: str = None,
                                models_used: List[str] = None) -> int:
        """Save data mining session metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO data_mining_sessions
                (session_id, session_type, config_snapshot, start_time,
                 data_version, models_used, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, session_type,
                json.dumps(config_snapshot),
                start_time.isoformat(),
                data_version,
                json.dumps(models_used) if models_used else None,
                'running'
            ))

            conn.commit()
            return cursor.lastrowid

    def update_data_mining_session(self, session_id: str,
                                 end_time: datetime = None,
                                 status: str = None,
                                 results_summary: Dict[str, Any] = None,
                                 error_message: str = None) -> bool:
        """Update data mining session with completion info."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            updates = []
            params = []

            if end_time:
                updates.append("end_time = ?")
                params.append(end_time.isoformat())

            if status:
                updates.append("status = ?")
                params.append(status)

            if results_summary:
                updates.append("results_summary = ?")
                params.append(json.dumps(results_summary))

            if error_message:
                updates.append("error_message = ?")
                params.append(error_message)

            if updates:
                params.append(session_id)
                cursor.execute(f"""
                    UPDATE data_mining_sessions
                    SET {', '.join(updates)}
                    WHERE session_id = ?
                """, params)

                conn.commit()
                return cursor.rowcount > 0

            return False


# Global repository instance
data_mining_repository = DataMiningRepository()