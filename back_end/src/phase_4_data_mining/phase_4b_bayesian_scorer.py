"""
Phase 4b: Bayesian Evidence Scoring for Canonical Groups.

Scores canonical groups from Phase 3 using Bayesian statistics,
pooling evidence across cluster members for better statistical power.

Key Features:
- Integrates with Phase 3 clustering (canonical groups)
- Pools evidence across cluster members (better statistical power)
- Solves innovation penalty problem (statistical confidence, not raw counts)
- Beta distribution priors (uniform by default)
- Conservative scoring (10th percentile)
- Database integration (saves Bayesian scores)

Migration from standalone data_mining/bayesian_scorer.py to Phase 4b.
"""

import math
import sys
from pathlib import Path
from typing import Dict, Union, Tuple, Optional, List
from scipy import stats
import numpy as np
from datetime import datetime

# Import shared utilities
from .scoring_utils import EffectivenessScorer, ConfidenceCalculator, ScoringResult, StatisticalHelpers

try:
    from back_end.src.phase_1_data_collection.data_mining_repository import DataMiningRepository, BayesianScore
    from back_end.src.data.config import setup_logging, config
except ImportError as e:
    print(f"Warning: Could not import database components: {e}")
    DataMiningRepository = None
    BayesianScore = None
    config = None

logger = setup_logging(__name__, 'bayesian_scorer.log') if 'setup_logging' in globals() else None


class BayesianEvidenceScorer:
    """
    Unified scoring system that handles the innovation penalty problem
    using Bayesian statistics with Beta distribution priors.
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0,
                 save_to_database: bool = True, analysis_model: str = "bayesian_v1"):
        """
        Initialize the Bayesian scorer with Beta distribution priors.

        Args:
            alpha_prior: Prior for positive outcomes (default: 1.0 = uniform prior)
            beta_prior: Prior for negative outcomes (default: 1.0 = uniform prior)
            save_to_database: Whether to save analysis results to database
            analysis_model: Model identifier for tracking analysis versions
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.save_to_database = save_to_database
        self.analysis_model = analysis_model

        # Initialize database repository if available
        self.repository = None
        if save_to_database and DataMiningRepository:
            try:
                self.repository = DataMiningRepository()
            except Exception as e:
                if logger:
                    logger.warning(f"Could not initialize database repository: {e}")
                self.save_to_database = False

    def score_intervention(
        self,
        positive: Union[int, str],
        negative: Union[int, str, None] = None,
        neutral: int = 0,
        knowledge_graph = None
    ) -> Dict[str, float]:
        """
        Score an intervention using Bayesian evidence aggregation.

        Two usage modes:
        1. Direct counts: score_intervention(positive_count, negative_count, neutral_count)
        2. Knowledge graph: score_intervention(intervention_name, condition_name, knowledge_graph=kg)

        Args:
            positive: Number of positive outcomes OR intervention name (if str)
            negative: Number of negative outcomes OR condition name (if str) OR None
            neutral: Number of neutral study outcomes (excluded from scoring)
            knowledge_graph: MedicalKnowledgeGraph instance (required for mode 2)

        Returns:
            Dictionary containing:
            - score: Bayesian posterior mean
            - conservative_score: 10th percentile (worst-case scenario)
            - confidence: Statistical confidence measure
            - evidence_count: Total evidence count
        """
        # Handle two different calling conventions
        if isinstance(positive, str):
            # Mode 2: Knowledge graph mode
            if knowledge_graph is None:
                raise ValueError("knowledge_graph is required when using intervention/condition names")
            intervention_name = positive
            condition_name = negative
            if condition_name is None:
                raise ValueError("condition_name is required in knowledge graph mode")

            # Extract evidence counts from knowledge graph
            pos_count, neg_count, neut_count = self._extract_evidence_counts(
                intervention_name, condition_name, knowledge_graph
            )
            positive, negative, neutral = pos_count, neg_count, neut_count

        # Mode 1: Direct counts mode
        if positive < 0 or negative < 0 or neutral < 0:
            raise ValueError("All counts must be non-negative")

        total_evidence = positive + negative + neutral
        relevant_evidence = positive + negative

        if relevant_evidence == 0:
            return {
                'score': 0.5,  # No evidence = neutral
                'conservative_score': 0.5,
                'confidence': 0.0,
                'evidence_count': total_evidence
            }

        # Beta distribution parameters (Bayesian update)
        alpha_posterior = self.alpha_prior + positive
        beta_posterior = self.beta_prior + negative

        # Posterior mean (Bayesian score)
        score = alpha_posterior / (alpha_posterior + beta_posterior)

        # Conservative score (10th percentile)
        conservative_score = stats.beta.ppf(0.1, alpha_posterior, beta_posterior)

        # Confidence measure based on evidence and uncertainty
        confidence = self._calculate_confidence(
            alpha_posterior, beta_posterior, relevant_evidence
        )

        result = {
            'score': round(score, 2),
            'conservative_score': round(conservative_score, 2),
            'confidence': round(confidence, 2),
            'evidence_count': total_evidence
        }

        # Save to database if we have intervention/condition names and database is available
        if (isinstance(positive, str) and self.save_to_database and
            self.repository and BayesianScore):
            try:
                self._save_bayesian_score(
                    intervention_name=intervention_name,
                    condition_name=condition_name,
                    positive_count=pos_count,
                    negative_count=neg_count,
                    neutral_count=neut_count,
                    alpha_posterior=alpha_posterior,
                    beta_posterior=beta_posterior,
                    result=result
                )
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save Bayesian score to database: {e}")

        return result

    def _calculate_confidence(
        self,
        alpha: float,
        beta: float,
        evidence_count: int
    ) -> float:
        """
        Calculate confidence based on posterior distribution and evidence count.

        Combines:
        1. Evidence quantity (more evidence = higher confidence)
        2. Posterior certainty (lower variance = higher confidence)

        Args:
            alpha: Beta distribution alpha parameter
            beta: Beta distribution beta parameter
            evidence_count: Total relevant evidence count

        Returns:
            Confidence score between 0 and 1
        """
        # Use shared confidence calculator
        calc = ConfidenceCalculator()

        # Evidence-based confidence using shared utility
        evidence_confidence = calc.sample_size_confidence(
            evidence_count, target_size=100, method='logarithmic'
        )

        # Posterior precision (inverse of variance)
        # Higher precision = lower uncertainty = higher confidence
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        precision_confidence = 1 - min(variance * 10, 1.0)  # Scale and cap at 1

        # Combined confidence (geometric mean)
        confidence = math.sqrt(evidence_confidence * precision_confidence)

        return min(confidence, 1.0)

    def compare_interventions(
        self,
        intervention_a: Dict[str, int],
        intervention_b: Dict[str, int]
    ) -> Dict[str, Union[str, float, Dict]]:
        """
        Compare two interventions and determine which has better evidence.

        Args:
            intervention_a: Dict with 'positive', 'negative', 'neutral' counts
            intervention_b: Dict with 'positive', 'negative', 'negative' counts

        Returns:
            Comparison results with recommendations
        """
        score_a = self.score_intervention(**intervention_a)
        score_b = self.score_intervention(**intervention_b)

        # Determine winner based on confidence-weighted scoring
        weighted_score_a = score_a['score'] * score_a['confidence']
        weighted_score_b = score_b['score'] * score_b['confidence']

        if weighted_score_a > weighted_score_b:
            winner = 'A'
            confidence_diff = weighted_score_a - weighted_score_b
        elif weighted_score_b > weighted_score_a:
            winner = 'B'
            confidence_diff = weighted_score_b - weighted_score_a
        else:
            winner = 'Tie'
            confidence_diff = 0.0

        return {
            'winner': winner,
            'confidence_difference': round(confidence_diff, 3),
            'intervention_a_scores': score_a,
            'intervention_b_scores': score_b,
            'recommendation': self._generate_recommendation(score_a, score_b, winner)
        }

    def _generate_recommendation(
        self,
        score_a: Dict,
        score_b: Dict,
        winner: str
    ) -> str:
        """Generate human-readable recommendation based on comparison."""
        if winner == 'Tie':
            return "Both interventions have similar evidence quality. Consider other factors."

        winner_scores = score_a if winner == 'A' else score_b
        loser_scores = score_b if winner == 'A' else score_a

        if winner_scores['confidence'] > 0.7 and loser_scores['confidence'] < 0.3:
            return f"Strong evidence favors intervention {winner}. High confidence recommendation."
        elif winner_scores['confidence'] < 0.5:
            return f"Intervention {winner} shows promise but needs more evidence."
        else:
            return f"Moderate evidence favors intervention {winner}. Consider risk tolerance."

    def _extract_evidence_counts(self, intervention: str, condition: str, knowledge_graph) -> Tuple[int, int, int]:
        """
        Extract evidence counts from knowledge graph for intervention-condition pair.

        Args:
            intervention: Intervention name
            condition: Condition name
            knowledge_graph: MedicalKnowledgeGraph instance

        Returns:
            Tuple of (positive_count, negative_count, neutral_count)
        """
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        # Access edges through the knowledge graph API
        # Check if the knowledge graph has a backward_edges property or method
        if hasattr(knowledge_graph, 'backward_edges'):
            edges_data = knowledge_graph.backward_edges
        elif hasattr(knowledge_graph, 'reverse_edges'):
            edges_data = knowledge_graph.reverse_edges
        else:
            # Try to use the query API
            try:
                treatments = knowledge_graph.query_treatments_for_condition(condition)
                for treatment in treatments:
                    if treatment.get('intervention') == intervention:
                        # Extract counts from aggregated data
                        evidence_details = treatment.get('evidence', [])
                        for detail in evidence_details:
                            evidence_type = detail.get('evidence_type', 'neutral')
                            if evidence_type == 'positive':
                                positive_count += 1
                            elif evidence_type == 'negative':
                                negative_count += 1
                            elif evidence_type == 'neutral':
                                neutral_count += 1
                        return positive_count, negative_count, neutral_count
            except:
                pass

        # Direct access to edges data
        if edges_data and condition in edges_data and intervention in edges_data[condition]:
            edges = edges_data[condition][intervention]
            for edge in edges:
                evidence_type = edge.evidence.evidence_type
                if evidence_type == 'positive':
                    positive_count += 1
                elif evidence_type == 'negative':
                    negative_count += 1
                elif evidence_type == 'neutral':
                    neutral_count += 1

        return positive_count, negative_count, neutral_count

    def _save_bayesian_score(self, intervention_name: str, condition_name: str,
                           positive_count: int, negative_count: int, neutral_count: int,
                           alpha_posterior: float, beta_posterior: float,
                           result: Dict) -> None:
        """Save Bayesian analysis results to the database."""
        try:
            # Calculate credible interval (95% by default)
            credible_lower = stats.beta.ppf(0.025, alpha_posterior, beta_posterior)
            credible_upper = stats.beta.ppf(0.975, alpha_posterior, beta_posterior)

            # Calculate Bayes factor (relative to neutral hypothesis)
            # Using evidence strength relative to no effect
            bayes_factor = self._calculate_bayes_factor(
                positive_count, negative_count, alpha_posterior, beta_posterior
            )

            bayesian_score = BayesianScore(
                intervention_name=intervention_name,
                condition_name=condition_name,
                posterior_mean=result['score'],
                posterior_variance=self._calculate_posterior_variance(alpha_posterior, beta_posterior),
                credible_interval_lower=credible_lower,
                credible_interval_upper=credible_upper,
                bayes_factor=bayes_factor,
                positive_evidence_count=positive_count,
                negative_evidence_count=negative_count,
                neutral_evidence_count=neutral_count,
                total_studies=positive_count + negative_count + neutral_count,
                alpha_prior=self.alpha_prior,
                beta_prior=self.beta_prior,
                confidence_adjusted_score=result['conservative_score'],
                analysis_model=self.analysis_model,
                analysis_timestamp=datetime.now()
            )

            self.repository.save_bayesian_score(bayesian_score)

            if logger:
                if config and not config.fast_mode:
                    logger.info(f"Saved Bayesian score for {intervention_name} -> {condition_name}")

        except Exception as e:
            if logger:
                logger.error(f"Error saving Bayesian score: {e}")
            raise

    def _calculate_posterior_variance(self, alpha: float, beta: float) -> float:
        """Calculate posterior variance for Beta distribution."""
        return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    def _calculate_bayes_factor(self, positive: int, negative: int,
                              alpha_posterior: float, beta_posterior: float) -> float:
        """Calculate Bayes factor comparing evidence strength to neutral hypothesis."""
        if positive + negative == 0:
            return 1.0  # No evidence = no preference

        # Simple Bayes factor based on posterior vs uniform prior
        # Higher values indicate stronger evidence for effectiveness
        posterior_odds = alpha_posterior / beta_posterior
        prior_odds = self.alpha_prior / self.beta_prior

        return min(posterior_odds / prior_odds, 100.0)  # Cap at 100 for numerical stability

    def score_all_interventions_for_condition(self, condition: str,
                                            knowledge_graph=None,
                                            min_evidence: int = 1) -> Dict[str, Dict]:
        """
        Score all interventions for a given condition and save to database.

        Args:
            condition: Health condition name
            knowledge_graph: MedicalKnowledgeGraph instance
            min_evidence: Minimum evidence count to include intervention

        Returns:
            Dictionary mapping intervention names to scores
        """
        if not knowledge_graph:
            raise ValueError("knowledge_graph is required")

        results = {}

        try:
            # Get all interventions for this condition from knowledge graph
            interventions = knowledge_graph.get_interventions_for_condition(condition)

            for intervention in interventions:
                intervention_name = intervention if isinstance(intervention, str) else intervention.get('name', str(intervention))

                try:
                    # Get evidence counts
                    pos_count, neg_count, neut_count = self._extract_evidence_counts(
                        intervention_name, condition, knowledge_graph
                    )

                    total_evidence = pos_count + neg_count + neut_count
                    if total_evidence >= min_evidence:
                        score_result = self.score_intervention(
                            intervention_name, condition, knowledge_graph=knowledge_graph
                        )
                        results[intervention_name] = score_result

                except Exception as e:
                    if logger:
                        logger.warning(f"Error scoring {intervention_name} for {condition}: {e}")
                    continue

        except Exception as e:
            if logger:
                logger.error(f"Error scoring interventions for {condition}: {e}")

        return results

    # =================================================================
    # PHASE 3 INTEGRATION - Canonical Group Scoring
    # =================================================================

    def score_canonical_group(
        self,
        canonical_group_id: int,
        condition: str,
        db_path: str,
        knowledge_graph=None
    ) -> Dict[str, float]:
        """
        Score a canonical group (cluster from Phase 3) for a condition.

        Pools evidence across all cluster members for better statistical power.

        Args:
            canonical_group_id: ID of canonical group from Phase 3
            condition: Health condition
            db_path: Path to intervention_research.db
            knowledge_graph: Optional knowledge graph (for extracting evidence)

        Returns:
            Bayesian score dictionary
        """
        import sqlite3

        if logger:
            logger.debug(f"Scoring canonical group {canonical_group_id} for {condition}")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Get canonical group name
            cursor.execute("""
                SELECT canonical_name
                FROM canonical_groups
                WHERE id = ?
            """, (canonical_group_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Canonical group {canonical_group_id} not found")

            canonical_name = row['canonical_name']

            # Get all cluster members using layer_1_canonical
            cursor.execute("""
                SELECT entity_name
                FROM semantic_hierarchy
                WHERE layer_1_canonical = ? AND entity_type = 'intervention'
            """, (canonical_name,))
            members = [r['entity_name'] for r in cursor.fetchall()]

            if not members:
                if logger:
                    logger.warning(f"No members found for canonical group {canonical_group_id}")
                return {
                    'score': 0.5,
                    'conservative_score': 0.5,
                    'confidence': 0.0,
                    'evidence_count': 0,
                    'canonical_name': canonical_name,
                    'cluster_members': []
                }

            # Aggregate evidence counts across all members
            total_positive = 0
            total_negative = 0
            total_neutral = 0

            for member in members:
                if knowledge_graph:
                    # Use knowledge graph if provided
                    pos, neg, neut = self._extract_evidence_counts(
                        member, condition, knowledge_graph
                    )
                else:
                    # Query database directly
                    pos, neg, neut = self._extract_evidence_counts_from_db(
                        member, condition, cursor
                    )

                total_positive += pos
                total_negative += neg
                total_neutral += neut

            # Score with pooled evidence
            result = self.score_intervention(total_positive, total_negative, total_neutral)

            # Add metadata
            result['canonical_name'] = canonical_name
            result['cluster_members'] = members
            result['cluster_size'] = len(members)

            # Save to database manually (since we used direct counts mode)
            if self.save_to_database and self.repository and BayesianScore and total_positive + total_negative > 0:
                try:
                    # Calculate alpha and beta posteriors for saving
                    alpha_posterior = self.alpha_prior + total_positive
                    beta_posterior = self.beta_prior + total_negative

                    self._save_bayesian_score(
                        intervention_name=canonical_name,  # Use canonical name
                        condition_name=condition,
                        positive_count=total_positive,
                        negative_count=total_negative,
                        neutral_count=total_neutral,
                        alpha_posterior=alpha_posterior,
                        beta_posterior=beta_posterior,
                        result=result
                    )
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to save Bayesian score for {canonical_name} -> {condition}: {e}")

            if logger:
                logger.debug(f"Scored {canonical_name}: {result['score']:.2f} (evidence: {result['evidence_count']})")

            return result

        finally:
            conn.close()

    def _extract_evidence_counts_from_db(
        self,
        intervention_name: str,
        condition: str,
        cursor
    ) -> Tuple[int, int, int]:
        """
        Extract evidence counts directly from database.

        Args:
            intervention_name: Raw intervention name
            condition: Condition name
            cursor: Database cursor

        Returns:
            Tuple of (positive_count, negative_count, neutral_count)
        """
        cursor.execute("""
            SELECT
                i.correlation_type as evidence_type,
                COUNT(*) as count
            FROM interventions i
            WHERE i.intervention_name = ? AND i.health_condition = ?
            GROUP BY i.correlation_type
        """, (intervention_name, condition))

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for row in cursor.fetchall():
            evidence_type = row['evidence_type'].lower() if row['evidence_type'] else 'neutral'
            count = row['count']

            if 'positive' in evidence_type:
                positive_count += count
            elif 'negative' in evidence_type:
                negative_count += count
            else:
                neutral_count += count

        return positive_count, negative_count, neutral_count

    def score_all_canonical_groups(
        self,
        knowledge_graph,
        db_path: str,
        min_evidence: int = 1
    ) -> Dict[str, Dict]:
        """
        Score all canonical groups from Phase 3.

        Args:
            knowledge_graph: MedicalKnowledgeGraph instance
            db_path: Path to intervention_research.db
            min_evidence: Minimum evidence count to include

        Returns:
            Dictionary mapping canonical group names to scores
        """
        import sqlite3

        if logger:
            logger.info("Scoring all canonical groups from Phase 3...")

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        results = {}
        scored_count = 0

        try:
            # Get all intervention canonical groups
            cursor.execute("""
                SELECT DISTINCT cg.id, cg.canonical_name
                FROM canonical_groups cg
                WHERE cg.entity_type = 'intervention'
                ORDER BY cg.canonical_name
            """)
            canonical_groups = cursor.fetchall()

            if logger:
                logger.info(f"Found {len(canonical_groups)} canonical groups to score")

            # Get all unique conditions from knowledge graph
            all_conditions = knowledge_graph.get_all_conditions()

            # Score each canonical group against each condition
            for group in canonical_groups:
                canonical_id = group['id']
                canonical_name = group['canonical_name']

                group_scores = {}

                for condition in all_conditions:
                    try:
                        score_result = self.score_canonical_group(
                            canonical_group_id=canonical_id,
                            condition=condition,
                            db_path=db_path,
                            knowledge_graph=knowledge_graph
                        )

                        # Only include if meets minimum evidence threshold
                        if score_result['evidence_count'] >= min_evidence:
                            group_scores[condition] = score_result
                            scored_count += 1

                    except Exception as e:
                        if logger:
                            logger.warning(f"Error scoring {canonical_name} for {condition}: {e}")
                        continue

                if group_scores:
                    results[canonical_name] = group_scores

                if len(results) % 50 == 0 and logger:
                    logger.info(f"Scored {len(results)}/{len(canonical_groups)} canonical groups...")

            if logger:
                logger.info(f"Scoring complete: {scored_count} total scores generated")

            return results

        finally:
            conn.close()

    def get_interventions_for_condition(self, condition: str, db_path: str) -> List[str]:
        """
        Get list of interventions (canonical groups) for a condition.

        Args:
            condition: Condition name
            db_path: Path to intervention_research.db

        Returns:
            List of intervention names
        """
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # Use layer_1_canonical instead of canonical_id join
            cursor.execute("""
                SELECT DISTINCT cg.canonical_name
                FROM canonical_groups cg
                JOIN semantic_hierarchy sh ON cg.canonical_name = sh.layer_1_canonical
                JOIN interventions i ON sh.entity_name = i.intervention_name
                WHERE i.health_condition = ? AND cg.entity_type = 'intervention' AND sh.entity_type = 'intervention'
            """, (condition,))

            return [row[0] for row in cursor.fetchall()]

        finally:
            conn.close()