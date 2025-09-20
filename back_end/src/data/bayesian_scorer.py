"""
Bayesian Evidence Aggregation System for Intervention Scoring.

Solves the innovation penalty problem by using statistical confidence
rather than raw counts, ensuring new treatments with limited evidence
aren't automatically penalized.
"""

import math
from typing import Dict, Union, Tuple
from scipy import stats
import numpy as np


class BayesianEvidenceScorer:
    """
    Unified scoring system that handles the innovation penalty problem
    using Bayesian statistics with Beta distribution priors.
    """

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize the Bayesian scorer with Beta distribution priors.

        Args:
            alpha_prior: Prior for positive outcomes (default: 1.0 = uniform prior)
            beta_prior: Prior for negative outcomes (default: 1.0 = uniform prior)
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

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

        return {
            'score': round(score, 2),
            'conservative_score': round(conservative_score, 2),
            'confidence': round(confidence, 2),
            'evidence_count': total_evidence
        }

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
        # Evidence-based confidence (asymptotic to 1)
        evidence_confidence = 1 - math.exp(-evidence_count / 20)

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