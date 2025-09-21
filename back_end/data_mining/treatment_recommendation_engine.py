"""
Treatment Recommendation Engine.

Multi-layer recommendation system that integrates all pipeline components
to provide personalized treatment suggestions with exploration bonuses for
emerging treatments. This is where everything comes together.

Key Features:
- Multi-signal evidence scoring (direct, mechanism, similar conditions)
- Exploration bonus for emerging treatments
- Treatment classification system
- Confidence-weighted final scoring
- Integration of all 7 pipeline steps
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
import random
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.paper_collection.data_mining_repository import (
        DataMiningRepository,
        TreatmentRecommendation as DBTreatmentRecommendation
    )
    from src.data.config import setup_logging
except ImportError as e:
    print(f"Warning: Could not import database components: {e}")
    DataMiningRepository = None
    DBTreatmentRecommendation = None

logger = setup_logging(__name__, 'treatment_recommendation.log') if 'setup_logging' in globals() else None


@dataclass
class RecommendationResult:
    """Individual treatment recommendation with scoring details (internal use)."""
    intervention: str
    final_score: float
    confidence: float
    evidence_type: str
    evidence_count: int
    classification: str
    explanation: str
    base_score: float
    exploration_bonus: float
    mechanism_matches: List[str]
    similar_conditions: List[str]
    risk_factors: List[str]


class TreatmentRecommendationEngine:
    """
    Multi-layer treatment recommendation system that integrates all pipeline
    components to provide personalized treatment suggestions.

    Architecture:
    1. Evidence Scoring (direct, mechanism, similar conditions)
    2. Exploration Bonus (for emerging treatments)
    3. Classification System (established, emerging_promising, moderate_evidence)
    4. Final Score Integration with confidence weighting
    5. Failed intervention filtering
    """

    def __init__(self,
                 explore_rate: float = 0.20,
                 min_confidence_threshold: float = 0.3,
                 diversity_bonus: float = 0.1,
                 save_to_database: bool = True,
                 generation_model: str = "recommendation_engine_v1"):
        """
        Initialize treatment recommendation engine.

        Args:
            explore_rate: Exploration bonus rate for emerging treatments (default 20%)
            min_confidence_threshold: Minimum confidence for recommendations
            diversity_bonus: Bonus for treatment diversity
            save_to_database: Whether to save recommendations to database
            generation_model: Model identifier for tracking recommendation versions
        """
        self.explore_rate = explore_rate
        self.min_confidence_threshold = min_confidence_threshold
        self.diversity_bonus = diversity_bonus
        self.save_to_database = save_to_database
        self.generation_model = generation_model

        # Initialize database repository if available
        self.repository = None
        if save_to_database and DataMiningRepository:
            try:
                self.repository = DataMiningRepository()
            except Exception as e:
                if logger:
                    logger.warning(f"Could not initialize database repository: {e}")
                self.save_to_database = False

        # Evidence type weights
        self.evidence_weights = {
            'direct': 1.0,           # Direct evidence for condition
            'mechanism': 0.7,        # Mechanism-based evidence
            'similar_condition': 0.6, # Evidence from similar conditions
            'combination': 0.8,       # Evidence from combinations
            'fundamental': 0.9        # Evidence from fundamental functions
        }

        # Classification thresholds
        self.classification_thresholds = {
            'established': {'min_evidence': 8, 'min_confidence': 0.7},
            'emerging_promising': {'min_evidence': 2, 'min_confidence': 0.4},
            'moderate_evidence': {'min_evidence': 4, 'min_confidence': 0.5},
            'experimental': {'min_evidence': 1, 'min_confidence': 0.2}
        }

    def recommend_treatments(self,
                           condition: str,
                           knowledge_graph,
                           bayesian_scorer,
                           biological_patterns,
                           fundamental_functions,
                           research_gaps,
                           power_combinations,
                           failed_interventions,
                           patient_profile: Optional[Dict] = None,
                           top_n: int = 10) -> List[RecommendationResult]:
        """
        Generate treatment recommendations for a given condition.

        Args:
            condition: Target medical condition
            knowledge_graph: MedicalKnowledgeGraph instance
            bayesian_scorer: BayesianEvidenceScorer instance
            biological_patterns: BiologicalPatternDiscovery instance
            fundamental_functions: FundamentalFunctionDiscovery instance
            research_gaps: ResearchGapIdentification instance
            power_combinations: PowerCombinationAnalyzer instance
            failed_interventions: FailedInterventionCatalog instance
            patient_profile: Optional patient-specific information
            top_n: Number of recommendations to return

        Returns:
            List of TreatmentRecommendation objects sorted by final score
        """

        # Get all possible interventions from multiple sources
        candidate_interventions = self._gather_candidate_interventions(
            condition, knowledge_graph, research_gaps, power_combinations
        )

        # Score each intervention using multiple evidence pathways
        scored_recommendations = []

        for intervention in candidate_interventions:
            # Skip failed interventions
            if intervention in failed_interventions:
                continue

            recommendation = self._score_intervention(
                intervention, condition, knowledge_graph, bayesian_scorer,
                biological_patterns, fundamental_functions, research_gaps,
                power_combinations, patient_profile
            )

            if recommendation and recommendation.confidence >= self.min_confidence_threshold:
                scored_recommendations.append(recommendation)

        # Apply exploration bonuses
        scored_recommendations = self._apply_exploration_bonuses(scored_recommendations)

        # Sort by final score and return top N
        sorted_recommendations = sorted(
            scored_recommendations,
            key=lambda x: x.final_score,
            reverse=True
        )

        top_recommendations = sorted_recommendations[:top_n]

        # Save recommendations to database if enabled
        if self.save_to_database and self.repository and DBTreatmentRecommendation:
            try:
                self._save_recommendations_to_database(condition, top_recommendations)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save recommendations to database: {e}")

        return top_recommendations

    def _gather_candidate_interventions(self,
                                      condition: str,
                                      knowledge_graph,
                                      research_gaps,
                                      power_combinations) -> Set[str]:
        """Gather candidate interventions from multiple sources."""
        candidates = set()

        # 1. Direct evidence from knowledge graph
        if condition in knowledge_graph.backward_edges:
            candidates.update(knowledge_graph.backward_edges[condition].keys())

        # 2. Predicted interventions from research gaps
        try:
            gap_predictions = research_gaps.predict_intervention_effectiveness(condition)
            for pred in gap_predictions[:20]:  # Top 20 predictions
                candidates.add(pred['intervention'])
        except:
            pass

        # 3. Power combinations
        try:
            combinations = power_combinations.get_combinations_for_condition(condition)
            for combo_name, combo_data in combinations.items():
                candidates.add(combo_name)
        except:
            pass

        # 4. Common interventions (fallback)
        common_interventions = [
            'exercise', 'meditation', 'omega_3', 'vitamin_d', 'magnesium',
            'probiotics', 'CBT', 'sleep_hygiene', 'stress_reduction',
            'anti_inflammatory_diet', 'yoga', 'physical_therapy'
        ]
        candidates.update(common_interventions)

        return candidates

    def _score_intervention(self,
                          intervention: str,
                          condition: str,
                          knowledge_graph,
                          bayesian_scorer,
                          biological_patterns,
                          fundamental_functions,
                          research_gaps,
                          power_combinations,
                          patient_profile: Optional[Dict] = None) -> Optional[RecommendationResult]:
        """Score an individual intervention using multiple evidence pathways."""

        evidence_scores = {}
        evidence_counts = {}
        explanations = []
        mechanism_matches = []
        similar_conditions = []

        # 1. Direct Evidence Scoring
        direct_score, direct_confidence, direct_count = self._score_direct_evidence(
            intervention, condition, knowledge_graph, bayesian_scorer
        )
        if direct_score > 0:
            evidence_scores['direct'] = direct_score
            evidence_counts['direct'] = direct_count
            explanations.append(f"Direct evidence: {direct_count} studies, {direct_score:.0%} success rate")

        # 2. Mechanism-Based Evidence
        mechanism_score, mechanism_confidence, mechanisms = self._score_mechanism_evidence(
            intervention, condition, biological_patterns, knowledge_graph
        )
        if mechanism_score > 0:
            evidence_scores['mechanism'] = mechanism_score
            evidence_counts['mechanism'] = len(mechanisms)
            mechanism_matches.extend(mechanisms)
            explanations.append(f"Targets {mechanisms[0]} mechanism | Early results very positive")

        # 3. Similar Condition Evidence
        similar_score, similar_confidence, similar_conds = self._score_similar_condition_evidence(
            intervention, condition, knowledge_graph, bayesian_scorer
        )
        if similar_score > 0:
            evidence_scores['similar_condition'] = similar_score
            evidence_counts['similar_condition'] = len(similar_conds)
            similar_conditions.extend(similar_conds)
            explanations.append(f"Works for similar {' and '.join(similar_conds[:2])}")

        # 4. Fundamental Function Evidence
        fundamental_score, fundamental_confidence = self._score_fundamental_evidence(
            intervention, condition, fundamental_functions
        )
        if fundamental_score > 0:
            evidence_scores['fundamental'] = fundamental_score
            explanations.append(f"Affects fundamental body functions")

        # 5. Combination Evidence
        combination_score, combination_confidence = self._score_combination_evidence(
            intervention, condition, power_combinations
        )
        if combination_score > 0:
            evidence_scores['combination'] = combination_score
            explanations.append(f"Synergistic effects identified")

        # Calculate weighted base score
        if not evidence_scores:
            return None

        base_score = self._calculate_weighted_base_score(evidence_scores)

        # Calculate overall confidence
        all_confidences = [direct_confidence, mechanism_confidence, similar_confidence,
                          fundamental_confidence, combination_confidence]
        overall_confidence = np.mean([c for c in all_confidences if c > 0])

        # Determine total evidence count
        total_evidence = sum(evidence_counts.values())

        # Determine evidence type (primary source)
        primary_evidence_type = max(evidence_scores.keys(), key=lambda k: evidence_scores[k])

        # Classify intervention
        classification = self._classify_intervention(total_evidence, overall_confidence)

        # Create recommendation
        recommendation = RecommendationResult(
            intervention=intervention,
            final_score=base_score,  # Will be updated with exploration bonus
            confidence=overall_confidence,
            evidence_type=primary_evidence_type,
            evidence_count=total_evidence,
            classification=classification,
            explanation=' | '.join(explanations[:2]),  # Top 2 explanations
            base_score=base_score,
            exploration_bonus=0.0,  # Will be calculated later
            mechanism_matches=mechanism_matches,
            similar_conditions=similar_conditions,
            risk_factors=[]
        )

        return recommendation

    def _score_direct_evidence(self, intervention: str, condition: str,
                             knowledge_graph, bayesian_scorer) -> Tuple[float, float, int]:
        """Score direct evidence for intervention-condition pair."""
        try:
            if condition in knowledge_graph.backward_edges:
                if intervention in knowledge_graph.backward_edges[condition]:
                    edges = knowledge_graph.backward_edges[condition][intervention]

                    # Use Bayesian scorer if available
                    if bayesian_scorer:
                        score_result = bayesian_scorer.score_intervention(intervention, condition)
                        return score_result['score'], score_result['confidence'], len(edges)
                    else:
                        # Fallback scoring
                        positive_edges = [e for e in edges if e.evidence.evidence_type == 'positive']
                        if edges:
                            success_rate = len(positive_edges) / len(edges)
                            confidence = np.mean([e.evidence.confidence for e in edges])
                            return success_rate, confidence, len(edges)
        except:
            pass

        return 0.0, 0.0, 0

    def _score_mechanism_evidence(self, intervention: str, condition: str,
                                biological_patterns, knowledge_graph) -> Tuple[float, float, List[str]]:
        """Score mechanism-based evidence."""
        try:
            # Get mechanisms for condition
            condition_mechanisms = biological_patterns.get_condition_mechanisms(condition)
            intervention_mechanisms = biological_patterns.get_intervention_mechanisms(intervention)

            # Find matching mechanisms
            matching_mechanisms = list(set(condition_mechanisms) & set(intervention_mechanisms))

            if matching_mechanisms:
                # Score based on mechanism overlap and strength
                mechanism_score = len(matching_mechanisms) / max(len(condition_mechanisms), 1)
                mechanism_confidence = 0.6  # Moderate confidence for mechanism evidence
                return min(mechanism_score, 0.8), mechanism_confidence, matching_mechanisms
        except:
            pass

        return 0.0, 0.0, []

    def _score_similar_condition_evidence(self, intervention: str, condition: str,
                                        knowledge_graph, bayesian_scorer) -> Tuple[float, float, List[str]]:
        """Score evidence from similar conditions."""
        try:
            # Define condition similarity mapping
            condition_similarity = {
                'fibromyalgia': ['chronic_fatigue', 'chronic_pain', 'arthritis'],
                'depression': ['anxiety', 'chronic_fatigue', 'PTSD'],
                'anxiety': ['depression', 'PTSD', 'panic_disorder'],
                'chronic_pain': ['fibromyalgia', 'arthritis', 'neuropathy'],
                'arthritis': ['fibromyalgia', 'chronic_pain', 'inflammation']
            }

            similar_conditions = condition_similarity.get(condition, [])
            evidence_scores = []
            found_conditions = []

            for similar_condition in similar_conditions:
                score, confidence, count = self._score_direct_evidence(
                    intervention, similar_condition, knowledge_graph, bayesian_scorer
                )
                if score > 0:
                    evidence_scores.append(score * 0.8)  # Discount for similarity
                    found_conditions.append(similar_condition)

            if evidence_scores:
                avg_score = np.mean(evidence_scores)
                avg_confidence = 0.5  # Moderate confidence for similar conditions
                return avg_score, avg_confidence, found_conditions
        except:
            pass

        return 0.0, 0.0, []

    def _score_fundamental_evidence(self, intervention: str, condition: str,
                                  fundamental_functions) -> Tuple[float, float]:
        """Score fundamental function evidence."""
        try:
            # Check if intervention is a fundamental function
            fundamental_interventions = fundamental_functions.get_fundamental_interventions()

            if intervention in fundamental_interventions:
                fundamental_data = fundamental_interventions[intervention]
                mechanism_count = len(fundamental_data.get('mechanisms', []))
                cross_condition_score = fundamental_data.get('cross_condition_effectiveness', 0)

                # Score based on mechanism breadth and cross-condition effectiveness
                score = min(0.7, mechanism_count * 0.1 + cross_condition_score * 0.5)
                confidence = 0.65
                return score, confidence
        except:
            pass

        return 0.0, 0.0

    def _score_combination_evidence(self, intervention: str, condition: str,
                                  power_combinations) -> Tuple[float, float]:
        """Score combination/synergy evidence."""
        try:
            # Check if intervention is part of effective combinations
            combinations = power_combinations.get_combinations_for_condition(condition)

            for combo_name, combo_data in combinations.items():
                if intervention in combo_name.split('_'):
                    lift_score = combo_data.get('lift', 1.0)
                    if lift_score > 1.2:  # Meaningful synergy
                        score = min(0.6, (lift_score - 1) * 0.5)
                        confidence = 0.55
                        return score, confidence
        except:
            pass

        return 0.0, 0.0

    def _calculate_weighted_base_score(self, evidence_scores: Dict[str, float]) -> float:
        """Calculate weighted base score from multiple evidence types."""
        weighted_sum = 0
        total_weight = 0

        for evidence_type, score in evidence_scores.items():
            weight = self.evidence_weights.get(evidence_type, 0.5)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0

    def _classify_intervention(self, evidence_count: int, confidence: float) -> str:
        """Classify intervention based on evidence quantity and quality."""

        for classification, thresholds in self.classification_thresholds.items():
            if (evidence_count >= thresholds['min_evidence'] and
                confidence >= thresholds['min_confidence']):
                return classification

        return 'experimental'

    def _apply_exploration_bonuses(self, recommendations: List[RecommendationResult]) -> List[RecommendationResult]:
        """Apply exploration bonuses to emerging treatments."""

        for recommendation in recommendations:
            exploration_bonus = 0.0

            # Exploration bonus for emerging promising treatments
            if recommendation.classification == 'emerging_promising':
                exploration_bonus = self.explore_rate * random.uniform(0.8, 1.2)

            # Smaller bonus for experimental treatments
            elif recommendation.classification == 'experimental':
                exploration_bonus = self.explore_rate * 0.5 * random.uniform(0.9, 1.1)

            # Mechanism-based bonus
            if recommendation.evidence_type == 'mechanism' and len(recommendation.mechanism_matches) > 1:
                exploration_bonus += 0.05

            # Update final score
            recommendation.exploration_bonus = exploration_bonus
            recommendation.final_score = min(1.0, recommendation.base_score + exploration_bonus)

        return recommendations

    def get_treatment_explanation(self, recommendation: RecommendationResult) -> str:
        """Generate detailed explanation for a treatment recommendation."""

        explanation_parts = []

        # Classification context
        classification_descriptions = {
            'established': 'Well-established treatment with strong evidence',
            'emerging_promising': 'Promising emerging treatment with positive early results',
            'moderate_evidence': 'Moderate evidence base, worth considering',
            'experimental': 'Experimental treatment, limited evidence'
        }

        explanation_parts.append(classification_descriptions.get(
            recommendation.classification, 'Treatment under evaluation'))

        # Evidence details
        explanation_parts.append(f"Confidence: {recommendation.confidence:.0%}")
        explanation_parts.append(f"Evidence count: {recommendation.evidence_count}")

        # Mechanism information
        if recommendation.mechanism_matches:
            explanation_parts.append(f"Targets: {', '.join(recommendation.mechanism_matches[:2])}")

        # Similar condition evidence
        if recommendation.similar_conditions:
            explanation_parts.append(f"Also effective for: {', '.join(recommendation.similar_conditions[:2])}")

        # Exploration bonus explanation
        if recommendation.exploration_bonus > 0:
            explanation_parts.append(f"Exploration bonus: +{recommendation.exploration_bonus:.0%}")

        return ' | '.join(explanation_parts)

    def analyze_recommendation_diversity(self, recommendations: List[RecommendationResult]) -> Dict[str, Any]:
        """Analyze diversity of recommendations."""

        # Classification diversity
        classification_counts = Counter([r.classification for r in recommendations])

        # Evidence type diversity
        evidence_type_counts = Counter([r.evidence_type for r in recommendations])

        # Mechanism diversity
        all_mechanisms = []
        for r in recommendations:
            all_mechanisms.extend(r.mechanism_matches)
        mechanism_counts = Counter(all_mechanisms)

        return {
            'total_recommendations': len(recommendations),
            'classification_distribution': dict(classification_counts),
            'evidence_type_distribution': dict(evidence_type_counts),
            'mechanism_coverage': len(set(all_mechanisms)),
            'top_mechanisms': dict(mechanism_counts.most_common(5)),
            'avg_confidence': np.mean([r.confidence for r in recommendations]),
            'exploration_rate_applied': np.mean([r.exploration_bonus for r in recommendations])
        }

    def validate_recommendations(self, recommendations: List[RecommendationResult]) -> Dict[str, Any]:
        """Validate recommendation quality and consistency."""

        issues = []

        # Check for score consistency
        for rec in recommendations:
            if rec.final_score < rec.base_score - 0.01:  # Allow small floating point errors
                issues.append(f"{rec.intervention}: Final score lower than base score")

            if rec.confidence > 1.0 or rec.confidence < 0.0:
                issues.append(f"{rec.intervention}: Invalid confidence value")

            if rec.evidence_count < 0:
                issues.append(f"{rec.intervention}: Negative evidence count")

        # Check for proper sorting
        scores = [r.final_score for r in recommendations]
        if scores != sorted(scores, reverse=True):
            issues.append("Recommendations not properly sorted by final score")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'total_recommendations': len(recommendations),
            'score_range': (min(scores) if scores else 0, max(scores) if scores else 0)
        }

    def _save_recommendations_to_database(self, condition: str,
                                        recommendations: List[RecommendationResult]) -> None:
        """Save treatment recommendations to the database."""
        try:
            for rank, rec in enumerate(recommendations, 1):
                # Map evidence strength based on classification and confidence
                evidence_strength_mapping = {
                    'established': 'strong' if rec.confidence > 0.7 else 'moderate',
                    'emerging_promising': 'moderate' if rec.confidence > 0.6 else 'weak',
                    'moderate_evidence': 'moderate',
                    'experimental': 'weak'
                }
                evidence_strength = evidence_strength_mapping.get(rec.classification, 'weak')

                # Create database recommendation object
                db_recommendation = DBTreatmentRecommendation(
                    condition_name=condition,
                    recommended_intervention=rec.intervention,
                    recommendation_rank=rank,
                    confidence_score=rec.confidence,
                    evidence_strength=evidence_strength,
                    supporting_studies_count=rec.evidence_count,
                    recommendation_rationale=rec.explanation,
                    generation_model=self.generation_model,
                    model_version="1.0",
                    generated_at=datetime.now()
                )

                # Add optional fields if available
                if hasattr(rec, 'mechanism_matches') and rec.mechanism_matches:
                    db_recommendation.contraindications = f"Based on mechanisms: {', '.join(rec.mechanism_matches[:3])}"

                if hasattr(rec, 'similar_conditions') and rec.similar_conditions:
                    db_recommendation.population_specificity = f"Evidence from similar conditions: {', '.join(rec.similar_conditions[:3])}"

                if hasattr(rec, 'risk_factors') and rec.risk_factors:
                    db_recommendation.interaction_warnings = f"Consider risk factors: {', '.join(rec.risk_factors[:3])}"

                # Save to database
                self.repository.save_treatment_recommendation(db_recommendation)

            if logger:
                logger.info(f"Saved {len(recommendations)} recommendations for {condition}")

        except Exception as e:
            if logger:
                logger.error(f"Error saving recommendations to database: {e}")
            raise


def create_demo_data():
    """Create demonstration data for testing."""

    # Mock knowledge graph data
    class MockKnowledgeGraph:
        def __init__(self):
            self.backward_edges = {
                'fibromyalgia': {
                    'exercise': [self._create_mock_edge('positive', 0.75)],
                    'magnesium': [self._create_mock_edge('positive', 0.60)],
                    'CBT': [self._create_mock_edge('positive', 0.70)]
                },
                'chronic_pain': {
                    'exercise': [self._create_mock_edge('positive', 0.80)],
                    'CBT': [self._create_mock_edge('positive', 0.75)]
                },
                'chronic_fatigue': {
                    'exercise': [self._create_mock_edge('positive', 0.65)],
                    'CBT': [self._create_mock_edge('positive', 0.70)]
                }
            }

        def _create_mock_edge(self, evidence_type, confidence):
            class MockEvidence:
                def __init__(self, evidence_type, confidence):
                    self.evidence_type = evidence_type
                    self.confidence = confidence
                    self.sample_size = 100
                    self.study_design = 'RCT'
                    self.study_id = 'mock_study'

            class MockEdge:
                def __init__(self, evidence):
                    self.evidence = evidence

            return MockEdge(MockEvidence(evidence_type, confidence))

    # Mock Bayesian scorer
    class MockBayesianScorer:
        def score_intervention(self, intervention, condition):
            # Mock scoring based on intervention
            scores = {
                'exercise': {'score': 0.75, 'confidence': 0.85},
                'magnesium': {'score': 0.60, 'confidence': 0.45},
                'CBT': {'score': 0.70, 'confidence': 0.70}
            }
            return scores.get(intervention, {'score': 0.5, 'confidence': 0.4})

    # Mock biological patterns
    class MockBiologicalPatterns:
        def get_condition_mechanisms(self, condition):
            mechanisms = {
                'fibromyalgia': ['pain_mechanism', 'inflammation', 'stress_response'],
                'chronic_pain': ['pain_mechanism', 'inflammation'],
                'depression': ['neurotransmitter', 'stress_response']
            }
            return mechanisms.get(condition, [])

        def get_intervention_mechanisms(self, intervention):
            mechanisms = {
                'exercise': ['inflammation', 'stress_response', 'neurotransmitter'],
                'magnesium': ['pain_mechanism', 'stress_response'],
                'CBT': ['stress_response', 'neurotransmitter']
            }
            return mechanisms.get(intervention, [])

    # Mock fundamental functions
    class MockFundamentalFunctions:
        def get_fundamental_interventions(self):
            return {
                'exercise': {
                    'mechanisms': ['inflammation', 'metabolism', 'stress_response'],
                    'cross_condition_effectiveness': 0.8
                }
            }

    # Mock research gaps
    class MockResearchGaps:
        def predict_intervention_effectiveness(self, condition):
            return [
                {'intervention': 'omega_3', 'predicted_score': 0.65},
                {'intervention': 'meditation', 'predicted_score': 0.60}
            ]

    # Mock power combinations
    class MockPowerCombinations:
        def get_combinations_for_condition(self, condition):
            return {
                'magnesium_vitamin_d': {'lift': 1.3, 'confidence': 0.6}
            }

    return (MockKnowledgeGraph(), MockBayesianScorer(), MockBiologicalPatterns(),
            MockFundamentalFunctions(), MockResearchGaps(), MockPowerCombinations(), set())