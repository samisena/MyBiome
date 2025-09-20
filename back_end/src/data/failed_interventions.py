"""
Failed Intervention Catalog System.

Tracks interventions with consistent negative results to avoid wasting
time, resources, and hope. Creates a "don't bother" database that
identifies treatments that keep getting tried despite consistent failure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import math


@dataclass
class FailedIntervention:
    """Intervention with consistent negative results."""
    intervention: str
    failure_rate: float
    confidence: float
    evidence_count: int
    waste_score: float
    recommendation: str
    failed_conditions: List[str]
    negative_studies: List[Dict[str, Any]]
    research_waste_estimate: float
    harm_potential: str
    alternative_suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format matching expected output."""
        return {
            'failure_rate': round(self.failure_rate, 2),
            'confidence': round(self.confidence, 2),
            'evidence_count': self.evidence_count,
            'waste_score': round(self.waste_score, 2),
            'recommendation': self.recommendation,
            'failed_conditions': self.failed_conditions,
            'harm_potential': self.harm_potential,
            'research_waste_estimate': round(self.research_waste_estimate, 1)
        }


class FailedInterventionCatalog:
    """
    Failed intervention catalog system that identifies interventions
    with consistent negative results to prevent waste of resources.

    Key approaches:
    1. Failure rate analysis (percentage of negative studies)
    2. Confidence assessment (statistical certainty)
    3. Waste score calculation (evidence count × failure rate)
    4. Two-tier recommendation system
    """

    def __init__(self,
                 min_failure_rate: float = 0.6,
                 min_evidence_count: int = 3,
                 avoid_confidence_threshold: float = 0.8):
        """
        Initialize failed intervention catalog system.

        Args:
            min_failure_rate: Minimum failure rate to be considered failed
            min_evidence_count: Minimum studies needed for assessment
            avoid_confidence_threshold: Confidence threshold for "AVOID" vs "UNLIKELY"
        """
        self.min_failure_rate = min_failure_rate
        self.min_evidence_count = min_evidence_count
        self.avoid_confidence_threshold = avoid_confidence_threshold

        # Known problematic interventions with historical context
        self.known_failed_interventions = {
            'homeopathy': {
                'expected_failure_rate': 0.85,
                'harm_potential': 'low_direct_high_opportunity',
                'research_waste': 'massive',
                'alternative_suggestions': ['evidence_based_medicine', 'placebo_counseling']
            },
            'magnetic_bracelets': {
                'expected_failure_rate': 0.78,
                'harm_potential': 'low',
                'research_waste': 'moderate',
                'alternative_suggestions': ['physical_therapy', 'exercise', 'proper_medical_treatment']
            },
            'copper_supplements_for_arthritis': {
                'expected_failure_rate': 0.72,
                'harm_potential': 'moderate',
                'research_waste': 'low_to_moderate',
                'alternative_suggestions': ['anti_inflammatory_medication', 'omega_3', 'exercise']
            },
            'detox_teas': {
                'expected_failure_rate': 0.80,
                'harm_potential': 'moderate',
                'research_waste': 'moderate',
                'alternative_suggestions': ['proper_nutrition', 'hydration', 'liver_support']
            },
            'alkaline_water_for_cancer': {
                'expected_failure_rate': 0.88,
                'harm_potential': 'very_high_opportunity_cost',
                'research_waste': 'high',
                'alternative_suggestions': ['evidence_based_oncology', 'clinical_trials']
            },
            'ear_candles': {
                'expected_failure_rate': 0.90,
                'harm_potential': 'moderate_direct',
                'research_waste': 'low',
                'alternative_suggestions': ['proper_ear_cleaning', 'medical_evaluation']
            },
            'crystal_healing': {
                'expected_failure_rate': 0.92,
                'harm_potential': 'low_direct_high_opportunity',
                'research_waste': 'low',
                'alternative_suggestions': ['meditation', 'therapy', 'evidence_based_treatments']
            }
        }

        # Intervention categories more likely to fail
        self.high_risk_categories = {
            'pseudoscience': ['homeopathy', 'crystal_healing', 'chakra_alignment'],
            'unproven_devices': ['magnetic_bracelets', 'ion_bracelets', 'balance_bracelets'],
            'questionable_supplements': ['detox_pills', 'fat_burner_pills', 'miracle_cures'],
            'dangerous_alternatives': ['alkaline_water_for_cancer', 'vitamin_c_megadose_for_cancer'],
            'ineffective_procedures': ['ear_candles', 'colon_cleansing', 'foot_detox']
        }

    def identify_failed_interventions(self,
                                    knowledge_graph,
                                    discovered_mechanisms: List,
                                    bayesian_scorer=None) -> Dict[str, FailedIntervention]:
        """
        Identify interventions with consistent negative results.

        Args:
            knowledge_graph: MedicalKnowledgeGraph instance
            discovered_mechanisms: List of BiologicalMechanism objects
            bayesian_scorer: BayesianEvidenceScorer instance (optional)

        Returns:
            Dictionary of failed interventions
        """
        # Analyze all interventions for failure patterns
        intervention_analysis = self._analyze_intervention_failures(knowledge_graph)

        # Filter and create failed intervention objects
        failed_interventions = {}

        for intervention_name, analysis in intervention_analysis.items():
            if self._meets_failure_criteria(analysis):
                failed_intervention = self._create_failed_intervention(
                    intervention_name, analysis, knowledge_graph, bayesian_scorer
                )
                failed_interventions[intervention_name] = failed_intervention

        # Sort by waste score (highest waste first)
        sorted_failures = dict(
            sorted(failed_interventions.items(),
                   key=lambda x: x[1].waste_score, reverse=True)
        )

        return sorted_failures

    def _analyze_intervention_failures(self, knowledge_graph) -> Dict[str, Dict[str, Any]]:
        """Analyze failure patterns for each intervention."""
        intervention_analysis = defaultdict(lambda: {
            'total_studies': 0,
            'negative_studies': 0,
            'neutral_studies': 0,
            'positive_studies': 0,
            'failed_conditions': [],
            'evidence_details': [],
            'avg_confidence': 0
        })

        # Analyze each intervention's track record
        for intervention in knowledge_graph.forward_edges:
            total_confidence = 0
            total_studies = 0

            for condition, edges in knowledge_graph.forward_edges[intervention].items():
                for edge in edges:
                    evidence_type = edge.evidence.evidence_type
                    confidence = edge.evidence.confidence

                    analysis = intervention_analysis[intervention]
                    analysis['total_studies'] += 1
                    total_studies += 1
                    total_confidence += confidence

                    # Categorize study results
                    if evidence_type == 'negative':
                        analysis['negative_studies'] += 1
                        analysis['failed_conditions'].append(condition)
                    elif evidence_type == 'neutral':
                        analysis['neutral_studies'] += 1
                    elif evidence_type == 'positive':
                        analysis['positive_studies'] += 1

                    # Store evidence details
                    analysis['evidence_details'].append({
                        'condition': condition,
                        'evidence_type': evidence_type,
                        'confidence': confidence,
                        'sample_size': edge.evidence.sample_size,
                        'study_design': edge.evidence.study_design,
                        'study_id': edge.evidence.study_id
                    })

            # Calculate average confidence
            if total_studies > 0:
                intervention_analysis[intervention]['avg_confidence'] = total_confidence / total_studies

        return dict(intervention_analysis)

    def _meets_failure_criteria(self, analysis: Dict[str, Any]) -> bool:
        """Check if intervention meets criteria for being classified as failed."""
        total_studies = analysis['total_studies']
        negative_studies = analysis['negative_studies']
        neutral_studies = analysis['neutral_studies']

        if total_studies < self.min_evidence_count:
            return False

        # Calculate failure rate (negative + neutral as "not effective")
        failure_count = negative_studies + neutral_studies
        failure_rate = failure_count / total_studies

        return failure_rate >= self.min_failure_rate

    def _create_failed_intervention(self,
                                  intervention_name: str,
                                  analysis: Dict[str, Any],
                                  knowledge_graph,
                                  bayesian_scorer=None) -> FailedIntervention:
        """Create a FailedIntervention object from analysis."""

        total_studies = analysis['total_studies']
        negative_studies = analysis['negative_studies']
        neutral_studies = analysis['neutral_studies']

        # Calculate failure rate
        failure_count = negative_studies + neutral_studies
        failure_rate = failure_count / total_studies

        # Calculate confidence using multiple approaches
        confidence = self._calculate_failure_confidence(analysis, intervention_name, bayesian_scorer)

        # Calculate waste score
        waste_score = self._calculate_waste_score(total_studies, failure_rate, confidence)

        # Determine recommendation level
        recommendation = self._determine_recommendation(confidence, failure_rate)

        # Get intervention-specific information
        intervention_info = self.known_failed_interventions.get(intervention_name, {})

        # Estimate research waste
        research_waste = self._estimate_research_waste(total_studies, intervention_name)

        # Get harm potential
        harm_potential = intervention_info.get('harm_potential', 'unknown')

        # Get alternative suggestions
        alternatives = intervention_info.get('alternative_suggestions', ['consult_healthcare_provider'])

        # Get negative studies details
        negative_studies_details = [
            detail for detail in analysis['evidence_details']
            if detail['evidence_type'] in ['negative', 'neutral']
        ]

        return FailedIntervention(
            intervention=intervention_name,
            failure_rate=failure_rate,
            confidence=confidence,
            evidence_count=total_studies,
            waste_score=waste_score,
            recommendation=recommendation,
            failed_conditions=list(set(analysis['failed_conditions'])),
            negative_studies=negative_studies_details,
            research_waste_estimate=research_waste,
            harm_potential=harm_potential,
            alternative_suggestions=alternatives
        )

    def _calculate_failure_confidence(self,
                                    analysis: Dict[str, Any],
                                    intervention_name: str,
                                    bayesian_scorer=None) -> float:
        """Calculate confidence in failure assessment."""

        total_studies = analysis['total_studies']
        failure_rate = (analysis['negative_studies'] + analysis['neutral_studies']) / total_studies
        avg_confidence = analysis['avg_confidence']

        # Base confidence from study quality and consistency
        study_quality_confidence = avg_confidence

        # Evidence quantity confidence (more studies = higher confidence)
        quantity_confidence = min(1.0, total_studies / 20.0)  # Asymptotic to 1.0

        # Consistency confidence (how consistent the failure is)
        if total_studies > 1:
            positive_rate = analysis['positive_studies'] / total_studies
            # High failure rate with low positive rate = high consistency
            consistency_confidence = failure_rate * (1 - positive_rate)
        else:
            consistency_confidence = 0.5

        # Known intervention bonus
        if intervention_name in self.known_failed_interventions:
            known_intervention_bonus = 0.2
        else:
            known_intervention_bonus = 0.0

        # Combined confidence
        confidence = (
            study_quality_confidence * 0.4 +
            quantity_confidence * 0.3 +
            consistency_confidence * 0.2 +
            known_intervention_bonus
        )

        return min(1.0, confidence)

    def _calculate_waste_score(self, evidence_count: int, failure_rate: float, confidence: float) -> float:
        """
        Calculate waste score - how much research resources were wasted.

        Higher score = more waste (more studies × higher failure rate × higher confidence)
        """
        # Base waste: evidence count × failure rate
        base_waste = evidence_count * failure_rate

        # Confidence multiplier (more confident failure = more waste)
        confidence_multiplier = 1 + confidence

        # Waste score
        waste_score = base_waste * confidence_multiplier

        return waste_score

    def _determine_recommendation(self, confidence: float, failure_rate: float) -> str:
        """Determine recommendation level based on confidence and failure rate."""

        if confidence >= self.avoid_confidence_threshold:
            if failure_rate >= 0.8:
                return "AVOID"
            else:
                return "AVOID"  # High confidence in failure regardless of rate
        else:
            if failure_rate >= 0.7:
                return "UNLIKELY TO HELP"
            else:
                return "INCONCLUSIVE"

    def _estimate_research_waste(self, study_count: int, intervention_name: str) -> float:
        """Estimate research waste in millions of dollars."""

        # Rough estimates of study costs
        study_cost_estimates = {
            'RCT': 2.0,           # $2M average
            'observational': 0.5,  # $500K average
            'meta-analysis': 0.3,  # $300K average
            'review': 0.1         # $100K average
        }

        # Known intervention waste patterns
        known_waste_multipliers = {
            'homeopathy': 1.5,    # Lots of unnecessary replication
            'magnetic_bracelets': 1.2,
            'detox_teas': 1.0,
            'alkaline_water_for_cancer': 2.0  # Very expensive studies
        }

        # Base estimate
        avg_study_cost = 1.0  # $1M average
        waste_multiplier = known_waste_multipliers.get(intervention_name, 1.0)

        estimated_waste = study_count * avg_study_cost * waste_multiplier

        return estimated_waste

    def get_worst_offenders(self, failed_interventions: Dict[str, FailedIntervention],
                          top_n: int = 10) -> Dict[str, FailedIntervention]:
        """Get worst offenders by waste score."""
        sorted_items = sorted(failed_interventions.items(),
                            key=lambda x: x[1].waste_score, reverse=True)
        return dict(sorted_items[:top_n])

    def get_avoid_recommendations(self, failed_interventions: Dict[str, FailedIntervention]) -> Dict[str, FailedIntervention]:
        """Get interventions with 'AVOID' recommendation."""
        return {
            name: intervention for name, intervention in failed_interventions.items()
            if intervention.recommendation == "AVOID"
        }

    def get_high_harm_potential(self, failed_interventions: Dict[str, FailedIntervention]) -> Dict[str, FailedIntervention]:
        """Get interventions with high harm potential."""
        high_harm_keywords = ['high', 'very_high', 'dangerous']
        return {
            name: intervention for name, intervention in failed_interventions.items()
            if any(keyword in intervention.harm_potential for keyword in high_harm_keywords)
        }

    def analyze_failure_patterns(self, failed_interventions: Dict[str, FailedIntervention]) -> Dict[str, Any]:
        """Analyze patterns in failed interventions."""

        if not failed_interventions:
            return {'error': 'No failed interventions to analyze'}

        # Failure statistics
        avg_failure_rate = np.mean([fi.failure_rate for fi in failed_interventions.values()])
        avg_confidence = np.mean([fi.confidence for fi in failed_interventions.values()])
        total_waste = sum([fi.waste_score for fi in failed_interventions.values()])

        # Recommendation distribution
        recommendation_counts = Counter([fi.recommendation for fi in failed_interventions.values()])

        # Harm potential distribution
        harm_counts = Counter([fi.harm_potential for fi in failed_interventions.values()])

        # Most wasteful categories
        category_waste = defaultdict(float)
        for intervention_name, failed_intervention in failed_interventions.items():
            for category, interventions in self.high_risk_categories.items():
                if intervention_name in interventions:
                    category_waste[category] += failed_intervention.waste_score

        return {
            'total_failed_interventions': len(failed_interventions),
            'average_failure_rate': round(avg_failure_rate, 3),
            'average_confidence': round(avg_confidence, 3),
            'total_waste_score': round(total_waste, 2),
            'recommendation_distribution': dict(recommendation_counts),
            'harm_potential_distribution': dict(harm_counts),
            'most_wasteful_categories': dict(sorted(category_waste.items(),
                                                  key=lambda x: x[1], reverse=True)),
            'estimated_total_research_waste_millions': round(
                sum([fi.research_waste_estimate for fi in failed_interventions.values()]), 1
            )
        }

    def get_alternatives_for_failed_intervention(self, failed_intervention: FailedIntervention) -> List[str]:
        """Get alternative suggestions for a failed intervention."""
        return failed_intervention.alternative_suggestions

    def check_intervention_status(self, intervention_name: str,
                                failed_interventions: Dict[str, FailedIntervention]) -> Dict[str, Any]:
        """Check if an intervention is in the failed catalog."""

        if intervention_name in failed_interventions:
            failed_int = failed_interventions[intervention_name]
            return {
                'status': 'FAILED',
                'recommendation': failed_int.recommendation,
                'failure_rate': failed_int.failure_rate,
                'confidence': failed_int.confidence,
                'alternatives': failed_int.alternative_suggestions
            }
        else:
            return {
                'status': 'NOT_IN_CATALOG',
                'note': 'This intervention is not in the failed interventions catalog'
            }