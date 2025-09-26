"""
Power Combination Pattern Analysis System.

Identifies synergistic treatment combinations where 1+1=3 - combined effect
is greater than the sum of individual effects. Discovers complementary
pathways and helps design optimal treatment protocols.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
from itertools import combinations
import math


@dataclass
class PowerCombination:
    """Synergistic combination of interventions."""
    combination: Tuple[str, str]
    confidence: float
    lift: float
    conditions_count: int
    recommendation: str
    mechanism_explanation: str
    individual_effectiveness: Dict[str, float]
    combined_effectiveness: float
    synergy_score: float
    evidence_strength: float
    complementary_mechanisms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format matching expected output."""
        combination_name = ' + '.join(self.combination)
        return {
            'confidence': round(self.confidence, 3),
            'lift': round(self.lift, 1),
            'conditions_count': self.conditions_count,
            'recommendation': self.recommendation,
            'mechanism_explanation': self.mechanism_explanation,
            'synergy_score': round(self.synergy_score, 2),
            'individual_effectiveness': {k: round(v, 2) for k, v in self.individual_effectiveness.items()},
            'combined_effectiveness': round(self.combined_effectiveness, 2)
        }


class PowerCombinationAnalysis:
    """
    Power combination analysis system that identifies synergistic
    treatment combinations using multiple approaches:

    1. Co-occurrence analysis (how often used together)
    2. Lift calculation (likelihood boost when combined)
    3. Mechanism complementarity (different pathways)
    4. Effectiveness enhancement (1+1=3 effect)
    """

    def __init__(self,
                 min_confidence: float = 0.7,
                 min_lift: float = 1.5,
                 min_conditions: int = 3):
        """
        Initialize power combination analysis system.

        Args:
            min_confidence: Minimum confidence for valid combinations
            min_lift: Minimum lift for synergistic effect
            min_conditions: Minimum conditions to establish pattern
        """
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.min_conditions = min_conditions

        # Known synergistic combinations with mechanisms
        self.known_synergies = {
            ('probiotics', 'prebiotics'): {
                'mechanism': 'Prebiotics feed probiotics',
                'recommendation': 'Take together - prebiotics feed the probiotics',
                'pathways': ['microbiome_enhancement', 'gut_barrier_function']
            },
            ('vitamin_d', 'magnesium'): {
                'mechanism': 'Magnesium activates vitamin D',
                'recommendation': 'Magnesium helps vitamin D absorption and activation',
                'pathways': ['bone_metabolism', 'immune_function']
            },
            ('cbt', 'exercise'): {
                'mechanism': 'Exercise boosts mood; CBT maintains gains',
                'recommendation': 'Exercise provides biological boost; CBT maintains psychological gains',
                'pathways': ['neuroplasticity', 'stress_response']
            },
            ('omega_3', 'vitamin_d'): {
                'mechanism': 'Both support anti-inflammatory pathways',
                'recommendation': 'Synergistic anti-inflammatory effects',
                'pathways': ['inflammation_reduction', 'immune_modulation']
            },
            ('meditation', 'exercise'): {
                'mechanism': 'Meditation enhances exercise benefits via stress reduction',
                'recommendation': 'Meditation amplifies exercise-induced neuroplasticity',
                'pathways': ['stress_response', 'neuroplasticity', 'autonomic_balance']
            },
            ('caffeine', 'l_theanine'): {
                'mechanism': 'L-theanine smooths caffeine stimulation',
                'recommendation': 'L-theanine provides focus without jitters',
                'pathways': ['neurotransmitter_balance', 'attention_enhancement']
            },
            ('curcumin', 'black_pepper'): {
                'mechanism': 'Piperine increases curcumin absorption',
                'recommendation': 'Black pepper increases curcumin bioavailability 20x',
                'pathways': ['bioavailability_enhancement', 'anti_inflammatory']
            },
            ('zinc', 'vitamin_c'): {
                'mechanism': 'Synergistic immune support',
                'recommendation': 'Enhanced immune function when combined',
                'pathways': ['immune_enhancement', 'antioxidant_activity']
            }
        }

        # Mechanism complementarity patterns
        self.complementary_mechanisms = {
            'gut_health': {
                'probiotics': 'adds beneficial bacteria',
                'prebiotics': 'feeds beneficial bacteria',
                'fiber': 'provides bacterial substrate',
                'fermented_foods': 'provides diverse bacteria'
            },
            'inflammation_reduction': {
                'omega_3': 'provides EPA/DHA anti-inflammatory compounds',
                'turmeric': 'provides curcumin anti-inflammatory compounds',
                'cold_exposure': 'activates anti-inflammatory pathways',
                'exercise': 'reduces inflammatory markers'
            },
            'stress_management': {
                'meditation': 'trains attention and awareness',
                'exercise': 'provides biological stress resilience',
                'yoga': 'combines physical and mental practices',
                'breathing': 'activates parasympathetic nervous system'
            },
            'cognitive_enhancement': {
                'exercise': 'increases BDNF and neuroplasticity',
                'meditation': 'improves attention and working memory',
                'omega_3': 'provides brain structural support',
                'sleep_optimization': 'enables memory consolidation'
            },
            'metabolic_optimization': {
                'exercise': 'improves insulin sensitivity',
                'intermittent_fasting': 'enhances metabolic flexibility',
                'cold_exposure': 'activates brown fat',
                'metformin': 'improves glucose metabolism'
            }
        }

    def analyze_power_combinations(self,
                                 knowledge_graph,
                                 discovered_mechanisms: List,
                                 fundamental_interventions: Dict) -> Dict[str, PowerCombination]:
        """
        Analyze power combinations using multiple approaches.

        Args:
            knowledge_graph: MedicalKnowledgeGraph instance
            discovered_mechanisms: List of BiologicalMechanism objects
            fundamental_interventions: Dict of FundamentalIntervention objects

        Returns:
            Dictionary of power combinations
        """
        # Get all interventions and their effectiveness data
        interventions = self._get_intervention_data(knowledge_graph, fundamental_interventions)

        # Find all possible intervention pairs
        intervention_pairs = list(combinations(interventions.keys(), 2))

        print(f"Analyzing {len(intervention_pairs)} potential intervention combinations...")

        # Analyze each combination
        power_combinations = {}

        for intervention_a, intervention_b in intervention_pairs:
            combination = self._analyze_combination(
                intervention_a, intervention_b,
                interventions, knowledge_graph, discovered_mechanisms
            )

            if combination and self._meets_power_criteria(combination):
                combination_name = f"{intervention_a} + {intervention_b}"
                power_combinations[combination_name] = combination

        # Sort by synergy score
        sorted_combinations = dict(
            sorted(power_combinations.items(),
                   key=lambda x: x[1].synergy_score, reverse=True)
        )

        # Store for later access by other components
        self._discovered_combinations = sorted_combinations

        return sorted_combinations

    def _get_intervention_data(self, knowledge_graph, fundamental_interventions: Dict) -> Dict[str, Dict]:
        """Get intervention effectiveness data."""
        interventions = {}

        # From knowledge graph
        for intervention in knowledge_graph.forward_edges:
            conditions = list(knowledge_graph.forward_edges[intervention].keys())
            if len(conditions) >= 2:  # Need multiple conditions for combination analysis
                effectiveness_scores = []

                for condition in conditions:
                    if condition in knowledge_graph.reverse_edges:
                        if intervention in knowledge_graph.reverse_edges[condition]:
                            edges = knowledge_graph.reverse_edges[condition][intervention]
                            for edge in edges:
                                score = edge.weight * edge.evidence.confidence
                                effectiveness_scores.append(score)

                if effectiveness_scores:
                    interventions[intervention] = {
                        'conditions': conditions,
                        'avg_effectiveness': np.mean(effectiveness_scores),
                        'evidence_count': len(effectiveness_scores)
                    }

        # From fundamental interventions
        for name, fund_intervention in fundamental_interventions.items():
            if name not in interventions:
                interventions[name] = {
                    'conditions': fund_intervention.conditions,
                    'avg_effectiveness': fund_intervention.fundamentality_score / 100.0,
                    'evidence_count': fund_intervention.total_evidence_count
                }

        return interventions

    def _analyze_combination(self,
                           intervention_a: str,
                           intervention_b: str,
                           interventions: Dict,
                           knowledge_graph,
                           discovered_mechanisms: List) -> Optional[PowerCombination]:
        """Analyze a specific intervention combination."""

        if intervention_a not in interventions or intervention_b not in interventions:
            return None

        # Calculate co-occurrence and lift
        co_occurrence_data = self._calculate_co_occurrence(
            intervention_a, intervention_b, interventions
        )

        if not co_occurrence_data:
            return None

        confidence = co_occurrence_data['confidence']
        lift = co_occurrence_data['lift']
        shared_conditions = co_occurrence_data['shared_conditions']

        # Calculate synergy metrics
        synergy_metrics = self._calculate_synergy(
            intervention_a, intervention_b, interventions, shared_conditions
        )

        # Check for mechanism complementarity
        complementarity = self._analyze_mechanism_complementarity(
            intervention_a, intervention_b, discovered_mechanisms
        )

        # Generate recommendation
        recommendation = self._generate_combination_recommendation(
            intervention_a, intervention_b, complementarity
        )

        # Calculate overall synergy score
        synergy_score = self._calculate_overall_synergy_score(
            confidence, lift, synergy_metrics, complementarity
        )

        return PowerCombination(
            combination=(intervention_a, intervention_b),
            confidence=confidence,
            lift=lift,
            conditions_count=len(shared_conditions),
            recommendation=recommendation['text'],
            mechanism_explanation=recommendation['mechanism'],
            individual_effectiveness=synergy_metrics['individual'],
            combined_effectiveness=synergy_metrics['combined'],
            synergy_score=synergy_score,
            evidence_strength=min(interventions[intervention_a]['evidence_count'],
                                interventions[intervention_b]['evidence_count']) / 10.0,
            complementary_mechanisms=complementarity['mechanisms']
        )

    def _calculate_co_occurrence(self, intervention_a: str, intervention_b: str,
                               interventions: Dict) -> Optional[Dict]:
        """Calculate co-occurrence statistics for intervention pair."""

        conditions_a = set(interventions[intervention_a]['conditions'])
        conditions_b = set(interventions[intervention_b]['conditions'])

        # Shared conditions where both interventions are used
        shared_conditions = conditions_a.intersection(conditions_b)

        if len(shared_conditions) < self.min_conditions:
            return None

        # Calculate confidence: P(B|A) = P(A and B) / P(A)
        confidence = len(shared_conditions) / len(conditions_a)

        # Calculate lift: P(B|A) / P(B)
        # Approximation: lift = confidence / (len(conditions_b) / total_conditions)
        total_conditions = len(conditions_a.union(conditions_b))
        expected_prob = len(conditions_b) / total_conditions
        lift = confidence / max(expected_prob, 0.1)  # Avoid division by zero

        return {
            'confidence': confidence,
            'lift': lift,
            'shared_conditions': list(shared_conditions)
        }

    def _calculate_synergy(self, intervention_a: str, intervention_b: str,
                         interventions: Dict, shared_conditions: List[str]) -> Dict:
        """Calculate synergy metrics (1+1=3 effect)."""

        effectiveness_a = interventions[intervention_a]['avg_effectiveness']
        effectiveness_b = interventions[intervention_b]['avg_effectiveness']

        # Estimate combined effectiveness
        # Model: synergy can increase effectiveness beyond additive
        baseline_combined = (effectiveness_a + effectiveness_b) / 2

        # Known synergy bonus for specific combinations
        combination_tuple = tuple(sorted([intervention_a, intervention_b]))
        if combination_tuple in self.known_synergies:
            synergy_bonus = 0.3  # 30% boost for known synergies
        else:
            # Estimate synergy based on mechanism complementarity
            synergy_bonus = self._estimate_synergy_bonus(intervention_a, intervention_b)

        combined_effectiveness = min(1.0, baseline_combined * (1 + synergy_bonus))

        return {
            'individual': {intervention_a: effectiveness_a, intervention_b: effectiveness_b},
            'combined': combined_effectiveness,
            'synergy_factor': combined_effectiveness / max(baseline_combined, 0.1)
        }

    def _estimate_synergy_bonus(self, intervention_a: str, intervention_b: str) -> float:
        """Estimate synergy bonus based on intervention characteristics."""

        # Check if interventions work through complementary mechanisms
        complementary_score = 0

        for mechanism, intervention_roles in self.complementary_mechanisms.items():
            if intervention_a in intervention_roles and intervention_b in intervention_roles:
                # Both work in same mechanism domain - potential synergy
                role_a = intervention_roles[intervention_a]
                role_b = intervention_roles[intervention_b]
                if role_a != role_b:  # Different roles = complementary
                    complementary_score += 0.2

        # Known synergistic pairs
        known_pairs = [
            ('probiotics', 'prebiotics'), ('vitamin_d', 'magnesium'),
            ('caffeine', 'l_theanine'), ('curcumin', 'black_pepper'),
            ('omega_3', 'vitamin_d'), ('meditation', 'exercise')
        ]

        for pair in known_pairs:
            if (intervention_a in pair and intervention_b in pair):
                complementary_score += 0.3

        return min(0.5, complementary_score)  # Cap at 50% bonus

    def _analyze_mechanism_complementarity(self, intervention_a: str, intervention_b: str,
                                         discovered_mechanisms: List) -> Dict:
        """Analyze if interventions work through complementary mechanisms."""

        mechanisms_a = []
        mechanisms_b = []

        # Find mechanisms for each intervention
        for mechanism in discovered_mechanisms:
            if intervention_a in mechanism.defining_interventions:
                mechanisms_a.append(mechanism.name)
            if intervention_b in mechanism.defining_interventions:
                mechanisms_b.append(mechanism.name)

        # Check complementarity
        shared_mechanisms = set(mechanisms_a).intersection(set(mechanisms_b))
        different_mechanisms = set(mechanisms_a).symmetric_difference(set(mechanisms_b))

        complementarity_score = len(different_mechanisms) / max(len(mechanisms_a) + len(mechanisms_b), 1)

        return {
            'mechanisms': list(shared_mechanisms) + list(different_mechanisms),
            'complementarity_score': complementarity_score,
            'shared_pathways': list(shared_mechanisms),
            'different_pathways': list(different_mechanisms)
        }

    def _generate_combination_recommendation(self, intervention_a: str, intervention_b: str,
                                           complementarity: Dict) -> Dict[str, str]:
        """Generate practical recommendation for combination."""

        combination_tuple = tuple(sorted([intervention_a, intervention_b]))

        # Check known combinations first
        if combination_tuple in self.known_synergies:
            synergy_info = self.known_synergies[combination_tuple]
            return {
                'text': synergy_info['recommendation'],
                'mechanism': synergy_info['mechanism']
            }

        # Generate recommendation based on complementarity
        if complementarity['complementarity_score'] > 0.5:
            mechanism_text = f"{intervention_a} and {intervention_b} work through different pathways"
            recommendation = f"Combine for complementary effects - {mechanism_text}"
        else:
            mechanism_text = f"{intervention_a} and {intervention_b} work through similar pathways"
            recommendation = f"May enhance effects when used together - {mechanism_text}"

        return {
            'text': recommendation,
            'mechanism': mechanism_text
        }

    def _calculate_overall_synergy_score(self, confidence: float, lift: float,
                                       synergy_metrics: Dict, complementarity: Dict) -> float:
        """Calculate overall synergy score combining all factors."""

        # Weight different factors
        confidence_weight = 0.3
        lift_weight = 0.3
        synergy_weight = 0.2
        complementarity_weight = 0.2

        # Normalize lift (typical range 1-5)
        normalized_lift = min(1.0, (lift - 1.0) / 4.0)

        # Get synergy factor
        synergy_factor = synergy_metrics.get('synergy_factor', 1.0)
        normalized_synergy = min(1.0, (synergy_factor - 1.0) / 1.0)

        # Complementarity score
        complementarity_score = complementarity['complementarity_score']

        # Combined score
        overall_score = (
            confidence * confidence_weight +
            normalized_lift * lift_weight +
            normalized_synergy * synergy_weight +
            complementarity_score * complementarity_weight
        )

        return min(1.0, overall_score)

    def _meets_power_criteria(self, combination: PowerCombination) -> bool:
        """Check if combination meets criteria for being a power combination."""
        return (
            combination.confidence >= self.min_confidence and
            combination.lift >= self.min_lift and
            combination.conditions_count >= self.min_conditions and
            combination.synergy_score >= 0.5  # Minimum synergy threshold
        )

    def get_top_combinations(self, power_combinations: Dict[str, PowerCombination],
                           top_n: int = 10) -> Dict[str, PowerCombination]:
        """Get top N power combinations by synergy score."""
        sorted_items = sorted(power_combinations.items(),
                            key=lambda x: x[1].synergy_score, reverse=True)
        return dict(sorted_items[:top_n])

    def get_combinations_for_intervention(self, power_combinations: Dict[str, PowerCombination],
                                        intervention: str) -> Dict[str, PowerCombination]:
        """Get all combinations involving a specific intervention."""
        return {
            name: combo for name, combo in power_combinations.items()
            if intervention in combo.combination
        }

    def get_mechanism_based_combinations(self, power_combinations: Dict[str, PowerCombination],
                                       mechanism: str) -> Dict[str, PowerCombination]:
        """Get combinations that work through a specific mechanism."""
        return {
            name: combo for name, combo in power_combinations.items()
            if mechanism.lower() in combo.mechanism_explanation.lower()
        }

    def analyze_combination_trends(self, power_combinations: Dict[str, PowerCombination]) -> Dict[str, Any]:
        """Analyze trends in power combinations."""

        if not power_combinations:
            return {'error': 'No power combinations to analyze'}

        # Most common interventions in combinations
        intervention_frequency = Counter()
        for combo in power_combinations.values():
            intervention_frequency.update(combo.combination)

        # Average metrics
        avg_confidence = np.mean([combo.confidence for combo in power_combinations.values()])
        avg_lift = np.mean([combo.lift for combo in power_combinations.values()])
        avg_synergy = np.mean([combo.synergy_score for combo in power_combinations.values()])

        # Mechanism patterns
        mechanism_patterns = Counter()
        for combo in power_combinations.values():
            for mechanism in combo.complementary_mechanisms:
                mechanism_patterns[mechanism] += 1

        return {
            'total_combinations': len(power_combinations),
            'most_synergistic_interventions': dict(intervention_frequency.most_common(5)),
            'average_confidence': round(avg_confidence, 3),
            'average_lift': round(avg_lift, 1),
            'average_synergy_score': round(avg_synergy, 2),
            'common_mechanisms': dict(mechanism_patterns.most_common(5))
        }

    def get_combinations_for_condition(self, condition: str) -> Dict[str, Dict[str, Any]]:
        """
        Get power combinations relevant for a specific condition.

        Args:
            condition: Target condition

        Returns:
            Dictionary of combinations that may help with the condition
        """
        if not hasattr(self, '_discovered_combinations'):
            return {}

        relevant_combinations = {}

        for combo_name, combination in self._discovered_combinations.items():
            # Check if any of the combination's interventions target this condition
            if condition in combination.primary_conditions:
                relevant_combinations[combo_name] = {
                    'lift': combination.lift,
                    'confidence': combination.confidence,
                    'synergy_score': combination.synergy_score,
                    'interventions': combination.interventions,
                    'mechanisms': combination.complementary_mechanisms,
                    'conditions': combination.primary_conditions
                }

        return relevant_combinations