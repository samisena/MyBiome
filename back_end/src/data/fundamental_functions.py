"""
Fundamental Body Functions Discovery System.

Finds interventions that work across multiple unrelated conditions,
revealing fundamental biological processes like inflammation, cellular
metabolism, and stress response - the "master keys" of medicine.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import math


@dataclass
class FundamentalIntervention:
    """Intervention that works across multiple biological mechanisms."""
    name: str
    conditions: List[str]
    mechanisms: List[str]
    fundamentality_score: float
    innovation_index: float
    classification: str  # 'established', 'emerging', 'experimental'
    evidence_strength: float
    cross_mechanism_effects: Dict[str, float]
    inferred_body_functions: List[str]
    total_evidence_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format matching expected output."""
        return {
            'conditions': self.conditions,
            'mechanisms': self.mechanisms,
            'fundamentality_score': round(self.fundamentality_score, 1),
            'innovation_index': round(self.innovation_index, 2),
            'classification': self.classification,
            'evidence_strength': round(self.evidence_strength, 2),
            'inferred_functions': self.inferred_body_functions
        }


class FundamentalFunctionsDiscovery:
    """
    Discovers fundamental body functions by analyzing interventions
    that work across multiple unrelated biological mechanisms.

    Key insight: If something helps depression AND diabetes AND arthritis,
    it's targeting fundamental processes like inflammation or metabolism.
    """

    def __init__(self, min_mechanisms: int = 3, min_conditions: int = 4):
        """
        Initialize fundamental functions discovery system.

        Args:
            min_mechanisms: Minimum mechanisms to be considered fundamental
            min_conditions: Minimum conditions to be considered fundamental
        """
        self.min_mechanisms = min_mechanisms
        self.min_conditions = min_conditions

        # Body function inference patterns
        self.function_patterns = {
            'inflammation': {
                'conditions': ['arthritis', 'ibs', 'crohns', 'autoimmune', 'inflammatory_bowel_disease', 'psoriasis'],
                'interventions': ['anti_inflammatory', 'omega_3', 'turmeric', 'curcumin', 'cold_exposure'],
                'mechanisms': ['inflammatory', 'immune', 'gut_microbiome']
            },
            'stress_response': {
                'conditions': ['anxiety', 'depression', 'insomnia', 'hypertension', 'ibs', 'chronic_pain'],
                'interventions': ['meditation', 'yoga', 'breathing', 'mindfulness', 'therapy'],
                'mechanisms': ['neuropsychological', 'cardiovascular', 'gut_microbiome']
            },
            'cellular_metabolism': {
                'conditions': ['diabetes', 'obesity', 'metabolic_syndrome', 'pcos', 'fatigue'],
                'interventions': ['exercise', 'intermittent_fasting', 'cold_exposure', 'metformin'],
                'mechanisms': ['metabolic', 'cardiovascular', 'neuropsychological']
            },
            'microbiome_regulation': {
                'conditions': ['ibs', 'depression', 'anxiety', 'autoimmune', 'allergies'],
                'interventions': ['probiotics', 'fiber', 'fermented_foods', 'prebiotics'],
                'mechanisms': ['gut_microbiome', 'neuropsychological', 'immune']
            },
            'circadian_rhythm': {
                'conditions': ['insomnia', 'depression', 'metabolic_syndrome', 'fatigue'],
                'interventions': ['light_therapy', 'sleep_hygiene', 'melatonin', 'exercise'],
                'mechanisms': ['neuropsychological', 'metabolic', 'hormonal']
            },
            'autonomic_nervous_system': {
                'conditions': ['anxiety', 'hypertension', 'arrhythmia', 'ibs', 'chronic_pain'],
                'interventions': ['breathing_exercises', 'cold_exposure', 'meditation', 'vagus_nerve_stimulation'],
                'mechanisms': ['neuropsychological', 'cardiovascular', 'gut_microbiome']
            }
        }

        # Classification thresholds
        self.classification_thresholds = {
            'established': {'evidence_count': 10, 'innovation_index_max': 0.6},
            'emerging': {'evidence_count': 5, 'innovation_index_min': 0.6, 'innovation_index_max': 0.9},
            'experimental': {'evidence_count': 2, 'innovation_index_min': 0.8}
        }

    def discover_fundamental_interventions(
        self,
        knowledge_graph,
        discovered_mechanisms: List
    ) -> Dict[str, FundamentalIntervention]:
        """
        Discover fundamental interventions that work across multiple mechanisms.

        Args:
            knowledge_graph: MedicalKnowledgeGraph instance
            discovered_mechanisms: List of BiologicalMechanism objects

        Returns:
            Dictionary of fundamental interventions
        """
        # Analyze cross-mechanism effects for each intervention
        intervention_analysis = self._analyze_cross_mechanism_effects(
            knowledge_graph, discovered_mechanisms
        )

        # Filter and score fundamental interventions
        fundamental_interventions = {}

        for intervention_name, analysis in intervention_analysis.items():
            if self._meets_fundamentality_criteria(analysis):
                fundamental_intervention = self._create_fundamental_intervention(
                    intervention_name, analysis, knowledge_graph
                )
                fundamental_interventions[intervention_name] = fundamental_intervention

        # Sort by fundamentality score
        sorted_interventions = dict(
            sorted(fundamental_interventions.items(),
                   key=lambda x: x[1].fundamentality_score, reverse=True)
        )

        # Store for later access by other components
        self._discovered_fundamental_interventions = sorted_interventions

        return sorted_interventions

    def _analyze_cross_mechanism_effects(
        self,
        knowledge_graph,
        discovered_mechanisms: List
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze how each intervention affects multiple mechanisms."""
        intervention_analysis = defaultdict(lambda: {
            'mechanisms': set(),
            'conditions': set(),
            'mechanism_effects': defaultdict(list),
            'total_evidence': 0,
            'avg_effectiveness': 0
        })

        # Map mechanisms to their names for lookup
        mechanism_map = {tuple(sorted(m.conditions)): m.name for m in discovered_mechanisms}

        # Analyze each intervention's effects across mechanisms
        for intervention in knowledge_graph.forward_edges:
            intervention_conditions = set(knowledge_graph.forward_edges[intervention].keys())

            # Find which mechanisms this intervention affects
            affected_mechanisms = []
            total_effectiveness = 0
            evidence_count = 0

            for mechanism in discovered_mechanisms:
                mechanism_conditions = set(mechanism.conditions)
                overlap = intervention_conditions.intersection(mechanism_conditions)

                if overlap:
                    # Calculate effectiveness in this mechanism
                    mechanism_effectiveness = []
                    mechanism_evidence = 0

                    for condition in overlap:
                        if condition in knowledge_graph.reverse_edges:
                            if intervention in knowledge_graph.reverse_edges[condition]:
                                edges = knowledge_graph.reverse_edges[condition][intervention]
                                for edge in edges:
                                    effectiveness = edge.weight * edge.evidence.confidence
                                    mechanism_effectiveness.append(effectiveness)
                                    mechanism_evidence += 1

                    if mechanism_effectiveness:
                        avg_mechanism_effect = np.mean(mechanism_effectiveness)
                        affected_mechanisms.append(mechanism.name)

                        intervention_analysis[intervention]['mechanisms'].add(mechanism.name)
                        intervention_analysis[intervention]['mechanism_effects'][mechanism.name] = avg_mechanism_effect
                        intervention_analysis[intervention]['total_evidence'] += mechanism_evidence

                        total_effectiveness += avg_mechanism_effect
                        evidence_count += mechanism_evidence

            # Store overall analysis
            intervention_analysis[intervention]['conditions'] = intervention_conditions
            intervention_analysis[intervention]['avg_effectiveness'] = (
                total_effectiveness / len(affected_mechanisms) if affected_mechanisms else 0
            )

        return dict(intervention_analysis)

    def _meets_fundamentality_criteria(self, analysis: Dict[str, Any]) -> bool:
        """Check if intervention meets criteria for being fundamental."""
        return (
            len(analysis['mechanisms']) >= self.min_mechanisms and
            len(analysis['conditions']) >= self.min_conditions and
            analysis['avg_effectiveness'] > 0.3 and  # Minimum effectiveness threshold
            analysis['total_evidence'] >= 3  # Minimum evidence requirement
        )

    def _create_fundamental_intervention(
        self,
        intervention_name: str,
        analysis: Dict[str, Any],
        knowledge_graph
    ) -> FundamentalIntervention:
        """Create a FundamentalIntervention object from analysis."""

        # Calculate fundamentality score
        fundamentality_score = self._calculate_fundamentality_score(analysis)

        # Calculate innovation index
        innovation_index = self._calculate_innovation_index(analysis, knowledge_graph)

        # Classify intervention
        classification = self._classify_intervention(analysis, innovation_index)

        # Infer body functions
        inferred_functions = self._infer_body_functions(
            intervention_name, list(analysis['conditions']), list(analysis['mechanisms'])
        )

        # Calculate evidence strength
        evidence_strength = min(1.0, analysis['total_evidence'] / 20.0)  # Normalize to 0-1

        return FundamentalIntervention(
            name=intervention_name,
            conditions=sorted(list(analysis['conditions'])),
            mechanisms=sorted(list(analysis['mechanisms'])),
            fundamentality_score=fundamentality_score,
            innovation_index=innovation_index,
            classification=classification,
            evidence_strength=evidence_strength,
            cross_mechanism_effects=dict(analysis['mechanism_effects']),
            inferred_body_functions=inferred_functions,
            total_evidence_count=analysis['total_evidence']
        )

    def _calculate_fundamentality_score(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate fundamentality score based on:
        - Number of mechanisms affected
        - Number of conditions helped
        - Average effectiveness
        - Evidence diversity
        """
        mechanism_count = len(analysis['mechanisms'])
        condition_count = len(analysis['conditions'])
        avg_effectiveness = analysis['avg_effectiveness']
        evidence_count = analysis['total_evidence']

        # Base score from breadth (mechanisms × conditions)
        breadth_score = mechanism_count * condition_count

        # Effectiveness multiplier
        effectiveness_multiplier = avg_effectiveness

        # Evidence confidence boost
        evidence_boost = min(2.0, 1 + math.log10(evidence_count))

        fundamentality_score = breadth_score * effectiveness_multiplier * evidence_boost

        return fundamentality_score

    def _calculate_innovation_index(
        self,
        analysis: Dict[str, Any],
        knowledge_graph
    ) -> float:
        """
        Calculate innovation index based on:
        - Recency of evidence
        - Evidence volume (low volume = emerging)
        - Mechanism diversity
        """
        evidence_count = analysis['total_evidence']
        mechanism_diversity = len(analysis['mechanisms'])

        # Evidence volume factor (less evidence = more innovative/emerging)
        if evidence_count < 5:
            volume_factor = 0.9
        elif evidence_count < 15:
            volume_factor = 0.6
        elif evidence_count < 30:
            volume_factor = 0.3
        else:
            volume_factor = 0.1

        # Mechanism diversity factor (more diverse = more innovative)
        diversity_factor = min(1.0, mechanism_diversity / 5.0)

        # Innovation index combines these factors
        innovation_index = (volume_factor + diversity_factor) / 2

        return min(1.0, innovation_index)

    def _classify_intervention(self, analysis: Dict[str, Any], innovation_index: float) -> str:
        """Classify intervention as established, emerging, or experimental."""
        evidence_count = analysis['total_evidence']

        if (evidence_count >= self.classification_thresholds['established']['evidence_count'] and
            innovation_index <= self.classification_thresholds['established']['innovation_index_max']):
            return 'established'
        elif (evidence_count >= self.classification_thresholds['emerging']['evidence_count'] and
              self.classification_thresholds['emerging']['innovation_index_min'] <= innovation_index <=
              self.classification_thresholds['emerging']['innovation_index_max']):
            return 'emerging'
        else:
            return 'experimental'

    def _infer_body_functions(
        self,
        intervention_name: str,
        conditions: List[str],
        mechanisms: List[str]
    ) -> List[str]:
        """
        Infer which fundamental body functions this intervention affects
        based on patterns in conditions and mechanisms.
        """
        inferred_functions = []

        for function_name, pattern in self.function_patterns.items():
            score = 0

            # Score based on condition overlap
            condition_overlap = len(set(conditions).intersection(set(pattern['conditions'])))
            score += condition_overlap * 2

            # Score based on intervention name patterns
            if intervention_name.lower() in [i.lower() for i in pattern['interventions']]:
                score += 3

            # Score based on mechanism overlap
            mechanism_overlap = 0
            for mechanism in mechanisms:
                for pattern_mechanism in pattern['mechanisms']:
                    if pattern_mechanism.lower() in mechanism.lower():
                        mechanism_overlap += 1
            score += mechanism_overlap

            # If score is high enough, include this function
            if score >= 3:  # Threshold for inclusion
                inferred_functions.append(function_name)

        # Sort by confidence (could implement more sophisticated scoring)
        return sorted(inferred_functions)

    def get_master_keys_summary(
        self,
        fundamental_interventions: Dict[str, FundamentalIntervention]
    ) -> Dict[str, Any]:
        """Get summary of master key interventions and their functions."""
        summary = {
            'total_fundamental_interventions': len(fundamental_interventions),
            'classifications': defaultdict(int),
            'body_functions_affected': defaultdict(int),
            'top_interventions': []
        }

        for intervention in fundamental_interventions.values():
            summary['classifications'][intervention.classification] += 1

            for function in intervention.inferred_body_functions:
                summary['body_functions_affected'][function] += 1

            summary['top_interventions'].append({
                'name': intervention.name,
                'score': intervention.fundamentality_score,
                'classification': intervention.classification,
                'functions': intervention.inferred_body_functions
            })

        # Sort top interventions by score
        summary['top_interventions'].sort(key=lambda x: x['score'], reverse=True)
        summary['top_interventions'] = summary['top_interventions'][:10]  # Top 10

        return dict(summary)

    def find_function_targets(
        self,
        fundamental_interventions: Dict[str, FundamentalIntervention],
        target_function: str
    ) -> List[Dict[str, Any]]:
        """Find all interventions that target a specific body function."""
        function_interventions = []

        for intervention in fundamental_interventions.values():
            if target_function in intervention.inferred_body_functions:
                function_interventions.append({
                    'intervention': intervention.name,
                    'fundamentality_score': intervention.fundamentality_score,
                    'classification': intervention.classification,
                    'conditions': intervention.conditions,
                    'mechanisms': intervention.mechanisms
                })

        # Sort by fundamentality score
        function_interventions.sort(key=lambda x: x['fundamentality_score'], reverse=True)
        return function_interventions

    def get_fundamental_interventions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get fundamental interventions in format expected by other components.

        Returns:
            Dictionary mapping intervention names to their fundamental data
        """
        if not hasattr(self, '_discovered_fundamental_interventions'):
            return {}

        # Convert FundamentalIntervention objects to dictionaries
        result = {}
        for name, intervention in self._discovered_fundamental_interventions.items():
            result[name] = {
                'mechanisms': intervention.mechanisms,
                'cross_condition_effectiveness': intervention.fundamentality_score,
                'classification': intervention.classification,
                'conditions': intervention.conditions,
                'body_functions': intervention.inferred_body_functions
            }

        return result