"""
Research Gap Identification System.

Predicts effectiveness of untested condition-intervention pairs using
multiple evidence sources. Acts as a "recommendation engine for medical research"
to identify promising areas for study among thousands of possible combinations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from back_end.src.data_collection.data_mining_repository import (
        DataMiningRepository,
        ResearchGap as DBResearchGap
    )
    from back_end.src.data.config import setup_logging
except ImportError as e:
    print(f"Warning: Could not import database components: {e}")
    DataMiningRepository = None
    DBResearchGap = None

logger = setup_logging(__name__, 'research_gaps.log') if 'setup_logging' in globals() else None


@dataclass
class ResearchGap:
    """Untested condition-intervention pair with predicted effectiveness."""
    condition: str
    intervention: str
    predicted_score: float
    confidence_level: str
    reasoning: List[str]
    priority: str
    evidence_sources: Dict[str, float]
    similar_conditions: List[str]
    mechanism_matches: List[str]
    innovation_potential: float
    research_feasibility: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format matching expected output."""
        return {
            'condition': self.condition,
            'intervention': self.intervention,
            'predicted_score': round(self.predicted_score, 2),
            'confidence_level': self.confidence_level,
            'reasoning': self.reasoning,
            'priority': self.priority,
            'evidence_sources': {k: round(v, 3) for k, v in self.evidence_sources.items()},
            'innovation_potential': round(self.innovation_potential, 2)
        }


class ResearchGapIdentification:
    """
    Research gap identification system that predicts effectiveness of
    untested condition-intervention pairs using multiple evidence sources.

    Key approaches:
    1. Condition similarity (if it works for similar conditions)
    2. Mechanism matching (same biological pathways)
    3. Intervention profile analysis (intervention's track record)
    4. Innovation potential assessment
    """

    def __init__(self,
                 similarity_threshold: float = 0.3,
                 confidence_threshold: float = 0.5,
                 max_gaps_per_condition: int = 10):
        """
        Initialize research gap identification system.

        Args:
            similarity_threshold: Minimum similarity for condition matching
            confidence_threshold: Minimum confidence for recommendations
            max_gaps_per_condition: Maximum gaps to identify per condition
        """
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.max_gaps_per_condition = max_gaps_per_condition

        # Condition similarity patterns for medical knowledge
        self.condition_clusters = {
            'inflammatory': ['arthritis', 'ibs', 'crohns', 'inflammatory_bowel_disease',
                           'autoimmune_disease', 'psoriasis', 'lupus'],
            'neuropsychological': ['depression', 'anxiety', 'ptsd', 'ocd', 'adhd',
                                 'bipolar', 'schizophrenia'],
            'metabolic': ['diabetes', 'obesity', 'metabolic_syndrome', 'pcos',
                         'insulin_resistance', 'fatty_liver'],
            'cardiovascular': ['hypertension', 'heart_disease', 'arrhythmia',
                             'atherosclerosis', 'stroke'],
            'respiratory': ['asthma', 'copd', 'lung_fibrosis', 'covid', 'long_covid'],
            'neurological': ['alzheimers', 'parkinsons', 'multiple_sclerosis',
                           'epilepsy', 'migraine', 'chronic_fatigue'],
            'gastrointestinal': ['ibs', 'crohns', 'ulcerative_colitis', 'gerd',
                               'leaky_gut', 'sibo'],
            'pain_related': ['chronic_pain', 'fibromyalgia', 'arthritis', 'migraine',
                           'neuropathy'],
            'developmental': ['autism', 'adhd', 'autism_sensory', 'learning_disabilities'],
            'sleep_related': ['insomnia', 'sleep_apnea', 'narcolepsy', 'restless_legs']
        }

        # Mechanism-intervention mappings
        self.mechanism_interventions = {
            'anti_inflammatory': ['omega_3', 'turmeric', 'curcumin', 'cold_exposure',
                                'anti_inflammatory_diet', 'probiotics'],
            'neuroprotective': ['meditation', 'exercise', 'omega_3', 'lion_mane_mushroom',
                              'nootropics', 'cold_exposure'],
            'metabolic_enhancing': ['exercise', 'intermittent_fasting', 'cold_exposure',
                                  'metformin', 'ketogenic_diet'],
            'stress_reducing': ['meditation', 'yoga', 'breathing_exercises', 'massage',
                              'mindfulness', 'therapy'],
            'microbiome_modulating': ['probiotics', 'prebiotics', 'fermented_foods',
                                    'fiber_supplement', 'fasting'],
            'circulation_improving': ['exercise', 'sauna', 'compression_therapy',
                                    'hyperbaric_oxygen', 'massage'],
            'sleep_promoting': ['melatonin', 'magnesium', 'weighted_blankets',
                              'sleep_hygiene', 'light_therapy'],
            'sensory_regulating': ['weighted_blankets', 'sensory_therapy', 'massage',
                                 'music_therapy', 'aromatherapy']
        }

        # Emerging interventions with high innovation potential
        self.emerging_interventions = {
            'hyperbaric_oxygen': 0.85,
            'cold_exposure': 0.80,
            'psychedelics': 0.90,
            'fecal_transplant': 0.88,
            'stem_cell_therapy': 0.95,
            'weighted_blankets': 0.60,
            'light_therapy': 0.65,
            'music_therapy': 0.55,
            'forest_bathing': 0.70
        }

    def identify_research_gaps(self,
                             knowledge_graph,
                             discovered_mechanisms: List,
                             fundamental_interventions: Dict) -> List[ResearchGap]:
        """
        Identify promising research gaps using multiple evidence sources.

        Args:
            knowledge_graph: MedicalKnowledgeGraph instance
            discovered_mechanisms: List of BiologicalMechanism objects
            fundamental_interventions: Dict of FundamentalIntervention objects

        Returns:
            List of promising research gaps ranked by potential
        """
        # Get all tested condition-intervention pairs
        tested_pairs = self._get_tested_pairs(knowledge_graph)

        # Get all possible conditions and interventions
        all_conditions = self._get_all_conditions(knowledge_graph, discovered_mechanisms)
        all_interventions = self._get_all_interventions(knowledge_graph, fundamental_interventions)

        # Generate untested pairs
        untested_pairs = self._generate_untested_pairs(all_conditions, all_interventions, tested_pairs)

        print(f"Found {len(untested_pairs)} untested condition-intervention pairs")

        # Predict effectiveness for each untested pair
        research_gaps = []
        for condition, intervention in untested_pairs:
            gap = self._predict_gap_potential(
                condition, intervention, knowledge_graph,
                discovered_mechanisms, fundamental_interventions
            )
            if gap and gap.predicted_score >= self.confidence_threshold:
                research_gaps.append(gap)

        # Sort by predicted score and priority
        research_gaps.sort(key=lambda x: (x.priority == 'HIGH', x.predicted_score), reverse=True)

        # Store for later access by other components
        self._identified_gaps = research_gaps

        # Limit results
        return research_gaps[:50]  # Top 50 research gaps

    def _get_tested_pairs(self, knowledge_graph) -> Set[Tuple[str, str]]:
        """Get all condition-intervention pairs that have been tested."""
        tested_pairs = set()

        for intervention in knowledge_graph.forward_edges:
            for condition in knowledge_graph.forward_edges[intervention]:
                tested_pairs.add((condition, intervention))

        return tested_pairs

    def _get_all_conditions(self, knowledge_graph, discovered_mechanisms: List) -> Set[str]:
        """Get all conditions from knowledge graph and discovered mechanisms."""
        conditions = set()

        # From knowledge graph
        for intervention_targets in knowledge_graph.forward_edges.values():
            conditions.update(intervention_targets.keys())

        # From discovered mechanisms
        for mechanism in discovered_mechanisms:
            conditions.update(mechanism.conditions)

        # Add some emerging conditions
        emerging_conditions = [
            'long_covid', 'chronic_fatigue', 'autism_sensory',
            'treatment_resistant_depression', 'microbiome_dysbiosis',
            'vaccine_hesitancy', 'digital_addiction', 'climate_anxiety'
        ]
        conditions.update(emerging_conditions)

        return conditions

    def _get_all_interventions(self, knowledge_graph, fundamental_interventions: Dict) -> Set[str]:
        """Get all interventions from various sources."""
        interventions = set()

        # From knowledge graph
        interventions.update(knowledge_graph.forward_edges.keys())

        # From fundamental interventions
        interventions.update(fundamental_interventions.keys())

        # Add emerging interventions
        interventions.update(self.emerging_interventions.keys())

        return interventions

    def _generate_untested_pairs(self, conditions: Set[str], interventions: Set[str],
                               tested_pairs: Set[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Generate all untested condition-intervention pairs."""
        untested_pairs = []

        for condition in conditions:
            for intervention in interventions:
                if (condition, intervention) not in tested_pairs:
                    untested_pairs.append((condition, intervention))

        return untested_pairs

    def _predict_gap_potential(self,
                             condition: str,
                             intervention: str,
                             knowledge_graph,
                             discovered_mechanisms: List,
                             fundamental_interventions: Dict) -> Optional[ResearchGap]:
        """Predict the potential of an untested condition-intervention pair."""

        evidence_sources = {}
        reasoning = []
        similar_conditions = []
        mechanism_matches = []

        # 1. Condition similarity analysis
        similarity_score = self._analyze_condition_similarity(
            condition, intervention, knowledge_graph
        )
        if similarity_score > 0:
            evidence_sources['condition_similarity'] = similarity_score
            similar_conditions = self._find_similar_conditions(condition, intervention, knowledge_graph)
            if similar_conditions:
                reasoning.append(f'Works for {len(similar_conditions)} similar conditions')

        # 2. Mechanism matching
        mechanism_score = self._analyze_mechanism_matching(
            condition, intervention, discovered_mechanisms
        )
        if mechanism_score > 0:
            evidence_sources['mechanism_match'] = mechanism_score
            mechanism_matches = self._find_mechanism_matches(condition, intervention, discovered_mechanisms)
            if mechanism_matches:
                reasoning.append('Mechanism match')

        # 3. Intervention track record
        track_record_score = self._analyze_intervention_track_record(
            intervention, knowledge_graph, fundamental_interventions
        )
        if track_record_score > 0:
            evidence_sources['track_record'] = track_record_score
            reasoning.append('Strong intervention track record')

        # 4. Innovation potential
        innovation_score = self._calculate_innovation_potential(intervention, condition)
        if innovation_score > 0.6:
            evidence_sources['innovation_potential'] = innovation_score
            reasoning.append('High innovation potential')

        # Calculate overall predicted score
        if not evidence_sources:
            return None

        predicted_score = self._calculate_predicted_score(evidence_sources)

        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(evidence_sources, reasoning)

        # Determine priority
        priority = self._determine_priority(predicted_score, innovation_score, evidence_sources)

        # Calculate research feasibility
        feasibility = self._calculate_research_feasibility(intervention, condition)

        return ResearchGap(
            condition=condition,
            intervention=intervention,
            predicted_score=predicted_score,
            confidence_level=confidence_level,
            reasoning=reasoning,
            priority=priority,
            evidence_sources=evidence_sources,
            similar_conditions=similar_conditions,
            mechanism_matches=mechanism_matches,
            innovation_potential=innovation_score,
            research_feasibility=feasibility
        )

    def _analyze_condition_similarity(self, target_condition: str, intervention: str,
                                    knowledge_graph) -> float:
        """Analyze if intervention works for conditions similar to target."""
        if intervention not in knowledge_graph.forward_edges:
            return 0.0

        tested_conditions = set(knowledge_graph.forward_edges[intervention].keys())
        similar_conditions = self._find_similar_conditions_for_target(target_condition)

        overlap = tested_conditions.intersection(similar_conditions)
        if not overlap:
            return 0.0

        # Calculate average effectiveness for similar conditions
        total_effectiveness = 0
        evidence_count = 0

        for condition in overlap:
            if condition in knowledge_graph.reverse_edges:
                if intervention in knowledge_graph.reverse_edges[condition]:
                    edges = knowledge_graph.reverse_edges[condition][intervention]
                    for edge in edges:
                        effectiveness = edge.weight * edge.evidence.confidence
                        total_effectiveness += effectiveness
                        evidence_count += 1

        if evidence_count == 0:
            return 0.0

        avg_effectiveness = total_effectiveness / evidence_count
        similarity_bonus = len(overlap) / len(similar_conditions) if similar_conditions else 0

        return max(0.0, min(1.0, avg_effectiveness * (1 + similarity_bonus)))

    def _find_similar_conditions_for_target(self, target_condition: str) -> Set[str]:
        """Find conditions similar to target condition."""
        similar = set()

        for cluster_name, conditions in self.condition_clusters.items():
            if target_condition in conditions:
                similar.update(conditions)
                break

        # Remove the target condition itself
        similar.discard(target_condition)

        return similar

    def _find_similar_conditions(self, target_condition: str, intervention: str,
                               knowledge_graph) -> List[str]:
        """Find similar conditions where intervention has been tested."""
        if intervention not in knowledge_graph.forward_edges:
            return []

        tested_conditions = set(knowledge_graph.forward_edges[intervention].keys())
        similar_conditions = self._find_similar_conditions_for_target(target_condition)

        return list(tested_conditions.intersection(similar_conditions))

    def _analyze_mechanism_matching(self, condition: str, intervention: str,
                                  discovered_mechanisms: List) -> float:
        """Analyze if intervention's mechanisms match condition's needs."""

        # Find which mechanisms the condition belongs to
        condition_mechanisms = []
        for mechanism in discovered_mechanisms:
            if condition in mechanism.conditions:
                condition_mechanisms.append(mechanism.name.lower())

        # Find which mechanisms the intervention targets
        intervention_mechanisms = []
        for mechanism_name, interventions in self.mechanism_interventions.items():
            if intervention in interventions:
                intervention_mechanisms.append(mechanism_name)

        if not condition_mechanisms or not intervention_mechanisms:
            return 0.0

        # Calculate mechanism overlap score
        # Simple keyword matching for now
        overlap_score = 0
        for cond_mech in condition_mechanisms:
            for int_mech in intervention_mechanisms:
                # Check for keyword overlap
                cond_words = set(cond_mech.split('_'))
                int_words = set(int_mech.split('_'))
                if cond_words.intersection(int_words):
                    overlap_score += 0.3
                # Special mechanism matching
                if ('neuro' in cond_mech and 'neuroprotective' in int_mech) or \
                   ('metabolic' in cond_mech and 'metabolic' in int_mech) or \
                   ('inflammatory' in cond_mech and 'anti_inflammatory' in int_mech):
                    overlap_score += 0.5

        return min(1.0, overlap_score)

    def _find_mechanism_matches(self, condition: str, intervention: str,
                              discovered_mechanisms: List) -> List[str]:
        """Find mechanism matches between condition and intervention."""
        matches = []

        condition_mechanisms = []
        for mechanism in discovered_mechanisms:
            if condition in mechanism.conditions:
                condition_mechanisms.append(mechanism.name)

        for mechanism_name, interventions in self.mechanism_interventions.items():
            if intervention in interventions:
                matches.append(mechanism_name)

        return matches

    def _analyze_intervention_track_record(self, intervention: str, knowledge_graph,
                                         fundamental_interventions: Dict) -> float:
        """Analyze intervention's overall track record."""

        # Check if it's a fundamental intervention
        if intervention in fundamental_interventions:
            fund_intervention = fundamental_interventions[intervention]
            return fund_intervention.fundamentality_score / 100.0  # Normalize to 0-1

        # Otherwise analyze from knowledge graph
        if intervention not in knowledge_graph.forward_edges:
            return 0.0

        total_effectiveness = 0
        condition_count = 0

        for condition, edges in knowledge_graph.forward_edges[intervention].items():
            if edges:
                condition_effectiveness = 0
                edge_count = 0
                for edge in edges:
                    effectiveness = edge.weight * edge.evidence.confidence
                    condition_effectiveness += effectiveness
                    edge_count += 1

                if edge_count > 0:
                    avg_condition_effectiveness = condition_effectiveness / edge_count
                    total_effectiveness += avg_condition_effectiveness
                    condition_count += 1

        if condition_count == 0:
            return 0.0

        track_record = total_effectiveness / condition_count
        breadth_bonus = min(0.3, condition_count / 10)  # Bonus for treating many conditions

        return max(0.0, min(1.0, track_record + breadth_bonus))

    def _calculate_innovation_potential(self, intervention: str, condition: str) -> float:
        """Calculate innovation potential of the research gap."""
        base_innovation = self.emerging_interventions.get(intervention, 0.3)

        # Bonus for emerging conditions
        emerging_conditions = ['long_covid', 'autism_sensory', 'climate_anxiety',
                             'digital_addiction', 'treatment_resistant_depression']
        if condition in emerging_conditions:
            base_innovation += 0.2

        return min(1.0, base_innovation)

    def _calculate_predicted_score(self, evidence_sources: Dict[str, float]) -> float:
        """Calculate overall predicted score from evidence sources."""
        if not evidence_sources:
            return 0.0

        # Weighted combination of evidence sources
        weights = {
            'condition_similarity': 0.4,
            'mechanism_match': 0.3,
            'track_record': 0.2,
            'innovation_potential': 0.1
        }

        weighted_score = 0
        total_weight = 0

        for source, score in evidence_sources.items():
            weight = weights.get(source, 0.1)
            weighted_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return min(1.0, weighted_score / total_weight)

    def _calculate_confidence_level(self, evidence_sources: Dict[str, float],
                                  reasoning: List[str]) -> str:
        """Calculate confidence level based on evidence strength."""
        num_sources = len(evidence_sources)
        avg_strength = sum(evidence_sources.values()) / num_sources if num_sources > 0 else 0

        if num_sources >= 3 and avg_strength >= 0.7:
            return 'VERY HIGH - Multiple strong indicators'
        elif num_sources >= 2 and avg_strength >= 0.6:
            return 'HIGH - Strong evidence from similar conditions'
        elif num_sources >= 2 or avg_strength >= 0.5:
            return 'MODERATE - Some supporting evidence'
        else:
            return 'LOW - Limited evidence available'

    def _determine_priority(self, predicted_score: float, innovation_score: float,
                          evidence_sources: Dict[str, float]) -> str:
        """Determine research priority level."""

        if predicted_score >= 0.8 and innovation_score >= 0.7:
            return 'HIGH'  # High potential, high innovation
        elif predicted_score >= 0.7 and len(evidence_sources) >= 2:
            return 'HIGH'  # Strong evidence from multiple sources
        elif predicted_score >= 0.6:
            return 'NORMAL'  # Good potential
        else:
            return 'LOW'  # Lower priority

    def _calculate_research_feasibility(self, intervention: str, condition: str) -> float:
        """Calculate how feasible this research would be to conduct."""

        # Simple heuristic based on intervention complexity
        complex_interventions = ['stem_cell_therapy', 'gene_therapy', 'psychedelics']
        simple_interventions = ['meditation', 'exercise', 'weighted_blankets', 'light_therapy']

        if intervention in complex_interventions:
            return 0.3  # Hard to study
        elif intervention in simple_interventions:
            return 0.9  # Easy to study
        else:
            return 0.6  # Moderate complexity

    def get_gaps_by_condition(self, research_gaps: List[ResearchGap],
                            condition: str) -> List[ResearchGap]:
        """Get research gaps for a specific condition."""
        return [gap for gap in research_gaps if gap.condition == condition]

    def get_gaps_by_intervention(self, research_gaps: List[ResearchGap],
                               intervention: str) -> List[ResearchGap]:
        """Get research gaps for a specific intervention."""
        return [gap for gap in research_gaps if gap.intervention == intervention]

    def get_high_priority_gaps(self, research_gaps: List[ResearchGap]) -> List[ResearchGap]:
        """Get high priority research gaps."""
        return [gap for gap in research_gaps if gap.priority == 'HIGH']

    def get_innovation_opportunities(self, research_gaps: List[ResearchGap],
                                   min_innovation: float = 0.7) -> List[ResearchGap]:
        """Get research gaps with high innovation potential."""
        return [gap for gap in research_gaps
                if gap.innovation_potential >= min_innovation]

    def predict_intervention_effectiveness(self, condition: str) -> List[Dict[str, Any]]:
        """
        Predict effectiveness of interventions for a given condition.

        Args:
            condition: Target condition

        Returns:
            List of intervention predictions with scores
        """
        if not hasattr(self, '_identified_gaps'):
            return []

        # Find research gaps for this condition
        condition_gaps = [gap for gap in self._identified_gaps
                         if gap.condition == condition]

        # Convert to prediction format
        predictions = []
        for gap in condition_gaps:
            predictions.append({
                'intervention': gap.intervention,
                'predicted_score': gap.predicted_score,
                'priority': gap.priority,
                'innovation_potential': gap.innovation_potential,
                'rationale': gap.reasoning
            })

        # Sort by predicted score
        predictions.sort(key=lambda x: x['predicted_score'], reverse=True)
        return predictions