"""
Biological Patterns Discovery System using NMF and Clustering.

Discovers hidden biological mechanisms (e.g., gut-brain axis) by finding
groups of conditions that respond to similar treatments without being
told what the patterns are.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import warnings

# ML imports
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class BiologicalMechanism:
    """Discovered biological mechanism with conditions and interventions."""
    name: str
    conditions: List[str]
    defining_interventions: List[str]
    strength: float
    mechanism_vector: np.ndarray
    condition_loadings: Dict[str, float]
    intervention_loadings: Dict[str, float]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert mechanism to dictionary format."""
        return {
            'name': self.name,
            'conditions': self.conditions,
            'defining_interventions': self.defining_interventions,
            'strength': round(self.strength, 2),
            'confidence': round(self.confidence, 2),
            'condition_loadings': {k: round(v, 3) for k, v in self.condition_loadings.items()},
            'intervention_loadings': {k: round(v, 3) for k, v in self.intervention_loadings.items()}
        }


class BiologicalPatternDiscovery:
    """
    Discovers hidden biological mechanisms using NMF and clustering.

    Key features:
    - Builds condition-intervention effectiveness matrix
    - Uses NMF to find latent biological mechanisms
    - Clusters conditions by mechanism similarity
    - Automatically names mechanisms based on intervention patterns
    - Validates discovered patterns
    """

    def __init__(self,
                 n_mechanisms: int = 10,
                 min_conditions_per_mechanism: int = 3,
                 min_interventions_per_mechanism: int = 3,
                 random_state: int = 42):
        """
        Initialize biological pattern discovery system.

        Args:
            n_mechanisms: Number of biological mechanisms to discover
            min_conditions_per_mechanism: Minimum conditions to form a mechanism
            min_interventions_per_mechanism: Minimum interventions to define a mechanism
            random_state: Random seed for reproducibility
        """
        self.n_mechanisms = n_mechanisms
        self.min_conditions_per_mechanism = min_conditions_per_mechanism
        self.min_interventions_per_mechanism = min_interventions_per_mechanism
        self.random_state = random_state

        # Core data structures
        self.condition_intervention_matrix = None
        self.conditions = []
        self.interventions = []
        self.mechanisms = []

        # ML models
        self.nmf_model = None
        self.scaler = StandardScaler()

        # Mechanism naming
        self.mechanism_keywords = {
            'neuropsychological': ['ssri', 'therapy', 'meditation', 'counseling', 'antidepressant', 'anxiolytic'],
            'gut_microbiome': ['probiotic', 'prebiotic', 'fiber', 'fodmap', 'fermented', 'microbiome'],
            'metabolic': ['metformin', 'insulin', 'glucose', 'diet', 'low_carb', 'ketogenic'],
            'inflammatory': ['anti_inflammatory', 'nsaid', 'omega_3', 'turmeric', 'curcumin'],
            'hormonal': ['hormone', 'estrogen', 'testosterone', 'thyroid', 'cortisol'],
            'cardiovascular': ['beta_blocker', 'ace_inhibitor', 'statin', 'cardio', 'heart'],
            'immune': ['immunosuppressant', 'vaccine', 'antibody', 'immune', 'autoimmune'],
            'musculoskeletal': ['physical_therapy', 'exercise', 'strength', 'mobility', 'joint']
        }

    def build_matrix_from_graph(self, knowledge_graph) -> np.ndarray:
        """
        Build condition-intervention effectiveness matrix from knowledge graph.

        Args:
            knowledge_graph: MedicalKnowledgeGraph instance

        Returns:
            Matrix where rows=conditions, columns=interventions, values=effectiveness
        """
        # Get all unique conditions and interventions
        all_conditions = set()
        all_interventions = set()

        # Collect from forward edges (intervention -> condition)
        for intervention, conditions in knowledge_graph.forward_edges.items():
            all_interventions.add(intervention)
            all_conditions.update(conditions.keys())

        self.conditions = sorted(list(all_conditions))
        self.interventions = sorted(list(all_interventions))

        # Build matrix
        matrix = np.zeros((len(self.conditions), len(self.interventions)))

        for i, condition in enumerate(self.conditions):
            for j, intervention in enumerate(self.interventions):
                # Get aggregated evidence for this condition-intervention pair
                if condition in knowledge_graph.reverse_edges:
                    if intervention in knowledge_graph.reverse_edges[condition]:
                        edges = knowledge_graph.reverse_edges[condition][intervention]
                        if edges:
                            # Calculate weighted average effectiveness
                            total_weight = sum(edge.weight * edge.evidence.confidence for edge in edges)
                            total_confidence = sum(edge.evidence.confidence for edge in edges)
                            effectiveness = total_weight / len(edges) if edges else 0
                            confidence_weight = total_confidence / len(edges) if edges else 0
                            matrix[i, j] = effectiveness * confidence_weight

        self.condition_intervention_matrix = matrix
        return matrix

    def discover_mechanisms(self, knowledge_graph) -> List[BiologicalMechanism]:
        """
        Discover biological mechanisms using NMF decomposition.

        Args:
            knowledge_graph: MedicalKnowledgeGraph instance

        Returns:
            List of discovered biological mechanisms
        """
        # Build the condition-intervention matrix
        matrix = self.build_matrix_from_graph(knowledge_graph)

        # Handle negative values for NMF (which requires non-negative input)
        # Shift matrix to make all values non-negative while preserving patterns
        min_val = matrix.min()
        if min_val < 0:
            matrix_shifted = matrix - min_val + 0.01  # Small positive offset
        else:
            matrix_shifted = matrix + 0.01  # Avoid zeros

        # Apply NMF to discover latent mechanisms
        self.nmf_model = NMF(
            n_components=self.n_mechanisms,
            random_state=self.random_state,
            max_iter=500,
            init='random'
        )

        # W: condition loadings (conditions x mechanisms)
        # H: intervention loadings (mechanisms x interventions)
        condition_loadings = self.nmf_model.fit_transform(matrix_shifted)
        intervention_loadings = self.nmf_model.components_

        # Extract mechanisms
        mechanisms = []
        for mechanism_idx in range(self.n_mechanisms):
            mechanism = self._extract_mechanism(
                mechanism_idx,
                condition_loadings,
                intervention_loadings,
                matrix
            )
            if mechanism and self._validate_mechanism(mechanism):
                mechanisms.append(mechanism)

        # Sort mechanisms by strength
        mechanisms.sort(key=lambda m: m.strength, reverse=True)
        self.mechanisms = mechanisms

        return mechanisms

    def _extract_mechanism(self,
                          mechanism_idx: int,
                          condition_loadings: np.ndarray,
                          intervention_loadings: np.ndarray,
                          original_matrix: np.ndarray) -> Optional[BiologicalMechanism]:
        """Extract a single mechanism from NMF components."""

        # Get condition loadings for this mechanism
        condition_scores = condition_loadings[:, mechanism_idx]
        condition_threshold = np.percentile(condition_scores, 70)  # Top 30% of conditions

        # Get intervention loadings for this mechanism
        intervention_scores = intervention_loadings[mechanism_idx, :]
        intervention_threshold = np.percentile(intervention_scores, 70)  # Top 30% of interventions

        # Extract conditions and interventions above threshold
        strong_conditions = [
            self.conditions[i] for i, score in enumerate(condition_scores)
            if score > condition_threshold
        ]

        strong_interventions = [
            self.interventions[j] for j, score in enumerate(intervention_scores)
            if score > intervention_threshold
        ]

        # Validate minimum requirements
        if (len(strong_conditions) < self.min_conditions_per_mechanism or
            len(strong_interventions) < self.min_interventions_per_mechanism):
            return None

        # Calculate mechanism strength (average effectiveness within mechanism)
        mechanism_strength = self._calculate_mechanism_strength(
            strong_conditions, strong_interventions, original_matrix
        )

        # Calculate confidence based on distinctiveness
        confidence = self._calculate_mechanism_confidence(
            condition_scores, intervention_scores
        )

        # Generate mechanism name
        mechanism_name = self._name_mechanism(strong_interventions)

        # Create condition and intervention loading dictionaries
        condition_loadings_dict = {
            condition: float(condition_scores[i])
            for i, condition in enumerate(self.conditions)
            if condition in strong_conditions
        }

        intervention_loadings_dict = {
            intervention: float(intervention_scores[j])
            for j, intervention in enumerate(self.interventions)
            if intervention in strong_interventions
        }

        return BiologicalMechanism(
            name=mechanism_name,
            conditions=strong_conditions,
            defining_interventions=strong_interventions,
            strength=mechanism_strength,
            mechanism_vector=intervention_scores,
            condition_loadings=condition_loadings_dict,
            intervention_loadings=intervention_loadings_dict,
            confidence=confidence
        )

    def _calculate_mechanism_strength(self,
                                    conditions: List[str],
                                    interventions: List[str],
                                    matrix: np.ndarray) -> float:
        """Calculate the average effectiveness within a mechanism."""
        condition_indices = [self.conditions.index(c) for c in conditions]
        intervention_indices = [self.interventions.index(i) for i in interventions]

        # Get submatrix for this mechanism
        submatrix = matrix[np.ix_(condition_indices, intervention_indices)]

        # Calculate mean absolute effectiveness (handles positive and negative)
        non_zero_values = submatrix[submatrix != 0]
        if len(non_zero_values) > 0:
            return float(np.abs(non_zero_values).mean())
        else:
            return 0.0

    def _calculate_mechanism_confidence(self,
                                      condition_scores: np.ndarray,
                                      intervention_scores: np.ndarray) -> float:
        """Calculate confidence based on how distinct this mechanism is."""
        # Confidence = how concentrated the loadings are (vs uniform distribution)
        condition_entropy = -np.sum(condition_scores * np.log(condition_scores + 1e-10))
        intervention_entropy = -np.sum(intervention_scores * np.log(intervention_scores + 1e-10))

        # Normalize by maximum possible entropy
        max_condition_entropy = np.log(len(condition_scores))
        max_intervention_entropy = np.log(len(intervention_scores))

        condition_distinctiveness = 1 - (condition_entropy / max_condition_entropy)
        intervention_distinctiveness = 1 - (intervention_entropy / max_intervention_entropy)

        return float((condition_distinctiveness + intervention_distinctiveness) / 2)

    def _name_mechanism(self, interventions: List[str]) -> str:
        """Automatically name mechanism based on intervention patterns."""
        intervention_text = ' '.join(interventions).lower()

        # Score each mechanism type based on keyword matches
        scores = {}
        for mechanism_type, keywords in self.mechanism_keywords.items():
            score = sum(1 for keyword in keywords if keyword in intervention_text)
            if score > 0:
                scores[mechanism_type] = score

        # Return best match or generic name
        if scores:
            best_match = max(scores, key=scores.get)
            return best_match.replace('_', ' ').title()
        else:
            # Generate name from most common intervention types
            intervention_words = []
            for intervention in interventions[:3]:  # Top 3 interventions
                intervention_words.extend(intervention.split('_'))

            if intervention_words:
                common_word = Counter(intervention_words).most_common(1)[0][0]
                return f"{common_word.title()} Mechanism"
            else:
                return "Unknown Mechanism"

    def _validate_mechanism(self, mechanism: BiologicalMechanism) -> bool:
        """Validate that a discovered mechanism meets quality criteria."""
        return (
            len(mechanism.conditions) >= self.min_conditions_per_mechanism and
            len(mechanism.defining_interventions) >= self.min_interventions_per_mechanism and
            mechanism.strength > 0.1 and  # Minimum strength threshold
            mechanism.confidence > 0.2     # Minimum confidence threshold
        )

    def get_mechanism_summary(self) -> Dict[str, Any]:
        """Get summary of all discovered mechanisms."""
        if not self.mechanisms:
            return {'error': 'No mechanisms discovered yet. Run discover_mechanisms() first.'}

        summary = {}
        for i, mechanism in enumerate(self.mechanisms):
            summary[mechanism.name] = mechanism.to_dict()

        return summary

    def find_cross_mechanism_conditions(self) -> Dict[str, List[str]]:
        """Find conditions that appear in multiple mechanisms (cross-talk)."""
        condition_mechanisms = defaultdict(list)

        for mechanism in self.mechanisms:
            for condition in mechanism.conditions:
                condition_mechanisms[condition].append(mechanism.name)

        # Return conditions that appear in multiple mechanisms
        cross_talk = {
            condition: mechanisms
            for condition, mechanisms in condition_mechanisms.items()
            if len(mechanisms) > 1
        }

        return cross_talk

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about discovered patterns."""
        if not self.mechanisms:
            return {'error': 'No mechanisms discovered yet.'}

        stats = {
            'total_mechanisms': len(self.mechanisms),
            'avg_conditions_per_mechanism': np.mean([len(m.conditions) for m in self.mechanisms]),
            'avg_interventions_per_mechanism': np.mean([len(m.defining_interventions) for m in self.mechanisms]),
            'avg_mechanism_strength': np.mean([m.strength for m in self.mechanisms]),
            'avg_mechanism_confidence': np.mean([m.confidence for m in self.mechanisms]),
            'matrix_shape': self.condition_intervention_matrix.shape if self.condition_intervention_matrix is not None else None,
            'cross_talk_conditions': len(self.find_cross_mechanism_conditions())
        }

        return stats

    def get_condition_mechanisms(self, condition: str) -> List[str]:
        """
        Get mechanisms that are relevant for a given condition.

        Args:
            condition: Condition name

        Returns:
            List of mechanism names that involve this condition
        """
        if not self.mechanisms:
            return []

        relevant_mechanisms = []
        for mechanism in self.mechanisms:
            if condition in mechanism.conditions:
                relevant_mechanisms.append(mechanism.name)

        return relevant_mechanisms

    def get_intervention_mechanisms(self, intervention: str) -> List[str]:
        """
        Get mechanisms that are influenced by a given intervention.

        Args:
            intervention: Intervention name

        Returns:
            List of mechanism names that involve this intervention
        """
        if not self.mechanisms:
            return []

        relevant_mechanisms = []
        for mechanism in self.mechanisms:
            if intervention in mechanism.defining_interventions:
                relevant_mechanisms.append(mechanism.name)

        return relevant_mechanisms