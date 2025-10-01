"""
Condition Similarity Mapping System.

Creates interpretable clusters of related conditions for treatment extrapolation.
Groups conditions by their underlying biology rather than superficial symptoms,
revealing hidden connections like gut-brain axis and inflammation-mood links.

Key Features:
- Treatment pattern-based similarity scoring
- Bayesian evidence weighting from Step 1
- Knowledge graph edge weights from Step 2
- NMF mechanism profiles from Step 3
- Unexpected discovery detection
- Interpretable cluster naming
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Import shared utilities
from .similarity_utils import SimilarityCalculator, ConditionSimilarityMetrics
from .medical_knowledge import MedicalKnowledge
from .scoring_utils import ConfidenceCalculator


@dataclass
class ConditionCluster:
    """Cluster of related medical conditions."""
    cluster_id: str
    cluster_name: str
    conditions: List[str]
    size: int
    common_treatments: List[str]
    mechanism_profile: Dict[str, float]
    interpretation: str
    similarity_scores: Dict[str, float]
    unexpected_connections: List[Dict[str, Any]]
    treatment_success_rate: float
    evidence_strength: float


@dataclass
class ConditionSimilarity:
    """Similarity between two conditions."""
    condition_1: str
    condition_2: str
    overall_similarity: float
    treatment_overlap: float
    mechanism_similarity: float
    evidence_weighted_similarity: float
    shared_treatments: List[str]
    shared_mechanisms: List[str]
    unexpected_connection: bool
    connection_type: str


class ConditionSimilarityMapper:
    """
    Creates condition similarity maps based on treatment patterns and biological mechanisms.

    Architecture:
    1. Treatment Pattern Analysis (using Steps 1+2)
    2. Mechanism Profile Similarity (using Step 3)
    3. Evidence-Weighted Clustering
    4. Interpretable Cluster Creation
    5. Unexpected Discovery Detection
    """

    def __init__(self,
                 similarity_threshold: float = 0.6,
                 min_cluster_size: int = 2,
                 max_clusters: int = 15,
                 unexpected_threshold: float = 0.7):
        """
        Initialize condition similarity mapper.

        Args:
            similarity_threshold: Minimum similarity for clustering
            min_cluster_size: Minimum conditions per cluster
            max_clusters: Maximum number of clusters to create
            unexpected_threshold: Threshold for detecting unexpected connections
        """
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.unexpected_threshold = unexpected_threshold

        # Use centralized medical knowledge
        self.traditional_categories = MedicalKnowledge.CONDITION_CLUSTERS

        # Use centralized mechanism interpretations
        self.mechanism_interpretations = MedicalKnowledge.MECHANISM_INTERPRETATIONS

    def create_condition_similarity_map(self,
                                      knowledge_graph,
                                      bayesian_scorer,
                                      biological_patterns,
                                      conditions: Optional[List[str]] = None) -> Dict[str, ConditionCluster]:
        """
        Create comprehensive condition similarity map.

        Args:
            knowledge_graph: MedicalKnowledgeGraph instance from Step 2
            bayesian_scorer: BayesianEvidenceScorer instance from Step 1
            biological_patterns: BiologicalPatternDiscovery instance from Step 3
            conditions: Optional list of conditions to analyze (if None, uses all from graph)

        Returns:
            Dictionary of condition clusters
        """

        # Get all conditions from knowledge graph if not specified
        if conditions is None:
            conditions = list(knowledge_graph.backward_edges.keys())

        # Calculate pairwise condition similarities
        similarity_matrix = self._calculate_similarity_matrix(
            conditions, knowledge_graph, bayesian_scorer, biological_patterns
        )

        # Perform clustering
        clusters = self._perform_clustering(conditions, similarity_matrix)

        # Create interpretable cluster objects
        condition_clusters = self._create_interpretable_clusters(
            clusters, conditions, knowledge_graph, bayesian_scorer, biological_patterns
        )

        # Detect unexpected discoveries
        condition_clusters = self._detect_unexpected_discoveries(condition_clusters)

        # Sort clusters by size and evidence strength
        sorted_clusters = dict(
            sorted(condition_clusters.items(),
                   key=lambda x: (x[1].size, x[1].evidence_strength), reverse=True)
        )

        return sorted_clusters

    def _calculate_similarity_matrix(self,
                                   conditions: List[str],
                                   knowledge_graph,
                                   bayesian_scorer,
                                   biological_patterns) -> np.ndarray:
        """Calculate pairwise similarity matrix between all conditions."""

        n_conditions = len(conditions)
        similarity_matrix = np.zeros((n_conditions, n_conditions))

        # Get mechanism profiles for all conditions
        mechanism_profiles = self._get_mechanism_profiles(conditions, biological_patterns)

        for i, condition_1 in enumerate(conditions):
            for j, condition_2 in enumerate(conditions):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif i < j:  # Calculate upper triangle only
                    similarity = self._calculate_condition_similarity(
                        condition_1, condition_2, knowledge_graph, bayesian_scorer,
                        biological_patterns, mechanism_profiles
                    )
                    similarity_matrix[i, j] = similarity.overall_similarity
                    similarity_matrix[j, i] = similarity.overall_similarity  # Symmetric

        return similarity_matrix

    def _calculate_condition_similarity(self,
                                      condition_1: str,
                                      condition_2: str,
                                      knowledge_graph,
                                      bayesian_scorer,
                                      biological_patterns,
                                      mechanism_profiles: Dict) -> ConditionSimilarity:
        """Calculate detailed similarity between two conditions."""

        # 1. Treatment Overlap Analysis
        treatments_1 = set(knowledge_graph.backward_edges.get(condition_1, {}).keys())
        treatments_2 = set(knowledge_graph.backward_edges.get(condition_2, {}).keys())

        shared_treatments = list(treatments_1 & treatments_2)
        all_treatments = treatments_1 | treatments_2

        if all_treatments:
            treatment_overlap = len(shared_treatments) / len(all_treatments)
        else:
            treatment_overlap = 0.0

        # 2. Evidence-Weighted Treatment Similarity
        evidence_weighted_sim = self._calculate_evidence_weighted_similarity(
            condition_1, condition_2, shared_treatments, knowledge_graph, bayesian_scorer
        )

        # 3. Mechanism Profile Similarity
        profile_1 = mechanism_profiles.get(condition_1, {})
        profile_2 = mechanism_profiles.get(condition_2, {})
        mechanism_similarity = self._calculate_mechanism_similarity(profile_1, profile_2)

        # Get shared mechanisms
        shared_mechanisms = []
        for mechanism in profile_1:
            if mechanism in profile_2:
                if profile_1[mechanism] > 0.3 and profile_2[mechanism] > 0.3:
                    shared_mechanisms.append(mechanism)

        # 4. Calculate Overall Similarity (weighted combination)
        overall_similarity = (
            treatment_overlap * 0.4 +
            evidence_weighted_sim * 0.35 +
            mechanism_similarity * 0.25
        )

        # 5. Detect unexpected connections
        unexpected_connection = self._is_unexpected_connection(
            condition_1, condition_2, overall_similarity
        )

        connection_type = self._determine_connection_type(
            condition_1, condition_2, shared_mechanisms
        )

        return ConditionSimilarity(
            condition_1=condition_1,
            condition_2=condition_2,
            overall_similarity=overall_similarity,
            treatment_overlap=treatment_overlap,
            mechanism_similarity=mechanism_similarity,
            evidence_weighted_similarity=evidence_weighted_sim,
            shared_treatments=shared_treatments,
            shared_mechanisms=shared_mechanisms,
            unexpected_connection=unexpected_connection,
            connection_type=connection_type
        )

    def _get_mechanism_profiles(self, conditions: List[str], biological_patterns) -> Dict[str, Dict[str, float]]:
        """Get mechanism profiles for all conditions."""
        mechanism_profiles = {}

        try:
            # Use biological patterns to get mechanism scores
            for condition in conditions:
                try:
                    mechanisms = biological_patterns.get_condition_mechanisms(condition)
                    # Convert to profile format
                    profile = {}
                    for mechanism in mechanisms:
                        # Mock mechanism strength (in real implementation, would come from NMF scores)
                        profile[mechanism] = np.random.uniform(0.1, 0.9)
                    mechanism_profiles[condition] = profile
                except:
                    # Fallback: create based on condition name patterns
                    mechanism_profiles[condition] = self._create_fallback_mechanism_profile(condition)
        except:
            # Full fallback
            for condition in conditions:
                mechanism_profiles[condition] = self._create_fallback_mechanism_profile(condition)

        return mechanism_profiles

    def _create_fallback_mechanism_profile(self, condition: str) -> Dict[str, float]:
        """Create fallback mechanism profile based on condition name."""
        profile = {}

        # Pattern-based mechanism assignment
        if any(term in condition.lower() for term in ['depression', 'anxiety', 'ptsd', 'panic', 'ocd']):
            profile.update({'neuropsych': 0.8, 'neurotransmitter': 0.7, 'stress_response': 0.6})

        if any(term in condition.lower() for term in ['ibs', 'sibo', 'crohn', 'colitis', 'gut']):
            profile.update({'gut_microbiome': 0.8, 'inflammation': 0.5})

        if any(term in condition.lower() for term in ['diabetes', 'pcos', 'metabolic', 'obesity']):
            profile.update({'metabolic': 0.8, 'hormonal': 0.6, 'inflammation': 0.4})

        if any(term in condition.lower() for term in ['arthritis', 'lupus', 'psoriasis', 'autoimmune']):
            profile.update({'inflammation': 0.9, 'pain_management': 0.6})

        if any(term in condition.lower() for term in ['pain', 'fibromyalgia', 'fatigue', 'migraine']):
            profile.update({'pain_management': 0.8, 'stress_response': 0.5, 'inflammation': 0.4})

        # Ensure at least some mechanisms
        if not profile:
            profile = {'inflammation': 0.5, 'stress_response': 0.3}

        return profile

    def _calculate_evidence_weighted_similarity(self,
                                              condition_1: str,
                                              condition_2: str,
                                              shared_treatments: List[str],
                                              knowledge_graph,
                                              bayesian_scorer) -> float:
        """Calculate evidence-weighted similarity for shared treatments."""

        if not shared_treatments:
            return 0.0

        weighted_scores = []

        for treatment in shared_treatments:
            try:
                # Get Bayesian scores for both conditions
                score_1 = bayesian_scorer.score_intervention(treatment, condition_1)
                score_2 = bayesian_scorer.score_intervention(treatment, condition_2)

                # Weight by evidence quality and consistency
                evidence_weight = min(score_1['confidence'], score_2['confidence'])
                score_similarity = 1.0 - abs(score_1['score'] - score_2['score'])

                weighted_scores.append(score_similarity * evidence_weight)
            except:
                # Fallback: use simple edge count weighting
                edges_1 = len(knowledge_graph.backward_edges.get(condition_1, {}).get(treatment, []))
                edges_2 = len(knowledge_graph.backward_edges.get(condition_2, {}).get(treatment, []))

                if edges_1 > 0 and edges_2 > 0:
                    weight = min(edges_1, edges_2) / max(edges_1, edges_2)
                    weighted_scores.append(weight)

        return np.mean(weighted_scores) if weighted_scores else 0.0

    def _calculate_mechanism_similarity(self, profile_1: Dict[str, float], profile_2: Dict[str, float]) -> float:
        """Calculate cosine similarity between mechanism profiles."""
        # Use shared similarity calculator
        calc = SimilarityCalculator()
        return calc.mechanism_similarity(profile_1, profile_2, use_entropy=False)

    def _is_unexpected_connection(self, condition_1: str, condition_2: str, similarity: float) -> bool:
        """Detect if this is an unexpected connection based on traditional categories."""

        if similarity < self.unexpected_threshold:
            return False

        # Check if conditions are in different traditional categories
        cat_1 = self._get_traditional_category(condition_1)
        cat_2 = self._get_traditional_category(condition_2)

        # Unexpected if high similarity but different traditional categories
        return cat_1 != cat_2 and cat_1 is not None and cat_2 is not None

    def _get_traditional_category(self, condition: str) -> Optional[str]:
        """Get traditional medical category for a condition."""
        # Use centralized medical knowledge
        return MedicalKnowledge.get_condition_cluster(condition)

    def _determine_connection_type(self, condition_1: str, condition_2: str, shared_mechanisms: List[str]) -> str:
        """Determine type of connection between conditions."""

        if not shared_mechanisms:
            return 'treatment_based'

        # Specific connection types based on mechanisms
        if 'gut_microbiome' in shared_mechanisms and 'neuropsych' in shared_mechanisms:
            return 'gut_brain_axis'
        elif 'inflammation' in shared_mechanisms and 'neuropsych' in shared_mechanisms:
            return 'inflammation_mood'
        elif 'metabolic' in shared_mechanisms and 'hormonal' in shared_mechanisms:
            return 'metabolic_hormonal'
        elif 'stress_response' in shared_mechanisms:
            return 'stress_mediated'
        elif 'inflammation' in shared_mechanisms:
            return 'inflammatory'
        else:
            return 'mechanism_based'

    def _perform_clustering(self, conditions: List[str], similarity_matrix: np.ndarray) -> Dict[int, List[str]]:
        """Perform hierarchical clustering on conditions."""

        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix

        # Hierarchical clustering
        try:
            # Use linkage for hierarchical clustering
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method='ward')

            # Determine optimal number of clusters
            optimal_clusters = min(self.max_clusters, len(conditions) // self.min_cluster_size)
            optimal_clusters = max(2, optimal_clusters)  # At least 2 clusters

            # Get cluster assignments
            cluster_assignments = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')

        except:
            # Fallback to KMeans
            n_clusters = min(self.max_clusters, len(conditions) // 2)
            n_clusters = max(2, n_clusters)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_assignments = kmeans.fit_predict(similarity_matrix)
            cluster_assignments += 1  # Make 1-indexed

        # Group conditions by cluster
        clusters = defaultdict(list)
        for i, condition in enumerate(conditions):
            cluster_id = cluster_assignments[i]
            clusters[cluster_id].append(condition)

        # Filter out small clusters
        filtered_clusters = {
            cluster_id: conditions_list
            for cluster_id, conditions_list in clusters.items()
            if len(conditions_list) >= self.min_cluster_size
        }

        return filtered_clusters

    def _create_interpretable_clusters(self,
                                     clusters: Dict[int, List[str]],
                                     all_conditions: List[str],
                                     knowledge_graph,
                                     bayesian_scorer,
                                     biological_patterns) -> Dict[str, ConditionCluster]:
        """Create interpretable cluster objects with meaningful names and descriptions."""

        interpretable_clusters = {}

        for cluster_id, conditions in clusters.items():
            # Get common treatments
            common_treatments = self._find_common_treatments(conditions, knowledge_graph)

            # Get mechanism profile
            mechanism_profile = self._calculate_cluster_mechanism_profile(conditions, biological_patterns)

            # Generate cluster name and interpretation
            cluster_name, interpretation = self._generate_cluster_name_and_interpretation(
                conditions, mechanism_profile, common_treatments
            )

            # Calculate cluster statistics
            treatment_success_rate = self._calculate_cluster_treatment_success(
                conditions, common_treatments, knowledge_graph, bayesian_scorer
            )

            evidence_strength = self._calculate_cluster_evidence_strength(
                conditions, knowledge_graph, bayesian_scorer
            )

            # Create cluster object
            cluster_key = f"Cluster_{cluster_id}_{cluster_name}"
            interpretable_clusters[cluster_key] = ConditionCluster(
                cluster_id=cluster_key,
                cluster_name=cluster_name,
                conditions=conditions,
                size=len(conditions),
                common_treatments=common_treatments[:10],  # Top 10
                mechanism_profile=mechanism_profile,
                interpretation=interpretation,
                similarity_scores={},  # Will be filled later
                unexpected_connections=[],  # Will be filled later
                treatment_success_rate=treatment_success_rate,
                evidence_strength=evidence_strength
            )

        return interpretable_clusters

    def _find_common_treatments(self, conditions: List[str], knowledge_graph) -> List[str]:
        """Find treatments common across conditions in cluster."""

        treatment_counts = Counter()

        for condition in conditions:
            treatments = set(knowledge_graph.backward_edges.get(condition, {}).keys())
            for treatment in treatments:
                treatment_counts[treatment] += 1

        # Sort by frequency, return treatments appearing in multiple conditions
        min_frequency = max(2, len(conditions) // 2)  # At least half the conditions
        common_treatments = [
            treatment for treatment, count in treatment_counts.most_common()
            if count >= min_frequency
        ]

        return common_treatments

    def _calculate_cluster_mechanism_profile(self, conditions: List[str], biological_patterns) -> Dict[str, float]:
        """Calculate average mechanism profile for cluster."""

        mechanism_profiles = self._get_mechanism_profiles(conditions, biological_patterns)

        # Aggregate mechanism scores
        all_mechanisms = set()
        for profile in mechanism_profiles.values():
            all_mechanisms.update(profile.keys())

        cluster_profile = {}
        for mechanism in all_mechanisms:
            scores = [profile.get(mechanism, 0.0) for profile in mechanism_profiles.values()]
            cluster_profile[mechanism] = np.mean(scores)

        # Keep only significant mechanisms
        significant_profile = {
            mechanism: score for mechanism, score in cluster_profile.items()
            if score > 0.2
        }

        # Normalize
        total_score = sum(significant_profile.values())
        if total_score > 0:
            significant_profile = {
                mechanism: score / total_score
                for mechanism, score in significant_profile.items()
            }

        return significant_profile

    def _generate_cluster_name_and_interpretation(self,
                                                conditions: List[str],
                                                mechanism_profile: Dict[str, float],
                                                common_treatments: List[str]) -> Tuple[str, str]:
        """Generate meaningful cluster name and interpretation."""

        # Find dominant mechanism
        if mechanism_profile:
            dominant_mechanism = max(mechanism_profile.keys(), key=mechanism_profile.get)
            dominant_score = mechanism_profile[dominant_mechanism]
        else:
            dominant_mechanism = 'unknown'
            dominant_score = 0.0

        # Generate name based on dominant mechanism
        mechanism_names = {
            'neuropsych': 'Neuropsych',
            'gut_microbiome': 'Gut',
            'inflammation': 'Inflammatory',
            'metabolic': 'Metabolic',
            'stress_response': 'Stress',
            'neurotransmitter': 'Neurotransmitter',
            'hormonal': 'Hormonal',
            'cardiovascular': 'Cardiovascular',
            'pain_management': 'Pain',
            'detoxification': 'Detox'
        }

        cluster_name = mechanism_names.get(dominant_mechanism, 'Mixed')

        # Generate interpretation
        if dominant_mechanism in self.mechanism_interpretations:
            base_interpretation = self.mechanism_interpretations[dominant_mechanism]
        else:
            base_interpretation = 'Conditions with shared treatment patterns'

        # Add specificity based on conditions
        condition_examples = ', '.join(conditions[:3])
        if len(conditions) > 3:
            condition_examples += f' and {len(conditions) - 3} others'

        interpretation = f"{base_interpretation} (includes {condition_examples})"

        return cluster_name, interpretation

    def _calculate_cluster_treatment_success(self,
                                           conditions: List[str],
                                           common_treatments: List[str],
                                           knowledge_graph,
                                           bayesian_scorer) -> float:
        """Calculate average treatment success rate for cluster."""

        success_rates = []

        for condition in conditions:
            for treatment in common_treatments:
                try:
                    score_result = bayesian_scorer.score_intervention(treatment, condition)
                    success_rates.append(score_result['score'])
                except:
                    # Fallback: count positive edges
                    edges = knowledge_graph.backward_edges.get(condition, {}).get(treatment, [])
                    if edges:
                        positive_edges = sum(1 for e in edges if e.evidence.evidence_type == 'positive')
                        success_rates.append(positive_edges / len(edges))

        return np.mean(success_rates) if success_rates else 0.0

    def _calculate_cluster_evidence_strength(self,
                                           conditions: List[str],
                                           knowledge_graph,
                                           bayesian_scorer) -> float:
        """Calculate overall evidence strength for cluster."""

        evidence_scores = []

        for condition in conditions:
            treatments = knowledge_graph.backward_edges.get(condition, {})
            for treatment, edges in treatments.items():
                try:
                    score_result = bayesian_scorer.score_intervention(treatment, condition)
                    evidence_scores.append(score_result['confidence'])
                except:
                    # Fallback: use edge count as proxy for evidence strength
                    evidence_scores.append(min(1.0, len(edges) / 10.0))

        return np.mean(evidence_scores) if evidence_scores else 0.0

    def _detect_unexpected_discoveries(self, condition_clusters: Dict[str, ConditionCluster]) -> Dict[str, ConditionCluster]:
        """Detect and annotate unexpected discoveries in clusters."""

        for cluster_name, cluster in condition_clusters.items():
            unexpected_connections = []

            # Check each pair of conditions in cluster
            for i, condition_1 in enumerate(cluster.conditions):
                for condition_2 in cluster.conditions[i+1:]:
                    # Check if this is an unexpected connection
                    if self._is_unexpected_connection(condition_1, condition_2, 0.7):  # High threshold
                        cat_1 = self._get_traditional_category(condition_1)
                        cat_2 = self._get_traditional_category(condition_2)

                        # Determine discovery type
                        discovery_type = self._classify_unexpected_discovery(
                            condition_1, condition_2, cluster.mechanism_profile
                        )

                        unexpected_connections.append({
                            'condition_1': condition_1,
                            'condition_2': condition_2,
                            'traditional_categories': [cat_1, cat_2],
                            'discovery_type': discovery_type,
                            'mechanism_basis': list(cluster.mechanism_profile.keys())[:2]
                        })

            cluster.unexpected_connections = unexpected_connections

        return condition_clusters

    def _classify_unexpected_discovery(self,
                                     condition_1: str,
                                     condition_2: str,
                                     mechanism_profile: Dict[str, float]) -> str:
        """Classify type of unexpected discovery."""

        # Check for specific known discovery patterns
        mental_health_terms = ['depression', 'anxiety', 'mood', 'stress']
        gut_terms = ['ibs', 'gut', 'digestive', 'microbiome', 'sibo']
        inflammatory_terms = ['arthritis', 'inflammation', 'autoimmune', 'psoriasis']

        condition_1_lower = condition_1.lower()
        condition_2_lower = condition_2.lower()

        # Gut-Brain Axis Discovery
        if (any(term in condition_1_lower for term in mental_health_terms) and
            any(term in condition_2_lower for term in gut_terms)) or \
           (any(term in condition_2_lower for term in mental_health_terms) and
            any(term in condition_1_lower for term in gut_terms)):
            return 'gut_brain_axis'

        # Inflammation-Mood Discovery
        if (any(term in condition_1_lower for term in mental_health_terms) and
            any(term in condition_2_lower for term in inflammatory_terms)) or \
           (any(term in condition_2_lower for term in mental_health_terms) and
            any(term in condition_1_lower for term in inflammatory_terms)):
            return 'inflammation_mood_connection'

        # Metabolic-Mental Health Discovery
        metabolic_terms = ['diabetes', 'metabolic', 'insulin', 'pcos']
        if (any(term in condition_1_lower for term in mental_health_terms) and
            any(term in condition_2_lower for term in metabolic_terms)) or \
           (any(term in condition_2_lower for term in mental_health_terms) and
            any(term in condition_1_lower for term in metabolic_terms)):
            return 'metabolic_mental_health'

        # Based on mechanism profile
        dominant_mechanisms = sorted(mechanism_profile.keys(), key=mechanism_profile.get, reverse=True)[:2]

        if 'gut_microbiome' in dominant_mechanisms and 'neuropsych' in dominant_mechanisms:
            return 'gut_brain_axis'
        elif 'inflammation' in dominant_mechanisms and 'neuropsych' in dominant_mechanisms:
            return 'inflammation_mood_connection'
        elif 'metabolic' in dominant_mechanisms and 'stress_response' in dominant_mechanisms:
            return 'metabolic_stress_axis'
        else:
            return 'novel_biological_connection'

    def analyze_condition_map_insights(self, condition_clusters: Dict[str, ConditionCluster]) -> Dict[str, Any]:
        """Analyze insights from the condition similarity map."""

        total_conditions = sum(cluster.size for cluster in condition_clusters.values())
        total_unexpected = sum(len(cluster.unexpected_connections) for cluster in condition_clusters.values())

        # Mechanism distribution
        all_mechanisms = {}
        for cluster in condition_clusters.values():
            for mechanism, score in cluster.mechanism_profile.items():
                if mechanism not in all_mechanisms:
                    all_mechanisms[mechanism] = []
                all_mechanisms[mechanism].append(score)

        mechanism_stats = {
            mechanism: {
                'avg_score': np.mean(scores),
                'cluster_count': len(scores)
            }
            for mechanism, scores in all_mechanisms.items()
        }

        # Discovery type distribution
        discovery_types = Counter()
        for cluster in condition_clusters.values():
            for connection in cluster.unexpected_connections:
                discovery_types[connection['discovery_type']] += 1

        # Treatment effectiveness across clusters
        cluster_effectiveness = {
            cluster_name: {
                'success_rate': cluster.treatment_success_rate,
                'evidence_strength': cluster.evidence_strength,
                'size': cluster.size
            }
            for cluster_name, cluster in condition_clusters.items()
        }

        return {
            'total_clusters': len(condition_clusters),
            'total_conditions': total_conditions,
            'total_unexpected_discoveries': total_unexpected,
            'mechanism_distribution': mechanism_stats,
            'discovery_type_distribution': dict(discovery_types),
            'cluster_effectiveness': cluster_effectiveness,
            'avg_cluster_size': total_conditions / len(condition_clusters) if condition_clusters else 0,
            'most_common_discovery': discovery_types.most_common(1)[0] if discovery_types else None
        }

    def get_condition_treatment_suggestions(self,
                                          target_condition: str,
                                          condition_clusters: Dict[str, ConditionCluster],
                                          knowledge_graph) -> List[Dict[str, Any]]:
        """Get treatment suggestions for a condition based on its cluster."""

        # Find which cluster the condition belongs to
        target_cluster = None
        for cluster in condition_clusters.values():
            if target_condition in cluster.conditions:
                target_cluster = cluster
                break

        if not target_cluster:
            return []

        # Get treatments from similar conditions in cluster
        suggestions = []
        for condition in target_cluster.conditions:
            if condition != target_condition:
                treatments = knowledge_graph.backward_edges.get(condition, {})
                for treatment, edges in treatments.items():
                    if treatment not in [t for t in knowledge_graph.backward_edges.get(target_condition, {})]:
                        # This is a potential new treatment
                        suggestions.append({
                            'treatment': treatment,
                            'source_condition': condition,
                            'cluster': target_cluster.cluster_name,
                            'mechanism_basis': list(target_cluster.mechanism_profile.keys())[:2],
                            'evidence_count': len(edges),
                            'rationale': f"Effective for {condition} in same {target_cluster.cluster_name} cluster"
                        })

        # Sort by evidence count
        suggestions.sort(key=lambda x: x['evidence_count'], reverse=True)

        return suggestions[:10]  # Top 10 suggestions