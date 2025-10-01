"""
Centralized similarity calculation utilities for data mining modules.
Eliminates redundancy across multiple files that calculate similar metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from scipy.stats import entropy
from scipy.spatial.distance import jaccard


class SimilarityCalculator:
    """Unified similarity calculation utilities."""

    @staticmethod
    def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            Cosine similarity score [0, 1]
        """
        if len(vector1) == 0 or len(vector2) == 0:
            return 0.0

        # Handle 1D arrays
        if vector1.ndim == 1:
            vector1 = vector1.reshape(1, -1)
        if vector2.ndim == 1:
            vector2 = vector2.reshape(1, -1)

        try:
            similarity = sklearn_cosine(vector1, vector2)[0, 0]
            return float(max(0, min(1, similarity)))  # Ensure [0, 1] range
        except:
            return 0.0

    @staticmethod
    def jaccard_similarity(set1: Set, set2: Set) -> float:
        """
        Calculate Jaccard similarity between two sets.

        Args:
            set1: First set
            set2: Second set

        Returns:
            Jaccard similarity score [0, 1]
        """
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def dice_coefficient(set1: Set, set2: Set) -> float:
        """
        Calculate Dice coefficient between two sets.

        Args:
            set1: First set
            set2: Second set

        Returns:
            Dice coefficient [0, 1]
        """
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        return (2.0 * intersection) / (len(set1) + len(set2))

    @staticmethod
    def mechanism_similarity(
        profile1: Dict[str, float],
        profile2: Dict[str, float],
        use_entropy: bool = False
    ) -> float:
        """
        Calculate similarity between two mechanism profiles.

        Args:
            profile1: First mechanism profile {mechanism: score}
            profile2: Second mechanism profile {mechanism: score}
            use_entropy: Whether to weight by entropy (diversity)

        Returns:
            Mechanism similarity score [0, 1]
        """
        if not profile1 or not profile2:
            return 0.0

        # Get all mechanisms
        all_mechanisms = set(profile1.keys()) | set(profile2.keys())

        if not all_mechanisms:
            return 0.0

        # Create vectors
        vec1 = np.array([profile1.get(m, 0.0) for m in all_mechanisms])
        vec2 = np.array([profile2.get(m, 0.0) for m in all_mechanisms])

        # Normalize if needed
        sum1, sum2 = vec1.sum(), vec2.sum()
        if sum1 > 0:
            vec1 = vec1 / sum1
        if sum2 > 0:
            vec2 = vec2 / sum2

        # Calculate base similarity
        similarity = SimilarityCalculator.cosine_similarity(vec1, vec2)

        # Apply entropy weighting if requested
        if use_entropy and sum1 > 0 and sum2 > 0:
            # Calculate entropy for diversity
            entropy1 = entropy(vec1 + 1e-10)
            entropy2 = entropy(vec2 + 1e-10)
            avg_entropy = (entropy1 + entropy2) / 2
            max_entropy = np.log(len(all_mechanisms))

            if max_entropy > 0:
                # Higher diversity = higher weight
                diversity_factor = avg_entropy / max_entropy
                similarity *= (0.5 + 0.5 * diversity_factor)

        return similarity

    @staticmethod
    def intervention_similarity(
        interventions1: Dict[str, float],
        interventions2: Dict[str, float],
        min_overlap: int = 3
    ) -> float:
        """
        Calculate similarity between intervention profiles.

        Args:
            interventions1: First intervention profile {intervention: effectiveness}
            interventions2: Second intervention profile {intervention: effectiveness}
            min_overlap: Minimum number of shared interventions required

        Returns:
            Intervention similarity score [0, 1]
        """
        shared = set(interventions1.keys()) & set(interventions2.keys())

        if len(shared) < min_overlap:
            return 0.0

        # Calculate correlation on shared interventions
        scores1 = np.array([interventions1[i] for i in shared])
        scores2 = np.array([interventions2[i] for i in shared])

        if len(scores1) == 0:
            return 0.0

        # Use cosine similarity on effectiveness scores
        return SimilarityCalculator.cosine_similarity(scores1, scores2)

    @staticmethod
    def weighted_similarity(
        similarities: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Combine multiple similarity scores with weights.

        Args:
            similarities: Dictionary of similarity scores
            weights: Optional weights for each similarity type

        Returns:
            Weighted similarity score [0, 1]
        """
        if not similarities:
            return 0.0

        if weights is None:
            # Equal weights by default
            weights = {k: 1.0 / len(similarities) for k in similarities}
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

        weighted_sum = sum(
            similarities.get(k, 0) * weights.get(k, 0)
            for k in similarities
        )

        return max(0, min(1, weighted_sum))  # Ensure [0, 1] range

    @staticmethod
    def evidence_weighted_similarity(
        shared_interventions: Set[str],
        effectiveness1: Dict[str, float],
        effectiveness2: Dict[str, float],
        evidence_counts: Optional[Dict[str, int]] = None
    ) -> float:
        """
        Calculate similarity weighted by evidence strength.

        Args:
            shared_interventions: Set of shared interventions
            effectiveness1: Effectiveness scores for condition 1
            effectiveness2: Effectiveness scores for condition 2
            evidence_counts: Optional evidence counts for weighting

        Returns:
            Evidence-weighted similarity score [0, 1]
        """
        if not shared_interventions:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for intervention in shared_interventions:
            eff1 = effectiveness1.get(intervention, 0)
            eff2 = effectiveness2.get(intervention, 0)

            # Calculate similarity for this intervention
            if eff1 > 0 and eff2 > 0:
                # Ratio-based similarity
                similarity = min(eff1, eff2) / max(eff1, eff2)
            else:
                similarity = 0.0

            # Apply evidence weighting if available
            weight = 1.0
            if evidence_counts:
                count = evidence_counts.get(intervention, 1)
                # Logarithmic weighting to prevent dominance
                weight = np.log1p(count)

            weighted_sum += similarity * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0


class ConditionSimilarityMetrics:
    """Specialized similarity metrics for medical conditions."""

    @staticmethod
    def treatment_response_similarity(
        condition1_treatments: Dict[str, float],
        condition2_treatments: Dict[str, float],
        min_shared: int = 5
    ) -> float:
        """
        Calculate similarity based on treatment response patterns.

        Args:
            condition1_treatments: Treatment effectiveness for condition 1
            condition2_treatments: Treatment effectiveness for condition 2
            min_shared: Minimum shared treatments required

        Returns:
            Treatment response similarity [0, 1]
        """
        calc = SimilarityCalculator()
        return calc.intervention_similarity(
            condition1_treatments,
            condition2_treatments,
            min_overlap=min_shared
        )

    @staticmethod
    def biological_mechanism_overlap(
        mechanisms1: Set[str],
        mechanisms2: Set[str],
        mechanism_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate overlap in biological mechanisms.

        Args:
            mechanisms1: Mechanisms for condition 1
            mechanisms2: Mechanisms for condition 2
            mechanism_weights: Optional importance weights

        Returns:
            Mechanism overlap score [0, 1]
        """
        if not mechanisms1 or not mechanisms2:
            return 0.0

        if mechanism_weights:
            # Weighted Jaccard
            shared = mechanisms1 & mechanisms2
            all_mechanisms = mechanisms1 | mechanisms2

            shared_weight = sum(mechanism_weights.get(m, 1.0) for m in shared)
            total_weight = sum(mechanism_weights.get(m, 1.0) for m in all_mechanisms)

            return shared_weight / total_weight if total_weight > 0 else 0.0
        else:
            # Standard Jaccard
            calc = SimilarityCalculator()
            return calc.jaccard_similarity(mechanisms1, mechanisms2)

    @staticmethod
    def condition_similarity_composite(
        treatment_similarity: float,
        mechanism_similarity: float,
        shared_evidence_count: int,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate composite similarity score for conditions.

        Args:
            treatment_similarity: Treatment response similarity
            mechanism_similarity: Biological mechanism similarity
            shared_evidence_count: Number of shared evidence points
            weights: Optional weights for components

        Returns:
            Composite similarity score [0, 1]
        """
        # Default weights
        if weights is None:
            weights = {
                'treatment': 0.5,
                'mechanism': 0.3,
                'evidence': 0.2
            }

        # Evidence score based on count (logarithmic)
        evidence_score = min(1.0, np.log1p(shared_evidence_count) / np.log(100))

        similarities = {
            'treatment': treatment_similarity,
            'mechanism': mechanism_similarity,
            'evidence': evidence_score
        }

        calc = SimilarityCalculator()
        return calc.weighted_similarity(similarities, weights)