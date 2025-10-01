"""
Unified scoring utilities for data mining modules.
Centralizes effectiveness scoring, confidence calculations, and statistical helpers.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from scipy import stats
from dataclasses import dataclass


@dataclass
class ScoringResult:
    """Container for scoring results with metadata."""
    score: float
    confidence: float
    evidence_count: int
    method: str
    metadata: Dict = None


class EffectivenessScorer:
    """Unified effectiveness scoring utilities."""

    @staticmethod
    def calculate_weighted_score(
        scores: List[float],
        weights: Optional[List[float]] = None,
        normalize: bool = True
    ) -> float:
        """
        Calculate weighted average of scores.

        Args:
            scores: List of scores
            weights: Optional weights for each score
            normalize: Whether to normalize result to [0, 1]

        Returns:
            Weighted score
        """
        if not scores:
            return 0.0

        scores_array = np.array(scores)

        if weights is None:
            weights = np.ones(len(scores))
        else:
            weights = np.array(weights)

        # Handle zero weights
        if weights.sum() == 0:
            return 0.0

        weighted_score = np.average(scores_array, weights=weights)

        if normalize:
            weighted_score = max(0, min(1, weighted_score))

        return float(weighted_score)

    @staticmethod
    def aggregate_evidence(
        positive_count: int,
        negative_count: int,
        neutral_count: int = 0,
        method: str = 'ratio'
    ) -> ScoringResult:
        """
        Aggregate evidence counts into effectiveness score.

        Args:
            positive_count: Number of positive outcomes
            negative_count: Number of negative outcomes
            neutral_count: Number of neutral outcomes
            method: Aggregation method ('ratio', 'wilson', 'laplace')

        Returns:
            ScoringResult with score and confidence
        """
        total = positive_count + negative_count + neutral_count

        if total == 0:
            return ScoringResult(score=0.5, confidence=0.0, evidence_count=0, method=method)

        if method == 'ratio':
            # Simple ratio
            score = positive_count / total if total > 0 else 0.5
            # Confidence based on sample size
            confidence = min(1.0, np.log1p(total) / np.log(100))

        elif method == 'wilson':
            # Wilson score interval (better for small samples)
            p = positive_count / total if total > 0 else 0.5
            z = 1.96  # 95% confidence
            denominator = 1 + z**2 / total
            score = (p + z**2 / (2 * total)) / denominator
            # Wilson confidence
            confidence = 1 - np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))

        elif method == 'laplace':
            # Laplace smoothing
            score = (positive_count + 1) / (total + 2)
            # Confidence increases with evidence
            confidence = 1 - (2 / (total + 2))

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return ScoringResult(
            score=float(score),
            confidence=float(confidence),
            evidence_count=total,
            method=method
        )

    @staticmethod
    def combine_scores(
        scores: List[ScoringResult],
        weights: Optional[Dict[str, float]] = None
    ) -> ScoringResult:
        """
        Combine multiple scoring results.

        Args:
            scores: List of ScoringResult objects
            weights: Optional weights by method type

        Returns:
            Combined ScoringResult
        """
        if not scores:
            return ScoringResult(score=0.0, confidence=0.0, evidence_count=0, method='combined')

        # Default weights if not provided
        if weights is None:
            weights = {
                'bayesian': 1.0,
                'direct': 0.9,
                'mechanism': 0.7,
                'similar': 0.6,
                'combined': 0.8
            }

        total_score = 0.0
        total_weight = 0.0
        total_evidence = 0
        max_confidence = 0.0

        for result in scores:
            weight = weights.get(result.method, 0.5)
            # Weight by confidence too
            effective_weight = weight * result.confidence

            total_score += result.score * effective_weight
            total_weight += effective_weight
            total_evidence += result.evidence_count
            max_confidence = max(max_confidence, result.confidence)

        if total_weight == 0:
            return ScoringResult(score=0.0, confidence=0.0, evidence_count=0, method='combined')

        return ScoringResult(
            score=total_score / total_weight,
            confidence=max_confidence,  # Use max confidence
            evidence_count=total_evidence,
            method='combined'
        )


class ConfidenceCalculator:
    """Utilities for calculating confidence scores."""

    @staticmethod
    def sample_size_confidence(
        sample_size: int,
        target_size: int = 100,
        method: str = 'logarithmic'
    ) -> float:
        """
        Calculate confidence based on sample size.

        Args:
            sample_size: Current sample size
            target_size: Target sample size for full confidence
            method: Calculation method ('logarithmic', 'linear', 'sqrt')

        Returns:
            Confidence score [0, 1]
        """
        if sample_size <= 0:
            return 0.0

        if method == 'logarithmic':
            # Logarithmic growth
            confidence = np.log1p(sample_size) / np.log1p(target_size)
        elif method == 'linear':
            # Linear growth
            confidence = sample_size / target_size
        elif method == 'sqrt':
            # Square root growth (faster initially)
            confidence = np.sqrt(sample_size / target_size)
        else:
            raise ValueError(f"Unknown method: {method}")

        return float(min(1.0, confidence))

    @staticmethod
    def variance_confidence(
        values: List[float],
        max_variance: float = 0.25
    ) -> float:
        """
        Calculate confidence based on variance (lower variance = higher confidence).

        Args:
            values: List of values
            max_variance: Maximum expected variance

        Returns:
            Confidence score [0, 1]
        """
        if len(values) < 2:
            return 0.0

        variance = np.var(values)
        # Inverse relationship: lower variance = higher confidence
        confidence = max(0, 1 - (variance / max_variance))

        return float(confidence)

    @staticmethod
    def consensus_confidence(
        agreement_count: int,
        total_count: int,
        threshold: float = 0.7
    ) -> float:
        """
        Calculate confidence based on consensus/agreement.

        Args:
            agreement_count: Number in agreement
            total_count: Total number of observations
            threshold: Agreement threshold for high confidence

        Returns:
            Confidence score [0, 1]
        """
        if total_count == 0:
            return 0.0

        agreement_ratio = agreement_count / total_count

        if agreement_ratio >= threshold:
            # High confidence when above threshold
            confidence = 0.7 + 0.3 * ((agreement_ratio - threshold) / (1 - threshold))
        else:
            # Lower confidence below threshold
            confidence = 0.7 * (agreement_ratio / threshold)

        return float(confidence)


class StatisticalHelpers:
    """Common statistical utilities."""

    @staticmethod
    def normalize_scores(
        scores: Dict[str, float],
        method: str = 'minmax'
    ) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: Dictionary of scores
            method: Normalization method ('minmax', 'zscore', 'sigmoid')

        Returns:
            Normalized scores
        """
        if not scores:
            return {}

        values = np.array(list(scores.values()))

        if len(values) == 0:
            return scores

        if method == 'minmax':
            # Min-max normalization
            min_val, max_val = values.min(), values.max()
            if max_val - min_val == 0:
                normalized = {k: 0.5 for k in scores}
            else:
                normalized = {
                    k: (v - min_val) / (max_val - min_val)
                    for k, v in scores.items()
                }

        elif method == 'zscore':
            # Z-score normalization then sigmoid
            mean, std = values.mean(), values.std()
            if std == 0:
                normalized = {k: 0.5 for k in scores}
            else:
                z_scores = {k: (v - mean) / std for k, v in scores.items()}
                # Apply sigmoid to map to [0, 1]
                normalized = {
                    k: 1 / (1 + np.exp(-z))
                    for k, z in z_scores.items()
                }

        elif method == 'sigmoid':
            # Direct sigmoid transformation
            normalized = {
                k: 1 / (1 + np.exp(-v))
                for k, v in scores.items()
            }

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    @staticmethod
    def calculate_percentile_rank(
        value: float,
        distribution: List[float]
    ) -> float:
        """
        Calculate percentile rank of value in distribution.

        Args:
            value: Value to rank
            distribution: List of values forming distribution

        Returns:
            Percentile rank [0, 100]
        """
        if not distribution:
            return 50.0

        return float(stats.percentileofscore(distribution, value))

    @staticmethod
    def exponential_decay(
        value: float,
        time_delta: float,
        half_life: float
    ) -> float:
        """
        Apply exponential decay to a value.

        Args:
            value: Initial value
            time_delta: Time elapsed
            half_life: Half-life period

        Returns:
            Decayed value
        """
        if half_life <= 0:
            return value

        decay_constant = np.log(2) / half_life
        return float(value * np.exp(-decay_constant * time_delta))

    @staticmethod
    def calculate_lift(
        observed_rate: float,
        expected_rate: float
    ) -> float:
        """
        Calculate lift (observed / expected).

        Args:
            observed_rate: Observed rate
            expected_rate: Expected rate

        Returns:
            Lift value
        """
        if expected_rate == 0:
            return 0.0 if observed_rate == 0 else float('inf')

        return observed_rate / expected_rate

    @staticmethod
    def binomial_confidence_interval(
        successes: int,
        trials: int,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate binomial confidence interval.

        Args:
            successes: Number of successes
            trials: Total number of trials
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if trials == 0:
            return (0.0, 1.0)

        # Using Clopper-Pearson method
        alpha = 1 - confidence_level

        if successes == 0:
            lower = 0.0
        else:
            lower = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)

        if successes == trials:
            upper = 1.0
        else:
            upper = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)

        return (float(lower), float(upper))


class ThresholdClassifier:
    """Utilities for threshold-based classification."""

    @staticmethod
    def classify_evidence_level(
        evidence_count: int,
        confidence: float,
        thresholds: Optional[Dict] = None
    ) -> str:
        """
        Classify evidence level based on count and confidence.

        Args:
            evidence_count: Number of evidence points
            confidence: Confidence score
            thresholds: Optional custom thresholds

        Returns:
            Classification string
        """
        if thresholds is None:
            thresholds = {
                'strong': {'min_evidence': 10, 'min_confidence': 0.75},
                'moderate': {'min_evidence': 5, 'min_confidence': 0.60},
                'weak': {'min_evidence': 2, 'min_confidence': 0.40},
                'insufficient': {'min_evidence': 0, 'min_confidence': 0.0}
            }

        # Check from strongest to weakest
        for level in ['strong', 'moderate', 'weak', 'insufficient']:
            if level in thresholds:
                threshold = thresholds[level]
                if (evidence_count >= threshold['min_evidence'] and
                    confidence >= threshold['min_confidence']):
                    return level

        return 'insufficient'

    @staticmethod
    def classify_innovation_stage(
        age_days: int,
        growth_rate: float,
        evidence_count: int
    ) -> str:
        """
        Classify innovation stage based on metrics.

        Args:
            age_days: Days since first evidence
            growth_rate: Growth rate of evidence
            evidence_count: Total evidence count

        Returns:
            Innovation stage
        """
        if age_days < 365 and growth_rate > 0.5:
            return 'breakthrough'
        elif age_days < 730 and growth_rate > 0.3:
            return 'rising_star'
        elif evidence_count > 50:
            return 'established'
        elif age_days > 1825:  # 5 years
            return 'mature'
        else:
            return 'emerging'