"""
Innovation Tracking System.

Monitors emerging treatments and adjusts recommendations based on evolving evidence.
Ensures today's breakthrough doesn't get buried under yesterday's common treatment.

Key Features:
- Innovation lifecycle tracking (Discovery → Breakthrough → Rising Star → Established → Mature → Legacy)
- Revival detection for old treatments with new applications
- Temporal evidence weighting and growth tracking
- Breakthrough detection algorithms
- Integration with recommendation engine for innovation-aware suggestions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import math
from datetime import datetime, timedelta
import statistics
import warnings
warnings.filterwarnings('ignore')


@dataclass
class InnovationMetrics:
    """Innovation metrics for a treatment."""
    intervention: str
    innovation_index: float
    classification: str
    recency: float
    growth: float
    breakthrough: float
    diversity: float
    recommendation: str
    lifecycle_stage: str
    evidence_velocity: float
    application_novelty: float
    research_momentum: float


@dataclass
class TreatmentLifecycle:
    """Lifecycle information for a treatment."""
    intervention: str
    current_stage: str
    stage_duration: float
    previous_stages: List[str]
    first_evidence_year: int
    peak_research_year: int
    conditions_treated: List[str]
    revival_applications: List[Dict[str, Any]]
    stage_transitions: List[Dict[str, Any]]


@dataclass
class RevivalDetection:
    """Revival detection for old treatments with new applications."""
    intervention: str
    original_application: str
    new_application: str
    revival_year: int
    revival_strength: float
    evidence_quality: float
    novelty_score: float
    adoption_rate: float


class InnovationTrackingSystem:
    """
    Innovation tracking system that monitors emerging treatments and adjusts
    recommendations based on evolving evidence.

    Key Concepts:
    1. Innovation Lifecycle: Discovery → Breakthrough → Rising Star → Established → Mature → Legacy
    2. Revival Detection: Old treatments finding new applications
    3. Temporal Evidence Weighting: Recent evidence weighted more heavily
    4. Breakthrough Detection: Rapid emergence with strong results
    5. Growth Tracking: Year-over-year research and adoption growth
    """

    def __init__(self,
                 recency_half_life: float = 3.0,  # years
                 growth_window: int = 3,          # years
                 breakthrough_threshold: float = 0.3,
                 diversity_threshold: float = 0.6):
        """
        Initialize innovation tracking system.

        Args:
            recency_half_life: Half-life for temporal evidence weighting (years)
            growth_window: Window for calculating growth metrics (years)
            breakthrough_threshold: Threshold for breakthrough classification
            diversity_threshold: Threshold for diversity scoring
        """
        self.recency_half_life = recency_half_life
        self.growth_window = growth_window
        self.breakthrough_threshold = breakthrough_threshold
        self.diversity_threshold = diversity_threshold

        # Innovation lifecycle stages
        self.lifecycle_stages = {
            'discovery': {
                'description': 'Initial research, limited evidence',
                'evidence_threshold': 1,
                'innovation_range': (0.8, 1.0),
                'typical_duration': 2
            },
            'breakthrough': {
                'description': 'Rapidly emerging with strong results',
                'evidence_threshold': 3,
                'innovation_range': (0.7, 0.95),
                'typical_duration': 3
            },
            'rising_star': {
                'description': 'Growing evidence base',
                'evidence_threshold': 8,
                'innovation_range': (0.6, 0.85),
                'typical_duration': 5
            },
            'established': {
                'description': 'Well-supported with good evidence',
                'evidence_threshold': 20,
                'innovation_range': (0.4, 0.7),
                'typical_duration': 10
            },
            'mature': {
                'description': 'Well-understood, stable knowledge base',
                'evidence_threshold': 50,
                'innovation_range': (0.2, 0.5),
                'typical_duration': 15
            },
            'legacy': {
                'description': 'Historical treatment, limited new research',
                'evidence_threshold': 100,
                'innovation_range': (0.0, 0.3),
                'typical_duration': float('inf')
            }
        }

        # Known innovation examples for calibration
        self.innovation_examples = {
            'psilocybin_therapy': {
                'expected_innovation_index': 0.92,
                'expected_classification': 'BREAKTHROUGH - Rapidly emerging with strong results',
                'key_characteristics': ['very_recent', 'high_growth', 'breakthrough_results', 'multiple_conditions']
            },
            'cold_therapy': {
                'expected_innovation_index': 0.78,
                'expected_classification': 'RISING STAR - Growing evidence base',
                'key_characteristics': ['recent', 'moderate_growth', 'good_results', 'expanding_applications']
            },
            'vitamin_C': {
                'expected_innovation_index': 0.35,
                'expected_classification': 'MATURE - Well-understood, stable',
                'key_characteristics': ['historical', 'low_growth', 'stable_results', 'broad_applications']
            },
            'ketamine_depression': {
                'expected_innovation_index': 0.85,
                'expected_classification': 'REVIVAL - Old treatment, new breakthrough application',
                'key_characteristics': ['revival', 'breakthrough_application', 'rapid_adoption']
            }
        }

    def track_treatment_innovations(self,
                                  knowledge_graph,
                                  bayesian_scorer,
                                  treatment_list: Optional[List[str]] = None,
                                  current_year: int = 2024) -> Dict[str, InnovationMetrics]:
        """
        Track innovation metrics for treatments.

        Args:
            knowledge_graph: MedicalKnowledgeGraph instance
            bayesian_scorer: BayesianEvidenceScorer instance
            treatment_list: Optional list of treatments to analyze
            current_year: Current year for temporal calculations

        Returns:
            Dictionary of innovation metrics for each treatment
        """

        # Get all treatments if not specified
        if treatment_list is None:
            treatment_list = self._extract_all_treatments(knowledge_graph)

        innovation_metrics = {}

        for treatment in treatment_list:
            # Calculate innovation metrics
            metrics = self._calculate_innovation_metrics(
                treatment, knowledge_graph, bayesian_scorer, current_year
            )

            # Classify innovation level
            classification = self._classify_innovation_level(metrics)

            # Generate recommendation
            recommendation = self._generate_innovation_recommendation(metrics, classification)

            # Determine lifecycle stage
            lifecycle_stage = self._determine_lifecycle_stage(metrics)

            innovation_metrics[treatment] = InnovationMetrics(
                intervention=treatment,
                innovation_index=metrics['innovation_index'],
                classification=classification,
                recency=metrics['recency'],
                growth=metrics['growth'],
                breakthrough=metrics['breakthrough'],
                diversity=metrics['diversity'],
                recommendation=recommendation,
                lifecycle_stage=lifecycle_stage,
                evidence_velocity=metrics['evidence_velocity'],
                application_novelty=metrics['application_novelty'],
                research_momentum=metrics['research_momentum']
            )

        # Sort by innovation index
        sorted_metrics = dict(
            sorted(innovation_metrics.items(),
                   key=lambda x: x[1].innovation_index, reverse=True)
        )

        return sorted_metrics

    def _extract_all_treatments(self, knowledge_graph) -> List[str]:
        """Extract all treatments from knowledge graph."""
        treatments = set()

        for condition, treatment_edges in knowledge_graph.backward_edges.items():
            treatments.update(treatment_edges.keys())

        return list(treatments)

    def _calculate_innovation_metrics(self,
                                    treatment: str,
                                    knowledge_graph,
                                    bayesian_scorer,
                                    current_year: int) -> Dict[str, float]:
        """Calculate core innovation metrics for a treatment."""

        # Get all evidence for treatment
        evidence_data = self._gather_evidence_data(treatment, knowledge_graph)

        # 1. Recency Score
        recency = self._calculate_recency_score(evidence_data, current_year)

        # 2. Growth Score
        growth = self._calculate_growth_score(evidence_data, current_year)

        # 3. Breakthrough Score
        breakthrough = self._calculate_breakthrough_score(evidence_data, bayesian_scorer)

        # 4. Diversity Score
        diversity = self._calculate_diversity_score(evidence_data)

        # 5. Evidence Velocity (rate of new evidence)
        evidence_velocity = self._calculate_evidence_velocity(evidence_data, current_year)

        # 6. Application Novelty (new condition applications)
        application_novelty = self._calculate_application_novelty(evidence_data, current_year)

        # 7. Research Momentum (accelerating research)
        research_momentum = self._calculate_research_momentum(evidence_data, current_year)

        # Calculate overall innovation index
        innovation_index = self._calculate_innovation_index(
            recency, growth, breakthrough, diversity, evidence_velocity, application_novelty
        )

        return {
            'innovation_index': innovation_index,
            'recency': recency,
            'growth': growth,
            'breakthrough': breakthrough,
            'diversity': diversity,
            'evidence_velocity': evidence_velocity,
            'application_novelty': application_novelty,
            'research_momentum': research_momentum
        }

    def _gather_evidence_data(self, treatment: str, knowledge_graph) -> List[Dict[str, Any]]:
        """Gather all evidence data for a treatment across conditions."""
        evidence_data = []

        for condition, treatment_edges in knowledge_graph.backward_edges.items():
            if treatment in treatment_edges:
                for edge in treatment_edges[treatment]:
                    evidence_data.append({
                        'condition': condition,
                        'evidence_type': edge.evidence.evidence_type,
                        'confidence': edge.evidence.confidence,
                        'sample_size': edge.evidence.sample_size,
                        'study_design': edge.evidence.study_design,
                        'study_id': edge.evidence.study_id,
                        'year': self._extract_year_from_study_id(edge.evidence.study_id)
                    })

        return evidence_data

    def _extract_year_from_study_id(self, study_id: str) -> int:
        """Extract publication year from study ID (mock implementation)."""
        # In real implementation, this would parse actual study metadata
        # For demo, create realistic year distribution
        import random
        if 'psilocybin' in study_id.lower():
            return random.randint(2018, 2024)  # Very recent
        elif 'cold_therapy' in study_id.lower():
            return random.randint(2015, 2024)  # Recent with some history
        elif 'vitamin' in study_id.lower():
            return random.randint(1980, 2024)  # Long history
        elif 'ketamine' in study_id.lower() and 'depression' in study_id.lower():
            return random.randint(2010, 2024)  # Revival period
        else:
            return random.randint(2000, 2024)  # General distribution

    def _calculate_recency_score(self, evidence_data: List[Dict], current_year: int) -> float:
        """Calculate recency score using exponential decay."""
        if not evidence_data:
            return 0.0

        recency_scores = []
        for evidence in evidence_data:
            years_ago = current_year - evidence['year']
            # Handle negative years_ago (future dates) and extreme values
            years_ago = max(0, years_ago)  # Clamp to non-negative
            years_ago = min(100, years_ago)  # Clamp to reasonable maximum

            # Exponential decay with half-life
            decay_factor = 0.5 ** (years_ago / self.recency_half_life)
            # Ensure decay factor is within valid range
            decay_factor = max(0.0, min(1.0, decay_factor))
            recency_scores.append(decay_factor)

        return np.mean(recency_scores)

    def _calculate_growth_score(self, evidence_data: List[Dict], current_year: int) -> float:
        """Calculate growth score based on evidence publication trends."""
        if len(evidence_data) < 2:
            return 0.0

        # Count evidence by year
        year_counts = Counter([evidence['year'] for evidence in evidence_data])

        # Get counts for recent years
        recent_years = list(range(current_year - self.growth_window, current_year + 1))
        recent_counts = [year_counts.get(year, 0) for year in recent_years]

        if sum(recent_counts) == 0:
            return 0.0

        # Calculate year-over-year growth
        growth_rates = []
        for i in range(1, len(recent_counts)):
            if recent_counts[i-1] > 0:
                growth_rate = (recent_counts[i] - recent_counts[i-1]) / recent_counts[i-1]
                growth_rates.append(growth_rate)

        if not growth_rates:
            return 0.0

        # Convert to 0-1 scale, with 0.5 = 50% growth, 1.0 = 100%+ growth
        avg_growth = np.mean(growth_rates)
        normalized_growth = min(1.0, max(0.0, (avg_growth + 0.5) / 1.5))

        return normalized_growth

    def _calculate_breakthrough_score(self, evidence_data: List[Dict], bayesian_scorer) -> float:
        """Calculate breakthrough score based on recent exceptional results."""
        if not evidence_data:
            return 0.0

        # Get recent evidence (last 3 years)
        current_year = 2024
        recent_evidence = [e for e in evidence_data if current_year - e['year'] <= 3]

        if not recent_evidence:
            return 0.0

        # Get historical baseline performance
        historical_evidence = [e for e in evidence_data if current_year - e['year'] > 3]

        # Calculate recent vs historical performance improvement
        recent_performance = np.mean([e['confidence'] for e in recent_evidence])

        if historical_evidence:
            historical_performance = np.mean([e['confidence'] for e in historical_evidence])
            improvement = recent_performance - historical_performance
        else:
            # If no historical data, high recent performance suggests breakthrough
            improvement = recent_performance - 0.5

        # Normalize to 0-1 scale
        breakthrough_score = min(1.0, max(0.0, improvement / 0.4))  # 0.4 = 40% improvement = max score

        return breakthrough_score

    def _calculate_diversity_score(self, evidence_data: List[Dict]) -> float:
        """Calculate diversity score based on breadth of applications."""
        if not evidence_data:
            return 0.0

        # Count unique conditions treated
        unique_conditions = set([evidence['condition'] for evidence in evidence_data])
        condition_count = len(unique_conditions)

        # Count evidence types
        unique_evidence_types = set([evidence['evidence_type'] for evidence in evidence_data])
        evidence_type_count = len(unique_evidence_types)

        # Count study designs
        unique_study_designs = set([evidence['study_design'] for evidence in evidence_data])
        study_design_count = len(unique_study_designs)

        # Normalize each component
        condition_diversity = min(1.0, condition_count / 10.0)  # 10+ conditions = max diversity
        evidence_diversity = min(1.0, evidence_type_count / 3.0)  # 3+ evidence types = max
        design_diversity = min(1.0, study_design_count / 4.0)  # 4+ designs = max

        # Weighted combination
        diversity_score = (
            condition_diversity * 0.5 +
            evidence_diversity * 0.3 +
            design_diversity * 0.2
        )

        return diversity_score

    def _calculate_evidence_velocity(self, evidence_data: List[Dict], current_year: int) -> float:
        """Calculate rate of new evidence generation."""
        if len(evidence_data) < 2:
            return 0.0

        # Count evidence per year
        year_counts = Counter([evidence['year'] for evidence in evidence_data])

        # Calculate evidence per year for recent period
        recent_years = range(current_year - 5, current_year + 1)
        recent_evidence_per_year = [year_counts.get(year, 0) for year in recent_years]

        # Calculate velocity (average evidence per year)
        velocity = np.mean(recent_evidence_per_year)

        # Normalize to 0-1 scale (10+ studies per year = max velocity)
        normalized_velocity = min(1.0, velocity / 10.0)

        return normalized_velocity

    def _calculate_application_novelty(self, evidence_data: List[Dict], current_year: int) -> float:
        """Calculate novelty of new condition applications."""
        if not evidence_data:
            return 0.0

        # Get recent new condition applications (last 2 years)
        recent_conditions = set([
            e['condition'] for e in evidence_data
            if current_year - e['year'] <= 2
        ])

        # Get historical conditions
        historical_conditions = set([
            e['condition'] for e in evidence_data
            if current_year - e['year'] > 2
        ])

        # Calculate new applications
        new_applications = recent_conditions - historical_conditions
        novelty_count = len(new_applications)

        # Normalize (3+ new applications = max novelty)
        novelty_score = min(1.0, novelty_count / 3.0)

        return novelty_score

    def _calculate_research_momentum(self, evidence_data: List[Dict], current_year: int) -> float:
        """Calculate research momentum (accelerating research activity)."""
        if len(evidence_data) < 3:
            return 0.0

        # Get evidence counts for last 6 years
        years = range(current_year - 5, current_year + 1)
        counts = [sum(1 for e in evidence_data if e['year'] == year) for year in years]

        # Calculate acceleration (second derivative)
        if len(counts) >= 3:
            # Simple acceleration: difference between recent and early slopes
            early_slope = (counts[2] - counts[0]) / 2 if counts[0] > 0 else 0
            late_slope = (counts[-1] - counts[-3]) / 2 if counts[-3] > 0 else 0
            acceleration = late_slope - early_slope

            # Normalize to 0-1 scale
            momentum = min(1.0, max(0.0, acceleration / 5.0))  # 5 studies/year acceleration = max
            return momentum

        return 0.0

    def _calculate_innovation_index(self,
                                  recency: float,
                                  growth: float,
                                  breakthrough: float,
                                  diversity: float,
                                  evidence_velocity: float,
                                  application_novelty: float) -> float:
        """Calculate overall innovation index."""

        # Weighted combination of metrics
        innovation_index = (
            recency * 0.25 +           # How recent is the research?
            growth * 0.20 +            # Is research activity growing?
            breakthrough * 0.25 +      # Are recent results exceptional?
            diversity * 0.15 +         # How broad are the applications?
            evidence_velocity * 0.10 + # How fast is evidence accumulating?
            application_novelty * 0.05  # Are there new applications?
        )

        return min(1.0, innovation_index)

    def _classify_innovation_level(self, metrics: Dict[str, float]) -> str:
        """Classify innovation level based on metrics."""
        innovation_index = metrics['innovation_index']
        breakthrough = metrics['breakthrough']
        growth = metrics['growth']
        recency = metrics['recency']

        # Breakthrough classification
        if innovation_index >= 0.85 and breakthrough >= 0.4:
            return "BREAKTHROUGH - Rapidly emerging with strong results"
        elif innovation_index >= 0.75 and growth >= 0.6:
            return "RISING STAR - Growing evidence base"
        elif innovation_index >= 0.6:
            return "EMERGING - Early evidence shows promise"
        elif innovation_index >= 0.4:
            return "ESTABLISHED - Well-supported treatment option"
        elif innovation_index >= 0.25:
            return "MATURE - Well-understood, stable"
        else:
            return "LEGACY - Historical treatment, limited new research"

    def _generate_innovation_recommendation(self, metrics: Dict[str, float], classification: str) -> str:
        """Generate recommendation based on innovation metrics."""
        innovation_index = metrics['innovation_index']
        breakthrough = metrics['breakthrough']
        growth = metrics['growth']
        recency = metrics['recency']

        if "BREAKTHROUGH" in classification:
            return "Consider for treatment-resistant cases. Monitor closely as protocols are still evolving."
        elif "RISING STAR" in classification:
            return "Good option with growing support. Check recent studies for optimal protocols."
        elif "EMERGING" in classification:
            return "Promising early evidence. Consider for patients who haven't responded to established treatments."
        elif "ESTABLISHED" in classification:
            return "Solid treatment option with good evidence base. Standard consideration for appropriate patients."
        elif "MATURE" in classification:
            return "Traditional treatment with extensive history. Consider if newer options have failed."
        else:
            return "Historical treatment with limited recent research. Consider only if specific expertise available."

    def _determine_lifecycle_stage(self, metrics: Dict[str, float]) -> str:
        """Determine lifecycle stage based on metrics."""
        innovation_index = metrics['innovation_index']

        for stage, properties in self.lifecycle_stages.items():
            min_innovation, max_innovation = properties['innovation_range']
            if min_innovation <= innovation_index <= max_innovation:
                return stage

        return 'unknown'

    def detect_treatment_revivals(self,
                                knowledge_graph,
                                innovation_metrics: Dict[str, InnovationMetrics],
                                current_year: int = 2024) -> Dict[str, RevivalDetection]:
        """Detect treatments experiencing revival (old treatments, new applications)."""
        revivals = {}

        for treatment, metrics in innovation_metrics.items():
            # Check for revival patterns
            revival_data = self._analyze_potential_revival(
                treatment, knowledge_graph, metrics, current_year
            )

            if revival_data and revival_data['revival_strength'] > 0.5:
                revivals[treatment] = RevivalDetection(
                    intervention=treatment,
                    original_application=revival_data['original_application'],
                    new_application=revival_data['new_application'],
                    revival_year=revival_data['revival_year'],
                    revival_strength=revival_data['revival_strength'],
                    evidence_quality=revival_data['evidence_quality'],
                    novelty_score=revival_data['novelty_score'],
                    adoption_rate=revival_data['adoption_rate']
                )

        return revivals

    def _analyze_potential_revival(self,
                                 treatment: str,
                                 knowledge_graph,
                                 metrics: InnovationMetrics,
                                 current_year: int) -> Optional[Dict[str, Any]]:
        """Analyze if treatment shows revival patterns."""

        # Get evidence data
        evidence_data = self._gather_evidence_data(treatment, knowledge_graph)

        if len(evidence_data) < 5:  # Need sufficient history
            return None

        # Check for bimodal distribution (early evidence, gap, then recent surge)
        years = [e['year'] for e in evidence_data]
        min_year, max_year = min(years), max(years)

        # Need at least 10 year span for revival detection
        if max_year - min_year < 10:
            return None

        # Look for gap in middle and recent surge
        year_counts = Counter(years)
        recent_count = sum(year_counts.get(year, 0) for year in range(current_year - 5, current_year + 1))
        historical_count = sum(year_counts.get(year, 0) for year in range(min_year, current_year - 10))
        middle_count = sum(year_counts.get(year, 0) for year in range(current_year - 10, current_year - 5))

        # Revival pattern: historical evidence, low middle activity, recent surge
        if historical_count >= 2 and recent_count >= 3 and recent_count > middle_count * 2:
            # Identify original vs new applications
            historical_conditions = set([
                e['condition'] for e in evidence_data if e['year'] <= current_year - 10
            ])
            recent_conditions = set([
                e['condition'] for e in evidence_data if e['year'] >= current_year - 5
            ])

            new_applications = recent_conditions - historical_conditions

            if new_applications:
                # Calculate revival metrics
                revival_strength = min(1.0, recent_count / (historical_count + 1))
                evidence_quality = np.mean([
                    e['confidence'] for e in evidence_data if e['year'] >= current_year - 5
                ])
                novelty_score = len(new_applications) / len(recent_conditions)
                adoption_rate = recent_count / 5.0  # Studies per year

                return {
                    'original_application': list(historical_conditions)[0] if historical_conditions else 'unknown',
                    'new_application': list(new_applications)[0],
                    'revival_year': current_year - 5,  # Approximate revival start
                    'revival_strength': revival_strength,
                    'evidence_quality': evidence_quality,
                    'novelty_score': novelty_score,
                    'adoption_rate': min(1.0, adoption_rate)
                }

        return None

    def integrate_with_recommendations(self,
                                     treatment_recommendations: List,
                                     innovation_metrics: Dict[str, InnovationMetrics],
                                     innovation_boost_factor: float = 0.2) -> List:
        """Integrate innovation metrics with treatment recommendations."""

        enhanced_recommendations = []

        for recommendation in treatment_recommendations:
            treatment_name = recommendation.intervention
            enhanced_rec = recommendation

            # Add innovation boost if treatment is innovative
            if treatment_name in innovation_metrics:
                innovation_data = innovation_metrics[treatment_name]

                # Apply innovation boost
                innovation_boost = innovation_data.innovation_index * innovation_boost_factor
                enhanced_rec.final_score = min(1.0, enhanced_rec.final_score + innovation_boost)

                # Update explanation to include innovation info
                if innovation_data.innovation_index > 0.7:
                    enhanced_rec.explanation += f" | {innovation_data.classification}"

                # Add innovation metadata
                enhanced_rec.innovation_index = innovation_data.innovation_index
                enhanced_rec.innovation_classification = innovation_data.classification

            enhanced_recommendations.append(enhanced_rec)

        # Re-sort by enhanced final score
        enhanced_recommendations.sort(key=lambda x: x.final_score, reverse=True)

        return enhanced_recommendations

    def analyze_innovation_trends(self, innovation_metrics: Dict[str, InnovationMetrics]) -> Dict[str, Any]:
        """Analyze trends in innovation across treatments."""

        if not innovation_metrics:
            return {'error': 'No innovation metrics to analyze'}

        # Classification distribution
        classification_counts = Counter([m.classification.split(' - ')[0] for m in innovation_metrics.values()])

        # Lifecycle stage distribution
        lifecycle_counts = Counter([m.lifecycle_stage for m in innovation_metrics.values()])

        # Innovation index statistics
        innovation_indices = [m.innovation_index for m in innovation_metrics.values()]

        # High innovation treatments
        high_innovation = {
            name: metrics for name, metrics in innovation_metrics.items()
            if metrics.innovation_index > 0.7
        }

        # Growth leaders
        growth_leaders = sorted(
            innovation_metrics.items(),
            key=lambda x: x[1].growth,
            reverse=True
        )[:5]

        # Breakthrough treatments
        breakthrough_treatments = {
            name: metrics for name, metrics in innovation_metrics.items()
            if metrics.breakthrough > 0.3
        }

        return {
            'total_treatments_analyzed': len(innovation_metrics),
            'avg_innovation_index': np.mean(innovation_indices),
            'std_innovation_index': np.std(innovation_indices),
            'classification_distribution': dict(classification_counts),
            'lifecycle_distribution': dict(lifecycle_counts),
            'high_innovation_count': len(high_innovation),
            'high_innovation_treatments': list(high_innovation.keys()),
            'growth_leaders': [(name, metrics.growth) for name, metrics in growth_leaders],
            'breakthrough_count': len(breakthrough_treatments),
            'breakthrough_treatments': list(breakthrough_treatments.keys()),
            'innovation_range': (min(innovation_indices), max(innovation_indices))
        }

    def get_innovation_recommendations(self, innovation_metrics: Dict[str, InnovationMetrics]) -> Dict[str, List[str]]:
        """Get recommendations for different innovation categories."""

        recommendations = {
            'monitor_closely': [],
            'consider_early_adoption': [],
            'standard_consideration': [],
            'research_needed': [],
            'historical_interest': []
        }

        for treatment, metrics in innovation_metrics.items():
            if metrics.innovation_index >= 0.85:
                recommendations['monitor_closely'].append(treatment)
            elif metrics.innovation_index >= 0.7:
                recommendations['consider_early_adoption'].append(treatment)
            elif metrics.innovation_index >= 0.4:
                recommendations['standard_consideration'].append(treatment)
            elif metrics.innovation_index >= 0.2:
                recommendations['research_needed'].append(treatment)
            else:
                recommendations['historical_interest'].append(treatment)

        return recommendations


def create_demo_innovation_data():
    """Create demonstration data for innovation tracking."""

    # Mock Knowledge Graph with temporal evidence
    class MockKnowledgeGraph:
        def __init__(self):
            self.backward_edges = {
                # Psilocybin therapy (breakthrough)
                'depression': {
                    'psilocybin_therapy': [self._create_mock_edge('positive', 0.85, '2023_psilocybin_depression_study')],
                    'SSRIs': [self._create_mock_edge('positive', 0.70, '2010_SSRI_depression_study')]
                },
                'PTSD': {
                    'psilocybin_therapy': [self._create_mock_edge('positive', 0.80, '2022_psilocybin_PTSD_study')]
                },
                'anxiety': {
                    'psilocybin_therapy': [self._create_mock_edge('positive', 0.75, '2023_psilocybin_anxiety_study')]
                },

                # Cold therapy (rising star)
                'inflammation': {
                    'cold_therapy': [self._create_mock_edge('positive', 0.70, '2020_cold_therapy_inflammation_study')],
                    'NSAIDs': [self._create_mock_edge('positive', 0.65, '2005_NSAID_inflammation_study')]
                },
                'recovery': {
                    'cold_therapy': [self._create_mock_edge('positive', 0.75, '2021_cold_therapy_recovery_study')]
                },
                'chronic_pain': {
                    'cold_therapy': [self._create_mock_edge('positive', 0.60, '2019_cold_therapy_pain_study')]
                },

                # Vitamin C (mature)
                'scurvy': {
                    'vitamin_C': [self._create_mock_edge('positive', 0.95, '1980_vitamin_C_scurvy_study')]
                },
                'common_cold': {
                    'vitamin_C': [self._create_mock_edge('neutral', 0.40, '1990_vitamin_C_cold_study')]
                },
                'immune_support': {
                    'vitamin_C': [self._create_mock_edge('positive', 0.60, '2000_vitamin_C_immune_study')]
                },

                # Ketamine depression (revival)
                'treatment_resistant_depression': {
                    'ketamine': [self._create_mock_edge('positive', 0.85, '2020_ketamine_depression_study')]
                },
                'anesthesia': {
                    'ketamine': [self._create_mock_edge('positive', 0.90, '1975_ketamine_anesthesia_study')]
                }
            }

        def _create_mock_edge(self, evidence_type, confidence, study_id):
            class MockEvidence:
                def __init__(self, evidence_type, confidence, study_id):
                    self.evidence_type = evidence_type
                    self.confidence = confidence
                    self.sample_size = 100
                    self.study_design = 'RCT'
                    self.study_id = study_id

            class MockEdge:
                def __init__(self, evidence):
                    self.evidence = evidence

            return MockEdge(MockEvidence(evidence_type, confidence, study_id))

    # Mock Bayesian Scorer
    class MockBayesianScorer:
        def score_intervention(self, intervention, condition):
            scores = {
                ('psilocybin_therapy', 'depression'): {'score': 0.85, 'confidence': 0.80},
                ('cold_therapy', 'inflammation'): {'score': 0.70, 'confidence': 0.65},
                ('vitamin_C', 'scurvy'): {'score': 0.95, 'confidence': 0.95},
                ('ketamine', 'treatment_resistant_depression'): {'score': 0.85, 'confidence': 0.85}
            }
            return scores.get((intervention, condition), {'score': 0.5, 'confidence': 0.5})

    return MockKnowledgeGraph(), MockBayesianScorer()