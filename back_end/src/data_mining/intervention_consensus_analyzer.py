#!/usr/bin/env python3
"""
Intervention Consensus Analyzer - LEGACY DATA MINING TOOL

⚠️  WARNING: This is a legacy data mining tool from back_end/src/data_mining/.
    Kept for backward compatibility with standalone analysis scripts.

    DEPRECATED FIELD REFERENCES:
    - correlation_strength field was removed Oct 16, 2025
    - References commented out but kept for reference

This module provides data mining and research analysis functionality for creating
consensus interventions from multiple sources. It groups similar interventions
across papers and creates high-quality consensus records for research purposes.

This is separate from core entity processing and focuses on research data quality
enhancement through evidence accumulation and cross-paper consensus building.

Usage:
    from back_end.src.data_mining.intervention_consensus_analyzer import InterventionConsensusAnalyzer

    analyzer = InterventionConsensusAnalyzer(batch_processor)
    consensus_results = analyzer.create_research_consensus(interventions, confidence_threshold=0.5)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime


class InterventionConsensusAnalyzer:
    """
    Analyzes and creates consensus from multiple intervention extractions for research purposes.

    This class focuses on data mining and research quality enhancement by:
    - Grouping similar interventions across different papers
    - Creating weighted consensus from multiple evidence sources
    - Enhancing confidence through cross-validation
    - Accumulating evidence for stronger research conclusions

    This is distinct from basic duplicate removal and focuses on research data quality.
    """

    def __init__(self, batch_processor):
        """
        Initialize the consensus analyzer.

        Args:
            batch_processor: Instance of BatchEntityProcessor for entity operations
        """
        self.batch_processor = batch_processor
        self.logger = batch_processor.logger

    def create_research_consensus(self, interventions: List[Dict],
                                confidence_threshold: float = 0.5,
                                min_sources: int = 1) -> Dict[str, Any]:
        """
        Create research-quality consensus interventions from multiple sources.

        This method is designed for data mining and research analysis, focusing on
        creating high-quality consensus records by combining evidence across papers.

        Args:
            interventions: List of interventions with canonical entities resolved
            confidence_threshold: Minimum confidence for final consensus
            min_sources: Minimum number of sources required for consensus

        Returns:
            Dictionary containing consensus results and analysis metadata
        """
        if not interventions:
            return {'consensus_interventions': [], 'analysis_metadata': {}}

        self.logger.info(f"Creating research consensus from {len(interventions)} interventions")

        # Group interventions by canonical entities for cross-paper analysis
        grouped_interventions = self._group_for_research_analysis(interventions)

        # Create consensus for each research group
        consensus_interventions = []
        analysis_stats = {
            'total_groups': len(grouped_interventions),
            'single_source_groups': 0,
            'multi_source_groups': 0,
            'evidence_accumulation_cases': 0,
            'confidence_enhanced_cases': 0
        }

        for group_key, intervention_group in grouped_interventions.items():
            # Skip groups with insufficient sources if required
            if len(intervention_group) < min_sources:
                continue

            try:
                consensus = self._create_research_consensus_intervention(intervention_group)

                if self._validate_research_consensus(consensus, confidence_threshold):
                    consensus_interventions.append(consensus)

                    # Update analysis statistics
                    if len(intervention_group) == 1:
                        analysis_stats['single_source_groups'] += 1
                    else:
                        analysis_stats['multi_source_groups'] += 1
                        if consensus.get('consensus_confidence', 0) > max(i.get('confidence_score', 0) for i in intervention_group):
                            analysis_stats['confidence_enhanced_cases'] += 1
                        analysis_stats['evidence_accumulation_cases'] += 1

            except Exception as e:
                self.logger.error(f"Failed to create research consensus for group {group_key}: {e}")

        # Generate comprehensive analysis metadata
        analysis_metadata = self._generate_research_analysis_metadata(
            interventions, consensus_interventions, analysis_stats
        )

        self.logger.info(f"Research consensus complete: {len(consensus_interventions)} consensus interventions created")

        return {
            'consensus_interventions': consensus_interventions,
            'analysis_metadata': analysis_metadata
        }

    def _group_for_research_analysis(self, interventions: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group interventions for cross-paper research analysis.

        Groups by canonical entities to identify the same intervention studied
        across different papers or by different models.

        Args:
            interventions: Interventions with canonical entities resolved

        Returns:
            Dictionary of grouped interventions for research analysis
        """
        if not interventions:
            return {}

        grouped_interventions = {}

        for intervention in interventions:
            # Create research grouping key based on canonical entities
            intervention_canonical_id = intervention.get('intervention_canonical_id', 'unknown')
            condition_canonical_id = intervention.get('condition_canonical_id', 'unknown')
            correlation_type = intervention.get('correlation_type', 'unknown')

            # Group by intervention-condition-correlation for research analysis
            research_key = f"{intervention_canonical_id}|{condition_canonical_id}|{correlation_type}"

            if research_key not in grouped_interventions:
                grouped_interventions[research_key] = []
            grouped_interventions[research_key].append(intervention)

        # Sort groups by evidence strength (number of sources)
        sorted_groups = dict(sorted(
            grouped_interventions.items(),
            key=lambda x: len(x[1]),
            reverse=True
        ))

        self.logger.info(f"Research grouping: {len(interventions)} interventions → {len(sorted_groups)} research groups")

        # Log interesting research groups
        multi_source_groups = {k: v for k, v in sorted_groups.items() if len(v) > 1}
        if multi_source_groups:
            self.logger.info(f"Found {len(multi_source_groups)} multi-source research groups for evidence accumulation")

        return sorted_groups

    def _create_research_consensus_intervention(self, intervention_group: List[Dict]) -> Dict:
        """
        Create a research-quality consensus intervention from multiple sources.

        This method focuses on creating high-quality research records by:
        - Accumulating evidence across papers
        - Enhancing confidence through cross-validation
        - Preserving research provenance and methodology

        Args:
            intervention_group: Group of similar interventions from research analysis

        Returns:
            Research consensus intervention with enhanced metadata
        """
        if len(intervention_group) == 1:
            # Single source - enhance with research metadata
            return self._enhance_single_source_for_research(intervention_group[0])

        # Multiple sources - create evidence-accumulated consensus
        return self._create_multi_source_research_consensus(intervention_group)

    def _enhance_single_source_for_research(self, intervention: Dict) -> Dict:
        """
        Enhance single-source intervention with research metadata.

        Args:
            intervention: Single intervention to enhance

        Returns:
            Enhanced intervention with research metadata
        """
        enhanced = intervention.copy()

        # Add research metadata
        enhanced['research_consensus_type'] = 'single_source'
        enhanced['evidence_sources'] = 1
        enhanced['research_confidence'] = enhanced.get('confidence_score', 0.5)
        enhanced['cross_validation'] = False
        enhanced['research_created_at'] = datetime.now().isoformat()

        # Maintain original confidence with research annotation
        enhanced['research_notes'] = 'Single source intervention - no cross-validation available'

        return enhanced

    def _create_multi_source_research_consensus(self, intervention_group: List[Dict]) -> Dict:
        """
        Create consensus from multiple research sources with evidence accumulation.

        Args:
            intervention_group: Multiple interventions to combine

        Returns:
            Research consensus with accumulated evidence
        """
        # Use highest confidence intervention as base for research purposes
        base_intervention = max(intervention_group, key=lambda x: x.get('confidence_score', 0) or 0)
        consensus = base_intervention.copy()

        # Collect research evidence
        evidence_sources = self._collect_research_evidence(intervention_group)
        consensus_confidence = self._calculate_research_consensus_confidence(intervention_group, evidence_sources)

        # Enhanced research metadata
        consensus.update({
            'research_consensus_type': 'multi_source',
            'evidence_sources': len(intervention_group),
            'research_confidence': consensus_confidence,
            'cross_validation': len(set(i.get('extraction_model', 'unknown') for i in intervention_group)) > 1,
            'research_created_at': datetime.now().isoformat(),

            # Source tracking for research purposes
            'source_papers': list(set(i.get('paper_pmid', i.get('pmid', 'unknown')) for i in intervention_group)),
            'source_models': list(set(i.get('extraction_model', 'unknown') for i in intervention_group)),
            'contributing_sources': len(intervention_group),

            # Evidence accumulation
            'accumulated_evidence': evidence_sources['combined_quotes'],
            'confidence_range': evidence_sources['confidence_range'],
            # 'correlation_strength_range': evidence_sources['correlation_strength_range'],  # REMOVED: field no longer exists

            # Research quality indicators
            'evidence_consistency': evidence_sources['consistency_score'],
            'methodological_diversity': len(evidence_sources['unique_methods']),
        })

        # Research notes
        consensus['research_notes'] = f"Consensus from {len(intervention_group)} sources with {evidence_sources['consistency_score']:.2f} consistency"

        return consensus

    def _collect_research_evidence(self, intervention_group: List[Dict]) -> Dict[str, Any]:
        """
        Collect and analyze research evidence from intervention group.

        Args:
            intervention_group: Group of interventions to analyze

        Returns:
            Dictionary with collected research evidence and analysis
        """
        evidence = {
            'combined_quotes': [],
            'confidence_scores': [],
            # 'correlation_strengths': [],  # REMOVED: correlation_strength field no longer exists
            'unique_methods': set(),
            'papers': set(),
            'models': set()
        }

        # Collect evidence from all sources
        for intervention in intervention_group:
            # Supporting quotes with attribution
            quote = intervention.get('supporting_quote', '').strip()
            if quote:
                paper_id = intervention.get('paper_pmid', intervention.get('pmid', 'unknown'))
                model = intervention.get('extraction_model', 'unknown')
                confidence = intervention.get('confidence_score', 0.5)
                evidence['combined_quotes'].append(f"[Paper {paper_id}, {model}, conf={confidence:.2f}]: {quote}")

            # Numerical evidence
            conf_score = intervention.get('confidence_score')
            if conf_score is not None:
                evidence['confidence_scores'].append(conf_score)

            # REMOVED: correlation_strength field no longer exists (removed Oct 16, 2025)
            # corr_strength = intervention.get('correlation_strength')
            # if corr_strength is not None:
            #     evidence['correlation_strengths'].append(corr_strength)

            # Methodological tracking
            method = intervention.get('extraction_method', intervention.get('mapping_method', 'unknown'))
            evidence['unique_methods'].add(method)

            # Source tracking
            evidence['papers'].add(intervention.get('paper_pmid', intervention.get('pmid', 'unknown')))
            evidence['models'].add(intervention.get('extraction_model', 'unknown'))

        # Calculate evidence quality metrics
        evidence['confidence_range'] = (
            (min(evidence['confidence_scores']), max(evidence['confidence_scores']))
            if evidence['confidence_scores'] else (0, 0)
        )

        # REMOVED: correlation_strength field no longer exists (removed Oct 16, 2025)
        # evidence['correlation_strength_range'] = (
        #     (min(evidence['correlation_strengths']), max(evidence['correlation_strengths']))
        #     if evidence['correlation_strengths'] else (0, 0)
        # )

        # Evidence consistency score (based on confidence score variance)
        if len(evidence['confidence_scores']) > 1:
            mean_conf = sum(evidence['confidence_scores']) / len(evidence['confidence_scores'])
            variance = sum((x - mean_conf) ** 2 for x in evidence['confidence_scores']) / len(evidence['confidence_scores'])
            evidence['consistency_score'] = max(0, 1 - variance)  # Higher = more consistent
        else:
            evidence['consistency_score'] = 1.0

        return evidence

    def _calculate_research_consensus_confidence(self, intervention_group: List[Dict],
                                               evidence: Dict[str, Any]) -> float:
        """
        Calculate research consensus confidence based on evidence accumulation.

        Args:
            intervention_group: Group of interventions
            evidence: Collected evidence analysis

        Returns:
            Research consensus confidence score
        """
        if not evidence['confidence_scores']:
            return 0.5

        # Base confidence from evidence
        mean_confidence = sum(evidence['confidence_scores']) / len(evidence['confidence_scores'])

        # Enhancement factors for research purposes
        enhancements = 0.0

        # Multi-source boost (evidence accumulation)
        if len(intervention_group) > 1:
            source_boost = min(0.1, len(intervention_group) * 0.03)  # Up to 10% boost
            enhancements += source_boost

        # Cross-model validation boost
        if len(evidence['models']) > 1:
            enhancements += 0.05  # 5% boost for cross-model validation

        # Cross-paper validation boost
        if len(evidence['papers']) > 1:
            enhancements += 0.05  # 5% boost for cross-paper validation

        # Consistency bonus
        consistency_bonus = evidence['consistency_score'] * 0.05  # Up to 5% for high consistency
        enhancements += consistency_bonus

        # Calculate final research confidence
        research_confidence = min(0.98, mean_confidence + enhancements)

        return research_confidence

    def _validate_research_consensus(self, consensus: Dict, confidence_threshold: float) -> bool:
        """
        Validate research consensus for quality and safety.

        Args:
            consensus: Research consensus to validate
            confidence_threshold: Minimum confidence threshold

        Returns:
            True if consensus passes research validation
        """
        # Basic field validation
        required_fields = ['intervention_name', 'research_confidence']
        for field in required_fields:
            if field not in consensus or consensus[field] is None:
                self.logger.error(f"Research consensus validation failed: missing {field}")
                return False

        # Research confidence threshold
        research_conf = consensus.get('research_confidence', 0)
        if research_conf < confidence_threshold:
            self.logger.warning(f"Research confidence {research_conf:.2f} below threshold {confidence_threshold}")
            return False

        # Medical safety validation using existing system
        intervention_name = consensus.get('intervention_name', '')
        condition_name = consensus.get('health_condition', '')

        if intervention_name and condition_name:
            if self.batch_processor._is_dangerous_match(intervention_name, condition_name):
                self.logger.error(f"DANGEROUS: Research consensus intervention-condition pair: {intervention_name} + {condition_name}")
                return False

        return True

    def _generate_research_analysis_metadata(self, original_interventions: List[Dict],
                                           consensus_interventions: List[Dict],
                                           analysis_stats: Dict[str, int]) -> Dict[str, Any]:
        """
        Generate comprehensive metadata about the research analysis process.

        Args:
            original_interventions: Original intervention list
            consensus_interventions: Final consensus interventions
            analysis_stats: Analysis statistics

        Returns:
            Research analysis metadata
        """
        # Calculate research quality metrics
        research_metadata = {
            'analysis_summary': {
                'input_interventions': len(original_interventions),
                'output_consensus': len(consensus_interventions),
                'reduction_ratio': len(consensus_interventions) / len(original_interventions) if original_interventions else 0,
                'evidence_accumulation_rate': analysis_stats['evidence_accumulation_cases'] / analysis_stats['total_groups'] if analysis_stats['total_groups'] > 0 else 0
            },

            'research_quality': {
                'multi_source_groups': analysis_stats['multi_source_groups'],
                'single_source_groups': analysis_stats['single_source_groups'],
                'confidence_enhanced_cases': analysis_stats['confidence_enhanced_cases'],
                'cross_validation_available': any(c.get('cross_validation', False) for c in consensus_interventions)
            },

            'evidence_analysis': self._analyze_evidence_distribution(consensus_interventions),

            'research_methodology': {
                'grouping_method': 'canonical_entity_based',
                'confidence_calculation': 'evidence_accumulated_weighted',
                'validation_approach': 'medical_safety_plus_threshold',
                'created_at': datetime.now().isoformat()
            }
        }

        return research_metadata

    def _analyze_evidence_distribution(self, consensus_interventions: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the distribution of evidence across consensus interventions.

        Args:
            consensus_interventions: List of consensus interventions

        Returns:
            Evidence distribution analysis
        """
        if not consensus_interventions:
            return {}

        evidence_counts = [c.get('evidence_sources', 1) for c in consensus_interventions]
        confidence_scores = [c.get('research_confidence', 0) for c in consensus_interventions]

        return {
            'evidence_source_distribution': {
                'min_sources': min(evidence_counts),
                'max_sources': max(evidence_counts),
                'avg_sources': sum(evidence_counts) / len(evidence_counts),
                'multi_source_percentage': len([c for c in evidence_counts if c > 1]) / len(evidence_counts) * 100
            },

            'confidence_distribution': {
                'min_confidence': min(confidence_scores),
                'max_confidence': max(confidence_scores),
                'avg_confidence': sum(confidence_scores) / len(confidence_scores),
                'high_confidence_percentage': len([c for c in confidence_scores if c > 0.8]) / len(confidence_scores) * 100
            }
        }

    def generate_research_summary_report(self, consensus_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable research summary report.

        Args:
            consensus_results: Results from create_research_consensus

        Returns:
            Formatted research summary report
        """
        consensus_interventions = consensus_results.get('consensus_interventions', [])
        metadata = consensus_results.get('analysis_metadata', {})

        report_lines = [
            "=== Intervention Research Consensus Analysis Report ===",
            "",
            f"Analysis Summary:",
            f"• Input interventions: {metadata.get('analysis_summary', {}).get('input_interventions', 0)}",
            f"• Consensus interventions: {len(consensus_interventions)}",
            f"• Data reduction: {metadata.get('analysis_summary', {}).get('reduction_ratio', 0):.1%}",
            "",
            f"Research Quality:",
            f"• Multi-source evidence groups: {metadata.get('research_quality', {}).get('multi_source_groups', 0)}",
            f"• Single-source groups: {metadata.get('research_quality', {}).get('single_source_groups', 0)}",
            f"• Confidence enhanced cases: {metadata.get('research_quality', {}).get('confidence_enhanced_cases', 0)}",
            "",
            f"Evidence Distribution:",
            f"• Average sources per intervention: {metadata.get('evidence_analysis', {}).get('evidence_source_distribution', {}).get('avg_sources', 0):.1f}",
            f"• Multi-source interventions: {metadata.get('evidence_analysis', {}).get('evidence_source_distribution', {}).get('multi_source_percentage', 0):.1f}%",
            f"• Average confidence: {metadata.get('evidence_analysis', {}).get('confidence_distribution', {}).get('avg_confidence', 0):.2f}",
            f"• High confidence interventions (>0.8): {metadata.get('evidence_analysis', {}).get('confidence_distribution', {}).get('high_confidence_percentage', 0):.1f}%",
            "",
            "=== End Report ==="
        ]

        return "\n".join(report_lines)


# === CONVENIENCE FUNCTIONS ===

def create_consensus_analyzer(batch_processor) -> InterventionConsensusAnalyzer:
    """
    Create an InterventionConsensusAnalyzer instance.

    Args:
        batch_processor: BatchEntityProcessor instance

    Returns:
        Configured InterventionConsensusAnalyzer
    """
    return InterventionConsensusAnalyzer(batch_processor)


def analyze_intervention_consensus(interventions: List[Dict], batch_processor,
                                 confidence_threshold: float = 0.5,
                                 min_sources: int = 1) -> Dict[str, Any]:
    """
    Convenience function to analyze intervention consensus for research.

    Args:
        interventions: List of interventions with canonical entities resolved
        batch_processor: BatchEntityProcessor instance
        confidence_threshold: Minimum confidence for consensus
        min_sources: Minimum sources required

    Returns:
        Research consensus results with metadata
    """
    analyzer = create_consensus_analyzer(batch_processor)
    return analyzer.create_research_consensus(
        interventions,
        confidence_threshold=confidence_threshold,
        min_sources=min_sources
    )