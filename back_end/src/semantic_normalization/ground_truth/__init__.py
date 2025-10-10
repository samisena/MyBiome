"""
Ground Truth Management for Semantic Normalization

Tools for creating and managing ground truth datasets for hierarchical
semantic normalization evaluation.

Components:
- labeling_interface: Interactive terminal-based labeling interface
- pair_generator: Strategic candidate pair generation (similarity-based, random, targeted)
- ground_truth_cli: Unified CLI for workflow (generate, label, status, clean)
- data_exporter: Export interventions from database for labeling

Features:
- 50 existing labeled pairs (hierarchical_ground_truth_50_pairs.json)
- 500 candidate pairs ready for labeling (hierarchical_candidates_500_pairs.json)
- Stratified sampling (60% similarity + 20% random + 20% targeted)
- 6 relationship types (EXACT_MATCH, VARIANT, SUBTYPE, SAME_CATEGORY, DOSAGE_VARIANT, DIFFERENT)
- 4-layer hierarchy (Category → Canonical → Variant → Dosage)
- Batch mode (50 pairs per session)
- Progress tracking and resume capability
- Undo, skip, review later features

Usage:
    # Unified CLI interface
    cd back_end/src/semantic_normalization/ground_truth

    # Generate candidate pairs
    python ground_truth_cli.py generate

    # Start labeling session
    python ground_truth_cli.py label --batch-size 50

    # Check progress
    python ground_truth_cli.py status

    # Remove duplicate labels
    python ground_truth_cli.py clean
"""

from .labeling_interface import HierarchicalLabelingInterface
from .pair_generator import SmartPairGenerator
from .data_exporter import InterventionDataExporter

__all__ = [
    'HierarchicalLabelingInterface',
    'SmartPairGenerator',
    'InterventionDataExporter',
]
