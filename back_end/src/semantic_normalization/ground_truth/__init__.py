"""
Ground Truth Management for Semantic Normalization

Tools for creating and managing ground truth datasets for hierarchical
semantic normalization evaluation.

Components:
- labeling_interface: Interactive terminal-based labeling interface
- pair_generator: Strategic candidate pair generation (similarity-based, random, targeted)
- label_in_batches: Batch labeling session management
- generate_candidates: 500-pair candidate generation script
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
    # Generate candidates
    from back_end.src.semantic_normalization.ground_truth.generate_candidates import main
    main()

    # Start labeling session
    python -m back_end.src.semantic_normalization.ground_truth.label_in_batches --batch-size 50

    # Check progress
    python -m back_end.src.semantic_normalization.ground_truth.label_in_batches --status
"""

from .labeling_interface import HierarchicalLabelingInterface
from .pair_generator import SmartPairGenerator
from .data_exporter import InterventionDataExporter

__all__ = [
    'HierarchicalLabelingInterface',
    'SmartPairGenerator',
    'InterventionDataExporter',
]
