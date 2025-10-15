"""
Phase 3d: Hierarchical Cluster Merging

Multi-level hierarchical clustering with cross-category functional grouping.

Stages:
- Stage 0: Hyperparameter optimization (optional)
- Stage 1: Centroid computation
- Stage 2: Candidate generation (top-k similar pairs)
- Stage 3: LLM validation (semantic coherence)
- Stage 3.5: Functional grouping (cross-category detection)
- Stage 4: Cross-category detection (deprecated - merged into 3.5)
- Stage 5: Merge application (hierarchy construction)

Key Features:
- Multi-category support (primary, functional, therapeutic, etc.)
- HDBSCAN clustering with embedding-based similarity
- LLM validation for semantic coherence (qwen3:14b)
- Up to 4-level hierarchies (great-grandparent → grandparent → parent → child)
- Cross-category pattern discovery

Usage:
    from back_end.src.phase_3_semantic_normalization.phase_3d import config
    from back_end.src.phase_3_semantic_normalization.phase_3d.stage_3_5_functional_grouping import FunctionalGrouper
"""

from .config import Phase3dConfig

__all__ = ['Phase3dConfig']
