"""
Semantic Normalization Module

Unified Phase 3 pipeline with clustering-first architecture:
- Phase 3a: Semantic Embedding (mxbai-embed-large, 1024-dim)
- Phase 3b: Clustering (Hierarchical with threshold=0.7 + Singleton Handler)
- Phase 3c: LLM Canonical Naming (qwen3:14b with temperature control)

Components:

Phase 3a (Embedders):
- InterventionEmbedder, ConditionEmbedder, MechanismEmbedder
- BaseEmbedder (abstract base class)

Phase 3b (Clusterers):
- HierarchicalClusterer, HDBSCANClusterer
- SingletonHandler (100% assignment guarantee)
- BaseClusterer (abstract base class)

Phase 3c (Namers):
- LLMNamer (canonical naming + categorization)
- BaseNamer (abstract base class)

Legacy Components (OLD - naming-first architecture):
- EmbeddingEngine, LLMClassifier, HierarchyManager, MainNormalizer, SemanticNormalizer

Ground Truth Tools:
- ground_truth.labeling_interface: Interactive labeling interface
- ground_truth.pair_generator: Candidate pair generation
- ground_truth.label_in_batches: Batch labeling session management
"""

# Phase 3a: Embedders
from .phase_3a_base_embedder import BaseEmbedder
from .phase_3a_intervention_embedder import InterventionEmbedder
from .phase_3a_condition_embedder import ConditionEmbedder
from .phase_3a_mechanism_embedder import MechanismEmbedder

# Phase 3b: Clusterers
from .phase_3b_base_clusterer import BaseClusterer
from .phase_3b_hierarchical_clusterer import HierarchicalClusterer
from .phase_3b_hdbscan_clusterer import HDBSCANClusterer
from .phase_3b_singleton_handler import SingletonHandler

# Phase 3c: Namers
from .phase_3c_base_namer import BaseNamer, ClusterData, NamingResult
from .phase_3c_llm_namer import LLMNamer

# Main Orchestrator
from .phase_3abc_orchestrator import UnifiedPhase3Orchestrator, EntityResults

# Phase 3c Stage 2: Category Consolidation
from .phase_3c_category_consolidator import CategoryConsolidator, CategoryInfo, ConsolidationMapping, ConsolidationResult

# Legacy imports removed - files deleted during migration

__all__ = [
    # Phase 3a
    'BaseEmbedder',
    'InterventionEmbedder',
    'ConditionEmbedder',
    'MechanismEmbedder',
    # Phase 3b
    'BaseClusterer',
    'HierarchicalClusterer',
    'HDBSCANClusterer',
    'SingletonHandler',
    # Phase 3c
    'BaseNamer',
    'ClusterData',
    'NamingResult',
    'LLMNamer',
    'CategoryConsolidator',
    'CategoryInfo',
    'ConsolidationMapping',
    'ConsolidationResult',
    # Orchestrator
    'UnifiedPhase3Orchestrator',
    'EntityResults',
]

__version__ = '1.0.0'
__author__ = 'MyBiome Research Team'
