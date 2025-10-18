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
from .phase_3_orchestrator import UnifiedPhase3Orchestrator, EntityResults

# Backward compatibility wrapper for legacy code
# TODO: Update phase_3_semantic_normalizer.py to use UnifiedPhase3Orchestrator directly
class SemanticNormalizer(UnifiedPhase3Orchestrator):
    """Legacy wrapper for UnifiedPhase3Orchestrator with simplified API."""
    def __init__(self, db_path: str):
        # Get config path from package directory
        from pathlib import Path
        config_path = Path(__file__).parent / "phase_3_config.yaml"
        cache_dir = Path(__file__).parent.parent.parent / "data" / "semantic_normalization_cache"
        super().__init__(
            config_path=str(config_path),
            db_path=db_path,
            cache_dir=str(cache_dir)
        )

    def normalize_interventions(self, interventions=None, entity_type: str = 'intervention',
                                source_table: str = 'interventions', force_reembed: bool = False,
                                force_recluster: bool = False, batch_size: int = 20):
        """
        Legacy method for backward compatibility - maps to run_pipeline().

        Note: The 'interventions' parameter is ignored because UnifiedPhase3Orchestrator
        processes all entities in the database, not a subset.
        """
        result = self.run_pipeline(
            entity_type=entity_type,
            force_reembed=force_reembed,
            force_recluster=force_recluster
        )

        # Convert EntityResults dataclass to legacy dict format
        return {
            'total_processed': result.embeddings_generated,
            'canonical_groups_created': result.num_clusters,
            'relationships_created': 0,  # Legacy field not used by new orchestrator
            'errors': 0
        }

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
    # Legacy Compatibility
    'SemanticNormalizer',
]

__version__ = '1.0.0'
__author__ = 'MyBiome Research Team'
