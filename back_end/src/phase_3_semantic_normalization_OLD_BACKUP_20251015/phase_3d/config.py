"""
Configuration for Phase 3d Hierarchical Cluster Merging

Centralized configuration management for all stages.
"""

from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "back_end" / "data" / "intervention_research.db"
CACHE_DIR = PROJECT_ROOT / "back_end" / "data" / "semantic_normalization_cache"
RESULTS_DIR = Path(__file__).parent / "results"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Phase3dConfig:
    """Configuration for Phase 3d pipeline."""

    # Database
    db_path: str = str(DB_PATH)
    cache_dir: str = str(CACHE_DIR)
    results_dir: str = str(RESULTS_DIR)

    # Entity type to process
    entity_type: str = 'mechanism'  # 'mechanism', 'intervention', or 'condition'

    # Hyperparameter optimization (Stage 0)
    threshold_search_space: Dict[str, List[float]] = field(default_factory=lambda: {
        'level_3_to_2': [0.82, 0.84, 0.86, 0.88],
        'level_2_to_1': [0.78, 0.80, 0.82, 0.84],
        'level_1_to_0': [0.74, 0.76, 0.78, 0.80]
    })

    # Scoring weights for hyperparameter optimization
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        'reduction': 25.0,      # Cluster count reduction
        'depth': 15.0,          # Hierarchy depth
        'size_distribution': 20.0,  # Balanced cluster sizes
        'coherence': 25.0,      # Intra-cluster similarity
        'separation': 15.0      # Inter-cluster distinction
    })

    # Target ranges for optimal configuration
    target_reduction_range: tuple = (0.40, 0.60)  # 40-60% reduction
    target_depth_range: tuple = (2, 3)  # 2-3 levels deep
    target_singleton_threshold: float = 0.50  # <50% singletons
    target_coherence_min: float = 0.65  # Avg intra-cluster similarity
    target_separation_min: float = 0.35  # Avg inter-cluster distinction

    # LLM validation (Stage 3)
    llm_model: str = "qwen3:14b"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.4  # Temperature for LLM calls
    llm_timeout: int = 60  # seconds
    llm_batch_size: int = 1  # Process candidates one at a time

    # Auto-approval thresholds
    auto_approve_confidence: str = "HIGH"  # Only auto-approve HIGH confidence
    auto_approve_name_quality_min: int = 60  # Name quality score >= 60
    auto_approve_diversity_max_severity: str = "MODERATE"  # Allow MODERATE warnings

    # Name quality scoring
    forbidden_generic_terms: List[str] = field(default_factory=lambda: [
        "intervention", "treatment", "therapy", "approach",
        "mechanism", "supplement", "medication", "procedure",
        "interventions", "treatments", "therapies", "approaches"
    ])

    specific_term_indicators: List[str] = field(default_factory=lambda: [
        "probiotic", "anti-inflammatory", "glucose", "insulin",
        "cardiovascular", "gut microbiome", "enzyme", "receptor",
        "aerobic", "resistance", "cox", "vitamin", "mineral"
    ])

    # Diversity check
    diversity_warning_threshold: float = 0.40  # Warn if inter-child similarity <0.40
    diversity_severe_threshold: float = 0.30  # SEVERE if <0.30

    # Embedding model
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768

    # Database tables
    clusters_table: str = 'mechanism_clusters'  # or 'canonical_groups' for interventions/conditions
    membership_table: str = 'mechanism_cluster_membership'

    # Logging
    log_level: str = "INFO"
    save_intermediate_results: bool = True

    # Performance
    max_hierarchy_depth: int = 4  # Stop after 4 levels

    def get_table_names(self, entity_type: str) -> Dict[str, str]:
        """
        Get appropriate table names for entity type.

        Args:
            entity_type: 'mechanism', 'intervention', or 'condition'

        Returns:
            Dict with table names
        """
        if entity_type == 'mechanism':
            return {
                'clusters': 'mechanism_clusters',
                'membership': 'mechanism_cluster_membership',
                'interventions_junction': 'intervention_mechanisms'
            }
        elif entity_type in ['intervention', 'condition']:
            return {
                'clusters': 'canonical_groups',
                'hierarchy': 'semantic_hierarchy',
                'relationships': 'entity_relationships'
            }
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")


# Global configuration instance
config = Phase3dConfig()


def update_config(**kwargs):
    """
    Update global configuration.

    Usage:
        update_config(entity_type='intervention', llm_model='qwen3:14b')
    """
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")


def get_config() -> Phase3dConfig:
    """Get current configuration."""
    return config
