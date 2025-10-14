"""
Configuration for Semantic Normalization Module

Centralized configuration for hierarchical semantic normalization system.
Paths are now relative to the main codebase structure.
"""

from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent  # back_end/
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"

# Semantic normalization paths
SEMANTIC_DIR = SRC_DIR / "semantic_normalization"
GROUND_TRUTH_DIR = SEMANTIC_DIR / "ground_truth"
GROUND_TRUTH_DATA_DIR = GROUND_TRUTH_DIR / "data"

# Cache and output paths
CACHE_DIR = DATA_DIR / "semantic_normalization_cache"
RESULTS_DIR = DATA_DIR / "semantic_normalization_results"
LOGS_DIR = DATA_DIR / "logs" / "semantic_normalization"

# Database paths
DB_PATH = DATA_DIR / "processed" / "intervention_research.db"

# Ensure directories exist
for directory in [CACHE_DIR, RESULTS_DIR, LOGS_DIR, GROUND_TRUTH_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Database configuration
DATABASE_CONFIG = {
    "path": str(DB_PATH),
    "export_limit": 500,  # Maximum interventions to export
}

# Data export configuration
EXPORT_CONFIG = {
    "min_frequency": 1,  # Minimum occurrence count to include
    "include_metadata": True,
    "output_dir": str(GROUND_TRUTH_DATA_DIR / "samples"),
}

# Ground truth labeling configuration
LABELING_CONFIG = {
    "target_pairs": 500,  # Target number of labeled pairs
    "candidate_pool_size": 600,  # Number of candidate pairs to generate
    "similarity_threshold_min": 0.30,  # Minimum similarity for candidates
    "similarity_threshold_max": 0.95,  # Maximum similarity for candidates
    "output_dir": str(GROUND_TRUTH_DATA_DIR),

    # Hierarchical relationship types
    "relationship_types": {
        1: {
            "code": "EXACT_MATCH",
            "display": "Exact Match (same intervention, same formulation)",
            "description": "Identical interventions, including synonyms and equivalent names",
            "aggregation": "merge_completely",
            "examples": [
                "vitamin D = cholecalciferol",
                "PPI = proton pump inhibitor"
            ]
        },
        2: {
            "code": "VARIANT",
            "display": "Variant (same concept, different formulation)",
            "description": "Same therapeutic concept but different formulation or biosimilar",
            "aggregation": "share_layer_1_link_layer_2",
            "examples": [
                "Cetuximab vs Cetuximab-beta (biosimilar)",
                "insulin glargine vs insulin detemir"
            ]
        },
        3: {
            "code": "SUBTYPE",
            "display": "Subtype (related but clinically distinct)",
            "description": "Related subtypes of the same parent condition or intervention class",
            "aggregation": "share_layer_1_separate_layer_2",
            "examples": [
                "IBS-D vs IBS-C",
                "type 1 diabetes vs type 2 diabetes"
            ]
        },
        4: {
            "code": "SAME_CATEGORY",
            "display": "Same Category (different entities in same class)",
            "description": "Different members of the same intervention category",
            "aggregation": "separate_all_layers",
            "examples": [
                "L. reuteri vs S. boulardii (both probiotics)",
                "atorvastatin vs simvastatin (both statins)"
            ]
        },
        5: {
            "code": "DOSAGE_VARIANT",
            "display": "Dosage Variant (same intervention, different dose)",
            "description": "Same intervention with explicit dosage differences",
            "aggregation": "share_layers_1_2",
            "examples": [
                "metformin vs metformin 500mg",
                "vitamin D 1000 IU vs vitamin D 5000 IU"
            ]
        },
        6: {
            "code": "DIFFERENT",
            "display": "Different (completely unrelated interventions)",
            "description": "No relationship between interventions",
            "aggregation": "no_relationship",
            "examples": [
                "vitamin D vs chemotherapy",
                "exercise vs surgery"
            ]
        }
    }
}

# Fuzzy matching configuration (for candidate generation)
FUZZY_MATCHING_CONFIG = {
    "algorithm": "jaro_winkler",  # Options: levenshtein, jaro_winkler, token_sort
    "score_threshold": 0.40,  # Minimum score to consider as candidate
}

# Hierarchical semantic layers configuration
HIERARCHY_CONFIG = {
    "similarity_thresholds": {
        "layer_0_category": 1.0,    # Exact category match (from taxonomy)
        "layer_1_canonical": 0.70,  # Broad semantic grouping
        "layer_2_variant": 0.90,    # Specific entity matching
        "layer_3_detail": 0.95      # Dosage/formulation exact match
    },

    "layer_definitions": {
        "layer_0": {
            "name": "Category",
            "description": "Intervention category from taxonomy (supplement, medication, etc.)",
            "source": "existing_intervention_categories_table"
        },
        "layer_1": {
            "name": "Canonical Entity",
            "description": "Semantic group (probiotics, statins, cetuximab, IBS)",
            "aggregation_level": "broad"
        },
        "layer_2": {
            "name": "Specific Variant",
            "description": "Exact entity (L. reuteri, atorvastatin, cetuximab-β, IBS-D)",
            "aggregation_level": "specific"
        },
        "layer_3": {
            "name": "Dosage/Details",
            "description": "Granular details (dosage, administration method)",
            "aggregation_level": "granular"
        }
    }
}

# Embedding engine configuration
EMBEDDING_CONFIG = {
    "model": "nomic-embed-text",
    "base_url": "http://localhost:11434",
    "cache_path": str(CACHE_DIR / "embeddings.pkl"),
    "batch_size": 32,
    "dimension": 768  # nomic-embed-text dimension
}

# LLM classifier configuration
LLM_CONFIG = {
    "model": "qwen3:14b",
    "base_url": "http://localhost:11434",
    "cache_path": str(CACHE_DIR / "llm_decisions.pkl"),
    "temperature": 0.1,  # Low temperature for consistency
    "timeout": 60  # seconds
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_enabled": True,
    "file_path": str(LOGS_DIR / "semantic_normalization.log")
}


def get_config() -> Dict[str, Any]:
    """
    Get complete configuration dictionary.

    Returns:
        Dictionary with all configuration sections
    """
    return {
        "paths": {
            "base_dir": str(BASE_DIR),
            "data_dir": str(DATA_DIR),
            "semantic_dir": str(SEMANTIC_DIR),
            "ground_truth_dir": str(GROUND_TRUTH_DIR),
            "ground_truth_data_dir": str(GROUND_TRUTH_DATA_DIR),
            "cache_dir": str(CACHE_DIR),
            "results_dir": str(RESULTS_DIR),
            "logs_dir": str(LOGS_DIR),
            "db_path": str(DB_PATH)
        },
        "database": DATABASE_CONFIG,
        "export": EXPORT_CONFIG,
        "labeling": LABELING_CONFIG,
        "fuzzy_matching": FUZZY_MATCHING_CONFIG,
        "hierarchy": HIERARCHY_CONFIG,
        "embedding": EMBEDDING_CONFIG,
        "llm": LLM_CONFIG,
        "logging": LOGGING_CONFIG
    }


def get_ground_truth_files() -> Dict[str, Path]:
    """
    Get paths to ground truth data files.

    Returns:
        Dictionary with ground truth file paths
    """
    return {
        "ground_truth_50": GROUND_TRUTH_DATA_DIR / "hierarchical_ground_truth_50_pairs.json",
        "candidates_500": GROUND_TRUTH_DATA_DIR / "hierarchical_candidates_500_pairs.json",
        "session_latest": max(
            GROUND_TRUTH_DATA_DIR.glob("labeling_session_*.json"),
            key=lambda p: p.stat().st_mtime,
            default=None
        ) if GROUND_TRUTH_DATA_DIR.exists() else None
    }


# Convenience exports
__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'SEMANTIC_DIR',
    'GROUND_TRUTH_DIR',
    'GROUND_TRUTH_DATA_DIR',
    'CACHE_DIR',
    'RESULTS_DIR',
    'LOGS_DIR',
    'DB_PATH',
    'DATABASE_CONFIG',
    'EXPORT_CONFIG',
    'LABELING_CONFIG',
    'FUZZY_MATCHING_CONFIG',
    'HIERARCHY_CONFIG',
    'EMBEDDING_CONFIG',
    'LLM_CONFIG',
    'LOGGING_CONFIG',
    'get_config',
    'get_ground_truth_files'
]
