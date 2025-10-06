"""
Semantic Normalization Module

Hierarchical semantic normalization system for intervention names using:
- Embedding-based similarity (nomic-embed-text)
- LLM-based canonical extraction (qwen3:14b)
- 4-layer hierarchical structure
- 6 relationship types

Components:
- embedding_engine: Generate semantic embeddings
- llm_classifier: LLM-based canonical extraction and relationship classification
- hierarchy_manager: Database operations for hierarchical schema
- normalizer: Main normalization pipeline
- evaluator: Ground truth accuracy testing
- test_runner: Batch testing framework
- cluster_reviewer: Interactive manual review
- experiment_logger: Experiment documentation

Ground Truth Tools:
- ground_truth.labeling_interface: Interactive labeling interface
- ground_truth.pair_generator: Candidate pair generation
- ground_truth.label_in_batches: Batch labeling session management
"""

from .embedding_engine import EmbeddingEngine
from .llm_classifier import LLMClassifier
from .hierarchy_manager import HierarchyManager
from .normalizer import MainNormalizer
from .semantic_normalizer import SemanticNormalizer
# Note: evaluator, test_runner, cluster_reviewer, experiment_logger have import issues - fix later if needed

__all__ = [
    'EmbeddingEngine',
    'LLMClassifier',
    'HierarchyManager',
    'MainNormalizer',
    'SemanticNormalizer',
]

__version__ = '1.0.0'
__author__ = 'MyBiome Research Team'
