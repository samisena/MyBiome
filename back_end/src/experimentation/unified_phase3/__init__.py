"""
Unified Phase 3 Pipeline - Semantic Clustering for Interventions, Conditions, and Mechanisms

This module replaces Phase 3, 3.5, and 3.6 with a unified approach:
- Phase 3a: Semantic Embedding (nomic-embed-text or mxbai-embed-large)
- Phase 3b: Clustering (HDBSCAN or Hierarchical + Singleton Handler)
- Phase 3c: LLM Canonical Naming (qwen3:14b with configurable temperature)

Key Features:
- 100% assignment guarantee (no entity left uncategorized)
- Unified architecture for all entity types (interventions, conditions, mechanisms)
- Experiment framework for testing different configurations
- Temperature experimentation (0.0, 0.2, 0.3, 0.4) for naming quality

Directory Structure:
- config/: YAML configuration files
- embedders/: Embedding engine implementations
- clusterers/: Clustering algorithm implementations
- namers/: LLM naming implementations
- orchestrator.py: Main pipeline coordinator
- experiment_runner.py: Experiment execution framework
- evaluation.py: Result comparison and metrics

Usage:
    from back_end.src.experimentation.unified_phase3.orchestrator import UnifiedPhase3Orchestrator

    orchestrator = UnifiedPhase3Orchestrator(config_path='config/base_config.yaml')
    result = orchestrator.run()
"""

__version__ = "0.1.0"
__author__ = "MyBiome Research Team"
