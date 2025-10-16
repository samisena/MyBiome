"""
Phase 4: Data Mining Pipeline

Integrates knowledge graph construction and Bayesian scoring into the main pipeline.
Consumes Phase 3 canonical groups for better statistical power and cleaner analytics.

Phases:
- Phase 4a: Knowledge Graph Construction (medical_knowledge_graph)
- Phase 4b: Bayesian Evidence Scoring (bayesian_scorer)
"""

__version__ = "1.0.0"
__all__ = [
    "MedicalKnowledgeGraph",
    "BayesianEvidenceScorer",
    "Phase4DataMiningOrchestrator"
]
