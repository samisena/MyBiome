# Phase 3d: Multi-Level Hierarchical Cluster Merging

## Overview

Automated pipeline that builds 2-4 level hierarchies for mechanism/intervention/condition clusters through iterative bottom-up merging with hyperparameter optimization.

## Architecture

**Pipeline Stages**:
1. **Stage 0**: Hyperparameter optimization (grid search for optimal thresholds)
2. **Stage 1**: Centroid computation (cluster-level embeddings)
3. **Stage 2**: Candidate generation (similarity-based merge candidates)
4. **Stage 3**: LLM validation (auto-approval with quality checks)
5. **Stage 4**: Cross-category detection (identify mis-categorizations)
6. **Stage 5**: Merge application (database updates)

## File Structure

```
phase_3d_hierarchical_merging/
├── config.py                           # Configuration management
├── validation_metrics.py               # Quality scoring functions
├── stage_0_hyperparameter_optimizer.py # Grid search
├── stage_1_centroid_computation.py     # Centroids
├── stage_2_candidate_generation.py     # Candidates
├── stage_3_llm_validation.py          # LLM validation
├── stage_4_cross_category_detection.py # Category detection
├── stage_5_merge_application.py       # Database updates
├── orchestrator.py                    # Main coordinator
├── test_stage_*.py                    # Unit tests per stage
├── test_integration.py                # Full pipeline test
└── results/                           # Output directory
```

## Usage

### Run Individual Stages (Testing)

```python
# Stage 0: Hyperparameter optimization
from stage_0_hyperparameter_optimizer import HyperparameterOptimizer
optimizer = HyperparameterOptimizer(db_path='...', entity_type='mechanism')
optimal_configs = optimizer.optimize()

# Stage 1: Compute centroids
from stage_1_centroid_computation import compute_centroids
centroids = compute_centroids(clusters, embeddings)

# ... etc for other stages
```

### Run Full Pipeline

```python
from orchestrator import PhaseThreeDOrchestrator

orchestrator = PhaseThreeDOrchestrator(
    db_path='back_end/data/intervention_research.db',
    entity_type='mechanism'
)

result = orchestrator.run()
```

### Run Tests

```bash
# Individual stage tests
python test_stage_0.py
python test_stage_1.py
python test_stage_2.py
...

# Integration test
python test_integration.py
```

## Expected Outputs

**Mechanisms**: 415 clusters → ~200-250 top-level (2-3 levels deep)
**Interventions**: ~571 groups → ~350-400 top-level (2-4 levels deep)
**Conditions**: ~200-300 groups → ~150-200 top-level (2-3 levels deep)

## Implementation Timeline

**Week 1**: Setup + Stages 0-1 (hyperparameter, centroids)
**Week 2**: Stages 2-3 (candidates, LLM validation)
**Week 3**: Stages 4-5 (detection, database updates)
**Week 4**: Orchestrator + integration + production

## Success Criteria

- All unit tests pass
- Integration test passes
- Hyperparameter optimization score >70
- Cluster reduction 40-50%
- Hierarchy depth 2-3 levels
- No database corruption
- Execution time <30 minutes

## References

- CLAUDE.md: Project documentation
- Phase 3: Semantic normalization (back_end/src/semantic_normalization/)
- Phase 3.6: Mechanism clustering (rotation_mechanism_clustering.py)
