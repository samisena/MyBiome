# Unified Phase 3 Pipeline - Semantic Clustering Experiment Framework

**Status**: Experimental (Development Phase)
**Version**: 0.1.0
**Goal**: Merge Phase 3, 3.5, and 3.6 into unified semantic clustering pipeline with temperature experimentation

---

## Overview

This experiment framework replaces the current 3-phase architecture (Phases 3, 3.5, 3.6) with a **unified approach** based on:
1. **Phase 3a**: Semantic Embedding (nomic-embed-text or mxbai-embed-large)
2. **Phase 3b**: Clustering (HDBSCAN or Hierarchical + Singleton Handler)
3. **Phase 3c**: LLM Canonical Naming (qwen3:14b with configurable temperature: 0.0, 0.2, 0.3, 0.4)

### Key Features

- **100% Assignment Guarantee**: No entity left uncategorized (via singleton clusters)
- **Temperature Experimentation**: Test 4 different temperatures (0.0, 0.2, 0.3, 0.4) for LLM naming quality/creativity trade-off
- **Modular Design**: Easy to swap embedding models, clustering algorithms, or naming strategies
- **Universal Architecture**: Same framework handles interventions, conditions, AND mechanisms
- **Aggressive Caching**: Embeddings cached to enable fast re-runs with different hyperparameters

---

## Directory Structure

```
unified_phase3/
├── __init__.py                 # Module initialization
├── README.md                   # This file
├── experiment_schema.sql       # Database schema for tracking experiments
├── config/
│   ├── base_config.yaml        # Default configuration
│   └── experiment_configs/     # 6 experiment configurations
│       ├── exp_001_nomic_hdbscan_temp00.yaml  # Baseline (deterministic)
│       ├── exp_002_nomic_hdbscan_temp02.yaml  # Slightly creative
│       ├── exp_003_nomic_hdbscan_temp03.yaml  # Balanced
│       ├── exp_004_nomic_hdbscan_temp04.yaml  # More diverse
│       ├── exp_005_mxbai_hdbscan_temp00.yaml  # Better for long text
│       └── exp_006_nomic_hierarchical_temp00.yaml  # Alternative clustering
├── embedders/                  # Phase 3a: Semantic Embedding
│   ├── __init__.py
│   ├── base_embedder.py        # Abstract base class
│   ├── intervention_embedder.py
│   ├── condition_embedder.py
│   └── mechanism_embedder.py
├── clusterers/                 # Phase 3b: Clustering
│   ├── __init__.py
│   ├── base_clusterer.py       # Abstract base class
│   ├── hdbscan_clusterer.py    # Density-based clustering
│   ├── hierarchical_clusterer.py  # Distance-based clustering
│   └── singleton_handler.py    # 100% assignment guarantee
├── namers/                     # Phase 3c: LLM Naming
│   ├── __init__.py
│   ├── base_namer.py           # Abstract base class
│   └── llm_namer.py            # qwen3:14b with temperature support
├── orchestrator.py             # Main pipeline coordinator (TODO)
├── experiment_runner.py        # Experiment execution framework (TODO)
├── evaluation.py               # Result comparison and metrics (TODO)
└── tests/                      # Unit tests (TODO)
    ├── test_embedders.py
    ├── test_clusterers.py
    └── test_namers.py
```

---

## Configuration

### Base Configuration ([base_config.yaml](config/base_config.yaml))

The base config defines default settings for all components:

```yaml
# Phase 3a: Embedding settings per entity type
embedding:
  interventions:
    model: "nomic-embed-text"
    dimension: 768
    batch_size: 32
    normalization: "l2"
  conditions:
    model: "nomic-embed-text"
    dimension: 768
  mechanisms:
    model: "nomic-embed-text"  # or "mxbai-embed-large"
    dimension: 768

# Phase 3b: Clustering settings per entity type
clustering:
  interventions:
    algorithm: "hdbscan"
    min_cluster_size: 2
    min_samples: 1
  conditions:
    algorithm: "hdbscan"
    min_cluster_size: 2
  mechanisms:
    algorithm: "hdbscan"
    min_cluster_size: 2

# Phase 3c: LLM naming settings
naming:
  llm:
    model: "qwen3:14b"
    temperature: 0.0  # Experiment: 0.0, 0.2, 0.3, 0.4
    max_tokens: 2000
    timeout: 60
```

### Experiment Configurations

Six pre-configured experiments are provided in [config/experiment_configs/](config/experiment_configs/):

| Experiment | Embedding | Clustering | Temperature | Purpose |
|-----------|-----------|------------|-------------|---------|
| exp_001   | nomic     | HDBSCAN    | 0.0         | **Baseline** (deterministic) |
| exp_002   | nomic     | HDBSCAN    | 0.2         | Slightly creative naming |
| exp_003   | nomic     | HDBSCAN    | 0.3         | Balanced creativity |
| exp_004   | nomic     | HDBSCAN    | 0.4         | More diverse naming |
| exp_005   | mxbai*    | HDBSCAN    | 0.0         | Better for long mechanisms |
| exp_006   | nomic     | Hierarchical | 0.0       | Alternative clustering |

*exp_005 uses mxbai-embed-large (1024-dim) for mechanisms only, nomic for interventions/conditions

---

## Components

### Phase 3a: Embedders

**Base Class**: [BaseEmbedder](embedders/base_embedder.py)
- Hash-based caching (JSON/pickle)
- Batch processing
- L2/unit-sphere normalization
- Performance tracking (cache hits, total embeddings)

**Implementations**:
1. **InterventionEmbedder** ([intervention_embedder.py](embedders/intervention_embedder.py))
   - Uses nomic-embed-text (768-dim) via Ollama API
   - Optional context enhancement (dosage, duration)
   - Method: `embed_interventions_from_db(db_path)`

2. **ConditionEmbedder** ([condition_embedder.py](embedders/condition_embedder.py))
   - Uses nomic-embed-text (768-dim)
   - Optional medical context (symptoms, ICD codes)
   - Method: `embed_conditions_from_db(db_path)`

3. **MechanismEmbedder** ([mechanism_embedder.py](embedders/mechanism_embedder.py))
   - Supports **nomic-embed-text (768-dim)** OR **mxbai-embed-large (1024-dim)**
   - Designed for longer text (mechanism descriptions)
   - Optional truncation for very long mechanisms
   - Method: `embed_mechanisms_from_db(db_path)`

**Example Usage**:
```python
from embedders import InterventionEmbedder

embedder = InterventionEmbedder(
    model="nomic-embed-text",
    dimension=768,
    cache_path="cache/interventions.pkl"
)

embeddings, intervention_names = embedder.embed_interventions_from_db(
    "intervention_research.db"
)

print(f"Generated {len(embeddings)} embeddings")
print(f"Cache hit rate: {embedder.get_stats()['hit_rate']:.2%}")
```

### Phase 3b: Clusterers

**Base Class**: [BaseClusterer](clusterers/base_clusterer.py)
- Clustering result caching (pickle)
- Quality metrics (silhouette, Davies-Bouldin)
- Hyperparameter tracking
- Metadata generation (cluster sizes, assignment rate)

**Implementations**:
1. **HDBSCANClusterer** ([hdbscan_clusterer.py](clusterers/hdbscan_clusterer.py))
   - Density-based clustering with hierarchical structure
   - Automatic cluster discovery (no need to specify count)
   - Handles noise/outliers (assigns -1 label)
   - Configurable: `min_cluster_size`, `min_samples`, `metric`

2. **HierarchicalClusterer** ([hierarchical_clusterer.py](clusterers/hierarchical_clusterer.py))
   - Agglomerative clustering with distance-based merging
   - Deterministic results (same input = same clusters)
   - No noise points (100% assignment by design)
   - Configurable: `linkage`, `distance_threshold`, `n_clusters`

3. **SingletonHandler** ([singleton_handler.py](clusterers/singleton_handler.py))
   - **Guarantees 100% assignment** (no -1 labels after processing)
   - Creates unique singleton clusters for outliers
   - Preserves existing cluster structure
   - Optional: Merge similar singletons based on embedding similarity

**Example Usage**:
```python
from clusterers import HDBSCANClusterer, SingletonHandler

# Cluster embeddings
clusterer = HDBSCANClusterer(
    min_cluster_size=2,
    min_samples=1,
    cache_path="cache/clusters.pkl"
)

cluster_labels, metadata = clusterer.cluster(embeddings)
print(f"Discovered {metadata['num_clusters']} clusters")
print(f"Noise points: {metadata['num_noise']}")

# Ensure 100% assignment
handler = SingletonHandler()
final_labels, singleton_metadata = handler.process_labels(
    cluster_labels,
    entity_names=intervention_names
)

print(f"Total clusters: {singleton_metadata['total_clusters']}")
print(f"Assignment rate: {singleton_metadata['assignment_rate']:.0%}")  # Should be 100%
```

### Phase 3c: Namers

**Base Class**: [BaseNamer](namers/base_namer.py)
- Cluster data abstraction (`ClusterData`, `NamingResult`)
- Batch processing (20 clusters per LLM call)
- Naming result caching (JSON)
- Fallback name generation on LLM failure

**Implementation**:
**LLMNamer** ([llm_namer.py](namers/llm_namer.py))
- Uses qwen3:14b via Ollama API
- **Temperature experimentation**: 0.0, 0.2, 0.3, 0.4
- Entity-type-specific prompts (interventions, conditions, mechanisms)
- JSON parsing with fallback
- Retry logic (3 attempts with exponential backoff)
- Tracks provenance (model, temperature, members shown)

**Example Usage**:
```python
from namers import LLMNamer, ClusterData

# Create namer with temperature
namer = LLMNamer(
    model="qwen3:14b",
    temperature=0.2,  # Slightly creative
    cache_path="cache/naming_temp02.json"
)

# Prepare cluster data
clusters = [
    ClusterData(
        cluster_id=0,
        entity_type='intervention',
        member_entities=['Vitamin D3', 'cholecalciferol', 'vitamin D 1000IU'],
        member_frequencies=[12, 8, 5],
        singleton=False
    ),
    # ... more clusters
]

# Name clusters
results = namer.name_clusters(clusters, batch_size=20)

for cluster_id, naming_result in results.items():
    print(f"Cluster {cluster_id}: {naming_result.canonical_name} ({naming_result.category})")
```

---

## Temperature Experimentation

### Goal
Find optimal LLM temperature for canonical naming by testing 4 values:
- **0.0**: Deterministic, most conservative (baseline)
- **0.2**: Slightly creative, consistent
- **0.3**: Balanced creativity
- **0.4**: More diverse, risk of inconsistency

### Metrics
1. **Consistency**: Same cluster gets similar name across temperatures (Levenshtein distance)
2. **Quality**: Manual review + heuristics (name length, semantic appropriateness)
3. **Creativity**: Better names for ambiguous clusters
4. **Stability**: JSON parsing success rate, LLM failure rate

### Expected Results
- **Temperature 0.0**: Highest consistency (baseline)
- **Temperature 0.2**: Similar consistency, slightly better creativity
- **Temperature 0.3**: Moderate creativity, acceptable consistency
- **Temperature 0.4**: High creativity, potential instability (monitor closely)

### Optimal Temperature
Likely **0.0** or **0.2** based on medical domain requirements (consistency > creativity)

---

## Experiment Database

Track experiments using [experiment_schema.sql](experiment_schema.sql):

**Tables**:
1. **experiments**: Experiment metadata (config, status, duration)
2. **experiment_results**: Detailed results per entity type (embedding, clustering, naming stats)
3. **cluster_details**: Individual cluster information (canonical name, members, quality)
4. **naming_comparisons**: Compare naming across different temperatures
5. **temperature_analysis**: Aggregated temperature effect analysis
6. **experiment_logs**: Debugging and auditing logs

**Views**:
- `v_experiment_summary`: Overview of all experiments
- `v_temperature_comparison`: Compare temperatures side-by-side
- `v_cluster_size_distribution`: Cluster size statistics

**Initialize Database**:
```bash
sqlite3 experiment_results.db < experiment_schema.sql
```

---

## Usage (When Orchestrator is Complete)

### Run Single Experiment
```python
from orchestrator import UnifiedPhase3Orchestrator

# Load configuration
orchestrator = UnifiedPhase3Orchestrator(
    config_path='config/experiment_configs/exp_001_nomic_hdbscan_temp00.yaml',
    db_path='intervention_research.db'
)

# Run pipeline
result = orchestrator.run()

print(f"Experiment: {result['experiment_name']}")
print(f"Duration: {result['duration_seconds']:.1f}s")
print(f"Interventions: {result['results']['interventions']['num_clusters']} clusters")
print(f"Conditions: {result['results']['conditions']['num_clusters']} clusters")
print(f"Mechanisms: {result['results']['mechanisms']['num_clusters']} clusters")
```

### Run Multiple Experiments
```python
from experiment_runner import ExperimentRunner

# Run temperature experiments (0.0, 0.2, 0.3, 0.4)
runner = ExperimentRunner(
    config_paths=[
        'config/experiment_configs/exp_001_nomic_hdbscan_temp00.yaml',
        'config/experiment_configs/exp_002_nomic_hdbscan_temp02.yaml',
        'config/experiment_configs/exp_003_nomic_hdbscan_temp03.yaml',
        'config/experiment_configs/exp_004_nomic_hdbscan_temp04.yaml',
    ],
    db_path='intervention_research.db'
)

# Run all experiments
results = runner.run_all()

# Compare results
comparison = runner.compare_experiments(
    metric='naming_quality',
    entity_type='intervention'
)

print(f"Best temperature: {comparison['best_temperature']}")
```

---

## Testing Strategy

### Phase 3a: Embedders
- Test caching behavior (hit rate)
- Validate embedding dimensions
- Test normalization methods
- Cosine similarity sanity checks (e.g., "vitamin D" similar to "Vitamin D3")

### Phase 3b: Clusterers
- Test 100% assignment guarantee (no -1 labels after SingletonHandler)
- Validate silhouette score > 0.2 (acceptable for medical domain)
- Test cluster size distribution (not all singletons, not one giant cluster)
- Compare HDBSCAN vs Hierarchical results

### Phase 3c: Namers
- Test JSON parsing at all temperatures
- Validate category assignments
- Test retry logic on LLM failures
- Compare naming consistency across temperatures

### Integration Tests
- Run on small dataset (10 interventions, 10 conditions, 20 mechanisms)
- Validate end-to-end pipeline
- Check database writes

---

## Performance Targets

### Phase 3a (Embedding)
- Cache hit rate > 90% after first run
- Embedding generation < 30s for 100 texts
- Memory efficient (< 2GB RAM for 1000 embeddings)

### Phase 3b (Clustering)
- 100% assignment guarantee (validated)
- Silhouette score > 0.2
- Runtime < 5 minutes for 1000 entities

### Phase 3c (Naming)
- 100% naming success (after retries)
- JSON parsing success rate > 95% at all temperatures
- Runtime < 10 minutes for 500 clusters

### Overall Pipeline
- Total runtime < 30 minutes for full dataset (interventions + conditions + mechanisms)
- Results quality >= current Phase 3 + 3.5 + 3.6 combined

---

## Next Steps (Implementation Roadmap)

### Week 1: ✅ COMPLETED
- [x] Experiment framework directory structure
- [x] Base configuration (YAML)
- [x] 6 experiment configs
- [x] BaseEmbedder + 3 embedder implementations
- [x] BaseClusterer + 2 clusterer implementations + SingletonHandler
- [x] BaseNamer + LLMNamer with temperature support
- [x] Experiment database schema
- [x] README documentation

### Week 2-3: TODO
- [ ] Implement UnifiedPhase3Orchestrator (main pipeline coordinator)
- [ ] Implement ExperimentRunner (multi-experiment execution)
- [ ] Implement evaluation.py (result comparison, metrics)
- [ ] Write unit tests (embedders, clusterers, namers)

### Week 4: TODO
- [ ] Test on small dataset (10 interventions, 10 conditions, 20 mechanisms)
- [ ] Run temperature experiments on 50 sample clusters
- [ ] Evaluate results and select optimal temperature
- [ ] Document findings

### Week 5-6: TODO
- [ ] Integration with main pipeline ([batch_medical_rotation.py](../../orchestration/batch_medical_rotation.py))
- [ ] Backward compatibility (keep old Phase 3.5 tables populated)
- [ ] Add `--use-unified-phase3` flag for gradual rollout
- [ ] Performance optimization

---

## Contributing

This is an experimental framework. To add new features:

1. **New Embedder**: Extend `BaseEmbedder`, implement `_generate_embedding_batch()`
2. **New Clusterer**: Extend `BaseClusterer`, implement `_perform_clustering()`
3. **New Namer**: Extend `BaseNamer`, implement `_generate_names_batch()`
4. **New Experiment**: Create YAML config in [config/experiment_configs/](config/experiment_configs/)

---

## References

- **Current Phase 3**: [rotation_semantic_normalizer.py](../../orchestration/rotation_semantic_normalizer.py)
- **Current Phase 3.5**: [rotation_group_categorization.py](../../orchestration/rotation_group_categorization.py)
- **Current Phase 3.6**: [rotation_mechanism_clustering.py](../../orchestration/rotation_mechanism_clustering.py)
- **Main Pipeline**: [batch_medical_rotation.py](../../orchestration/batch_medical_rotation.py)
- **Database Schema**: [mechanism_schema.sql](../../semantic_normalization/mechanism_schema.sql)

---

**Last Updated**: January 2025
**Status**: Components implemented, orchestrator pending
**Contact**: MyBiome Research Team
