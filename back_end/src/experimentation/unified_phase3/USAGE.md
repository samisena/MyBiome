# Unified Phase 3 Pipeline - Usage Guide

Complete guide for running experiments, testing, and evaluating results.

---

## Quick Start

### 1. Test on Small Dataset

First, validate that everything works with a small subset of data:

```bash
# Navigate to experimentation directory
cd back_end/src/experimentation/unified_phase3

# Run test on small dataset (10 interventions, 10 conditions, 20 mechanisms)
python test_small_dataset.py \
    --source-db "../../../data/processed/intervention_research.db" \
    --test-db "../../../data/test_intervention_research.db" \
    --config "config/experiment_configs/exp_001_nomic_hdbscan_temp00.yaml"
```

**Expected Output**:
```
[SUCCESS] ALL VALIDATIONS PASSED
  - 100% assignment rate for all entity types
  - Clusters created successfully
  - Naming mostly successful
  - Embeddings generated
```

---

## Running Single Experiment

### Basic Usage

```bash
python -c "
from orchestrator import UnifiedPhase3Orchestrator

orchestrator = UnifiedPhase3Orchestrator(
    config_path='config/experiment_configs/exp_001_nomic_hdbscan_temp00.yaml',
    db_path='../../../data/processed/intervention_research.db'
)

result = orchestrator.run()

if result['success']:
    print(f'Experiment completed in {result[\"duration_seconds\"]:.1f}s')
    print(f'Experiment ID: {result[\"experiment_id\"]}')
"
```

### Python Script

Create `run_single_experiment.py`:

```python
from experimentation.unified_phase3.orchestrator import UnifiedPhase3Orchestrator
import sys

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/experiment_configs/exp_001_nomic_hdbscan_temp00.yaml"
    db_path = "../../../data/processed/intervention_research.db"

    orchestrator = UnifiedPhase3Orchestrator(
        config_path=config_path,
        db_path=db_path
    )

    result = orchestrator.run()

    if result['success']:
        print(f"\n[SUCCESS] Experiment: {result['experiment_name']}")
        print(f"Duration: {result['duration_seconds']:.1f}s")
        print(f"Experiment ID: {result['experiment_id']}")

        # Print results for each entity type
        for entity_type in ['interventions', 'conditions', 'mechanisms']:
            entity_results = result['results'].get(entity_type)
            if entity_results:
                print(f"\n{entity_type.upper()}:")
                print(f"  Entities: {len(entity_results.entity_names)}")
                print(f"  Clusters: {entity_results.num_clusters}")
                print(f"  Singletons: {entity_results.num_singleton_clusters}")
                print(f"  Silhouette: {entity_results.silhouette_score:.3f}" if entity_results.silhouette_score else "  Silhouette: N/A")
    else:
        print(f"\n[FAILED] Error: {result.get('error')}")
        sys.exit(1)
```

Run:
```bash
python run_single_experiment.py config/experiment_configs/exp_001_nomic_hdbscan_temp00.yaml
```

---

## Running Temperature Experiments

### Run All 4 Temperatures

```bash
python experiment_runner.py \
    --db "../../../data/processed/intervention_research.db" \
    --configs \
        config/experiment_configs/exp_001_nomic_hdbscan_temp00.yaml \
        config/experiment_configs/exp_002_nomic_hdbscan_temp02.yaml \
        config/experiment_configs/exp_003_nomic_hdbscan_temp03.yaml \
        config/experiment_configs/exp_004_nomic_hdbscan_temp04.yaml \
    --exp-db "experiment_results.db" \
    --cache-dir "../../../data/semantic_normalization_cache" \
    --report "temperature_comparison_report.json"
```

**What This Does**:
1. Runs 4 experiments sequentially (temp=0.0, 0.2, 0.3, 0.4)
2. Saves results to `experiment_results.db`
3. Reuses cached embeddings/clusters across experiments (only naming changes)
4. Generates comparison report as JSON

**Expected Duration**:
- First experiment: ~30 minutes (embedding + clustering + naming)
- Subsequent experiments: ~5-10 minutes each (only re-run naming with different temperature)
- **Total**: ~1 hour for all 4 experiments

---

## Evaluating Results

### Compare Temperatures

```bash
python evaluation.py \
    --exp-db "experiment_results.db" \
    --entity-type "intervention" \
    --report "evaluation_report.json"
```

**Output**:
```
Temperature Comparison (interventions):
Temp   Experiments  Clusters   Silhouette   Failures
------------------------------------------------------------
0.0    1            571        0.228        3
0.2    1            571        0.228        2
0.3    1            571        0.228        5
0.4    1            571        0.228        8

Optimal temperature: 0.2
Recommendation score: 0.856
```

### Python API

```python
from experimentation.unified_phase3.evaluation import ExperimentEvaluator

evaluator = ExperimentEvaluator("experiment_results.db")

# Compare temperatures
temp_metrics = evaluator.compare_temperatures('intervention')

for temp, metrics in temp_metrics.items():
    print(f"Temperature {temp}:")
    print(f"  Avg silhouette: {metrics.avg_silhouette:.3f}")
    print(f"  Avg failures: {metrics.avg_failures:.1f}")

# Select optimal
optimal_temp, analysis = evaluator.select_optimal_temperature('intervention')
print(f"\nRecommended temperature: {optimal_temp}")

# Analyze naming consistency
experiment_ids = [1, 2, 3, 4]  # temp 0.0, 0.2, 0.3, 0.4
consistency = evaluator.compute_naming_consistency(experiment_ids, 'intervention')
print(f"Naming consistency: {consistency['overall_consistency']:.2%}")
```

---

## Advanced Usage

### Custom Configuration

Create `config/experiment_configs/exp_custom.yaml`:

```yaml
_base: "../base_config.yaml"

experiment:
  name: "exp_custom_mxbai_temp02"
  description: "Custom: mxbai for mechanisms, temperature 0.2"
  tags:
    - custom
    - mxbai
    - temperature_0.2

# Use mxbai-embed-large for mechanisms (better for long text)
embedding:
  mechanisms:
    model: "mxbai-embed-large"
    dimension: 1024
    batch_size: 5  # Slower model

# More permissive clustering for mechanisms
clustering:
  mechanisms:
    algorithm: "hdbscan"
    min_cluster_size: 3
    min_samples: 2

# Slightly creative naming
naming:
  llm:
    temperature: 0.2
```

Run:
```bash
python run_single_experiment.py config/experiment_configs/exp_custom.yaml
```

### Query Experiment Database

```bash
# View all experiments
sqlite3 experiment_results.db "SELECT experiment_id, experiment_name, status, duration_seconds FROM experiments;"

# View temperature comparison
sqlite3 experiment_results.db "SELECT * FROM v_temperature_comparison;"

# Get cluster details for experiment ID 1
sqlite3 experiment_results.db "SELECT cluster_id, canonical_name, category, member_count FROM cluster_details WHERE experiment_id = 1 AND entity_type = 'intervention' ORDER BY member_count DESC LIMIT 10;"
```

---

## Troubleshooting

### Issue: "HDBSCAN not installed"

```bash
conda activate venv
pip install hdbscan
```

### Issue: "Ollama connection refused"

Make sure Ollama is running:
```bash
ollama serve
```

Pull required models:
```bash
ollama pull nomic-embed-text
ollama pull qwen3:14b
```

### Issue: "Out of memory"

Reduce batch sizes in config:
```yaml
embedding:
  interventions:
    batch_size: 16  # Default: 32
  mechanisms:
    batch_size: 5   # Default: 10
```

### Issue: "Naming failures"

Check LLM timeout and retries:
```yaml
naming:
  llm:
    timeout: 120    # Increase from 60
    max_retries: 5  # Increase from 3
```

### Issue: "Cache taking too much space"

Clear caches:
```bash
rm -rf back_end/data/semantic_normalization_cache/*
```

---

## Performance Optimization

### Use Cached Embeddings

First run generates embeddings (~30 minutes). Subsequent runs reuse them:

```bash
# Run 1: temp=0.0 (full pipeline, ~30 min)
python run_single_experiment.py exp_001_nomic_hdbscan_temp00.yaml

# Run 2: temp=0.2 (only naming changes, ~5 min)
python run_single_experiment.py exp_002_nomic_hdbscan_temp02.yaml

# Runs 3-4: temp=0.3, 0.4 (~5 min each)
```

### Parallel Execution (Future)

```python
# TODO: Implement in experiment_runner.py
runner.run_all(sequential=False, max_workers=4)
```

---

## Expected Results

### Small Dataset (10 interventions, 10 conditions, 20 mechanisms)

- **Duration**: ~2-3 minutes
- **Interventions**: ~8-10 clusters (mostly singletons)
- **Conditions**: ~8-10 clusters (mostly singletons)
- **Mechanisms**: ~15-20 clusters (mostly singletons)
- **Assignment rate**: 100% for all
- **Naming failures**: 0-2 per entity type

### Full Dataset (777 interventions, 406 conditions, 666 mechanisms)

- **Duration**: ~30 minutes (first run with fresh cache)
- **Interventions**: ~500-600 clusters (~200 natural + ~400 singletons)
- **Conditions**: ~300-400 clusters (~150 natural + ~250 singletons)
- **Mechanisms**: ~400-500 clusters (~150 natural + ~350 singletons)
- **Assignment rate**: 100% for all (guaranteed)
- **Silhouette score**: 0.2-0.3 (acceptable for medical domain)
- **Naming failures**: <5% per entity type

---

## Next Steps

After running temperature experiments:

1. **Analyze Results**: Use `evaluation.py` to compare temperatures
2. **Select Optimal**: Likely temperature 0.0 or 0.2 based on consistency
3. **Update Base Config**: Set optimal temperature in `base_config.yaml`
4. **Integration**: Integrate with main pipeline ([batch_medical_rotation.py](../../orchestration/batch_medical_rotation.py))
5. **Production**: Replace Phase 3, 3.5, 3.6 with unified pipeline

---

## Files Overview

| File | Purpose | Usage |
|------|---------|-------|
| `orchestrator.py` | Main pipeline coordinator | Single experiment execution |
| `experiment_runner.py` | Multi-experiment runner | Temperature experiments |
| `evaluation.py` | Result comparison | Post-experiment analysis |
| `test_small_dataset.py` | Small dataset validation | Testing before full run |
| `base_config.yaml` | Default configuration | Base settings |
| `exp_00X_*.yaml` | Experiment configs | Specific experiment settings |

---

## Contact & Support

For issues or questions:
- Check [README.md](README.md) for architecture details
- Review experiment logs in `experiment_results.db`
- Check cache directories for stale data
- Consult MyBiome team for medical domain questions

---

**Last Updated**: January 2025
**Version**: 0.1.0
