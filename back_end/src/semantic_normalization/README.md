# Semantic Normalization Module

**Hierarchical semantic normalization system for intervention names**

Uses embedding-based similarity and LLM-based classification to group intervention names into a 4-layer hierarchy with 6 relationship types.

---

## Overview

**Problem**: Intervention names vary across papers
- "vitamin D" vs "Vitamin D3" vs "cholecalciferol" (same thing)
- "atorvastatin" vs "simvastatin" (different statins, same class)
- "metformin" vs "metformin 500mg" (dosage variant)

**Solution**: Hierarchical semantic normalization
- **Layer 0**: Category (supplement, medication, etc.)
- **Layer 1**: Canonical Entity ("vitamin D", "statins", "metformin")
- **Layer 2**: Specific Variant ("cholecalciferol", "atorvastatin", "metformin")
- **Layer 3**: Dosage/Details ("vitamin D 1000 IU", "atorvastatin 20mg", "metformin 500mg")

---

## Architecture

### Components

1. **embedding_engine.py** - Semantic embeddings (nomic-embed-text via Ollama)
2. **llm_classifier.py** - Canonical extraction & relationship classification (qwen3:14b)
3. **hierarchy_manager.py** - Database operations for hierarchical schema
4. **normalizer.py** - Main normalization pipeline
5. **evaluator.py** - Ground truth accuracy testing
6. **test_runner.py** - Batch testing framework
7. **cluster_reviewer.py** - Interactive manual review
8. **experiment_logger.py** - Experiment documentation

### Ground Truth Tools

9. **ground_truth/labeling_interface.py** - Interactive labeling interface
10. **ground_truth/pair_generator.py** - Candidate pair generation (stratified sampling)
11. **ground_truth/label_in_batches.py** - Batch labeling session management
12. **ground_truth/generate_candidates.py** - 500-pair candidate generator
13. **ground_truth/data_exporter.py** - Export interventions from database

---

## Relationship Types

| Type | Code | Description | Examples |
|------|------|-------------|----------|
| 1 | `EXACT_MATCH` | Same intervention, synonyms | "vitamin D" = "cholecalciferol" |
| 2 | `VARIANT` | Same concept, different formulation | "Cetuximab" vs "Cetuximab-β" (biosimilar) |
| 3 | `SUBTYPE` | Related but clinically distinct | "IBS-D" vs "IBS-C" |
| 4 | `SAME_CATEGORY` | Different entities in same class | "atorvastatin" vs "simvastatin" (statins) |
| 5 | `DOSAGE_VARIANT` | Same intervention, different dose | "metformin" vs "metformin 500mg" |
| 6 | `DIFFERENT` | Completely unrelated | "vitamin D" vs "chemotherapy" |

---

## Usage

### 1. Generate Ground Truth Candidates

```python
from back_end.src.semantic_normalization.ground_truth.generate_candidates import main

# Generate 500 candidate pairs using stratified sampling
main()
```

**Output**: `ground_truth/data/hierarchical_candidates_500_pairs.json`

---

### 2. Label Candidate Pairs

```bash
cd back_end/src/semantic_normalization/ground_truth

# Start labeling first batch (50 pairs)
python label_in_batches.py --batch-size 50 --start-from 0

# Check progress
python label_in_batches.py --status

# Continue with next batch
python label_in_batches.py --batch-size 50 --start-from 50
```

**Features**:
- Progress bar and time estimation
- Undo last label (u key)
- Mark for review later (r key)
- Auto-save every 10 labels
- Resume capability

---

### 3. Run Semantic Normalization

```python
from back_end.src.semantic_normalization.normalizer import SemanticNormalizer
from back_end.src.semantic_normalization.config import DB_PATH

# Initialize normalizer
normalizer = SemanticNormalizer(db_path=str(DB_PATH))

# Run normalization on all interventions
results = normalizer.normalize_all_interventions()

print(f"Processed {results['total_interventions']} interventions")
print(f"Created {results['canonical_groups']} canonical groups")
print(f"Clustering rate: {results['clustering_rate']:.1%}")
```

---

### 4. Evaluate Accuracy

```python
from back_end.src.semantic_normalization.evaluator import Evaluator
from back_end.src.semantic_normalization.config import GROUND_TRUTH_DATA_DIR

# Load ground truth
evaluator = Evaluator(
    ground_truth_file=str(GROUND_TRUTH_DATA_DIR / "hierarchical_ground_truth_50_pairs.json")
)

# Evaluate accuracy
results = evaluator.evaluate()

print(f"Accuracy: {results['accuracy']:.1%}")
print(f"Precision: {results['precision']:.1%}")
print(f"Recall: {results['recall']:.1%}")
```

---

### 5. Interactive Cluster Review

```bash
cd back_end/src/semantic_normalization

# Review clustering results
python cluster_reviewer.py results/test_run_YYYYMMDD_HHMMSS.json
```

**Features**:
- Review each cluster interactively
- Mark as correct/incorrect
- Identify false positives/negatives
- Export corrections

---

## Configuration

**File**: [`config.py`](config.py)

```python
from back_end.src.semantic_normalization.config import (
    CACHE_DIR,          # Cache for embeddings & LLM decisions
    RESULTS_DIR,        # Test results and outputs
    GROUND_TRUTH_DIR,   # Ground truth data
    EMBEDDING_CONFIG,   # nomic-embed-text settings
    LLM_CONFIG,         # qwen3:14b settings
    HIERARCHY_CONFIG,   # 4-layer hierarchy settings
)
```

---

## Ground Truth Data

### Existing Datasets

1. **50 labeled pairs** (completed October 2025)
   - File: `ground_truth/data/hierarchical_ground_truth_50_pairs.json`
   - Manually labeled with 6 relationship types
   - Includes hierarchical layer annotations

2. **500 candidate pairs** (ready for labeling)
   - File: `ground_truth/data/hierarchical_candidates_500_pairs.json`
   - Stratified sampling:
     - 60% similarity-based (300 pairs)
     - 20% random low-similarity (100 pairs)
     - 20% targeted same-category (100 pairs)

### Labeling Progress

Check current labeling progress:
```bash
python back_end/src/semantic_normalization/ground_truth/label_in_batches.py --status
```

---

## Performance

**Test Results** (October 2025, 542 interventions):
- **Runtime**: ~1.5 hours with qwen3:14b
- **Cache hit rate**: 40.6% (220/542 cached)
- **Unique canonical groups**: 486 (from 542 interventions)
- **Clustering rate**: 10.3% (56 interventions merged into 44 groups)
- **Average cluster size**: 1.12

**Top Semantic Groups**:
1. Proton pump inhibitors (6 members)
2. Probiotics (4 members)
3. JAK inhibitors (3 members)
4. Robot-assisted gait training (3 members)
5. Metformin, Statins, Corticosteroids, Vitamin D (3 members each)

---

## Database Schema

**Tables** (in `intervention_research.db`):

1. **semantic_hierarchy** - 4-layer hierarchical intervention structure
2. **entity_relationships** - Pairwise relationship types (6 types)
3. **canonical_groups** - Canonical entity groupings

See [`HIERARCHICAL_SCHEMA.sql`](../../../experiments/semantic_normalization/HIERARCHICAL_SCHEMA.sql) for schema definition.

---

## Dependencies

### Required
- **Python 3.13+**
- **Ollama** (local LLM inference)
  - nomic-embed-text (embeddings)
  - qwen3:14b (classification)
- **rapidfuzz** (fuzzy string matching)
- **numpy** (vector operations)
- **PyYAML** (configuration)

### Installation

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull nomic-embed-text
ollama pull qwen3:14b

# Install Python dependencies
pip install rapidfuzz numpy pyyaml
```

---

## Workflow

### Phase 1: Ground Truth Creation ✅ COMPLETE
1. Generate 500 candidate pairs (stratified sampling)
2. Label pairs interactively (6 relationship types, 4-layer hierarchy)
3. Validate ground truth dataset

### Phase 2: Integration ⏳ IN PROGRESS
1. Migrate code from experiments/ to src/
2. Consolidate configuration
3. Update imports and paths
4. Clean up deprecated code

### Phase 3: Production Deployment
1. Integrate with main pipeline
2. Add to batch_medical_rotation.py
3. Replace old semantic grouping in batch_entity_processor.py
4. Performance benchmarking

---

## Comparison: Old vs New

| Feature | Old (batch_entity_processor.py) | New (semantic_normalization/) |
|---------|--------------------------------|-------------------------------|
| **Approach** | Simple LLM grouping | Hierarchical 4-layer system |
| **Embeddings** | ❌ None | ✅ nomic-embed-text (768-dim) |
| **Relationship Types** | ❌ Binary (same/different) | ✅ 6 types (EXACT_MATCH, VARIANT, etc.) |
| **Hierarchy** | ❌ Flat | ✅ 4 layers (Category → Canonical → Variant → Dosage) |
| **Ground Truth** | ❌ None | ✅ 500 labeled pairs (50 existing + 450 new) |
| **Evaluation** | ❌ No metrics | ✅ Accuracy, Precision, Recall |
| **Caching** | ❌ None | ✅ Embeddings + LLM decisions |
| **Interactive Review** | ❌ None | ✅ Cluster reviewer tool |

---

## Migration Notes

**Original Location**: `back_end/experiments/semantic_normalization/`
**New Location**: `back_end/src/semantic_normalization/`

**Files Migrated**:
- ✅ 8 core modules
- ✅ 6 ground truth tools
- ✅ Ground truth data (50 + 500 pairs)
- ✅ Cache files (embeddings, LLM decisions, canonicals)
- ✅ Configuration (merged into config.py)

**Files Archived**:
- Test logs (*.log)
- Test results (results/*.json)
- Old documentation (*.md from experiments/)

**Files Deleted**:
- One-time scripts (apply_corrections.py, run_phase1.py)
- Temporary files (nul, __pycache__)
- Old batch files (START_LABELING.bat)

---

## Documentation

- **[Phase 1 Implementation](../../experiments/semantic_normalization/PHASE1_IMPLEMENTATION.md)** - Ground truth expansion details
- **[Quick Start Guide](../../experiments/semantic_normalization/QUICK_START_GUIDE.md)** - Labeling walkthrough
- **[Phase 2 Cleanup Plan](../../experiments/semantic_normalization/PHASE2_CLEANUP_PLAN.md)** - Migration strategy

---

## Contact

**Module**: Semantic Normalization
**Author**: MyBiome Research Team
**Version**: 1.0.0
**Last Updated**: October 6, 2025
