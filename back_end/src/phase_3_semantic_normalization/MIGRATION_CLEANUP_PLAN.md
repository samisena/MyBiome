# Phase 3 Migration Cleanup Plan

## Overview

This document lists files to DELETE after successful migration from naming-first to clustering-first architecture.

**IMPORTANT**: Only delete these files AFTER confirming the new pipeline works correctly.

## Backup Location

All old files are backed up at:
```
back_end/src/phase_3_semantic_normalization_OLD_BACKUP_20251015/
```

---

## Files to DELETE (Old Naming-First Architecture)

### Core Old Pipeline (5 files)
- `phase_3_embedding_engine.py` - OLD embedding engine (replaced by phase_3a_*.py)
- `phase_3_llm_classifier.py` - OLD LLM classifier (replaced by phase_3c_llm_namer.py)
- `phase_3_hierarchy_manager.py` - OLD hierarchy manager (replaced by phase_3c database operations)
- `phase_3_normalizer.py` - OLD main normalizer (replaced by phase_3abc_orchestrator.py)
- `semantic_normalizer.py` - OLD semantic normalizer wrapper

### Legacy Config (2 items)
- `config.py` - OLD config (replaced by phase_3_config.yaml)
- `config/` - OLD config directory

### Mechanism-Specific Old Files (9 files)
- `mechanism_baseline_test.py` - OLD mechanism testing
- `mechanism_canonical_extractor.py` - OLD mechanism extraction
- `mechanism_db_manager.py` - OLD mechanism DB operations
- `mechanism_experiments.py` - OLD mechanism experiments
- `mechanism_normalizer.py` - OLD mechanism normalizer
- `mechanism_preprocessing_comparison.py` - OLD preprocessing comparison
- `mechanism_preprocessor.py` - OLD preprocessor
- `mechanism_schema.sql` - OLD schema
- `prompts.py` - OLD prompt definitions

### Old Testing/Experiment Files (4 files)
- `complete_assignment_test.py` - OLD assignment testing
- `hyperparameter_experiment.py` - OLD hyperparameter experiments
- `real_embedding_test.py` - OLD embedding testing
- `cluster_reviewer.py` - OLD manual review tool

### Validation Files (Keep or Delete - TBD)
- `evaluator.py` - Ground truth evaluator (MAY KEEP for accuracy testing)
- `validation.py` - Categorization validation (MAY KEEP for validation)

### New Testing Files (KEEP - created during migration)
- `test_imports.py` - NEW import test (KEEP)
- `test_orchestrator.py` - NEW orchestrator test (KEEP)

---

## Files to KEEP (New Clustering-First Architecture)

### Phase 3a: Embedders (4 files)
- `phase_3a_base_embedder.py`
- `phase_3a_intervention_embedder.py`
- `phase_3a_condition_embedder.py`
- `phase_3a_mechanism_embedder.py`

### Phase 3b: Clusterers (4 files)
- `phase_3b_base_clusterer.py`
- `phase_3b_hierarchical_clusterer.py`
- `phase_3b_hdbscan_clusterer.py`
- `phase_3b_singleton_handler.py`

### Phase 3c: Namers (2 files)
- `phase_3c_base_namer.py`
- `phase_3c_llm_namer.py`

### Phase 3c: Old Categorizers (2 files - used by old pipeline)
**NOTE**: These are part of the OLD Phase 3b (group categorization), not the NEW clustering-first Phase 3b
- `phase_3b_condition_categorizer.py` - OLD condition categorization
- `phase_3b_intervention_categorizer.py` - OLD intervention categorization

**Decision**: DELETE after confirming new pipeline handles categorization

### Phase 3c: Old Mechanism Clustering (2 files)
**NOTE**: These are part of the OLD Phase 3c (mechanism clustering), not the NEW clustering-first Phase 3c
- `phase_3c_mechanism_hierarchical_clustering.py` - OLD mechanism clustering
- `phase_3c_mechanism_pipeline_orchestrator.py` - OLD mechanism orchestrator

**Decision**: DELETE after confirming new pipeline handles mechanism clustering

### Core Orchestrator (2 files)
- `phase_3abc_orchestrator.py` - NEW unified orchestrator
- `phase_3_config.yaml` - NEW configuration

### Phase 3d: Hierarchical Merging (Keep all)
- `phase_3d/` - Entire directory (KEEP - experimental feature)

### Supporting Directories (Keep)
- `__init__.py` - Package initialization
- `ground_truth/` - Ground truth labeling tools
- `research/` - Research and experiment files
- `README.md` - Documentation

---

## Cleanup Commands

### Step 1: Verify Backup Exists
```bash
ls "C:\Users\samis\Desktop\MyBiome\back_end\src\phase_3_semantic_normalization_OLD_BACKUP_20251015"
```

### Step 2: Delete Old Files (after testing confirms success)

```bash
cd "C:\Users\samis\Desktop\MyBiome\back_end\src\phase_3_semantic_normalization"

# Delete core old pipeline
rm phase_3_embedding_engine.py
rm phase_3_llm_classifier.py
rm phase_3_hierarchy_manager.py
rm phase_3_normalizer.py
rm semantic_normalizer.py

# Delete legacy config
rm config.py
rm -r config/

# Delete mechanism-specific old files
rm mechanism_baseline_test.py
rm mechanism_canonical_extractor.py
rm mechanism_db_manager.py
rm mechanism_experiments.py
rm mechanism_normalizer.py
rm mechanism_preprocessing_comparison.py
rm mechanism_preprocessor.py
rm mechanism_schema.sql
rm prompts.py

# Delete old testing files
rm complete_assignment_test.py
rm hyperparameter_experiment.py
rm real_embedding_test.py
rm cluster_reviewer.py

# Delete old Phase 3b/3c files (conflicting names)
rm phase_3b_condition_categorizer.py
rm phase_3b_intervention_categorizer.py
rm phase_3c_mechanism_hierarchical_clustering.py
rm phase_3c_mechanism_pipeline_orchestrator.py

# Optional: Delete validation files (if not needed)
# rm evaluator.py
# rm validation.py
```

### Step 3: Delete Experimentation Folder

```bash
cd "C:\Users\samis\Desktop\MyBiome\back_end\src"
rm -r experimentation/
```

---

## Verification

After deletion, the phase_3_semantic_normalization folder should contain:

**Phase 3a/3b/3c Files**: 12 files (4 embedders + 4 clusterers + 2 namers + orchestrator + config)
**Phase 3d**: 1 directory
**Support**: ground_truth/, research/, __init__.py, README.md, test files
**Optional**: evaluator.py, validation.py (if kept)

---

## Rollback Plan

If something goes wrong, restore from backup:
```bash
cd "C:\Users\samis\Desktop\MyBiome\back_end\src"
rm -r phase_3_semantic_normalization/
cp -r phase_3_semantic_normalization_OLD_BACKUP_20251015/ phase_3_semantic_normalization/
```

---

**Created**: October 15, 2025
**Migration**: Naming-first → Clustering-first architecture
**Backup**: phase_3_semantic_normalization_OLD_BACKUP_20251015/
