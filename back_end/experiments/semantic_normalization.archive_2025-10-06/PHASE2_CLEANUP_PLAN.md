# Phase 2: Code Cleanup & Deprecation Plan

**Status**: In Progress
**Date**: October 6, 2025
**Objective**: Migrate experimental code to main codebase and remove deprecated files

---

## Analysis Summary

### Current State

**Experiment Directory**: `back_end/experiments/semantic_normalization/`
- Contains 8 core modules (embedding, LLM, hierarchy, etc.)
- Contains ground truth labeling tools (50 existing + 450 new pairs)
- Contains test artifacts, logs, and cache files
- Contains configuration files

**Main Codebase**: `back_end/src/`
- Has simple semantic grouping in `batch_entity_processor.py`
- Uses basic LLM approach (no hierarchy, no embeddings)
- Will be superseded by new hierarchical system

---

## Cleanup Categories

### ❌ Category 1: Test Artifacts (DELETE)

**Test logs** (5 files):
- `test_output.log`
- `test_output_50.log`
- `test_run_monitored.log`
- `test_run_full_469.log`
- `test_run_full_542.log`

**Reason**: One-time test runs, no longer needed

---

### ❌ Category 2: One-Time Scripts (DELETE)

**Scripts** (2 files):
- `apply_corrections.py` - One-time correction script
- `run_phase1.py` - Phase 1 test runner (superseded by label_in_batches.py)

**Reason**: One-time use, completed

---

### ❌ Category 3: Temporary Files (DELETE)

**Files**:
- `nul` - Empty file
- `START_LABELING.bat` - Windows batch file (can be recreated if needed)
- `__pycache__/` directories (all Python cache files)

**Reason**: Not needed in production

---

### ❌ Category 4: Old Documentation (DELETE after migration)

**Files** (keep until Phase 2 complete):
- `README.md` - Old experiment README
- `PHASE1_COMPLETE.md` - Phase 1 checkpoint
- `HIERARCHICAL_GUIDE.md` - Implementation guide
- `HIERARCHICAL_IMPLEMENTATION_COMPLETE.md` - Completion report
- `HIERARCHICAL_SCHEMA.sql` - SQL schema (integrate into main schema)
- `PROGRESS_SUMMARY.md` - Old progress tracking

**Reason**: Information consolidated into main docs

---

### ❌ Category 5: Test Results (ARCHIVE, don't delete)

**Files**:
- `results/test_run_20251005_223504.json`
- `results/test_run_20251006_114027.json`
- `results/test_run_20251006_132440.json`

**Action**: Move to `back_end/data/semantic_normalization_results/` for reference

**Reason**: Validation data, may be useful for comparison

---

### ✅ Category 6: Core Modules (MIGRATE)

**Files to migrate**:
1. `embedding_engine.py` → `back_end/src/semantic_normalization/embedding_engine.py`
2. `llm_classifier.py` → `back_end/src/semantic_normalization/llm_classifier.py`
3. `hierarchy_manager.py` → `back_end/src/semantic_normalization/hierarchy_manager.py`
4. `main_normalizer.py` → `back_end/src/semantic_normalization/normalizer.py` (rename)
5. `evaluator.py` → `back_end/src/semantic_normalization/evaluator.py`
6. `test_runner.py` → `back_end/src/semantic_normalization/test_runner.py`
7. `cluster_reviewer.py` → `back_end/src/semantic_normalization/cluster_reviewer.py`
8. `experiment_logger.py` → `back_end/src/semantic_normalization/experiment_logger.py`

**Reason**: Production-ready code

---

### ✅ Category 7: Ground Truth Tools (MIGRATE)

**Files to migrate**:
1. `core/labeling_interface.py` → `back_end/src/semantic_normalization/ground_truth/labeling_interface.py`
2. `core/pair_generator.py` → `back_end/src/semantic_normalization/ground_truth/pair_generator.py`
3. `core/data_exporter.py` → `back_end/src/semantic_normalization/ground_truth/data_exporter.py`
4. `core/prompts.py` → `back_end/src/semantic_normalization/ground_truth/prompts.py`
5. `label_in_batches.py` → `back_end/src/semantic_normalization/ground_truth/label_in_batches.py`
6. `generate_500_candidates.py` → `back_end/src/semantic_normalization/ground_truth/generate_candidates.py`

**Reason**: Ground truth management tools

---

### ✅ Category 8: Ground Truth Data (MIGRATE)

**Files to migrate**:
- `data/ground_truth/labeling_session_hierarchical_ground_truth_20251005_184757.json` (50 labels)
- `data/ground_truth/labeling_session_hierarchical_candidates_500_20251006_164058.json` (500 candidates)
- `data/ground_truth/candidate_pairs_20251005_173745.json` (old candidates - archive)

**New location**: `back_end/src/semantic_normalization/ground_truth/data/`

---

### ✅ Category 9: Configuration (MERGE)

**Files**:
- `config/config.yaml` → Merge into `back_end/src/data/config.py`
- `config/config_phase2.yaml` → Merge into `back_end/src/data/config.py`

**Strategy**: Add semantic normalization config section to main config

---

### ✅ Category 10: Cache Files (MIGRATE)

**Files**:
- `data/cache/embeddings.pkl`
- `data/cache/llm_decisions.pkl`
- `data/cache/canonicals.pkl`

**New location**: `back_end/data/semantic_normalization_cache/`

**Reason**: Performance cache, should persist

---

### ✅ Category 11: Database (MIGRATE)

**File**:
- `data/hierarchical_normalization.db`

**Action**: Merge schema into main `intervention_research.db`

**Tables to add**:
- `semantic_hierarchy` (4-layer hierarchy)
- `entity_relationships` (6 relationship types)
- `canonical_groups` (canonical entity groupings)

---

### ⚠️ Category 12: Old Semantic Grouping (DEPRECATE)

**File**: `back_end/src/llm_processing/batch_entity_processor.py`

**Current approach**:
- Simple LLM-based grouping
- No hierarchy
- No embeddings
- No relationship types

**New approach** (from experiment):
- Hierarchical 4-layer system
- Embedding-based similarity
- 6 relationship types (EXACT_MATCH, VARIANT, SUBTYPE, etc.)
- Ground truth validation

**Action**:
1. Keep old code for now (backward compatibility)
2. Add new system as separate module
3. Add flag to choose between old/new approach
4. Deprecate old approach in Phase 3 (after testing)

---

## New Directory Structure

```
back_end/src/semantic_normalization/
├── __init__.py
├── embedding_engine.py
├── llm_classifier.py
├── hierarchy_manager.py
├── normalizer.py (renamed from main_normalizer.py)
├── evaluator.py
├── test_runner.py
├── cluster_reviewer.py
├── experiment_logger.py
├── config.py (semantic normalization specific config)
│
├── ground_truth/
│   ├── __init__.py
│   ├── labeling_interface.py
│   ├── pair_generator.py
│   ├── data_exporter.py
│   ├── prompts.py
│   ├── label_in_batches.py
│   ├── generate_candidates.py
│   │
│   └── data/
│       ├── hierarchical_ground_truth_50_pairs.json (original)
│       ├── hierarchical_candidates_500_pairs.json (new)
│       └── README.md (explains ground truth data)
│
└── README.md (main semantic normalization documentation)
```

**Archive location**: `back_end/data/archives/semantic_normalization_experiment/`
- Test results
- Old documentation
- Backup of experiment folder

**Cache location**: `back_end/data/semantic_normalization_cache/`
- embeddings.pkl
- llm_decisions.pkl
- canonicals.pkl

---

## Migration Checklist

### Phase 2.1: Preparation
- [x] Analyze deprecated code
- [ ] Create new directory structure
- [ ] Create archive directory
- [ ] Backup experiment folder

### Phase 2.2: Migrate Core Modules
- [ ] Copy 8 core modules to `semantic_normalization/`
- [ ] Update imports in migrated files
- [ ] Create __init__.py with proper exports

### Phase 2.3: Migrate Ground Truth Tools
- [ ] Copy 6 ground truth files to `semantic_normalization/ground_truth/`
- [ ] Update imports in migrated files
- [ ] Create ground_truth/__init__.py

### Phase 2.4: Migrate Data & Cache
- [ ] Move cache files to new location
- [ ] Move ground truth data to new location
- [ ] Update paths in migrated code

### Phase 2.5: Configuration Consolidation
- [ ] Extract config from config.yaml
- [ ] Create semantic_normalization/config.py
- [ ] Add config section to main config.py

### Phase 2.6: Database Schema
- [ ] Extract schema from hierarchical_normalization.db
- [ ] Document schema additions needed
- [ ] Create migration script (optional)

### Phase 2.7: Cleanup
- [ ] Delete test logs
- [ ] Delete one-time scripts
- [ ] Delete __pycache__ directories
- [ ] Delete nul and temp files
- [ ] Archive test results
- [ ] Archive old documentation

### Phase 2.8: Verification
- [ ] Test imports from new location
- [ ] Run test_runner.py from new location
- [ ] Verify cache loading works
- [ ] Verify ground truth tools work

### Phase 2.9: Final Cleanup
- [ ] Delete experiment folder
- [ ] Update claude.md with new paths
- [ ] Create migration notes

---

## Import Changes

**Old imports**:
```python
from back_end.experiments.semantic_normalization.embedding_engine import EmbeddingEngine
from back_end.experiments.semantic_normalization.llm_classifier import LLMClassifier
```

**New imports**:
```python
from back_end.src.semantic_normalization.embedding_engine import EmbeddingEngine
from back_end.src.semantic_normalization.llm_classifier import LLMClassifier
```

---

## Risk Assessment

**Low Risk**:
- Deleting test logs
- Deleting one-time scripts
- Deleting __pycache__

**Medium Risk**:
- Moving cache files (paths need updating)
- Merging configuration (need to preserve all settings)

**High Risk**:
- Deleting experiment folder (ensure everything migrated)
- Deprecating old semantic grouping (may break existing code)

---

## Rollback Plan

If migration fails:
1. Experiment folder backed up to archive
2. Git commit before deletion
3. Can restore from archive
4. Can revert git commit

---

**Next Steps**: Execute migration in phases with verification at each step
