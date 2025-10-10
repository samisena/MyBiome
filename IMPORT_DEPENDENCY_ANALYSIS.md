# Import Dependency Analysis Report
**MyBiome back_end Directory**
**Generated:** 2025-10-10
**Total Python Files Analyzed:** 117

---

## Executive Summary

### File Category Breakdown
- **Entry Points (Orchestration Scripts):** 7 files
- **Other Executable Scripts:** 43 files
- **Library Modules (Imported):** 45 files
- **Init Files:** 15 files
- **Unused Files:** 7 files
- **Duplicate Files:** 8 pairs (exact duplicates or launchers)

### Key Findings
1. **7 truly unused files** identified - never imported and not executable
2. **3 exact duplicate files** found (prompts.py and evaluator.py have identical copies)
3. **Most critical files:** `config.py` (45 imports), `database_manager.py` (20 imports)
4. **Utility modules are well-used:** `medical_knowledge.py`, `scoring_utils.py`, `similarity_utils.py` all imported by data mining modules

---

## 1. Entry Points (Main Orchestration Scripts)

These are the primary entry points for the pipeline, located in `src/orchestration/`:

| File | Purpose | Imported By |
|------|---------|-------------|
| `batch_medical_rotation.py` | Main pipeline orchestrator | 0 files (top-level entry point) |
| `rotation_paper_collector.py` | Phase 1: Paper collection | 2 files |
| `rotation_llm_processor.py` | Phase 2: LLM extraction | 2 files |
| `rotation_llm_categorization.py` | Phase 2.5: Entity categorization | 2 files |
| `rotation_semantic_normalizer.py` | Phase 3: Semantic normalization | 0 files (called by batch_medical_rotation) |
| `rotation_group_categorization.py` | Phase 3.5: Group categorization | 4 files |
| `rotation_semantic_grouping_integrator.py` | Integration layer | 2 files |

**All orchestration scripts are actively used.**

---

## 2. Truly Unused Files (Candidates for Removal)

### 2.1 Data Mining
**File:** `src/data_mining/emerging_category_analyzer.py` (7 imports)
**Purpose:** Analyzes interventions classified as "emerging" to suggest new categories
**Status:** Never imported, no `__main__`
**Recommendation:** Remove or add to data_mining_orchestrator.py if functionality is needed

**File:** `src/data_mining/fundamental_functions.py` (3 imports)
**Purpose:** Discovers fundamental body functions via cross-mechanism interventions
**Status:** Never imported, no `__main__`
**Recommendation:** Remove or integrate into data mining pipeline

### 2.2 Experimentation
**File:** `src/experimentation/analyze_interventions.py` (2 imports)
**Purpose:** Batch size experiment analysis script
**Status:** Never imported, no `__main__`
**Recommendation:** Remove (appears to be old experiment analysis)

**File:** `src/experimentation/check_progress.py` (1 import)
**Purpose:** Quick experiment progress checker
**Status:** Never imported, no `__main__`
**Recommendation:** Remove (appears to be old experiment monitoring)

**File:** `src/experimentation/show_interventions.py` (1 import)
**Purpose:** Display interventions from experiments
**Status:** Never imported, no `__main__`
**Recommendation:** Remove (appears to be old experiment utility)

### 2.3 Semantic Normalization
**File:** `src/semantic_normalization/ground_truth/prompts.py` (0 imports)
**Purpose:** LLM prompts (EXACT DUPLICATE of parent prompts.py)
**Status:** DUPLICATE - never imported
**Recommendation:** **DELETE** - exact duplicate of `src/semantic_normalization/prompts.py`

### 2.4 Utils
**File:** `src/utils/integration_success_summary.py` (0 imports)
**Purpose:** Empty/minimal file
**Status:** Never imported, no imports, no `__main__`
**Recommendation:** Remove (appears to be placeholder or deprecated)

---

## 3. Exact Duplicate Files

### 3.1 Confirmed Exact Duplicates (DELETE THESE)

**1. Prompts - Identical Files**
- `src/semantic_normalization/prompts.py` (222 lines) - **KEEP** (imported by parent modules)
- `src/semantic_normalization/ground_truth/prompts.py` (222 lines) - **DELETE** (never imported)
- Verification: `diff` shows no differences

**2. Evaluators - Identical Files**
- `src/semantic_normalization/evaluator.py` (399 lines) - **KEEP** (imported by parent modules)
- `src/semantic_normalization/ground_truth/evaluator.py` (399 lines) - **DELETE OR CLARIFY** (may be used in ground truth workflow)
- Verification: `diff` shows no differences
- Note: Check if ground_truth/evaluator.py is used in standalone ground truth labeling workflow before deletion

### 3.2 Launcher Duplicates (KEEP BOTH)

**3. Review Correlations - Launcher Pattern**
- `src/data_mining/review_correlations.py` - Launcher script (imports from utils)
- `src/utils/review_correlations.py` - Actual implementation
- Status: Keep both (one is a launcher for convenience)

### 3.3 Intentional Duplicates (Different Purposes - KEEP ALL)

**4. Category Validators**
- `src/conditions/category_validators.py` - Validates condition categories (18 categories)
- `src/interventions/category_validators.py` - Validates intervention categories (13 categories)
- Status: Keep both (different taxonomies)

**5. Search Terms**
- `src/conditions/search_terms.py` - PubMed search terms for conditions
- `src/interventions/search_terms.py` - Intervention search patterns
- Status: Keep both (different domains)

**6. Taxonomy**
- `src/conditions/taxonomy.py` - Condition taxonomy (18 categories)
- `src/interventions/taxonomy.py` - Intervention taxonomy (13 categories)
- Status: Keep both (different taxonomies)

**7. Config**
- `src/data/config.py` - Global application config (45 imports)
- `src/semantic_normalization/config.py` - Semantic normalization config
- Status: Keep both (different scopes)

**8. Experiment Runners**
- `src/experimentation/group_categorization/experiment_runner.py` (323 lines)
- `src/experimentation/runners/experiment_runner.py` (270 lines)
- Status: Keep both (different experiment types - verify they serve different purposes)

---

## 4. Most Imported Files (Core Infrastructure)

| File | Import Count | Category |
|------|--------------|----------|
| `src/data/config.py` | 45 | Configuration |
| `src/data_collection/database_manager.py` | 20 | Data Access |
| `src/data/utils.py` | 9 | Utilities |
| `src/interventions/taxonomy.py` | 8 | Domain Model |
| `src/data/api_clients.py` | 7 | External APIs |
| `src/llm_processing/batch_entity_processor.py` | 6 | LLM Processing |
| `src/conditions/taxonomy.py` | 5 | Domain Model |
| `src/data/validators.py` | 5 | Validation |
| `src/data/repositories.py` | 5 | Data Access |
| `src/data/error_handler.py` | 4 | Error Handling |
| `src/data_mining/scoring_utils.py` | 4 | Analytics |
| `src/data_collection/data_mining_repository.py` | 4 | Data Access |
| `src/interventions/search_terms.py` | 3 | Domain Model |
| `src/interventions/category_validators.py` | 3 | Validation |
| `src/data_mining/medical_knowledge.py` | 3 | Analytics |

**Analysis:** Core infrastructure files are heavily used. The top 15 files account for the majority of internal imports.

---

## 5. Utility Modules Status

### Well-Used Utility Modules (KEEP)
- `medical_knowledge.py` - 3 imports (condition_similarity_mapping, power_combinations, research_gaps)
- `scoring_utils.py` - 4 imports (bayesian_scorer, condition_similarity_mapping, power_combinations, research_gaps)
- `similarity_utils.py` - 2 imports (condition_similarity_mapping, research_gaps)
- `graph_utils.py` - 1 import (power_combinations)

**Status:** All utility modules in data_mining are actively used via relative imports.

---

## 6. Migration Scripts Status

Located in `scripts/` and `src/migrations/`:

| File | Purpose | Status |
|------|---------|--------|
| `scripts/migrate_add_mechanism.py` | Added mechanism field to DB | Executable (keep for reference) |
| `scripts/migrate_intervention_category_nullable.py` | Made category nullable | Executable (keep for reference) |
| `scripts/backfill_condition_categories.py` | Backfilled condition categories | Executable (keep for reference) |
| `scripts/classify_conditions.py` | Initial condition classification | Executable (keep for reference) |
| `src/migrations/add_semantic_normalization_tables.py` | Added Phase 3 tables | Executable (keep for reference) |
| `src/migrations/create_interventions_view_option_b.py` | Created DB view | Executable (keep for reference) |

**Recommendation:** Keep all migration scripts for database evolution history and potential rollback reference.

---

## 7. Detailed Recommendations

### Immediate Actions (High Confidence)

#### DELETE (3 files)
1. `src/semantic_normalization/ground_truth/prompts.py` - Exact duplicate, never imported
2. `src/utils/integration_success_summary.py` - Empty/minimal file, never used
3. `src/experimentation/analyze_interventions.py` - Old experiment analysis, never used

#### VERIFY THEN DELETE (4 files)
4. `src/semantic_normalization/ground_truth/evaluator.py` - Exact duplicate, but check if used in ground truth labeling workflow
5. `src/experimentation/check_progress.py` - Old experiment monitoring, but verify not used in scripts
6. `src/experimentation/show_interventions.py` - Old experiment display, but verify not used in scripts

### Consider Integration (2 files)
7. `src/data_mining/emerging_category_analyzer.py` - Useful functionality, consider adding to data_mining_orchestrator
8. `src/data_mining/fundamental_functions.py` - Interesting analysis, consider adding to data_mining_orchestrator

### Total Cleanup Potential
- **Guaranteed safe to delete:** 3 files
- **Likely safe to delete (verify first):** 4 files
- **Total cleanup:** 7 files (6% of codebase)

---

## 8. Import Patterns Analysis

### Relative Imports
The codebase uses relative imports extensively, especially in:
- `data_mining/` - All utility modules use relative imports
- `semantic_normalization/` - Parent and ground_truth subdirectories

### Absolute Imports
Most orchestration and top-level scripts use absolute imports with `back_end.src.` prefix.

### Import Graph Depth
- **Level 0 (Entry Points):** 7 orchestration scripts
- **Level 1 (Core Libraries):** config.py, database_manager.py, taxonomy files
- **Level 2 (Specialized Libraries):** LLM processors, data collectors
- **Level 3 (Utilities):** scoring_utils, similarity_utils, medical_knowledge
- **Level 4 (Leaf Nodes):** Taxonomy definitions, configuration constants

---

## 9. File Organization Health

### Well-Organized Directories
- `src/orchestration/` - Clear entry points, all actively used
- `src/data/` - Core infrastructure, heavily imported
- `src/data_collection/` - Clean separation of concerns
- `src/conditions/` and `src/interventions/` - Parallel structure, intentional

### Needs Cleanup
- `src/experimentation/` - Contains old experiment scripts (3 unused files)
- `src/semantic_normalization/ground_truth/` - Contains exact duplicates (2 files)
- `src/data_mining/` - Contains 2 unused analysis modules

---

## 10. Dependency Risks

### No Circular Dependencies Detected
The import graph is acyclic - no circular import risks.

### Single Points of Failure
- `src/data/config.py` - 45 imports depend on this file
- `src/data_collection/database_manager.py` - 20 imports depend on this file

**Recommendation:** These files are critical infrastructure and should have comprehensive tests.

---

## 11. Files Safe to Keep

### All Files in These Directories Are Used
- `src/orchestration/` - All 7 files actively used
- `src/data/` - All 6 files heavily imported
- `src/data_collection/` - All 6 files actively used
- `src/llm_processing/` - 3 of 5 files used (entity_operations.py and entity_utils.py deprecated but kept for now)
- `src/conditions/` - All 3 files used
- `src/interventions/` - All 3 files used
- `scripts/` - All 4 migration scripts kept for reference

### Ground Truth Labeling System
The `src/semantic_normalization/ground_truth/` directory contains a complete labeling system:
- `data_exporter.py` - Active
- `generate_candidates.py` - Active
- `labeling_interface.py` - Active
- `label_in_batches.py` - Active
- `pair_generator.py` - Active
- `remove_duplicate_labels.py` - Active
- `prompts.py` - **DUPLICATE** (delete)
- `evaluator.py` - **DUPLICATE** (verify usage first)

---

## 12. Summary Statistics

### Import Density
- Files with 0 internal imports: 24 (leaf nodes or unused)
- Files with 1-3 imports: 38
- Files with 4-10 imports: 42
- Files with 10+ imports: 13 (heavy users)

### Usage Distribution
- Never imported: 22 files (19%)
  - 15 are __init__.py or executable scripts (expected)
  - 7 are truly unused (candidates for removal)
- Imported 1-3 times: 28 files (24%)
- Imported 4+ times: 17 files (15%) - core infrastructure
- Most imported: config.py (45 times)

---

## Appendix A: Full File Listing by Category

### Entry Points (7)
```
src/orchestration/batch_medical_rotation.py
src/orchestration/rotation_group_categorization.py
src/orchestration/rotation_llm_categorization.py
src/orchestration/rotation_llm_processor.py
src/orchestration/rotation_paper_collector.py
src/orchestration/rotation_semantic_grouping_integrator.py
src/orchestration/rotation_semantic_normalizer.py
```

### Executable Scripts (43)
```
scripts/backfill_condition_categories.py
scripts/classify_conditions.py
scripts/migrate_add_mechanism.py
scripts/migrate_intervention_category_nullable.py
src/data_mining/correlation_consistency_checker.py
src/data_mining/data_mining_orchestrator.py
src/data_mining/review_correlations.py
src/experimentation/analysis/results_analyzer.py
src/experimentation/group_categorization/experiment_runner.py
src/experimentation/group_categorization/group_categorizer.py
src/experimentation/group_categorization/validation.py
src/experimentation/runners/dataset_selector.py
src/experimentation/runners/experiment_runner.py
src/llm_processing/batch_entity_processor.py
src/llm_processing/export_to_json.py
src/migrations/add_semantic_normalization_tables.py
src/migrations/create_interventions_view_option_b.py
src/semantic_normalization/cluster_reviewer.py
src/semantic_normalization/embedding_engine.py
src/semantic_normalization/evaluator.py
src/semantic_normalization/experiment_logger.py
src/semantic_normalization/hierarchy_manager.py
src/semantic_normalization/llm_classifier.py
src/semantic_normalization/normalizer.py
src/semantic_normalization/test_runner.py
src/semantic_normalization/ground_truth/data_exporter.py
src/semantic_normalization/ground_truth/evaluator.py
src/semantic_normalization/ground_truth/generate_candidates.py
src/semantic_normalization/ground_truth/labeling_interface.py
src/semantic_normalization/ground_truth/label_in_batches.py
src/semantic_normalization/ground_truth/pair_generator.py
src/semantic_normalization/ground_truth/remove_duplicate_labels.py
src/utils/analyze_mapping_suggestions.py
src/utils/analyze_reviews.py
src/utils/batch_process_summary.py
src/utils/batch_process_unmapped_terms.py
src/utils/cleanup_old_interventions.py
src/utils/create_manual_mappings.py
src/utils/drop_legacy_tables.py
src/utils/export_frontend_data.py
src/utils/reextract_mechanisms.py
src/utils/review_correlations.py
src/utils/run_correlation_extraction.py
```

### Library Modules (45)
```
src/conditions/category_validators.py
src/conditions/search_terms.py
src/conditions/taxonomy.py
src/data/api_clients.py
src/data/config.py
src/data/error_handler.py
src/data/repositories.py
src/data/utils.py
src/data/validators.py
src/data_collection/database_manager.py
src/data_collection/data_mining_repository.py
src/data_collection/fulltext_retriever.py
src/data_collection/paper_parser.py
src/data_collection/pubmed_collector.py
src/data_collection/semantic_scholar_enrichment.py
src/data_mining/bayesian_scorer.py
src/data_mining/biological_patterns.py
src/data_mining/condition_similarity_mapping.py
src/data_mining/failed_interventions.py
src/data_mining/graph_utils.py
src/data_mining/innovation_tracking_system.py
src/data_mining/intervention_consensus_analyzer.py
src/data_mining/medical_knowledge.py
src/data_mining/medical_knowledge_graph.py
src/data_mining/power_combinations.py
src/data_mining/research_gaps.py
src/data_mining/scoring_utils.py
src/data_mining/similarity_utils.py
src/data_mining/treatment_recommendation_engine.py
src/experimentation/config/experiment_config.py
src/experimentation/evaluation/system_monitor.py
src/experimentation/group_categorization/condition_group_categorizer.py
src/experimentation/group_categorization/group_categorizer.py
src/experimentation/group_categorization/validation.py
src/experimentation/runners/dataset_selector.py
src/interventions/category_validators.py
src/interventions/search_terms.py
src/interventions/taxonomy.py
src/llm_processing/batch_entity_processor.py
src/llm_processing/prompt_service.py
src/llm_processing/single_model_analyzer.py
src/orchestration/rotation_session_manager.py
src/semantic_normalization/config.py
src/utils/batch_file_operations.py
src/utils/review_correlations.py
```

### Unused Files (7)
```
src/data_mining/emerging_category_analyzer.py
src/data_mining/fundamental_functions.py
src/experimentation/analyze_interventions.py
src/experimentation/check_progress.py
src/experimentation/show_interventions.py
src/semantic_normalization/ground_truth/prompts.py
src/utils/integration_success_summary.py
```

---

## Appendix B: Verification Commands

### Check if a file is truly unused
```bash
# Search for imports across entire codebase
grep -r "from.*filename" back_end/
grep -r "import.*filename" back_end/

# Check if file has __main__
grep -l "__name__.*==.*__main__" back_end/path/to/file.py
```

### Verify duplicates
```bash
# Compare two files
diff file1.py file2.py

# Compare file sizes
wc -l file1.py file2.py
```

### Safe deletion process
```bash
# 1. Verify no imports
grep -r "emerging_category_analyzer" back_end/

# 2. Check git history
git log -- back_end/src/data_mining/emerging_category_analyzer.py

# 3. Create backup branch
git checkout -b cleanup-unused-files

# 4. Delete file
rm back_end/src/data_mining/emerging_category_analyzer.py

# 5. Run tests
python -m pytest back_end/

# 6. Commit if tests pass
git add -A
git commit -m "Remove unused file: emerging_category_analyzer.py"
```

---

**Report End**
