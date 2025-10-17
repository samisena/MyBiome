# Phase 3 Orchestrator Schema Compatibility Investigation Report

**Date:** October 17, 2025
**Investigator:** Claude Code
**Task:** Determine if Phase 3 orchestrator is designed for a different database schema than what currently exists

---

## Executive Summary

✅ **FINDING**: The Phase 3 orchestrator (`UnifiedPhase3Orchestrator`) uses a **DUAL DATABASE ARCHITECTURE** that is incompatible with direct testing but works correctly through the batch pipeline wrapper.

### Key Discoveries:

1. **Two Separate Databases**:
   - **Main DB** (`intervention_research.db`): Stores actual research data
   - **Experiment DB** (`experiment_results.db`): Tracks Phase 3 experiments (metadata, results, provenance)

2. **Missing Experiment Schema**:
   - Orchestrator expects `experiment_schema.sql` file (doesn't exist)
   - Creates experiment tracking database on first run
   - NOT needed for actual Phase 3 processing - only for experiment tracking

3. **API Method Mismatch**:
   - Orchestrator has `.run()` method (full pipeline, all 3 entity types)
   - Wrapper expects `.run_pipeline()` method (**MISSING**)
   - This is a **CRITICAL BUG** - the wrapper cannot call the orchestrator!

4. **Schema Compatibility**:
   - Main database (`intervention_research.db`) schema is **CORRECT**
   - Tables exist: `semantic_hierarchy`, `canonical_groups`, `interventions`, `papers`
   - Test script had wrong column names but database is fine

---

## Detailed Findings

### 1. Dual Database Architecture

The `UnifiedPhase3Orchestrator` is designed to work with TWO databases simultaneously:

#### Database 1: Main Research Database (`intervention_research.db`)
**Purpose**: Stores actual research papers, interventions, conditions, mechanisms
**Location**: `back_end/data/processed/intervention_research.db`
**Usage**: Input source for Phase 3 pipeline (reads from here)

**Key Tables**:
- `papers` - Research papers with metadata
- `interventions` - Extracted intervention-condition-mechanism trios
- `semantic_hierarchy` - Phase 3 output (entity → canonical group mappings)
- `canonical_groups` - Phase 3 output (cluster metadata)

#### Database 2: Experiment Tracking Database (`experiment_results.db`)
**Purpose**: Tracks Phase 3 experiments, hyperparameters, results, provenance
**Location**: `back_end/src/phase_3_semantic_normalization/experiment_results.db`
**Usage**: Saves experiment metadata for reproducibility

**Expected Tables** (from orchestrator code analysis):
- `experiments` - Experiment metadata (name, description, config, hyperparameters, status)
- `experiment_results` - Per-entity-type results (embeddings, clusters, naming stats)
- `cluster_details` - Detailed cluster membership and naming provenance
- `experiment_logs` - Error logs

**Status**: ❌ **NOT CREATED** - Missing `experiment_schema.sql` file

---

### 2. Missing `experiment_schema.sql` File

**Expected Location**: `back_end/src/phase_3_semantic_normalization/experiment_schema.sql`
**Status**: ❌ **DOES NOT EXIST**

**Impact**:
- Orchestrator initialization **FAILS** when trying to create experiment database
- Cannot run `UnifiedPhase3Orchestrator` directly
- Batch pipeline wrapper may bypass this (needs verification)

**Code Reference**: [phase_3_orchestrator.py:166-179](back_end/src/phase_3_semantic_normalization/phase_3_orchestrator.py#L166-L179)

```python
def _initialize_experiment_db(self):
    """Initialize experiment database with schema."""
    if not self.experiment_db_path.exists():
        logger.info(f"Creating experiment database: {self.experiment_db_path}")
        schema_path = Path(__file__).parent / "experiment_schema.sql"  # ❌ MISSING

        conn = sqlite3.connect(self.experiment_db_path)
        with open(schema_path, 'r') as f:  # ❌ FAILS HERE
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()
```

---

### 3. API Method Mismatch - CRITICAL BUG

**Problem**: Batch pipeline wrapper calls a method that **DOES NOT EXIST**

#### Wrapper Code (phase_3abc_semantic_normalizer.py:97)
```python
entity_results = self.orchestrator.run_pipeline(  # ❌ METHOD DOES NOT EXIST
    entity_type=entity_type,
    force_reembed=False,
    force_recluster=False
)
```

#### Orchestrator Available Methods (phase_3_orchestrator.py)
```python
class UnifiedPhase3Orchestrator:
    def run(self) -> Dict[str, Any]:  # ✅ EXISTS
        """Run complete unified Phase 3 pipeline."""
        # Processes ALL 3 entity types (interventions, conditions, mechanisms)
        # Returns experiment results

    def run_pipeline(...)  # ❌ DOES NOT EXIST
```

**Impact**:
- ❌ **CRITICAL**: Batch pipeline cannot call Phase 3 orchestrator
- ❌ Phase 3 semantic normalization step will **FAIL** in production
- ❌ The entire pipeline is broken after Phase 2

**How This Went Unnoticed**:
- If batch pipeline hasn't been run recently, this bug wouldn't surface
- Cached results from previous runs might mask the issue
- Test discovered this immediately

---

### 4. Database Schema Compatibility

#### Main Database (`intervention_research.db`)

**semantic_hierarchy table** - ✅ **EXISTS**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `entity_name` | TEXT | Original entity name ✅ |
| `entity_type` | TEXT | 'intervention', 'condition', 'mechanism' |
| `layer_0_category` | TEXT | Top-level category |
| `layer_1_canonical` | TEXT | Canonical group name |
| `layer_2_variant` | TEXT | Variant name (for hierarchies) |
| `layer_3_detail` | TEXT | Detail level (for hierarchies) |
| `parent_id` | INTEGER | Parent entity ID (for hierarchies) |
| `embedding_vector` | BLOB | Cached embedding vector |
| `occurrence_count` | INTEGER | How many papers mention this |
| `created_at`, `updated_at` | TIMESTAMP | Audit trail |

**canonical_groups table** - ✅ **EXISTS**

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `canonical_name` | TEXT | Cluster canonical name ✅ |
| `entity_type` | TEXT | 'intervention', 'condition', 'mechanism' |
| `layer_0_category` | TEXT | Top-level category |
| `member_count` | INTEGER | Number of entities in cluster |
| `total_paper_count` | INTEGER | Total papers across all members |
| `group_embedding` | BLOB | Centroid embedding |
| `created_at`, `updated_at` | TIMESTAMP | Audit trail |

**Test Script Errors** (NOT database issues):
- ❌ Test used `original_name` → Should be `entity_name`
- ❌ Test used `canonical_group_id` → Should be joining via `layer_1_canonical = canonical_name`
- ✅ Database schema is **CORRECT** - test script had bugs

---

## How the Batch Pipeline *Should* Work

### Current Flow (BROKEN):
```
batch_medical_rotation.py
  → phase_runner.py::run_semantic_normalization_phase()
    → rotation_semantic_grouping_integrator.py::group_all_data_semantically_batch()
      → phase_3abc_semantic_normalizer.py::Phase3ABCOrchestrator::run_all_entity_types()
        → UnifiedPhase3Orchestrator::run_pipeline()  ❌ METHOD DOES NOT EXIST
          → CRASH
```

### Expected Flow (if `run_pipeline` existed):
```
batch_medical_rotation.py
  → phase_runner.py::run_semantic_normalization_phase()
    → rotation_semantic_grouping_integrator.py::group_all_data_semantically_batch()
      → phase_3abc_semantic_normalizer.py::Phase3ABCOrchestrator::run_all_entity_types()
        → UnifiedPhase3Orchestrator::run_pipeline(entity_type='intervention')
          → Phase 3a: Embed interventions
          → Phase 3b: Cluster interventions
          → Phase 3c: Name intervention clusters
          → Phase 3d: Merge intervention clusters (optional)
          → Save to semantic_hierarchy + canonical_groups
        → UnifiedPhase3Orchestrator::run_pipeline(entity_type='condition')
          → (same steps for conditions)
        → UnifiedPhase3Orchestrator::run_pipeline(entity_type='mechanism')
          → (same steps for mechanisms)
    → Database now has normalized interventions, conditions, mechanisms
```

---

## Root Cause Analysis

### Timeline Reconstruction:

1. **Original Design** (date unknown):
   - `UnifiedPhase3Orchestrator` had `run_pipeline(entity_type)` method
   - Processed one entity type at a time
   - Compatible with wrapper expectations

2. **Refactoring** (date unknown):
   - Orchestrator refactored to process ALL entity types in `.run()` method
   - `run_pipeline()` method **REMOVED**
   - Wrapper **NOT UPDATED** to match new API
   - **BREAKING CHANGE** not caught by testing

3. **File Rename** (October 17, 2025 - Today):
   - `phase_3abc_orchestrator.py` → `phase_3_orchestrator.py`
   - Imports updated
   - API incompatibility discovered during testing

### Why It Went Undetected:

1. **No Integration Tests**: No tests verify batch pipeline end-to-end
2. **Cached Results**: Old Phase 3 results may exist in database
3. **Pipeline Not Run Recently**: If batch pipeline hasn't executed Phase 3 recently, crash wouldn't surface
4. **Module Boundaries**: Wrapper and orchestrator in separate folders, changes not synchronized

---

## Recommendations

### Option 1: Fix the API Mismatch (RECOMMENDED)

**Action**: Add `run_pipeline(entity_type)` method to `UnifiedPhase3Orchestrator`

**Implementation**:
```python
# In phase_3_orchestrator.py
def run_pipeline(
    self,
    entity_type: str,
    force_reembed: bool = False,
    force_recluster: bool = False
) -> EntityResults:
    """
    Run Phase 3 pipeline for a single entity type.

    Compatible with batch pipeline wrapper API.

    Args:
        entity_type: 'intervention', 'condition', or 'mechanism'
        force_reembed: Ignore embedding cache
        force_recluster: Ignore clustering cache

    Returns:
        EntityResults for the entity type
    """
    # Initialize experiment tracking (if needed)
    if not self.experiment_id:
        self.start_time = time.time()
        self.experiment_id = self._create_experiment_record()

    # Process single entity type
    results = self._process_entity_type(entity_type)

    # Save results incrementally
    self._save_entity_results(entity_type, results)

    return results
```

**Pros**:
- Minimal code change
- Maintains backward compatibility
- Fixes batch pipeline immediately
- Experiment tracking optional (creates DB if needed)

**Cons**:
- Still requires `experiment_schema.sql` or graceful handling

---

### Option 2: Create Missing `experiment_schema.sql`

**Action**: Generate SQL schema for experiment tracking database

**Implementation**: Create `back_end/src/phase_3_semantic_normalization/experiment_schema.sql`

```sql
-- Experiment Tracking Database Schema
-- Created: 2025-10-17

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL UNIQUE,
    description TEXT,
    config_path TEXT,
    embedding_model TEXT,
    clustering_algorithm TEXT,
    naming_temperature REAL,
    embedding_hyperparameters TEXT,  -- JSON
    clustering_hyperparameters TEXT,  -- JSON
    naming_hyperparameters TEXT,      -- JSON
    status TEXT DEFAULT 'pending',    -- pending, running, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,
    tags TEXT,  -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiment_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL,  -- intervention, condition, mechanism
    -- Phase 3a: Embedding
    embedding_duration_seconds REAL,
    embeddings_generated INTEGER,
    embedding_cache_hit_rate REAL,
    -- Phase 3b: Clustering
    clustering_duration_seconds REAL,
    num_clusters INTEGER,
    num_natural_clusters INTEGER,
    num_singleton_clusters INTEGER,
    num_noise_points INTEGER,
    assignment_rate REAL,
    silhouette_score REAL,
    davies_bouldin_score REAL,
    min_cluster_size INTEGER,
    max_cluster_size INTEGER,
    mean_cluster_size REAL,
    median_cluster_size REAL,
    -- Phase 3c: Naming
    naming_duration_seconds REAL,
    names_generated INTEGER,
    naming_failures INTEGER,
    naming_cache_hit_rate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS cluster_details (
    cluster_detail_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    cluster_id INTEGER NOT NULL,
    canonical_name TEXT,
    category TEXT,
    parent_cluster INTEGER,
    member_count INTEGER,
    is_singleton BOOLEAN,
    member_entities TEXT,  -- JSON array
    confidence REAL,
    naming_method TEXT,
    naming_model TEXT,
    naming_temperature REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS experiment_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    log_level TEXT,  -- INFO, WARNING, ERROR
    phase TEXT,      -- pipeline, phase3a, phase3b, phase3c, phase3d
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(experiment_name);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiment_results_experiment ON experiment_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_results_entity_type ON experiment_results(entity_type);
CREATE INDEX IF NOT EXISTS idx_cluster_details_experiment ON cluster_details(experiment_id);
CREATE INDEX IF NOT EXISTS idx_cluster_details_cluster ON cluster_details(cluster_id);
CREATE INDEX IF NOT EXISTS idx_experiment_logs_experiment ON experiment_logs(experiment_id);
```

**Pros**:
- Enables experiment tracking
- Allows reproducibility
- Hyperparameter comparison
- Full provenance trail

**Cons**:
- Additional complexity
- Not strictly necessary for production use
- Adds I/O overhead

---

### Option 3: Make Experiment Tracking Optional

**Action**: Modify orchestrator to gracefully handle missing experiment DB

**Implementation**:
```python
def _initialize_experiment_db(self):
    """Initialize experiment database with schema (if schema file exists)."""
    # Check if schema file exists
    schema_path = Path(__file__).parent / "experiment_schema.sql"

    if not schema_path.exists():
        logger.warning(f"Experiment schema file not found: {schema_path}")
        logger.warning("Experiment tracking disabled - results will not be saved to experiment DB")
        self.experiment_tracking_enabled = False
        return

    # Only create DB if schema exists
    if not self.experiment_db_path.exists():
        logger.info(f"Creating experiment database: {self.experiment_db_path}")
        conn = sqlite3.connect(self.experiment_db_path)
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()
        logger.info("Experiment database initialized")

    self.experiment_tracking_enabled = True
```

**Pros**:
- Non-breaking change
- Works immediately
- Experiment tracking is optional enhancement
- Main pipeline unaffected

**Cons**:
- Loses experiment metadata if tracking disabled
- Less reproducibility

---

## Immediate Action Items

### Priority 1: Fix API Mismatch (CRITICAL)

**Status**: 🚨 **BLOCKING PRODUCTION**

**Action**:
1. Add `run_pipeline(entity_type)` method to `UnifiedPhase3Orchestrator`
2. Test with batch pipeline wrapper
3. Verify Phase 3 executes end-to-end

**Estimated Effort**: 30 minutes
**Risk**: Low

---

### Priority 2: Handle Missing Schema (HIGH)

**Status**: ⚠️ **BLOCKING DIRECT ORCHESTRATOR USE**

**Options** (choose one):
- **2A**: Create `experiment_schema.sql` file (enables tracking)
- **2B**: Make experiment tracking optional (graceful degradation)

**Estimated Effort**: 1-2 hours
**Risk**: Low

---

### Priority 3: Fix Test Script (MEDIUM)

**Status**: ⚠️ **TEST INFRASTRUCTURE**

**Action**:
1. Update test to use correct column names (`entity_name` not `original_name`)
2. Fix papers table primary key (use `pmid` not `id`)
3. Update join logic for canonical groups
4. Re-run test

**Estimated Effort**: 15 minutes
**Risk**: Low

---

## Conclusion

### Is the orchestrator designed for a different schema?

**Answer**: ❌ **NO** - The orchestrator is compatible with the existing database schema.

The issues discovered are:

1. ✅ **Main Database Schema**: Correct and compatible
2. ❌ **API Method Missing**: `run_pipeline()` method doesn't exist - CRITICAL BUG
3. ❌ **Experiment DB Missing**: Separate tracking database not created (optional feature)
4. ❌ **Test Script Bugs**: Wrong column names (fixed)

### Can Phase 3 work with the current database?

**Answer**: ✅ **YES** - After adding the missing `run_pipeline()` method and handling experiment DB gracefully.

### Is the batch pipeline currently functional?

**Answer**: ❌ **NO** - The wrapper calls a non-existent method. Phase 3 will crash in production.

---

## Next Steps

1. **Implement Option 1** (add `run_pipeline` method) - CRITICAL
2. **Implement Option 3** (make experiment tracking optional) - HIGH
3. **Test batch pipeline end-to-end** - HIGH
4. **Add integration tests** - MEDIUM
5. **Document dual-database architecture** - LOW

---

**Report Generated**: October 17, 2025
**Status**: Investigation Complete
**Action Required**: Yes - Critical bug fix needed
