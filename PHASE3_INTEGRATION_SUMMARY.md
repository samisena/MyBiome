# Phase 3: Main Pipeline Integration - Summary

**Status**: ⏳ IN PROGRESS (2/5 complete)
**Date**: October 6, 2025
**Objective**: Integrate hierarchical semantic normalization into main MyBiome pipeline

---

## Completed ✅

### 1. Database Migration Script ✅
**File**: [`back_end/src/migrations/add_semantic_normalization_tables.py`](back_end/src/migrations/add_semantic_normalization_tables.py)

**Creates**:
- `semantic_hierarchy` table (4-layer hierarchical structure)
- `entity_relationships` table (pairwise relationship tracking)
- `canonical_groups` table (Layer 1 aggregation entities)
- Indexes for performance (6 indexes)
- Views for common queries (3 views)

**Usage**:
```bash
# Create tables
python -m back_end.src.migrations.add_semantic_normalization_tables

# Create tables AND drop old ones (DESTRUCTIVE)
python -m back_end.src.migrations.add_semantic_normalization_tables --drop-old
```

**Features**:
- Checks if tables already exist
- Shows old table status (canonical_entities, entity_mappings)
- Optional drop of old tables with confirmation
- Creates helpful views (v_intervention_hierarchy, v_intervention_by_canonical, v_intervention_by_variant)

---

### 2. Semantic Normalization Orchestrator ✅
**File**: [`back_end/src/orchestration/rotation_semantic_normalizer.py`](back_end/src/orchestration/rotation_semantic_normalizer.py)

**Purpose**: Run hierarchical semantic normalization after LLM processing

**Features**:
- ✅ Load interventions for condition from database
- ✅ Generate embeddings (cached)
- ✅ Extract canonical groups via LLM (cached)
- ✅ Find similar interventions and classify relationships
- ✅ Populate semantic_hierarchy tables
- ✅ Update canonical_groups aggregations
- ✅ Resumable (checks what's already normalized)
- ✅ Incremental (only processes new interventions)
- ✅ Batch-aware (processes in chunks of 50)
- ✅ Session management with progress tracking

**Usage**:
```bash
# Normalize single condition
python -m back_end.src.orchestration.rotation_semantic_normalizer diabetes

# Normalize all conditions
python -m back_end.src.orchestration.rotation_semantic_normalizer --all

# Resume from latest session
python -m back_end.src.orchestration.rotation_semantic_normalizer --resume

# Check status
python -m back_end.src.orchestration.rotation_semantic_normalizer --status
```

---

## Remaining Tasks ⏳

### 3. Integrate into Batch Medical Rotation ⏳
**File**: `back_end/src/orchestration/batch_medical_rotation.py`

**Changes Needed**:
```python
# Current phases:
# 1. Collection
# 2. Processing
# 2.5. Categorization
# 3. Semantic Grouping (OLD - batch_entity_processor.py)

# New phases:
# 1. Collection
# 2. Processing
# 2.5. Categorization
# 3. Semantic Normalization (NEW - rotation_semantic_normalizer.py)
# 4. Data Mining (moved from 3)
```

**Command**:
```bash
# Run only semantic normalization phase
python -m back_end.src.orchestration.batch_medical_rotation --start-phase semantic_normalization

# Run full pipeline with new phase
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10
```

---

### 4. Update Data Mining Modules ⏳
**Files to update**:
- `back_end/src/data_mining/medical_knowledge_graph.py`
- `back_end/src/data_mining/bayesian_scorer.py`
- `back_end/src/data_mining/treatment_recommendation_engine.py`

**Query Updates**:

**OLD**:
```python
# Query canonical_entities (old table)
SELECT canonical_name, COUNT(*)
FROM canonical_entities
GROUP BY canonical_name
```

**NEW**:
```python
# Query semantic_hierarchy (new table)
SELECT layer_1_canonical, SUM(occurrence_count)
FROM semantic_hierarchy
WHERE entity_type = 'intervention'
GROUP BY layer_1_canonical

# Or use view
SELECT * FROM v_intervention_by_canonical
```

**Benefits**:
- Access to 4-layer hierarchy (not just flat canonical)
- Relationship type awareness (EXACT_MATCH, VARIANT, etc.)
- Better aggregation rules
- Embedding-based similarity scores

---

### 5. Deprecate Old Semantic Grouping ⏳
**File**: `back_end/src/orchestration/rotation_semantic_grouping_integrator.py`

**Action**: ❌ DELETE (after verification)

**Old tables** (in `intervention_research.db`):
- `canonical_entities` (71 entities)
- `entity_mappings` (204 mappings)
- `llm_normalization_cache`
- `normalized_terms_cache`

**Action**:
1. Run new system in parallel with old for testing
2. Compare results (old vs new)
3. Verify new system works correctly
4. Drop old tables with migration script: `--drop-old`
5. Delete `rotation_semantic_grouping_integrator.py`

---

## Migration Workflow

### Step 1: Database Migration
```bash
cd back_end

# Check current tables
sqlite3 data/processed/intervention_research.db ".tables"

# Run migration
python -m back_end.src.migrations.add_semantic_normalization_tables

# Verify tables created
sqlite3 data/processed/intervention_research.db ".schema semantic_hierarchy"
```

---

### Step 2: Test Normalization on Single Condition
```bash
# Run on one condition
python -m back_end.src.orchestration.rotation_semantic_normalizer diabetes

# Check status
python -m back_end.src.orchestration.rotation_semantic_normalizer --status

# Verify data in database
sqlite3 data/processed/intervention_research.db "SELECT COUNT(*) FROM semantic_hierarchy"
```

---

### Step 3: Run on All Conditions
```bash
# Process all conditions
python -m back_end.src.orchestration.rotation_semantic_normalizer --all

# Monitor progress (in another terminal)
watch -n 10 "python -m back_end.src.orchestration.rotation_semantic_normalizer --status"
```

---

### Step 4: Integrate into Main Pipeline
```bash
# Add to batch_medical_rotation.py
# (manual code changes needed)

# Test integration
python -m back_end.src.orchestration.batch_medical_rotation --start-phase semantic_normalization
```

---

### Step 5: Update Data Mining
```bash
# Update queries in data mining modules
# (manual code changes needed)

# Test data mining with new schema
python -m back_end.src.data_mining.data_mining_orchestrator --all
```

---

### Step 6: Deprecate Old System
```bash
# Verify new system works
# Compare old vs new results

# Drop old tables (DESTRUCTIVE - no undo!)
python -m back_end.src.migrations.add_semantic_normalization_tables --drop-old

# Delete old orchestrator
rm back_end/src/orchestration/rotation_semantic_grouping_integrator.py
```

---

## Comparison: Old vs New

| Feature | Old (batch_entity_processor) | New (semantic_normalization) |
|---------|------------------------------|------------------------------|
| **Structure** | Flat canonical entities | 4-layer hierarchy |
| **Relationships** | None | 6 types (EXACT_MATCH, VARIANT, etc.) |
| **Embeddings** | ❌ | ✅ nomic-embed-text (768-dim) |
| **LLM Model** | qwen2.5:14b | qwen3:14b (optimized) |
| **Caching** | Basic | Advanced (embeddings + LLM) |
| **Ground Truth** | ❌ | ✅ 50 labeled (500 ready) |
| **Evaluation** | ❌ | ✅ Accuracy metrics |
| **Aggregation** | Simple counts | Hierarchical aggregation |
| **Tables** | 4 tables | 3 tables + 3 views |
| **Orchestrator** | rotation_semantic_grouping_integrator.py | rotation_semantic_normalizer.py |

---

## Database Schema Comparison

### Old Schema
```sql
-- Flat structure
canonical_entities (id, canonical_name, ...)
entity_mappings (id, intervention_name, canonical_id, ...)
```

### New Schema
```sql
-- Hierarchical structure
semantic_hierarchy (
    id,
    entity_name,
    layer_0_category,      -- From taxonomy
    layer_1_canonical,     -- "probiotics", "statins"
    layer_2_variant,       -- "L. reuteri", "atorvastatin"
    layer_3_detail,        -- "L. reuteri 10^9 CFU"
    embedding_vector,      -- Semantic embedding
    ...
)

entity_relationships (
    entity_1_id,
    entity_2_id,
    relationship_type,     -- EXACT_MATCH, VARIANT, etc.
    similarity_score,
    ...
)

canonical_groups (
    canonical_name,        -- Layer 1 canonical
    member_count,
    total_paper_count,
    ...
)
```

---

## Performance Estimates

**Single Condition** (e.g., diabetes with 50 interventions):
- Embedding generation: ~5 seconds (first time) or instant (cached)
- LLM canonical extraction: ~20 seconds per batch of 20
- Relationship classification: ~15 seconds per batch
- **Total**: ~2-3 minutes

**All Conditions** (60 conditions × ~20 interventions each = 1,200 interventions):
- Embedding generation: ~60 seconds (first time)
- LLM processing: ~90 minutes (with caching)
- Relationship classification: ~45 minutes
- **Total**: ~2-2.5 hours

**With Full Caching** (subsequent runs):
- ~5-10 minutes (only new interventions)

---

## Testing Checklist

Before integrating into main pipeline:

- [ ] Run database migration successfully
- [ ] Test normalization on single condition (diabetes)
- [ ] Verify semantic_hierarchy table populated correctly
- [ ] Check canonical_groups aggregations
- [ ] Test relationship tracking
- [ ] Verify embeddings cached correctly
- [ ] Test resumability (interrupt and resume)
- [ ] Run on all 60 conditions
- [ ] Compare results with old system
- [ ] Verify data mining queries work with new schema
- [ ] Test end-to-end pipeline with new phase

---

## Risk Assessment

**Low Risk**:
- Running new system in parallel with old
- Creating new tables (doesn't affect existing data)
- Testing on single condition first

**Medium Risk**:
- Integrating into batch_medical_rotation.py
- Updating data mining queries
- Performance impact on pipeline

**High Risk**:
- Dropping old tables (`--drop-old`) - IRREVERSIBLE
- Deleting old orchestrator files
- Full replacement without testing

---

## Rollback Plan

If new system fails:
1. Old tables still exist (unless `--drop-old` used)
2. Old orchestrator still exists (unless deleted)
3. Can revert to old system by:
   - Commenting out new phase in batch_medical_rotation.py
   - Using old data mining queries
   - Running old orchestrator

**Important**: Don't use `--drop-old` until new system fully validated!

---

## Next Steps

### Immediate (Today)
1. ✅ Create database migration script
2. ✅ Create semantic normalization orchestrator
3. ⏳ Test migration on development database
4. ⏳ Test normalization on single condition

### Short-term (This Week)
1. ⏳ Integrate into batch_medical_rotation.py
2. ⏳ Update data mining modules
3. ⏳ Test full pipeline with new phase
4. ⏳ Document changes in claude.md

### Medium-term (Next Week)
1. ⏳ Run on all 60 conditions
2. ⏳ Compare old vs new results
3. ⏳ Validate accuracy with ground truth
4. ⏳ Deprecate old system (if validated)

---

## Summary

**Phase 3 Progress**: 40% complete (2/5 tasks)

**Completed**:
- ✅ Database migration script (add_semantic_normalization_tables.py)
- ✅ Semantic normalization orchestrator (rotation_semantic_normalizer.py)

**Remaining**:
- ⏳ Integrate into batch_medical_rotation.py
- ⏳ Update data mining modules
- ⏳ Deprecate old semantic grouping

**Status**: Ready for testing and integration

**Estimated Time to Complete**: 4-6 hours
- Database migration + testing: 1 hour
- Integration into pipeline: 2-3 hours
- Data mining updates: 1-2 hours
- Testing and validation: 1 hour

---

**Date**: October 6, 2025
**Next Phase**: Testing & Validation OR Ground Truth Labeling (500 pairs)
