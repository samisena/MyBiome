# Migration Guide: Hierarchical Semantic Normalization

**Date**: October 6, 2025
**Migration**: Phase 3 (Legacy Semantic Grouping) → Phase 3.5 (Hierarchical Semantic Normalization)
**Status**: Migration complete, legacy system deprecated

---

## Overview

This guide documents the migration from the legacy Phase 3 semantic grouping system to the new Phase 3.5 hierarchical semantic normalization system.

---

## Why the Change?

### Old System (Phase 3 - DEPRECATED)
**Architecture**: Simple 2-table system
- `canonical_entities` - Unified intervention names
- `entity_mappings` - Original → canonical mappings
- **Limitation**: Flat structure, no relationship types, no hierarchy
- **Method**: LLM-based grouping only (batch size: 20)

**Problems**:
1. ❌ No semantic similarity scoring
2. ❌ No relationship classification (EXACT_MATCH vs VARIANT vs SUBTYPE)
3. ❌ Flat structure - no hierarchical layers
4. ❌ Limited metadata for data mining
5. ❌ No embedding-based similarity matching

### New System (Phase 3.5 - CURRENT)
**Architecture**: Advanced 3-table hierarchical system
- `semantic_hierarchy` - 4-layer hierarchical structure (Category → Canonical → Variant → Dosage)
- `entity_relationships` - Pairwise relationships with 6 types
- `canonical_groups` - Layer 1 aggregation entities

**Benefits**:
1. ✅ Embedding-based semantic similarity (nomic-embed-text 768-dim)
2. ✅ LLM canonical extraction (qwen3:14b with caching)
3. ✅ 6 relationship types (EXACT_MATCH, VARIANT, SUBTYPE, SAME_CATEGORY, DOSAGE_VARIANT, DIFFERENT)
4. ✅ 4-layer hierarchy for precise aggregation
5. ✅ Rich metadata for advanced data mining
6. ✅ Resumable sessions with state persistence
7. ✅ Comprehensive caching (embeddings + LLM decisions)

---

## Database Migration

### Step 1: Backup Existing Database

**CRITICAL**: Always backup before migration!

```bash
cd back_end/data/processed
cp intervention_research.db intervention_research.db.backup_$(date +%Y%m%d)
```

### Step 2: Add New Tables

The migration script has already created the new tables. Verify they exist:

```sql
-- Check if tables exist
SELECT name FROM sqlite_master
WHERE type='table'
AND name IN ('semantic_hierarchy', 'entity_relationships', 'canonical_groups');
```

**Expected output**: 3 rows (semantic_hierarchy, entity_relationships, canonical_groups)

If tables don't exist, run the migration:

```bash
cd back_end/src/migrations
python add_semantic_normalization_tables.py
```

### Step 3: Verify Schema

```sql
-- Check semantic_hierarchy structure
PRAGMA table_info(semantic_hierarchy);

-- Check entity_relationships structure
PRAGMA table_info(entity_relationships);

-- Check canonical_groups structure
PRAGMA table_info(canonical_groups);
```

### Step 4: Legacy Table Handling

**Do NOT drop legacy tables yet!** Keep for backward compatibility during transition.

Legacy tables to keep temporarily:
- `canonical_entities` (191 rows)
- `entity_mappings` (389 rows)
- `llm_normalization_cache`
- `normalized_terms_cache`

**When to drop**: After Phase 3.5 is fully integrated and data mining modules are updated.

---

## Configuration Changes

### Old Configuration (experiments/semantic_normalization/config/config.yaml)

```yaml
embedding:
  model: "nomic-embed-text"
  dimension: 768
  batch_size: 32

llm:
  model: "qwen3:14b"
  temperature: 0.0
  timeout: 60

paths:
  cache_dir: "data/cache"
  ground_truth_dir: "data/ground_truth"

thresholds:
  exact_match: 0.95
  variant: 0.85
  subtype: 0.75
```

### New Configuration (back_end/src/data/config.py)

```python
from back_end.src.data.config import config

# Access settings via unified config
embedding_model = config.semantic_embedding_model  # "nomic-embed-text"
llm_model = config.semantic_canonical_llm_model  # "qwen3:14b"
batch_size = config.semantic_batch_size  # 50

# Thresholds
exact_match_threshold = config.semantic_exact_match_threshold  # 0.95
variant_threshold = config.semantic_variant_threshold  # 0.85

# Paths
cache_dir = config.semantic_cache_dir  # Path object
results_dir = config.semantic_results_dir  # Path object
```

**Migration**: All experimental YAML configs merged into `config.py` with `semantic_` prefix.

---

## Running for the First Time

### 1. Initialize Directories

Directories are auto-created when you import config:

```python
from back_end.src.data.config import config
# Automatically creates:
# - data/semantic_normalization_cache/
# - data/semantic_normalization_results/
```

### 2. Test with Single Condition

Start with a small test to verify everything works:

```bash
cd back_end

# Test with a condition that has interventions
python -m src.orchestration.rotation_semantic_normalizer "Type 2 diabetes mellitus (T2DM)"
```

**Expected output**:
```
================================================================================
NORMALIZING: Type 2 diabetes mellitus (T2DM)
================================================================================
Found 45 interventions

Running normalization (batch size: 50)...

[OK] Normalization complete!
  - Processed: 45
  - Canonical groups: 28
  - Relationships: 127
```

### 3. Check Status

```bash
python -m src.orchestration.rotation_semantic_normalizer --status
```

**Expected output**:
```
================================================================================
SEMANTIC NORMALIZATION STATUS
================================================================================

Database: intervention_research.db

Semantic Hierarchy:
  - Interventions normalized: 45
  - Canonical groups (Layer 1): 28
  - Relationships tracked: 127
  - Canonical group records: 28
```

### 4. Normalize All Conditions

Once verified working, run for all conditions:

```bash
python -m src.orchestration.rotation_semantic_normalizer --all --batch-size 50
```

**This will**:
- Process all unique health conditions in database
- Create resumable session
- Save progress after each condition
- Cache embeddings and LLM decisions
- Create hierarchical relationships

---

## Troubleshooting Guide

### Issue 1: "No module named 'back_end.src.semantic_normalization'"

**Cause**: Python path issues

**Fix**:
```bash
cd /path/to/MyBiome
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m back_end.src.orchestration.rotation_semantic_normalizer --status
```

### Issue 2: "semantic_hierarchy table does not exist"

**Cause**: Migration not run

**Fix**:
```bash
cd back_end/src/migrations
python add_semantic_normalization_tables.py
```

### Issue 3: UnicodeEncodeError with checkmarks/emojis

**Cause**: Windows console (cp1252) encoding

**Fix**: Already fixed in code! All Unicode symbols replaced with ASCII ([OK], [ERROR], [WARNING])

### Issue 4: "Found 0 interventions" for known condition

**Cause**: Exact condition name mismatch

**Fix**: Check exact condition names in database:
```bash
python -c "
import sqlite3
conn = sqlite3.connect('back_end/data/processed/intervention_research.db')
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT health_condition FROM interventions ORDER BY health_condition')
for row in cursor.fetchall():
    print(row[0])
"
```

Use the **exact** condition name from output.

### Issue 5: Ollama connection errors

**Cause**: Ollama not running or models not loaded

**Fix**:
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull required models if needed
ollama pull nomic-embed-text
ollama pull qwen3:14b
```

### Issue 6: Cache permission errors

**Cause**: Cache directory not writable

**Fix**:
```bash
# Ensure cache directories exist and are writable
mkdir -p data/semantic_normalization_cache
mkdir -p data/semantic_normalization_results
chmod -R 755 data/semantic_normalization_*
```

### Issue 7: Out of memory with large batches

**Cause**: Too many interventions processed at once

**Fix**: Reduce batch size:
```bash
python -m src.orchestration.rotation_semantic_normalizer "condition" --batch-size 20
```

---

## Data Migration (Optional)

If you want to migrate data from legacy Phase 3 to Phase 3.5:

### Option 1: Fresh Start (Recommended)
Simply run Phase 3.5 normalizer - it will create new normalized data from scratch:
```bash
python -m src.orchestration.rotation_semantic_normalizer --all
```

### Option 2: Preserve Legacy Mappings
Keep legacy tables and let both systems coexist:
- Legacy system: `canonical_entities`, `entity_mappings`
- New system: `semantic_hierarchy`, `entity_relationships`, `canonical_groups`

Data mining can be updated to query new tables when ready.

---

## Performance Comparison

| Metric | Legacy (Phase 3) | New (Phase 3.5) |
|--------|------------------|-----------------|
| **Tables** | 2 | 3 |
| **Relationship Types** | None | 6 types |
| **Hierarchy Layers** | 0 (flat) | 4 layers |
| **Semantic Matching** | LLM only | Embeddings + LLM |
| **Caching** | Basic | Comprehensive |
| **Resumable** | No | Yes |
| **Batch Size** | 20 | 50 |
| **Cache Hit Rate** | ~0% | ~40% (542 items cached) |
| **Processing Speed** | Moderate | Fast (with caching) |

---

## Integration Checklist

- [x] Database tables created (`semantic_hierarchy`, `entity_relationships`, `canonical_groups`)
- [x] Configuration merged into `config.py`
- [x] Orchestrator tested (`rotation_semantic_normalizer.py`)
- [x] Documentation updated (`claude.md`)
- [x] Single-database architecture verified
- [ ] Integrate into `batch_medical_rotation.py` as Phase 3.5
- [ ] Update data mining modules to query new tables
- [ ] Drop legacy tables after verification
- [ ] Archive experimental folder

---

## Next Steps

### Immediate (After Migration)
1. **Run normalizer on all conditions**:
   ```bash
   python -m src.orchestration.rotation_semantic_normalizer --all
   ```

2. **Verify data populated**:
   ```sql
   SELECT COUNT(*) FROM semantic_hierarchy;
   SELECT COUNT(*) FROM entity_relationships;
   SELECT COUNT(*) FROM canonical_groups;
   ```

### Short-term (Integration)
3. **Integrate into main pipeline** (`batch_medical_rotation.py`):
   - Add Phase 3.5 after Phase 2.5 (Categorization)
   - Before Phase 4 (Data Mining)

4. **Update data mining modules**:
   - Query `semantic_hierarchy` instead of `canonical_entities`
   - Use `entity_relationships` for relationship analysis
   - Aggregate by `canonical_groups` Layer 1

### Long-term (Cleanup)
5. **Drop legacy tables** (after verification):
   ```sql
   DROP TABLE IF EXISTS canonical_entities;
   DROP TABLE IF EXISTS entity_mappings;
   DROP TABLE IF EXISTS llm_normalization_cache;
   DROP TABLE IF EXISTS normalized_terms_cache;
   ```

6. **Archive experimental folder**:
   ```bash
   mv back_end/experiments/semantic_normalization back_end/experiments/semantic_normalization.archive
   ```

---

## Support & Questions

**Documentation**: See `PHASE4_CONFIGURATION_COMPLETE.md`, `OPTION_A_IMPLEMENTATION_COMPLETE.md`

**Testing**: Refer to `SESSION_SUMMARY_2025-10-06.md` for test results

**Issues**: Check troubleshooting section above first

---

## Summary

**Migration Status**: ✅ COMPLETE
**System Status**: ✅ PRODUCTION READY
**Legacy System**: DEPRECATED (kept for backward compatibility)
**New System**: Phase 3.5 Hierarchical Semantic Normalization

The migration from Phase 3 to Phase 3.5 provides a more robust, scalable, and feature-rich semantic normalization system with embedding-based similarity, hierarchical structure, and comprehensive relationship classification.
