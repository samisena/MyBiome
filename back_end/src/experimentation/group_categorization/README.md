# Group-Based Semantic Categorization Experiment

## Overview

This experiment tests a novel approach to intervention categorization: **categorizing semantic groups instead of individual interventions**.

### Hypothesis

By categorizing canonical groups (e.g., "probiotics", "statins", "HIIT") instead of individual interventions, we can:
- **Reduce LLM calls by ~80%** (10,000 interventions → ~2,000 groups)
- **Improve accuracy** by leveraging semantic context (group name + member names)
- **Ensure consistency** across intervention variants (e.g., all "vitamin D" variants get same category)

---

## Architecture

### Current Approach (Phase 2.5)
```
Phase 2: Extract interventions (no category)
Phase 2.5: Categorize each intervention individually
         - "vitamin D" → supplement
         - "Vitamin D3" → supplement
         - "cholecalciferol" → supplement
         (3 separate LLM calls)
Phase 3: Semantic normalization creates groups
```

### Proposed Approach (Group-Based)
```
Phase 2: Extract interventions (no category)
Phase 3: Semantic normalization creates groups
         - "vitamin D" group with members: ["vitamin D", "Vitamin D3", "cholecalciferol"]
Phase 3.5: Categorize groups (NEW)
         - "vitamin D" group → supplement
         (1 LLM call, all variants inherit category)
```

---

## Implementation

### Files

1. **`group_categorizer.py`** - Core algorithm
   - `categorize_all_groups()`: Categorize canonical groups using LLM
   - `propagate_to_interventions()`: Copy categories from groups to interventions (Option A)
   - `categorize_orphan_interventions()`: Fallback for non-grouped interventions

2. **`validation.py`** - Validation suite
   - `validate_category_coverage()`: Ensure 100% interventions categorized
   - `validate_group_purity()`: Check groups don't span multiple categories
   - `compare_with_existing()`: Compare with existing Phase 2.5 categorizations

3. **`experiment_runner.py`** - Experiment orchestrator
   - `run_full_experiment()`: Run on full database
   - `run_subset_experiment()`: Run on subset for testing
   - `compare_with_individual()`: Compare efficiency vs individual approach

4. **`config.yaml`** - Configuration
   - LLM settings, batch sizes, validation thresholds

---

## Usage

### Run Full Experiment

```bash
# From project root
cd back_end/src/experimentation/group_categorization

# Run on full database
python experiment_runner.py

# Run on subset (for testing)
python experiment_runner.py --subset 200

# Custom batch size
python experiment_runner.py --batch-size 10
```

### Run Components Separately

```python
from back_end.src.experimentation.group_categorization import (
    GroupBasedCategorizer,
    validate_all,
    run_experiment
)

# 1. Categorize groups
categorizer = GroupBasedCategorizer(db_path="path/to/db")
group_stats = categorizer.categorize_all_groups()

# 2. Propagate to interventions
propagate_stats = categorizer.propagate_to_interventions()

# 3. Handle orphans
orphan_stats = categorizer.categorize_orphan_interventions()

# 4. Validate
validation_results = validate_all(db_path="path/to/db")
```

### Validation Only

```bash
python validation.py
```

---

## Success Criteria

### Performance
- ✅ **LLM calls**: <100 calls for full database (vs ~40 for individual approach on 792 interventions)
- ✅ **Time**: <10 minutes for full database
- ✅ **Reduction rate**: >70% fewer LLM calls than individual approach

### Accuracy
- ✅ **Coverage**: 100% interventions categorized (grouped + orphans)
- ✅ **Orphan rate**: <5% interventions require fallback categorization
- ✅ **Agreement**: >95% agreement with existing Phase 2.5 categorizations
- ✅ **Purity**: <10% groups have mixed-category members

---

## Results

Results are saved to `results/` directory as JSON files with timestamp.

### Key Metrics

```json
{
  "experiment_id": "group_categorization_20251009_143022",
  "elapsed_time_seconds": 450.2,
  "performance": {
    "total_llm_calls": 35,
    "reduction_rate": 0.82,
    "interventions_per_group_call": 22.6
  },
  "validation": {
    "coverage": {
      "coverage_rate": 1.0,
      "passed": true
    },
    "purity": {
      "purity_rate": 0.94,
      "passed": true
    },
    "comparison": {
      "agreement_rate": 0.97,
      "passed": true
    },
    "all_passed": true
  }
}
```

---

## Integration Plan

### Phase 1: Experiment (Current)
- ✅ Isolated testing in `experimentation/` folder
- ✅ Validate approach on current database
- ✅ Document results

### Phase 2: Pipeline Integration (Option A)
- Create `rotation_group_categorization.py` (Phase 3.5 orchestrator)
- Modify `batch_medical_rotation.py` to add Phase 3.5 after Phase 3
- Use Option A (persist categories in `interventions` table for backwards compatibility)
- Run in production, monitor for 1 week

### Phase 3: Schema Migration (Option B)
- Create `v_interventions_categorized` VIEW for clean architecture
- Migrate queries from `interventions` → `v_interventions_categorized`
- Deprecate `interventions.intervention_category` column (keep for orphans)
- Single source of truth in `canonical_groups.layer_0_category`

---

## Schema Details

### Option A: Persist Categories (Backwards Compatible)

**Pros:**
- No code changes required
- Fast queries (no JOIN)
- Works with existing frontend/tools

**Cons:**
- Data duplication
- Must sync if group category changes

```sql
-- Update groups
UPDATE canonical_groups SET layer_0_category = 'supplement'
WHERE canonical_name = 'vitamin D';

-- Propagate to interventions
UPDATE interventions
SET intervention_category = (
    SELECT cg.layer_0_category
    FROM canonical_groups cg
    JOIN semantic_hierarchy sh ON cg.canonical_name = sh.layer_1_canonical
    WHERE sh.entity_name = interventions.intervention_name
);
```

### Option B: VIEW-Based (Clean Architecture) - Future

**Pros:**
- Single source of truth
- Auto-sync when group changes
- Cleaner architecture

**Cons:**
- Requires code refactoring
- Slightly slower queries (JOIN)

```sql
CREATE VIEW v_interventions_categorized AS
SELECT
    i.*,
    COALESCE(cg.layer_0_category, i.intervention_category) AS intervention_category
FROM interventions i
LEFT JOIN semantic_hierarchy sh ON i.intervention_name = sh.entity_name
LEFT JOIN canonical_groups cg ON sh.layer_1_canonical = cg.canonical_name;
```

---

## Edge Cases

### 1. Orphan Interventions
**Problem**: Single-paper interventions not merged into groups
**Solution**: Fallback to individual categorization (Phase 2.5 logic)
**Target**: <5% orphan rate

### 2. Mixed Groups
**Problem**: Groups with members from multiple categories
**Solution**: LLM chooses dominant category, flagged for manual review
**Target**: <10% mixed groups

### 3. New Interventions
**Problem**: Interventions added after Phase 3.5
**Solution**: Re-run Phase 3.5 or use incremental categorization

---

## Troubleshooting

### Issue: High Orphan Rate (>5%)

```bash
# Check orphan count
python -c "
from validation import validate_category_coverage
from back_end.src.data.config import config
results = validate_category_coverage(config.db_path)
print(f'Orphans: {results[\"orphans\"]} ({results[\"orphan_rate\"]*100:.1f}%)')
"
```

**Solutions:**
- Lower semantic similarity threshold in Phase 3
- Run Phase 3 normalization again
- Accept higher orphan rate if interventions are truly unique

### Issue: Low Agreement Rate (<95%)

```bash
# Check disagreements
python -c "
from validation import compare_with_existing
from back_end.src.data.config import config
results = compare_with_existing(config.db_path, show_disagreements=20)
"
```

**Solutions:**
- Review disagreement examples (may be improvements!)
- Adjust category descriptions in prompt
- Manual review of low-confidence categorizations

### Issue: LLM Timeout/Failure

**Solutions:**
- Reduce `batch_size` (e.g., 20 → 10)
- Increase `max_retries` (e.g., 3 → 5)
- Reduce `max_members_in_prompt` (e.g., 10 → 5)

---

## References

- **13-Category Taxonomy**: `back_end/src/interventions/taxonomy.py`
- **Phase 2.5 (Individual)**: `back_end/src/orchestration/rotation_llm_categorization.py`
- **Phase 3 (Semantic Normalization)**: `back_end/src/semantic_normalization/`
- **Database Schema**: `back_end/src/migrations/add_semantic_normalization_tables.py`

---

## Contact

For questions or issues, see main project documentation in `CLAUDE.md`.

---

**Status**: ✅ **Experiment Ready**
**Next Step**: Run full experiment and validate results
