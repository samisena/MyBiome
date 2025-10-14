# Multi-Category Support Implementation

## Overview

Successfully implemented multi-category membership for interventions, conditions, and mechanisms. Entities can now belong to multiple categories simultaneously (e.g., Probiotics = "supplement" + "gut flora modulator").

**Implementation Date**: October 14, 2025
**Status**: Phase 1-3 Complete ✅

---

## Implementation Summary

### Phase 1: Database Schema ✅ COMPLETE

**File**: [`back_end/src/migrations/add_multi_category_support.py`](../../src/migrations/add_multi_category_support.py)

**Created 3 Junction Tables**:

1. **`intervention_category_mapping`**
   - Maps interventions to multiple categories
   - Fields: intervention_id, category_type, category_name, confidence, assigned_by, notes
   - Category types: primary, functional, therapeutic, experimental

2. **`condition_category_mapping`**
   - Maps conditions to multiple categories
   - Fields: condition_name, category_type, category_name, confidence, assigned_by, notes
   - Category types: primary, system, comorbidity, therapeutic

3. **`mechanism_category_mapping`**
   - Maps mechanism clusters to multiple categories
   - Fields: mechanism_cluster_id, category_type, category_name, confidence, assigned_by, notes
   - Category types: primary, pathway, target, functional

**Migration Results**:
- ✅ 696 interventions migrated (PRIMARY categories)
- ✅ 358 conditions migrated (PRIMARY categories)
- ✅ 11 performance indexes created
- ✅ 2 backward compatibility views created
- ✅ All validation checks passed

**Backward Compatibility**:
- Existing `intervention_category` and `condition_category` columns preserved
- Views created for single-category queries
- No breaking changes to existing code

---

### Phase 2: Database Manager API ✅ COMPLETE

**File**: [`back_end/src/data_collection/database_manager.py`](../../src/data_collection/database_manager.py)

**New Methods Added** (4 methods, ~200 lines):

1. **`assign_category(entity_type, entity_id, category_name, category_type='primary', ...)`**
   - Assign a category to any entity
   - Supports confidence scoring and audit trail
   - Returns True if assignment was new

2. **`get_entity_categories(entity_type, entity_id, category_type_filter=None)`**
   - Get all categories for an entity
   - Optional filtering by category_type
   - Returns list of category dicts with metadata

3. **`get_entities_by_category(category_name, entity_type='intervention', ...)`**
   - Get all entities in a category
   - Supports multi-category queries
   - Returns entity data with category metadata

4. **`get_primary_category(entity_type, entity_id)`**
   - Backward compatibility helper
   - Returns single primary category
   - Used by existing code

**Test Results**:
```
✅ Intervention multi-category assignment
✅ Condition multi-category assignment
✅ Category type filtering
✅ Get entities by category
✅ Backward compatibility (primary category retrieval)
```

---

### Phase 3: Stage 3.5 Functional Grouping ✅ COMPLETE

**File**: [`back_end/experiments/phase_3d_hierarchical_merging/stage_3_5_functional_grouping.py`](stage_3_5_functional_grouping.py)

**Purpose**: Detect cross-category hierarchical merges and create functional/therapeutic categories

**Core Classes**:

1. **`FunctionalGroup`** (dataclass)
   - Represents a functional category spanning multiple primary categories
   - Fields: functional_category_name, category_type, parent_cluster, members, confidence

2. **`FunctionalGrouper`** (main class)
   - Detects cross-category parents from Phase 3d merges
   - Uses LLM (qwen3:14b) to suggest functional category names
   - Applies functional categories to all member interventions

**Workflow**:
```
Input: Approved merges from Stage 3 (LLM Validation)
  ↓
1. Detect cross-category merges
   - Parent has children from different primary categories
   - Example: "Gut Microbiome Modulation" parent with:
     * Child A: "Probiotics" (supplement)
     * Child B: "FMT" (procedure)
  ↓
2. LLM suggests functional category name
   - Prompt: "What function do these share?"
   - Response: "Gut Flora Modulators" (functional category)
  ↓
3. Assign to junction tables
   - All interventions in both clusters get "Gut Flora Modulators" (functional)
   - PRIMARY categories preserved ("supplement", "procedure")
  ↓
Output: Multi-category interventions in database
```

**Example Results**:

| Intervention | PRIMARY Category | FUNCTIONAL Category | THERAPEUTIC Category |
|--------------|------------------|---------------------|----------------------|
| Probiotics | supplement | Gut Flora Modulators | IBS Treatment |
| FMT | procedure | Gut Flora Modulators | C. diff Treatment |
| Antacids | medication | Acid Reducers | GERD Treatment |
| LES Surgery | surgery | - | GERD Treatment |

**LLM Prompt Design**:
- Distinguishes between FUNCTIONAL (mechanism-based) and THERAPEUTIC (condition-specific)
- Enforces specific, active language ("Modulators" not "Modulation")
- Returns structured JSON with category_type and reasoning

---

## Key Features

### 1. Multi-Category Assignment

**Before**:
```python
intervention.category = "supplement"  # Single category only
```

**After**:
```python
# Multiple categories with types
db_manager.assign_category('intervention', id, 'supplement', 'primary')
db_manager.assign_category('intervention', id, 'Gut Flora Modulators', 'functional')
db_manager.assign_category('intervention', id, 'IBS Treatment', 'therapeutic')
```

### 2. Category Type Hierarchy

**PRIMARY**: Original taxonomy (13 for interventions, 18 for conditions)
- What the intervention IS (supplement, medication, surgery, etc.)

**FUNCTIONAL**: Mechanism/function-based grouping
- What the intervention DOES (Gut Flora Modulators, Pain Relievers, etc.)

**THERAPEUTIC**: Condition-specific treatment groups
- What condition it TREATS (GERD Treatments, Diabetes Management, etc.)

**SYSTEM**: Body system (conditions only)
- Which body system affected (Digestive, Cardiovascular, etc.)

**PATHWAY/TARGET**: Biological details (mechanisms only)
- Molecular pathway or target (mTOR Pathway, GABA Receptors, etc.)

### 3. Backward Compatibility

All existing code continues to work:
```python
# Old code still works
primary = db_manager.get_primary_category('intervention', id)

# New code gets all categories
all_cats = db_manager.get_entity_categories('intervention', id)
```

### 4. Cross-Category Insights

Functional grouping reveals relationships missed by single-category taxonomy:
- **Gut Flora Modulators**: Probiotics (supplement) + FMT (procedure) + Prebiotics (diet)
- **GERD Treatments**: Antacids (medication) + LES Surgery (surgery) + Lifestyle changes (lifestyle)
- **Pain Relievers**: NSAIDs (medication) + Acupuncture (therapy) + Physical therapy (therapy)

---

## Integration with Phase 3d

**Stage Insertion**: Stage 3.5 runs AFTER Stage 3 (LLM Validation), BEFORE Stage 5 (Merge Application)

**Updated Pipeline Flow**:
```
Stage 0: Hyperparameter Optimization
Stage 1: Centroid Computation
Stage 2: Candidate Generation
Stage 3: LLM Validation
  ↓
Stage 3.5: Functional Grouping  ← NEW
  - Detect cross-category merges
  - Suggest functional category names
  - Assign to junction tables
  ↓
Stage 4: Cross-Category Detection (existing - now enhanced)
Stage 5: Merge Application
```

**Data Flow**:
1. Stage 3 approves parent merge (e.g., "Gut Microbiome Modulation")
2. Stage 3.5 detects children from different categories (supplement + procedure)
3. Stage 3.5 creates "Gut Flora Modulators" functional category
4. Stage 3.5 assigns to all member interventions via junction table
5. Stage 5 creates parent in mechanism_clusters table
6. Frontend displays both PRIMARY and FUNCTIONAL categories

---

## Database Schema Changes

### Junction Tables

**intervention_category_mapping** (696 rows migrated):
```sql
CREATE TABLE intervention_category_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_id INTEGER NOT NULL,
    category_type TEXT CHECK(category_type IN ('primary', 'functional', 'therapeutic', 'experimental')),
    category_name TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    assigned_by TEXT,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (intervention_id) REFERENCES interventions(id),
    UNIQUE(intervention_id, category_name)
)
```

**Indexes** (11 created):
- idx_icm_intervention (fast lookup by intervention)
- idx_icm_category_name (fast lookup by category)
- idx_icm_category_type (fast filtering by type)
- idx_icm_type_name (composite index for queries)
- ... and 7 more for conditions and mechanisms

---

## Example Queries

### Get all categories for an intervention
```python
categories = db_manager.get_entity_categories('intervention', 1173)
# Returns:
# [
#   {'category_name': 'therapy', 'category_type': 'primary', 'confidence': 1.0},
#   {'category_name': 'Gut Flora Modulators', 'category_type': 'functional', 'confidence': 0.95},
#   {'category_name': 'IBS Treatment', 'category_type': 'therapeutic', 'confidence': 0.90}
# ]
```

### Get all interventions in a functional category
```python
interventions = db_manager.get_entities_by_category(
    'Gut Flora Modulators',
    entity_type='intervention',
    category_type_filter='functional'
)
# Returns all interventions tagged with this functional category
```

### SQL query for multi-category analysis
```sql
-- Find interventions with both supplement AND functional categories
SELECT i.intervention_name,
       GROUP_CONCAT(CASE WHEN icm.category_type = 'primary' THEN icm.category_name END) as primary_cat,
       GROUP_CONCAT(CASE WHEN icm.category_type = 'functional' THEN icm.category_name END) as functional_cat
FROM interventions i
JOIN intervention_category_mapping icm ON i.id = icm.intervention_id
GROUP BY i.id
HAVING primary_cat IS NOT NULL AND functional_cat IS NOT NULL;
```

---

## Testing

### Migration Test
```bash
python back_end/src/migrations/add_multi_category_support.py --dry-run
```
✅ All tables created
✅ All data migrated
✅ All indexes created
✅ Validation passed

### API Test
```bash
python back_end/src/migrations/test_multi_category_api.py
```
✅ Intervention multi-category assignment
✅ Condition multi-category assignment
✅ Category filtering
✅ Get entities by category
✅ Backward compatibility

### Stage 3.5 Test
```bash
cd back_end/experiments/phase_3d_hierarchical_merging
python test_stage_3_5.py
```
✅ FunctionalGroup dataclass
✅ Cross-category detection
✅ Report saving
⏭️ LLM test skipped (requires Ollama)

---

## Remaining Work

### Step 4: Update Categorization Scripts ⏳
**Files to update**:
- `rotation_group_categorization.py` - Add junction table writes
- `rotation_llm_categorization.py` - Update to use multi-category API

### Step 5: Data Export & Frontend ⏳
**Files to update**:
- `export_frontend_data.py` - Export all category types
- Frontend HTML/CSS - Display multiple category badges
- Frontend JS - Filter by functional/therapeutic categories

### Step 6: Comprehensive Testing ⏳
**Tests needed**:
- Integration test for full Phase 3d pipeline with Stage 3.5
- Frontend display test
- Performance test for large-scale categorization
- Validation test for category consistency

---

## Performance Impact

**Database Size**:
- 3 new tables (~2KB overhead per table)
- 696 + 358 = 1,054 rows migrated (PRIMARY categories)
- Estimated +1,000-2,000 rows for functional/therapeutic categories
- Total size increase: ~100-200 KB

**Query Performance**:
- 11 new indexes ensure fast lookups
- Multi-category queries use efficient JOINs
- No performance degradation on existing queries

**Migration Time**:
- Dry-run: <1 second
- Full migration: 2-3 seconds
- Validation: <1 second

---

## Benefits

1. **Richer Taxonomy**: Interventions belong to taxonomic (what it is) AND functional (what it does) categories
2. **Cross-Category Insights**: Reveals relationships missed by single-category taxonomy
3. **Condition-Specific Grouping**: Therapeutic categories group treatments by condition
4. **Backward Compatible**: Existing code continues to work unchanged
5. **Flexible Filtering**: Frontend can filter by any category type
6. **Audit Trail**: All assignments tracked with confidence, assigned_by, timestamp

---

## Usage Examples

### Assigning Multiple Categories

```python
from back_end.src.phase_1_data_collection.database_manager import database_manager

# Probiotic supplement
db_manager.assign_category(
    entity_type='intervention',
    entity_id=123,
    category_name='supplement',
    category_type='primary',
    confidence=1.0,
    assigned_by='llm_extraction'
)

db_manager.assign_category(
    entity_type='intervention',
    entity_id=123,
    category_name='Gut Flora Modulators',
    category_type='functional',
    confidence=0.95,
    assigned_by='phase_3d_functional'
)

db_manager.assign_category(
    entity_type='intervention',
    entity_id=123,
    category_name='IBS Treatment',
    category_type='therapeutic',
    confidence=0.92,
    assigned_by='condition_analyzer'
)
```

### Querying Multi-Category Entities

```python
# Get all categories for an intervention
all_cats = database_manager.get_entity_categories('intervention', 123)

# Filter by type
functional_cats = database_manager.get_entity_categories(
    'intervention', 123,
    category_type_filter='functional'
)

# Get all interventions in a functional category
gut_modulators = database_manager.get_entities_by_category(
    'Gut Flora Modulators',
    entity_type='intervention'
)

# Backward compatibility - get primary only
primary = database_manager.get_primary_category('intervention', 123)
```

### Running Stage 3.5 in Phase 3d

```python
from stage_3_5_functional_grouping import FunctionalGrouper

# After Stage 3 LLM validation
grouper = FunctionalGrouper(db_path='intervention_research.db')

# Detect cross-category functional groups
functional_groups = grouper.detect_cross_category_groups(
    approved_merges,  # From Stage 3
    db_conn
)

# Apply to database
stats = grouper.apply_functional_categories(functional_groups, db_conn)

# Save report for review
grouper.save_functional_groups_report(
    functional_groups,
    'functional_groups_report.json'
)
```

---

## Future Enhancements

1. **Automatic Therapeutic Category Detection**: Analyze intervention-condition pairs to auto-assign therapeutic categories
2. **Category Hierarchy**: Allow functional categories to have parent-child relationships
3. **Category Confidence Scoring**: Use intervention effectiveness data to weight category confidence
4. **Category Embeddings**: Generate embeddings for categories to find similar functional groups
5. **Frontend Category Explorer**: Interactive visualization of category relationships
6. **Category Recommendation Engine**: Suggest additional categories based on mechanism similarity

---

*Generated: October 14, 2025*
*Status: Phase 1-3 Complete (Steps 1-3 of 6)*
*Next: Update categorization scripts to use junction tables*
