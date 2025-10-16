# Removal Checklist: correlation_strength AND extraction_confidence

## Completed ✅

### 1. phase_2_prompt_service.py ✅
- Removed `correlation_strength` from schema definition (line 411)
- Removed from 3 examples (lines 457, 489, 522, 536, 550)
- Removed `extraction_confidence` from schema definition (line 413)
- Removed from 3 examples (lines 459, 490, 522, 536, 550)

**Result**: LLM will NOT extract these fields anymore

---

## In Progress 🔄

### 2. database_manager.py (LARGE FILE - Multiple locations)

#### CREATE TABLE statement (lines 264-268):
**BEFORE:**
```sql
correlation_type TEXT CHECK(...),

-- Dual confidence metrics
extraction_confidence REAL CHECK(...),
study_confidence REAL CHECK(...),
```

**AFTER:**
```sql
correlation_type TEXT CHECK(...),

-- Study confidence metric (for future use)
study_confidence REAL CHECK(...),
```

#### INSERT statement #1 (lines 687-724):
**Column list (line 690-691)**: Remove `extraction_confidence`
**VALUES list (line 696)**: Remove one `?`
**Values tuple (line 706)**: Remove `.get('extraction_confidence')`

#### INSERT statement #2 (lines 1143-1184):
**Column list (line 1146-1147)**: Remove `extraction_confidence`
**VALUES list (line 1153)**: Remove one `?`
**Values tuple (line 1163)**: Remove `.get('extraction_confidence')`

#### migrate_to_dual_confidence() function (lines 153-190):
Rename to `migrate_to_study_confidence()` and remove extraction_confidence migration logic

---

## Pending Files

### 3. validators.py
- Remove `VALID_CONFIDENCE_LEVELS` list (keep only for study_confidence if needed)
- Remove `CONFIDENCE_TO_NUMERIC` mapping for extraction_confidence
- Remove validation block for extraction_confidence (lines 365-379)
- Keep study_confidence validation

### 4. phase_2_single_model_analyzer.py
- Check `_flatten_hierarchical_to_interventions()` - remove extraction_confidence handling
- Check `_validate_and_enhance_interventions()` - remove references

### 5. phase_2_batch_entity_processor.py
- Search and remove extraction_confidence references

### 6. phase_2_entity_operations.py
- Search and remove references

### 7. phase_2_entity_utils.py
- Search and remove references

### 8. export_frontend_data.py
- Remove from SQL SELECT query
- Remove from JSON export dict

### 9. phase_4a_knowledge_graph.py
- Remove if used (likely just passes through)

### 10. phase_2_export_to_json.py
- Remove from export format

### 11. frontend/script.js
- Remove column from DataTables `columns` definition
- Remove from `formatInterventionDetails()` or similar functions
- Remove sorting/filtering logic

### 12. frontend/index.html
- Remove column header `<th>`

### 13. frontend/style.css
- Remove any `.extraction-confidence` or `.correlation-strength` styles

### 14. frontend/data/interventions.json
- Regenerate after backend changes

### 15. drop_correlation_strength_column.py → rename to drop_confidence_columns.py
- Update to drop BOTH columns
- Update table recreation to exclude both
- Update copy statement to exclude both

### 16. CLAUDE.md
- Remove all mentions of correlation_strength and extraction_confidence
- Update intervention schema examples
- Update database schema section
- Update Bayesian scoring documentation

### 17. CORRELATION_STRENGTH_REMOVAL_STATUS.md
- Rename to CONFIDENCE_FIELDS_REMOVAL_STATUS.md
- Update to track BOTH fields
- Mark completed items

---

## SQL to Run (After All Code Changes)

```sql
-- Check current columns
PRAGMA table_info(interventions);

-- Drop both columns (requires table recreation in SQLite)
-- Run the migration script instead
python -m back_end.src.migrations.drop_confidence_columns
```

---

## Testing Steps

1. Run Phase 2 on 1 test paper → should NOT have correlation_strength or extraction_confidence in database
2. Check database schema → columns should be gone
3. Run frontend export → interventions.json should NOT have these fields
4. Load frontend → no errors, columns not displayed
5. Run Phase 4a/4b → should still work
6. Full pipeline end-to-end

---

*Status: 1/17 files completed (6%)*
*Last Updated: October 16, 2025*
