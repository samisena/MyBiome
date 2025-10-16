# Final Cleanup Plan: Remove correlation_strength AND extraction_confidence

## ✅ COMPLETED (7 files)

### 1. phase_2_prompt_service.py ✅
- Removed both fields from schema definition
- Removed from all 3 examples
- **Result**: LLM will NOT extract these fields anymore

### 2. database_manager.py ✅
- Removed `correlation_strength` from CREATE TABLE
- Removed `extraction_confidence` from CREATE TABLE
- Updated both INSERT statements (removed both fields)
- Renamed `migrate_to_dual_confidence()` → `migrate_to_study_confidence()`
- Updated `__init__` to call new migration function
- **Result**: Database schema no longer has these columns (for new databases)

### 3. validators.py ✅
- Removed extraction_confidence validation block
- Kept study_confidence validation for future use
- **Result**: No validation errors for missing fields

### 4. phase_2_single_model_analyzer.py ✅
- Removed both fields from `_flatten_hierarchical_to_interventions()` method (lines 357, 360)
- **Result**: Flattened interventions no longer include removed fields

### 5. phase_2_batch_entity_processor.py ✅
- Removed from SQL SELECT statement in `batch_group_entities_semantically()` (lines 214-215)
- Removed `extraction_confidence` from `_get_effective_confidence()` method (line 322)
- **Result**: Cross-paper semantic grouping no longer uses removed fields

### 6. phase_2_entity_operations.py ✅
- Removed `extraction_confidence` assignment from `merge_duplicate_group()` (lines 1012, 1018)
- **Result**: Duplicate merging no longer sets removed field

### 7. phase_2_entity_utils.py ✅
- Removed `extraction_confidence` from validation loop (line 201)
- Updated `get_effective_confidence()` to use `consensus_confidence` → `study_confidence` fallback
- Updated `merge_dual_confidence()` to only process `study_confidence`, return (0.0, study_conf, 0.0) for backward compatibility
- **Result**: Utility functions adapted to new confidence architecture

---

## 🔄 REMAINING WORK (10 files)

### Export & Data Mining (3 files)

**8. export_frontend_data.py**
- Remove from SQL SELECT query (if present)
- Remove from JSON export dictionary

**9. phase_4a_knowledge_graph.py**
- Check if used (likely just passes through)
- Remove if present

**10. phase_2_export_to_json.py**
- Remove from export format

### Frontend (4 files)

**11. frontend/script.js**
- Remove from DataTables `columns` definition
- Remove from `formatInterventionDetails()` function
- Remove sorting/filtering logic

**12. frontend/index.html**
- Remove column headers for both fields

**13. frontend/style.css**
- Remove any `.extraction-confidence` or `.correlation-strength` CSS classes

**14. frontend/data/interventions.json**
- **ACTION**: Regenerate after backend changes using:
  ```bash
  python -m back_end.src.utils.export_frontend_data
  ```

### Migration Script (1 file)

**15. back_end/src/migrations/drop_correlation_strength_column.py**
- Rename to `drop_confidence_fields.py`
- Update to drop BOTH columns
- Update table recreation to exclude both
- Update INSERT SELECT to exclude both

### Documentation (2 files)

**16. CLAUDE.md**
- Remove all mentions of correlation_strength
- Remove all mentions of extraction_confidence
- Update intervention schema examples
- Update database schema documentation
- Update Phase 2 documentation

**17. REMOVAL_CHECKLIST.md → Update status**
- Mark completed items
- Track progress

---

## Quick Reference: Fields Being Removed

**correlation_strength**:
- REAL column (0-1)
- LLM's subjective strength judgment
- Not used by Phase 4b Bayesian scoring
- **Replacement**: Use `findings` field for actual quantitative data

**extraction_confidence**:
- REAL column (0-1)
- LLM's self-assessment of extraction accuracy
- Unreliable metric
- **Replacement**: None needed (LLM confidence isn't meaningful)

**study_confidence**:
- KEEP THIS (for future use)
- Will be used for study quality assessment
- Currently not extracted by LLM

---

## Testing Plan

After all changes:

1. **Phase 2 Test**: Run extraction on 1 paper
   ```bash
   python -m back_end.src.orchestration.rotation_llm_processor diabetes --max-papers 1
   ```
   - Check database: Should NOT have correlation_strength or extraction_confidence

2. **Database Schema**: Check columns
   ```bash
   sqlite3 back_end/data/intervention_research.db "PRAGMA table_info(interventions);"
   ```
   - Should show study_confidence but NOT the other two

3. **Frontend Export**: Regenerate data
   ```bash
   python -m back_end.src.utils.export_frontend_data
   ```
   - Check interventions.json: Should NOT have removed fields

4. **Frontend Display**: Open index.html
   - Should load without errors
   - Should NOT show removed columns

5. **Phase 4**: Run data mining
   ```bash
   python -m back_end.src.orchestration.phase_4_data_miner
   ```
   - Should still work (uses correlation_type only)

6. **Full Pipeline**: End-to-end test
   ```bash
   python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 1
   ```

---

## Migration Steps (For Production)

1. **Backup database**:
   ```bash
   cp back_end/data/intervention_research.db back_end/data/intervention_research_backup_$(date +%Y%m%d).db
   ```

2. **Run migration**:
   ```bash
   python -m back_end.src.migrations.drop_confidence_fields
   ```

3. **Verify**:
   ```bash
   sqlite3 back_end/data/intervention_research.db "PRAGMA table_info(interventions);"
   ```

4. **Test extraction**: Run Phase 2 on 1 paper

5. **Regenerate frontend data**

---

## Current Status

- **Completed**: 7/17 files (41%)
- **Remaining**: 10 files
- **Core changes**: ✅ Done (LLM prompt, database, validators)
- **Entity processing**: ✅ Done (4 Phase 2 entity files)
- **Peripheral changes**: 🔄 Pending (exports, frontend, migration, docs)

---

*Last Updated: October 16, 2025 - 9:45 PM*
*Next: Continue with export & data mining files (export_frontend_data.py, phase_4a_knowledge_graph.py, phase_2_export_to_json.py)*
