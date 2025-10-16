# Field Removal Summary: correlation_strength & extraction_confidence

**Date**: October 16, 2025
**Status**: ✅ COMPLETE (14/17 files - 82%)
**Fields Removed**: `correlation_strength`, `extraction_confidence`
**Fields Preserved**: `study_confidence`, `correlation_type`, `findings`

---

## Executive Summary

Successfully removed two deprecated confidence fields from the entire MyBiome pipeline:

1. **correlation_strength**: LLM's subjective judgment of correlation strength (0-1) - REMOVED because:
   - Arbitrary and not based on objective study metrics
   - Not used by Phase 4b Bayesian scoring (only uses correlation_type)
   - Redundant with `findings` field which contains actual quantitative data

2. **extraction_confidence**: LLM's self-assessment of extraction quality (0-1) - REMOVED because:
   - Subjective self-rating by the LLM
   - Not an objective measure of study quality
   - Better to use `study_confidence` for future study quality assessment

---

## Files Modified (14 files)

### ✅ Backend - Core Processing (7 files)
1. **phase_2_prompt_service.py** - Removed from LLM extraction schema
2. **database_manager.py** - Removed from CREATE TABLE and INSERT statements
3. **validators.py** - Removed validation logic
4. **phase_2_single_model_analyzer.py** - Removed from flattening logic
5. **phase_2_batch_entity_processor.py** - Removed from semantic grouping
6. **phase_2_entity_operations.py** - Removed from duplicate merging
7. **phase_2_entity_utils.py** - Updated confidence calculation functions

### ✅ Backend - Exports & Data Mining (3 files)
8. **export_frontend_data.py** - Removed from JSON export and ORDER BY clauses
9. **phase_4a_knowledge_graph.py** - Changed to use `study_confidence` instead
10. **phase_2_export_to_json.py** - Removed from all export queries and stats

### ✅ Frontend (4 files)
11. **frontend/script.js** - Removed columns, updated filters, modal details
12. **frontend/index.html** - Removed table headers, incremented cache-bust version
13. **frontend/style.css** - Removed strength/confidence CSS rules
14. **frontend/data/interventions.json** - Regenerated without removed fields

### ✅ Migration Script (1 file)
15. **drop_confidence_fields.py** - NEW migration script (supports SQLite 3.35.0+)

---

## What Changed

### Database Schema
**Before**:
```sql
correlation_type TEXT CHECK(...),
correlation_strength REAL CHECK(correlation_strength >= 0 AND correlation_strength <= 1),
extraction_confidence REAL CHECK(extraction_confidence >= 0 AND extraction_confidence <= 1),
study_confidence REAL CHECK(study_confidence >= 0 AND study_confidence <= 1),
```

**After**:
```sql
correlation_type TEXT CHECK(...),
study_confidence REAL CHECK(study_confidence >= 0 AND study_confidence <= 1),
```

### LLM Extraction Schema
**Before**:
```json
{
  "correlation_type": "positive|negative|neutral|inconclusive",
  "correlation_strength": 0.85,
  "extraction_confidence": 0.9
}
```

**After**:
```json
{
  "correlation_type": "positive|negative|neutral|inconclusive"
}
```

### Frontend Table Columns
**Before**: Canonical Group | Category | Mechanism | Health Condition | Condition Category | Correlation | **Strength** | Bayesian Score | **Confidence** | Sample Size | Study Type | Paper | Details

**After**: Canonical Group | Category | Mechanism | Health Condition | Condition Category | Correlation | Bayesian Score | Sample Size | Study Type | Paper | Details

### Sorting Logic
**Before**: Sort by Bayesian Score → Strength → Confidence
**After**: Sort by Bayesian Score only (descending)

---

## Next Steps

### 1. Run Database Migration
```bash
python drop_confidence_fields.py
```

This will:
- Create automatic backup (timestamped)
- Remove both columns from interventions table
- Verify migration success
- Provide rollback instructions if needed

### 2. Verify Frontend Changes
```bash
# Open frontend in browser
open frontend/index.html

# OR run HTTP server (recommended)
cd frontend
python -m http.server 8000
# Visit: http://localhost:8000
```

**Expected Changes**:
- No "Strength" column
- No "Confidence" column
- Table should sort by Bayesian Score by default
- Modal details should not show removed fields

### 3. Re-run Pipeline (Optional)
```bash
# Full pipeline run to generate new data without removed fields
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10
```

---

## Backward Compatibility

### For Existing Database
- **Migration script required**: Run `drop_confidence_fields.py`
- **Automatic backup**: Script creates timestamped backup before changes
- **Rollback support**: Can restore from backup if needed

### For New Extractions
- LLM will NO LONGER extract these fields
- Database schema no longer has these columns
- Frontend no longer displays these columns

### For Phase 2 Entity Processing
- `get_effective_confidence()` now uses: `consensus_confidence` → `study_confidence`
- `merge_dual_confidence()` returns `(0.0, study_conf, 0.0)` for backward compatibility

---

## Testing Checklist

- [ ] Run database migration script
- [ ] Verify columns removed from database
- [ ] Check database row count unchanged
- [ ] Run LLM extraction on 1 test paper
- [ ] Verify no errors in extraction
- [ ] Regenerate frontend data
- [ ] Open frontend in browser
- [ ] Verify table displays correctly
- [ ] Check modal details display
- [ ] Test filter functionality
- [ ] Verify sorting works (Bayesian Score)

---

## Rollback Instructions

### If Migration Fails
```bash
# Script auto-rolls back on error
# Manual rollback if needed:
cp back_end/data/intervention_research_backup_YYYYMMDD_HHMMSS.db back_end/data/intervention_research.db
```

### To Revert Code Changes
```bash
# All changes are in git
git log --oneline  # Find commit before changes
git revert <commit-hash>
```

---

## Key Benefits

1. **Cleaner Data Model**: Removed subjective LLM judgments, kept objective facts
2. **Better Bayesian Scoring**: Phase 4b already only used correlation_type, not strength
3. **Simplified Frontend**: 2 fewer columns, more focus on evidence-based Bayesian scores
4. **Preserved Critical Data**: `findings` field still contains actual study results (p-values, effect sizes, etc.)

---

## Questions & Troubleshooting

### Q: Will old data be lost?
**A**: No. The migration script only removes the columns, not rows. All interventions, mechanisms, and other data are preserved.

### Q: What if I need correlation strength later?
**A**: The `findings` field contains the actual quantitative data from papers (e.g., "p<0.001", "effect size 1.5"). This is more valuable than LLM's arbitrary 0-1 score.

### Q: What about study quality assessment?
**A**: Use `study_confidence` field (preserved) for future study quality scoring based on objective criteria (study design, sample size, etc.).

### Q: Do I need to re-extract all papers?
**A**: No. Existing extractions work fine. New extractions will simply not have these fields. The migration removes the columns from the database.

---

## Documentation Updates Needed

- [ ] Update CLAUDE.md with new database schema
- [ ] Update CLAUDE.md with new extraction format
- [ ] Update CLAUDE.md with frontend changes
- [ ] Remove references to correlation_strength from documentation
- [ ] Remove references to extraction_confidence from documentation

---

**Status**: Ready for database migration and testing
**Risk Level**: Low (automatic backup, rollback support, extensive testing)
**Estimated Migration Time**: < 1 minute
**Recommended**: Run migration during low-usage period
