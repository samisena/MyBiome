# Correlation_Strength Removal Status

## Completed ✅

1. **back_end/src/phase_2_llm_processing/phase_2_prompt_service.py** ✅
   - Removed from schema definition (line 411)
   - Removed from 3 examples (lines 457, 489, 522, 536, 550)

2. **back_end/src/phase_1_data_collection/database_manager.py** ✅
   - Removed from CREATE TABLE statement (line 265)
   - Removed from INSERT statement #1 (lines 690, 706)
   - Removed from INSERT statement #2 (lines 1146, 1163)

3. **back_end/src/data/validators.py** ✅
   - Removed VALID_STRENGTH_LEVELS constant (line 250)
   - Removed STRENGTH_TO_NUMERIC mapping (lines 261-267)
   - Removed validation logic (lines 374-388)

4. **back_end/src/migrations/drop_correlation_strength_column.py** ✅
   - Created migration script to drop column from database

---

## Remaining Files (16 total)

### Critical Backend Files (Need Immediate Attention)

**5. back_end/src/utils/export_frontend_data.py** 🔴
   - Likely exports correlation_strength to frontend JSON
   - Action: Remove from export query/logic

**6. back_end/src/phase_4_data_mining/phase_4a_knowledge_graph.py** 🔴
   - May use correlation_strength in graph construction
   - Action: Remove references, use correlation_type only

**7. back_end/src/phase_2_llm_processing/phase_2_export_to_json.py** 🔴
   - Exports intervention data to JSON
   - Action: Remove correlation_strength from export

**8. back_end/src/phase_2_llm_processing/phase_2_batch_entity_processor.py** 🔴
   - Entity processing logic
   - Action: Remove any correlation_strength handling

**9. back_end/src/utils/run_correlation_extraction.py** 🔴
   - Utility script for correlation extraction
   - Action: Remove correlation_strength logic

**10. back_end/src/utils/review_correlations.py** 🔴
   - Review utility
   - Action: Remove correlation_strength display/logic

**11. back_end/src/phase_3_semantic_normalization/phase_3d/stage_5_merge_application.py** 🟡
   - Phase 3d merge logic
   - Action: Check if correlation_strength is used, remove if present

**12. back_end/src/orchestration/phase_3c_mechanism_clustering.py** 🟡
   - Mechanism clustering orchestrator
   - Action: Check for references, remove if any

### Frontend Files (Need Immediate Attention)

**13. frontend/script.js** 🔴 (NOT in grep results - need to search)
   - Main frontend JavaScript
   - Action: Remove correlation_strength display logic, remove column from DataTables

**14. frontend/index.html** 🔴 (NOT in grep results - need to search)
   - Main HTML file
   - Action: Remove correlation_strength column header

**15. frontend/style.css** 🔴 (NOT in grep results - need to search)
   - Styling
   - Action: Remove any correlation_strength-specific styles

**16. frontend/data/correlations.json** 🟡
   - Static data file
   - Action: Regenerate after backend changes complete (no code changes needed)

### Legacy/Experimental Files (Lower Priority)

**17-18. back_end/data/experimentation/temperature_experiment_41038680_*.json|md** ⚪
   - Old experiment results (4 files total)
   - Action: Leave as historical record (no changes needed)

**19. back_end/src/migrations/create_interventions_view_option_b.py** ⚪
   - Old migration script
   - Action: Check if still used; if deprecated, leave as-is or delete

**20. back_end/scripts/migrate_intervention_category_nullable.py** ⚪
   - Old migration script
   - Action: Check if still used; if deprecated, leave as-is

**21. back_end/src/data_mining/intervention_consensus_analyzer.py** ⚪
   - Legacy data mining file (likely not used with new Phase 4 architecture)
   - Action: Low priority - remove references if time permits

### Documentation

**22. CLAUDE.md** 🔴
   - Main project documentation
   - Action: Remove all correlation_strength references, update examples

---

## Next Steps (Priority Order)

1. **Export Scripts** (Files 5, 7) - Fix export_frontend_data.py and phase_2_export_to_json.py
2. **Phase 4 Integration** (File 6) - Remove from phase_4a_knowledge_graph.py
3. **Frontend Files** (Files 13-15) - Search for and remove from script.js, index.html, style.css
4. **Utility Scripts** (Files 8-10) - Clean up phase_2_batch_entity_processor.py, run_correlation_extraction.py, review_correlations.py
5. **Documentation** (File 22) - Update CLAUDE.md
6. **Phase 3 Files** (Files 11-12) - Remove from phase_3d and orchestration files (low risk)
7. **Regenerate Data** (File 16) - Regenerate frontend/data/correlations.json after all backend changes
8. **Run Migration** - Execute drop_correlation_strength_column.py script

---

## Testing Checklist

After all changes:
- [ ] Run Phase 2 extraction on 1 test paper
- [ ] Verify no correlation_strength in database
- [ ] Run frontend export script
- [ ] Check frontend displays without errors
- [ ] Run Phase 4a and 4b
- [ ] Verify Bayesian scoring still works
- [ ] Run full pipeline end-to-end

---

## Rollback Plan

If issues arise:
- Database backup available at: `back_end/data/intervention_research_backup_before_drop_correlation_strength_YYYYMMDD_HHMMSS.db`
- Git commit before changes: [To be filled in]
- Restore files from git: `git checkout HEAD^ <file_path>`

---

*Last Updated: October 16, 2025*
*Status: 3/22 files completed (14%)*
