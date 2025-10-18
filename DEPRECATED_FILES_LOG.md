# Deprecated Files Log (October 18, 2025)

This log tracks files removed during Round 3 Codebase Cleanup.

## Summary

**Total Files Deleted**: 20 files
**Lines of Code Removed**: ~2,500+ lines
**Cleanup Scope**: Deprecated exports, obsolete backups, one-time tests, historical migrations, legacy implementations, duplicate utilities

---

## Deleted Files & Migration Paths

| Category | Deleted File | Replacement | Reason | Date Removed |
|----------|--------------|-------------|--------|--------------|
| **Deprecated Exports** | `back_end/src/utils/export_frontend_data.py` | `back_end/src/phase_5_frontend_export/phase_5_table_view_exporter.py` + `back_end/src/orchestration/phase_5_frontend_updater.py` | Phase 5 automation with atomic writes, backups, validation | Oct 18, 2025 |
| **Deprecated Exports** | `back_end/src/utils/export_network_visualization_data.py` | `back_end/src/phase_5_frontend_export/phase_5_network_viz_exporter.py` + `back_end/src/orchestration/phase_5_frontend_updater.py` | Phase 5 automation with atomic writes, backups, validation | Oct 18, 2025 |
| **Obsolete Backups** | `back_end/src/phase_1_data_collection/database_manager_OLD_BACKUP.py` | `back_end/src/phase_1_data_collection/database_manager.py` | Git version control preserves history | Oct 18, 2025 |
| **Obsolete Backups** | `back_end/src/orchestration/batch_medical_rotation_OLD_BACKUP.py` | `back_end/src/orchestration/batch_medical_rotation.py` | Git version control preserves history | Oct 18, 2025 |
| **Database Backups** | `back_end/data/processed/intervention_research_backup_before_table_drop_20251016_211518.db` | N/A (migration completed) | Migrations completed successfully | Oct 18, 2025 |
| **Database Backups** | `back_end/data/processed/intervention_research_backup_before_table_drop_20251016_211537.db` | N/A (migration completed) | Migrations completed successfully | Oct 18, 2025 |
| **Database Backups** | `back_end/data/processed/intervention_research_backup_20251017_141043.db` | N/A (migration completed) | Migrations completed successfully | Oct 18, 2025 |
| **One-Time Tests** | `back_end/src/phase_3_semantic_normalization/test_imports.py` | N/A (verification complete) | Phase 3 migration verification (Oct 2025) | Oct 18, 2025 |
| **One-Time Tests** | `back_end/src/phase_3_semantic_normalization/test_orchestrator.py` | N/A (verification complete) | Phase 3 orchestrator verification (Oct 2025) | Oct 18, 2025 |
| **One-Time Tests** | `back_end/src/migrations/test_multi_category_api.py` | N/A (verification complete) | Multi-category migration verification (Oct 2025) | Oct 18, 2025 |
| **One-Time Tests** | `back_end/src/utils/test_multi_category_integration.py` | N/A (verification complete) | Multi-category integration verification (Oct 2025) | Oct 18, 2025 |
| **One-Time Tests** | `back_end/src/utils/test_export_multi_category.py` | N/A (verification complete) | Multi-category export verification (Oct 2025) | Oct 18, 2025 |
| **Historical Migrations** | `back_end/src/utils/update_imports_phase_rename.py` | N/A (migration complete) | One-time folder renaming (data_collection → phase_1_data_collection) | Oct 18, 2025 |
| **Historical Migrations** | `back_end/src/utils/update_relative_imports.py` | N/A (migration complete) | One-time import path fixing | Oct 18, 2025 |
| **Historical Migrations** | `back_end/src/utils/update_phase_3_nomenclature.py` | N/A (migration complete) | One-time Phase 3 file renaming | Oct 18, 2025 |
| **Historical Migrations** | `back_end/src/utils/update_file_renames.py` | N/A (migration complete) | One-time file renaming refactoring | Oct 18, 2025 |
| **Deprecated Orchestrators** | `back_end/src/orchestration/rotation_llm_categorization.py` | `back_end/src/phase_3_semantic_normalization/phase_3c_llm_namer.py` | Replaced by clustering-first architecture (Phase 3c) | Oct 18, 2025 |
| **Legacy Data Mining** | `back_end/src/data_mining/medical_knowledge_graph.py` | `back_end/src/phase_4_data_mining/phase_4a_knowledge_graph.py` + `back_end/src/orchestration/phase_4_data_miner.py` | Replaced by Phase 4a with canonical groups integration | Oct 18, 2025 |
| **Legacy Data Mining** | `back_end/src/data_mining/bayesian_scorer.py` | `back_end/src/phase_4_data_mining/phase_4b_bayesian_scorer.py` + `back_end/src/orchestration/phase_4_data_miner.py` | Replaced by Phase 4b with pooled evidence scoring | Oct 18, 2025 |
| **Unused Migrations** | `back_end/src/migrations/create_interventions_view_option_b.py` | N/A (alternative not used) | "Option B" alternative never used in production | Oct 18, 2025 |
| **Duplicate Utilities** | `back_end/src/data_mining/scoring_utils.py` | `back_end/src/phase_4_data_mining/scoring_utils.py` | Consolidation to Phase 4 location | Oct 18, 2025 |
| **Duplicate Utilities** | `back_end/src/data_mining/review_correlations.py` | `back_end/src/utils/review_correlations.py` | Consolidation to utils location | Oct 18, 2025 |

---

## Recovery Instructions

All deleted files are preserved in Git history. To recover any deleted file:

### View Deleted File Contents
```bash
# Replace <filepath> with the full path from the table above
git show HEAD~1:<filepath>
```

### Restore Deleted File
```bash
# Replace <filepath> with the full path from the table above
git checkout HEAD~1 -- <filepath>
```

### Examples
```bash
# View deleted export_frontend_data.py
git show HEAD~1:back_end/src/utils/export_frontend_data.py

# Restore deleted export_frontend_data.py
git checkout HEAD~1 -- back_end/src/utils/export_frontend_data.py

# View deleted medical_knowledge_graph.py
git show HEAD~1:back_end/src/data_mining/medical_knowledge_graph.py

# Restore deleted bayesian_scorer.py
git checkout HEAD~1 -- back_end/src/data_mining/bayesian_scorer.py
```

---

## Migration Guidance

### For Deprecated Export Scripts

**Old Way** (DEPRECATED):
```bash
python -m back_end.src.utils.export_frontend_data
python -m back_end.src.utils.export_network_visualization_data
```

**New Way** (Recommended):
```bash
# Automated export (runs after Phase 4b in pipeline)
python -m back_end.src.orchestration.batch_medical_rotation

# Manual export (if needed)
python -m back_end.src.orchestration.phase_5_frontend_updater
```

**Phase 5 Benefits**:
- Atomic file writes (no corrupted JSON)
- Automatic backups (.bak files)
- Post-export validation
- Session tracking
- Integrated into main pipeline

---

### For Legacy Data Mining Tools

**Old Way** (DEPRECATED):
```bash
# Legacy knowledge graph
python -m back_end.src.data_mining.medical_knowledge_graph

# Legacy Bayesian scoring
python -m back_end.src.data_mining.bayesian_scorer
```

**New Way** (Recommended):
```bash
# Integrated Phase 4 (both 4a + 4b)
python -m back_end.src.orchestration.phase_4_data_miner

# Phase 4a only (knowledge graph)
python -m back_end.src.orchestration.phase_4_data_miner --phase-4a-only

# Phase 4b only (Bayesian scoring)
python -m back_end.src.orchestration.phase_4_data_miner --phase-4b-only
```

**Phase 4 Benefits**:
- Integrates with Phase 3 canonical groups (cleaner nodes)
- Pools evidence across cluster members (better statistical power)
- Database integration with session tracking
- Integrated into main pipeline

---

### For Deprecated Categorization Orchestrator

**Old Way** (DEPRECATED):
```bash
python -m back_end.src.orchestration.rotation_llm_categorization --interventions-only
```

**New Way** (Recommended):
```bash
# Clustering-first architecture (Phase 3c handles categorization)
python -m back_end.src.orchestration.phase_3abc_semantic_normalizer --all
```

**Architecture Change**:
- Old: Naming-first (LLM extraction → grouping)
- New: Clustering-first (embedding → clustering → LLM naming with categories)

---

## Preserved Legacy Tools (NOT DELETED)

The following legacy data_mining tools are preserved for backward compatibility with external analysis scripts:

- `fundamental_functions.py` - Discovers fundamental interventions across unrelated conditions
- `intervention_consensus_analyzer.py` - Analyzes consensus across studies
- `treatment_recommendation_engine.py` - AI treatment recommendations
- `research_gaps.py` - Identifies under-researched areas
- `innovation_tracking_system.py` - Tracks emerging treatments
- `biological_patterns.py` - Mechanism and pattern discovery
- `correlation_consistency_checker.py` - Data quality validation
- `condition_similarity_mapping.py` - Condition similarity matrix
- `power_combinations.py` - Synergistic combination analysis
- `medical_knowledge.py` - Centralized medical domain knowledge
- `similarity_utils.py` - Unified similarity calculations
- `graph_utils.py` - Graph utilities

**Status**: STANDALONE ANALYTICAL TOOLS - Not integrated into main pipeline, kept for reference/research.

---

## Cleanup Impact

### Before Cleanup
- **Total Python Files**: ~120 files
- **Deprecated Exports**: 2 files (~1,079 lines)
- **Obsolete Backups**: 2 code files + 3 database files
- **One-Time Tests**: 5 files (~500 lines)
- **Historical Migrations**: 4 files (~400 lines)
- **Legacy Implementations**: 3 files (~800 lines)
- **Duplicate Utilities**: 2 files (~400 lines)

### After Cleanup
- **Total Files Deleted**: 20 files
- **Lines of Code Removed**: ~2,500+ lines
- **Duplicate Code Eliminated**: ~800 lines
- **Cleaner Architecture**: Phase-based organization without legacy overhead

### Benefits
✅ Reduced codebase complexity
✅ Eliminated duplicate utilities
✅ Clearer migration paths
✅ Improved maintainability
✅ Zero breaking changes to main pipeline

---

*Last Updated: October 18, 2025*
*Cleanup Performed By: Round 3 Codebase Optimization*
*Git Commit: [To be added after commit]*
