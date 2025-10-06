# Hierarchical Semantic Normalization - Implementation Complete

## Summary

Successfully implemented a **multi-layer hierarchical normalization system** to address the limitation of binary match/no-match classification for medical interventions.

## ✅ Implementation Status

### 1. Configuration System
**File**: [config/config.yaml](config/config.yaml)

**Features**:
- ✅ 6 relationship types defined (EXACT_MATCH, VARIANT, SUBTYPE, SAME_CATEGORY, DOSAGE_VARIANT, DIFFERENT)
- ✅ Aggregation rules per relationship type
- ✅ Layer definitions (Layer 0-3) with thresholds
- ✅ Examples and descriptions for each relationship type

**Key Configuration**:
```yaml
relationship_types:
  1: EXACT_MATCH (merge_completely)
  2: VARIANT (share_layer_1_link_layer_2)
  3: SUBTYPE (share_layer_1_separate_layer_2)
  4: SAME_CATEGORY (separate_all_layers)
  5: DOSAGE_VARIANT (share_layers_1_2)
  6: DIFFERENT (no_relationship)

hierarchy:
  similarity_thresholds:
    layer_1_canonical: 0.70  # Broad grouping
    layer_2_variant: 0.90    # Specific matching
    layer_3_detail: 0.95     # Exact matching
```

---

### 2. Hierarchical Labeling Interface
**File**: [core/labeling_interface.py](core/labeling_interface.py)

**Features**:
- ✅ Interactive relationship type selection (1-6)
- ✅ Canonical group prompting (Layer 1)
- ✅ Variant status prompting (Layer 2)
- ✅ Enhanced JSON output with hierarchical metadata
- ✅ Resumable sessions with auto-save
- ✅ Relationship type distribution tracking

**Workflow**:
1. Display intervention pair
2. Select relationship type (1-6/s/q)
3. Enter canonical group (e.g., "probiotics")
4. Specify variant status (same/different)
5. Auto-save every 5 pairs

**Output Format**:
```json
{
  "pair_id": 1,
  "intervention_1": "L. reuteri",
  "intervention_2": "S. boulardii",
  "similarity_score": 0.65,
  "relationship": {
    "type_code": "SAME_CATEGORY",
    "type_display": "Same Category (different entities in same class)",
    "aggregation_rule": "separate_all_layers",
    "hierarchy": {
      "layer_1_canonical": "probiotics",
      "layer_2_variant_1": "L. reuteri",
      "layer_2_variant_2": "S. boulardii",
      "same_variant_layer_2": false
    }
  },
  "labeled_at": "2025-10-05T18:00:00"
}
```

---

### 3. Database Schema Design
**File**: [HIERARCHICAL_SCHEMA.sql](HIERARCHICAL_SCHEMA.sql)

**Tables**:
1. **`semantic_hierarchy`** - Main hierarchical entity table
   - 4 layers (category, canonical, variant, detail)
   - Parent-child relationships
   - Semantic embedding storage
   - Aggregation rules

2. **`entity_relationships`** - Explicit relationship tracking
   - Relationship types (EXACT_MATCH, VARIANT, etc.)
   - Confidence scores
   - Source tracking (manual/llm/embedding)
   - Hierarchical aggregation flags

3. **`canonical_groups`** - Layer 1 canonical entities
   - Group-level metadata
   - Member counts
   - Group embeddings

**Views**:
- `v_intervention_hierarchy` - Full hierarchical context
- `v_intervention_by_canonical` - Layer 1 aggregation
- `v_intervention_by_variant` - Layer 2 aggregation

---

### 4. Comprehensive Documentation
**Files**:
- [HIERARCHICAL_GUIDE.md](HIERARCHICAL_GUIDE.md) - Complete usage guide
- [README.md](README.md) - Project overview
- [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) - Phase 1 completion status

**Documentation Includes**:
- Problem statement and examples
- 4-layer architecture explanation
- 6 relationship type definitions with examples
- Labeling workflow step-by-step
- Query examples for different aggregation levels
- Tips and best practices

---

## 🎯 Key Design Decisions

### Relationship Type Semantics

| Relationship | Layer 1 | Layer 2 | Use Case |
|--------------|---------|---------|----------|
| **EXACT_MATCH** | Same | Same | Synonyms (vitamin D = cholecalciferol) |
| **VARIANT** | Same | Different | Biosimilars (Cetuximab vs Cetuximab-β) |
| **SUBTYPE** | Same | Different | Subtypes (IBS-D vs IBS-C) |
| **SAME_CATEGORY** | Different* | Different | Category members (L. reuteri vs S. boulardii) |
| **DOSAGE_VARIANT** | Same | Same | Dosage variants (metformin vs metformin 500mg) |
| **DIFFERENT** | Different | Different | Unrelated (vitamin D vs chemotherapy) |

*SAME_CATEGORY shares the category but NOT the canonical entity

### Aggregation Levels

```
Level 1 Query: "How effective are probiotics for IBS?"
→ Aggregates ALL probiotic strains (L. reuteri, S. boulardii, etc.)
→ Uses Layer 1 canonical grouping

Level 2 Query: "Compare L. reuteri vs S. boulardii for IBS"
→ Separates specific strains
→ Uses Layer 2 variant distinction

Level 3 Query: "Optimal dosage of L. reuteri"
→ Analyzes dosage-specific data
→ Uses Layer 3 detail granularity
```

---

## 📊 Current Status

### Completed Components

✅ **Configuration**: Relationship types, thresholds, layer definitions
✅ **Labeling Interface**: Interactive hierarchical labeling with 6 relationship types
✅ **Database Schema**: Complete schema with hierarchy support
✅ **Documentation**: Comprehensive guides and examples
✅ **JSON Format**: Enhanced ground truth format with hierarchical metadata

### Ready for Use

✅ **Labeling**: Can start labeling 50 intervention pairs immediately
✅ **Session Management**: Resumable sessions with auto-save
✅ **Export/Import**: Candidate pairs generated and ready

### Pending (Phase 2)

⏳ **Semantic Embedding Model**: Train/implement embedding-based matching
⏳ **Hierarchical Normalizer**: Multi-layer similarity matching
⏳ **Evaluation System**: Test against ground truth
⏳ **Database Migration**: Implement schema in production database

---

## 🚀 How to Start

### Run Hierarchical Labeling

```bash
cd back_end/experiments/semantic_normalization
python -m core.labeling_interface
```

### Labeling Process

1. **Pair Display**: See intervention pair with similarity score
2. **Relationship Selection**: Choose 1-6 based on relationship type
3. **Canonical Group**: Enter broad category (e.g., "probiotics")
4. **Variant Status**: Specify if same (y) or different (n) variant
5. **Auto-save**: Progress saved every 5 pairs
6. **Resume Anytime**: Press 'q' to quit, resume later

### Example Labeling Session

```
PAIR 1 of 50
================================
Intervention 1: L. reuteri
Intervention 2: S. boulardii
Similarity Score: 0.6500

What is the relationship?
4. SAME_CATEGORY (different entities in same class)

Select relationship (1-6/s/q): 4

Canonical group (Layer 1): probiotics

Same variant? (y/n): n

✓ Labeled as SAME_CATEGORY
```

---

## 💡 Real-World Examples

### Example 1: Probiotics
```
L. reuteri vs S. boulardii

Relationship: SAME_CATEGORY
Layer 1 Canonical: "probiotics"
Layer 2 Variant 1: "L. reuteri"
Layer 2 Variant 2: "S. boulardii"
Same Variant: NO

Result: Aggregate for "probiotics" analysis, separate for strain comparison
```

### Example 2: Cetuximab Variants
```
Cetuximab vs Cetuximab-β

Relationship: VARIANT
Layer 1 Canonical: "cetuximab"
Layer 2 Variant 1: "cetuximab (original)"
Layer 2 Variant 2: "cetuximab-β (biosimilar)"
Same Variant: NO

Result: Link as variants of same therapeutic concept
```

### Example 3: IBS Subtypes
```
IBS-D vs IBS-C

Relationship: SUBTYPE
Layer 1 Canonical: "IBS"
Layer 2 Variant 1: "IBS-D (diarrhea)"
Layer 2 Variant 2: "IBS-C (constipation)"
Same Variant: NO

Result: Group under "IBS" for general research, separate for subtype-specific analysis
```

---

## 📁 File Structure

```
semantic_normalization/
├── core/
│   ├── labeling_interface.py         ✅ Hierarchical labeling (UPDATED)
│   ├── data_exporter.py               ✅ Data export
│   ├── pair_generator.py              ✅ Candidate generation
│   ├── normalizer.py                  ⏳ Phase 2
│   ├── test_runner.py                 ⏳ Phase 2
│   └── evaluator.py                   ⏳ Phase 2
├── config/
│   └── config.yaml                    ✅ Hierarchical config (UPDATED)
├── data/
│   ├── samples/
│   │   └── interventions_export_*.json  ✅ 500 interventions
│   └── ground_truth/
│       ├── candidate_pairs_*.json       ✅ 150 candidates
│       └── labeling_session_*.json      ⏳ To be created by labeling
├── HIERARCHICAL_SCHEMA.sql            ✅ Database design
├── HIERARCHICAL_GUIDE.md              ✅ Usage guide
├── HIERARCHICAL_IMPLEMENTATION_COMPLETE.md  ✅ This file
└── README.md                          ✅ Project overview
```

---

## 🎯 Next Steps

### Immediate: Complete Labeling
1. Run labeling interface: `python -m core.labeling_interface`
2. Label 50 intervention pairs with hierarchical relationships
3. Review ground truth distribution

### Phase 2: Semantic Embedding System
1. Choose embedding model (e.g., SentenceTransformers)
2. Implement hierarchical matching with layer-specific thresholds
3. Build normalizer with multi-layer similarity
4. Integrate with database schema

### Phase 3: Evaluation
1. Test embedding system against ground truth
2. Calculate precision, recall, F1 per relationship type
3. Compare with existing LLM-based approach
4. Optimize thresholds per layer

### Phase 4: Integration
1. Migrate production database to new schema
2. Replace LLM-based normalization
3. Update query interfaces for hierarchical access
4. Deploy to production pipeline

---

## ✨ Benefits Achieved

✅ **Clinical Accuracy**: Preserves IBS-D ≠ IBS-C distinction while grouping under IBS
✅ **Flexible Queries**: Aggregate at Layer 1, compare at Layer 2
✅ **Research Value**: Analyze "all probiotics" OR compare specific strains
✅ **Evolutionary Design**: Can add Layer 4, 5 as needed
✅ **Embedding-Ready**: Hierarchical relationships captured for ML
✅ **Production-Ready Schema**: Complete database design with views
✅ **User-Friendly Interface**: Clear labeling workflow with examples

---

**Implementation Date**: October 5, 2025
**Status**: ✅ READY FOR LABELING
**Next Action**: Run `python -m core.labeling_interface` to start labeling 50 pairs
