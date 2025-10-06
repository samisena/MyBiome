# Hierarchical Semantic Normalization - Complete Guide

## Overview

This system implements **multi-layer hierarchical normalization** for medical interventions and conditions, addressing the limitation of binary "match/no-match" classification.

## Problem Statement

Real-world medical interventions have complex relationships that can't be captured with simple binary matching:

### Example 1: Probiotics
- **L. reuteri** and **S. boulardii** are BOTH probiotics (Layer 1)
- But they are DIFFERENT specific strains (Layer 2)
- **Decision**: Aggregate when analyzing "probiotics for IBS", but keep separate when comparing specific strains

### Example 2: Cetuximab Variants
- **Cetuximab** and **Cetuximab-β** are the SAME concept (Layer 1)
- But they are DIFFERENT formulations (Layer 2: biosimilar)
- **Decision**: Link them as variants of the same canonical entity

### Example 3: IBS Subtypes
- **IBS-D** and **IBS-C** are both subtypes of IBS (Layer 1)
- But they are clinically DISTINCT conditions (Layer 2)
- **Decision**: Group under "IBS" for broad analysis, separate for subtype-specific research

## Hierarchical Architecture

### 4-Layer System

```
Layer 0: Category (Taxonomy)
    ├─ supplement, medication, therapy, etc. (13 intervention categories)
    └─ cardiac, neurological, digestive, etc. (18 condition categories)

Layer 1: Canonical Entity (Semantic Grouping)
    ├─ probiotics, statins, cetuximab, IBS
    └─ Broad aggregation level for general queries

Layer 2: Specific Variant (Exact Entity)
    ├─ L. reuteri, atorvastatin, cetuximab-β, IBS-D
    └─ Specific entity level for detailed analysis

Layer 3: Dosage/Details (Granular)
    ├─ L. reuteri 10^9 CFU, atorvastatin 20mg
    └─ Dosage-specific optimization queries
```

## Relationship Types

### 1. EXACT_MATCH
**Definition**: Identical interventions (synonyms, equivalent names)

**Examples**:
- "vitamin D" = "cholecalciferol"
- "PPI" = "proton pump inhibitor"
- "cognitive behavioral therapy" = "CBT"

**Aggregation**: `merge_completely`
- Layer 1: Same canonical entity
- Layer 2: Same variant
- **Action**: Treat as identical in all analyses

---

### 2. VARIANT
**Definition**: Same therapeutic concept but different formulation or biosimilar

**Examples**:
- "Cetuximab" ~ "Cetuximab-β" (biosimilar)
- "insulin glargine" ~ "insulin detemir" (long-acting insulins)

**Aggregation**: `share_layer_1_link_layer_2`
- Layer 1: Share canonical entity ("cetuximab", "long-acting insulin")
- Layer 2: Different variants (linked as related)
- **Action**: Aggregate for broad efficacy, compare for specific formulation differences

---

### 3. SUBTYPE
**Definition**: Related subtypes of the same parent condition or intervention class

**Examples**:
- "IBS-D" ~ "IBS-C" (IBS subtypes)
- "type 1 diabetes" ~ "type 2 diabetes"

**Aggregation**: `share_layer_1_separate_layer_2`
- Layer 1: Share parent entity ("IBS", "diabetes")
- Layer 2: Separate variants (clinically distinct)
- **Action**: Aggregate for general "IBS" research, separate for subtype-specific interventions

---

### 4. SAME_CATEGORY
**Definition**: Different members of the same intervention category

**Examples**:
- "L. reuteri" ≠ "S. boulardii" (both probiotics)
- "atorvastatin" ≠ "simvastatin" (both statins)

**Aggregation**: `separate_all_layers`
- Layer 1: Same category ("probiotics", "statins") but NO canonical linking
- Layer 2: Completely separate variants
- **Action**: Keep separate in all analyses, but can group by category for broad queries

---

### 5. DOSAGE_VARIANT
**Definition**: Same intervention with explicit dosage differences

**Examples**:
- "metformin" ~ "metformin 500mg"
- "vitamin D 1000 IU" ~ "vitamin D 5000 IU"

**Aggregation**: `share_layers_1_2`
- Layer 1: Same canonical entity ("metformin", "vitamin D")
- Layer 2: Same variant (same drug)
- Layer 3: Different dosage details
- **Action**: Aggregate for general efficacy, separate for dose-response analysis

---

### 6. DIFFERENT
**Definition**: No relationship between interventions

**Examples**:
- "vitamin D" ≠ "chemotherapy"
- "exercise" ≠ "surgery"

**Aggregation**: `no_relationship`
- Layer 1: Different canonical entities
- Layer 2: Different variants
- **Action**: No aggregation or linking

---

## Labeling Workflow

### Step 1: Relationship Type Selection

```
What is the relationship between these interventions?

1. EXACT_MATCH (same intervention, same formulation)
2. VARIANT (same concept, different formulation)
3. SUBTYPE (related but clinically distinct)
4. SAME_CATEGORY (different entities in same class)
5. DOSAGE_VARIANT (same intervention, different dose)
6. DIFFERENT (completely unrelated)

Select relationship (1-6/s/q):
```

### Step 2: Canonical Group (Layer 1)

For relationships 1-5, you'll be asked:

```
What is the canonical group (Layer 1) for these interventions?
(e.g., 'probiotics', 'statins', 'IBS', 'cetuximab')

Intervention 1: L. reuteri
Intervention 2: S. boulardii

Canonical group (or press Enter to skip): probiotics
```

### Step 3: Variant Status (Layer 2)

```
Are these the SAME specific variant or DIFFERENT variants?

Intervention 1: L. reuteri
Intervention 2: S. boulardii

Examples:
  - SAME: 'metformin' = 'metformin therapy' (same drug)
  - DIFFERENT: 'Cetuximab' != 'Cetuximab-β' (biosimilar)
  - DIFFERENT: 'L. reuteri' != 'S. boulardii' (different strains)

Same variant? (y/n): n
```

## Output Format

### Labeled Pair JSON Structure

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

## Query Examples

### Query 1: Broad Category Analysis
**Question**: "How effective are probiotics for IBS?"

**Query**:
```sql
SELECT * FROM v_intervention_by_canonical
WHERE layer_1_canonical = 'probiotics'
AND health_condition LIKE '%IBS%';
```

**Aggregates**: All probiotic strains (L. reuteri, S. boulardii, etc.)

---

### Query 2: Specific Variant Comparison
**Question**: "Compare L. reuteri vs S. boulardii for IBS"

**Query**:
```sql
SELECT * FROM v_intervention_by_variant
WHERE layer_1_canonical = 'probiotics'
AND layer_2_variant IN ('L. reuteri', 'S. boulardii')
AND health_condition LIKE '%IBS%';
```

**Separates**: Individual probiotic strains

---

### Query 3: Variant Relationships
**Question**: "Find all biosimilars of Cetuximab"

**Query**:
```sql
SELECT sh2.entity_name AS biosimilar
FROM entity_relationships er
JOIN semantic_hierarchy sh1 ON er.entity_1_id = sh1.id
JOIN semantic_hierarchy sh2 ON er.entity_2_id = sh2.id
WHERE sh1.entity_name = 'Cetuximab'
AND er.relationship_type = 'VARIANT';
```

**Returns**: Cetuximab-β (and other variants)

---

## Usage Instructions

### Start Labeling

```bash
cd back_end/experiments/semantic_normalization
python -m core.labeling_interface
```

### Session Management

- **Auto-save**: Every 5 labeled pairs
- **Resume**: Automatically detects and offers to resume previous session
- **Quit**: Press 'q' to save progress and exit anytime

### Tips for Labeling

1. **Canonical Groups**: Use broad, intuitive names
   - ✅ Good: "probiotics", "statins", "IBS"
   - ❌ Avoid: "lactobacillus_species", "hmg_coa_reductase_inhibitors"

2. **Same vs Different Variants**:
   - Same: Synonyms, equivalent names (metformin = metformin therapy)
   - Different: Distinct entities, even if related (L. reuteri ≠ S. boulardii)

3. **When Uncertain**: Choose the most conservative relationship
   - If unsure between VARIANT and SUBTYPE → choose SUBTYPE
   - If unsure between SAME_CATEGORY and DIFFERENT → choose DIFFERENT

## Benefits of Hierarchical System

✅ **Clinical Accuracy**: Preserves important distinctions (IBS-D ≠ IBS-C)
✅ **Flexible Aggregation**: Query at any granularity level
✅ **Research Value**: Analyze broad categories AND specific variants
✅ **Future-Proof**: Can add more layers as needed
✅ **Embedding-Friendly**: Semantic embeddings naturally capture hierarchy

## Next Steps

After completing labeling (50 pairs):

1. **Analyze Ground Truth**: Review relationship type distribution
2. **Phase 2**: Build semantic embedding system with hierarchical matching
3. **Evaluation**: Test embedding system against labeled ground truth
4. **Integration**: Replace existing LLM-based normalization
5. **Production**: Deploy hierarchical system to main pipeline

## Files

- **Config**: `config/config.yaml` (relationship types, thresholds)
- **Labeling**: `core/labeling_interface.py` (interactive interface)
- **Schema**: `HIERARCHICAL_SCHEMA.sql` (database design)
- **Output**: `data/ground_truth/labeling_session_*.json` (labeled pairs)
