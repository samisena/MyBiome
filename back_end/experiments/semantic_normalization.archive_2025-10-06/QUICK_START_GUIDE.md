# Quick Start Guide: Labeling 500 Intervention Pairs

**Goal**: Label 500 intervention pairs with hierarchical relationships

**Time Estimate**: ~20 hours total (10 sessions × 2 hours each)

---

## Prerequisites

1. Python 3.13+ installed
2. Dependencies installed:
   ```bash
   pip install rapidfuzz pyyaml
   ```
3. Intervention data exported (run once):
   ```bash
   python -m back_end.experiments.semantic_normalization.core.data_exporter
   ```

---

## Step-by-Step Instructions

### 1. Generate 500 Candidate Pairs (One-Time Setup)

```bash
cd back_end/experiments/semantic_normalization
python generate_500_candidates.py
```

**Expected Output**:
- Creates `labeling_session_hierarchical_candidates_500_YYYYMMDD.json`
- Shows distribution: ~300 high similarity, ~100 medium, ~100 low
- Displays sample pairs

**Time**: ~2-3 minutes

---

### 2. Label First Batch (Pairs 1-50)

```bash
python label_in_batches.py --batch-size 50 --start-from 0
```

**What happens**:
1. Shows session overview
2. Displays first pair with similarity score
3. Asks for relationship type (1-6)
4. Asks for canonical group (Layer 1)
5. Asks if same variant (Layer 2)
6. Moves to next pair
7. Auto-saves every 10 pairs
8. Shows summary when done

**Time**: ~2 hours

---

### 3. Check Your Progress

```bash
python label_in_batches.py --status
```

**Shows**:
- Total progress (e.g., 50/500 = 10%)
- Batch grid (✓ DONE, ⏳ IN PROGRESS, ○ PENDING)
- Relationship type distribution
- Suggested next batch

**Time**: Instant

---

### 4. Continue with Next Batches

**Batch 2 (Pairs 51-100)**:
```bash
python label_in_batches.py --batch-size 50 --start-from 50
```

**Batch 3 (Pairs 101-150)**:
```bash
python label_in_batches.py --batch-size 50 --start-from 100
```

**...continue until Batch 10 (Pairs 451-500)**:
```bash
python label_in_batches.py --batch-size 50 --start-from 450
```

---

## Keyboard Shortcuts (During Labeling)

| Key | Action | Description |
|-----|--------|-------------|
| **1** | EXACT_MATCH | Same intervention (synonyms) |
| **2** | VARIANT | Same concept, different formulation |
| **3** | SUBTYPE | Related but clinically distinct |
| **4** | SAME_CATEGORY | Different entities in same class |
| **5** | DOSAGE_VARIANT | Same intervention, different dose |
| **6** | DIFFERENT | Completely unrelated |
| **s** | Skip | Skip this pair (move to next) |
| **u** | Undo | Undo last label |
| **r** | Review Later | Mark for later review |
| **q** | Quit | Save and quit (can resume later) |

---

## Tips for Efficient Labeling

### 1. Use Batch Mode (Recommended)
- Label 50 pairs per session (~2 hours)
- Take breaks between batches
- Prevents fatigue and improves accuracy

### 2. Undo is Your Friend
- Made a mistake? Press **u** immediately
- Can undo last 10 labels

### 3. Review Later for Difficult Pairs
- Uncertain? Press **r** to mark for review
- Focus on clear-cut pairs first
- Revisit challenging pairs at the end

### 4. Check Progress Regularly
```bash
python label_in_batches.py --status
```
- See how far you've come
- Motivation boost!

### 5. Don't Rush
- Quality > Speed
- Accurate labels = better model performance
- 2-3 minutes per pair is normal

---

## Relationship Type Guidelines

### EXACT_MATCH (1)
**Definition**: Same intervention, different spelling/capitalization
- "vitamin D" = "Vitamin D"
- "PPI" = "proton pump inhibitor"
- "CBT" = "cognitive behavioral therapy"

### VARIANT (2)
**Definition**: Same concept, different formulation/biosimilar
- "Cetuximab" vs "Cetuximab-β" (biosimilar)
- "insulin glargine" vs "insulin detemir" (different formulations)
- "standard dose" vs "high dose" (protocol variants)

### SUBTYPE (3)
**Definition**: Related subtypes of same parent
- "IBS-D" vs "IBS-C" (different IBS types)
- "absorbable suture" vs "non-absorbable suture"
- "type 1 diabetes" vs "type 2 diabetes"

### SAME_CATEGORY (4)
**Definition**: Different members of same drug/therapy class
- "atorvastatin" vs "simvastatin" (both statins)
- "L. reuteri" vs "S. boulardii" (both probiotics)
- "omeprazole" vs "lansoprazole" (both PPIs)

### DOSAGE_VARIANT (5)
**Definition**: Same intervention, explicit dosage difference
- "metformin" vs "metformin 500mg"
- "vitamin D 1000 IU" vs "vitamin D 5000 IU"

### DIFFERENT (6)
**Definition**: Completely unrelated interventions
- "vitamin D" vs "chemotherapy"
- "exercise" vs "surgery"
- "probiotics" vs "acupuncture"

---

## Troubleshooting

### Q: Session won't resume?
**A**: Check if session file exists:
```bash
ls back_end/experiments/semantic_normalization/data/ground_truth/labeling_session_*.json
```

### Q: Lost progress after crash?
**A**: Auto-save happens every 10 labels. You'll resume from last auto-save point.

### Q: Want to change a label from 20 pairs ago?
**A**:
1. Undo is limited to last 10 labels
2. For older labels, need to manually edit JSON file
3. Best practice: Use "review later" (r) when uncertain

### Q: Candidate file not found?
**A**: Generate candidates first:
```bash
python generate_500_candidates.py
```

### Q: How do I know which batch to do next?
**A**: Use the suggest command:
```bash
python label_in_batches.py --suggest
```

---

## Labeling Schedule Suggestion

**Goal**: Complete 500 labels in 2 weeks

| Week | Day | Batch | Pairs | Time | Cumulative |
|------|-----|-------|-------|------|------------|
| 1 | Mon | 1 | 1-50 | 2h | 50 (10%) |
| 1 | Tue | 2 | 51-100 | 2h | 100 (20%) |
| 1 | Wed | 3 | 101-150 | 2h | 150 (30%) |
| 1 | Thu | 4 | 151-200 | 2h | 200 (40%) |
| 1 | Fri | 5 | 201-250 | 2h | 250 (50%) |
| 2 | Mon | 6 | 251-300 | 2h | 300 (60%) |
| 2 | Tue | 7 | 301-350 | 2h | 350 (70%) |
| 2 | Wed | 8 | 351-400 | 2h | 400 (80%) |
| 2 | Thu | 9 | 401-450 | 2h | 450 (90%) |
| 2 | Fri | 10 | 451-500 | 2h | 500 (100%) ✅ |

**Total**: 10 sessions × 2 hours = 20 hours

---

## Example Labeling Session

```
================================================================================
HIERARCHICAL INTERVENTION PAIR LABELING SESSION
================================================================================

Instructions:
  - Review each pair of intervention names
  - Select the hierarchical relationship type (1-6)
  - Provide canonical group and variant information
  - Press 'u' to undo last label, 'r' to mark for review

Relationship Types:
  1. EXACT_MATCH - Same intervention (synonyms)
  2. VARIANT - Same concept, different formulation
  3. SUBTYPE - Related but clinically distinct
  4. SAME_CATEGORY - Different entities in same class
  5. DOSAGE_VARIANT - Same intervention, different dose
  6. DIFFERENT - Completely unrelated

Batch Mode: Labeling 50 pairs starting from 0
================================================================================

Progress: [████████████████████████████████░░░░░░░░░░░░░░░░] 32/50 (64.0%)
Estimated time remaining: 36m

================================================================================
PAIR 33 of 50
Progress: [████████████████████████████████░░░░░░░░░░░░░░░░] 32/50 (64.0%)
================================================================================

Intervention 1: atorvastatin
Intervention 2: simvastatin

Similarity Score: 0.6923
Length Difference: 1 characters
Word Counts: 1 vs 1
--------------------------------------------------------------------------------

What is the relationship between these interventions?
--------------------------------------------------------------------------------
1. EXACT_MATCH
   Exact Match (same intervention, same formulation)
   Examples: vitamin D = cholecalciferol

2. VARIANT
   Variant (same concept, different formulation)
   Examples: Cetuximab vs Cetuximab-beta (biosimilar)

3. SUBTYPE
   Subtype (related but clinically distinct)
   Examples: IBS-D vs IBS-C

4. SAME_CATEGORY
   Same Category (different entities in same class)
   Examples: L. reuteri vs S. boulardii (both probiotics)

5. DOSAGE_VARIANT
   Dosage Variant (same intervention, different dose)
   Examples: metformin vs metformin 500mg

6. DIFFERENT
   Different (completely unrelated interventions)
   Examples: vitamin D vs chemotherapy

s - SKIP this pair
u - UNDO last label (Ctrl+Z)
r - Review later (mark for review)
q - QUIT and save progress
--------------------------------------------------------------------------------

Select relationship (1-6/s/u/r/q): 4

What is the canonical group (Layer 1) for these interventions?
(e.g., 'probiotics', 'statins', 'IBS', 'cetuximab')
Intervention 1: atorvastatin
Intervention 2: simvastatin
Canonical group (or press Enter to skip): statins

Are these the SAME specific variant or DIFFERENT variants?
Intervention 1: atorvastatin
Intervention 2: simvastatin
Examples:
  - SAME: 'metformin' = 'metformin therapy' (same drug)
  - DIFFERENT: 'Cetuximab' != 'Cetuximab-β' (biosimilar)
  - DIFFERENT: 'L. reuteri' != 'S. boulardii' (different strains)
Same variant? (y/n): n

[Pair labeled as SAME_CATEGORY - Different statins]
```

---

## When You're Done

After completing all 500 labels:

1. **Verify completion**:
   ```bash
   python label_in_batches.py --status
   ```
   Should show: `Progress: 500/500 pairs (100.0%)`

2. **Review ground truth file**:
   - Location: `data/ground_truth/labeling_session_hierarchical_ground_truth_*.json`
   - Contains all 500 labeled pairs

3. **Check review_later queue** (if you used 'r'):
   - Location: `data/ground_truth/review_later_*.json`
   - Go back and label these pairs

4. **Notify team**:
   - Ground truth expansion complete
   - Ready for Phase 2 (integration)

---

**Questions?** Check [`PHASE1_IMPLEMENTATION.md`](PHASE1_IMPLEMENTATION.md) for technical details.

**Good luck!** 🚀
