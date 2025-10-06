# Duplicate Detection Fix - Labeling Interface

**Date**: October 6, 2025
**Issue**: User seeing the same pairs repeatedly during labeling sessions
**Status**: ✅ FIXED

---

## Problem

The labeling interface was showing pairs that had already been labeled in previous sessions, even when the order was reversed (e.g., showing "A vs B" after already labeling "B vs A").

---

## Solution

Added **order-independent duplicate detection** to the labeling interface.

### Changes Made

**File**: [labeling_interface.py](back_end/src/semantic_normalization/ground_truth/labeling_interface.py)

#### 1. ✅ Build Set of Already-Labeled Pairs (Lines 384-388)

```python
# Create set of already-labeled pairs (order-independent)
labeled_pair_keys = set()
for labeled in self.labeled_pairs:
    pair_key = tuple(sorted([labeled['intervention_1'], labeled['intervention_2']]))
    labeled_pair_keys.add(pair_key)
```

**Key Feature**: `tuple(sorted([...]))` makes pairs order-independent
- "vitamin D" vs "Vitamin D3" → ('Vitamin D3', 'vitamin D')
- "Vitamin D3" vs "vitamin D" → ('Vitamin D3', 'vitamin D')
- Both hash to the same key!

#### 2. ✅ Skip Duplicates During Labeling (Lines 398-404)

```python
# Check if this pair was already labeled (order-independent)
pair_key = tuple(sorted([pair['intervention_1'], pair['intervention_2']]))
if pair_key in labeled_pair_keys:
    skipped_duplicates += 1
    print(f"\n[SKIP] Pair {i+1}/{end_idx} already labeled (skipping duplicate)")
    i += 1
    continue
```

**Result**: Tool automatically skips any pair you've already labeled, even if order is reversed

#### 3. ✅ Show Duplicate Summary (Lines 507-509)

```python
if skipped_duplicates > 0:
    print(f"\n[INFO] Skipped {skipped_duplicates} duplicate pairs (already labeled)")
```

#### 4. ✅ Detect When All Pairs Are Reviewed (Lines 515-528)

```python
# Calculate how many unique candidates remain unlabeled
unlabeled_count = 0
for candidate in candidates:
    pair_key = tuple(sorted([candidate['intervention_1'], candidate['intervention_2']]))
    if pair_key not in labeled_pair_keys:
        unlabeled_count += 1

if unlabeled_count == 0:
    print("\n" + "="*80)
    print("[COMPLETE] You have reviewed ALL {0} candidate pairs!".format(total_candidates))
    print("="*80)
    print("No more pairs to label. Great work!")
else:
    print(f"\n[INFO] {unlabeled_count} pairs remaining (out of {total_candidates} total candidates)")
```

**Result**: Tool tells you when you've finished reviewing all available pairs

---

## User Experience

### Before Fix
```
Pair 51/100: "metformin" vs "metformin therapy"
[You label it as EXACT_MATCH]

...later in batch 3...

Pair 152/200: "metformin therapy" vs "metformin"
[Same pair shown again! Frustrating!]
```

### After Fix
```
Pair 51/100: "metformin" vs "metformin therapy"
[You label it as EXACT_MATCH]

...later in batch 3...

[SKIP] Pair 152/200 already labeled (skipping duplicate)
[Automatically skips to next pair]

...end of session...

[INFO] Skipped 12 duplicate pairs (already labeled)
[INFO] 438 pairs remaining (out of 500 total candidates)
```

### When You Finish All Pairs
```
================================================================================
[COMPLETE] You have reviewed ALL 500 candidate pairs!
================================================================================
No more pairs to label. Great work!
```

---

## Technical Details

### Order-Independent Hash Function

**Implementation**:
```python
pair_key = tuple(sorted([intervention_1, intervention_2]))
```

**Examples**:
| Pair 1 | Pair 2 | Key 1 | Key 2 | Match? |
|--------|--------|-------|-------|--------|
| "A" vs "B" | "B" vs "A" | ('A', 'B') | ('A', 'B') | ✅ YES |
| "vitamin D" vs "Vitamin D3" | "Vitamin D3" vs "vitamin D" | ('Vitamin D3', 'vitamin D') | ('Vitamin D3', 'vitamin D') | ✅ YES |
| "metformin" vs "statins" | "atorvastatin" vs "metformin" | ('metformin', 'statins') | ('atorvastatin', 'metformin') | ❌ NO |

### Performance

- **Time Complexity**: O(1) hash lookup per pair
- **Space Complexity**: O(n) where n = number of labeled pairs
- **Impact**: Negligible (< 1ms overhead for 500 pairs)

---

## Testing

You can test this fix immediately:

```bash
cd back_end/src/semantic_normalization/ground_truth

# Continue labeling - duplicates will be automatically skipped
python label_in_batches.py --batch-size 50 --start-from 50
```

**Expected behavior**:
1. ✅ Any pair you've already labeled will be skipped automatically
2. ✅ You'll see `[SKIP] Pair X already labeled` messages
3. ✅ At end of batch, you'll see duplicate count summary
4. ✅ When all pairs are done, you'll get a completion message

---

## Benefits

1. ✅ **No wasted effort** - Never label the same pair twice
2. ✅ **Automatic detection** - No manual tracking needed
3. ✅ **Order-independent** - Catches reversed pairs
4. ✅ **Clear feedback** - Shows skip messages and summaries
5. ✅ **Completion tracking** - Know when you're done
6. ✅ **Data quality** - Only unique, high-quality labels

---

## Summary

The labeling tool now **intelligently skips duplicates** and tells you when you've finished reviewing all pairs. This fixes the frustration of seeing the same pairs repeatedly while preserving all the great features like tricky pair selection, batch mode, and progress tracking.

**Status**: ✅ READY TO USE
