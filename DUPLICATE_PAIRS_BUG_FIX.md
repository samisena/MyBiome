# Duplicate Pairs Bug - Root Cause & Fix

**Date**: October 6, 2025
**Reporter**: User
**Status**: ✅ IDENTIFIED | 🔧 FIX IN PROGRESS

---

## Problem Report

**Symptom**: Reviewing the same pairs multiple times during labeling
**Impact**: Reduced data quality (duplicate labels)

---

## Root Cause Analysis

### Issue #1: Insufficient Candidates Generated ❌

**Expected**: 500 candidate pairs
**Actual**: 351 candidate pairs (only 70%)

**Evidence**:
```json
{
  "metadata": {
    "target_count": 500,
    "total_candidates": 351  // <-- ONLY 351!
  }
}
```

**Why this causes duplicates**:
- You run `--start-from 350` (batch 8)
- Only 351 pairs exist (batch 8 would be pairs 350-399)
- Tool shows pair 350 then runs out
- May cycle back or show errors

---

### Issue #2: No Validation in label_in_batches.py ❌

**Current behavior**:
- Doesn't check if `start_from` exceeds available pairs
- Doesn't warn if batch extends beyond candidates
- Silently fails or shows unexpected pairs

**Example**:
```python
# start_from = 350, batch_size = 50
# candidates = 351 pairs
# Expected end = 400, but only 351 exist
# Shows pair 350, then ??? (undefined behavior)
```

---

### Issue #3: Stratified Sampling Underperformed

**Target distribution** (from config):
- Similarity-based (60%): 300 pairs
  - 0.85-0.95: 100 pairs
  - 0.75-0.85: 100 pairs
  - 0.65-0.75: 100 pairs
- Random (20%): 100 pairs (0.40-0.65)
- Targeted (20%): 100 pairs (same category)

**Actual distribution**:
- Likely match (>0.75): 45 pairs (expected 200+)
- Edge case (0.50-0.75): 127 pairs (expected 100)
- Likely no match (<0.50): 179 pairs (expected 100)

**Total**: 351 pairs (expected 500)

**Why sampling failed**:
- Not enough interventions in database? (check exports)
- Similarity thresholds too strict?
- Deduplication removed too many?
- Insufficient same-category pairs?

---

## Diagnostic Commands

### Check Export Data
```bash
cd back_end/src/semantic_normalization/ground_truth

python -c "
from data_exporter import InterventionDataExporter

exporter = InterventionDataExporter()
export_data = exporter.get_latest_export()

if export_data:
    print(f'Unique interventions: {len(export_data[\"unique_names\"])}')
    print(f'Full data records: {len(export_data.get(\"full_data\", []))}')
else:
    print('No export found')
"
```

### Calculate Theoretical Max Pairs
```bash
python -c "
# If N interventions, max pairs = N * (N-1) / 2
N = 542  # From your database
max_pairs = N * (N - 1) // 2
print(f'With {N} interventions:')
print(f'  Max possible pairs: {max_pairs:,}')
print(f'  Requested: 500')
print(f'  Feasibility: {\"OK\" if max_pairs > 500 else \"INSUFFICIENT\"}')'
"
```

---

## Solution

### Fix #1: Regenerate Candidates with Adjusted Strategy

**Option A**: Lower similarity thresholds
```python
# In pair_generator.py, adjust ranges:
{
    (0.75, 0.95): 100,  # Was 0.85-0.95
    (0.65, 0.85): 100,  # Was 0.75-0.85
    (0.55, 0.75): 100,  # Was 0.65-0.75
}
```

**Option B**: Fill remaining with random sampling
```python
# After stratified sampling:
if len(candidates) < target_count:
    needed = target_count - len(candidates)
    # Sample from ALL pairs, any similarity
    additional = sample_random(all_pairs, needed)
    candidates.extend(additional)
```

**Option C**: Use ALL similarity ranges
```python
# Don't filter by min/max similarity
# Sample from full range 0.0-1.0
```

---

### Fix #2: Add Validation to label_in_batches.py

```python
def run_batch_session(self, batch_size: int = 50, start_from: int = 0):
    # Load candidates first
    candidates = load_candidates()

    # VALIDATE before starting
    if start_from >= len(candidates):
        print(f"ERROR: start_from ({start_from}) exceeds available candidates ({len(candidates)})")
        print(f"Available range: 0-{len(candidates)-1}")
        return

    if start_from + batch_size > len(candidates):
        actual_batch = len(candidates) - start_from
        print(f"WARNING: Only {actual_batch} pairs available in this batch")
        print(f"(Requested {batch_size}, but only {len(candidates)} total candidates)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
```

---

### Fix #3: Display Available Batches

```python
def display_batch_status(candidates_file):
    with open(candidates_file, 'r') as f:
        data = json.load(f)

    total_candidates = len(data['all_candidates'])
    batch_size = 50
    num_batches = (total_candidates + batch_size - 1) // batch_size

    print(f"Total candidates: {total_candidates}")
    print(f"Batch size: {batch_size}")
    print(f"Available batches: {num_batches}")
    print("\nBatch breakdown:")

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total_candidates)
        count = end - start
        print(f"  Batch {i+1}: Pairs {start}-{end-1} ({count} pairs)")
```

---

## Immediate Actions

### For You (User)

**Stop labeling temporarily** until we:
1. Regenerate 500 full candidates
2. Verify no duplicates
3. Update tool with validation

**Check your current progress**:
```bash
python label_in_batches.py --status
```

This will show:
- How many pairs you've labeled
- Which batches are complete
- If there are any duplicates

---

### For Me (Assistant)

1. **Regenerate candidates file** ✅ Urgent
   - Adjust sampling strategy
   - Ensure 500 pairs minimum
   - Verify no duplicates

2. **Add validation to label_in_batches.py** ✅ Urgent
   - Check start_from vs available
   - Warn on batch overflow
   - Display available batches

3. **Check for duplicate labels** ✅ Important
   - Scan existing session file
   - Identify any duplicate pair_ids
   - Remove duplicates if found

4. **Update generate_candidates.py** ✅ Important
   - Add fallback sampling
   - Guarantee minimum count
   - Better error messages

---

## Testing Plan

After fixes:

```bash
# 1. Regenerate candidates
python generate_candidates.py

# Should output:
# "Generated 500 candidate pairs" (not 351!)

# 2. Verify count
python -c "
import json
with open('hierarchical_candidates_500_pairs.json', 'r') as f:
    data = json.load(f)
print(f'Actual count: {len(data[\"all_candidates\"])}')
assert len(data['all_candidates']) == 500, 'Not enough candidates!'
print('[OK] 500 candidates verified')
"

# 3. Check for duplicates in candidates
python -c "
import json
with open('hierarchical_candidates_500_pairs.json', 'r') as f:
    data = json.load(f)

pairs_set = set()
duplicates = []

for i, pair in enumerate(data['all_candidates']):
    key = tuple(sorted([pair['intervention_1'], pair['intervention_2']]))
    if key in pairs_set:
        duplicates.append((i, pair))
    pairs_set.add(key)

if duplicates:
    print(f'[ERROR] Found {len(duplicates)} duplicate pairs!')
    for idx, pair in duplicates[:5]:
        print(f'  [{idx}] {pair[\"intervention_1\"]} vs {pair[\"intervention_2\"]}')
else:
    print('[OK] No duplicates in candidates')
"

# 4. Test batch validation
python label_in_batches.py --start-from 450 --batch-size 50
# Should show: "50 pairs available in batch 10"

python label_in_batches.py --start-from 500 --batch-size 50
# Should show: "ERROR: start_from exceeds available candidates"
```

---

## Data Quality Impact

**If you've labeled duplicates**:
- Need to identify which pairs were duplicated
- Keep only first label for each unique pair
- Discard duplicate labels

**Script to check**:
```python
# Check for duplicates in your session
with open('labeling_session_*.json', 'r') as f:
    session = json.load(f)

labeled_pairs = session['labeled_pairs']
pair_keys = {}
duplicates = []

for label in labeled_pairs:
    key = tuple(sorted([label['intervention_1'], label['intervention_2']]))
    if key in pair_keys:
        duplicates.append((pair_keys[key], label))
    pair_keys[key] = label

if duplicates:
    print(f'Found {len(duplicates)} duplicate labels')
    # Show details and offer to clean
```

---

## Summary

**Root Cause**: Only 351 candidates generated instead of 500
**Impact**: Ran out of pairs before completing 10 batches
**Result**: Tool behavior undefined, possibly showing duplicates

**Fixes Needed**:
1. ✅ Regenerate 500 candidates
2. ✅ Add batch validation
3. ✅ Check/remove duplicate labels
4. ✅ Update generate_candidates.py

**Status**: Fixing now
**ETA**: 30 minutes

---

**Next**: I'll implement all fixes and regenerate the candidates file properly.
