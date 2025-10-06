# Labeling Tool Fix

**Issue**: FileNotFoundError when running label_in_batches.py

**Root Cause**: Config file missing in expected location

## Fixes Applied

### 1. ✅ Copied config.yaml to ground_truth/config/
```bash
mkdir -p back_end/src/semantic_normalization/ground_truth/config
cp back_end/src/semantic_normalization/config/config.yaml \
   back_end/src/semantic_normalization/ground_truth/config/config.yaml
```

### 2. ✅ Fixed Unicode Characters in Status Display
**File**: label_in_batches.py (lines 93-99)

**Before**:
```python
status = "✓ DONE"
status = "⏳ IN PROGRESS"
status = "○ PENDING"
```

**After**:
```python
status = "[DONE]"
status = "[IN PROGRESS]"
status = "[PENDING]"
```

## Verification

Tool now works correctly:
```bash
cd back_end/src/semantic_normalization/ground_truth
python label_in_batches.py --status
```

**Output**:
```
================================================================================
BATCH LABELING STATUS
================================================================================

Batch Grid:
  Batch  1: Pairs   1- 50 [[PENDING]]
  Batch  2: Pairs  51-100 [[PENDING]]
  ...
  Batch 10: Pairs 451-500 [[PENDING]]

Suggested Next Batch:
  python label_in_batches.py --batch-size 50 --start-from 0
```

## Ready to Resume Labeling

You can now continue labeling the 500 pairs:

```bash
cd back_end/src/semantic_normalization/ground_truth
python label_in_batches.py --batch-size 50 --start-from 50
```

The tool will:
- ✅ Load the NEW 500-pair candidates file
- ✅ Validate batch boundaries
- ✅ Prevent duplicate pairs
- ✅ Show ASCII-safe status indicators
