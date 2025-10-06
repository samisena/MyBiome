# Bug Fix: Session Loading Issue

**Date**: October 6, 2025
**Issue**: KeyError when loading labeling session
**Status**: ✅ FIXED

---

## Problem

When running `label_in_batches.py`, the tool was finding the candidates file instead of the session file:

```
Found existing session: labeling_session_hierarchical_candidates_500_20251006_164058.json
Resume this session? (y/n): y
Traceback (most recent call last):
  ...
  File "core/labeling_interface.py", line 90, in load_or_create_session
    self.session_id = session_data['session_id']
KeyError: 'session_id'
```

**Root Cause**: The glob pattern `labeling_session_*.json` was matching BOTH:
- Session files: `labeling_session_hierarchical_ground_truth_*.json`
- Candidate files: `labeling_session_hierarchical_candidates_500_*.json`

Candidate files don't have a `session_id` field, causing the KeyError.

---

## Solution

**File**: [`core/labeling_interface.py`](core/labeling_interface.py)

### Change 1: Filter Out Candidates Files

**Before**:
```python
session_files = list(self.output_dir.glob("labeling_session_*.json"))
```

**After**:
```python
session_files = [
    f for f in self.output_dir.glob("labeling_session_*.json")
    if 'candidates' not in f.name
]
```

### Change 2: Add Validation

**Before**:
```python
session_data = json.load(f)
self.session_id = session_data['session_id']
```

**After**:
```python
session_data = json.load(f)

# Validate it's a session file
if 'session_id' not in session_data:
    print(f"ERROR: {latest_session.name} is not a valid session file")
    print("Creating new session instead...")
else:
    self.session_id = session_data['session_id']
    ...
```

### Change 3: Improve Candidates Loading

**Before**:
```python
candidate_files = list(self.output_dir.glob("candidate_pairs_*.json"))
```

**After**:
```python
candidate_files = []
# Pattern 1: Old format
candidate_files.extend(self.output_dir.glob("candidate_pairs_*.json"))
# Pattern 2: New format with "candidates" in name
candidate_files.extend([
    f for f in self.output_dir.glob("labeling_session_*candidates*.json")
    if 'candidates' in f.name
])
```

---

## File Types in ground_truth/

| File Type | Pattern | Has session_id? | Purpose |
|-----------|---------|----------------|---------|
| **Session files** | `labeling_session_hierarchical_ground_truth_*.json` | ✅ Yes | Stores labeled pairs |
| **Candidate files** | `labeling_session_hierarchical_candidates_500_*.json` | ❌ No | Stores candidate pairs to label |
| **Old candidate files** | `candidate_pairs_*.json` | ❌ No | Old format candidates |

---

## Verification

Current files in `data/ground_truth/`:
```
Session files (3):
  - candidate_pairs_20251005_173745.json (old candidates)
  - labeling_session_hierarchical_ground_truth_20251005_184757.json (50 labels)
  - labeling_session_hierarchical_ground_truth_20251005_184757_BACKUP.json (backup)

Candidate files (1):
  - labeling_session_hierarchical_candidates_500_20251006_164058.json (500 candidates)
```

The fix correctly separates:
- **3 session files** (excluding candidates file)
- **1 candidate file** (matching 'candidates' in name)

---

## Testing

To verify the fix works:

```bash
cd back_end/experiments/semantic_normalization

# Should now correctly load session (not candidates)
python label_in_batches.py --status

# Should create new session if no existing session found
python label_in_batches.py --batch-size 50 --start-from 0
```

---

## Status

✅ **Fixed in both locations**:
1. `back_end/experiments/semantic_normalization/core/labeling_interface.py`
2. `back_end/src/semantic_normalization/ground_truth/labeling_interface.py`

Ready for labeling!
