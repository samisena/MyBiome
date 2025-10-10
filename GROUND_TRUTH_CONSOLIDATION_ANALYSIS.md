# Ground Truth Files - Consolidation Analysis

**Date:** October 10, 2025
**Current State:** 7 files, 1,723 lines

---

## Current File Structure

| File | Lines | Type | Purpose | Dependencies |
|------|-------|------|---------|--------------|
| `data_exporter.py` | 160 | **Class** | Export interventions from DB | None |
| `pair_generator.py` | 442 | **Class** | Generate candidate pairs via fuzzy matching | None |
| `generate_candidates.py` | 159 | **Script** | Generate 500 pairs (uses exporter + generator) | ✅ Uses #1 + #2 |
| `labeling_interface.py` | 559 | **Class** | Interactive terminal labeling UI | None |
| `label_in_batches.py` | 247 | **Script** | Batch labeling session manager | ✅ Uses #4 |
| `remove_duplicate_labels.py` | 112 | **Script** | Clean duplicate labels | None |
| `__init__.py` | 44 | Module | Package init | None |

**Total:** 1,723 lines across 7 files

---

## Dependency Graph

```
data_exporter.py (Class - standalone)
    ↓
pair_generator.py (Class - standalone)
    ↓
generate_candidates.py (Script - USES both above)
    ↓ (creates candidate file)
labeling_interface.py (Class - standalone, READS candidate file)
    ↓
label_in_batches.py (Script - USES labeling_interface)
    ↓ (creates session file)
remove_duplicate_labels.py (Script - READS session file)
```

**Key Insight:** Only 2 direct imports (`generate_candidates` → exporter+generator, `label_in_batches` → interface)

---

## User's Proposed Merges

### Option 1: Merge `labeling_interface.py` + `label_in_batches.py`
**Current:**
- `labeling_interface.py` (559 lines) - HierarchicalLabelingInterface class
- `label_in_batches.py` (247 lines) - BatchLabelingManager class + CLI

**Analysis:**

| Aspect | Assessment |
|--------|------------|
| **Coupling** | ✅ Low - `label_in_batches` only uses `labeling_interface`, doesn't modify it |
| **Cohesion** | ⚠️ Medium - Related but different concerns (UI vs batch management) |
| **Reusability** | ❌ Hurts - `labeling_interface` is reusable standalone, batch manager is optional |
| **File size** | ❌ 806 lines (too large) |
| **Testability** | ❌ Harder to test separately |

**Recommendation:** ❌ **DO NOT MERGE**

**Reason:** `labeling_interface.py` is a **library** (reusable UI component), `label_in_batches.py` is a **convenience script** (optional workflow tool). Merging would violate Single Responsibility Principle.

---

### Option 2: Merge `generate_candidates.py` + `data_exporter.py` + `remove_duplicate_labels.py`
**Current:**
- `data_exporter.py` (160 lines) - InterventionDataExporter class
- `generate_candidates.py` (159 lines) - Script using exporter + generator
- `remove_duplicate_labels.py` (112 lines) - Script for cleaning duplicates

**Analysis:**

| Aspect | Assessment |
|--------|------------|
| **Coupling** | ❌ None - `remove_duplicate_labels` doesn't use exporter or generator |
| **Cohesion** | ❌ Low - 3 completely different operations (export → generate → clean) |
| **Workflow** | ❌ Non-sequential - clean happens AFTER labeling, not after generation |
| **Reusability** | ❌ Hurts - `data_exporter` is reusable, scripts are one-time runners |
| **File size** | ⚠️ 431 lines (acceptable but questionable) |

**Recommendation:** ❌ **DO NOT MERGE**

**Reason:** These are **3 separate workflow steps** that don't belong together:
1. Export (data extraction)
2. Generate (pair creation)
3. Clean (post-labeling maintenance)

Merging would create a confusing "utility grab bag" file.

---

## Alternative Consolidation Strategies

### Strategy A: Merge Scripts Only ✅ **RECOMMENDED**

**Merge:** `generate_candidates.py` + `label_in_batches.py` + `remove_duplicate_labels.py` → `ground_truth_cli.py`

**Rationale:**
- All 3 are **executable scripts** (have `if __name__ == "__main__"`)
- All 3 are **workflow tools** (not reusable libraries)
- All 3 are **sequential steps** in the ground truth workflow

**New structure:**
```python
# ground_truth_cli.py (518 lines)

class Commands:
    @staticmethod
    def export():
        """Export interventions from DB"""
        exporter = InterventionDataExporter()
        exporter.export_interventions()

    @staticmethod
    def generate():
        """Generate 500 candidate pairs"""
        # Code from generate_candidates.py

    @staticmethod
    def label():
        """Start labeling session (batch mode)"""
        # Code from label_in_batches.py

    @staticmethod
    def clean():
        """Remove duplicate labels"""
        # Code from remove_duplicate_labels.py

    @staticmethod
    def status():
        """Show labeling progress"""
        # From label_in_batches.py

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Export subcommand
    export_parser = subparsers.add_parser('export')

    # Generate subcommand
    gen_parser = subparsers.add_parser('generate')
    gen_parser.add_argument('--count', default=500)

    # Label subcommand
    label_parser = subparsers.add_parser('label')
    label_parser.add_argument('--batch-size', default=50)
    label_parser.add_argument('--start-from', default=0)

    # Clean subcommand
    clean_parser = subparsers.add_parser('clean')

    # Status subcommand
    status_parser = subparsers.add_parser('status')
```

**Usage:**
```bash
# Old way (3 separate commands):
python data_exporter.py
python generate_candidates.py
python label_in_batches.py --batch-size 50 --start-from 0
python remove_duplicate_labels.py

# New way (single CLI):
python ground_truth_cli.py export
python ground_truth_cli.py generate
python ground_truth_cli.py label --batch-size 50 --start-from 0
python ground_truth_cli.py clean
python ground_truth_cli.py status
```

**Result:**
- **Before:** 7 files (1,723 lines)
- **After:** 5 files (1,723 lines, but better organized)
- **Reduction:** 2 files removed (scripts consolidated)

**Files after consolidation:**
1. ✅ `data_exporter.py` (160 lines) - Library class (KEEP)
2. ✅ `pair_generator.py` (442 lines) - Library class (KEEP)
3. ✅ `labeling_interface.py` (559 lines) - Library class (KEEP)
4. ✅ **`ground_truth_cli.py`** (518 lines) - **NEW** (consolidates 3 scripts)
5. ✅ `__init__.py` (44 lines) - Module init (KEEP)

---

### Strategy B: Merge Everything Aggressively ❌ **NOT RECOMMENDED**

**Merge:** All 6 files → `ground_truth.py` (1,679 lines)

**Result:**
- Single 1,679-line file
- Classes + scripts mixed together
- Violates Single Responsibility Principle
- Hard to test, maintain, extend

**Verdict:** ❌ **Terrible idea**

---

## Analysis: Is Consolidation Worth It?

### Current Workflow Complexity

**User experience (current):**
```bash
# Step 1: Export data
python data_exporter.py

# Step 2: Generate pairs
python generate_candidates.py

# Step 3: Label in batches
python label_in_batches.py --batch-size 50 --start-from 0
python label_in_batches.py --batch-size 50 --start-from 50
python label_in_batches.py --batch-size 50 --start-from 100
# ... repeat for 10 batches

# Step 4: Check status
python label_in_batches.py --status

# Step 5: Clean duplicates
python remove_duplicate_labels.py
```

**User experience (after Strategy A consolidation):**
```bash
# Step 1: Export data
python ground_truth_cli.py export

# Step 2: Generate pairs
python ground_truth_cli.py generate

# Step 3: Label in batches
python ground_truth_cli.py label --batch-size 50 --start-from 0
python ground_truth_cli.py label --batch-size 50 --start-from 50
python ground_truth_cli.py label --batch-size 50 --start-from 100
# ... repeat for 10 batches

# Step 4: Check status
python ground_truth_cli.py status

# Step 5: Clean duplicates
python ground_truth_cli.py clean
```

**Improvement:** Slightly cleaner (single entry point), but not dramatically different.

---

## Pros/Cons of Strategy A (Merge Scripts)

### ✅ Pros
1. **Single CLI entry point** - Cleaner UX
2. **Fewer files** - 7 → 5 files (29% reduction)
3. **Consistent interface** - All commands use same CLI pattern
4. **Easier discovery** - `python ground_truth_cli.py --help` shows all commands
5. **Shared utilities** - Common code (arg parsing, file loading) can be factored out

### ❌ Cons
1. **Larger file** - 518 lines (manageable but larger than current scripts)
2. **More complex** - Subcommand parsing adds complexity
3. **Breaking change** - Existing documentation/scripts would need updating
4. **Testing complexity** - Single file has multiple responsibilities
5. **Import overhead** - Loading all 3 libraries even if only using one command

---

## My Recommendation: **Keep Current Structure** ✅

### Reasoning

1. **Current structure is already good**
   - Clean separation: libraries vs scripts
   - Small file sizes (112-247 lines per script)
   - Easy to understand (one file = one task)

2. **Low actual pain**
   - Users run these scripts maybe 2-3 times total (during ground truth setup)
   - Not a frequently-used workflow (one-time labeling task)
   - Scripts are simple and self-documenting

3. **Consolidation benefits are minimal**
   - UX improvement: Minor (still same number of commands)
   - Code reuse: None (scripts don't share code)
   - Maintenance: Actually harder (more complex file)

4. **Current usage pattern**
   - 80/500 pairs labeled (16% complete)
   - User runs `label_in_batches.py` occasionally
   - Other scripts rarely used after initial setup
   - **Not worth disrupting for marginal gains**

5. **Unix philosophy**
   - "Do one thing well"
   - Small, focused tools are easier to understand and debug
   - Composition > monolithic design

---

## Final Verdict

### ❌ DO NOT CONSOLIDATE

**Keep all 7 files as-is:**
1. `data_exporter.py` - Library class
2. `pair_generator.py` - Library class
3. `generate_candidates.py` - Script (uses #1 + #2)
4. `labeling_interface.py` - Library class
5. `label_in_batches.py` - Script (uses #4)
6. `remove_duplicate_labels.py` - Script (standalone)
7. `__init__.py` - Module init

**Reasons:**
- ✅ Already well-organized (libraries vs scripts)
- ✅ Appropriate file sizes (112-559 lines)
- ✅ Clear single responsibilities
- ✅ Low coupling, high cohesion
- ✅ Easy to test and maintain
- ✅ Rarely used workflow (one-time ground truth setup)
- ✅ Consolidation provides minimal benefit
- ✅ Would violate Single Responsibility Principle

---

## Alternative: If You MUST Consolidate

**If you really want fewer files, the ONLY acceptable merge is:**

### Merge Scripts → `ground_truth_cli.py` (Strategy A)

**Merge these 3 scripts:**
- `generate_candidates.py` (159 lines)
- `label_in_batches.py` (247 lines)
- `remove_duplicate_labels.py` (112 lines)

**Into:** `ground_truth_cli.py` (518 lines with subcommands)

**Keep these as libraries:**
- `data_exporter.py` - Reusable class
- `pair_generator.py` - Reusable class
- `labeling_interface.py` - Reusable class

**Result:** 7 files → 5 files (29% reduction)

**BUT:** I still recommend keeping current structure. The benefit doesn't justify the refactoring effort.

---

## Summary

| Strategy | Files | Lines | Pros | Cons | Recommendation |
|----------|-------|-------|------|------|----------------|
| **Current** | 7 | 1,723 | Simple, clear, follows SRP | More files | ✅ **KEEP** |
| **Strategy A** (Merge scripts) | 5 | 1,723 | Single CLI, fewer files | Larger file, more complex | ⚠️ Acceptable |
| **Strategy B** (Merge all) | 2 | 1,679 | Fewest files | Monolithic, violates SRP | ❌ **NO** |

**Final Answer:** Keep current structure. If you must merge, use Strategy A (scripts only).