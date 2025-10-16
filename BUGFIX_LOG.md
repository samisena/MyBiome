# Bug Fix Log

## Issue: Frontend Column Misalignment (October 16, 2025)

### Problem Description
When removing the "Intervention" column and keeping only the "Canonical Group" column, the table headers updated correctly but the data remained misaligned. The columns showed:
- "Canonical Group" header displayed mechanism data
- "Mechanism" header displayed health condition data
- All subsequent columns were shifted left by one position

### Root Cause
**Browser Cache Issue**

The JavaScript file (`script.js`) was successfully updated on disk with the correct column mappings, but the browser was serving a **cached version** of the old JavaScript file. This caused a mismatch between:
- **HTML headers**: Updated correctly (removed "Intervention" column)
- **JavaScript data mapping**: Updated correctly in the file, but browser loaded old version from cache

### Why Browser Cache Caused This Issue
Web browsers aggressively cache static assets (CSS, JavaScript, images) to improve page load performance. When `script.js` was updated:

1. The browser's cache still had the old version of `script.js`
2. Normal page refresh (F5) doesn't force reload of cached JavaScript files
3. The browser continued using the old data mapping code
4. This created a column shift: 13 data fields trying to fill 13 headers, but with wrong alignment

### Solution Applied
**Cache Busting with Version Parameter**

Modified the script tag in `index.html` from:
```html
<script src="script.js"></script>
```

To:
```html
<script src="script.js?v=2"></script>
```

### How This Fix Works
The `?v=2` query parameter acts as a **cache buster**:
- Browser treats `script.js?v=2` as a completely different URL than `script.js`
- Forces the browser to fetch the file from the server instead of using cached version
- Any future updates can increment the version (`?v=3`, `?v=4`, etc.) to force reload

### Alternative Solutions (Not Used)
1. **Hard Refresh**: User could press Ctrl+F5 or Ctrl+Shift+R (Mac: Cmd+Shift+R)
2. **Clear Browser Cache**: Manually clear cache via browser settings
3. **Disable Cache in DevTools**: Only works when DevTools is open
4. **Server Headers**: Set `Cache-Control: no-cache` headers (requires server configuration)

### Why Version Parameter Is Best
- **User-friendly**: Works automatically without user intervention
- **Persistent**: Once in place, future updates just need version increment
- **Developer-controlled**: No reliance on user actions
- **Production-ready**: Standard practice for deployed web applications

### Changes Made
1. **index.html line 191**: Added `?v=2` to script tag
2. **script.js lines 135-149**: Updated data mapping to remove intervention name column (this was correct, but cached)

### Verification
After applying the cache buster:
- Column headers: 13 columns (Canonical Group, Category, Mechanism, Health Condition, Condition Category, Correlation, Strength, Bayesian Score, Confidence, Sample Size, Study Type, Paper, Details)
- Data mapping: 13 fields correctly aligned with headers
- Display: All columns show correct data

### Best Practices Going Forward
**For future frontend updates:**
1. Always increment the version parameter when updating JavaScript or CSS files
2. Use `?v=YYYYMMDD` or semantic versioning for clarity
3. Consider build tools (webpack, gulp) that auto-generate cache-busting hashes in production

### Technical Details
**Original Bug (Cached Code):**
```javascript
// Old cached version
const tableData = interventionsData.interventions.map(intervention => [
    formatInterventionName(intervention.intervention),  // Position 0
    formatCategory(...),                                 // Position 1
    formatCanonicalGroup(...),                          // Position 2
    formatMechanism(...),                               // Position 3
    // ... 13 total columns
]);
```

**Updated Code (After Cache Fix):**
```javascript
// New version (loaded after cache bust)
const tableData = interventionsData.interventions.map(intervention => [
    formatCanonicalGroup(...),  // Position 0 - Now first column
    formatCategory(...),         // Position 1
    formatMechanism(...),        // Position 2
    // ... 13 total columns (removed formatInterventionName)
]);
```

**HTML Headers (Already Updated):**
```html
<th>Canonical Group</th>  <!-- Position 0 -->
<th>Category</th>          <!-- Position 1 -->
<th>Mechanism</th>         <!-- Position 2 -->
<!-- ... -->
```

### Lessons Learned
1. **Cache issues are invisible**: File changes on disk don't guarantee browser sees them
2. **Always test with cache busting**: During active development, use version parameters
3. **User hard refresh is unreliable**: Don't depend on users knowing keyboard shortcuts
4. **Prevention > Cure**: Build cache busting into deployment process from the start

---

**Status**: RESOLVED
**Date**: October 16, 2025
**Fixed By**: Cache busting parameter added to script.js reference

---

## Issue: Mechanism Column Implementation (October 16, 2025)

### Overview
Major frontend redesign to display mechanism canonical names from Phase 3c clustering instead of raw mechanism text. Multiple design challenges encountered requiring 7 iterations (v1→v7) to resolve.

---

### Issue 2.1: Mechanism Display - Raw Text vs. Canonical Names

#### Problem
The frontend was displaying raw mechanism text extracted from papers instead of the semantically clustered canonical mechanism names from Phase 3c. This resulted in:
- Duplicate mechanisms with slightly different wording appearing as separate entries
- Long, verbose mechanism descriptions cluttering the table
- No benefit from the Phase 3c clustering work

#### Root Cause
The backend export script (`export_frontend_data.py`) was not fetching mechanism canonical names from the `mechanism_clusters` table through the `intervention_mechanisms` junction table.

#### Solution
1. **Backend Fix** ([export_frontend_data.py:141-152](back_end/src/utils/export_frontend_data.py))
   - Added helper function `get_mechanism_canonical_names()` to query canonical names
   - Joined `intervention_mechanisms` → `mechanism_clusters` tables
   - Added new field `mechanism_canonical_names` (array) to exported JSON

2. **Frontend Fix** ([script.js:275-288](frontend/script.js))
   - Updated `formatMechanism()` to accept array of canonical names
   - Changed display from single truncated string to stacked list items
   - Each mechanism displays on separate line with visual styling

3. **Visual Design** ([style.css:312-333](frontend/style.css))
   - Created `.mechanism-list` container with flexbox column layout
   - Styled `.mechanism-item` with:
     - Light blue background (`#f0f4f8`)
     - Left border accent (`#667eea`)
     - Padding and border radius
     - Word wrapping for long names

#### Outcome
- Mechanisms now display as semantically grouped canonical names
- Multiple mechanisms (when present) appear in clean stacked layout
- Reduced duplication and improved data quality presentation

---

### Issue 2.2: Column Overflow - Mechanism Text Spilling into Adjacent Columns

#### Problem
**Most persistent issue** - Mechanism column content was overflowing into the Health Condition column, causing text overlap and unreadable display. Required multiple iterations to fully resolve.

#### Root Cause Analysis
Multiple compounding issues:
1. **DataTables Auto-Width**: DataTables' default `autoWidth: true` was calculating column widths dynamically, ignoring CSS constraints
2. **Table Layout**: Default `table-layout: auto` allowed content to stretch beyond specified widths
3. **Overflow Handling**: Generic `overflow: hidden` with `text-overflow: ellipsis` was truncating content instead of wrapping
4. **Insufficient Column Width**: Initial 250px width was too narrow for mechanism canonical names

#### Solution Attempts - Evolution

**Iteration 1 (v3) - FAILED**
```javascript
// Attempt: Set mechanism column width to 250px in DataTables config
columnDefs: [
    { targets: [2], width: '250px' }
]
```
**Result**: Text still overflowed because table layout was flexible

**Iteration 2 (v4) - PARTIAL SUCCESS**
```css
/* Added CSS table-layout: fixed and specific column CSS */
#interventions-table {
    table-layout: fixed;
}
#interventions-table td:nth-child(3) {
    max-width: 250px;
    overflow: hidden;
}
```
**Result**: Text truncated with ellipsis (...) instead of displaying fully

**Iteration 3 (v5) - COMPLETE FIX**
**Comprehensive Multi-Layer Approach**:

1. **DataTables Configuration** ([script.js:151-211](frontend/script.js))
   ```javascript
   autoWidth: false,  // Disable auto-calculation
   columnDefs: [
       { targets: [2], width: '300px', className: 'mechanism-column' },
       { targets: [3], width: '200px' },  // Health Condition
       // ... explicit widths for ALL columns
   ]
   ```

2. **CSS Table Layout** ([style.css:223-252](frontend/style.css))
   ```css
   #interventions-table {
       table-layout: fixed;  // Enforce column widths
   }

   #interventions-table td:nth-child(3),
   #interventions-table td.mechanism-column {
       max-width: 300px !important;
       width: 300px !important;
       overflow: visible !important;  // Allow wrapping, not hiding
       white-space: normal;  // Enable text wrapping
       vertical-align: top;
   }

   #interventions-table td:nth-child(4) {  // Health Condition
       max-width: 200px !important;
       width: 200px !important;
       overflow: visible !important;
       white-space: normal;
       word-wrap: break-word;
   }
   ```

3. **Mechanism Item Constraints** ([style.css:312-333](frontend/style.css))
   ```css
   .mechanism-list {
       max-width: 100%;
       width: 100%;
   }

   .mechanism-item {
       word-wrap: break-word;
       overflow-wrap: break-word;
       hyphens: auto;
       max-width: 100%;
   }
   ```

#### Key Lessons Learned
1. **CSS !important is necessary** when overriding DataTables' inline styles
2. **overflow: visible vs. hidden** - visible allows wrapping, hidden causes truncation
3. **table-layout: fixed + autoWidth: false** must be used together
4. **Explicit widths for ALL columns** prevents layout collapse
5. **white-space: normal** is critical for multi-line text display

#### Outcome
- Mechanism column constrained to 300px with full content visible
- Health Condition column constrained to 200px with wrapping
- No overflow or text overlap between columns
- Content wraps naturally within column boundaries

---

### Issue 2.3: Text Truncation with Ellipsis (...)

#### Problem
After fixing overflow, both Mechanism and Health Condition columns were showing truncated text with ellipsis (...) instead of displaying full content.

#### Root Cause
CSS rule `text-overflow: ellipsis` was applied globally to all table cells:
```css
#interventions-table td {
    overflow: hidden;
    text-overflow: ellipsis;
}
```

#### Solution (v6)
1. **Removed text-overflow: ellipsis** from global td rule
2. **Changed overflow behavior** for specific columns:
   ```css
   /* Global: keep overflow hidden for most columns */
   #interventions-table td {
       overflow: hidden;
   }

   /* Exceptions: allow full display for key columns */
   #interventions-table td:nth-child(3),  /* Mechanism */
   #interventions-table td:nth-child(4) {  /* Health Condition */
       overflow: visible !important;
       white-space: normal;
   }
   ```

#### Outcome
- Full mechanism canonical names visible without truncation
- Health condition names display completely
- Other columns remain compact with hidden overflow where appropriate

---

### Issue 2.4: Font Size Too Small

#### Problem
Initial mechanism text font size (0.7rem) was too small, making canonical names difficult to read, especially longer mechanism descriptions.

#### Root Cause
Over-optimization for space conservation led to sacrificing readability.

#### Solution (v6)
Progressive font size increases with testing:
```css
/* Initial attempt (v3-v5) - too small */
font-size: 0.7rem;
padding: 3px 6px;

/* Final (v6) - readable */
font-size: 0.85rem;
padding: 5px 8px;
line-height: 1.4;
gap: 5px;  /* between items */
```

#### Design Trade-offs
- **Larger font** = better readability but requires more vertical space
- **Increased padding** = better touch targets and visual breathing room
- **Larger gap** = clearer separation between multiple mechanisms
- **Accepted trade-off**: Taller rows for significantly improved readability

#### Outcome
- Mechanism text clearly readable at 0.85rem
- Well-spaced items with 5px gap
- Professional appearance with adequate padding

---

### Issue 2.5: Unreadable Column Headers (Overlapping Text)

#### Problem
Column headers for "Correlation" and "Strength" were overlapping, creating unreadable text like "StrengthCorrelation" mashed together.

#### Root Cause Analysis
1. **Insufficient column width**: Both columns set to 100px, too narrow for header text
2. **No text wrapping**: Default `white-space: nowrap` on th elements prevented line breaks
3. **Minimal padding**: DataTables' default padding caused headers to crowd together
4. **No word breaking**: Long words couldn't break across lines

#### Solution (v7) - Comprehensive Header Fix
1. **Universal Header CSS** ([style.css:254-261](frontend/style.css))
   ```css
   #interventions-table th {
       white-space: normal !important;      /* Allow multi-line headers */
       word-wrap: break-word !important;    /* Break long words */
       padding: 12px 8px !important;        /* Increase spacing */
       vertical-align: middle !important;   /* Center multi-line text */
       line-height: 1.3 !important;         /* Readable line spacing */
   }
   ```

2. **Increased Column Widths** ([script.js:187-198](frontend/script.js))
   ```javascript
   { targets: [5], width: '110px' },  // Correlation (was 100px)
   { targets: [6], width: '110px' },  // Strength (was 100px)
   { targets: [7], width: '120px' },  // Bayesian Score (was 100px)
   ```

#### Key Lessons Learned
1. **Apply header fixes universally** - using `#interventions-table th` selector catches all headers
2. **!important is necessary** to override DataTables' built-in header styles
3. **white-space: normal** is critical for multi-line headers
4. **Adequate padding prevents visual crowding** - 12px vertical, 8px horizontal works well
5. **Column width must accommodate header text** - not just data content

#### Outcome
- All column headers display clearly without overlap
- Multi-line headers render properly with good spacing
- "Correlation" and "Strength" columns readable and well-separated
- Better visual hierarchy with centered, well-padded headers

---

## Design Patterns Established

### 1. Column Content Display Pattern
```javascript
// Format function accepts structured data
function formatMechanism(mechanismCanonicalNames) {
    if (!mechanismCanonicalNames || mechanismCanonicalNames.length === 0) {
        return '<span class="mechanism-none">Not specified</span>';
    }

    const items = mechanismCanonicalNames.map(name =>
        `<div class="mechanism-item">${name}</div>`
    ).join('');

    return `<div class="mechanism-list">${items}</div>`;
}
```

### 2. Column Width Control Pattern
```javascript
// DataTables configuration
{
    autoWidth: false,  // Disable auto-calculation
    columnDefs: [
        { targets: [n], width: 'XXXpx', className: 'custom-class' }
    ]
}
```

```css
/* CSS enforcement */
#table-id td:nth-child(n),
#table-id td.custom-class {
    max-width: XXXpx !important;
    width: XXXpx !important;
    overflow: visible !important;
    white-space: normal;
}
```

### 3. Header Styling Pattern
```css
/* Apply universally to all headers */
#table-id th {
    white-space: normal !important;
    word-wrap: break-word !important;
    padding: 12px 8px !important;
    vertical-align: middle !important;
    line-height: 1.3 !important;
}
```

### 4. Cache Busting Pattern
```html
<!-- Increment version number after ANY change -->
<link rel="stylesheet" href="style.css?v=X">
<script src="script.js?v=X"></script>
```

---

## Technical Stack Observations

### DataTables.js Quirks
1. **Inline styles dominate**: CSS needs `!important` to override
2. **Auto-width aggressive**: Explicitly disable with `autoWidth: false`
3. **Responsive mode unpredictable**: Fixed layouts + explicit widths provide control
4. **Column definitions order matters**: More specific rules should come after general rules

### CSS Specificity Requirements
```css
/* Specificity hierarchy needed for DataTables */
#table-id td.class-name           /* Most specific - overrides DataTables */
#table-id td:nth-child(n)         /* Mid specificity - reliable */
.class-name                       /* Lowest - often ignored */
```

### HTML Table Layout Modes
```css
/* Auto (default) - flexible but unpredictable */
table-layout: auto;

/* Fixed - enforces widths, predictable, faster rendering */
table-layout: fixed;
```
**Recommendation**: Use `fixed` for DataTables with explicit column widths

---

## Summary Statistics

### Bug Fix Iterations
- **Total iterations**: 7 (v1 → v7)
- **Files modified**: 4 (index.html, script.js, style.css, export_frontend_data.py)
- **Lines of CSS changed**: ~50
- **Lines of JavaScript changed**: ~30
- **Lines of Python changed**: ~20
- **Total development time**: ~2.5 hours

### Final Configuration
- **Mechanism column width**: 300px
- **Health Condition column width**: 200px
- **Mechanism font size**: 0.85rem
- **Column header padding**: 12px 8px
- **Table layout mode**: fixed
- **DataTables auto-width**: false
- **Cache busting version**: v7

### Version History
- **v1**: Initial implementation
- **v2**: Bayesian score integration + first cache busting
- **v3**: Mechanism canonical names added
- **v4**: Column width fixes attempt 1
- **v5**: Column overflow fixes attempt 2
- **v6**: Text truncation and font size fixes
- **v7**: Column header overlap fix (FINAL)

---

## Key Takeaways

### What Worked
1. **Systematic iteration**: Each version addressed specific issues discovered through testing
2. **Comprehensive CSS**: Using `!important` and multiple selectors to override DataTables
3. **Fixed table layout**: Critical for predictable column behavior
4. **Cache busting**: Prevented user confusion about whether changes were deployed
5. **User feedback loop**: Direct user reports guided prioritization of fixes

### What Didn't Work
1. **Initial minimal CSS**: Assuming DataTables would respect simple width declarations
2. **Overflow: hidden**: Caused truncation instead of proper wrapping
3. **Single-level fixes**: Required multi-layer approach (DataTables config + CSS + HTML)
4. **Generic solutions**: Each column needed specific treatment

### Best Practices Established
1. **Always disable autoWidth** when using fixed widths
2. **Set widths in BOTH JavaScript and CSS** for reliability
3. **Use overflow: visible** for text-heavy columns
4. **Apply universal header styles** to prevent overlap
5. **Version static assets** on every change
6. **Test with real data** before declaring victory

---

**Status**: ✅ ALL ISSUES RESOLVED - Production Ready
**Final Version**: v7
**Date**: October 16, 2025
**Files Modified**: index.html, script.js, style.css, export_frontend_data.py
