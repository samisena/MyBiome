# Threshold Comparison Test for Phase 3 Semantic Normalization

## Overview

This test compares three similarity thresholds (0.70, 0.65, 0.60) to optimize Phase 3 semantic normalization performance.

## What the Test Does

### 1. **Stratified Sampling**
- Loads 50 interventions across all 13 categories (proportional sampling)
- Ensures representative coverage of medication, supplement, diet, therapy, etc.

### 2. **Threshold Testing**
For each threshold (0.70, 0.65, 0.60):
- Generates embeddings for sampled interventions
- Finds similar interventions above the threshold
- Classifies relationships using LLM (qwen3:14b)
- Tracks performance metrics

### 3. **Metrics Collected**

**Per Threshold:**
- Total candidate pairs generated
- Average candidates per intervention
- Number of LLM classification calls
- Relationship type distribution (EXACT_MATCH, VARIANT, SUBTYPE, etc.)
- Processing time
- Category-specific performance

**Comparison Analysis:**
- Candidate increase percentage (0.65 vs 0.70, 0.60 vs 0.70)
- LLM call increase (cost impact)
- Category sensitivity to threshold changes
- Recommendations for optimal threshold

### 4. **Outputs**

**Console Output:**
- Real-time progress for each intervention
- Summary table comparing all thresholds
- Relationship type distribution
- Performance impact analysis
- Top recommendations

**Saved Files** (in `back_end/data/threshold_analysis/`):
- `threshold_test_results_YYYYMMDD_HHMMSS.json` - Full data
- `threshold_comparison_report_YYYYMMDD_HHMMSS.txt` - Human-readable report

## How to Run

### Basic Usage

```bash
# Activate conda environment
conda activate venv

# Run the test
python -m back_end.scripts.test_threshold_comparison
```

### Expected Runtime

- **Sample size**: 50 interventions
- **LLM calls**: ~150-300 (depending on candidates found)
- **Estimated time**: 60-120 minutes
  - Embeddings: Fast (cached after first run)
  - LLM classification: ~25s per call
  - Total: ~60-120 min for 150-300 calls

### Customization

Edit these constants in the script:

```python
THRESHOLDS_TO_TEST = [0.70, 0.65, 0.60]  # Add more thresholds
SAMPLE_SIZE = 50                         # Increase for more data
TOP_K_SIMILAR = 5                        # Max candidates per intervention
```

## Interpreting Results

### Key Metrics to Watch

**1. Candidate Increase**
```
Threshold 0.70: 120 candidates (2.4 avg/intervention)
Threshold 0.65: 180 candidates (3.6 avg/intervention) [+50%]
Threshold 0.60: 250 candidates (5.0 avg/intervention) [+108%]
```
- **Low increase (<30%)**: Safe to lower threshold
- **Moderate increase (30-60%)**: Test with ground truth first
- **High increase (>60%)**: Risk of false positives

**2. Relationship Type Distribution**
```
Threshold 0.70:
  VARIANT: 45 (37.5%)
  EXACT_MATCH: 30 (25.0%)
  SUBTYPE: 25 (20.8%)
  DIFFERENT: 10 (8.3%)
  SAME_CATEGORY: 8 (6.7%)
  DOSAGE_VARIANT: 2 (1.7%)
```
- Higher **DIFFERENT** percentage = more false positives
- Lower **VARIANT/EXACT_MATCH** = missing relationships
- Optimal: High VARIANT/EXACT_MATCH, low DIFFERENT

**3. Category Sensitivity**
```
Category 'supplement':
  0.70: 30 candidates
  0.65: 50 candidates (+67%)
  0.60: 75 candidates (+150%)
```
- High sensitivity → Consider adaptive threshold for this category
- Low sensitivity → Single threshold works well

**4. LLM Call Increase**
```
0.65 vs 0.70: +50% LLM calls
0.60 vs 0.70: +108% LLM calls
```
- Direct cost impact (each call ~25s processing time)
- 100% increase = 2x processing time and GPU usage

### Decision Guidelines

**Choose 0.65 if:**
- Candidate increase <40%
- DIFFERENT relationships stay <15%
- You want better recall (catch more variants)
- Processing time increase is acceptable

**Choose 0.60 if:**
- Candidate increase <60%
- You need maximum recall
- You can tolerate more false positives
- You have time for extensive ground truth validation

**Keep 0.70 if:**
- Lower thresholds show >60% candidate increase
- DIFFERENT relationships spike significantly
- Current precision is satisfactory
- You want conservative, high-precision grouping

## Adaptive Thresholds (Advanced)

After analyzing results, consider implementing **category-specific thresholds**:

### Example Configuration (based on test results)

```python
ADAPTIVE_THRESHOLDS = {
    # Strict categories (avoid over-grouping)
    'medication': 0.75,
    'biologics': 0.75,
    'device': 0.75,
    'surgery': 0.75,
    'gene_therapy': 0.80,

    # Moderate categories
    'therapy': 0.70,
    'procedure': 0.72,
    'test': 0.70,

    # Loose categories (many variants)
    'supplement': 0.65,
    'diet': 0.65,
    'exercise': 0.60,
    'lifestyle': 0.60,
    'emerging': 0.68,
}
```

### How to Identify Optimal Category Thresholds

1. **Run this test** with current thresholds
2. **Review category-specific performance** in JSON output
3. **Look for categories** with:
   - High candidate increase → Needs stricter threshold
   - Low candidate increase → Can use looser threshold
   - High DIFFERENT percentage → Too loose
4. **Adjust** thresholds per category
5. **Re-test** with adaptive thresholds

## Next Steps After Testing

### 1. Validate with Ground Truth
- Label more pairs in `ground_truth/labeling_session.json` (currently 80/500)
- Run `evaluator.py` to compare automated vs human labels
- Calculate precision/recall for each threshold

### 2. Implement Adaptive Thresholds
- Modify [normalizer.py:256](back_end/src/semantic_normalization/normalizer.py#L256)
- Add category-to-threshold mapping
- Test with full pipeline

### 3. Full Pipeline Test
- Run full normalization with new threshold
- Compare canonical groups created
- Validate relationship quality

## Troubleshooting

**Issue: "LLM timeout"**
- Increase timeout in `llm_classifier.py`
- Check Ollama server status: `ollama list`

**Issue: "Cache not found"**
- First run generates embeddings (slower)
- Subsequent runs use cache (faster)

**Issue: "Memory error"**
- Reduce `SAMPLE_SIZE` to 25 or 30
- Close other applications

**Issue: "Too slow"**
- Reduce `SAMPLE_SIZE` for quick test (e.g., 20)
- Results still informative with smaller sample

## Files Modified

None - this is a standalone test script that reads from the database without modifying it.

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Full pipeline documentation
- [semantic_normalization/README.md](../src/semantic_normalization/README.md) - Phase 3 details
- [normalizer.py](../src/semantic_normalization/normalizer.py) - Implementation
- [embedding_engine.py](../src/semantic_normalization/embedding_engine.py) - Similarity calculation

---

*Last Updated: October 12, 2025*