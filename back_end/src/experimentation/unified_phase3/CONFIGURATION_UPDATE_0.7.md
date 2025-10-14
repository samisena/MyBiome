# Configuration Update: distance_threshold = 0.7

**Date**: 2025-10-14
**Author**: Threshold Experiment Analysis (0.4-0.8 comparison)

## Summary

Updated the Phase 3b clustering configuration to use hierarchical clustering with `distance_threshold=0.7` for all three entity types (interventions, conditions, mechanisms). This is based on comprehensive experiments comparing thresholds 0.4, 0.5, 0.6, 0.7, and 0.8.

## Changes Made

### 1. base_config.yaml
**File**: `back_end/src/experimentation/unified_phase3/config/base_config.yaml`

**Changes**:
- **Interventions**: Changed from `algorithm: "hdbscan"` to `algorithm: "hierarchical"` with `distance_threshold: 0.7`
- **Conditions**: Changed from `algorithm: "hdbscan"` to `algorithm: "hierarchical"` with `distance_threshold: 0.7`
- **Mechanisms**: Changed from `algorithm: "hdbscan"` to `algorithm: "hierarchical"` with `distance_threshold: 0.7`

**Before**:
```yaml
clustering:
  interventions:
    algorithm: "hdbscan"
    distance_threshold: null

  conditions:
    algorithm: "hdbscan"

  mechanisms:
    algorithm: "hdbscan"
```

**After**:
```yaml
clustering:
  interventions:
    algorithm: "hierarchical"
    linkage: "ward"
    distance_threshold: 0.7    # Optimal threshold from experiments (0.4-0.8)

  conditions:
    algorithm: "hierarchical"
    linkage: "ward"
    distance_threshold: 0.7    # Optimal threshold from experiments

  mechanisms:
    algorithm: "hierarchical"
    linkage: "ward"
    distance_threshold: 0.7    # Optimal threshold from experiments
```

### 2. hierarchical_clusterer.py
**File**: `back_end/src/experimentation/unified_phase3/clusterers/hierarchical_clusterer.py`

**Changes**:
- Line 62: Changed default from `0.5` to `0.7`
- Line 63: Updated log message to reflect 0.7 as optimal

**Before**:
```python
if distance_threshold is None and n_clusters is None:
    distance_threshold = 0.5
    logger.info("Neither distance_threshold nor n_clusters set, using distance_threshold=0.5")
```

**After**:
```python
if distance_threshold is None and n_clusters is None:
    distance_threshold = 0.7
    logger.info("Neither distance_threshold nor n_clusters set, using distance_threshold=0.7 (optimal)")
```

## Justification

Based on comprehensive threshold experiments documented in:
- [threshold_analysis/THRESHOLD_0.7_vs_0.8_FINAL_RECOMMENDATION.md](threshold_analysis/THRESHOLD_0.7_vs_0.8_FINAL_RECOMMENDATION.md)
- [threshold_analysis/OVERMERGING_EXAMPLES_0.8.md](threshold_analysis/OVERMERGING_EXAMPLES_0.8.md)
- [threshold_analysis/compare_0.7_vs_0.8.py](threshold_analysis/compare_0.7_vs_0.8.py)

### Quantitative Evidence

**Silhouette Score (higher is better):**
- Interventions: 0.118 (0.7) → 0.135 (0.8) = +14.8% [0.8 wins]
- Conditions: 0.104 (0.7) → 0.118 (0.8) = +13.5% [0.8 wins]
- Mechanisms: 0.134 (0.7) → 0.146 (0.8) = +9.3% [0.8 wins]

**Davies-Bouldin Score (lower is better):**
- Interventions: 0.588 (0.7) → 0.849 (0.8) = +44.5% [0.7 wins]
- Conditions: 0.586 (0.7) → 0.864 (0.8) = +47.5% [0.7 wins]
- Mechanisms: 0.666 (0.7) → 0.903 (0.8) = +35.6% [0.7 wins]

**Cluster Counts:**
- Interventions: 538 (0.7) vs 420 (0.8) = 22% reduction
- Conditions: 330 (0.7) vs 251 (0.8) = 24% reduction
- Mechanisms: 130 (0.7) vs 101 (0.8) = 22% reduction

### Qualitative Evidence

**0.8 Over-Merging Examples:**

1. **Cognitive Function Mechanisms (SEVERE)**:
   - 0.7: 8 distinct clusters (VR training, exercise-based, CBT therapy, digital training, etc.)
   - 0.8: 1 mega-cluster (loses critical distinctions)
   - **Impact**: Cannot distinguish between VR-based training ($500+ headset) vs exercise (gym) vs CBT (licensed therapist)

2. **Cancer Treatment Mechanisms (MODERATE)**:
   - 0.7: Keeps electric field therapy (Tumor Treating Fields) separate from chemotherapy
   - 0.8: Merges non-chemical mechanisms into cytotoxic chemotherapy cluster
   - **Impact**: Loses mechanistically distinct treatment modalities

3. **Anti-Inflammatory Mechanisms (APPROPRIATE SEPARATION)**:
   - 0.7: Preserves corticosteroid pathway vs antioxidant pathway vs cytokine blockade
   - 0.8: Merges pathways with different side effect profiles and clinical uses

### Decision Rationale

**Why 0.7 > 0.8:**
1. **Medical domain requires precision**: Better to keep distinct mechanisms separate than risk inappropriate merging
2. **Davies-Bouldin degradation is severe**: +36-48% loss of cluster separation
3. **Silhouette improvement is modest**: +9-15% gain in cohesion doesn't justify separation loss
4. **Real-world impact**: VR training, exercise-based training, and CBT therapy have completely different:
   - Implementation costs
   - Required equipment/expertise
   - Target populations
   - Biological mechanisms

**Why 0.7 > 0.6:**
From previous comparison (documented in threshold_analysis/):
- 0.7 has 26% better silhouette score than 0.6
- 0.7 shows superior semantic grouping (e.g., H. pylori antibiotics properly consolidated)
- 0.6 exhibited over-fragmentation (behavioral adherence split into 2 members instead of 5)

## Impact on Production Pipeline

### Before This Update:
- Default algorithm: HDBSCAN (density-based, non-deterministic)
- Hierarchical clustering: Used 0.5 threshold as fallback

### After This Update:
- Default algorithm: Hierarchical AgglomerativeClustering (deterministic, 100% assignment)
- Distance threshold: 0.7 (optimal for medical domain)
- Benefits:
  - Consistent results across runs
  - 100% entity assignment (no noise points)
  - Appropriate balance between consolidation and separation

## Testing Recommendations

1. **Create test experiment** inheriting from base_config.yaml:
   ```bash
   cd back_end/src/experimentation/unified_phase3
   python orchestrator.py --config config/base_config.yaml
   ```

2. **Verify threshold is applied**:
   - Check logs for: "distance_threshold=0.7"
   - Verify cluster counts are similar to experiments 18, 22, 26 (0.7 threshold experiments)

3. **Compare with threshold 0.8**:
   - Review cluster member lists for inappropriate merging
   - Spot-check cognitive function, cancer, and anti-inflammatory clusters

## Rollback Plan

If 0.7 produces unexpected results in production:

1. **Revert to HDBSCAN** (previous default):
   ```yaml
   clustering:
     interventions:
       algorithm: "hdbscan"
       min_cluster_size: 2
   ```

2. **Use threshold 0.6** (more conservative):
   ```yaml
   clustering:
     interventions:
       algorithm: "hierarchical"
       distance_threshold: 0.6
   ```

3. **Test threshold 0.75** (intermediate):
   - May balance silhouette vs Davies-Bouldin trade-off
   - Would require new experiments

## References

- Experiments 10-17: Thresholds 0.4-0.7 for interventions and conditions
- Experiments 18-21: Thresholds 0.4-0.7 for mechanisms
- Experiments 22-24: Threshold 0.8 (not recommended)
- Experiments 27-29: Threshold 0.8 full analysis (completed)

## Notes

- Experiment configs (exp_010-024) preserved as-is for historical record
- This update affects **default production behavior** only
- Individual experiments can still override with custom thresholds
- Cache keys include hyperparameters, so 0.7 results won't conflict with 0.5/0.6/0.8 cached results
