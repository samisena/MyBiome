# Phase 2 Batch Size Optimization - Quick Reference

## Overview

**Goal**: Find the optimal batch size for Phase 2 extraction (Sequential architecture)

**Given**: Phase 2.5 batching (20 items) is already highly efficient - no optimization needed

**Approach**: Test 4 batch sizes with 16 papers (controlled comparison)

---

## Experiments (4 Total)

### Experimental Control
- **Total papers**: 16 papers (same for all experiments)
- **Architecture**: Sequential (Phase 2 → Phase 2.5)
- **Variable**: Batch size only

### Experiment Matrix

| Experiment ID | Batch Size | Batches Needed | Papers per Batch | Expected Phase 2 Time |
|--------------|-----------|----------------|------------------|---------------------|
| **EXP-001** | 4 | 4 batches | 4+4+4+4 = 16 | 16 × 93s = 1,488s (~25 min) |
| **EXP-002** | 8 | 2 batches | 8+8 = 16 | 16 × 93s = 1,488s (~25 min) |
| **EXP-003** | 12 | 2 batches | 12+4 = 16 | 16 × 93s = 1,488s (~25 min) |
| **EXP-004** | 16 | 1 batch | 16 = 16 | 16 × 93s = 1,488s (~25 min) |

**Phase 2.5 time** (same for all): ~50s (assumes ~64 interventions ÷ 20 per batch)

**Total expected time per experiment**: ~26 minutes

---

## Key Metrics

### Primary Metric (Speed)
- **Total pipeline time**: Phase 2 + Phase 2.5
- **Papers per hour**: 3600 ÷ (total_time ÷ 16 papers)
- **Throughput efficiency**: Papers/hour ÷ GPU memory utilization

### Secondary Metrics (Quality)
- **Field extraction completeness**: % of required fields populated
- **JSON parse success rate**: % responses successfully parsed
- **Validation pass rate**: % interventions passing validator
- **Error rate**: % papers failing extraction

### Safety Metrics (System)
- **GPU temperature**: Max and average during experiment
- **GPU memory peak**: Maximum memory usage
- **Thermal throttling events**: Count of cooling pauses
- **Processing stability**: Variance in per-paper timing

---

## Test Dataset

**16 papers** stratified by:
- 4 high citation (>100 citations) - complex studies
- 4 medium citation (20-100) - typical studies
- 4 low citation (<20) - simple studies
- 2 full-text available - comprehensive extraction
- 2 abstract-only - minimal extraction

Covering:
- Diverse condition categories
- Diverse intervention categories

---

## Hypotheses

### H1: Larger Batches = Better Throughput (Within Limits)
**Prediction**: Batch size 12-16 will be optimal
- Larger batches reduce overhead between batches
- Single-model architecture can handle larger batches than dual-model
- Current default (8) may be suboptimal

**Risk**: Very large batches (16) may cause:
- GPU memory pressure
- Thermal throttling
- Increased error rate

### H2: Batch Size vs System Stability
**Prediction**: Batch size 8-12 balances speed and stability
- Smaller batches (4) = more stable but slower (more overhead)
- Larger batches (16) = faster but less stable (thermal/memory issues)
- Sweet spot likely at 12

### H3: Diminishing Returns
**Prediction**: Gains plateau after batch size 12
- Overhead reduction benefit diminishes
- Thermal/memory constraints kick in
- Batch size 12 vs 16 will show minimal time difference

---

## Expected Outcomes

### Scenario 1: Batch 12 Optimal (Most Likely)
**Why**: Balances throughput, memory, and thermal safety
- 25% fewer batches than batch=4
- Still manages thermal limits well
- GPU memory stays under 90%

**Recommendation**: Update `optimal_batch_size` from 8 → 12

### Scenario 2: Batch 16 Optimal (Possible)
**Why**: System has sufficient cooling and memory headroom
- 75% fewer batches than batch=4
- Maximum throughput
- GPU can handle sustained load

**Recommendation**: Update `optimal_batch_size` from 8 → 16
**Caveat**: Monitor thermal status in production

### Scenario 3: Batch 8 Already Optimal (Status Quo)
**Why**: Current batch size hits sweet spot
- No significant gains from larger batches
- Current configuration already optimized

**Recommendation**: Keep `optimal_batch_size` at 8

### Scenario 4: Batch 4 More Stable (Conservative)
**Why**: Larger batches cause thermal or memory issues
- Better error rate with smaller batches
- More predictable performance

**Recommendation**: Reduce `optimal_batch_size` from 8 → 4 (prioritize stability)

---

## Decision Logic

```
Run all 4 experiments (EXP-001 through EXP-004)

Compare metrics:
├─ Total pipeline time (primary)
├─ GPU temperature (safety constraint)
├─ Error rate (quality constraint)
└─ Memory usage (system constraint)

IF batch=4 is fastest AND stable:
    └─> Reduce optimal_batch_size to 4

IF batch=8 is fastest AND stable:
    └─> Keep optimal_batch_size at 8 (no change)

IF batch=12 is fastest AND stable (temp <85°C, errors <5%):
    └─> Increase optimal_batch_size to 12

IF batch=16 is fastest AND stable (temp <85°C, errors <5%):
    └─> Increase optimal_batch_size to 16
    └─> Add thermal monitoring safeguards

IF multiple batch sizes perform similarly (<5% difference):
    └─> Choose smaller batch size for stability
```

---

## Success Criteria

### Minimum
- Complete all 4 experiments without critical errors
- Valid metrics for all batch sizes
- Clear recommendation on optimal batch size

### Target
- 10%+ throughput improvement from optimal batch size
- GPU temperature stays <85°C for all batch sizes
- Error rate <3% for winning batch size

### Stretch
- 25%+ throughput improvement
- Findings applicable to Phase 2.5 optimization
- Dynamic batch sizing algorithm based on system metrics

---

## Experimental Procedure

### Stage 1: Setup (1 hour)
1. Create isolated test database: `experiment_phase2.db`
2. Select and clone 16 test papers (stratified sample)
3. Implement experiment runner infrastructure
4. Set up metrics collection and GPU monitoring

### Stage 2: Run Experiments (~2 hours)
1. **EXP-001**: Batch size 4 (~26 min)
2. **EXP-002**: Batch size 8 (~26 min)
3. **EXP-003**: Batch size 12 (~26 min)
4. **EXP-004**: Batch size 16 (~26 min)

**Allow 10-minute cooling period between experiments**

### Stage 3: Analysis (1 hour)
1. Compare total pipeline times
2. Assess GPU thermal performance
3. Validate error rates and quality metrics
4. Determine optimal batch size

**Total Time**: ~4 hours

---

## Implementation Files

```
back_end/src/experimentation/
├── config/
│   ├── __init__.py
│   └── experiment_config.py           # Batch size configurations
├── runners/
│   ├── __init__.py
│   ├── experiment_runner.py           # Main orchestrator
│   └── batch_size_optimizer.py        # Batch size tester
├── evaluation/
│   ├── __init__.py
│   ├── metrics_calculator.py          # Speed/quality/system metrics
│   └── system_monitor.py              # GPU/thermal monitoring
├── analysis/
│   ├── __init__.py
│   ├── results_analyzer.py            # Compare experiments
│   └── visualization.py               # Performance plots
└── data/
    ├── test_dataset.json              # 16 test papers
    ├── results/
    │   ├── EXP-001_batch4.json       # Batch 4 results
    │   ├── EXP-002_batch8.json       # Batch 8 results
    │   ├── EXP-003_batch12.json      # Batch 12 results
    │   └── EXP-004_batch16.json      # Batch 16 results
    └── experiment_phase2.db           # Isolated test database
```

---

## Timeline

**Day 1 (Morning - 2 hours)**:
- Build experiment infrastructure
- Create test dataset (16 papers)

**Day 1 (Afternoon - 2 hours)**:
- Run all 4 experiments
- Collect metrics

**Day 2 (Morning - 1 hour)**:
- Analyze results
- Generate recommendation

**Total**: ~5 hours

---

## Deliverables

1. **Experiment Results**: 4 JSON files with comprehensive metrics
2. **Performance Analysis**: Statistical comparison of batch sizes
3. **Visualization**: Plots showing:
   - Total time vs batch size
   - GPU temperature vs batch size
   - Error rate vs batch size
   - Throughput vs batch size
4. **Recommendation Report**: Optimal batch size with justification
5. **Updated Config**: Modified `config.py` with optimal `intervention_batch_size`

---

## The Core Question

**What is the optimal number of papers to process per batch?**

### Trade-offs to Consider:

**Batch Size 4** (Conservative):
- ✅ Lowest memory usage
- ✅ Most stable (low thermal risk)
- ❌ Most overhead (4 batches = 4× batch setup cost)
- ❌ Slowest throughput

**Batch Size 8** (Current Default):
- ✅ Good balance (proven in production)
- ✅ Moderate memory usage
- ✅ Moderate thermal risk
- ⚠️ May not be optimal

**Batch Size 12** (Aggressive):
- ✅ 33% fewer batches than batch=8
- ✅ Higher throughput potential
- ⚠️ Higher memory usage
- ⚠️ Moderate thermal risk

**Batch Size 16** (Maximum):
- ✅ Maximum throughput (1 batch only)
- ✅ Minimal overhead
- ❌ Highest memory usage
- ❌ Highest thermal risk
- ❌ May cause stability issues

---

## Next Steps

1. ✅ Review experimental design
2. Approve infrastructure build-out
3. Select 16 test papers (stratified sample)
4. Execute all 4 experiments
5. Analyze results and update production config

---

**Key Design Principles**:
- ✅ Controlled experiment (same 16 papers for all tests)
- ✅ Minimal experiments (4 batch sizes)
- ✅ Fast execution (~4 hours total)
- ✅ Clear decision logic
- ✅ Actionable recommendation (update `config.py`)