# Phase 2 Optimization Experiment Design

## Executive Summary

Comprehensive experimental framework to optimize Phase 2 LLM extraction pipeline by systematically testing **batch size** and **prompt strategies**. Current baseline: ~93s per paper (~38-39 papers/hour) using qwen3:14b with mechanism extraction.

---

## 1. Baseline Configuration

### Current Phase 2 Setup
- **Model**: qwen3:14b (single-model architecture)
- **Batch Size**: 8 papers per batch
- **Prompt**: Extraction-only (no categorization in prompt)
- **Processing Flow**: Extract → Phase 2.5 Categorization (separate step)
- **Performance**: ~93s per paper
- **Mechanism Extraction**: Enabled (biological/behavioral/psychological pathways)

### Current Prompt Structure
- **Location**: `back_end/src/llm_processing/prompt_service.py`
- **Method**: `create_extraction_prompt()`
- **Fields Extracted**: 23 fields including intervention_name, mechanism, dosage, etc.
- **Categorization**: NOT included in Phase 2 extraction prompt

---

## 2. Experimental Variables

### Variable 1: Batch Size
**Hypothesis**: Larger batches may improve throughput but risk memory issues and thermal limits.

**Test Values**:
- `batch_size = 1` (baseline for per-paper timing)
- `batch_size = 4` (conservative)
- `batch_size = 8` (current default)
- `batch_size = 12` (aggressive)
- `batch_size = 16` (stress test)

**Rationale**:
- Single-model architecture can handle larger batches than previous dual-model
- GPU memory: ~8GB VRAM available
- Need to balance throughput vs thermal safety
- Current optimal_batch_size logic uses GPU memory monitoring

### Variable 2: Prompt Strategy
**Hypothesis**: Merging Phase 2 + Phase 2.5 prompts may reduce total pipeline time despite longer per-paper processing.

**Test Configurations**:

#### Strategy A: Current (Separate Extraction + Categorization)
- **Phase 2**: Extract 23 fields WITHOUT categories
- **Phase 2.5**: LLM categorizes 20 interventions + conditions per batch
- **Total Time**: Extraction time + Categorization time

#### Strategy B: Merged (Extraction + Categorization in One Call)
- **Phase 2**: Extract 25 fields INCLUDING intervention_category + condition_category
- **Phase 2.5**: SKIP (no separate categorization needed)
- **Total Time**: Extraction time only
- **Trade-off**: Longer extraction per paper, but eliminates Phase 2.5

#### Strategy C: Hybrid (Intervention Category Only)
- **Phase 2**: Extract 24 fields + intervention_category only
- **Phase 2.5**: Only categorize conditions (lighter workload)
- **Total Time**: Extraction time + Reduced categorization time

**Categorization Context**:
- Intervention categories: 13 categories (exercise, diet, supplement, medication, therapy, lifestyle, surgery, test, device, procedure, biologics, gene_therapy, emerging)
- Condition categories: 18 categories (cardiac, neurological, digestive, etc.)

---

## 3. Experimental Design

### Test Matrix (15 Experiments)

| Experiment ID | Batch Size | Prompt Strategy | Expected Outcome |
|--------------|-----------|-----------------|------------------|
| EXP-001 | 1 | A (Current) | Baseline per-paper timing |
| EXP-002 | 4 | A (Current) | Conservative throughput |
| EXP-003 | 8 | A (Current) | Current production baseline |
| EXP-004 | 12 | A (Current) | Higher throughput test |
| EXP-005 | 16 | A (Current) | Maximum throughput test |
| EXP-006 | 1 | B (Merged) | Baseline merged timing |
| EXP-007 | 4 | B (Merged) | Conservative merged |
| EXP-008 | 8 | B (Merged) | Standard merged |
| EXP-009 | 12 | B (Merged) | High merged |
| EXP-010 | 16 | B (Merged) | Max merged |
| EXP-011 | 4 | C (Hybrid) | Conservative hybrid |
| EXP-012 | 8 | C (Hybrid) | Standard hybrid |
| EXP-013 | 12 | C (Hybrid) | High hybrid |
| EXP-014 | 1 | B (Merged) + Aggressive Suppress | Ultra-fast single |
| EXP-015 | 8 | B (Merged) + Aggressive Suppress | Ultra-fast batch |

**Note on EXP-014/015**: Test aggressive chain-of-thought suppression techniques beyond current `<think>` tag stripping.

---

## 4. Evaluation Metrics

### Speed Metrics (Primary)
1. **Papers per Hour**: Total papers / total time (including all phases)
2. **Seconds per Paper**: Average processing time per paper
3. **Batch Processing Time**: Time to process one batch
4. **Total Pipeline Time**: Phase 2 + Phase 2.5 combined
5. **Throughput Efficiency**: Papers per hour / GPU memory utilization

### Accuracy Metrics (Critical)
1. **Field Extraction Completeness**: % of required fields populated
2. **Category Accuracy**: % of interventions/conditions correctly categorized
3. **Mechanism Quality**: % of mechanisms with meaningful content (not "unknown")
4. **JSON Parse Success Rate**: % of responses successfully parsed
5. **Validation Pass Rate**: % of interventions passing validator
6. **Ground Truth Comparison**: Compare against 80 labeled pairs in ground truth dataset

### Quality Metrics (Secondary)
1. **Intervention Specificity**: % of intervention_name fields with specific names (not generic "therapy")
2. **Condition Precision**: % of health_condition fields with primary condition (not underlying disease)
3. **Dosage Extraction Rate**: % of pharmaceutical interventions with dosage data
4. **Sample Size Capture**: % of extractions including sample_size
5. **Mechanism Detail**: Average token count in mechanism field (target: 5-15 tokens)

### System Metrics (Safety)
1. **GPU Temperature**: Max temperature during experiment
2. **GPU Memory Peak**: Peak memory usage
3. **Error Rate**: % of papers failing extraction
4. **Thermal Throttling Events**: Count of cooling pauses
5. **LLM Timeout Rate**: % of requests exceeding timeout

---

## 5. Test Dataset

### Dataset Composition
- **Size**: 50 papers (representative sample)
- **Selection Criteria**:
  - 10 papers: High citation (>100 citations) - complex studies
  - 10 papers: Medium citation (20-100) - typical studies
  - 10 papers: Low citation (<20) - simple studies
  - 10 papers: Full-text available - comprehensive extraction
  - 10 papers: Abstract-only - minimal extraction
- **Conditions Coverage**: Diverse across all 18 condition categories
- **Intervention Mix**: All 13 intervention categories represented

### Ground Truth Reference
- Use existing 80 labeled intervention pairs from `back_end/src/semantic_normalization/ground_truth/`
- Manually validate 10 papers from test dataset for accuracy benchmarking

---

## 6. Experimental Procedure

### Phase 1: Environment Setup (1 hour)
1. Create isolated test database: `experiment_phase2.db`
2. Clone 50 test papers into test database
3. Implement experiment runner with configuration management
4. Build evaluation framework with automated metrics calculation
5. Set up logging and result storage

### Phase 2: Baseline Measurement (2 hours)
1. Run EXP-003 (current production config) 3 times
2. Calculate mean ± std dev for all metrics
3. Establish baseline thresholds for speed/accuracy trade-offs

### Phase 3: Batch Size Experiments (4 hours)
1. Run EXP-001, 002, 003, 004, 005 (Strategy A with varying batch sizes)
2. Monitor GPU thermal status continuously
3. Identify optimal batch size for current prompt strategy

### Phase 4: Prompt Strategy Experiments (6 hours)
1. Implement Strategy B (merged prompt) and Strategy C (hybrid prompt)
2. Run EXP-006 through EXP-013
3. Compare total pipeline time (Phase 2 + Phase 2.5)
4. Validate categorization accuracy for merged strategies

### Phase 5: Aggressive Optimization (2 hours)
1. Implement additional chain-of-thought suppression techniques
2. Run EXP-014 and EXP-015
3. Test edge cases and failure modes

### Phase 6: Analysis and Recommendation (2 hours)
1. Statistical analysis of all experiments
2. Generate performance vs accuracy plots
3. Recommend optimal configuration with confidence intervals
4. Document trade-offs and implementation requirements

**Total Estimated Time**: 17 hours of experiments + analysis

---

## 7. Implementation Plan

### File Structure
```
back_end/src/experimentation/
├── PHASE2_OPTIMIZATION_DESIGN.md          # This document
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── experiment_config.py               # Experiment configurations
│   └── prompt_strategies.py               # Prompt variants (A, B, C)
├── runners/
│   ├── __init__.py
│   ├── experiment_runner.py               # Main experiment orchestrator
│   ├── batch_size_tester.py               # Batch size experiments
│   └── prompt_strategy_tester.py          # Prompt strategy experiments
├── evaluation/
│   ├── __init__.py
│   ├── metrics_calculator.py              # Speed/accuracy/quality metrics
│   ├── ground_truth_validator.py          # Accuracy validation
│   └── system_monitor.py                  # GPU/thermal monitoring
├── analysis/
│   ├── __init__.py
│   ├── statistical_analysis.py            # Statistical tests and confidence intervals
│   ├── visualization.py                   # Performance plots
│   └── recommendation_engine.py           # Optimal config recommendation
└── data/
    ├── test_dataset.json                  # 50 test papers
    ├── results/
    │   ├── EXP-001_results.json          # Individual experiment results
    │   ├── ...
    │   └── summary_report.json            # Aggregate analysis
    └── experiment_phase2.db               # Isolated test database
```

### Key Components

#### 1. `experiment_config.py`
```python
@dataclass
class ExperimentConfig:
    experiment_id: str
    batch_size: int
    prompt_strategy: str  # "A", "B", or "C"
    model_name: str = "qwen3:14b"
    test_dataset_size: int = 50
    repetitions: int = 1
    enable_thermal_monitoring: bool = True
```

#### 2. `prompt_strategies.py`
```python
class PromptStrategy:
    @staticmethod
    def create_strategy_a_prompt(paper): ...  # Current

    @staticmethod
    def create_strategy_b_prompt(paper): ...  # Merged

    @staticmethod
    def create_strategy_c_prompt(paper): ...  # Hybrid
```

#### 3. `experiment_runner.py`
```python
class ExperimentRunner:
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult
    def run_all_experiments(self) -> List[ExperimentResult]
    def save_results(self, results: List[ExperimentResult])
```

#### 4. `metrics_calculator.py`
```python
class MetricsCalculator:
    def calculate_speed_metrics(self, results) -> Dict
    def calculate_accuracy_metrics(self, results) -> Dict
    def calculate_quality_metrics(self, results) -> Dict
    def calculate_system_metrics(self, results) -> Dict
```

---

## 8. Expected Outcomes

### Hypothesis 1: Optimal Batch Size
- **Prediction**: Batch size 8-12 will be optimal
- **Reasoning**: Balance between throughput and thermal safety
- **Risk**: Larger batches may cause GPU memory issues or thermal throttling

### Hypothesis 2: Merged Prompt Strategy
- **Prediction**: Strategy B (merged) will reduce total pipeline time by 20-30%
- **Reasoning**: Eliminates Phase 2.5 categorization step
- **Risk**: Longer extraction time per paper, potential accuracy loss

### Hypothesis 3: Speed-Accuracy Trade-off
- **Prediction**: Larger batch sizes will improve speed without accuracy loss
- **Reasoning**: Batch size affects throughput, not extraction quality
- **Risk**: May increase error rate due to memory pressure

### Hypothesis 4: Category Extraction Quality
- **Prediction**: Strategy A (current) will have highest category accuracy
- **Reasoning**: Focused categorization prompts in Phase 2.5 are more accurate
- **Risk**: Merged prompts may confuse model with too many tasks

---

## 9. Success Criteria

### Minimum Viable Optimization
- 10% improvement in papers per hour
- No degradation in accuracy metrics (>95% of baseline)
- GPU temperature stays <85°C
- Error rate <5%

### Target Optimization
- 25% improvement in papers per hour
- Accuracy maintained at 98%+ of baseline
- Reduced GPU memory usage
- Error rate <3%

### Stretch Goals
- 50% improvement in papers per hour (from ~39 to ~58 papers/hour)
- Improved accuracy through better prompts
- Mechanism extraction quality improvement
- Reduced Phase 2.5 processing time

---

## 10. Risk Mitigation

### Risk 1: GPU Overheating
- **Mitigation**: Thermal monitoring with automatic cooling pauses
- **Fallback**: Reduce batch size or increase inter-batch delays

### Risk 2: Accuracy Degradation
- **Mitigation**: Ground truth validation after each experiment
- **Fallback**: Revert to current configuration if accuracy drops >5%

### Risk 3: Database Corruption
- **Mitigation**: Use isolated test database
- **Fallback**: Database backups before each experiment

### Risk 4: LLM Instability
- **Mitigation**: Retry logic with exponential backoff
- **Fallback**: Skip failed papers and continue

---

## 11. Timeline and Resources

### Execution Timeline
- **Week 1**: Implementation (file structure, runners, evaluation)
- **Week 2**: Experiments (baseline, batch size, prompt strategies)
- **Week 3**: Analysis and documentation
- **Total**: 3 weeks part-time or 1 week full-time

### Required Resources
- GPU: NVIDIA GPU with 8GB+ VRAM
- Storage: ~5GB for test database and results
- Ollama: qwen3:14b model available locally
- Ground Truth: 80 labeled pairs from existing dataset

---

## 12. Deliverables

### Technical Deliverables
1. **Experiment Runner**: Automated experimentation framework
2. **Results Database**: All 15 experiments with metrics
3. **Analysis Report**: Statistical analysis with recommendations
4. **Optimal Configuration**: Production-ready config with justification

### Documentation Deliverables
1. **Experiment Log**: Detailed logs for all experiments
2. **Performance Plots**: Speed vs accuracy visualizations
3. **Recommendation Report**: Executive summary with implementation guide
4. **CLAUDE.md Update**: Updated with optimal Phase 2 configuration

---

## 13. Next Steps

### Immediate Actions
1. Review and approve this experimental design
2. Prioritize experiments (which to run first?)
3. Confirm test dataset size (50 papers sufficient?)
4. Validate success criteria and thresholds

### Implementation Sequence
1. Build experiment infrastructure (runners, metrics)
2. Create test dataset and isolated database
3. Run baseline experiments (EXP-003)
4. Execute batch size experiments
5. Execute prompt strategy experiments
6. Analyze and recommend

---

## Appendix A: Prompt Strategy Examples

### Strategy A: Current (Extraction Only)
```
REQUIRED FIELDS:
- intervention_name
- dosage
- duration
- frequency
- health_condition
- mechanism
- correlation_type
- correlation_strength
... (23 fields total, NO categories)
```

**Phase 2.5 Categorization Prompt** (separate call, 20 items per batch):
```
Classify each intervention into ONE category:
CATEGORIES: exercise, diet, supplement, medication, therapy, lifestyle, surgery, test, device, procedure, biologics, gene_therapy, emerging

INTERVENTIONS:
1. aspirin
2. walking
...
```

### Strategy B: Merged (Extraction + Categorization)
```
REQUIRED FIELDS:
- intervention_name
- intervention_category: ONE of [exercise, diet, supplement, medication, therapy, lifestyle, surgery, test, device, procedure, biologics, gene_therapy, emerging]
- dosage
- duration
- frequency
- health_condition
- condition_category: ONE of [cardiac, neurological, digestive, respiratory, endocrine, renal, musculoskeletal, dermatological, psychiatric, reproductive, immune, hematological, oncological, ophthalmological, ent, urological, infectious, genetic]
- mechanism
- correlation_type
- correlation_strength
... (25 fields total, WITH categories)
```

### Strategy C: Hybrid (Intervention Category Only)
```
REQUIRED FIELDS:
- intervention_name
- intervention_category: ONE of [exercise, diet, supplement, medication, therapy, lifestyle, surgery, test, device, procedure, biologics, gene_therapy, emerging]
- dosage
- duration
- frequency
- health_condition
- mechanism
- correlation_type
- correlation_strength
... (24 fields total, intervention_category only)
```

**Phase 2.5 Condition Categorization** (separate call, conditions only):
```
Classify each condition into ONE category:
CATEGORIES: cardiac, neurological, digestive, respiratory, endocrine, renal, musculoskeletal, dermatological, psychiatric, reproductive, immune, hematological, oncological, ophthalmological, ent, urological, infectious, genetic

CONDITIONS:
1. type 2 diabetes
2. hypertension
...
```

---

## Appendix B: Statistical Analysis Plan

### Comparative Analysis
- **ANOVA**: Compare mean papers-per-hour across batch sizes
- **T-tests**: Pairwise comparison of prompt strategies
- **Regression**: Model relationship between batch size and throughput
- **Correlation**: Speed vs accuracy trade-off analysis

### Confidence Intervals
- 95% CI for all speed metrics
- Bootstrapping for accuracy metrics
- Effect size calculation (Cohen's d)

### Visualization
- Box plots: Batch size vs papers-per-hour
- Scatter plots: Speed vs accuracy by strategy
- Heatmaps: Experiment matrix with metric overlays
- Time series: GPU temperature and memory over time

---

*Document Version: 1.0*
*Last Updated: 2025-10-08*
*Status: Design Phase - Awaiting Approval*