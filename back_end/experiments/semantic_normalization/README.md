# Semantic Normalization Experiment

**Phase 3 Intervention Name Normalization using Semantic Embeddings**

## Overview

This experiment replaces the existing LLM-based intervention name normalization system with a semantic embedding approach. The goal is to improve accuracy, speed, and consistency in identifying intervention variants (synonyms, abbreviations, dosage variations).

## Directory Structure

```
semantic_normalization/
├── core/
│   ├── data_exporter.py           # Export interventions from DB
│   ├── pair_generator.py          # Generate candidate pairs with fuzzy matching
│   ├── labeling_interface.py      # Interactive terminal labeling UI
│   ├── normalizer.py              # [TODO] Semantic embedding normalizer
│   ├── test_runner.py             # [TODO] Test execution engine
│   └── evaluator.py               # [TODO] Performance metrics
├── config/
│   └── config.yaml                # Configuration settings
├── data/
│   ├── samples/                   # Exported intervention names
│   └── ground_truth/              # Manually labeled pairs
├── results/                       # Timestamped test results
├── logs/                          # Execution logs
├── run_phase1.py                  # Phase 1 orchestrator
└── README.md                      # This file
```

## Phase 1: Setup & Data Preparation

### Prerequisites

Install required dependencies:

```bash
pip install rapidfuzz pyyaml
```

### Step 1: Run Complete Workflow

Execute the full Phase 1 workflow (export → generate → label):

```bash
cd back_end/experiments/semantic_normalization
python run_phase1.py
```

This will:
1. Export 300-500 intervention names from the database
2. Generate ~150 candidate pairs using fuzzy matching
3. Launch interactive labeling interface

### Step 2: Label Ground Truth Pairs

The labeling interface will guide you through labeling 50 intervention pairs:

- **y** = YES (these interventions should match)
- **n** = NO (these are different interventions)
- **s** = SKIP (skip this pair)
- **q** = QUIT (save progress and exit)

**Resumable**: Progress is auto-saved every 5 labels. You can quit and resume anytime.

### Alternative: Run Steps Individually

```bash
# Step 1: Export data
python -m core.data_exporter

# Step 2: Generate candidate pairs
python -m core.pair_generator

# Step 3: Label pairs
python -m core.labeling_interface
```

## Configuration

Edit `config/config.yaml` to customize:

- **export_limit**: Number of interventions to export (default: 500)
- **target_pairs**: Number of pairs to label (default: 50)
- **similarity_threshold_min/max**: Fuzzy matching range (default: 0.40-0.95)
- **fuzzy_matching.algorithm**: Similarity algorithm (default: jaro_winkler)

## Output Files

### Data Exports
`data/samples/interventions_export_YYYYMMDD_HHMMSS.json`

Contains exported intervention data with metadata:
```json
{
  "metadata": {
    "export_timestamp": "20251005_143022",
    "total_records": 450,
    "unique_intervention_names": 350
  },
  "interventions": [...],
  "unique_names": [...]
}
```

### Candidate Pairs
`data/ground_truth/candidate_pairs_YYYYMMDD_HHMMSS.json`

Contains generated candidate pairs with similarity scores:
```json
{
  "metadata": {...},
  "categorized": {
    "likely_match": [...],
    "edge_case": [...],
    "likely_no_match": [...]
  },
  "all_candidates": [...]
}
```

### Labeled Pairs (Ground Truth)
`data/ground_truth/labeling_session_ground_truth_YYYYMMDD_HHMMSS.json`

Contains manually labeled pairs (source of truth for evaluation):
```json
{
  "session_id": "ground_truth_20251005_143500",
  "labeled_pairs": [
    {
      "pair_id": 1,
      "intervention_1": "metformin",
      "intervention_2": "metformin therapy",
      "similarity_score": 0.78,
      "label": "match",
      "labeled_at": "2025-10-05T14:35:22"
    }
  ],
  "progress": {"total": 50, "labeled": 50}
}
```

## Edge Cases to Label

When labeling, pay special attention to:

1. **Synonyms**: "vitamin D" vs "cholecalciferol" → MATCH
2. **Abbreviations**: "CBT" vs "cognitive behavioral therapy" → MATCH
3. **Dosage variants**: "metformin" vs "metformin 500mg" → MATCH (or NO MATCH? You decide!)
4. **Similar names**: "vitamin D" vs "vitamin C" → NO MATCH
5. **Treatment subtypes**: "aerobic exercise" vs "resistance training" → NO MATCH

## Next Steps

After completing Phase 1:

1. **Review labeled pairs**: Ensure quality and diversity of ground truth
2. **Phase 2**: Develop semantic embedding normalizer
3. **Phase 3**: Evaluate performance against ground truth
4. **Phase 4**: Integrate into production pipeline

## Notes

- **Isolated workspace**: This experiment is separate from production code
- **Database**: Uses existing `intervention_research.db` (read-only)
- **Ignores**: `canonical_entities` and `entity_mappings` tables (will be replaced)
- **Resumable**: All operations support interruption and recovery
