# Data Mining Module

This module contains specialized tools for research-oriented data mining and analysis of intervention data.

## Intervention Consensus Analyzer

### Purpose
The `InterventionConsensusAnalyzer` provides advanced research consensus functionality that goes beyond basic duplicate removal. It's designed for researchers who want to:

- Accumulate evidence across multiple papers studying the same intervention
- Create high-confidence research records through cross-validation
- Enhance statistical power through multi-source evidence
- Track research provenance and methodological diversity

### Key Features

**Cross-Paper Evidence Accumulation**: Groups interventions from different papers that study the same intervention-condition relationships, providing stronger evidence.

**Enhanced Confidence Calculation**: Boosts confidence scores when multiple independent sources agree, with sophisticated algorithms accounting for:
- Multi-source validation (up to 10% boost)
- Cross-model validation (5% boost)
- Cross-paper validation (5% boost)
- Evidence consistency (up to 5% boost)

**Research Metadata**: Tracks comprehensive research information including:
- Source papers and models
- Evidence consistency scores
- Methodological diversity
- Research quality indicators

### Usage

```python
from back_end.src.data_mining.intervention_consensus_analyzer import create_consensus_analyzer
from back_end.src.llm_processing.batch_entity_processor import create_batch_processor

# Create components
batch_processor = create_batch_processor('path/to/database.db')
consensus_analyzer = create_consensus_analyzer(batch_processor)

# Create research consensus
results = consensus_analyzer.create_research_consensus(
    interventions,
    confidence_threshold=0.5,
    min_sources=1
)

# Access results
consensus_interventions = results['consensus_interventions']
research_metadata = results['analysis_metadata']

# Generate human-readable report
report = consensus_analyzer.generate_research_summary_report(results)
print(report)
```

### Example Output

**Input**: 3 interventions studying probiotics for IBS across 2 papers
**Output**: 1 high-confidence research consensus with:
- Enhanced confidence: 0.96 (vs individual scores of 0.7, 0.8)
- Evidence sources: 2 papers
- Cross-validation: True
- Accumulated supporting quotes from both studies

## Separation from Core Processing

The consensus analyzer is intentionally separated from core entity processing (`batch_entity_processor.py`) because:

1. **Different Use Cases**: Core processing focuses on essential duplicate removal for data integrity, while this module focuses on research quality enhancement
2. **Optional Complexity**: Researchers can choose whether to use advanced consensus features
3. **Performance**: Core processing remains fast for basic deduplication workflows
4. **Maintainability**: Clear separation of concerns between entity processing and research analysis

## When to Use

**Use the Consensus Analyzer when:**
- Conducting research meta-analysis
- Building high-confidence intervention databases
- Studying intervention effectiveness across multiple papers
- Need detailed research provenance tracking

**Use Basic Deduplication when:**
- Just need to remove true duplicates from same papers
- Want to preserve individual paper records
- Prioritize processing speed over research enhancement
- Building simple intervention catalogs