# Multi-LLM Consensus Analysis

This system processes research papers with multiple LLMs (gemma2:9b and qwen2.5:14b) to extract probiotic-health correlations with consensus validation.

## How it Works

1. **Multiple Extraction**: Each paper is processed by both configured LLMs
2. **Consensus Analysis**: Correlations are compared between models
3. **Agreement Storage**: Only correlations that both models agree on are stored in the main database
4. **Conflict Flagging**: Disagreements are flagged for manual review

## Quick Start

```python
from src.llm.pipeline import EnhancedResearchPipeline

# Use multi-LLM consensus (default)
pipeline = EnhancedResearchPipeline(use_consensus=True)

# Process papers with consensus analysis
results = pipeline.analyze_correlations(limit_papers=10)

print(f"Agreed correlations: {results['agreed_correlations']}")
print(f"Conflicts: {results['conflicts']}")
print(f"Papers needing review: {results['papers_needing_review']}")
```

## Configuration

The default configuration uses:
- **gemma2:9b** - Fast and efficient model
- **qwen2.5:14b** - Larger, more capable model

To change models, modify `config.py`:

```python
# In MultiLLMConfig.__post_init__():
self.models = [
    LLMConfig(model_name="your_model_1"),
    LLMConfig(model_name="your_model_2"),
    # Add more models easily
]
```

## Database Schema

### New Tables

1. **correlation_extractions** - Individual LLM extractions
2. **correlation_consensus** - Consensus tracking and results

### Existing Tables

- **correlations** - Only stores agreed-upon correlations
- **papers** - Unchanged

## Adding New Models

To add a third LLM (e.g., for tie-breaking):

1. Add to config:
```python
self.models = [
    LLMConfig(model_name="gemma2:9b"),
    LLMConfig(model_name="qwen2.5:14b"), 
    LLMConfig(model_name="llama3.1:8b")  # New model
]
```

2. The system automatically handles N models
3. Consensus logic adapts to any number of models

## Conflict Resolution

When models disagree:

- **Status**: `needs_review = True`
- **Reason**: Stored in `review_reason` field
- **Data**: All individual extractions preserved
- **Action**: Manual review or third-model arbitration

## Review Interface

Papers needing review can be found with:

```sql
SELECT * FROM correlation_consensus 
WHERE needs_review = TRUE;
```

## Performance Considerations

- **Batch Size**: Automatically reduced for multi-LLM (5 vs 15 papers)
- **Processing Time**: ~2-3x longer due to multiple model calls
- **Token Usage**: Tracked separately per model
- **API Delays**: Built-in delays between model calls

## Backward Compatibility

The system maintains full backward compatibility:

```python
# Old single-LLM approach still works
pipeline = EnhancedResearchPipeline(use_consensus=False)
```

## Testing

Run the test script to verify setup:

```bash
python test_consensus_pipeline.py
```

## Key Benefits

1. **Higher Accuracy**: Cross-validation between models
2. **Conflict Detection**: Identifies uncertain extractions
3. **Easy Extension**: Simple to add more models
4. **Quality Assurance**: Only stores high-confidence results
5. **Transparency**: All individual extractions preserved