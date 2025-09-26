# LLM Paper Processing System

This system processes research papers with dual LLMs (gemma2:9b and qwen2.5:14b) to extract health intervention correlations from abstracts and full-text papers.

## How it Works

1. **Dual Extraction**: Each paper is processed by both configured LLMs independently
2. **Model Comparison**: Both models extract interventions from the same papers
3. **All Results Stored**: Valid interventions from both models are stored with model attribution
4. **Status Tracking**: Papers are tracked through processing states (pending → processed/failed)

## Quick Start - Unified Processing Script

The easiest way to process papers is using the unified script:

```bash
# Check current processing status
python run_llm_processing.py --status-only

# Process all unprocessed papers in small batches
python run_llm_processing.py --batch-size 5

# Process only 20 papers for testing
python run_llm_processing.py --limit 20

# Process papers with larger batches (if you have good GPU)
python run_llm_processing.py --batch-size 10
```

## Programmatic Usage

```python
from src.llm.pipeline import InterventionResearchPipeline

# Initialize the pipeline
pipeline = InterventionResearchPipeline()

# Process unprocessed papers with limit
results = pipeline.analyze_interventions(limit_papers=10)

print(f"Papers processed: {results['papers_processed']}")
print(f"Interventions extracted: {results['interventions_extracted']}")
print(f"Success rate: {results['success_rate']:.1f}%")
```

## Configuration

The system uses two models configured in `DualModelAnalyzer`:
- **gemma2:9b** - Fast and efficient model for initial extraction
- **qwen2.5:14b** - Larger, more capable model for comprehensive analysis

Models are configured in `src/llm/dual_model_analyzer.py` with automatic GPU optimization and dynamic token limits.

## Database Schema

### Paper Processing States

Papers in the `papers` table have a `processing_status` field:
- **NULL/pending** - Not yet processed by LLMs
- **processed** - Successfully processed by both models
- **failed** - Processing failed (with error details)

### Intervention Storage

The `interventions` table stores all extracted interventions with:
- **extraction_model** - Which LLM extracted this intervention
- **paper_id** - Source paper reference
- **intervention_category** - Type of intervention (probiotic, dietary, etc.)
- **correlation_strength** - Confidence in the intervention effect (0-1)
- **supporting_quote** - Text evidence from the paper

## Adding New Models

To add additional models, modify the `models` dictionary in `DualModelAnalyzer.__init__()`:

```python
self.models = {
    'gemma2:9b': {
        'client': get_llm_client('gemma2:9b'),
        'temperature': 0.3,
        # ... configuration
    },
    'qwen2.5:14b': {
        'client': get_llm_client('qwen2.5:14b'),
        'temperature': 0.3,
        # ... configuration
    },
    'llama3.1:8b': {  # New model
        'client': get_llm_client('llama3.1:8b'),
        'temperature': 0.3,
        # ... configuration
    }
}
```

The system will automatically process papers with all configured models.

## Reviewing Results

Use the interactive database chat to review extracted interventions:

```bash
python database_chat.py
```

Or query the database directly:
```sql
-- See processing progress
SELECT processing_status, COUNT(*) FROM papers GROUP BY processing_status;

-- Review interventions by model
SELECT extraction_model, COUNT(*) FROM interventions GROUP BY extraction_model;

-- Find papers with interventions from both models
SELECT paper_id, COUNT(DISTINCT extraction_model) as model_count
FROM interventions
GROUP BY paper_id
HAVING model_count = 2;
```

## Performance Considerations

- **Batch Size**: Recommended 5-10 papers per batch depending on GPU memory
- **Processing Time**: ~2x longer due to dual model processing
- **Token Usage**: Tracked separately per model for cost monitoring
- **GPU Optimization**: Automatic GPU detection and memory management
- **Resumption**: Processing can be interrupted and resumed safely

## Processing Status Monitoring

The unified script provides detailed status reporting:

```bash
# Check what needs processing
python run_llm_processing.py --status-only

# Monitor progress during processing
tail -f back_end/data/logs/unified_llm_processing.log
```

## Troubleshooting

**Common Issues:**

1. **Out of GPU Memory**: Reduce `--batch-size` to 3 or lower
2. **Papers stuck in 'pending'**: Use `--force-reprocess` cautiously to retry
3. **No papers found**: Check if abstracts are available in the papers table
4. **Model errors**: Check logs in `back_end/data/logs/` directory

**Reset Processing Status:**
```sql
-- Reset all papers to pending (use with caution)
UPDATE papers SET processing_status = 'pending' WHERE processing_status = 'failed';
```

## Key Benefits

1. **Unified Interface**: Single script handles all processing scenarios
2. **Resumption**: Safely interrupt and resume processing
3. **Dual Model Analysis**: Cross-validation between gemma2:9b and qwen2.5:14b
4. **Progress Tracking**: Clear status reporting and error handling
5. **Flexible Batching**: Configurable batch sizes for different hardware