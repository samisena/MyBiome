# Ollama LLM Pipeline Test

This is a refactored version of the LLM pipeline test that uses **Ollama** (local open-source LLMs) instead of OpenRouter's free tier APIs.

## Features

- **Local LLM Processing**: Uses locally running Mistral 7B and Llama 3.1 8B models via Ollama
- **Model Switching**: Easy switching between different model combinations
- **Error Handling**: Robust error handling with retry logic
- **Connection Validation**: Automatic validation of Ollama setup and model availability
- **Comprehensive Logging**: Detailed logging of the entire pipeline process

## Prerequisites

### 1. Install Ollama
```bash
# Visit https://ollama.ai and install Ollama for your platform
# Or use package managers:
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama Server
```bash
ollama serve
```

### 3. Pull Required Models
```bash
ollama pull mistral:7b
ollama pull llama3.1:8b
```

### 4. Verify Models Are Available
```bash
ollama list
```

You should see both models listed.

## Usage

### Basic Usage (Default Configuration)
```python
# Run with default models (Mistral 7B vs Llama 3.1 8B)
python test_llm_pipeline.py
```

### Custom Configuration
```python
from test_llm_pipeline import main, OllamaModel

# Custom condition and paper count
success = main(
    condition="Crohn's disease", 
    max_papers=3,
    primary_model="mistral:7b",
    secondary_model="llama3.1:8b",
    ollama_url="http://localhost:11434/v1"
)
```

### Advanced Usage with Model Switching
```python
from test_llm_pipeline import LLMPipelineTester, OllamaModel

# Initialize tester
tester = LLMPipelineTester()

# Switch models mid-test
tester.switch_models(
    new_primary=OllamaModel.LLAMA_3_1_8B,
    new_secondary=OllamaModel.MISTRAL_7B
)

# Validate new setup
if tester.validate_setup():
    papers = tester.collect_test_data("IBS", 5)
    if papers:
        primary_results = tester.process_with_primary_model(papers)
        secondary_results = tester.process_with_secondary_model(papers)
        tester.compare_results(primary_results, secondary_results)
        tester.print_summary()
```

### Testing Different Model Combinations
```python
from test_llm_pipeline import run_with_custom_models

# This will test both model combinations and compare results
success = run_with_custom_models()
```

## Configuration Options

The system can be configured with the following parameters:

- **condition**: Health condition to search for (default: "IBS")
- **max_papers**: Maximum number of papers to process (default: 5)  
- **primary_model**: Primary Ollama model name (default: "mistral:7b")
- **secondary_model**: Secondary Ollama model name (default: "llama3.1:8b")
- **ollama_url**: Ollama server URL (default: "http://localhost:11434/v1")

## Available Models

The system currently supports:
- `mistral:7b` - Mistral 7B model
- `llama3.1:8b` - Llama 3.1 8B model

To add more models, update the `OllamaModel` enum in the code.

## Output

The script generates:
1. **Console Output**: Real-time progress and summary
2. **Log File**: `llm_pipeline_test.log` with detailed logging
3. **Results JSON**: `ollama_pipeline_test_results_<timestamp>.json` with complete results

## Error Handling

The system includes comprehensive error handling:
- **Connection Validation**: Checks if Ollama is running
- **Model Availability**: Verifies requested models are pulled
- **Retry Logic**: Automatic retries for transient failures
- **Graceful Degradation**: Continues with partial results when possible

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Check if the service is accessible
curl http://localhost:11434/api/tags
```

### "Model not available"
```bash
# Pull the missing model
ollama pull mistral:7b
ollama pull llama3.1:8b

# Verify installation
ollama list
```

### "Missing NCBI_API_KEY"
Set your NCBI API key in your environment or .env file:
```bash
export NCBI_API_KEY="your_ncbi_api_key_here"
```

## Performance Notes

- Local models are significantly faster than API calls once loaded
- First run may be slower as models need to be loaded into memory
- Processing time depends on your hardware specifications
- Models stay loaded in memory between runs for faster subsequent processing

## Key Improvements Over Original

1. **No API Keys Required**: Only needs NCBI API key for PubMed (not for LLMs)
2. **Cost-Free**: No charges for LLM usage
3. **Privacy**: All LLM processing happens locally
4. **Speed**: No network latency for LLM calls
5. **Reliability**: No rate limits or API downtime
6. **Flexibility**: Easy model switching and configuration