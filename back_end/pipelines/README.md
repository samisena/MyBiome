# MyBiome Pipeline Scripts

This directory contains the unified pipeline scripts for the MyBiome research platform.

## Core Scripts

### 📚 `paper_collector.py` - Unified Paper Collection
Robust overnight paper collection with comprehensive features.

**Usage**:
```bash
# Single condition
python paper_collector.py "ibs" --max-papers 100

# Multi-condition overnight campaign
python paper_collector.py --conditions "ibs,gerd,crohns" --target-per-condition 1000 --overnight

# Resume interrupted session
python paper_collector.py --resume
```

**Features**:
- Multi-condition batch processing
- Network resilience with retry logic
- Session persistence and recovery
- S2 enrichment (interleaved and traditional)
- Progress tracking and auto-save
- Overnight operation capacity

### 🤖 `llm_processor.py` - Unified LLM Processing
Robust LLM processing with thermal protection and session management.

**Usage**:
```bash
# Standard processing with thermal protection
python llm_processor.py --limit 50 --max-temp 80

# Process all papers overnight
python llm_processor.py --all --overnight

# Resume with thermal monitoring
python llm_processor.py --resume --thermal-status
```

**Features**:
- Advanced thermal protection with GPU monitoring
- Dual-model analysis (gemma2:9b + qwen2.5:14b)
- Memory optimization and cleanup
- Session persistence and recovery
- Progress checkpoints and auto-save
- Real-time performance metrics

### 🎯 `research_orchestrator.py` - Master Orchestrator
Coordinates complete research workflows by calling the other scripts.

**Usage**:
```bash
# Complete workflow
python research_orchestrator.py "ibs" --papers 500

# Multi-condition overnight campaign
python research_orchestrator.py --conditions "ibs,gerd,crohns" --papers-per-condition 1000 --overnight

# Collection only
python research_orchestrator.py --conditions "diabetes" --collection-only
```

**Features**:
- Intelligent workflow coordination
- Cross-phase session management
- Multi-condition campaigns
- Workflow phase control (collection → processing → validation → reporting)
- Complete fault tolerance
- Comprehensive reporting

## Workflow Phases

The orchestrator manages these workflow phases:

1. **Collection Phase**: Gather papers from PubMed/S2 (`paper_collector.py`)
2. **Processing Phase**: Extract interventions via LLM (`llm_processor.py`)
3. **Validation Phase**: Quality checks and data validation
4. **Reporting Phase**: Generate comprehensive reports

## Session Management

All scripts support session persistence:
- **Auto-save**: Progress saved every 60 seconds
- **Resume**: Continue from interruption points
- **Status monitoring**: Real-time progress tracking
- **Error recovery**: Automatic restart capabilities

## Thermal Protection

Scripts with GPU operations include:
- Real-time temperature monitoring
- Predictive cooling algorithms
- Automatic processing pauses
- Safe operating thresholds
- Hardware protection

## Configuration

### Command Line
All scripts support comprehensive command-line options.

### JSON Configuration
```bash
python script.py --config config.json
```

### Environment Variables
Scripts use `.env` file for API keys and database paths.

## Output

### Session Files
- `collection_session.json` - Collection progress
- `processing_session.json` - Processing progress
- `orchestration_session.json` - Orchestration progress

### Results
- `collection_results/` - Collection outputs and reports
- `processing_results/` - Processing outputs and metrics
- `orchestration_results/` - Campaign reports and summaries

## Utility Scripts

### `collect_process_validate.py`
Legacy full pipeline script (consider using `research_orchestrator.py` instead).

### `reprocess_abstracts.py`
Utility for reprocessing existing abstracts.

### `run_s2_enrichment.py`
Standalone Semantic Scholar enrichment.

## Legacy Scripts (Replaced)

The following scripts have been replaced by the unified versions:
- `collect_papers.py.OLD` → Use `paper_collector.py`
- `run_llm_processing.py.OLD` → Use `llm_processor.py`
- `robust_llm_processor.py.OLD` → Use `llm_processor.py`

## Quick Start

### For New Users
```bash
# 1. Complete research workflow for one condition
python research_orchestrator.py "inflammatory bowel disease" --papers 100

# 2. Monitor progress
python research_orchestrator.py --status

# 3. Resume if interrupted
python research_orchestrator.py --resume
```

### For Advanced Users
```bash
# 1. Collection only with custom settings
python paper_collector.py --conditions "ibs,gerd" --target-per-condition 500 --traditional-mode

# 2. Processing with thermal limits
python llm_processor.py --all --max-temp 75 --batch-size 3

# 3. Custom workflow phases
python research_orchestrator.py --conditions "diabetes" --phases "collection,processing" --no-validation
```

## Troubleshooting

### Import Errors
Ensure you're running from the `back_end` directory and Python path is correct.

### Thermal Issues
Use `--thermal-status` to monitor GPU temperature and adjust `--max-temp` if needed.

### Session Recovery
Use `--resume` to continue interrupted operations. Session files contain full state.

### Performance
- Adjust `--batch-size` for memory constraints
- Use `--checkpoint-interval` for frequent saving
- Monitor with `--status` for real-time progress

For detailed documentation, see the individual script help:
```bash
python script_name.py --help
```