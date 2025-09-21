# MyBiome Orchestration Guide

## Updated Folder Structure

```
back_end/
├── src/                       # Core source code
│   ├── data/                     # Core data management
│   │   ├── config.py                # Unified configuration management
│   │   ├── repositories.py          # Database abstraction layer
│   │   ├── validators.py            # Main validation system
│   │   ├── api_clients.py           # API clients (PubMed, PMC, Semantic Scholar, LLM)
│   │   ├── error_handler.py         # Circuit breaker & retry logic
│   │   └── utils.py                 # Helper functions
│   │
│   ├── paper_collection/            # Data collection pipeline
│   │   ├── database_manager.py      # Database schema & operations
│   │   ├── pubmed_collector.py      # PubMed API integration
│   │   ├── semantic_scholar_enrichment.py # S2 data enrichment
│   │   ├── fulltext_retriever.py    # PMC & Unpaywall fulltext
│   │   └── paper_parser.py          # Content parsing & extraction
│   │
│   ├── llm/                         # LLM processing pipeline
│   │   ├── dual_model_analyzer.py   # Core dual-model analysis (gemma2 + qwen2.5)
│   │   ├── prompt_service.py        # Prompt templates & management
│   │   ├── pipeline.py              # LLM processing pipeline
│   │   ├── pipeline_analyzer.py     # Pipeline analysis tools
│   │   └── emerging_category_analyzer.py # New category detection
│   │
│   ├── interventions/               # Intervention classification
│   │   ├── taxonomy.py              # Intervention type definitions
│   │   ├── category_validators.py   # Category-specific validation rules
│   │   └── search_terms.py          # Search term definitions
│   │
│   ├── action_functions/            # Modular pipeline operations
│   │   └── [various action functions]
│   │
│   └── utils/                       # Utility functions
│       └── imports.py               # Import helpers
│
├── pipelines/                       # 🚀 MAIN EXECUTION SCRIPTS
│   ├── paper_collector.py           # 📚 Unified Paper Collection
│   ├── llm_processor.py             # 🤖 Unified LLM Processing
│   ├── research_orchestrator.py     # 🎯 Master Research Orchestrator
│   ├── collect_process_validate.py  # Legacy full pipeline
│   ├── reprocess_abstracts.py       # Abstract reprocessing utility
│   ├── run_s2_enrichment.py         # S2 enrichment utility
│   └── test_new_scripts.py          # Testing framework
│
├── data_mining/                     # 🔬 ADVANCED ANALYTICS & PATTERN DISCOVERY
│   ├── data_mining_orchestrator.py  # 📊 Data Mining Orchestrator
│   ├── bayesian_scorer.py           # Bayesian evidence scoring
│   ├── medical_knowledge_graph.py   # Multi-edge medical knowledge graph
│   ├── biological_patterns.py       # Biological pattern recognition
│   ├── condition_similarity_mapping.py # Condition similarity analysis
│   ├── treatment_recommendation_engine.py # AI treatment recommendations
│   ├── innovation_tracking_system.py # Emerging treatment tracking
│   ├── research_gaps.py             # Research gap identification
│   ├── power_combinations.py        # Intervention combination analysis
│   ├── failed_interventions.py      # Failed treatment analysis
│   ├── fundamental_functions.py     # Core analytical functions
│   ├── correlation_consistency_checker.py # Data consistency analysis
│   └── review_correlations.py       # Correlation review tools
│
├── testing/                         # Testing & validation
├── utilities/                       # Administrative tools
└── [other directories...]
```

## 🎯 Main Orchestration Files

### 1. **`pipelines/research_orchestrator.py`** - Master Research Workflow
**Purpose**: Coordinates complete research campaigns from data collection to final reports

**What it does**:
- Orchestrates the entire research pipeline: Collection → Processing → Validation → Reporting
- Manages multi-condition research campaigns
- Coordinates between paper_collector.py and llm_processor.py
- Provides comprehensive session management across all phases
- Handles thermal protection and error recovery throughout the workflow

**When to use**:
- ✅ **Complete research workflows** (most common use case)
- ✅ **Multi-condition campaigns**
- ✅ **Overnight/unattended operations**
- ✅ **When you want everything automated**

**Usage Examples**:
```bash
# Complete research workflow for one condition
python research_orchestrator.py "inflammatory bowel disease" --papers 500

# Multi-condition overnight campaign
python research_orchestrator.py --conditions "ibs,gerd,crohns,colitis" --papers-per-condition 1000 --overnight

# Collection only (skip processing)
python research_orchestrator.py --conditions "diabetes" --collection-only

# Processing only (papers already collected)
python research_orchestrator.py --processing-only --limit 100

# Resume interrupted campaign
python research_orchestrator.py --resume

# Check campaign status
python research_orchestrator.py --status
```

**Key Features**:
- 🔄 **4-Phase Workflow**: Collection → Processing → Validation → Reporting
- 🌡️ **Thermal Protection**: Monitors GPU temperature throughout
- 💾 **Cross-Phase Sessions**: Maintains state across all workflow phases
- 📊 **Comprehensive Reporting**: Generates detailed campaign reports
- 🔁 **Auto-Restart**: Recovers from errors and continues operation

---

### 2. **`pipelines/paper_collector.py`** - Standalone Paper Collection
**Purpose**: Robust paper collection from PubMed and Semantic Scholar

**What it does**:
- Collects papers from PubMed with intelligent search strategies
- Performs Semantic Scholar enrichment (interleaved or traditional)
- Handles network failures with exponential backoff retry
- Manages collection sessions with progress tracking
- Supports multi-condition batch processing

**When to use**:
- ✅ **Only need paper collection** (no LLM processing)
- ✅ **Building up paper database** before processing
- ✅ **Testing collection strategies**
- ✅ **Want fine control over collection parameters**

**Usage Examples**:
```bash
# Single condition collection
python paper_collector.py "ibs" --max-papers 200

# Multi-condition collection campaign
python paper_collector.py --conditions "ibs,gerd,crohns" --target-per-condition 500

# Traditional mode (no interleaved S2)
python paper_collector.py "diabetes" --traditional-mode --max-papers 100

# Resume collection session
python paper_collector.py --resume

# Check collection status
python paper_collector.py --status
```

**Key Features**:
- 🌐 **Multi-Source Collection**: PubMed + Semantic Scholar + PMC + Unpaywall
- 🔄 **Network Resilience**: Exponential backoff retry for API failures
- 📈 **Session Persistence**: Resume from interruptions
- 🎯 **Smart Search**: Interleaved S2 discovery for 6x paper expansion
- 📊 **Progress Tracking**: Real-time collection statistics

---

### 3. **`pipelines/llm_processor.py`** - Standalone LLM Processing
**Purpose**: Robust LLM-based intervention extraction with thermal protection

**What it does**:
- Processes papers using dual-model analysis (gemma2:9b + qwen2.5:14b)
- Monitors GPU temperature with predictive cooling
- Manages memory usage and performs automatic cleanup
- Tracks processing sessions with detailed metrics
- Handles processing failures with retry logic

**When to use**:
- ✅ **Only need LLM processing** (papers already collected)
- ✅ **Reprocessing existing papers** with updated models
- ✅ **Want fine control over thermal/memory settings**
- ✅ **Testing LLM processing performance**

**Usage Examples**:
```bash
# Process all unprocessed papers
python llm_processor.py --all

# Process specific number with thermal limits
python llm_processor.py --limit 50 --max-temp 75 --batch-size 3

# Reprocess failed papers
python llm_processor.py --reprocess-failed

# Process overnight with thermal monitoring
python llm_processor.py --all --overnight --thermal-status

# Resume processing session
python llm_processor.py --resume

# Check thermal status
python llm_processor.py --thermal-status
```

**Key Features**:
- 🌡️ **Advanced Thermal Protection**: Real-time GPU monitoring with predictive cooling
- 🧠 **Dual-Model Analysis**: Uses both gemma2:9b and qwen2.5:14b for robust extraction
- 💾 **Memory Management**: Automatic cleanup and optimization
- 📊 **Performance Metrics**: Detailed timing and throughput statistics
- 🔁 **Session Recovery**: Resume from any interruption point

---

### 4. **`data_mining/data_mining_orchestrator.py`** - Advanced Analytics Orchestrator
**Purpose**: Orchestrates advanced data mining and pattern discovery operations

**What it does**:
- Builds medical knowledge graphs from extracted interventions
- Performs Bayesian evidence scoring and statistical analysis
- Generates treatment recommendations and identifies research gaps
- Analyzes intervention combinations and biological patterns
- Creates comprehensive analytical reports

**When to use**:
- ✅ **After data collection and processing is complete**
- ✅ **Want to discover hidden patterns in the data**
- ✅ **Need treatment recommendations and research insights**
- ✅ **Building knowledge graphs and analytical models**

**Usage Examples**:
```bash
# Complete data mining pipeline
python data_mining_orchestrator.py --all

# Specific analyses
python data_mining_orchestrator.py --knowledge-graph --bayesian-scoring --recommendations

# Target specific conditions
python data_mining_orchestrator.py --conditions "ibs,gerd" --all

# Resume data mining session
python data_mining_orchestrator.py --resume
```

## 🔄 Recommended Workflow

### **For New Users (Complete Workflow)**:
```bash
# 1. Complete research campaign (collection + processing + analysis)
python pipelines/research_orchestrator.py "inflammatory bowel disease" --papers 500

# 2. Monitor progress
python pipelines/research_orchestrator.py --status

# 3. After completion, run data mining
python data_mining/data_mining_orchestrator.py --all --conditions "inflammatory bowel disease"
```

### **For Advanced Users (Staged Workflow)**:
```bash
# Stage 1: Collection
python pipelines/paper_collector.py --conditions "ibs,gerd,crohns" --target-per-condition 1000

# Stage 2: Processing (after collection completes)
python pipelines/llm_processor.py --all --max-temp 80

# Stage 3: Data Mining (after processing completes)
python data_mining/data_mining_orchestrator.py --all --conditions "ibs,gerd,crohns"
```

### **For Overnight Operations**:
```bash
# Complete overnight campaign
python pipelines/research_orchestrator.py --conditions "ibs,gerd,crohns,colitis,diabetes" --papers-per-condition 1000 --overnight --auto-restart

# Monitor remotely
python pipelines/research_orchestrator.py --status
python pipelines/research_orchestrator.py --thermal-monitor
```

## 🛠️ Session Management

All orchestrators support:
- **Auto-save**: Progress saved every 60 seconds
- **Resume**: `--resume` continues from interruption points
- **Status**: `--status` shows real-time progress
- **Configuration**: `--config config.json` for complex setups
- **Error Recovery**: Automatic restart capabilities

## 🌡️ Thermal Protection

GPU-intensive operations include:
- Real-time temperature monitoring
- Predictive cooling algorithms
- Automatic processing pauses
- Configurable temperature thresholds
- Hardware protection safeguards

Use `--thermal-status` to monitor and `--max-temp` to set limits.

## 📊 Output Structure

```
back_end/
├── collection_session.json         # Collection progress
├── processing_session.json         # Processing progress
├── orchestration_session.json      # Orchestration progress
├── data_mining_session.json        # Data mining progress
├── collection_results/             # Collection outputs
├── processing_results/             # Processing outputs
├── orchestration_results/          # Campaign reports
└── data_mining_results/            # Analytics outputs
```

This unified structure eliminates redundancy while providing maximum flexibility for different research workflows.