# MyBiome Health Research Pipeline

## Project Overview

An automated biomedical research pipeline that collects research papers about health conditions, extracts key intervention-outcome relationships using local LLMs, stores data in SQLite, and performs advanced data mining for correlation discovery. The system will eventually present findings through an elegant web interface.

## Architecture

**Backend**: Python-based research automation pipeline
**Frontend**: Web interface for presenting research findings
**Database**: SQLite with comprehensive schema for papers, interventions, and correlations
**LLMs**: Local dual-model analysis (gemma2:9b + qwen2.5:14b) via Ollama

## Core Pipeline Stages

### 1. **Data Collection** (`back_end/src/data_collection/`)
- **PubMed Collection**: Automated paper retrieval via PubMed API (PRIMARY source for batch_medical_rotation)
- **Semantic Scholar**: Available but DISABLED in batch pipeline to prevent hanging (use_interleaved_s2=False)
- **Fulltext Retrieval**: PMC and Unpaywall integration for complete paper access
- **Database Management**: SQLite operations with robust schema

### 2. **LLM Processing** (`back_end/src/llm_processing/`)
- **Dual-Model Analysis**: Sequential processing with gemma2:9b and qwen2.5:14b (both models process each paper independently)
- **Intervention Extraction**: Structured extraction of treatment-outcome relationships
- **Batch Processing**: Efficient processing with thermal protection and memory management
- **Same-Paper Deduplication**: Intelligent merging of duplicate extractions from the same paper
- **Canonical Entity Merging**: Unification of intervention names across all papers (e.g., "vitamin D", "Vitamin D3", "cholecalciferol" → single canonical entity)

### 3. **Data Mining** (`back_end/src/data_mining/`)
- **Pattern Discovery**: Advanced correlation analysis and biological pattern recognition
- **Knowledge Graphs**: Multi-edge medical knowledge graphs for relationship mapping
- **Bayesian Scoring**: Evidence-based scoring for intervention effectiveness
- **Treatment Recommendations**: AI-powered treatment suggestion engine
- **Research Gap Analysis**: Identification of under-researched areas

### 4. **Orchestration** (`back_end/src/orchestration/`)
- **Main Medical Rotation**: Circular processing through 60 medical conditions (12 specialties × 5 conditions)
- **Session Management**: Resumable execution with comprehensive state persistence
- **Error Recovery**: Robust fault tolerance with automatic retry and recovery
- **Thermal Protection**: GPU monitoring with predictive cooling algorithms

## Key Execution Scripts

### Primary Orchestrators
- **`main_medical_rotation.py`**: Complete workflow orchestration for rotating through medical conditions
- **`rotation_llm_processor.py`**: LLM processing coordinator with thermal protection
- **`rotation_paper_collector.py`**: Multi-source paper collection orchestrator
- **`rotation_deduplication_integrator.py`**: Entity deduplication and integration

### Data Mining Tools
- **`data_mining_orchestrator.py`**: Advanced analytics and pattern discovery coordinator
- **`medical_knowledge_graph.py`**: Knowledge graph construction and analysis
- **`treatment_recommendation_engine.py`**: AI treatment recommendation system
- **`bayesian_scorer.py`**: Evidence-based intervention scoring

## Technology Stack

### Backend
- **Python 3.13**: Core language
- **SQLite**: Primary database with comprehensive medical schema
- **Ollama**: Local LLM inference (gemma2:9b, qwen2.5:14b)
- **Requests**: API integrations (PubMed, Semantic Scholar, PMC)
- **Circuit Breaker Pattern**: Robust error handling and retry logic

### APIs & Data Sources
- **PubMed API**: Primary research paper source
- **Semantic Scholar API**: Paper enrichment and discovery
- **PMC (PubMed Central)**: Fulltext paper access
- **Unpaywall API**: Open access paper retrieval

### Frontend (In Development)
- **HTML/CSS/JavaScript**: Core web technologies
- **DataTables.js**: Interactive data presentation
- **GitHub Pages**: Static hosting (planned)

## Database Schema

Comprehensive SQLite database with tables for:
- **Papers**: PubMed articles with metadata and fulltext
- **Interventions**: Extracted treatments and outcomes
- **Correlations**: Statistical relationships between interventions and conditions
- **Processing Sessions**: Session state and progress tracking
- **Quality Metrics**: Validation and scoring data

## Current Capabilities

### Data Collection
- ✅ Automated PubMed paper collection with intelligent search strategies
- ✅ Semantic Scholar enrichment (6x paper expansion through citations)
- ✅ PMC and Unpaywall fulltext retrieval
- ✅ Multi-condition batch processing with session persistence

### LLM Processing
- ✅ Dual-model consensus analysis for robust extraction
- ✅ Thermal protection with GPU monitoring (RTX 4090)
- ✅ Automatic memory management and cleanup
- ✅ Session recovery and resumable processing
- ✅ Quality validation and scoring

### Data Mining
- ✅ Advanced correlation analysis and pattern recognition
- ✅ Medical knowledge graph construction
- ✅ Bayesian evidence scoring
- ✅ Treatment recommendation engine
- ✅ Research gap identification

## Operational Commands

### Complete Workflow (batch_medical_rotation.py)
```bash
# Start batch medical rotation pipeline (recommended: 10 papers per condition across 60 conditions)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10

# Resume interrupted session
python -m back_end.src.orchestration.batch_medical_rotation --resume

# Resume from specific phase (collection, processing, or deduplication)
python -m back_end.src.orchestration.batch_medical_rotation --resume --start-phase processing

# Check current status
python -m back_end.src.orchestration.batch_medical_rotation --status
```

**Pipeline Phases**:
1. **Collection Phase**: Collects papers for all 60 conditions (PubMed only, S2 disabled, 2 parallel workers)
2. **Processing Phase**: Dual-model extraction (gemma2:9b → qwen2.5:14b sequential, creates duplicates)
3. **Deduplication Phase**: Same-paper duplicate removal + canonical entity merging

### Individual Components
```bash
# LLM processing only
python back_end/src/orchestration/rotation_llm_processor.py --limit 50

# Paper collection only
python back_end/src/orchestration/rotation_paper_collector.py --condition "diabetes" --papers 100

# Data mining and analysis
python back_end/src/data_mining/data_mining_orchestrator.py --all
```

### Testing & Utilities
```bash
# Test single condition
python back_end/src/orchestration/main_medical_rotation.py --test-condition "hypertension" --papers 5

# Check thermal status
python back_end/src/orchestration/rotation_llm_processor.py --thermal-status
```

## Project Conventions

### Code Style
- **No emojis in print statements or code comments**
- **Comprehensive error handling with circuit breaker patterns**
- **Session persistence for all long-running operations**
- **Detailed logging and progress tracking**
- **Thermal protection for GPU-intensive operations**

### File Organization
- **Source code**: All core modules in `back_end/src/`
- **Execution scripts**: Primary orchestrators in `back_end/src/orchestration/`
- **Data mining**: Advanced analytics in `back_end/src/data_mining/`
- **Configuration**: Centralized config management in `back_end/src/data/config.py`
- **Session files**: JSON session state files in `back_end/data/`

### Development Workflow
- **Resumable by design**: All operations support interruption and recovery
- **Modular architecture**: Clear separation between collection, processing, and analysis
- **Comprehensive validation**: Multi-level validation for data quality
- **Thermal awareness**: GPU temperature monitoring for sustainable operation

## Current Database Status
- **Medical conditions**: 60 conditions across 12 medical specialties
- **Processing capability**: 500+ papers per hour (dual-model analysis)
- **Session management**: Comprehensive state persistence and recovery
- **Quality assurance**: Multi-stage validation and scoring

## Critical Concepts: Deduplication vs Canonical Merging

### **Same-Paper Deduplication** (Within-Paper Duplicate Removal)
**Purpose**: Prevent double-counting when both LLM models extract the same finding from the same paper

**Problem**:
- Paper 41031311 is processed by gemma2:9b → extracts "vitamin D for cognitive impairment"
- Same paper 41031311 is processed by qwen2.5:14b → extracts "vitamin D for type 2 diabetes mellitus-induced cognitive impairment"
- Without deduplication: This appears as 2 separate findings, inflating evidence counts

**Solution** (in `batch_entity_processor.py`):
1. **Simple Normalization**: Try basic string matching first (fast, cheap)
2. **LLM Semantic Verification**: If normalization fails, use qwen2.5:14b to determine if conditions are semantically equivalent
3. **Merge Action**: Delete one record, keep ONE intervention per paper
4. **Model Attribution**: Update the kept record to show extraction by BOTH models (e.g., `models_used: "gemma2:9b, qwen2.5:14b"`)
5. **Consensus Naming**: Use LLM to decide which condition wording is most accurate

**Result**: Each paper contributes only ONE record per unique intervention, preventing statistical inflation

### **Canonical Entity Merging** (Cross-Paper Entity Unification)
**Purpose**: Aggregate all evidence about the same intervention across ALL papers

**Problem**:
- Paper A mentions "vitamin D"
- Paper B mentions "Vitamin D3"
- Paper C mentions "cholecalciferol"
- These are the same thing but with different names

**Solution**:
- All three interventions point to the same canonical entity (e.g., `canonical_id: 1, canonical_name: "vitamin D"`)
- Statistical analysis aggregates all evidence under the canonical entity
- Original intervention names are preserved for transparency

**Result**: Unified analysis showing "150 papers support vitamin D" instead of fragmented counts

### Key Differences

| Aspect | Same-Paper Deduplication | Canonical Merging |
|--------|-------------------------|-------------------|
| **Scope** | Within single paper | Across all papers |
| **Action** | DELETE duplicate records | LINK different names to same entity |
| **Problem** | Dual-model extraction creates duplicates | Different papers use different terminology |
| **Timing** | Happens immediately after LLM extraction | Happens during entity normalization |
| **Impact** | Prevents evidence inflation | Enables aggregated analysis |

## Restraints & Guidelines
- **No emojis in any output or code**
- **Local LLM only** (gemma2:9b, qwen2.5:14b via Ollama)
- **Thermal limits**: Max GPU temperature 85°C with predictive cooling
- **Session-based operation**: All long-running tasks must be resumable
- **Robust error handling**: Circuit breaker patterns for all external API calls