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
- **PubMed Collection**: Automated paper retrieval via PubMed API
- **Semantic Scholar Enrichment**: Additional metadata and paper discovery (6x expansion)
- **Fulltext Retrieval**: PMC and Unpaywall integration for complete paper access
- **Database Management**: SQLite operations with robust schema

### 2. **LLM Processing** (`back_end/src/llm_processing/`)
- **Dual-Model Analysis**: Parallel processing with gemma2:9b and qwen2.5:14b
- **Intervention Extraction**: Structured extraction of treatment-outcome relationships
- **Batch Processing**: Efficient processing with thermal protection and memory management
- **Entity Operations**: Deduplication and normalization of extracted entities

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

### Complete Workflow
```bash
# Start medical rotation pipeline (10 papers per condition across 60 conditions)
python back_end/src/orchestration/main_medical_rotation.py

# Resume interrupted session
python back_end/src/orchestration/main_medical_rotation.py --resume

# Check current status
python back_end/src/orchestration/main_medical_rotation.py --status
```

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

## Restraints & Guidelines
- **No emojis in any output or code**
- **Local LLM only** (gemma2:9b, qwen2.5:14b via Ollama)
- **Thermal limits**: Max GPU temperature 85°C with predictive cooling
- **Session-based operation**: All long-running tasks must be resumable
- **Robust error handling**: Circuit breaker patterns for all external API calls