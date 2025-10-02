# MyBiome Health Research Pipeline

## Project Overview

An automated biomedical research pipeline that collects research papers about health conditions, extracts key intervention-outcome relationships using local LLMs, stores data in SQLite, and performs advanced data mining for correlation discovery. The system will eventually present findings through an elegant web interface.

## Architecture

**Backend**: Python-based research automation pipeline
**Frontend**: Web interface for presenting research findings
**Database**: SQLite with comprehensive schema for papers, interventions, and correlations
**LLM**: Local single-model extraction (qwen2.5:14b) via Ollama 

## Core Pipeline Stages

### 1. **Data Collection** (`back_end/src/data_collection/`)
- **PubMed Collection**: Automated paper retrieval via PubMed API (PRIMARY source for batch_medical_rotation)
- **Semantic Scholar**: Available but DISABLED in batch pipeline to prevent hanging (use_interleaved_s2=False)
- **Fulltext Retrieval**: PMC and Unpaywall integration for complete paper access
- **Database Management**: SQLite operations with robust schema

### 2. **LLM Processing** (`back_end/src/llm_processing/`)
- **Single-Model Extraction**: Fast processing with qwen2.5:14b 
- **Intervention Extraction**: Structured extraction of treatment-outcome relationships with superior detail
- **Batch Processing**: Efficient processing with thermal protection and memory management
- **Phase 3 Canonical Merging**: Cross-paper unification of intervention names (e.g., "vitamin D", "Vitamin D3", "cholecalciferol" → single canonical entity)

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
- **Ollama**: Local LLM inference (qwen2.5:14b)
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
- ✅ Single-model extraction with qwen2.5:14b 
- ✅ Superior extraction detail preserved from Qwen model
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
2. **Processing Phase**: Single-model extraction with qwen2.5:14b 
3. **Deduplication Phase**: Phase 3 canonical entity merging (cross-paper unification)

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
- **Processing capability**: 1000+ papers per hour (single-model extraction, 2x improvement)
- **Session management**: Comprehensive state persistence and recovery
- **Quality assurance**: Multi-stage validation and scoring
- **Architecture**: Single-model (qwen2.5:14b) - October 2025 migration validated ✅

## Critical Concepts: Single-Model vs Canonical Merging

### **Architecture Change (October 2025)**: Dual-Model → Single-Model ✅
**Previous**: Used gemma2:9b + qwen2.5:14b with Phase 2 consensus-before-save deduplication
**Current**: Uses qwen2.5:14b only - eliminates Phase 2 entirely

**Benefits**:
- 2x speed improvement (no dual extraction overhead)
- No Phase 2 deduplication needed (single extraction = no same-paper duplicates)
- Preserves Qwen's superior extraction detail (avoids "atorvastatin" vs "atorvastatin pretreatment" conflicts)
- Simpler error handling and debugging
- Increased batch size from 5 to 8 papers

### **Phase 2: DEPRECATED** (Previously: Same-Paper Deduplication)
**Status**: No longer needed with single-model architecture
**Historical Purpose**: Prevented double-counting when both LLM models extracted the same finding
**Why Removed**: Single-model extraction creates only one record per finding per paper

**Historical Context** (Dual-Model Era):
- Paper 41031311 was processed by gemma2:9b → extracted "vitamin D for cognitive impairment"
- Same paper 41031311 was processed by qwen2.5:14b → extracted "vitamin D for type 2 diabetes mellitus-induced cognitive impairment"
- Complex consensus logic was needed to merge these duplicates

**Current Reality** (Single-Model Era):
- Paper 41031311 is processed by qwen2.5:14b ONCE → extracts "vitamin D for type 2 diabetes mellitus-induced cognitive impairment"
- No duplicates created, no consensus logic needed
- Preserves Qwen's superior detail without conflicts

### **Phase 3: Canonical Entity Merging** (Cross-Paper Entity Unification)
**Purpose**: Aggregate all evidence about the same intervention across ALL papers

**When**: Runs AFTER all extractions complete, operates on database records (cleanup and unification)

**Problem**:
- Paper A mentions "vitamin D"
- Paper B mentions "Vitamin D3"
- Paper C mentions "cholecalciferol"
- These are the same thing but with different names

**Solution** (in `batch_entity_processor.py` → `batch_deduplicate_entities()`):
- All three interventions point to the same canonical entity (e.g., `canonical_id: 1, canonical_name: "vitamin D"`)
- Statistical analysis aggregates all evidence under the canonical entity
- Original intervention names are preserved for transparency
- Uses LLM semantic analysis for intelligent grouping

**Result**: Unified analysis showing "150 papers support vitamin D" instead of fragmented counts