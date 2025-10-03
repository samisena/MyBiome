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
- **`batch_medical_rotation.py`**: Complete batch workflow orchestration for 60 medical conditions
- **`rotation_llm_processor.py`**: LLM processing coordinator with thermal protection
- **`rotation_paper_collector.py`**: Multi-source paper collection orchestrator
- **`rotation_semantic_grouping_integrator.py`**: Phase 3 semantic grouping and canonical entity merging

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

**Database**: `intervention_research.db` (SQLite)
**Total Tables**: 19 tables organized into 4 categories

### Core Data Tables (2 tables)
**Updated by**: Data collection pipeline

1. **`papers`** - PubMed articles with metadata and fulltext
   - Updated by: [`pubmed_collector.py`](back_end/src/data_collection/pubmed_collector.py), [`semantic_scholar_enrichment.py`](back_end/src/data_collection/semantic_scholar_enrichment.py)
   - Methods: `insert_paper()`, `insert_papers_batch()` in [`database_manager.py`](back_end/src/data_collection/database_manager.py)

2. **`interventions`** - Extracted treatments and outcomes with LLM processing metadata
   - Updated by: [`single_model_analyzer.py`](back_end/src/llm_processing/single_model_analyzer.py), [`batch_entity_processor.py`](back_end/src/llm_processing/batch_entity_processor.py)
   - Methods: `insert_intervention()`, `insert_intervention_normalized()` in [`database_manager.py`](back_end/src/data_collection/database_manager.py)

### Phase 3 Semantic Grouping Tables (4 tables)
**Updated by**: [`rotation_semantic_grouping_integrator.py`](back_end/src/orchestration/rotation_semantic_grouping_integrator.py) via [`batch_entity_processor.py`](back_end/src/llm_processing/batch_entity_processor.py)

3. **`canonical_entities`** - Unified intervention/condition names (e.g., "vitamin D" = "Vitamin D3" = "cholecalciferol")
   - Created by: `batch_group_entities_semantically()` → `create_canonical_entity()`
   - Current: 72 canonical entities

4. **`entity_mappings`** - Links original names to canonical entities
   - Created by: `batch_group_entities_semantically()` → `_link_interventions_to_canonical()`
   - Current: 133 mappings

5. **`llm_normalization_cache`** - LLM-based normalization cache for performance
   - Updated by: `get_or_compute_normalized_term()` in [`batch_entity_processor.py`](back_end/src/llm_processing/batch_entity_processor.py)

6. **`normalized_terms_cache`** - Fast lookup cache for normalized terms
   - Updated by: `get_or_compute_normalized_term()` in [`batch_entity_processor.py`](back_end/src/llm_processing/batch_entity_processor.py)

### Data Mining Analytics Tables (11 tables)
**Updated by**: [`data_mining_orchestrator.py`](back_end/src/data_mining/data_mining_orchestrator.py) via specialized analyzers

7. **`knowledge_graph_nodes`** - Nodes in medical knowledge graph (interventions, conditions)
   - Updated by: [`medical_knowledge_graph.py`](back_end/src/data_mining/medical_knowledge_graph.py)
   - Repository: `save_knowledge_graph_node()` in [`data_mining_repository.py`](back_end/src/data_collection/data_mining_repository.py)

8. **`knowledge_graph_edges`** - Multi-edge graph relationships with full study metadata
   - Updated by: [`medical_knowledge_graph.py`](back_end/src/data_mining/medical_knowledge_graph.py)
   - Repository: `save_knowledge_graph_edge()` in [`data_mining_repository.py`](back_end/src/data_collection/data_mining_repository.py)

9. **`bayesian_scores`** - Bayesian evidence scoring with Beta distribution analysis
   - Updated by: [`bayesian_scorer.py`](back_end/src/data_mining/bayesian_scorer.py)
   - Repository: `save_bayesian_score()` in [`data_mining_repository.py`](back_end/src/data_collection/data_mining_repository.py)

10. **`treatment_recommendations`** - AI-powered treatment recommendations
    - Updated by: [`treatment_recommendation_engine.py`](back_end/src/data_mining/treatment_recommendation_engine.py)
    - Repository: `save_treatment_recommendation()` in [`data_mining_repository.py`](back_end/src/data_collection/data_mining_repository.py)

11. **`research_gaps`** - Identified under-researched areas
    - Updated by: [`research_gaps.py`](back_end/src/data_mining/research_gaps.py)
    - Repository: `save_research_gap()` in [`data_mining_repository.py`](back_end/src/data_collection/data_mining_repository.py)

12. **`innovation_tracking`** - Emerging treatment tracking
    - Updated by: [`innovation_tracking_system.py`](back_end/src/data_mining/innovation_tracking_system.py)

13. **`biological_patterns`** - Mechanism and pattern discovery
    - Updated by: [`biological_patterns.py`](back_end/src/data_mining/biological_patterns.py)

14. **`condition_similarities`** - Condition similarity matrix
    - Updated by: [`condition_similarity_mapping.py`](back_end/src/data_mining/condition_similarity_mapping.py)

15. **`intervention_combinations`** - Synergistic combination analysis
    - Updated by: [`power_combinations.py`](back_end/src/data_mining/power_combinations.py)

16. **`failed_interventions`** - Catalog of ineffective treatments
    - Updated by: [`failed_interventions.py`](back_end/src/data_mining/failed_interventions.py)

17. **`data_mining_sessions`** - Session tracking for analytics pipeline
    - Updated by: [`data_mining_orchestrator.py`](back_end/src/data_mining/data_mining_orchestrator.py)
    - Repository: `save_data_mining_session()` in [`data_mining_repository.py`](back_end/src/data_collection/data_mining_repository.py)

### Configuration & System Tables (2 tables)

18. **`intervention_categories`** - Intervention taxonomy configuration
    - Updated by: `setup_intervention_categories()` in [`database_manager.py`](back_end/src/data_collection/database_manager.py)

19. **`sqlite_sequence`** - SQLite internal auto-increment tracking (system table)

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

# Resume from specific phase (collection, processing, or semantic_grouping)
python -m back_end.src.orchestration.batch_medical_rotation --resume --start-phase processing

# Check current status
python -m back_end.src.orchestration.batch_medical_rotation --status
```

**Pipeline Phases**:
1. **Collection Phase**: Collects papers for all 60 conditions (PubMed only, S2 disabled, 2 parallel workers)
2. **Processing Phase**: Single-model extraction with qwen2.5:14b (batch size: 8 papers)
3. **Semantic Grouping Phase**: Phase 3 canonical entity merging (batch size: 20 interventions, cross-paper unification)

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

### **Phase 3: Semantic Grouping** (Cross-Paper Entity Unification) ✅ VALIDATED
**Purpose**: Aggregate all evidence about the same intervention across ALL papers

**When**: Runs AFTER all extractions complete, operates on database records (cleanup and unification)

**Problem**:
- Paper A mentions "vitamin D"
- Paper B mentions "Vitamin D3"
- Paper C mentions "cholecalciferol"
- These are the same thing but with different names

**Solution** (in `batch_entity_processor.py` → `batch_group_entities_semantically()`):
- All three interventions point to the same canonical entity (e.g., `canonical_id: 1, canonical_name: "vitamin D"`)
- Statistical analysis aggregates all evidence under the canonical entity
- **CRITICAL**: Original intervention names are preserved for transparency (NO DELETIONS)
- Uses LLM semantic analysis (qwen2.5:14b) for intelligent grouping
- Batch size: 20 interventions per LLM call (2.5x larger than Phase 2)

**Performance** (Phase 3 vs Phase 2):
- Phase 3 is significantly faster than Phase 2 because:
  1. **Larger batch size**: 20 interventions vs 8 papers
  2. **Smaller input**: Compares 20 short names (~600 tokens) vs 8 full papers (~40,000-120,000 tokens)
  3. **Simpler task**: Name comparison vs structured extraction with 15+ fields

**Validation Results** (October 2025):
- ✅ 284 interventions preserved (0 deletions)
- ✅ 71 canonical entities created
- ✅ 204 entity mappings created
- ✅ 204 interventions linked to canonical entities
- ✅ Semantic grouping working correctly (e.g., "metformin" = "metformin therapy" = "metformin treatment")

**Result**: Unified analysis showing "150 papers support vitamin D" instead of fragmented counts