# MyBiome Health Research Pipeline

## Project Overview

An automated biomedical research pipeline that collects research papers about health conditions, extracts key intervention-outcome relationships using local LLMs, stores data in SQLite, and performs advanced data mining for correlation discovery. The system will eventually present findings through an elegant web interface.

## Architecture

**Backend**: Python-based research automation pipeline
**Frontend**: Web interface for presenting research findings
**Database**: SQLite with comprehensive schema for papers, interventions, and correlations
**LLM**: Local single-model extraction (qwen3:14b) via Ollama 

## Core Pipeline Stages

### 1. **Data Collection** (`back_end/src/data_collection/`)
- **PubMed Collection**: Automated paper retrieval via PubMed API (PRIMARY source for batch_medical_rotation)
- **Semantic Scholar**: Available but DISABLED in batch pipeline to prevent hanging (use_interleaved_s2=False)
- **Fulltext Retrieval**: PMC and Unpaywall integration for complete paper access
- **Database Management**: SQLite operations with robust schema

### 2. **LLM Processing** (`back_end/src/llm_processing/`)
- **Single-Model Extraction**: Fast processing with qwen3:14b (optimized with chain-of-thought suppression)
- **Intervention Extraction**: Structured extraction of treatment-outcome relationships including **mechanism of action** (biological/behavioral/psychological pathways)
- **Categorization Deferred**: Categories assigned in Phase 2.5 (categorization happens separately)
- **Batch Processing**: Efficient processing with thermal protection and memory management

### 2.5. **Categorization Phase** (`back_end/src/orchestration/`)
- **Intervention Categorization**: LLM-based categorization of interventions into 13 categories (exercise, diet, supplement, medication, therapy, lifestyle, surgery, test, device, procedure, biologics, gene_therapy, emerging)
- **Condition Categorization**: LLM-based categorization of health conditions into 18 categories (cardiac, neurological, digestive, etc.)
- **Batch Processing**: Processes 20 interventions/conditions per LLM call
- **Separation of Concerns**: Categorization separated from extraction for flexibility and re-categorization capability

### 3. **Semantic Grouping** (`back_end/src/llm_processing/`)
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
- **`rotation_paper_collector.py`**: Multi-source paper collection orchestrator (Phase 1)
- **`rotation_llm_processor.py`**: LLM extraction coordinator with thermal protection (Phase 2 - extracts WITHOUT categories)
- **`rotation_llm_categorization.py`**: LLM categorization coordinator (Phase 2.5 - categorizes interventions AND conditions)
- **`rotation_semantic_grouping_integrator.py`**: Semantic grouping and canonical entity merging (Phase 3)

### Data Mining Tools
- **`data_mining_orchestrator.py`**: Advanced analytics and pattern discovery coordinator
- **`medical_knowledge_graph.py`**: Knowledge graph construction and analysis
- **`treatment_recommendation_engine.py`**: AI treatment recommendation system
- **`bayesian_scorer.py`**: Evidence-based intervention scoring

## Technology Stack

### Backend
- **Python 3.13**: Core language
- **SQLite**: Primary database with comprehensive medical schema
- **Ollama**: Local LLM inference (qwen3:14b with optimized prompting)
- **Requests**: API integrations (PubMed, Semantic Scholar, PMC)
- **Circuit Breaker Pattern**: Robust error handling and retry logic
- **FAST_MODE**: Performance optimization via logging suppression (enabled by default)

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
   - Key fields: intervention_name, health_condition, **mechanism** (biological/behavioral/psychological pathway), correlation_type, sample_size, study_type

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

18. **`intervention_categories`** - Intervention taxonomy configuration (13 categories)
    - Updated by: `setup_intervention_categories()` in [`database_manager.py`](back_end/src/data_collection/database_manager.py)
    - Categories: exercise, diet, supplement, medication, therapy, lifestyle, surgery, test, **device**, **procedure**, **biologics**, **gene_therapy**, emerging

19. **`sqlite_sequence`** - SQLite internal auto-increment tracking (system table)

## Current Capabilities

### Data Collection
- ✅ Automated PubMed paper collection with intelligent search strategies
- ✅ Semantic Scholar enrichment (6x paper expansion through citations)
- ✅ PMC and Unpaywall fulltext retrieval
- ✅ Multi-condition batch processing with session persistence

### LLM Processing
- ✅ Single-model extraction with qwen3:14b (optimized with chain-of-thought suppression)
- ✅ Extraction WITHOUT categories (separate categorization phase)
- ✅ Superior extraction detail with 1.3x speed improvement over qwen2.5
- ✅ Thermal protection with GPU monitoring (RTX 4090)
- ✅ Automatic memory management and cleanup
- ✅ Session recovery and resumable processing
- ✅ Quality validation and scoring

### Categorization Phase (NEW - Phase 2.5)
- ✅ LLM-based intervention categorization (13 categories)
- ✅ LLM-based condition categorization (18 categories)
- ✅ Batch processing (20 items per LLM call)
- ✅ Separation of concerns for re-categorization flexibility

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
2. **Processing Phase**: Single-model extraction with qwen3:14b (batch size: 8 papers) - extracts interventions WITHOUT categories
2.5. **Categorization Phase**: LLM categorization of interventions AND conditions (batch size: 20 items)
3. **Semantic Grouping Phase**: Canonical entity merging (batch size: 20 interventions, cross-paper unification)

### Individual Components
```bash
# Paper collection only (Phase 1)
python -m back_end.src.orchestration.rotation_paper_collector diabetes --count 100 --no-s2

# LLM processing only (Phase 2 - extracts WITHOUT categories)
python -m back_end.src.orchestration.rotation_llm_processor diabetes --max-papers 50

# Categorization only (Phase 2.5 - categorizes interventions AND conditions)
python -m back_end.src.orchestration.rotation_llm_categorization --interventions-only
python -m back_end.src.orchestration.rotation_llm_categorization --conditions-only
python -m back_end.src.orchestration.rotation_llm_categorization  # Both

# Semantic grouping only (Phase 3)
python -m back_end.src.orchestration.rotation_semantic_grouping_integrator

# Data mining and analysis
python -m back_end.src.data_mining.data_mining_orchestrator --all
```

### Testing & Utilities
```bash
# Test single condition
python back_end/src/orchestration/main_medical_rotation.py --test-condition "hypertension" --papers 5

# Check thermal status
python back_end/src/orchestration/rotation_llm_processor.py --thermal-status

# Toggle FAST_MODE for debugging
# In .env: FAST_MODE=0 (enables full logging)
# In .env: FAST_MODE=1 (default, suppresses non-critical logs)
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

### Performance Optimization
- **FAST_MODE**: Logging suppression for high-throughput processing
  - **Enabled (default)**: `FAST_MODE=1` in `.env` - Only logs CRITICAL errors, no file logging
  - **Disabled**: `FAST_MODE=0` - Full logging (ERROR/INFO/DEBUG) with file output
  - **Impact**: Reduces I/O overhead during 1000+ papers/hour processing
  - **Implementation**: [`config.py:218-227`](back_end/src/data/config.py#L218), [`batch_file_operations.py`](back_end/src/utils/batch_file_operations.py)
  - **When to disable**: Active debugging or troubleshooting specific issues

## Current Database Status
- **Medical conditions**: 60 conditions across 12 medical specialties
- **Processing capability**: 1200+ papers per hour (qwen3:14b with optimized prompting)
- **Session management**: Comprehensive state persistence and recovery
- **Quality assurance**: Multi-stage validation and scoring
- **Architecture**: Single-model (qwen3:14b) - January 2025 optimization validated ✅

## Critical Concepts: Pipeline Architecture

### **Architecture Change (October 2025)**: Dual-Model → Single-Model ✅
**Previous**: Used gemma2:9b + qwen2.5:14b with Phase 2 consensus-before-save deduplication
**Current**: Uses qwen3:14b only - eliminates Phase 2 deduplication entirely

**Benefits**:
- 2.6x speed improvement (no dual extraction overhead + qwen3 optimization)
- No Phase 2 deduplication needed (single extraction = no same-paper duplicates)
- Preserves Qwen's superior extraction detail
- Simpler error handling and debugging
- Increased batch size from 5 to 8 papers

### **Model Optimization (January 2025)**: qwen2.5:14b → qwen3:14b ✅
**Challenge**: qwen3:14b natively generates chain-of-thought reasoning (8x slower)
**Solution**: System prompt suppression + think tag stripping

**Implementation**:
- **System Message**: `"Provide only the final JSON output without showing your reasoning process or using <think> tags. Start your response immediately with the [ character."`
- **Post-Processing**: Strip `<think>...</think>` tags from response via regex
- **Result**: qwen3:14b runs 1.3x faster than qwen2.5:14b (22.0s vs 28.9s per paper)

**Performance Comparison**:
- qwen3:14b (unoptimized): 257.5s, 7,001 chars (chain-of-thought reasoning)
- qwen3:14b (optimized): 22.0s, 630 chars (suppressed reasoning)
- qwen2.5:14b (baseline): 28.9s, 625 chars

**Benefits**:
- 1.3x faster extraction than qwen2.5:14b
- Same extraction quality and accuracy
- Same token efficiency as qwen2.5:14b
- Validates PRIMARY vs SECONDARY condition extraction correctly

### **Categorization Change (October 2025)**: Inline → Separate Phase ✅
**Previous**: Categorization happened during extraction (in prompt)
**Current**: Categorization happens in separate Phase 2.5 (after extraction)

**Benefits**:
- Separation of concerns (extraction vs classification)
- Can re-categorize without re-extraction
- Faster extraction (fewer tokens in prompt)
- Same LLM categorizes interventions AND conditions together
- More accurate categorization with focused prompts

**Workflow**:
1. **Phase 2 (Extraction)**: Extract interventions WITHOUT categories → `intervention_category = NULL`
2. **Phase 2.5 (Categorization)**: LLM categorizes interventions into 13 categories (including 'emerging' as fallback)
3. **Phase 3 (Semantic Grouping)**: Canonical entity merging across papers

### **Phase 2.5: Categorization Phase** (NEW - October 2025) ✅
**Purpose**: Categorize interventions AND conditions using LLM

**When**: Runs AFTER extraction, BEFORE semantic grouping

**How it works**:
- Interventions extracted with `intervention_category = NULL`
- LLM categorizes in batches of 20 items
- Assigns one of 13 categories (exercise, diet, supplement, medication, therapy, lifestyle, surgery, test, device, procedure, biologics, gene_therapy, emerging)
- 'emerging' used as fallback for interventions that don't fit other categories
- Same script also categorizes health conditions into 18 categories

**Stats Tracking**:
- During extraction: `category_counts['uncategorized']` tracks NULL categories
- After categorization: All interventions have proper categories (including 'emerging')

### **Phase 3: Semantic Grouping** (Cross-Paper Entity Unification) ✅ VALIDATED

**Problem**:
- Paper A mentions "vitamin D"
- Paper B mentions "Vitamin D3"
- Paper C mentions "cholecalciferol"
- These are the same thing but with different names

**Solution** (in `batch_entity_processor.py` → `batch_group_entities_semantically()`):
- All three interventions point to the same canonical entity (e.g., `canonical_id: 1, canonical_name: "vitamin D"`)
- Statistical analysis aggregates all evidence under the canonical entity
- **CRITICAL**: Original intervention names are preserved for transparency (NO DELETIONS)
- Uses LLM semantic analysis (qwen3:14b) for intelligent grouping
- Batch size: 20 interventions per LLM call

**Performance**:
- **Batch size**: 20 interventions per LLM call
- **Input size**: ~600 tokens (20 short intervention names)
- **Task**: Simple name comparison and grouping

**Validation Results** (October 2025):
- ✅ 284 interventions preserved (0 deletions)
- ✅ 71 canonical entities created
- ✅ 204 entity mappings created
- ✅ 204 interventions linked to canonical entities
- ✅ Semantic grouping working correctly (e.g., "metformin" = "metformin therapy" = "metformin treatment")

**Result**: Unified analysis showing "150 papers support vitamin D" instead of fragmented counts

## Intervention Taxonomy (13 Categories)

**Last Updated**: October 2025 - Expanded from 9 to 13 categories
**Subcategories**: Removed - all intervention classification is done at the primary category level

### Category Definitions

| # | Category | Display Name | Description | Examples |
|---|----------|-------------|-------------|----------|
| 1 | `exercise` | Exercise & Physical Activity | Physical exercise interventions | Aerobic exercise, resistance training, yoga, walking, swimming, HIIT |
| 2 | `diet` | Diet & Nutrition | Dietary interventions and nutritional modifications | Mediterranean diet, ketogenic diet, intermittent fasting, caloric restriction |
| 3 | `supplement` | Supplements & Nutraceuticals | Nutritional supplements | Vitamin D, probiotics, omega-3, herbal supplements, minerals |
| 4 | `medication` | Medications & Pharmaceuticals | Small molecule pharmaceutical drugs | Statins, metformin, antidepressants, antibiotics, pain medications |
| 5 | `therapy` | Therapy & Counseling | Psychological, physical, and behavioral therapies | Cognitive behavioral therapy, physical therapy, massage, acupuncture |
| 6 | `lifestyle` | Lifestyle Modifications | Behavioral and lifestyle changes | Sleep hygiene, stress management, smoking cessation, social support |
| 7 | `surgery` | Surgical Interventions | Surgical procedures and operations | Laparoscopic surgery, cardiac surgery, bariatric surgery, joint replacement |
| 8 | `test` | Tests & Diagnostics | Medical tests and diagnostic procedures | Blood tests, genetic testing, colonoscopy, biomarker analysis, imaging |
| 9 | **`device`** | Medical Devices & Implants | Medical devices, implants, wearables, monitors | **Pacemakers, insulin pumps, CPAP machines, continuous glucose monitors, hearing aids** |
| 10 | **`procedure`** | Medical Procedures | Non-surgical medical procedures | **Endoscopy, dialysis, blood transfusion, radiation therapy, chemotherapy** |
| 11 | **`biologics`** | Biological Medicines | Biological drugs and immunotherapies | **Monoclonal antibodies, vaccines, immunotherapies, insulin, TNF inhibitors** |
| 12 | **`gene_therapy`** | Gene & Cellular Therapy | Genetic and cellular interventions | **CRISPR gene editing, CAR-T cell therapy, stem cell therapy, gene transfer** |
| 13 | `emerging` | Emerging Interventions | Novel interventions that don't fit existing categories | Digital therapeutics, precision medicine, AI-guided interventions |

### Design Philosophy

**LLM-Based Classification**: All category assignment is performed by the LLM (qwen3:14b) in **Phase 2.5** (separate from extraction)
- **Broad categories**: Designed to group similar interventions (e.g., swimming + cycling = exercise)
- **Strict validation**: LLM must select exactly one of the 13 categories
- **No subcategories**: Removed for simplicity and flexibility
- **Emerging category**: Safety valve for truly novel interventions that don't fit any category

**Key Distinctions**:
- **medication vs biologics**: Small molecules vs biological drugs (antibodies, vaccines)
- **surgery vs procedure**: Surgical operations vs non-surgical procedures (dialysis, endoscopy)
- **device vs medication**: Hardware/implants vs pharmaceuticals
- **therapy vs lifestyle**: Professional therapeutic interventions vs self-directed behavioral changes

### Technical Implementation

**Files**:
- [`taxonomy.py`](back_end/src/interventions/taxonomy.py) - Category definitions and structures
- [`category_validators.py`](back_end/src/interventions/category_validators.py) - Validation logic
- [`validators.py`](back_end/src/data/validators.py) - Base validation rules (allows NULL categories)
- [`rotation_llm_categorization.py`](back_end/src/orchestration/rotation_llm_categorization.py) - Categorization orchestrator
- [`search_terms.py`](back_end/src/interventions/search_terms.py) - PubMed search terms per category

**Workflow** (Updated October 2025):
1. **Phase 2 (Extraction)**: LLM extracts intervention WITHOUT category → `intervention_category = NULL`
2. **Phase 2.5 (Categorization)**: LLM categorizes interventions in batches of 20
3. **Validation**: Category validated against 13 allowed values
4. **Database Update**: `intervention_category` field updated from NULL to assigned category

**Migration Notes** (October 2025):
- Expanded from 9 to 13 categories (added: device, procedure, biologics, gene_therapy)
- Removed subcategory handling entirely (subcategories no longer used)
- Database schema has no `intervention_subcategory` column
- All 284 existing interventions preserved during expansion