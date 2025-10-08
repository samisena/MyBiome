# MyBiome Health Research Pipeline

## Project Overview

Automated biomedical research pipeline that collects research papers about health conditions, extracts intervention-outcome relationships using local LLMs, and performs semantic normalization and data mining. Presents findings through an interactive web interface.

## Quick Start

**Environment**: Conda environment called 'venv'
```bash
conda activate venv

# Run complete pipeline (10 papers per condition across 60 conditions)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10

# Resume interrupted session
python -m back_end.src.orchestration.batch_medical_rotation --resume

# Check status
python -m back_end.src.orchestration.batch_medical_rotation --status
```

---

## Architecture

**Backend**: Python 3.13 research automation pipeline
**Frontend**: HTML/CSS/JavaScript web interface with DataTables.js
**Database**: SQLite with 19 tables for papers, interventions, and analytics
**LLM**: Local single-model extraction (qwen3:14b) via Ollama

---

## Core Pipeline Phases

### Phase 1: Data Collection
- **PubMed API**: Primary source for paper collection
- **PMC & Unpaywall**: Fulltext retrieval
- **Output**: Papers stored in `papers` table with `processing_status = 'pending'`

### Phase 2: LLM Processing
- **Model**: qwen3:14b (optimized with chain-of-thought suppression)
- **Extracts**: Intervention-outcome relationships + **mechanism of action** (biological/behavioral/psychological pathways)
- **Output**: Interventions saved WITHOUT categories → `intervention_category = NULL`
- **Performance**: ~38-39 papers/hour (~93s per paper)

### Phase 2.5: Categorization
- **Intervention Categories**: 13 categories (exercise, diet, supplement, medication, therapy, lifestyle, surgery, test, device, procedure, biologics, gene_therapy, emerging)
- **Condition Categories**: 18 categories (cardiac, neurological, digestive, etc.)
- **Batch Processing**: 20 items per LLM call (qwen3:14b)
- **Output**: All interventions and conditions categorized

### Phase 3.5: Hierarchical Semantic Normalization ✅
- **4-Layer Hierarchy**:
  - **Layer 0**: Category (from 13-category taxonomy)
  - **Layer 1**: Canonical group (e.g., "vitamin D", "statins", "probiotics")
  - **Layer 2**: Specific variant (e.g., "atorvastatin", "L. reuteri")
  - **Layer 3**: Dosage/details (e.g., "atorvastatin 20mg")
- **6 Relationship Types**: EXACT_MATCH, VARIANT, SUBTYPE, SAME_CATEGORY, DOSAGE_VARIANT, DIFFERENT
- **Technology**: nomic-embed-text embeddings (768-dim) + qwen3:14b classification
- **Output**: Cross-paper intervention unification (e.g., "vitamin D" = "Vitamin D3" = "cholecalciferol")

---

## Database Schema (19 Tables)

### Core Data Tables (2 tables)
1. **`papers`** - PubMed articles with metadata and fulltext
2. **`interventions`** - Extracted treatments and outcomes with mechanism data

### Phase 3.5 Hierarchical Normalization (3 tables)
3. **`semantic_hierarchy`** - 4-layer hierarchical structure
4. **`entity_relationships`** - Pairwise relationship types (6 types)
5. **`canonical_groups`** - Canonical entity groupings

### Data Mining Analytics (11 tables)
6. **`knowledge_graph_nodes`** - Nodes in medical knowledge graph
7. **`knowledge_graph_edges`** - Multi-edge graph relationships
8. **`bayesian_scores`** - Bayesian evidence scoring
9. **`treatment_recommendations`** - AI treatment recommendations
10. **`research_gaps`** - Under-researched areas
11. **`innovation_tracking`** - Emerging treatment tracking
12. **`biological_patterns`** - Mechanism and pattern discovery
13. **`condition_similarities`** - Condition similarity matrix
14. **`intervention_combinations`** - Synergistic combination analysis
15. **`failed_interventions`** - Catalog of ineffective treatments
16. **`data_mining_sessions`** - Session tracking

### Configuration & System (2 tables)
17. **`intervention_categories`** - 13-category taxonomy configuration
18. **`sqlite_sequence`** - SQLite internal auto-increment

### Legacy Tables (4 tables - DEPRECATED)
19. **`canonical_entities`, `entity_mappings`, `llm_normalization_cache`, `normalized_terms_cache`** - Replaced by Phase 3.5

---

## Key Commands

### Complete Workflow
```bash
# Collection → Processing → Categorization → Normalization
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10
python -m back_end.src.orchestration.batch_medical_rotation --resume --start-phase processing
```

### Individual Components
```bash
# Paper collection only (Phase 1)
python -m back_end.src.orchestration.rotation_paper_collector diabetes --count 100 --no-s2

# LLM processing only (Phase 2 - extracts WITHOUT categories)
python -m back_end.src.orchestration.rotation_llm_processor diabetes --max-papers 50

# Categorization only (Phase 2.5)
python -m back_end.src.orchestration.rotation_llm_categorization --interventions-only
python -m back_end.src.orchestration.rotation_llm_categorization --conditions-only
python -m back_end.src.orchestration.rotation_llm_categorization  # Both

# Hierarchical semantic normalization (Phase 3.5 - RECOMMENDED)
python -m back_end.src.orchestration.rotation_semantic_normalizer "type 2 diabetes"  # Single
python -m back_end.src.orchestration.rotation_semantic_normalizer --all  # All conditions
python -m back_end.src.orchestration.rotation_semantic_normalizer --status  # Check progress

# Data mining and analysis
python -m back_end.src.data_mining.data_mining_orchestrator --all
```

### Ground Truth Labeling
```bash
cd back_end/src/semantic_normalization/ground_truth

# Interactive labeling (with duplicate detection)
python label_in_batches.py --batch-size 20

# Validate accuracy
python evaluator.py
```

---

## Technology Stack

- **Python 3.13**: Core language
- **SQLite**: Database with connection pooling
- **Ollama**: Local LLM inference (qwen3:14b, nomic-embed-text)
- **PubMed API**: Primary paper source
- **PMC & Unpaywall**: Fulltext retrieval
- **Circuit Breaker Pattern**: Robust error handling
- **FAST_MODE**: Logging suppression for high-throughput (enabled by default)

---

## Current Status (October 8, 2025 - Post-Cleanup)

- **Papers**: 365 research papers (high quality, all with mechanism data)
- **Interventions**: 792 with **100% mechanism coverage** ✅
- **Interventions categorized**: 792/792 (100%) - 13 categories
- **Conditions**: 406 unique conditions
- **Conditions categorized**: 406/406 (100%) - 18 categories
- **Semantic normalization**: 676/676 interventions normalized (100%) ✅
- **Canonical groups**: 571 intervention groups created
- **Semantic relationships**: 141 cross-paper relationships tracked
- **Processing rate**: ~38-39 papers/hour (qwen3:14b with mechanism extraction)
- **Ground Truth Labeling**: 80/500 pairs labeled (16% complete)

---

## Project Conventions

### Code Style
- No emojis in print statements or code comments
- Comprehensive error handling with circuit breaker patterns
- Session persistence for all long-running operations
- Thermal protection for GPU-intensive operations

### File Organization
- **Source code**: `back_end/src/`
- **Execution scripts**: `back_end/src/orchestration/`
- **Data mining**: `back_end/src/data_mining/`
- **Configuration**: `back_end/src/data/config.py`
- **Session files**: `back_end/data/` (JSON state files)

### Performance Optimization
- **FAST_MODE**: Logging suppression for 1000+ papers/hour processing
  - **Enabled (default)**: `FAST_MODE=1` in `.env` - Only CRITICAL errors
  - **Disabled**: `FAST_MODE=0` - Full logging (ERROR/INFO/DEBUG)
  - **When to disable**: Active debugging or troubleshooting

---

## Frontend (Web Interface)

### File Structure
- **[index.html](frontend/index.html)** - Main webpage with DataTables integration
- **[script.js](frontend/script.js)** - Data loading, filtering, and display
- **[style.css](frontend/style.css)** - Custom styling and responsive design
- **[data/interventions.json](frontend/data/interventions.json)** - Exported data (generated by `export_frontend_data.py`)

### Key Features
- **Interactive DataTables**: Sortable, searchable, paginated intervention table
- **Summary Statistics**: Total interventions, conditions, papers, canonical groups, relationships
- **Correlation Strength Display**: Categorical labels (Very Strong ≥0.75, Strong ≥0.50, Weak ≥0.25, Very Weak <0.25)
- **Filtering**: By intervention category (13), condition category (18), correlation type, confidence threshold
- **Semantic Integration**: Displays canonical groups and 4-layer hierarchical classifications
- **Details Modal**: Full intervention data, mechanism of action, study details, paper information

### Data Export
```bash
python -m back_end.src.utils.export_frontend_data
```
Exports SQLite → JSON with Phase 3.5 hierarchical data, metadata, and top performers.

---

## Important Architecture Changes

### Single-Model Architecture (October 2025) ✅
**Previous**: Dual-model (gemma2:9b + qwen2.5:14b) with Phase 2 deduplication
**Current**: Single-model (qwen3:14b only) - no deduplication needed

**Benefits**:
- 2.6x speed improvement (no dual extraction overhead)
- Simpler error handling and debugging
- Preserves superior extraction detail
- Increased batch size from 5 to 8 papers

### Model Optimization: qwen3:14b (January 2025) ✅
**Challenge**: qwen3:14b generates chain-of-thought reasoning (8x slower)
**Solution**: System prompt suppression + `<think>` tag stripping
**Result**: qwen3:14b runs 1.3x faster than qwen2.5:14b (22.0s vs 28.9s per paper - baseline)

**Current Performance** (with mechanism extraction):
- qwen3:14b: ~93s per paper (~38-39 papers/hour)
- Trade-off: Richer mechanism data (biological/behavioral/psychological pathways)

### Categorization Architecture (October 2025) ✅
**Previous**: Categorization during extraction (in prompt)
**Current**: Separate Phase 2.5 after extraction

**Benefits**:
- Separation of concerns (extraction vs classification)
- Can re-categorize without re-extraction
- Faster extraction (fewer tokens in prompt)
- More accurate categorization with focused prompts

**Workflow**:
1. **Phase 2**: Extract interventions WITHOUT categories → `intervention_category = NULL`
2. **Phase 2.5**: LLM categorizes into 13 categories (batch of 20 items)
3. **Phase 3.5**: Hierarchical semantic normalization

### Phase 3.5: Hierarchical Semantic Normalization (October 2025) ✅

**Problem**: Cross-paper intervention name unification
**Solution**: 4-Layer hierarchical system with embedding-based similarity + LLM classification

**Performance** (Full Database Normalization - 1,028 interventions):
- **Runtime**: ~22 hours with qwen3:14b (~25s per uncached LLM call)
- **Canonical groups created**: 796 (Layer 1 hierarchy)
- **Relationships tracked**: 297 semantic connections
- **Embeddings cached**: 1,028 (nomic-embed-text 768-dim vectors)
- **Cache hit rate**: 40%+

**Result**: Unified analysis showing "150 papers support vitamin D" instead of fragmented counts

---

## Intervention Taxonomy (13 Categories)

| # | Category | Description | Examples |
|---|----------|-------------|----------|
| 1 | exercise | Physical exercise interventions | Aerobic exercise, resistance training, yoga, walking |
| 2 | diet | Dietary interventions | Mediterranean diet, ketogenic diet, intermittent fasting |
| 3 | supplement | Nutritional supplements | Vitamin D, probiotics, omega-3, herbal supplements |
| 4 | medication | Small molecule drugs | Statins, metformin, antidepressants, antibiotics |
| 5 | therapy | Psychological/physical/behavioral therapies | CBT, physical therapy, massage, acupuncture |
| 6 | lifestyle | Behavioral and lifestyle changes | Sleep hygiene, stress management, smoking cessation |
| 7 | surgery | Surgical procedures | Laparoscopic surgery, cardiac surgery, bariatric surgery |
| 8 | test | Medical tests and diagnostics | Blood tests, genetic testing, colonoscopy, imaging |
| 9 | device | Medical devices and implants | Pacemakers, insulin pumps, CPAP machines, CGMs |
| 10 | procedure | Non-surgical medical procedures | Endoscopy, dialysis, blood transfusion, radiation therapy |
| 11 | biologics | Biological drugs | Monoclonal antibodies, vaccines, immunotherapies, insulin |
| 12 | gene_therapy | Genetic and cellular interventions | CRISPR, CAR-T cell therapy, stem cell therapy |
| 13 | emerging | Novel interventions | Digital therapeutics, precision medicine, AI-guided |

**Key Distinctions**:
- medication vs biologics: Small molecules vs biological drugs
- surgery vs procedure: Surgical operations vs non-surgical procedures
- device vs medication: Hardware/implants vs pharmaceuticals

---

## Database Cleanup (October 8, 2025) ✅

**Issue**: 579 interventions (42.2%) extracted before mechanism field added (Oct 2-4, 2025)

**Actions Taken**:
1. Deleted 579 old interventions (without mechanisms)
2. Deleted 91 papers with only old interventions
3. Cleaned semantic hierarchy (352 entries, 156 relationships, 225 canonical groups)
4. Backup created: `intervention_research_backup_cleanup_20251008_173314.db`

**Results**:
- Interventions: 1,371 → **792** (-42.2%)
- Mechanism coverage: 57.8% → **100%** ✅
- Papers: 456 → **365** (-20.0%)
- All remaining interventions have complete mechanism data

**Documentation**: See [CLEANUP_SUMMARY_20251008.md](back_end/data/CLEANUP_SUMMARY_20251008.md)

---

## Data Mining Tools

Located in `back_end/src/data_mining/`:

### Shared Utility Modules
- **medical_knowledge.py**: Centralized medical domain knowledge (condition clusters, synergies, mechanisms)
- **similarity_utils.py**: Unified similarity calculations (cosine, Jaccard, Dice, mechanism similarity)
- **scoring_utils.py**: Scoring and statistical utilities (effectiveness, confidence, thresholds)

### Analytics Scripts
- **data_mining_orchestrator.py**: Coordinator for pattern discovery and analytics
- **medical_knowledge_graph.py**: Knowledge graph construction and analysis
- **bayesian_scorer.py**: Evidence-based intervention scoring (Beta distributions)
- **treatment_recommendation_engine.py**: AI treatment recommendation system
- **research_gaps.py**: Identification of under-researched areas
- **innovation_tracking_system.py**: Emerging treatment tracking
- **biological_patterns.py**: Mechanism and pattern discovery
- **condition_similarity_mapping.py**: Condition similarity matrix
- **power_combinations.py**: Synergistic combination analysis
- **failed_interventions.py**: Catalog of ineffective treatments

---

## Semantic Normalization Module

**Location**: `back_end/src/semantic_normalization/`

**Purpose**: Advanced intervention name normalization using hierarchical semantic grouping

### Core Components (8 files)
1. **embedding_engine.py** - Semantic embeddings (nomic-embed-text 768-dim)
2. **llm_classifier.py** - Canonical extraction & relationship classification (qwen3:14b)
3. **hierarchy_manager.py** - Database operations for hierarchical schema
4. **normalizer.py** - Full pipeline orchestration
5. **evaluator.py** - Ground truth accuracy testing
6. **test_runner.py** - Batch testing framework
7. **cluster_reviewer.py** - Interactive manual review
8. **experiment_logger.py** - Experiment documentation

### Ground Truth Tools (6 files in `ground_truth/`)
9. **labeling_interface.py** - Interactive labeling (undo, skip, review later)
10. **pair_generator.py** - Candidate pair generation (stratified sampling)
11. **label_in_batches.py** - Batch labeling session management
12. **generate_candidates.py** - 500-pair candidate generator
13. **data_exporter.py** - Export interventions from database
14. **prompts.py** - LLM prompts for classification

**Documentation**: See [back_end/src/semantic_normalization/README.md](back_end/src/semantic_normalization/README.md)

---

## Support & Troubleshooting

**For help**:
- Run status check: `python -m back_end.src.orchestration.batch_medical_rotation --status`
- Check semantic normalization: `python -m back_end.src.orchestration.rotation_semantic_normalizer --status`
- Review logs: `back_end/logs/*.log`
- Run tests: Check test files in `back_end/testing/`

**Common Issues**:
- GPU overheating: Check thermal status, wait for cooling
- LLM timeout: Increase timeout in config
- Database locked: Ensure no concurrent processes
- Missing dependencies: `conda activate venv`, check Ollama models

---

*Last Updated: October 8, 2025*
*Architecture: Single-model (qwen3:14b)*
*Status: Production Ready ✅*