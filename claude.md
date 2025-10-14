# MyBiome Health Research Pipeline

## Project Overview

Automated biomedical research pipeline that collects research papers about health conditions, extracts intervention-outcome relationships using local LLMs, and performs semantic normalization and data mining. Presents findings through an interactive web interface.

## Quick Start

**Environment**: Conda environment called 'venv'
```bash
conda activate venv

# Run single pipeline iteration (10 papers per condition across 60 conditions)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10

# Run continuous mode (infinite loop - restarts Phase 1 after Phase 3)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10 --continuous

# Run limited iterations (e.g., 5 complete cycles)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10 --continuous --max-iterations 5

# Custom delay between iterations (5 minutes = 300 seconds, default: 60s)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10 --continuous --iteration-delay 300

# Resume interrupted session
python -m back_end.src.orchestration.batch_medical_rotation --resume

# Resume in continuous mode
python -m back_end.src.orchestration.batch_medical_rotation --resume --continuous

# Check status (shows iteration history in continuous mode)
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

### Phase 2: LLM Processing (Hierarchical Extraction) ✅
- **Model**: qwen3:14b (optimized with chain-of-thought suppression)
- **Format**: Hierarchical extraction (study-level + intervention-level fields)
- **Extracts**:
  - **Study-level**: health_condition, study_focus (research questions), measured_metrics (measurement tools), findings (key results with data), study_location, publisher, sample_size, study_duration, study_type, population_details
  - **Intervention-level**: intervention_name, dosage, duration, frequency, intensity, mechanism (biological/behavioral pathway), correlation_type, correlation_strength, delivery_method, adverse_effects, extraction_confidence
- **Output**: Hierarchical JSON → Flattened to database (study fields duplicated per intervention)
- **Performance**: ~31 papers/hour (~115s per paper) - 25% slower than old format but provides richer study context
- **Database**: 5 new fields added (study_focus, measured_metrics, findings, study_location, publisher)

### Phase 2.5: Categorization (DEPRECATED - Now Phase 3.5)
- **Status**: Standalone script available but NOT integrated into main pipeline
- **Use Case**: Manual re-categorization outside pipeline
- **Replaced By**: Phase 3.5 group-based categorization

### Phase 3a: Semantic Normalization ✅
- **Scope**: Both interventions AND conditions
- **Technology**: nomic-embed-text embeddings (768-dim) + qwen3:14b classification
- **6 Relationship Types**: EXACT_MATCH, VARIANT, SUBTYPE, SAME_CATEGORY, DOSAGE_VARIANT, DIFFERENT
- **Output**:
  - **Interventions**: Cross-paper unification (e.g., "vitamin D" = "Vitamin D3" = "cholecalciferol")
  - **Conditions**: Hierarchical grouping (e.g., "IBS" → "IBS-C", "IBS-D", "IBS-M")
- **Performance**: Creates ~571 intervention groups + ~200-300 condition groups
- **Files**: `phase_3_normalizer.py`, `phase_3_embedding_engine.py`, `phase_3_llm_classifier.py`, `phase_3_hierarchy_manager.py`

### Phase 3b: Group-Based Categorization ✅
- **Scope**: Both interventions AND conditions
- **Efficiency**: Categorizes canonical groups instead of individual entities (~80% fewer LLM calls)
- **Intervention Categories**: 13 categories (medication, supplement, therapy, etc.)
- **Condition Categories**: 18 categories (cardiac, neurological, digestive, etc.)
- **Workflow**:
  - **PART A (Interventions)**:
    1. Categorize intervention canonical groups (571 groups in batches of 20)
    2. Propagate categories from groups to member interventions
    3. Fallback categorization for orphan interventions (no group membership)
  - **PART B (Conditions)**:
    4. Categorize condition canonical groups (batches of 20)
    5. Propagate categories from groups to conditions in interventions table
    6. Fallback categorization for orphan conditions (no group membership)
- **Performance**: 28% LLM call reduction vs individual categorization
- **Files**: `phase_3b_intervention_categorizer.py`, `phase_3b_condition_categorizer.py`

### Phase 3c: Mechanism Clustering ✅
- **Scope**: All intervention mechanisms (biological/behavioral/psychological pathways)
- **Technology**: HDBSCAN clustering + nomic-embed-text embeddings (768-dim)
- **100% Assignment Guarantee**: No mechanism left uncategorized
- **Strategy**: HDBSCAN for natural groupings + singleton clusters for unique mechanisms
- **Output**:
  - **mechanism_clusters**: Cluster metadata with canonical names
  - **mechanism_cluster_membership**: Mechanism-to-cluster assignments
  - **intervention_mechanisms**: Junction table linking interventions to mechanism clusters
  - **mechanism_condition_associations**: Analytics for which mechanisms work for which conditions
- **Performance**:
  - First run: ~25 minutes (embedding generation for 666 mechanisms)
  - Subsequent runs: <5 seconds (uses cached embeddings)
  - Incremental updates: Only embed new mechanisms
- **Results** (Current Database):
  - **666 mechanisms** processed → **415 clusters** created
  - **142 natural clusters** (HDBSCAN multi-member groups)
  - **273 singleton clusters** (unique mechanisms)
  - **Assignment rate: 100%** (all mechanisms categorized)
  - Average cluster size: 1.60 mechanisms per cluster
  - Silhouette score: 0.228 (acceptable for medical domain complexity)
- **Files**: `phase_3c_mechanism_pipeline_orchestrator.py`, `phase_3c_mechanism_hierarchical_clustering.py`

### Phase 3d: Hierarchical Cluster Merging (EXPERIMENTAL) 🧪
- **Location**: `back_end/src/semantic_normalization/phase_3d/`
- **Status**: Prototype implementation, not yet integrated into main pipeline
- **Scope**: All three entity types (interventions, conditions, mechanisms)
- **Purpose**: Build multi-level hierarchies by merging similar clusters
- **Technology**: HDBSCAN + nomic-embed-text embeddings (768-dim) + qwen3:14b validation
- **Stages**:
  1. **Initial Clustering** - HDBSCAN creates base clusters (conservative settings)
  2. **Similarity Calculation** - Embedding-based similarity between clusters
  3. **Merge Candidate Generation** - Top-k most similar cluster pairs per cluster
  4. **LLM Validation** - qwen3:14b validates semantic coherence of proposed merges
  5. **Hierarchical Merging** - Create parent clusters from validated merges (up to 4 levels)
  6. **Stage 3.5: Functional Grouping** - Detect cross-category merges, assign functional categories
- **Multi-Category Support** ✨:
  - Entities can belong to MULTIPLE categories simultaneously
  - **Category Types**: primary, functional, therapeutic, system, pathway, target, comorbidity
  - Example: "Probiotics" = supplement (PRIMARY) + gut flora modulator (FUNCTIONAL) + IBS treatment (THERAPEUTIC)
  - Cross-category merges: "Probiotics" (supplement) + "FMT" (procedure) → "Gut Flora Modulators" (functional group)
- **Output**:
  - Multi-level hierarchies (great-grandparent → grandparent → parent → child)
  - Junction tables for multi-category relationships
  - Functional category suggestions from LLM for cross-category groups
- **Expected Results**:
  - 40-50% cluster reduction (e.g., 571 intervention groups → ~280-340 parent clusters)
  - More generalizable treatment insights
  - Cross-category pattern discovery

---

## Database Schema (26 Tables)

### Core Data Tables (2 tables)
1. **`papers`** - PubMed articles with metadata and fulltext
2. **`interventions`** - Extracted treatments and outcomes with mechanism data

### Phase 3a & 3b Semantic Normalization & Categorization (3 tables)
3. **`semantic_hierarchy`** - Hierarchical structure linking interventions AND conditions to canonical groups
4. **`entity_relationships`** - Pairwise relationship types (6 types) for both entity types
5. **`canonical_groups`** - Canonical entity groupings with Layer 0 categories for both interventions and conditions

### Phase 3c Mechanism Clustering (4 tables)
6. **`mechanism_clusters`** - Mechanism cluster metadata with canonical names and hierarchy
7. **`mechanism_cluster_membership`** - Mechanism-to-cluster assignments (HDBSCAN or singleton)
8. **`intervention_mechanisms`** - Junction table linking interventions to mechanism clusters
9. **`mechanism_condition_associations`** - Analytics for which mechanisms work for which conditions

### Phase 3d Multi-Category Support (3 tables) 🧪
10. **`intervention_category_mapping`** - Many-to-many intervention-to-category relationships
11. **`condition_category_mapping`** - Many-to-many condition-to-category relationships
12. **`mechanism_category_mapping`** - Many-to-many mechanism-to-category relationships

### Data Mining Analytics (11 tables)
13. **`knowledge_graph_nodes`** - Nodes in medical knowledge graph
14. **`knowledge_graph_edges`** - Multi-edge graph relationships
15. **`bayesian_scores`** - Bayesian evidence scoring
16. **`treatment_recommendations`** - AI treatment recommendations
17. **`research_gaps`** - Under-researched areas
18. **`innovation_tracking`** - Emerging treatment tracking
19. **`biological_patterns`** - Mechanism and pattern discovery
20. **`condition_similarities`** - Condition similarity matrix
21. **`intervention_combinations`** - Synergistic combination analysis
22. **`failed_interventions`** - Catalog of ineffective treatments
23. **`data_mining_sessions`** - Session tracking

### Configuration & System (2 tables)
24. **`intervention_categories`** - 13-category taxonomy configuration
25. **`sqlite_sequence`** - SQLite internal auto-increment

### Legacy Tables (4 tables - DEPRECATED)
26. **`canonical_entities`, `entity_mappings`, `llm_normalization_cache`, `normalized_terms_cache`** - Replaced by Phase 3a semantic normalization

---

## Codebase Architecture

### Folder Structure
```
back_end/src/
├── phase_1_data_collection/          # Phase 1: Paper Collection
│   ├── phase_1_pubmed_collector.py
│   ├── phase_1_fulltext_retriever.py
│   ├── phase_1_paper_parser.py
│   ├── phase_1_semantic_scholar_enrichment.py
│   ├── database_manager.py          (generic - used by all phases)
│   └── data_mining_repository.py    (generic utility)
│
├── phase_2_llm_processing/           # Phase 2: LLM Extraction
│   ├── phase_2_single_model_analyzer.py
│   ├── phase_2_batch_entity_processor.py
│   ├── phase_2_entity_operations.py
│   ├── phase_2_entity_utils.py
│   ├── phase_2_prompt_service.py
│   └── phase_2_export_to_json.py
│
├── phase_3_semantic_normalization/   # Phase 3: Normalization & Categorization
│   ├── phase_3_normalizer.py                    # 3a: Core normalization orchestrator
│   ├── phase_3_embedding_engine.py              # 3a: Semantic embeddings
│   ├── phase_3_llm_classifier.py                # 3a: Relationship classification
│   ├── phase_3_hierarchy_manager.py             # 3a: Database operations
│   ├── phase_3b_intervention_categorizer.py     # 3b: Intervention categorization
│   ├── phase_3b_condition_categorizer.py        # 3b: Condition categorization
│   ├── phase_3c_mechanism_pipeline_orchestrator.py  # 3c: Mechanism clustering
│   ├── phase_3c_mechanism_hierarchical_clustering.py  # 3c: Hierarchical clusters
│   ├── mechanism_*.py                           # 3c: Supporting mechanism files
│   ├── phase_3d/                                # 3d: Experimental hierarchical merging
│   └── ground_truth/                            # Ground truth labeling workflow
│
├── orchestration/                    # Pipeline Orchestrators
│   ├── phase_1_paper_collector.py
│   ├── phase_2_llm_processor.py
│   ├── phase_3_semantic_normalizer.py
│   ├── phase_3b_group_categorization.py
│   ├── phase_3c_mechanism_clustering.py
│   └── batch_medical_rotation.py    (main pipeline controller)
│
├── data_mining/                      # Analytics & Insights
│   ├── data_mining_orchestrator.py
│   ├── medical_knowledge_graph.py
│   ├── bayesian_scorer.py
│   └── ...
│
├── utils/                            # General Utilities
├── migrations/                       # Database Migrations
└── data/                            # Configuration & Repositories
```

### Pipeline Flow
```
Phase 1 → Phase 2 → Phase 3a → Phase 3b → Phase 3c → [Phase 3d]
   ↓         ↓          ↓          ↓          ↓          ↓
Papers   Interventions Groups  Categories Mechanisms Hierarchies
                                                      (experimental)
```

### File Naming Convention
- **Phase-specific files**: `phase_X_descriptive_name.py` (e.g., `phase_2_single_model_analyzer.py`)
- **Sub-phase files**: `phase_Xa_name.py` (e.g., `phase_3b_intervention_categorizer.py`)
- **Generic utilities**: Keep descriptive names without phase prefix (e.g., `database_manager.py`)
- **Orchestrators**: Located in `orchestration/`, named by phase

---

## Key Commands

### Complete Workflow
```bash
# Single iteration: Collection → Processing → Semantic Normalization → Group Categorization → Mechanism Clustering
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10

# Continuous mode: Infinite loop (restarts Phase 1 after Phase 3c)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10 --continuous

# Limited iterations (e.g., 5 complete cycles)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10 --continuous --max-iterations 5

# Resume from specific phase
python -m back_end.src.orchestration.batch_medical_rotation --resume --start-phase processing
```

### Individual Components
```bash
# Paper collection only (Phase 1)
python -m back_end.src.orchestration.rotation_paper_collector diabetes --count 100 --no-s2

# LLM processing only (Phase 2 - extracts WITHOUT categories)
python -m back_end.src.orchestration.rotation_llm_processor diabetes --max-papers 50

# Semantic normalization (Phase 3)
python -m back_end.src.orchestration.rotation_semantic_normalizer "type 2 diabetes"  # Normalize interventions for single condition
python -m back_end.src.orchestration.rotation_semantic_normalizer --all  # Normalize interventions for all conditions
python -m back_end.src.orchestration.rotation_semantic_normalizer --normalize-conditions  # Normalize condition entities (e.g., "IBS" → "IBS-C", "IBS-D")
python -m back_end.src.orchestration.rotation_semantic_normalizer --status  # Check progress

# Group-based categorization (Phase 3.5)
python -m back_end.src.orchestration.rotation_group_categorization  # Categorize both interventions and conditions

# Manual categorization (standalone - NOT in pipeline)
python -m back_end.src.orchestration.rotation_llm_categorization --interventions-only
python -m back_end.src.orchestration.rotation_llm_categorization --conditions-only

# Data mining and analysis
python -m back_end.src.data_mining.data_mining_orchestrator --all
```

### Ground Truth Labeling
```bash
cd back_end/src/semantic_normalization/ground_truth

# Generate candidate pairs (500 pairs with stratified sampling)
python ground_truth_cli.py generate

# Interactive batch labeling (with auto-save, undo, skip)
python ground_truth_cli.py label --batch-size 50

# Check labeling progress
python ground_truth_cli.py status

# Remove duplicate labels
python ground_truth_cli.py clean

# Validate accuracy (parent folder)
cd ..
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
- **Retry Logic**: Automatic retry with exponential backoff for LLM failures
- **Continuous Mode**: Infinite loop execution for unattended multi-iteration data collection
- **FAST_MODE**: Logging suppression for high-throughput (enabled by default)

---

## Current Status (October 13, 2025)

- **Papers**: 533 research papers (high quality, all with mechanism data)
- **Interventions**: 777 with **100% mechanism coverage** ✅
- **Intervention Groups**: 571 canonical groups created ✅
- **Intervention Categorization**: 777/777 (100%) via group propagation ✅
- **Conditions**: 406 unique conditions
- **Condition Groups**: ~200-300 canonical groups (semantic normalization enabled) ✅
- **Condition Categorization**: 406/406 (100%) via group propagation - 18 categories ✅
- **Mechanism Clusters**: **666 mechanisms** → **415 clusters** (100% assignment) ✅
  - 142 natural clusters (HDBSCAN)
  - 273 singleton clusters (unique mechanisms)
- **Semantic Relationships**: 141+ cross-paper relationships tracked (both entity types)
- **Processing Rate**: ~38-39 papers/hour (qwen3:14b with mechanism extraction)
- **Architecture**: Phase 3a + 3b + 3c fully integrated ✅
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
- **Filtering**: By intervention category (13), condition category (18), functional category, therapeutic category, correlation type, confidence threshold
- **Multi-Category Display** 🧪: Color-coded badges for multiple category types (primary, functional, therapeutic, etc.)
- **Semantic Integration**: Displays canonical groups and 4-layer hierarchical classifications
- **Details Modal**: Full intervention data, mechanism of action, study details, paper information, all category types

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
**Current**: Group-based categorization in Phase 3.5 (after semantic normalization)

**Benefits**:
- Separation of concerns (extraction → normalization → categorization)
- Can re-categorize without re-extraction
- Faster extraction (fewer tokens in prompt)
- More accurate categorization with semantic context from groups
- Automatic retry logic prevents pipeline stoppage
- Enhanced prompts with edge case handling

**Current Pipeline Flow**:
1. **Phase 2**: Extract interventions WITHOUT categories → `intervention_category = NULL`, `condition_category = NULL`
2. **Phase 3**: Semantic normalization (create canonical groups for interventions AND conditions)
3. **Phase 3.5**: Group-based categorization
   - Categorize intervention groups (13 categories, batch of 20)
   - Propagate to member interventions
   - Categorize condition groups (18 categories, batch of 20)
   - Propagate to conditions in interventions table
   - Fallback categorization for orphans

### Phase 3: Semantic Normalization (October 2025) ✅

**Problem**: Cross-paper entity name unification (interventions AND conditions)
**Solution**: Embedding-based similarity + LLM classification to create canonical groups

**Scope**: Both interventions and conditions processed
- **Interventions**: Unify variant names (e.g., "vitamin D" = "Vitamin D3" = "cholecalciferol")
- **Conditions**: Hierarchical grouping (e.g., "IBS" parent with "IBS-C", "IBS-D", "IBS-M" subtypes)

**Performance** (Current Database):
- **Interventions**: 777 entities → 571 canonical groups
- **Conditions**: 406 entities → ~200-300 canonical groups
- **Total Relationships**: 141+ semantic connections
- **Embeddings**: nomic-embed-text 768-dim vectors for all entities
- **Runtime**: ~25s per uncached LLM call
- **Cache hit rate**: 40%+

**Result**:
- Unified cross-paper analysis (e.g., "150 papers support vitamin D" instead of fragmented counts)
- Condition variant tracking (e.g., "IBS" studies include "IBS-C", "IBS-D", "IBS-M" subtypes)

### Phase 3.5: Group-Based Categorization (October 2025) ✅

**Problem**: Categorizing individual entities is inefficient (792 LLM calls for interventions alone)
**Solution**: Categorize canonical groups then propagate to member entities (both interventions AND conditions)

**Benefits**:
- 28% reduction in LLM calls vs individual categorization
- Consistent categorization across variant names
- Can re-categorize all entities by updating group categories
- Single phase handles both entity types

**Workflow (6 Steps)**:
- **PART A (Interventions)**:
  1. Categorize intervention canonical groups (571 groups, 13 categories, batches of 20)
  2. Propagate categories from groups to member interventions
  3. Fallback categorization for orphan interventions (no group membership)
- **PART B (Conditions)**:
  4. Categorize condition canonical groups (~200-300 groups, 18 categories, batches of 20)
  5. Propagate categories from groups to conditions in interventions table
  6. Fallback categorization for orphan conditions (no group membership)

**Performance**:
- **Interventions**: 571 LLM calls (vs 792 individual) = 28% reduction
- **Conditions**: ~10-15 LLM calls (vs ~20 individual) = ~40% reduction
- **Total**: ~580-590 LLM calls for complete categorization of both entity types

**Integration**: Fully integrated into main pipeline as Phase 3.5 (runs after Phase 3 semantic normalization)

### Continuous Mode: Infinite Loop Execution (October 2025) ✅

**Problem**: Manual pipeline re-runs required for large-scale data collection
**Solution**: Continuous mode that automatically restarts Phase 1 after Phase 3 completion

**Features**:
- **Infinite loop**: Automatically restarts collection after semantic normalization completes
- **Iteration tracking**: Full history of all completed iterations with timestamps and statistics
- **Thermal protection**: Configurable delay between iterations (default: 60s) prevents GPU overheating
- **Graceful shutdown**: Ctrl+C stops cleanly between iterations (won't interrupt mid-phase)
- **Iteration limits**: Optional max iterations (unlimited by default)
- **Resumable**: Can resume continuous mode from saved session
- **Backward compatible**: Default behavior unchanged (single iteration)

**Usage**:
```bash
# Infinite loop (runs until Ctrl+C)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10 --continuous

# Limited iterations (e.g., 5 complete cycles)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10 --continuous --max-iterations 5

# Custom delay (5 minutes between iterations)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10 --continuous --iteration-delay 300
```

**Iteration Flow**:
1. Phase 1 → Phase 2 → Phase 3 → Phase 3.5 → Phase 3.6 (complete)
2. Save iteration statistics to history
3. Reset phase flags and counters
4. Wait `iteration_delay_seconds` (thermal protection)
5. Increment iteration counter
6. **Loop back to Phase 1** ↻
7. Continue until max iterations reached or Ctrl+C

**Benefits**:
- Unattended data collection over days/weeks
- Automatic expansion of research database
- No manual intervention required
- Full audit trail via iteration history
- GPU thermal protection between cycles

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
| 10 | procedure | Non-surgical medical procedures | Endoscopy, dialysis, **blood transfusion**, **fecal microbiota transplant**, radiation therapy, PRP injections |
| 11 | biologics | Biological drugs | Monoclonal antibodies, vaccines, immunotherapies, insulin |
| 12 | gene_therapy | Genetic and cellular interventions | CRISPR, CAR-T cell therapy, stem cell therapy |
| 13 | emerging | Novel interventions | Digital therapeutics, precision medicine, AI-guided |

**Key Distinctions**:
- **medication vs biologics**: Small molecules vs biological drugs
- **surgery vs procedure**: Surgical operations vs non-surgical procedures
- **device vs medication**: Hardware/implants vs pharmaceuticals
- **Edge Cases** (fixed with enhanced prompts):
  - Blood transfusion → **procedure** (NOT medication/biologics)
  - Fecal microbiota transplant → **procedure** (NOT supplement)
  - Probiotics pills → **supplement**; fecal transplant → **procedure**
  - Insulin, vaccines → **biologics** (NOT medication)

---

## Condition Taxonomy (18 Categories)

| # | Category | Description | Examples |
|---|----------|-------------|----------|
| 1 | cardiac | Heart and blood vessel conditions | Coronary artery disease, heart failure, hypertension, arrhythmias, MI |
| 2 | neurological | Brain, spinal cord, nervous system | Stroke, Alzheimer's, Parkinson's, epilepsy, MS, dementia, neuropathy |
| 3 | digestive | Gastrointestinal system | GERD, IBD, IBS, cirrhosis, Crohn's, ulcerative colitis, H. pylori |
| 4 | pulmonary | Lungs and respiratory system | COPD, asthma, pneumonia, pulmonary embolism, respiratory failure |
| 5 | endocrine | Hormones and metabolism | Diabetes, thyroid disorders, obesity, PCOS, metabolic syndrome |
| 6 | renal | Kidneys and urinary system | Chronic kidney disease, acute kidney injury, kidney stones, glomerulonephritis |
| 7 | oncological | All cancers and malignant neoplasms | Lung cancer, breast cancer, colorectal cancer, leukemia |
| 8 | rheumatological | Autoimmune and rheumatic diseases | Rheumatoid arthritis, lupus, gout, vasculitis, fibromyalgia |
| 9 | psychiatric | Mental health conditions | Depression, anxiety, bipolar disorder, schizophrenia, ADHD, PTSD |
| 10 | musculoskeletal | Bones, muscles, tendons, ligaments | Fractures, osteoarthritis, back pain, ACL injury, tendinitis |
| 11 | dermatological | Skin, hair, nails | Acne, psoriasis, eczema, atopic dermatitis, melanoma, rosacea |
| 12 | infectious | Bacterial, viral, fungal infections | HIV, tuberculosis, hepatitis, sepsis, COVID-19, influenza |
| 13 | immunological | Allergies and immune disorders | Food allergies, allergic rhinitis, immunodeficiency, anaphylaxis |
| 14 | hematological | Blood cells and clotting | Anemia, thrombocytopenia, hemophilia, sickle cell disease, thrombosis |
| 15 | nutritional | Nutrient deficiencies | Vitamin D deficiency, B12 deficiency, iron deficiency, malnutrition |
| 16 | toxicological | Poisoning and drug toxicity | Drug toxicity, heavy metal poisoning, overdose, carbon monoxide poisoning |
| 17 | parasitic | Parasitic infections | Malaria, toxoplasmosis, giardiasis, schistosomiasis, helminth infections |
| 18 | other | Conditions that don't fit standard categories | Rare diseases, multisystem conditions, unclassified syndromes |

**Key Distinctions**:
- **Type 2 diabetes** → **endocrine** (metabolic/hormonal, NOT digestive)
- **H. pylori infection** → **infectious** (infection itself, NOT digestive complications)
- **Osteoarthritis** → **rheumatological** (autoimmune context) or **musculoskeletal** (mechanical wear)
- **Diabetic foot ulcer** → **infectious** or **dermatological** (depending on context, NOT endocrine)
- **IBS variants** (IBS-C, IBS-D, IBS-M) → All **digestive** with hierarchical grouping

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

**Purpose**: Advanced entity name normalization using hierarchical semantic grouping (interventions AND conditions)

### Core Components (8 files, 2,996 lines)

**Phase 3: Semantic Normalization** (5 files, 1,618 lines)
1. **embedding_engine.py** (214 lines) - Semantic embeddings with caching
   - nomic-embed-text model (768-dimensional vectors)
   - Persistent embedding cache to avoid recomputation
   - Batch embedding support for efficiency
   - Cosine similarity calculations for finding related entities

2. **llm_classifier.py** (267 lines) - LLM-based relationship classification
   - Uses qwen3:14b for canonical extraction and relationship typing
   - 6 relationship types: EXACT_MATCH, VARIANT, SUBTYPE, SAME_CATEGORY, DOSAGE_VARIANT, DIFFERENT
   - Extracts canonical forms (e.g., "vitamin D" from "Vitamin D3 1000IU")
   - Chain-of-thought reasoning for accurate classification

3. **hierarchy_manager.py** (307 lines) - Database operations
   - Manages 3-table schema: `semantic_hierarchy`, `entity_relationships`, `canonical_groups`
   - CRUD operations for entities, relationships, and canonical groups
   - Query methods for fetching hierarchies and group members
   - Transaction support for atomic updates

4. **normalizer.py** (488 lines) - Pipeline orchestrator
   - MainNormalizer class coordinates full normalization workflow
   - Workflow: Load entities → Generate embeddings → Extract canonicals → Find similar → Classify relationships → Populate DB
   - Supports both intervention and condition normalization
   - Progress tracking and session persistence
   - Configurable similarity thresholds and batch sizes

5. **evaluator.py** (342 lines) - Accuracy validation
   - Tests automated system against ground truth labeled data
   - Generates 6×6 confusion matrix for relationship types
   - Per-type accuracy metrics and overall system accuracy
   - Error pattern identification (e.g., VARIANT misclassified as EXACT_MATCH)

**Phase 3.5: Group-Based Categorization** (3 files, 1,378 lines)
6. **group_categorizer.py** (561 lines) - Intervention group categorizer
   - Categorizes canonical intervention groups (13 categories)
   - Includes group members in prompt for semantic context
   - Batch processing (20 groups per LLM call)
   - Propagation to member interventions via UPDATE-JOIN
   - Orphan intervention handling (fallback categorization)

7. **condition_group_categorizer.py** (493 lines) - Condition group categorizer
   - Categorizes canonical condition groups (18 categories)
   - Includes group members in prompt for semantic context
   - Batch processing (20 groups per LLM call)
   - Propagation to conditions in interventions table
   - Orphan condition handling (fallback categorization)

8. **validation.py** (324 lines) - Categorization validation
   - Coverage validation (% of entities categorized)
   - Purity validation (consistency within groups)
   - Comparison with existing categorizations
   - Overall validation scoring with pass/fail thresholds

### Ground Truth Labeling Workflow (5 files, 1,776 lines)

Complete **human-in-the-loop** workflow for creating labeled training data

1. **Export** data from DB (**`data_exporter.py`** - 160 lines)
    - Exports unique intervention names with metadata (frequency, category, health conditions)
    - Query: Groups by intervention name, filters by min frequency, limits to top 500
    - Output: JSON file with intervention list + metadata for candidate generation

2. **Unified CLI** for workflow management (**`ground_truth_cli.py`** - 567 lines)
    - Single entry point with 4 subcommands: `generate`, `label`, `status`, `clean`
    - **Subcommand: generate** - Creates 500 candidate pairs using stratified sampling
        - Calls `pair_generator.py` (442 lines) with fuzzy matching (rapidfuzz/fuzzywuzzy)
        - Stratified sampling: 60% similarity-based + 20% random + 20% targeted same-category
        - Output: `hierarchical_candidates_500_pairs.json` (168 KB)
    - **Subcommand: label** - Interactive batch labeling with resume capability
        - Uses `labeling_interface.py` (559 lines) terminal UI with 6 relationship types
        - Manages batches (default: 50 pairs), auto-save every 5 labels, undo/skip features
        - Progress grid, time estimation, performance tracking (labels per minute)
        - Example: Batch 1 (pairs 1-50) → Batch 2 (pairs 51-100) → ... → Batch 10 (pairs 451-500)
    - **Subcommand: status** - Displays completion percentage, batch grid, time remaining
    - **Subcommand: clean** - Removes duplicate labels, creates backup, updates counters

3. **Pair generation library** (**`pair_generator.py`** - 442 lines)
    - SmartPairGenerator class with fuzzy matching algorithms
    - Stratified sampling strategy:
        - 60% similarity-based (ranges: 0.85-0.95, 0.75-0.85, 0.65-0.75)
        - 20% random low-similarity (0.40-0.65) for DIFFERENT examples
        - 20% targeted same-category (probiotics vs probiotics, statins vs statins)
    - Category-aware pair selection and deduplication logic

4. **Interactive labeling UI** (**`labeling_interface.py`** - 559 lines)
    - HierarchicalLabelingInterface class - reusable terminal interface
    - Features:
        - Undo history (last 10 labels)
        - Skip pairs (review later)
        - Performance tracking (labels per minute)
        - Session auto-save every 5 labels
        - Displays similarity score and metadata
    - Relationship types: EXACT_MATCH (1), VARIANT (2), SUBTYPE (3), SAME_CATEGORY (4), DOSAGE_VARIANT (5), DIFFERENT (6)
    - Output: Session JSON with labeled pairs + progress tracking

5. **Module documentation** (**`__init__.py`** - 48 lines)
    - Package-level documentation and imports
    - Usage examples for CLI workflow
    - Feature summary and component descriptions

**Current Progress**: 80/500 pairs labeled (16% complete)

**Usage**:
```bash
cd back_end/src/semantic_normalization/ground_truth
python ground_truth_cli.py generate              # Step 1: Generate candidates
python ground_truth_cli.py label --batch-size 50 # Step 2: Label in batches
python ground_truth_cli.py status                # Step 3: Check progress
python ground_truth_cli.py clean                 # Step 4: Remove duplicates
```

**Documentation**: See [back_end/src/semantic_normalization/README.md](back_end/src/semantic_normalization/README.md)

---

## Phase 3d: Hierarchical Cluster Merging (EXPERIMENTAL) 🧪

**Location**: `back_end/src/semantic_normalization/phase_3d/`
**Status**: Prototype implementation, not yet integrated into main pipeline

### Overview

Phase 3d builds multi-level hierarchies by merging similar clusters created in earlier phases. The goal is to discover higher-level patterns and enable cross-category functional grouping.

### Architecture

**6-Stage Pipeline**:
1. **Stage 1**: Initial HDBSCAN clustering (conservative settings)
2. **Stage 2**: Embedding-based similarity calculation
3. **Stage 3**: Merge candidate generation (top-k similar pairs)
4. **Stage 4**: LLM validation (qwen3:14b semantic coherence check)
5. **Stage 5**: Hierarchical parent cluster creation (up to 4 levels)
6. **Stage 3.5**: Functional grouping for cross-category merges

**Key Components** (7 files):
- **config.py** (156 lines) - Configuration with hyperparameter presets
- **models.py** (225 lines) - Data models (Cluster, MergeCandidate, HierarchyLevel)
- **stage_1_clustering.py** (294 lines) - HDBSCAN initial clustering
- **stage_2_similarity.py** (248 lines) - Centroid-based similarity calculation
- **stage_3_candidate_generation.py** (339 lines) - Top-k merge candidate selection
- **stage_4_llm_validation.py** (412 lines) - Semantic coherence validation
- **stage_5_hierarchical_merging.py** (468 lines) - Multi-level hierarchy construction
- **stage_3_5_functional_grouping.py** (561 lines) - Cross-category detection & functional categorization

### Multi-Category Support ✨

**Problem**: Single categories are too restrictive. Example: "Probiotics" (supplement) + "FMT" (procedure) both modulate gut flora, but traditional categorization can't capture this shared function.

**Solution**: Multi-category membership with junction tables

**Category Types**:
- **primary**: What the entity IS (e.g., supplement, medication, procedure)
- **functional**: What the entity DOES (e.g., gut flora modulator, anti-inflammatory)
- **therapeutic**: What condition it treats (e.g., IBS treatment, cardiovascular therapy)
- **system**: Body system (conditions only - e.g., digestive, cardiac)
- **pathway**: Biological pathway (mechanisms only - e.g., AMPK activation)
- **target**: Molecular target (mechanisms only - e.g., PPAR-gamma)
- **comorbidity**: Co-occurring conditions (conditions only - e.g., metabolic syndrome cluster)

**Database Schema** (3 junction tables):
- `intervention_category_mapping` - Intervention-to-category assignments
- `condition_category_mapping` - Condition-to-category assignments
- `mechanism_category_mapping` - Mechanism-to-category assignments

Each table includes:
- `entity_id` - Foreign key to entity
- `category_type` - Type of category (primary, functional, therapeutic, etc.)
- `category_name` - Category name
- `confidence` - Assignment confidence (0.0-1.0)
- `assigned_by` - Source (system, llm, group_propagation, user)
- `notes` - Optional explanation

**API Methods** ([database_manager.py:1268-1475](back_end/src/data_collection/database_manager.py#L1268-L1475)):
- `assign_category()` - Assign category to entity
- `get_entity_categories()` - Get all categories for entity
- `get_entities_by_category()` - Query entities by category
- `get_primary_category()` - Backward compatibility helper

**Dual-Write Pattern**: Updates both legacy columns (`intervention_category`, `condition_category`) AND junction tables for backward compatibility

**Example Use Case**:
```python
# Probiotics gets multiple categories
db.assign_category('intervention', probiotic_id, 'supplement', 'primary')
db.assign_category('intervention', probiotic_id, 'gut_flora_modulator', 'functional')
db.assign_category('intervention', probiotic_id, 'ibs_treatment', 'therapeutic')

# Query all gut flora modulators (includes both probiotics AND FMT)
entities = db.get_entities_by_category('gut_flora_modulator', 'intervention', 'functional')
```

### Frontend Integration

**Updated Components**:
- **HTML**: Functional/therapeutic category filter dropdowns + multi-category stat card
- **CSS**: Color-coded badges (primary=blue, functional=green, therapeutic=yellow, etc.)
- **JavaScript**: Multi-category display, filtering, and details modal

**Color Scheme**:
- Primary (blue): `#007bff`
- Functional (green): `#28a745`
- Therapeutic (yellow): `#ffc107`
- System (teal): `#17a2b8`
- Pathway (purple): `#6f42c1`
- Target (pink): `#e83e8c`
- Comorbidity (orange): `#fd7e14`

**Backward Compatibility**: All code handles both legacy single-category (string) and new multi-category (object) formats

### Usage

```bash
# Run multi-category migration (create junction tables)
python -m back_end.src.migrations.add_multi_category_support

# Run Stage 3.5 functional grouping (detect cross-category merges)
python -m back_end.src.semantic_normalization.phase_3d.stage_3_5_functional_grouping

# Export data with multi-category support
python -m back_end.src.utils.export_frontend_data
```

### Expected Results

- **Cluster Reduction**: 40-50% (e.g., 571 intervention groups → ~280-340 parent clusters)
- **Cross-Category Insights**: Discover functional groupings across traditional boundaries
- **More Generalizable**: Higher-level abstractions enable broader treatment insights

### Status

- **Backend**: Fully implemented (database, API, categorization scripts, data export)
- **Frontend**: Fully implemented (HTML, CSS, JavaScript with multi-category display)
- **Integration**: NOT yet integrated into main pipeline (experimental prototype)
- **Testing**: Unit tests passing, integration tests pending

---

## Support & Troubleshooting

**For help**:
- Run status check: `python -m back_end.src.orchestration.batch_medical_rotation --status`
- Check semantic normalization: `python -m back_end.src.orchestration.rotation_semantic_normalizer --status`
- Review logs: `back_end/logs/*.log`
- Run tests: Check test files in `back_end/testing/`

**Common Issues**:
- GPU overheating: Check thermal status, wait for cooling (or use `--iteration-delay` in continuous mode)
- LLM timeout: Increase timeout in config or use automatic retry logic (`--max-retries`)
- LLM categorization failures: Retry logic now handles transient failures automatically
- Database locked: Ensure no concurrent processes
- Missing dependencies: `conda activate venv`, check Ollama models
- Pipeline stops during Phase 2.5: Automatic retry logic (3 attempts) prevents stoppage

---

*Last Updated: October 14, 2025*
*Architecture: Single-model (qwen3:14b) with Phase 1 (collection) → Phase 2 (extraction) → Phase 3a (normalization) → Phase 3b (categorization) → Phase 3c (mechanism clustering) → Phase 3d (experimental hierarchical merging)*
*Status: Production Ready ✅ (Phase 3d: Experimental 🧪)*
*Naming Convention: Phase-prefixed files for clear pipeline organization*