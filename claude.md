# MyBiome Health Research Pipeline

## Project Overview

Automated biomedical research pipeline that collects research papers about health conditions using PubMed API (Phase 1), then extracts condition-intervention-outcome-mechanism relationships using local LLMs (Phase 2). After that the pipeline performs semantic embedding of the conditions, interventions and mechanisms extracted (Phase 3a), followed by clustering them (Phase 3b), cluster naming using local LLMs (Phase 3c) and then merges similar clusters into parent-child hierarchies (Phase 3d). Finally, Phase 4 builds a knowledge graph from canonical groups (Phase 4a) and generates Bayesian evidence scores with pooled evidence for better statistical power (Phase 4b). The findings are presented through an interactive web interface.

## Quick Start

**Environment**: Conda environment called 'venv'
```bash
conda activate venv
```
---

## Architecture

**Backend**: Python 3.13 research automation pipeline
**Frontend**: HTML/CSS/JavaScript web interface with DataTables.js
**Database**: SQLite with 25 tables
**LLM**: Local qwen3:14b via Ollama
**Embeddings**: mxbai-embed-large (1024-dim) via Ollama

---

## Core Pipeline Phases

### Phase 1: Data Collection
- **PubMed API**: Primary source for paper collection
- **PMC & Unpaywall**: Fulltext retrieval
- **Output**: Papers stored in `papers` table with `processing_status = 'pending'`

### Phase 2: LLM Processing
- **Model**: qwen3:14b (optimized with chain-of-thought suppression)
- **Format**: Hierarchical extraction (study-level + intervention-level fields)
- **Extracts**:
  - **Study-level**: health_condition, study_focus (research questions), measured_metrics (measurement tools), findings (key results with data), study_location, publisher, sample_size, study_duration, study_type, population_details, study_focus, measured_metrics, findings, study_location, publisher
  - **Intervention-level**: intervention_name, dosage, duration, frequency, intensity, mechanism (biological/behavioral pathway), correlation_type, correlation_strength, delivery_method, adverse_effects, extraction_confidence
- **Output**: Hierarchical JSON → Flattened to database (study fields duplicated per intervention)

### Phase 3a: Semantic Embedding ✅
- **Scope**: All three entity types (interventions, conditions, mechanisms)
- **Technology**: mxbai-embed-large (1024-dim) via Ollama
- **Process**: Generate embeddings for all entities → Cache for reuse
- **Performance**: 100% cache hit rate after first run, <1s to load
- **Output**: 716 intervention, ~400 condition, 666 mechanism embeddings
- **Files**: `phase_3a_base_embedder.py`, `phase_3a_intervention_embedder.py`, `phase_3a_condition_embedder.py`, `phase_3a_mechanism_embedder.py`

### Phase 3b: Hierarchical Clustering ✅
- **Scope**: All three entity types (interventions, conditions, mechanisms)
- **Technology**: Hierarchical clustering with distance_threshold=0.7 (optimal from hyperparameter experiments)
- **100% Assignment Guarantee**: Singleton handler ensures no entity unclustered
- **Performance**: Interventions: 716 entities → 538 clusters (0.2s), uses cached results after first run
- **Output**: Natural clusters (multi-member) + Singleton clusters (unique entities)
- **Files**: `phase_3b_base_clusterer.py`, `phase_3b_hierarchical_clusterer.py`, `phase_3b_hdbscan_clusterer.py`, `phase_3b_singleton_handler.py`

### Phase 3c: LLM Canonical Naming ✅
- **Scope**: All three entity types (interventions, conditions, mechanisms)
- **Technology**: qwen3:14b (temperature=0.0 for deterministic results)
- **Process**: Name clusters with LLM → Assign categories → Cache results
- **Performance**: Batch processing (20 clusters per call), ~70 minutes for 538 intervention clusters (uncached)
- **Output**: Canonical names + category assignments (13 intervention, 18 condition categories)
- **Files**: `phase_3c_base_namer.py`, `phase_3c_llm_namer.py` 

### Phase 3d: Hierarchical Cluster Merging 
- **Location**: `back_end/src/semantic_normalization/phase_3d/`
- **Purpose**: Build multi-level hierarchies by merging similar clusters
- **Technology**: HDBSCAN + qwen3:14b validation
- **Stages**:
  1. **Initial Clustering** - HDBSCAN creates base clusters (conservative settings)
  2. **Similarity Calculation** - Embedding-based similarity between clusters
  3. **Merge Candidate Generation** - Top-k most similar cluster pairs per cluster
  4. **LLM Validation** - qwen3:14b validates semantic coherence of proposed merges
  5. **Hierarchical Merging** - Create parent clusters from validated merges (up to 4 levels)
  6. **Stage 3.5: Functional Grouping** - Detect cross-category merges, assign functional categories
- **Multi-Category Support**:
  - Entities can belong to MULTIPLE categories simultaneously
  - **Category Types**: primary, functional, therapeutic, system, pathway, target, comorbidity
  - Example: "Probiotics" = supplement (PRIMARY) + gut flora modulator (FUNCTIONAL) + IBS treatment (THERAPEUTIC)
  - Cross-category merges: "Probiotics" (supplement) + "FMT" (procedure) → "Gut Flora Modulators" (functional group)
- **Output**:
  - Multi-level hierarchies (great-grandparent → grandparent → parent → child)
  - Junction tables for multi-category relationships
  - Functional category suggestions from LLM for cross-category groups
- **Expected Results**:
  - 40-50% cluster reduction
  - More generalizable treatment insights
  - Cross-category pattern discovery

### Phase 4a: Knowledge Graph Construction ✅
- **Location**: `back_end/src/phase_4_data_mining/`
- **Purpose**: Build multi-edge bidirectional knowledge graph from Phase 3 canonical groups
- **Technology**: Graph construction with canonical group nodes (not raw intervention names)
- **Key Features**:
  - **Canonical Group Nodes**: 538 nodes (vs. 716 duplicate raw names)
  - **Evidence Pooling**: Aggregate evidence across cluster members
  - **Multi-Edge Preservation**: All studies retained (positive, negative, neutral)
  - **Bidirectional Queries**: "What treats X?" and "What does Y treat?"
  - **Database Integration**: Saves to `knowledge_graph_nodes` and `knowledge_graph_edges` tables
- **Performance**: Builds complete graph from Phase 3 results in seconds
- **Files**: `phase_4a_knowledge_graph.py`

### Phase 4b: Bayesian Evidence Scoring ✅
- **Location**: `back_end/src/phase_4_data_mining/`
- **Purpose**: Score canonical groups using Bayesian statistics with pooled evidence
- **Technology**: Beta distribution priors + evidence aggregation
- **Key Features**:
  - **Canonical Group Scoring**: Score clusters (not raw names)
  - **Evidence Pooling**: Pool evidence across all cluster members for better statistical power
  - **Innovation Penalty Solution**: Uses statistical confidence (not raw counts)
  - **Conservative Scoring**: 10th percentile (worst-case scenario)
  - **Database Integration**: Saves to `bayesian_scores` table
- **Example**: "Probiotics" cluster (5 studies) vs. "probiotic supplementation" (2 studies) → Pool to 7 studies for higher confidence
- **Performance**: Scores all canonical groups against all conditions in minutes
- **Files**: `phase_4b_bayesian_scorer.py`

---

## Database Schema (25 Tables)

### Core Data Tables (2 tables)
1. **`papers`** - PubMed articles with metadata and fulltext
2. **`interventions`** - Extracted treatments and outcomes with mechanism data

### Phase 3a & 3b Semantic Normalization & Categorization (2 tables)
3. **`semantic_hierarchy`** - Hierarchical structure linking interventions AND conditions to canonical groups
4. **`canonical_groups`** - Canonical entity groupings with Layer 0 categories for both interventions and conditions

Note: `entity_relationships` table removed - relationship analysis moved to Phase 3d (cluster-level)

### Phase 3c Mechanism Clustering (4 tables)
5. **`mechanism_clusters`** - Mechanism cluster metadata with canonical names and hierarchy
6. **`mechanism_cluster_membership`** - Mechanism-to-cluster assignments (HDBSCAN or singleton)
7. **`intervention_mechanisms`** - Junction table linking interventions to mechanism clusters
8. **`mechanism_condition_associations`** - Analytics for which mechanisms work for which conditions

### Phase 3d Multi-Category Support (3 tables) 🧪
9. **`intervention_category_mapping`** - Many-to-many intervention-to-category relationships
10. **`condition_category_mapping`** - Many-to-many condition-to-category relationships
11. **`mechanism_category_mapping`** - Many-to-many mechanism-to-category relationships

### Data Mining Analytics (11 tables)
12. **`knowledge_graph_nodes`** - Nodes in medical knowledge graph
13. **`knowledge_graph_edges`** - Multi-edge graph relationships
14. **`bayesian_scores`** - Bayesian evidence scoring
15. **`treatment_recommendations`** - AI treatment recommendations
16. **`research_gaps`** - Under-researched areas
17. **`innovation_tracking`** - Emerging treatment tracking
18. **`biological_patterns`** - Mechanism and pattern discovery
19. **`condition_similarities`** - Condition similarity matrix
20. **`intervention_combinations`** - Synergistic combination analysis
21. **`failed_interventions`** - Catalog of ineffective treatments
22. **`data_mining_sessions`** - Session tracking

### Configuration & System (2 tables)
23. **`intervention_categories`** - 13-category taxonomy configuration
24. **`sqlite_sequence`** - SQLite internal auto-increment

### Legacy Tables (4 tables - DEPRECATED)
25. **`canonical_entities`, `entity_mappings`, `llm_normalization_cache`, `normalized_terms_cache`** - Replaced by Phase 3a semantic normalization

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
├── phase_3_semantic_normalization/   # Phase 3: Clustering-First Architecture
│   ├── phase_3a_base_embedder.py                # 3a: Base embedder class
│   ├── phase_3a_intervention_embedder.py        # 3a: Intervention embeddings
│   ├── phase_3a_condition_embedder.py           # 3a: Condition embeddings
│   ├── phase_3a_mechanism_embedder.py           # 3a: Mechanism embeddings
│   ├── phase_3b_base_clusterer.py               # 3b: Base clusterer class
│   ├── phase_3b_hierarchical_clusterer.py       # 3b: Hierarchical clustering
│   ├── phase_3b_hdbscan_clusterer.py            # 3b: HDBSCAN (alternative)
│   ├── phase_3b_singleton_handler.py            # 3b: 100% assignment guarantee
│   ├── phase_3c_base_namer.py                   # 3c: Base namer class
│   ├── phase_3c_llm_namer.py                    # 3c: LLM canonical naming
│   ├── phase_3abc_orchestrator.py               # Main orchestrator
│   ├── phase_3_config.yaml                      # Configuration
│   ├── phase_3d/                                # 3d: Hierarchical merging
│   └── ground_truth/                            # Ground truth labeling
│
├── phase_4_data_mining/              # Phase 4: Knowledge Graph + Bayesian Scoring
│   ├── phase_4a_knowledge_graph.py              # 4a: Knowledge graph construction
│   ├── phase_4b_bayesian_scorer.py              # 4b: Bayesian evidence scoring
│   ├── scoring_utils.py                         # Shared scoring utilities
│   └── phase_4_config.yaml                      # Configuration
│
├── orchestration/                    # Pipeline Orchestrators
│   ├── phase_1_paper_collector.py
│   ├── phase_2_llm_processor.py
│   ├── phase_3abc_semantic_normalizer.py        # Phase 3 orchestrator
│   ├── phase_4_data_miner.py                    # Phase 4 orchestrator (NEW)
│   └── batch_medical_rotation.py                # Main pipeline controller
│
├── data_mining/                      # Advanced Analytics (Legacy/Standalone)
│   ├── data_mining_orchestrator.py
│   ├── medical_knowledge_graph.py               # (Original - kept for compatibility)
│   ├── bayesian_scorer.py                       # (Original - kept for compatibility)
│   └── ...
│
├── utils/                            # General Utilities
├── migrations/                       # Database Migrations
└── data/                            # Configuration & Repositories
```

### Pipeline Flow
```
Phase 1 → Phase 2 → Phase 3a → Phase 3b → Phase 3c → Phase 3d → Phase 4a → Phase 4b
   ↓         ↓          ↓          ↓          ↓          ↓          ↓          ↓
Papers   Extracts   Embeddings Clusters    Names    Hierarchies  Graph    Scores
                   (1024-dim)   (538)   (canonical) (experimental) (538 nodes) (Bayesian)
```

**Phase 3 Details (Clustering-First)**:
- **3a**: Embed entities using mxbai-embed-large → Cache vectors
- **3b**: Cluster embeddings with hierarchical algorithm → 100% assignment
- **3c**: Name clusters with qwen3:14b → Assign categories
- **3d**: Merge similar clusters → Build parent-child hierarchies

**Phase 4 Details (Data Mining)**:
- **4a**: Build knowledge graph from canonical groups → Cleaner nodes, pooled evidence
- **4b**: Score canonical groups with Bayesian statistics → Better statistical power

### File Naming Convention
- **Phase-specific files**: `phase_X_descriptive_name.py` (e.g., `phase_2_single_model_analyzer.py`)
- **Sub-phase files**: `phase_Xa_name.py` (e.g., `phase_3b_intervention_categorizer.py`)
- **Generic utilities**: Keep descriptive names without phase prefix (e.g., `database_manager.py`)
- **Orchestrators**: Located in `orchestration/`, named by phase

---

## Key Commands

### Complete Workflow
```bash
# Single iteration: Collection → Processing → Semantic Normalization → Group Categorization → Mechanism Clustering → Data Mining
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10

# Continuous mode: Infinite loop (restarts Phase 1 after Phase 4)
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

# Phase 4: Data Mining (Knowledge Graph + Bayesian Scoring)
python -m back_end.src.orchestration.phase_4_data_miner  # Run complete Phase 4 (4a + 4b)
python -m back_end.src.orchestration.phase_4_data_miner --phase-4a-only  # Knowledge graph only
python -m back_end.src.orchestration.phase_4_data_miner --phase-4b-only  # Bayesian scoring only
python -m back_end.src.orchestration.phase_4_data_miner --status  # Check Phase 4 status

# Data mining and analysis (standalone/legacy tools)
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
- **SQLite**: Database with connection pooling (25 tables)
- **Ollama**: Local LLM inference
  - **qwen3:14b**: Extraction + naming (temperature=0.0)
  - **mxbai-embed-large**: Embeddings (1024-dim)
- **PubMed API**: Primary paper source
- **PMC & Unpaywall**: Fulltext retrieval
- **Circuit Breaker Pattern**: Robust error handling
- **Retry Logic**: Automatic retry with exponential backoff for LLM failures
- **Caching**: Embeddings, clusters, and naming results cached for speed
- **FAST_MODE**: Logging suppression for high-throughput (enabled by default)

---

## Project Conventions

### Code Style
- No emojis in print statements or code comments
- Comprehensive error handling with circuit breaker patterns
- Session persistence for all long-running operations
- Thermal protection for GPU-intensive operations

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
- **Bayesian Score Ranking (Phase 4b)** ✅: Default sorting by evidence-based Bayesian scores
  - Color-coded scores: Green (>0.7), Yellow (>0.5), Red (<0.5)
  - Posterior mean + conservative (10th percentile) estimates
  - Evidence breakdown (positive/negative/neutral counts)
  - Removes innovation penalty (new treatments fairly ranked)
- **Summary Statistics**: Total interventions, conditions, papers, canonical groups, relationships, high-scoring interventions
- **Correlation Strength Display**: Categorical labels (Very Strong ≥0.75, Strong ≥0.50, Weak ≥0.25, Very Weak <0.25)
- **Filtering**: By intervention category (13), condition category (18), functional category, therapeutic category, correlation type, confidence threshold
- **Multi-Category Display** 🧪: Color-coded badges for multiple category types (primary, functional, therapeutic, etc.)
- **Semantic Integration**: Displays canonical groups and 4-layer hierarchical classifications
- **Details Modal**: Full intervention data, mechanism of action, Bayesian statistics, study details, paper information, all category types

### Data Export
```bash
python -m back_end.src.utils.export_frontend_data
```
Exports SQLite → JSON with Phase 4b Bayesian scores, Phase 3.5 hierarchical data, metadata, and top performers.

### Bayesian Score Integration (October 15, 2025)
- **Backend**: [export_frontend_data.py](back_end/src/utils/export_frontend_data.py) joins `bayesian_scores` table
- **Frontend**: [script.js](frontend/script.js) formats and displays scores with `formatBayesianScore()`
- **Styling**: [style.css](frontend/style.css) provides color-coded visual indicators
- **Default Ranking**: Interventions sorted by Bayesian score (desc) → Strength (desc) → Confidence (desc)
- **Statistics**: New "High Bayesian Score (>0.7)" card in summary section

### Browser Cache Management (October 16, 2025)
**IMPORTANT**: When updating frontend files (JavaScript/CSS), browsers aggressively cache static assets.

**Problem**: Modified files on disk may not reflect in browser (old cached version loads instead)

**Solution**: Use cache-busting version parameters in HTML:
```html
<!-- Increment version number after ANY JavaScript/CSS changes -->
<script src="script.js?v=2"></script>
<link rel="stylesheet" href="style.css?v=2">
```

**Best Practices**:
- Increment version parameter (`?v=3`, `?v=4`, etc.) after each update
- Use date format `?v=YYYYMMDD` or semantic versioning for clarity
- Hard refresh (Ctrl+F5) alone is unreliable - users may not know the shortcut
- Version parameters force browsers to treat file as new URL, bypassing cache
- For production: Use build tools (webpack, gulp) to auto-generate cache-busting hashes

**Current Version**: `script.js?v=7`, `style.css?v=7` (updated October 16, 2025 - mechanism canonical names + layout fixes)

### Frontend Design Challenges & Solutions (October 16, 2025)

**See [BUGFIX_LOG.md](BUGFIX_LOG.md) for detailed technical documentation.**

#### Common Issues When Working with DataTables.js

**Issue 1: Column Overflow and Text Wrapping**
- **Problem**: Content spilling into adjacent columns
- **Root Cause**: DataTables' `autoWidth: true` + flexible `table-layout: auto` ignore CSS constraints
- **Solution Pattern**:
  ```javascript
  // DataTables config
  {
    autoWidth: false,
    columnDefs: [
      { targets: [n], width: 'XXXpx', className: 'custom-class' }
    ]
  }
  ```
  ```css
  /* CSS enforcement */
  #table-id {
    table-layout: fixed;
  }
  #table-id td.custom-class {
    max-width: XXXpx !important;
    width: XXXpx !important;
    overflow: visible !important;
    white-space: normal;
  }
  ```

**Issue 2: Text Truncation with Ellipsis (...)**
- **Problem**: Content hidden with "..." instead of wrapping
- **Root Cause**: Generic `text-overflow: ellipsis` on all cells
- **Solution**: Use `overflow: visible` + `white-space: normal` for text-heavy columns

**Issue 3: Column Headers Overlapping**
- **Problem**: Header text mashing together unreadably
- **Root Cause**: Default `white-space: nowrap` prevents line breaks, insufficient column width
- **Solution Pattern**:
  ```css
  #table-id th {
    white-space: normal !important;
    word-wrap: break-word !important;
    padding: 12px 8px !important;
    vertical-align: middle !important;
    line-height: 1.3 !important;
  }
  ```

**Issue 4: Browser Cache Not Updating**
- **Problem**: Changes to CSS/JS not reflecting in browser
- **Root Cause**: Aggressive browser caching of static assets
- **Solution**: Version parameters (already implemented above)

#### Key Lessons for Frontend Development

1. **CSS !important is necessary** when overriding DataTables' inline styles
2. **Set widths in BOTH JavaScript and CSS** for reliability
3. **overflow: visible vs. hidden**: visible allows wrapping, hidden causes truncation
4. **table-layout: fixed + autoWidth: false** must be used together
5. **Explicit widths for ALL columns** prevents layout collapse
6. **Apply header fixes universally** using `th` selector
7. **Test with real data** before declaring success

#### DataTables.js Quirks to Remember

- **Inline styles dominate**: CSS needs `!important` to override
- **Auto-width is aggressive**: Must explicitly disable with `autoWidth: false`
- **Responsive mode unpredictable**: Fixed layouts + explicit widths provide control
- **CSS specificity matters**: Use `#table-id td.class-name` for highest precedence

#### Design Patterns Established

**Column Content Display**:
```javascript
function formatColumnContent(dataArray) {
    if (!dataArray || dataArray.length === 0) {
        return '<span class="none">Not specified</span>';
    }
    const items = dataArray.map(item =>
        `<div class="item">${item}</div>`
    ).join('');
    return `<div class="list">${items}</div>`;
}
```

**Column Width Control**:
- Set in DataTables config: `{ targets: [n], width: 'XXXpx' }`
- Enforce in CSS: `max-width: XXXpx !important; width: XXXpx !important;`
- Allow wrapping: `overflow: visible; white-space: normal;`

**Cache Busting**:
- Always increment version after CSS/JS changes
- Format: `?v=X` (increment) or `?v=YYYYMMDD` (date-based)

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


## Current Status (October 15, 2025)

**Phase 3 Migration Complete**: Successfully migrated from naming-first to clustering-first architecture.
**Phase 4 Integration Complete**: Knowledge graph and Bayesian scoring now integrated into main pipeline.
**Frontend Bayesian Integration Complete**: Bayesian scores now drive default intervention ranking.

### Database Statistics
- **Papers**: 533 research papers (all with mechanism data)
- **Interventions**: 777 interventions (100% mechanism coverage)
- **Intervention Clusters**: 538 canonical groups (from 716 unique names)
- **Conditions**: ~400 unique conditions
- **Mechanisms**: 666 unique mechanisms
- **Bayesian Scores**: 259 intervention-condition relationships scored

### Performance Metrics
- **Phase 3a (Embedding)**: <1s (100% cache hit rate after first run)
- **Phase 3b (Clustering)**: 0.2s per entity type (hierarchical threshold=0.7)
- **Phase 3c (Naming)**: ~70 minutes for 538 clusters (uncached), instant with cache
- **Phase 4a (Knowledge Graph)**: Seconds (builds from Phase 3 canonical groups)
- **Phase 4b (Bayesian Scoring)**: ~3 minutes (scores 259 canonical group pairs)
- **Frontend Export**: ~2 seconds (generates interventions.json with Bayesian data)
- **Architecture**: Clustering-first with integrated data mining and Bayesian-ranked frontend

### Phase 3 Migration Details (October 15, 2025)
- **Old Architecture**: Naming-first (nomic-embed-text 768-dim → LLM canonical extraction → grouping)
- **New Architecture**: Clustering-first (mxbai-embed-large 1024-dim → hierarchical clustering → LLM naming)
- **Benefits**:
  - Better separation with 1024-dim embeddings
  - Optimal clustering threshold (0.7) from hyperparameter tuning
  - 100% entity assignment (no noise points)
  - Faster with aggressive caching
- **Backup**: Old code saved in `phase_3_semantic_normalization_OLD_BACKUP_20251015/`

### Phase 4 Integration Details (October 15, 2025)
- **Migration**: data_mining tools → phase_4_data_mining pipeline integration
- **New Components**:
  - `phase_4a_knowledge_graph.py` - Builds graph from canonical groups (not raw names)
  - `phase_4b_bayesian_scorer.py` - Scores canonical groups with pooled evidence
  - `phase_4_data_miner.py` - Orchestrator for Phase 4a + 4b
- **Benefits**:
  - Cleaner knowledge graph: 538 nodes vs. 716 duplicates
  - Better statistical power: Pooled evidence across cluster members
  - Integrated pipeline: Automatic execution after Phase 3
- **Backward Compatibility**: Original data_mining tools kept for standalone use

### Frontend Bayesian Integration (October 15, 2025)
- **Problem Solved**: Frontend was sorting by LLM extraction confidence (not effectiveness)
- **Solution**: Integrated Phase 4b Bayesian scores as default ranking method
- **Implementation**:
  - **Backend**: Fixed Phase 4b bugs (schema mismatch), generated 259 scores
  - **Export**: Updated `export_frontend_data.py` to JOIN `bayesian_scores` table
  - **Frontend**: Added Bayesian Score column with color-coded display (green/yellow/red)
  - **Sorting**: Changed default to Bayesian score (desc) → Strength (desc) → Confidence (desc)
  - **UI**: Added "High Bayesian Score (>0.7)" summary statistic
  - **Modal**: Displays full Bayesian statistics (posterior mean, conservative score, evidence breakdown, Bayes factor)
- **Technical Details**:
  - Scoring formula: Beta(α=1, β=1) prior → Posterior mean = (α+positive) / (α+β+positive+negative)
  - Conservative estimate: 10th percentile of posterior distribution
  - Evidence pooling: Aggregates across canonical group members (e.g., "probiotics" + "probiotic supplementation")
  - Color thresholds: Green >0.7, Yellow >0.5, Red <0.5
- **User Impact**:
  - Most effective interventions (by evidence) appear first
  - No innovation penalty (new treatments fairly ranked)
  - Transparent evidence breakdown (positive/negative/neutral counts)
  - Statistically rigorous ranking (Bayesian statistics)

---

*Last Updated: October 15, 2025*
*Architecture: End-to-End Pipeline (Phase 1 → 2 → 3a → 3b → 3c → 3d → 4a → 4b → Frontend)*
*Status: Production Ready with Bayesian-Ranked Frontend ✅*
