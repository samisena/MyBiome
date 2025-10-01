# MyBiome Health Research Pipeline

An automated biomedical research pipeline that collects papers, extracts intervention-outcome relationships using dual LLM analysis, and performs statistical analysis of treatment effectiveness across medical conditions.

## Quick Start

**Environment**: Conda virtual environment called 'venv'
```bash
conda activate venv
```

**Run Complete Pipeline**:
```bash
# Batch medical rotation: 3 phases (collection → processing → deduplication)
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10
```

## System Architecture

### Core Pipeline: batch_medical_rotation.py

The system operates in **3 sequential phases**:

#### Phase 1: Paper Collection
- **Source**: PubMed API (primary)
- **Note**: Semantic Scholar is DISABLED in batch pipeline (use_interleaved_s2=False)
- **Parallelism**: 2 concurrent workers (reduced from 8 to prevent API overload)
- **Output**: Papers stored in SQLite database

#### Phase 2: LLM Processing (Dual-Model Extraction)
- **Models**: gemma2:9b and qwen2.5:14b (sequential, not parallel)
- **Process**: Each paper is processed by BOTH models independently
- **Output**: Each paper generates 2 sets of interventions (one per model)
- **Side Effect**: Creates duplicate interventions from same paper (by design)

#### Phase 3: Deduplication & Canonical Merging

**Two distinct operations occur here:**

##### A. Same-Paper Deduplication (Prevents Evidence Inflation)
**Problem**: Dual-model extraction creates duplicates
```
Paper 41031311:
  - gemma2:9b → "vitamin D for cognitive impairment"
  - qwen2.5:14b → "vitamin D for type 2 diabetes mellitus-induced cognitive impairment"
```

**Solution** (in `batch_entity_processor.py`):
1. Simple normalization first (fast string matching)
2. LLM semantic verification if needed (qwen2.5:14b determines if conditions are equivalent)
3. **DELETE one duplicate, keep only ONE intervention**
4. Mark kept record with both models: `models_used: "gemma2:9b, qwen2.5:14b"`
5. LLM builds consensus on which condition name to use

**Result**: Each paper contributes only 1 record per intervention (prevents statistical inflation)

##### B. Canonical Entity Merging (Cross-Paper Unification)
**Problem**: Different papers use different names for same intervention
```
Paper A: "vitamin D"
Paper B: "Vitamin D3"
Paper C: "cholecalciferol"
```

**Solution**: All link to same canonical entity
```
Paper A intervention → canonical_id: 1 (vitamin D)
Paper B intervention → canonical_id: 1 (vitamin D)
Paper C intervention → canonical_id: 1 (vitamin D)
```

**Result**: Statistical analysis shows "150 papers support vitamin D" (aggregated evidence)

### Key Distinction Table

| Aspect | Same-Paper Deduplication | Canonical Merging |
|--------|-------------------------|-------------------|
| **Scope** | Within single paper | Across all papers |
| **Action** | DELETE duplicate records | LINK to canonical entity |
| **Why** | Dual extraction creates duplicates | Papers use different terminology |
| **When** | Immediately after LLM extraction | During entity normalization |
| **Impact** | Prevents double-counting | Enables aggregated analysis |

## Core Components

### Data Collection (`back_end/src/data_collection/`)

**PubMedCollector** (`pubmed_collector.py`)
- Collects papers investigating interventions and health conditions
- Checks for free fulltext availability in PMC or via DOI/Unpaywall
- Used by batch_medical_rotation with Semantic Scholar disabled

**PaperParser** (`paper_parser.py`)
- Converts PubMed XML metadata to Python format
- Stores papers in SQLite database: `back_end/data/processed/intervention_research.db`
- Automatically inserts papers during collection

**FulltextRetriever** (`fulltext_retriever.py`)
- Downloads free fulltext from PMC or Unpaywall when available

**DatabaseManager** (`database_manager.py`)
- Handles all database operations
- Main tables: `papers`, `interventions`, `canonical_entities`, `intervention_categories`
- Includes methods for querying, inserting, updating, and statistical analysis

### LLM Processing (`back_end/src/llm_processing/`)

**DualModelAnalyzer** (`dual_model_analyzer.py`)
- Processes each paper with gemma2:9b and qwen2.5:14b sequentially
- Extracts intervention-condition relationships with supporting evidence
- Stores both model outputs (creates intentional duplicates)

**BatchEntityProcessor** (`batch_entity_processor.py`)
- **Same-paper deduplication**: Merges duplicates from dual extraction
- **Canonical entity merging**: Unifies intervention names across papers
- Uses LLM for semantic verification when simple normalization fails
- Critical for preventing statistical inflation

### Orchestration (`back_end/src/orchestration/`)

**batch_medical_rotation.py** (PRIMARY pipeline)
- 3-phase batch processing: collection → processing → deduplication
- Processes 60 medical conditions (12 specialties × 5 conditions each)
- Session management with resumable execution
- Status tracking and error recovery

**rotation_paper_collector.py**
- Batch collection for all conditions
- PubMed-only (S2 disabled)
- 2 parallel workers for API stability

**rotation_llm_processor.py**
- Orchestrates dual-model processing
- Thermal monitoring for GPU safety
- Batch processing with VRAM optimization

**rotation_deduplication_integrator.py**
- Coordinates deduplication phase
- Calls BatchEntityProcessor for actual deduplication work

## Usage Examples

### Run Complete Pipeline
```bash
# Standard run
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10

# Resume interrupted session
python -m back_end.src.orchestration.batch_medical_rotation --resume

# Resume from specific phase
python -m back_end.src.orchestration.batch_medical_rotation --resume --start-phase processing

# Check status
python -m back_end.src.orchestration.batch_medical_rotation --status
```

### Database Queries

**Count papers**:
```python
from back_end.src.data_collection.database_manager import database_manager

with database_manager.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM papers")
    print(f"Total papers: {cursor.fetchone()[0]}")
```

**Accurate statistical analysis** (respects deduplication):
```sql
SELECT intervention_name, health_condition,
       COUNT(DISTINCT paper_id) as evidence_count
FROM interventions
WHERE is_consensus_primary = 1 OR consensus_group_id IS NULL
GROUP BY intervention_name, health_condition
ORDER BY evidence_count DESC;
```

## Important Notes

1. **Semantic Scholar**: Disabled in batch_medical_rotation to prevent hanging issues
2. **Dual Extraction**: Both models process every paper - duplicates are expected and handled
3. **Statistical Accuracy**: Always use deduplication-aware queries to prevent inflated counts
4. **Model Attribution**: Merged interventions show both models in `models_used` field
5. **Thermal Protection**: GPU temperature monitoring prevents overheating during LLM processing

## Database Schema

Main tables:
- `papers`: PubMed articles with metadata
- `interventions`: Extracted treatment-outcome relationships
- `canonical_entities`: Unified intervention names
- `intervention_categories`: Intervention type classifications
- `knowledge_graph_nodes/edges`: For relationship mapping
- `bayesian_scores`: Evidence quality metrics

## Technology Stack

- **Python 3.13**: Core language
- **SQLite**: Database (with connection pooling)
- **Ollama**: Local LLM inference (gemma2:9b, qwen2.5:14b)
- **PubMed API**: Primary paper source
- **PyTorch**: GPU optimization (optional)

## Development Guidelines

- No emojis in code or output
- Local LLMs only (no external API calls)
- Session persistence for all long-running operations
- Comprehensive error handling with retry logic
- Thermal protection for GPU-intensive tasks
