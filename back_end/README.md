# MyBiome Back-End

Biomedical research automation pipeline for intervention extraction and analysis.

## 📁 Folder Structure

### Core Source Code (`src/`)
- **`data/`**: Configuration, validation, repositories, and core data management
- **`paper_collection/`**: PubMed, Semantic Scholar, and fulltext retrieval
- **`llm/`**: LLM processing, dual-model analysis, and prompt management
- **`interventions/`**: Intervention taxonomy, validation, and search terms
- **`action_functions/`**: Modular action functions for pipeline operations
- **`utils/`**: Utility functions and imports

### Execution Scripts

#### 🔬 **data_mining/**
Scripts for analyzing patterns, correlations, and data consistency.

#### 🤖 **orchestration/**
High-level automation for autonomous research campaigns with fault tolerance.

#### ⚙️ **pipelines/**
Core processing pipelines for data collection and LLM analysis.

#### 🧪 **testing/**
Performance testing, validation, and system verification tools.

#### 🛠️ **utilities/**
Database management and interactive administrative tools.

## 🚀 Quick Start

1. **Configuration**: Set up environment variables in `.env`
2. **Data Collection**: Use scripts in `pipelines/` for collecting papers
3. **Processing**: Run LLM analysis using `pipelines/run_llm_processing.py`
4. **Automation**: Use `orchestration/` scripts for unattended operation
5. **Analysis**: Explore results with `data_mining/` tools

## 🏗️ Architecture

- **Dual-model LLM analysis** (gemma2:9b + qwen2.5:14b)
- **Multi-source data collection** (PubMed, Semantic Scholar, PMC, Unpaywall)
- **Robust error handling** with circuit breakers and retry logic
- **Thermal protection** for GPU operations
- **Session persistence** for long-running tasks
- **Comprehensive validation** with category-specific rules

## 📊 Data Flow

```
PubMed API → Paper Collection → LLM Analysis → Validation → Database
     ↓              ↓              ↓           ↓         ↓
Semantic Scholar → Enrichment → Dual Models → Quality → Storage
```