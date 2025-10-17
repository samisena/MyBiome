"""
Shared Constants for MyBiome Research Platform

Centralized location for all shared constants used across the codebase.
This prevents duplication and ensures consistency.

Created: October 16, 2025 (Repository cleanup)
"""

# === VALIDATION CONSTANTS ===

# Placeholder patterns for detecting invalid/placeholder text
PLACEHOLDER_PATTERNS = {
    '...', 'N/A', 'n/a', 'NA', 'na', 'null', 'NULL', 'none', 'None',
    'unknown', 'Unknown', 'UNKNOWN', 'placeholder', 'Placeholder', 'PLACEHOLDER',
    'TBD', 'tbd', 'TODO', 'todo', 'not specified', 'not available',
    'various', 'multiple', 'several', 'different', 'mixed', 'varies'
}

# Minimum text length for valid entity names
MIN_TEXT_LENGTH = 3

# === BAYESIAN SCORING CONSTANTS ===

# Bayesian score thresholds for quality classification
BAYESIAN_SCORE_HIGH = 0.7  # High confidence interventions
BAYESIAN_SCORE_MEDIUM = 0.5  # Medium confidence interventions
BAYESIAN_SCORE_LOW = 0.3  # Low confidence interventions

# Conservative estimate percentile for Bayesian scoring
BAYESIAN_CONSERVATIVE_PERCENTILE = 10  # 10th percentile (worst-case scenario)

# === PHASE 3 SEMANTIC NORMALIZATION CONSTANTS ===

# Default confidence threshold for semantic matching
CONFIDENCE_THRESHOLD_DEFAULT = 0.7

# Clustering distance threshold (hierarchical clustering)
CLUSTERING_DISTANCE_THRESHOLD = 0.7  # Optimal from hyperparameter experiments

# Minimum similarity score for semantic grouping
SEMANTIC_SIMILARITY_MINIMUM = 0.7

# === LLM PROCESSING CONSTANTS ===

# Batch sizes for LLM processing
LLM_BATCH_SIZE = 20  # Standard batch size for LLM calls
LLM_BATCH_SIZE_SMALL = 5  # Small batches for error recovery
LLM_BATCH_SIZE_LARGE = 50  # Large batches for bulk processing

# Embedding generation batch sizes
EMBEDDING_BATCH_SIZE = 32  # Standard embedding batch size
EMBEDDING_BATCH_SIZE_MECHANISMS = 10  # Smaller for longer mechanism text

# LLM temperature settings
LLM_TEMPERATURE_DETERMINISTIC = 0.0  # For consistent naming/categorization
LLM_TEMPERATURE_CREATIVE = 0.4  # For extraction (optimized from experiments)

# === MODEL CONFIGURATION ===

# Current embedding model (migrated Oct 16, 2025)
EMBEDDING_MODEL = "mxbai-embed-large"
EMBEDDING_DIMENSION = 1024

# Legacy embedding model (for reference/compatibility)
EMBEDDING_MODEL_LEGACY = "nomic-embed-text"
EMBEDDING_DIMENSION_LEGACY = 768

# Current LLM model
LLM_MODEL_CURRENT = "qwen3:14b"

# === TIMEOUT CONSTANTS ===

# API timeouts (in seconds)
# NOTE: These are conservative for local Ollama - may not be strictly necessary
TIMEOUT_EMBEDDING_DEFAULT = 30  # Standard embedding generation
TIMEOUT_EMBEDDING_LONG = 60  # For mxbai-embed-large (larger model)
TIMEOUT_LLM_DEFAULT = 60  # Standard LLM calls
TIMEOUT_LLM_BATCH = 300  # Large batch LLM processing (5 minutes)

# Ollama client constants (for OllamaClient class)
OLLAMA_API_URL = "http://localhost:11434"  # Default Ollama API endpoint
OLLAMA_TIMEOUT_SECONDS = 60  # Request timeout
OLLAMA_RETRY_DELAYS = [10, 30, 60]  # Retry delays in seconds (exponential backoff)

# === INTERVENTION CATEGORIES (13 categories) ===

INTERVENTION_CATEGORIES = [
    "exercise",
    "diet",
    "supplement",
    "medication",
    "therapy",
    "lifestyle",
    "surgery",
    "test",
    "device",
    "procedure",
    "biologics",
    "gene_therapy",
    "emerging"
]

# === CONDITION CATEGORIES (18 categories) ===

CONDITION_CATEGORIES = [
    "cardiac",
    "neurological",
    "digestive",
    "pulmonary",
    "endocrine",
    "renal",
    "oncological",
    "rheumatological",
    "psychiatric",
    "musculoskeletal",
    "dermatological",
    "infectious",
    "immunological",
    "hematological",
    "nutritional",
    "toxicological",
    "parasitic",
    "other"
]

# === OUTCOME TYPES (Health Impact Framework) ===

# Migrated from correlation_type → outcome_type (Oct 16, 2025)
OUTCOME_TYPE_IMPROVES = "improves"  # Intervention improves patient health
OUTCOME_TYPE_WORSENS = "worsens"  # Intervention worsens patient health
OUTCOME_TYPE_NO_EFFECT = "no_effect"  # No measurable health impact
OUTCOME_TYPE_INCONCLUSIVE = "inconclusive"  # Mixed/unclear evidence

OUTCOME_TYPES = [
    OUTCOME_TYPE_IMPROVES,
    OUTCOME_TYPE_WORSENS,
    OUTCOME_TYPE_NO_EFFECT,
    OUTCOME_TYPE_INCONCLUSIVE
]

# Legacy correlation type values (for backward compatibility)
CORRELATION_TYPE_POSITIVE = "positive"  # DEPRECATED - use OUTCOME_TYPE_IMPROVES
CORRELATION_TYPE_NEGATIVE = "negative"  # DEPRECATED - use OUTCOME_TYPE_WORSENS
CORRELATION_TYPE_NEUTRAL = "neutral"  # DEPRECATED - use OUTCOME_TYPE_NO_EFFECT

# === DATABASE CONSTANTS ===

# Maximum connections for SQLite connection pooling
DB_MAX_CONNECTIONS = 5

# Batch processing sizes
DB_BATCH_SIZE_DEFAULT = 50
DB_BATCH_SIZE_LARGE = 100

# === CACHE CONSTANTS ===

# Cache file names
CACHE_EMBEDDINGS_FILENAME = "embeddings.pkl"
CACHE_CANONICALS_FILENAME = "canonicals.pkl"
CACHE_LLM_DECISIONS_FILENAME = "llm_decisions.pkl"

# === COLOR CODING FOR FRONTEND ===

# Bayesian score color thresholds (matches frontend style.css)
COLOR_THRESHOLD_GREEN = 0.7  # High confidence (green)
COLOR_THRESHOLD_YELLOW = 0.5  # Medium confidence (yellow)
# Below 0.5 = Low confidence (red)

# === LOGGING CONSTANTS ===

# Fast mode threshold for logging suppression
FAST_MODE_ENABLED = True  # Default: suppress logs for performance

# Logging levels
LOG_LEVEL_FAST_MODE = "CRITICAL"  # Only critical errors in fast mode
LOG_LEVEL_DEBUG_MODE = "INFO"  # Full logging when debugging

# === PERFORMANCE CONSTANTS ===

# Iteration delay for continuous mode (thermal protection)
ITERATION_DELAY_DEFAULT = 60.0  # seconds

# Processing batch sizes for papers
PAPERS_BATCH_SIZE_SMALL = 5  # Thermal management
PAPERS_BATCH_SIZE_STANDARD = 10  # Standard processing
PAPERS_BATCH_SIZE_LARGE = 100  # Bulk collection

# === FILE PATHS (relative to project root) ===

# These are typically constructed from config.py, but provided for reference
PATH_DATA_ROOT = "back_end/data"
PATH_DB_PROCESSED = "back_end/data/processed"
PATH_LOGS = "back_end/data/logs"
PATH_CACHE = "back_end/data/semantic_normalization_cache"
PATH_RESULTS = "back_end/data/semantic_normalization_results"
PATH_FRONTEND_DATA = "frontend/data"

# === VERSION CONSTANTS ===

# Repository cleanup version
CLEANUP_VERSION = "2025-10-16"
CLEANUP_DESCRIPTION = "Comprehensive repository cleanup and optimization"

# Architecture versions
PHASE_3_ARCHITECTURE_VERSION = "clustering-first"  # Migrated Oct 15, 2025
PHASE_3_ARCHITECTURE_VERSION_OLD = "naming-first"  # Deprecated Oct 15, 2025

# Embedding model migration
EMBEDDING_MIGRATION_DATE = "2025-10-16"
EMBEDDING_MIGRATION_FROM = "nomic-embed-text (768-dim)"
EMBEDDING_MIGRATION_TO = "mxbai-embed-large (1024-dim)"

# Field removal migration
FIELD_REMOVAL_DATE = "2025-10-16"
FIELDS_REMOVED = ["correlation_strength", "extraction_confidence"]
FIELD_RENAMED = "correlation_type → outcome_type"
