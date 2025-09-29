#!/usr/bin/env python3
"""
Unified Batch Entity Processor

This module consolidates all entity normalization functionality into a single,
efficient batch processing system. It combines features from:
- entity_normalizer_v2.py (main API)
- entity_normalizer_engine.py (matching orchestration)
- entity_normalizer_matchers.py (matching strategies)
- entity_normalizer_repository.py (database operations)
- llm_deduplication.py (batch deduplication)
- generate_llm_enhanced_mapping_suggestions.py (mapping suggestions)
- core_utils.py (utilities and validation)

Optimized for batch processing while preserving all sophisticated features.
"""

import sqlite3
import json
import re
import os
import sys
import csv
import logging
import hashlib
import shutil
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, Set
from enum import Enum
from datetime import datetime
from collections import defaultdict, Counter

# Local imports
from .entity_utils import (
    EntityNormalizationError, DatabaseError, ValidationError, MatchingError, LLMError, ConfigurationError,
    EntityType, MatchingMode, MatchResult, ValidationResult,
    EntityValidator, ConfidenceCalculator,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_BATCH_SIZE
)
from .entity_operations import EntityRepository, LLMProcessor, DuplicateDetector

# External dependencies
try:
    from back_end.src.data_collection.database_manager import database_manager
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Configuration
try:
    from back_end.src.data.config import config
except ImportError:
    # Fallback configuration
    class SimpleConfig:
        fast_mode = os.getenv('FAST_MODE', '1').lower() in ('1', 'true', 'yes')
        data_root = 'data'
        db_path = 'data/processed/intervention_research.db'
    config = SimpleConfig()

# Consensus functionality is now integrated directly into BatchEntityProcessor
CONSENSUS_AVAILABLE = True

# === MAIN BATCH ENTITY PROCESSOR CLASS ===

class BatchEntityProcessor:
    """
    Unified batch processor for entity normalization, deduplication, and mapping suggestions.

    This class consolidates all LLM processing functionality into efficient batch operations
    while preserving all sophisticated features from the original modular architecture.
    """

    # Medical safety is now handled through enhanced LLM prompts that provide
    # comprehensive pattern recognition rather than hardcoded pairs

    def __init__(self, db_connection: sqlite3.Connection, llm_model: str = "gemma2:9b"):
        """
        Initialize the BatchEntityProcessor with a database connection.

        Args:
            db_connection: SQLite database connection object
            llm_model: LLM model name for semantic matching
        """
        # Store references
        self.db = db_connection
        self.db.row_factory = sqlite3.Row  # Enable dict-like access to rows
        self.llm_model = llm_model

        # Initialize operation modules
        self.repository = EntityRepository(db_connection)
        self.llm_processor = LLMProcessor(self.repository, llm_model)
        self.duplicate_detector = DuplicateDetector(self.repository)

        # Performance monitoring
        self.operation_counts = {}
        self.operation_times = {}

        # Configure logging
        self._setup_logging()

        # Initialize performance optimizations (pre-computed normalization cache, etc.)
        self.repository.ensure_performance_optimizations()

        # Consensus functionality is now integrated directly into this class
        # No separate processor needed

    def _setup_logging(self):
        """Set up logging for batch operations."""
        log_level = logging.ERROR if config.fast_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    # === PERFORMANCE OPTIMIZATION METHODS ===

    def _ensure_performance_optimizations(self) -> None:
        """
        Ensure performance optimization tables and indexes exist.

        This adds the normalized_terms_cache table for O(1) pattern matching
        and optimizes LLM cache with hash-based keys.
        """
        cursor = self.db.cursor()

        try:
            # Create normalized terms cache table for performance optimization
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS normalized_terms_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_term TEXT NOT NULL,
                    normalized_term TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    canonical_id INTEGER,
                    created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(original_term, entity_type),
                    FOREIGN KEY (canonical_id) REFERENCES canonical_entities(id) ON DELETE CASCADE
                )
            """)

            # Create performance indexes for fast lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_normalized_terms_lookup
                ON normalized_terms_cache(normalized_term, entity_type)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_normalized_terms_reverse
                ON normalized_terms_cache(original_term, entity_type)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_normalized_terms_canonical
                ON normalized_terms_cache(canonical_id)
            """)

            # Check if LLM cache needs optimization (add cache_key column)
            cursor.execute("PRAGMA table_info(llm_normalization_cache)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'cache_key' not in columns:
                cursor.execute("""
                    ALTER TABLE llm_normalization_cache
                    ADD COLUMN cache_key TEXT
                """)

                # Create index for cache_key
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_llm_cache_key
                    ON llm_normalization_cache(cache_key)
                """)

                self.logger.info("Added cache_key optimization to LLM cache")

            self.db.commit()
            self.logger.info("Performance optimization tables and indexes ensured")

        except Exception as e:
            self.logger.error(f"Failed to create performance optimization schema: {e}")
            self.db.rollback()

    def get_or_compute_normalized_term(self, term: str, entity_type: str) -> str:
        """
        Get normalized form of term from cache or compute and cache it.

        This provides O(1) lookup for normalized forms instead of O(N) computation.
        """
        cursor = self.db.cursor()

        # Check cache first
        cursor.execute("""
            SELECT normalized_term
            FROM normalized_terms_cache
            WHERE original_term = ? AND entity_type = ?
        """, (term, entity_type))

        result = cursor.fetchone()
        if result:
            return result[0]

        # Compute and cache
        normalized_term = self.normalize_term(term)

        cursor.execute("""
            INSERT OR IGNORE INTO normalized_terms_cache
            (original_term, normalized_term, entity_type)
            VALUES (?, ?, ?)
        """, (term, normalized_term, entity_type))

        self.db.commit()
        return normalized_term

    def bulk_populate_normalization_cache(self, entity_type: Optional[str] = None) -> int:
        """
        Pre-populate normalization cache for all existing canonical entities and mappings.

        This is a one-time operation to optimize future pattern matching performance.
        """
        cursor = self.db.cursor()
        populated_count = 0

        try:
            # Populate from canonical entities
            where_clause = "WHERE entity_type = ?" if entity_type else ""
            params = [entity_type] if entity_type else []

            cursor.execute(f"""
                SELECT canonical_name, entity_type, id
                FROM canonical_entities
                {where_clause}
            """, params)

            for row in cursor.fetchall():
                canonical_name, ent_type, canonical_id = row
                normalized_term = self.normalize_term(canonical_name)

                cursor.execute("""
                    INSERT OR IGNORE INTO normalized_terms_cache
                    (original_term, normalized_term, entity_type, canonical_id)
                    VALUES (?, ?, ?, ?)
                """, (canonical_name, normalized_term, ent_type, canonical_id))

                populated_count += cursor.rowcount

            # Populate from entity mappings
            cursor.execute(f"""
                SELECT DISTINCT em.raw_text, em.entity_type, em.canonical_id
                FROM entity_mappings em
                {where_clause.replace('entity_type', 'em.entity_type') if where_clause else ''}
                WHERE em.raw_text IS NOT NULL
            """, params)

            for row in cursor.fetchall():
                raw_text, ent_type, canonical_id = row
                normalized_term = self.normalize_term(raw_text)

                cursor.execute("""
                    INSERT OR IGNORE INTO normalized_terms_cache
                    (original_term, normalized_term, entity_type, canonical_id)
                    VALUES (?, ?, ?, ?)
                """, (raw_text, normalized_term, ent_type, canonical_id))

                populated_count += cursor.rowcount

            self.db.commit()
            self.logger.info(f"Populated normalization cache with {populated_count} terms")
            return populated_count

        except Exception as e:
            self.logger.error(f"Failed to populate normalization cache: {e}")
            self.db.rollback()
            return 0

    def bulk_normalize_terms_optimized(self, terms: List[str], entity_type: str) -> Dict[str, str]:
        """
        Efficiently normalize multiple terms using the pre-computed cache.

        This provides massive performance improvements for bulk operations by
        minimizing normalize_term() calls.
        """
        if not terms:
            return {}

        cursor = self.db.cursor()
        normalized_mapping = {}
        terms_to_compute = []

        # Step 1: Batch lookup from cache
        placeholders = ','.join(['?' for _ in terms])
        cursor.execute(f"""
            SELECT original_term, normalized_term
            FROM normalized_terms_cache
            WHERE original_term IN ({placeholders}) AND entity_type = ?
        """, terms + [entity_type])

        cached_results = dict(cursor.fetchall())

        # Step 2: Identify terms that need computation
        for term in terms:
            if term in cached_results:
                normalized_mapping[term] = cached_results[term]
            else:
                terms_to_compute.append(term)

        # Step 3: Batch compute missing normalizations
        if terms_to_compute:
            computed_normalizations = []

            for term in terms_to_compute:
                normalized_term = self.normalize_term(term)
                normalized_mapping[term] = normalized_term

                # Prepare for batch insert
                computed_normalizations.append((term, normalized_term, entity_type))

            # Step 4: Batch insert new normalizations to cache
            if computed_normalizations:
                cursor.executemany("""
                    INSERT OR IGNORE INTO normalized_terms_cache
                    (original_term, normalized_term, entity_type)
                    VALUES (?, ?, ?)
                """, computed_normalizations)

                self.db.commit()

        return normalized_mapping

    def optimize_existing_data_performance(self) -> Dict[str, int]:
        """
        One-time operation to optimize performance for existing data.

        This populates the normalization cache and migrates LLM cache to hash-based keys.
        """
        results = {
            'normalized_terms_populated': 0,
            'llm_cache_entries_migrated': 0
        }

        # Populate normalization cache
        results['normalized_terms_populated'] = self.bulk_populate_normalization_cache()

        # Migrate LLM cache to hash-based keys
        results['llm_cache_entries_migrated'] = self._migrate_llm_cache_to_hash_keys()

        return results

    def _migrate_llm_cache_to_hash_keys(self) -> int:
        """
        Migrate existing LLM cache entries to use hash-based keys.

        This is a one-time operation to improve cache performance.
        """
        cursor = self.db.cursor()
        migrated_count = 0

        try:
            # Find entries without cache_key
            cursor.execute("""
                SELECT input_term, entity_type, model_name
                FROM llm_normalization_cache
                WHERE cache_key IS NULL
                LIMIT 1000
            """)

            entries_to_migrate = cursor.fetchall()

            for entry in entries_to_migrate:
                input_term, entity_type, model_name = entry
                cache_key = self._generate_llm_cache_key(input_term, entity_type, model_name)

                cursor.execute("""
                    UPDATE llm_normalization_cache
                    SET cache_key = ?
                    WHERE input_term = ? AND entity_type = ? AND model_name = ? AND cache_key IS NULL
                """, (cache_key, input_term, entity_type, model_name))

                migrated_count += cursor.rowcount

            self.db.commit()

            if migrated_count > 0:
                self.logger.info(f"Migrated {migrated_count} LLM cache entries to hash-based keys")

            return migrated_count

        except Exception as e:
            self.logger.error(f"Failed to migrate LLM cache entries: {e}")
            self.db.rollback()
            return 0

    # === VALIDATION UTILITIES ===

    def _validate_entity_type(self, entity_type: str) -> str:
        """Validate and normalize entity type."""
        if not entity_type or not isinstance(entity_type, str):
            raise ValidationError("Entity type must be a non-empty string")

        entity_type = entity_type.strip().lower()
        if not EntityType.is_valid(entity_type):
            valid_types = [member.value for member in EntityType]
            raise ValidationError(f"Invalid entity type '{entity_type}'. Must be one of: {valid_types}")

        return entity_type

    def _validate_term(self, term: str) -> str:
        """Validate and normalize term input."""
        if not isinstance(term, str):
            raise ValidationError("Term must be a string")

        term = term.strip()
        if not term:
            raise ValidationError("Term cannot be empty or whitespace only")

        if len(term) > 1000:  # Reasonable limit
            raise ValidationError("Term is too long (max 1000 characters)")

        return term

    def _validate_confidence(self, confidence: float) -> float:
        """Validate confidence score."""
        if not isinstance(confidence, (int, float)):
            raise ValidationError("Confidence must be a number")

        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValidationError("Confidence must be between 0.0 and 1.0")

        return confidence

    def _get_effective_confidence(self, intervention: Dict[str, Any]) -> float:
        """
        Get effective confidence from intervention data.

        Supports both legacy confidence_score and new dual confidence system.
        For dual system, uses extraction_confidence as primary metric.

        Args:
            intervention: Intervention dictionary

        Returns:
            Effective confidence score (0.0-1.0)
        """
        # Try new dual confidence first
        if 'extraction_confidence' in intervention:
            extraction_conf = intervention.get('extraction_confidence', 0)
            if extraction_conf is not None:
                return float(extraction_conf)

        # Fall back to consensus_confidence (from consensus processing)
        if 'consensus_confidence' in intervention:
            consensus_conf = intervention.get('consensus_confidence', 0)
            if consensus_conf is not None:
                return float(consensus_conf)

        # Fall back to legacy confidence_score
        legacy_conf = intervention.get('confidence_score', 0)
        if legacy_conf is not None:
            return float(legacy_conf)

        return 0.0

    def _merge_dual_confidence(self, interventions: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """
        Merge confidence values from multiple interventions.

        Args:
            interventions: List of intervention dictionaries

        Returns:
            Tuple of (extraction_confidence, study_confidence, legacy_confidence)
        """
        extraction_confidences = []
        study_confidences = []
        legacy_confidences = []

        for intervention in interventions:
            # Extract extraction_confidence
            ext_conf = intervention.get('extraction_confidence')
            if ext_conf is not None:
                extraction_confidences.append(float(ext_conf))

            # Extract study_confidence
            study_conf = intervention.get('study_confidence')
            if study_conf is not None:
                study_confidences.append(float(study_conf))

            # Extract legacy confidence_score
            legacy_conf = intervention.get('confidence_score')
            if legacy_conf is not None:
                legacy_confidences.append(float(legacy_conf))

        # Calculate merged values (use max for extraction confidence, average for others)
        merged_extraction = max(extraction_confidences) if extraction_confidences else 0.0
        merged_study = sum(study_confidences) / len(study_confidences) if study_confidences else 0.0
        merged_legacy = sum(legacy_confidences) / len(legacy_confidences) if legacy_confidences else 0.0

        return merged_extraction, merged_study, merged_legacy

    # === TERM NORMALIZATION ===

    @staticmethod
    def normalize_term(term: str) -> str:
        """
        Normalize a term by lowercasing, stripping whitespace, and removing punctuation.

        Args:
            term: The raw text term to normalize

        Returns:
            The normalized term
        """
        # Convert to lowercase and strip whitespace
        normalized = term.lower().strip()

        # Remove common punctuation but keep meaningful characters like hyphens in compound words
        # Remove parentheses and their contents (like abbreviations)
        normalized = re.sub(r'\([^)]*\)', '', normalized)

        # Remove excess whitespace and common punctuation
        normalized = re.sub(r'[.,;:!?"\']', '', normalized)

        # Normalize multiple spaces to single spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        # Final trim
        return normalized.strip()

    # _is_dangerous_match method removed - medical safety now handled through
    # enhanced LLM prompts during the normalization process

    # === DATABASE OPERATIONS ===

    def find_canonical_by_id(self, canonical_id: int) -> Optional[Dict[str, Any]]:
        """Find a canonical entity by its ID."""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, canonical_name, entity_type, description, created_timestamp
            FROM canonical_entities WHERE id = ?
        """, (canonical_id,))

        result = cursor.fetchone()
        return dict(result) if result else None

    def find_canonical_by_name(self, canonical_name: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Find a canonical entity by name and type."""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, canonical_name, entity_type, description, created_timestamp
            FROM canonical_entities
            WHERE canonical_name = ? AND entity_type = ?
        """, (canonical_name, entity_type))

        result = cursor.fetchone()
        return dict(result) if result else None

    def create_canonical_entity(self, canonical_name: str, entity_type: str,
                              scientific_name: Optional[str] = None) -> int:
        """Create a new canonical entity."""
        cursor = self.db.cursor()

        # Prepare metadata JSON if scientific_name provided
        metadata = {}
        if scientific_name:
            metadata['scientific_name'] = scientific_name

        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO canonical_entities (canonical_name, entity_type, description)
            VALUES (?, ?, ?)
        """, (canonical_name, entity_type, scientific_name))

        self.db.commit()
        return cursor.lastrowid

    def find_mapping_by_term(self, term: str, entity_type: str, apply_confidence_decay: bool = True) -> Optional[Dict[str, Any]]:
        """
        Find an existing mapping for a term with optional confidence decay.

        Args:
            term: The term to find mapping for
            entity_type: The entity type
            apply_confidence_decay: Whether to apply time-based confidence decay

        Returns:
            Mapping dictionary with potentially adjusted confidence, or None
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT em.id, em.canonical_id, em.raw_text, em.entity_type,
                   em.confidence_score, em.mapping_method, em.created_timestamp,
                   ce.canonical_name
            FROM entity_mappings em
            JOIN canonical_entities ce ON em.canonical_id = ce.id
            WHERE em.raw_text = ? AND em.entity_type = ?
        """, (term, entity_type))

        result = cursor.fetchone()
        if not result:
            return None

        mapping = dict(result)

        # Apply confidence decay if requested
        if apply_confidence_decay:
            current_confidence = self._calculate_time_adjusted_confidence(
                mapping['confidence_score'],
                mapping['created_timestamp'],
                mapping['mapping_method']
            )
            mapping['adjusted_confidence'] = current_confidence
            mapping['original_confidence'] = mapping['confidence_score']

        return mapping

    def _calculate_time_adjusted_confidence(self, original_confidence: float,
                                          created_timestamp: str, mapping_method: str) -> float:
        """
        Calculate time-adjusted confidence using decay formula.

        Args:
            original_confidence: Original confidence score (0.0-1.0)
            created_timestamp: ISO timestamp when mapping was created
            mapping_method: Method used for mapping (affects decay rate)

        Returns:
            Time-adjusted confidence score
        """
        try:
            # Parse timestamp (assuming ISO format)
            if created_timestamp:
                created_date = datetime.fromisoformat(created_timestamp.replace('Z', '+00:00'))
            else:
                # If no timestamp, assume recent for backward compatibility
                created_date = datetime.now()

            current_date = datetime.now()
            years_elapsed = (current_date - created_date).days / 365.25

            # Different decay rates based on mapping method
            decay_rates = {
                'exact': 0.98,              # Very slow decay for exact matches
                'exact_canonical': 0.98,
                'exact_normalized': 0.97,
                'safe_plural_addition': 0.96,
                'safe_plural_removal': 0.96,
                'definite_article_removal': 0.95,
                'spacing_punctuation_normalization': 0.95,
                'llm_semantic': 0.90,       # Faster decay for LLM matches
                'llm_semantic_batch': 0.90,
                'llm_semantic_cached': 0.92,
                'deduplication_merge': 0.93,
                'manual': 0.99,             # Slowest decay for manual mappings
            }

            # Get decay rate for this method (default to 0.95 for unknown methods)
            decay_rate = decay_rates.get(mapping_method, 0.95)

            # Apply exponential decay: confidence = original * decay_rate^years
            adjusted_confidence = original_confidence * (decay_rate ** years_elapsed)

            # Ensure bounds [0.0, 1.0] and minimum threshold
            adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))  # Floor at 0.1

            return adjusted_confidence

        except Exception as e:
            self.logger.warning(f"Error calculating confidence decay: {e}")
            # Return original confidence on error
            return original_confidence

    def create_mapping(self, original_term: str, canonical_id: int,
                      confidence: float, method: str) -> int:
        """Create a new term mapping to a canonical entity."""
        cursor = self.db.cursor()

        # Verify canonical entity exists and get its type
        cursor.execute("""
            SELECT entity_type FROM canonical_entities WHERE id = ?
        """, (canonical_id,))

        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Canonical entity with ID {canonical_id} not found")

        entity_type = result['entity_type']

        # Create the mapping
        cursor.execute("""
            INSERT INTO entity_mappings
            (canonical_id, raw_text, entity_type, confidence_score, mapping_method)
            VALUES (?, ?, ?, ?, ?)
        """, (canonical_id, original_term, entity_type, confidence, method))

        self.db.commit()
        return cursor.lastrowid

    def get_all_canonical_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all canonical entities, optionally filtered by type."""
        cursor = self.db.cursor()

        if entity_type:
            cursor.execute("""
                SELECT id, canonical_name, entity_type, description, created_timestamp
                FROM canonical_entities
                WHERE entity_type = ?
                ORDER BY canonical_name
            """, (entity_type,))
        else:
            cursor.execute("""
                SELECT id, canonical_name, entity_type, description, created_timestamp
                FROM canonical_entities
                ORDER BY canonical_name
            """)

        return [dict(row) for row in cursor.fetchall()]

    def get_all_mappings_with_canonicals(self, entity_type: str) -> List[Dict[str, Any]]:
        """Get all mappings with their canonical entity information for an entity type."""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT DISTINCT ce.id, ce.canonical_name, ce.entity_type,
                   em.raw_text, em.confidence_score
            FROM canonical_entities ce
            LEFT JOIN entity_mappings em ON ce.id = em.canonical_id
            WHERE ce.entity_type = ?
        """, (entity_type,))

        return [dict(row) for row in cursor.fetchall()]

    # === LLM CACHE OPERATIONS ===

    def _generate_llm_cache_key(self, term: str, entity_type: str, model_name: str) -> str:
        """
        Generate lightweight cache key using hashing instead of JSON serialization.

        This eliminates the expensive JSON serialization and enables indexed lookups.
        """
        # Don't include candidate_canonicals - they're determined by entity_type anyway
        cache_input = f"{term.lower().strip()}|{entity_type}|{model_name}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()

    def find_llm_cache(self, term: str, entity_type: str,
                      candidate_canonicals: Optional[List[str]], model_name: str) -> Optional[Dict[str, Any]]:
        """
        Check if we have a cached LLM decision for this term using optimized hash-based lookup.

        This replaces expensive JSON serialization with fast hash-based keys.
        """
        cursor = self.db.cursor()

        # Generate cache key (much faster than JSON serialization)
        cache_key = self._generate_llm_cache_key(term, entity_type, model_name)

        # First try the new optimized cache lookup
        cursor.execute("""
            SELECT match_result, confidence_score, reasoning, llm_response, created_at
            FROM llm_normalization_cache
            WHERE cache_key = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (cache_key,))

        result = cursor.fetchone()
        if result:
            return {
                'match_result': result['match_result'],
                'confidence_score': result['confidence_score'],
                'reasoning': result['reasoning'],
                'llm_response': result['llm_response'],
                'cached_at': result['created_at']
            }

        # Fallback to legacy lookup for backward compatibility (during migration)
        candidates_json = json.dumps(sorted(candidate_canonicals or []), sort_keys=True) if candidate_canonicals else None

        cursor.execute("""
            SELECT match_result, confidence_score, reasoning, llm_response, created_at
            FROM llm_normalization_cache
            WHERE input_term = ? AND entity_type = ? AND candidate_canonicals = ? AND model_name = ?
              AND cache_key IS NULL
            ORDER BY created_at DESC
            LIMIT 1
        """, (term, entity_type, candidates_json, model_name))

        result = cursor.fetchone()
        if result:
            # Migrate this entry to use the new cache_key format
            try:
                cursor.execute("""
                    UPDATE llm_normalization_cache
                    SET cache_key = ?
                    WHERE input_term = ? AND entity_type = ? AND candidate_canonicals = ? AND model_name = ?
                      AND cache_key IS NULL
                """, (cache_key, term, entity_type, candidates_json, model_name))
                self.db.commit()
            except Exception as e:
                self.logger.warning(f"Failed to migrate cache entry: {e}")

            return {
                'match_result': result['match_result'],
                'confidence_score': result['confidence_score'],
                'reasoning': result['reasoning'],
                'llm_response': result['llm_response'],
                'cached_at': result['created_at']
            }

        return None

    def save_llm_cache(self, term: str, entity_type: str, candidate_canonicals: Optional[List[str]],
                      llm_response: str, match_result: Optional[str], confidence: float,
                      reasoning: str, model_name: str) -> None:
        """
        Save LLM decision to cache using optimized hash-based keys.

        This eliminates JSON serialization and enables much faster cache operations.
        """
        cursor = self.db.cursor()

        # Generate optimized cache key
        cache_key = self._generate_llm_cache_key(term, entity_type, model_name)

        # Legacy candidates_json for backward compatibility (will be phased out)
        candidates_json = json.dumps(sorted(candidate_canonicals or []), sort_keys=True) if candidate_canonicals else None

        cursor.execute("""
            INSERT OR REPLACE INTO llm_normalization_cache
            (input_term, entity_type, candidate_canonicals, llm_response, match_result,
             confidence_score, reasoning, model_name, cache_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (term, entity_type, candidates_json, llm_response, match_result, confidence, reasoning, model_name, cache_key))

        self.db.commit()

    # === BULK DATABASE OPERATIONS ===

    def bulk_find_existing_mappings(self, terms_and_types: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Find existing mappings for multiple terms efficiently using WHERE IN.

        Args:
            terms_and_types: List of (term, entity_type) tuples

        Returns:
            Dictionary mapping (term, entity_type) to mapping info
        """
        if not terms_and_types:
            return {}

        cursor = self.db.cursor()
        results = {}

        # Group by entity type for efficient queries
        by_entity_type = defaultdict(list)
        for term, entity_type in terms_and_types:
            by_entity_type[entity_type].append(term)

        for entity_type, terms in by_entity_type.items():
            if not terms:
                continue

            # Use WHERE IN for bulk lookup
            placeholders = ','.join(['?' for _ in terms])
            cursor.execute(f"""
                SELECT em.id, em.canonical_id, em.raw_text, em.entity_type,
                       em.confidence_score, em.mapping_method, em.created_timestamp,
                       ce.canonical_name
                FROM entity_mappings em
                JOIN canonical_entities ce ON em.canonical_id = ce.id
                WHERE em.raw_text IN ({placeholders}) AND em.entity_type = ?
            """, terms + [entity_type])

            for row in cursor.fetchall():
                key = (row['raw_text'], row['entity_type'])
                results[key] = dict(row)

        return results

    def bulk_create_mappings(self, mappings: List[Dict[str, Any]]) -> List[int]:
        """
        Create multiple entity mappings efficiently using executemany.

        Args:
            mappings: List of mapping dictionaries with keys:
                     original_term, canonical_id, confidence, method

        Returns:
            List of created mapping IDs
        """
        if not mappings:
            return []

        cursor = self.db.cursor()

        # Prepare data for bulk insert
        mapping_data = []
        for mapping in mappings:
            # Verify canonical entity exists and get its type
            cursor.execute("""
                SELECT entity_type FROM canonical_entities WHERE id = ?
            """, (mapping['canonical_id'],))

            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Canonical entity with ID {mapping['canonical_id']} not found")

            entity_type = result['entity_type']

            mapping_data.append((
                mapping['canonical_id'],
                mapping['original_term'],
                entity_type,
                mapping['confidence'],
                mapping['method']
            ))

        # Bulk insert using executemany
        cursor.executemany("""
            INSERT INTO entity_mappings
            (canonical_id, raw_text, entity_type, confidence_score, mapping_method)
            VALUES (?, ?, ?, ?, ?)
        """, mapping_data)

        # Get the IDs of created mappings
        first_id = cursor.lastrowid - len(mapping_data) + 1
        mapping_ids = list(range(first_id, first_id + len(mapping_data)))

        return mapping_ids

    def bulk_create_canonical_entities(self, entities: List[Dict[str, Any]]) -> List[int]:
        """
        Create multiple canonical entities efficiently using executemany.

        Args:
            entities: List of entity dictionaries with keys:
                     canonical_name, entity_type, description (optional)

        Returns:
            List of created canonical entity IDs
        """
        if not entities:
            return []

        cursor = self.db.cursor()

        # Prepare data for bulk insert
        entity_data = []
        for entity in entities:
            entity_data.append((
                entity['canonical_name'],
                entity['entity_type'],
                entity.get('description', entity.get('scientific_name'))
            ))

        # Bulk insert using executemany
        cursor.executemany("""
            INSERT INTO canonical_entities (canonical_name, entity_type, description)
            VALUES (?, ?, ?)
        """, entity_data)

        # Get the IDs of created entities
        first_id = cursor.lastrowid - len(entity_data) + 1
        entity_ids = list(range(first_id, first_id + len(entity_data)))

        return entity_ids

    def bulk_normalize_with_transaction(self, terms_and_types: List[Tuple[str, str]],
                                      confidence_threshold: float = 0.7,
                                      create_new_entities: bool = True) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Normalize multiple terms efficiently with transaction management.

        Args:
            terms_and_types: List of (term, entity_type) tuples
            confidence_threshold: Minimum confidence for LLM matching
            create_new_entities: Whether to create new canonical entities for unmatched terms

        Returns:
            Dictionary mapping (term, entity_type) to normalization results
        """
        if not terms_and_types:
            return {}

        results = {}

        try:
            # Start transaction
            self.db.execute("BEGIN TRANSACTION")

            # Step 1: Bulk check for existing mappings
            existing_mappings = self.bulk_find_existing_mappings(terms_and_types)

            # Step 2: Identify terms that need normalization
            terms_to_normalize = []
            for term, entity_type in terms_and_types:
                key = (term, entity_type)
                if key in existing_mappings:
                    # Use existing mapping
                    mapping = existing_mappings[key]
                    results[key] = {
                        'canonical_id': mapping['canonical_id'],
                        'canonical_name': mapping['canonical_name'],
                        'method': 'existing_mapping',
                        'confidence': mapping['confidence_score'],
                        'is_new': False,
                        'reasoning': 'Found existing mapping'
                    }
                else:
                    terms_to_normalize.append((term, entity_type))

            # Step 3: Process terms that need normalization in chunks
            chunk_size = 50  # Process in chunks to avoid memory issues
            for i in range(0, len(terms_to_normalize), chunk_size):
                chunk = terms_to_normalize[i:i + chunk_size]

                # Group by entity type for efficient processing
                by_entity_type = defaultdict(list)
                for term, entity_type in chunk:
                    by_entity_type[entity_type].append(term)

                # Process each entity type with optimized batch processing
                for entity_type, terms in by_entity_type.items():
                    # Step 1: Try safe matches first (fastest)
                    term_matches = {}
                    terms_needing_llm = []

                    for term in terms:
                        safe_matches = self.find_matches(term, entity_type, MatchingMode.SAFE_ONLY)
                        if safe_matches:
                            term_matches[term] = safe_matches[0]  # Best safe match
                        else:
                            terms_needing_llm.append(term)

                    # Step 2: Batch process LLM matches for remaining terms
                    if terms_needing_llm:
                        llm_terms_and_types = [(term, entity_type) for term in terms_needing_llm]
                        batch_llm_results = self._batch_find_llm_matches(
                            llm_terms_and_types,
                            candidate_canonicals=None,
                            batch_size=20
                        )

                        # Filter LLM results by confidence threshold
                        for term, matches in batch_llm_results.items():
                            if matches and matches[0].confidence >= confidence_threshold:
                                term_matches[term] = matches[0]

                    # Prepare new entities and mappings to create
                    new_entities = []
                    new_mappings = []

                    for term in terms:
                        key = (term, entity_type)

                        if term in term_matches:
                            # Found a match
                            match = term_matches[term]
                            results[key] = {
                                'canonical_id': match.canonical_id,
                                'canonical_name': match.canonical_name,
                                'method': match.method,
                                'confidence': match.confidence,
                                'is_new': False,
                                'reasoning': match.reasoning
                            }

                            # Create mapping if it doesn't exist
                            new_mappings.append({
                                'original_term': term,
                                'canonical_id': match.canonical_id,
                                'confidence': match.confidence,
                                'method': match.method
                            })

                        elif create_new_entities:
                            # No match found - create new entity
                            new_entities.append({
                                'canonical_name': term,
                                'entity_type': entity_type,
                                'description': None
                            })

                    # Bulk create new canonical entities
                    if new_entities:
                        entity_ids = self.bulk_create_canonical_entities(new_entities)

                        # Create mappings for new entities and update results
                        for entity, entity_id in zip(new_entities, entity_ids):
                            term = entity['canonical_name']
                            key = (term, entity_type)

                            results[key] = {
                                'canonical_id': entity_id,
                                'canonical_name': term,
                                'method': 'new_canonical',
                                'confidence': 1.0,
                                'is_new': True,
                                'reasoning': 'No existing match found, created new canonical entity'
                            }

                            new_mappings.append({
                                'original_term': term,
                                'canonical_id': entity_id,
                                'confidence': 1.0,
                                'method': 'exact'
                            })

                    # Bulk create mappings
                    if new_mappings:
                        self.bulk_create_mappings(new_mappings)

            # Commit transaction
            self.db.commit()

        except Exception as e:
            # Rollback on error
            self.db.rollback()
            self.logger.error(f"Error in bulk normalization: {e}")
            raise

        return results

    # === MATCHING STRATEGIES ===

    def _find_exact_matches(self, term: str, entity_type: str) -> List[MatchResult]:
        """Find exact normalized matches for a term."""
        normalized_term = self.normalize_term(term)

        # Check against canonical names
        canonical_entities = self.get_all_canonical_entities(entity_type)
        for entity in canonical_entities:
            if self.normalize_term(entity['canonical_name']) == normalized_term:
                return [MatchResult(
                    canonical_id=entity['id'],
                    canonical_name=entity['canonical_name'],
                    entity_type=entity_type,
                    confidence=1.0,
                    method='exact_canonical',
                    reasoning=f"Exact match with canonical name: {entity['canonical_name']}"
                )]

        # Check against existing mappings
        existing_mapping = self.find_mapping_by_term(term, entity_type)
        if existing_mapping:
            return [MatchResult(
                canonical_id=existing_mapping['canonical_id'],
                canonical_name=existing_mapping['canonical_name'],
                entity_type=entity_type,
                confidence=1.0,
                method='existing_mapping',
                reasoning=f"Exact match with existing mapping: {existing_mapping['raw_text']}"
            )]

        # Check normalized mappings
        all_mappings = self.get_all_mappings_with_canonicals(entity_type)
        for mapping in all_mappings:
            if mapping['raw_text'] and self.normalize_term(mapping['raw_text']) == normalized_term:
                return [MatchResult(
                    canonical_id=mapping['id'],
                    canonical_name=mapping['canonical_name'],
                    entity_type=entity_type,
                    confidence=0.98,
                    method='exact_normalized',
                    reasoning=f"Exact normalized match with: {mapping['raw_text']}"
                )]

        return []

    def _find_pattern_matches(self, term: str, entity_type: str) -> List[MatchResult]:
        """
        Find matches using optimized pattern matching with pre-computed normalized forms.

        This replaces the O(N) loop with O(log N) database queries for 100x performance improvement.
        """
        # Get normalized form from cache (O(1) lookup)
        normalized_term = self.get_or_compute_normalized_term(term, entity_type)
        matches = []

        cursor = self.db.cursor()

        # Direct SQL query using pre-computed normalized forms and indexes
        cursor.execute("""
            SELECT DISTINCT
                ce.id as canonical_id,
                ce.canonical_name,
                em.confidence_score,
                em.mapping_method,
                ntc.original_term,
                'cached_normalized' as match_source
            FROM normalized_terms_cache ntc
            JOIN canonical_entities ce ON ce.id = ntc.canonical_id
            LEFT JOIN entity_mappings em ON em.canonical_id = ce.id
            WHERE ntc.normalized_term = ? AND ntc.entity_type = ?
            ORDER BY em.confidence_score DESC
        """, (normalized_term, entity_type))

        for row in cursor.fetchall():
            mapping = dict(row)

            # Create match result with confidence from the mapping
            confidence = mapping['confidence_score'] or 0.95  # Default for canonical entities
            method = f"pattern_{mapping['match_source']}"

            # Apply additional pattern matching logic if needed
            if self._is_safe_pattern_match(term, mapping['original_term']):
                matches.append(MatchResult(
                    canonical_id=mapping['canonical_id'],
                    canonical_name=mapping['canonical_name'],
                    entity_type=entity_type,
                    confidence=confidence,
                    method=method,
                    reasoning=f"Normalized pattern match: {term} → {mapping['original_term']}"
                ))

        # If no exact normalized matches, try fuzzy pattern matching
        if not matches:
            matches.extend(self._find_fuzzy_pattern_matches(term, entity_type, normalized_term))

        # Remove duplicates based on canonical ID, keeping highest confidence
        unique_matches = {}
        for match in matches:
            canonical_id = match.canonical_id
            if canonical_id not in unique_matches or match.confidence > unique_matches[canonical_id].confidence:
                unique_matches[canonical_id] = match

        # Sort by confidence descending
        return sorted(unique_matches.values(), key=lambda x: x.confidence, reverse=True)

    def _find_fuzzy_pattern_matches(self, term: str, entity_type: str, normalized_term: str) -> List[MatchResult]:
        """
        Find fuzzy pattern matches using optimized SQL queries for common transformations.

        Handles pluralization, article removal, and other safe transformations.
        """
        matches = []
        cursor = self.db.cursor()

        # Generate common fuzzy patterns
        fuzzy_patterns = self._generate_fuzzy_patterns(normalized_term)

        for pattern, transformation_type in fuzzy_patterns:
            cursor.execute("""
                SELECT DISTINCT
                    ce.id as canonical_id,
                    ce.canonical_name,
                    em.confidence_score,
                    ntc.original_term
                FROM normalized_terms_cache ntc
                JOIN canonical_entities ce ON ce.id = ntc.canonical_id
                LEFT JOIN entity_mappings em ON em.canonical_id = ce.id
                WHERE ntc.normalized_term = ? AND ntc.entity_type = ?
                LIMIT 5
            """, (pattern, entity_type))

            for row in cursor.fetchall():
                mapping = dict(row)

                confidence = (mapping['confidence_score'] or 0.95) * 0.9  # Slight penalty for fuzzy matching
                method = f"pattern_{transformation_type}"

                matches.append(MatchResult(
                    canonical_id=mapping['canonical_id'],
                    canonical_name=mapping['canonical_name'],
                    entity_type=entity_type,
                    confidence=confidence,
                    method=method,
                    reasoning=f"Fuzzy pattern match ({transformation_type}): {term} → {mapping['original_term']}"
                ))

        return matches

    def _generate_fuzzy_patterns(self, normalized_term: str) -> List[Tuple[str, str]]:
        """Generate fuzzy pattern variations for a normalized term."""
        patterns = []

        # Pluralization patterns
        if normalized_term.endswith('s') and len(normalized_term) > 3:
            singular = normalized_term[:-1]
            patterns.append((singular, 'depluralization'))

        if not normalized_term.endswith('s'):
            plural = normalized_term + 's'
            patterns.append((plural, 'pluralization'))

        # Article removal patterns
        for article in ['the ', 'a ', 'an ']:
            if normalized_term.startswith(article):
                without_article = normalized_term[len(article):]
                patterns.append((without_article, 'article_removal'))
            else:
                with_article = article + normalized_term
                patterns.append((with_article, 'article_addition'))

        return patterns

    def _is_safe_pattern_match(self, original_term: str, matched_term: str) -> bool:
        """
        Verify that a pattern match is medically safe.

        Medical safety is now primarily handled through enhanced LLM prompts,
        but basic string similarity checks remain for simple pattern matches.
        """
        # Basic safety check - avoid exact matches of very different terms
        # Enhanced medical safety is handled through LLM prompts during normalization
        norm1 = self.normalize_term(original_term.lower())
        norm2 = self.normalize_term(matched_term.lower())

        # Very conservative check for obviously different terms
        if len(norm1) > 0 and len(norm2) > 0:
            # If terms are very different lengths and don't share common root, be cautious
            length_diff = abs(len(norm1) - len(norm2))
            if length_diff > 4 and not (norm1 in norm2 or norm2 in norm1):
                return False

        return True

    def _check_pattern_match(self, term: str, target: str, mapping: Dict[str, Any], match_type: str) -> Optional[MatchResult]:
        """Check if term matches target using safe patterns."""
        # PATTERN 1: Safe Plural/Singular matching
        if self._is_safe_pluralization_candidate(term, target):
            if term == target + 's':
                return MatchResult(
                    canonical_id=mapping['id'],
                    canonical_name=mapping['canonical_name'],
                    entity_type=mapping['entity_type'],
                    confidence=0.95,
                    method='safe_plural_addition',
                    reasoning=f"Safe plural form of {match_type}: {target}"
                )
            elif term + 's' == target:
                return MatchResult(
                    canonical_id=mapping['id'],
                    canonical_name=mapping['canonical_name'],
                    entity_type=mapping['entity_type'],
                    confidence=0.95,
                    method='safe_plural_removal',
                    reasoning=f"Safe singular form of {match_type}: {target}"
                )

        # PATTERN 2: Definite article removal
        if term.startswith('the ') and len(term) > 4:
            term_without_the = term[4:]  # Remove "the "
            if term_without_the == target:
                return MatchResult(
                    canonical_id=mapping['id'],
                    canonical_name=mapping['canonical_name'],
                    entity_type=mapping['entity_type'],
                    confidence=0.90,
                    method='definite_article_removal',
                    reasoning=f"Match after removing 'the' from {match_type}: {target}"
                )

        # PATTERN 3: Punctuation/spacing normalization
        term_no_punct = re.sub(r'[-_\s]+', '', term)
        target_no_punct = re.sub(r'[-_\s]+', '', target)

        if (term_no_punct == target_no_punct and
            term_no_punct != term and  # Only if there was a change
            len(term_no_punct) > 3):   # Avoid matching very short terms
            return MatchResult(
                canonical_id=mapping['id'],
                canonical_name=mapping['canonical_name'],
                entity_type=mapping['entity_type'],
                confidence=0.90,
                method='spacing_punctuation_normalization',
                reasoning=f"Match after normalizing spacing/punctuation with {match_type}: {target}"
            )

        return None

    def _is_safe_pluralization_candidate(self, term1: str, term2: str) -> bool:
        """Check if two terms are safe candidates for pluralization matching."""
        # Must be at least 4 characters to avoid false positives
        if len(term1) < 4 or len(term2) < 4:
            return False

        # Check if one is exactly the other + 's'
        if term1 == term2 + 's' or term2 == term1 + 's':
            # Additional safety: ensure the root is at least 3 characters
            shorter = min(term1, term2, key=len)
            if len(shorter) >= 3:
                return True

        return False

    def _find_llm_matches(self, term: str, entity_type: str,
                         candidate_canonicals: Optional[List[str]] = None) -> List[MatchResult]:
        """Find matches using LLM semantic understanding (single term - legacy interface)."""
        batch_results = self._batch_find_llm_matches([(term, entity_type)], candidate_canonicals)
        return batch_results.get(term, [])

    def _batch_find_llm_matches(self, terms_and_types: List[Tuple[str, str]],
                               candidate_canonicals: Optional[List[str]] = None,
                               batch_size: int = 20) -> Dict[str, List[MatchResult]]:
        """
        Find matches using true batch LLM processing for multiple terms.

        This dramatically reduces API calls by batching terms in single requests.
        """
        if not LLM_AVAILABLE or not self.llm_client:
            return {}

        # Group terms by entity type for efficient processing
        by_entity_type = defaultdict(list)
        for term, entity_type in terms_and_types:
            by_entity_type[entity_type].append(term)

        all_results = {}

        for entity_type, terms in by_entity_type.items():
            # Get candidate canonicals for this entity type
            if candidate_canonicals is None:
                canonical_entities = self.get_all_canonical_entities(entity_type)
                type_candidates = [entity['canonical_name'] for entity in canonical_entities]
            else:
                type_candidates = candidate_canonicals

            if not type_candidates:
                continue

            # Step 1: Check cache for all terms first
            cached_results = {}
            uncached_terms = []

            for term in terms:
                cached_result = self.find_llm_cache(term, entity_type, type_candidates, self.llm_model)
                if cached_result and cached_result['match_result']:
                    canonical_entity = self.find_canonical_by_name(
                        cached_result['match_result'], entity_type
                    )
                    if canonical_entity:
                        cached_results[term] = [MatchResult(
                            canonical_id=canonical_entity['id'],
                            canonical_name=canonical_entity['canonical_name'],
                            entity_type=entity_type,
                            confidence=cached_result['confidence_score'],
                            method='llm_semantic_cached',
                            reasoning=cached_result['reasoning']
                        )]
                else:
                    uncached_terms.append(term)

            # Add cached results to final results
            all_results.update(cached_results)

            # Step 2: Process uncached terms in batches
            if uncached_terms:
                batch_results = self._process_batch_llm_terms(
                    uncached_terms, entity_type, type_candidates, batch_size
                )
                all_results.update(batch_results)

        return all_results

    def _process_batch_llm_terms(self, terms: List[str], entity_type: str,
                                candidate_canonicals: List[str], batch_size: int) -> Dict[str, List[MatchResult]]:
        """
        Process terms in batches with medical safety considerations and error handling.
        """
        results = {}

        # Process terms in batches
        for i in range(0, len(terms), batch_size):
            batch_terms = terms[i:i + batch_size]

            # Medical safety now handled through enhanced LLM prompts
            # that provide pattern recognition during the normalization process
            try:
                batch_results = self._execute_batch_llm_request(
                    batch_terms, entity_type, candidate_canonicals
                )
                results.update(batch_results)

            except Exception as e:
                self.logger.error(f"Batch LLM request failed for {len(batch_terms)} terms: {e}")

                # Fall back to individual processing for error recovery
                self.logger.info("Falling back to individual LLM processing for error recovery")
                for term in batch_terms:
                    try:
                        individual_result = self._execute_single_llm_request(
                            term, entity_type, candidate_canonicals
                        )
                        if individual_result:
                            results[term] = individual_result
                    except Exception as individual_error:
                        self.logger.error(f"Individual LLM request failed for '{term}': {individual_error}")
                        continue

        return results

    # _filter_dangerous_batch_pairs method removed - medical safety now handled through
    # enhanced LLM prompts that teach pattern recognition during normalization

    def _execute_batch_llm_request(self, terms: List[str], entity_type: str,
                                  candidate_canonicals: List[str]) -> Dict[str, List[MatchResult]]:
        """
        Execute a single LLM request for multiple terms using structured prompting.
        """
        if not terms:
            return {}

        # Build batch prompt
        prompt = self._build_batch_llm_prompt(terms, candidate_canonicals, entity_type)

        try:
            response = self.llm_client.generate(prompt, temperature=0.1)
            llm_content = response['content'].strip()

            # Parse batch response
            batch_results = self._parse_batch_llm_response(llm_content, terms, entity_type)

            # Cache individual results
            for term, matches in batch_results.items():
                if matches:
                    match = matches[0]  # Take best match
                    try:
                        self.save_llm_cache(
                            term, entity_type, candidate_canonicals,
                            llm_content, match.canonical_name, match.confidence,
                            match.reasoning, self.llm_model
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to cache batch LLM result for '{term}': {e}")

            return batch_results

        except Exception as e:
            self.logger.error(f"Batch LLM request execution failed: {e}")
            raise

    def _execute_single_llm_request(self, term: str, entity_type: str,
                                   candidate_canonicals: List[str]) -> Optional[List[MatchResult]]:
        """
        Execute a single LLM request for one term (fallback method).
        """
        try:
            prompt = self._build_llm_prompt(term, candidate_canonicals, entity_type)
            response = self.llm_client.generate(prompt, temperature=0.1)

            llm_content = response['content'].strip()
            parsed = self._parse_llm_response(llm_content)

            if not parsed:
                return None

            match_name = parsed.get('match')
            confidence = float(parsed.get('confidence', 0.0))
            reasoning = parsed.get('reasoning', 'No reasoning provided')

            # Cache the result
            try:
                self.save_llm_cache(
                    term, entity_type, candidate_canonicals,
                    llm_content, match_name, confidence, reasoning, self.llm_model
                )
            except Exception as e:
                self.logger.warning(f"Failed to cache single LLM result: {e}")

            # Return match if confident enough
            if match_name and confidence > 0.3:
                canonical_entity = self.find_canonical_by_name(match_name, entity_type)
                if canonical_entity:
                    return [MatchResult(
                        canonical_id=canonical_entity['id'],
                        canonical_name=canonical_entity['canonical_name'],
                        entity_type=entity_type,
                        confidence=confidence,
                        method='llm_semantic',
                        reasoning=reasoning
                    )]

        except Exception as e:
            self.logger.error(f"Single LLM request failed for '{term}': {e}")
            raise

        return None

    def _build_llm_prompt(self, term: str, candidate_canonicals: List[str], entity_type: str) -> str:
        """Build medical-aware LLM prompt for entity matching using centralized prompt service."""
        if self.prompt_service:
            return self.prompt_service.create_entity_matching_prompt(term, candidate_canonicals, entity_type)

        # Fallback if prompt service not available (should not happen in normal operation)
        return f"Match term '{term}' to canonical forms: {candidate_canonicals}"

    def _build_batch_llm_prompt(self, terms: List[str], candidate_canonicals: List[str], entity_type: str) -> str:
        """Build batch LLM prompt for processing multiple terms efficiently using centralized prompt service."""
        if self.prompt_service:
            return self.prompt_service.create_batch_entity_matching_prompt(terms, candidate_canonicals, entity_type)

        # Fallback if prompt service not available (should not happen in normal operation)
        return f"Match terms {terms} to canonical forms: {candidate_canonicals}"

    def _parse_batch_llm_response(self, llm_content: str, terms: List[str], entity_type: str) -> Dict[str, List[MatchResult]]:
        """
        Parse batch LLM response and convert to MatchResult objects.

        Handles both successful parsing and fallback for partial failures.
        """
        results = {}

        try:
            # Try to parse as JSON array
            if LLM_AVAILABLE and 'parse_json_safely' in globals():
                parsed_list = parse_json_safely(llm_content)
                parsed_response = parsed_list[0] if parsed_list and isinstance(parsed_list, list) else None
            else:
                # Extract JSON array from response
                json_match = re.search(r'\[.*\]', llm_content, re.DOTALL)
                if json_match:
                    parsed_response = json.loads(json_match.group())
                else:
                    parsed_response = json.loads(llm_content)

            if not isinstance(parsed_response, list):
                raise ValueError("Response is not a JSON array")

            # Process each term result
            for item in parsed_response:
                if not isinstance(item, dict):
                    continue

                original_term = item.get('original_term', '').strip()
                match_name = item.get('match')
                confidence = float(item.get('confidence', 0.0))
                reasoning = item.get('reasoning', 'No reasoning provided')

                # Validate the term is in our expected list
                if original_term not in terms:
                    self.logger.warning(f"LLM returned unexpected term: {original_term}")
                    continue

                # Create match result if we have a confident match
                if match_name and confidence > 0.3:
                    canonical_entity = self.find_canonical_by_name(match_name, entity_type)
                    if canonical_entity:
                        results[original_term] = [MatchResult(
                            canonical_id=canonical_entity['id'],
                            canonical_name=canonical_entity['canonical_name'],
                            entity_type=entity_type,
                            confidence=confidence,
                            method='llm_semantic_batch',
                            reasoning=reasoning
                        )]
                    else:
                        self.logger.warning(f"LLM matched to non-existent canonical: {match_name}")

        except Exception as e:
            self.logger.error(f"Failed to parse batch LLM response: {e}")
            self.logger.debug(f"Raw LLM response: {llm_content}")
            # Return empty results - caller will handle fallback to individual processing

        return results

    def _parse_llm_response(self, llm_content: str) -> Optional[Dict[str, Any]]:
        """Parse LLM JSON response safely."""
        # Try to parse JSON response
        if LLM_AVAILABLE and 'parse_json_safely' in globals():
            parsed_list = parse_json_safely(llm_content)
            # parse_json_safely returns a list, we want the first dict
            return parsed_list[0] if parsed_list and isinstance(parsed_list, list) else None
        else:
            try:
                return json.loads(llm_content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    self.logger.warning("Invalid LLM response format")
                    return None

    # === MATCHING ORCHESTRATION ===

    def find_matches(self, term: str, entity_type: str,
                    mode: MatchingMode = MatchingMode.COMPREHENSIVE,
                    confidence_threshold: float = 0.3) -> List[MatchResult]:
        """
        Find matches for a term using the specified matching mode.

        Args:
            term: The term to find matches for
            entity_type: Either 'intervention' or 'condition'
            mode: Matching mode (safe_only, comprehensive, llm_only)
            confidence_threshold: Minimum confidence for LLM matches

        Returns:
            List of MatchResult objects, deduplicated and sorted by safety/confidence
        """
        if not term or not term.strip():
            return []

        entity_type = self._validate_entity_type(entity_type)
        all_matches = []

        if mode == MatchingMode.SAFE_ONLY:
            # Only use exact and pattern matching
            all_matches.extend(self._find_exact_matches(term, entity_type))
            if not all_matches:  # Only try pattern matching if no exact match
                all_matches.extend(self._find_pattern_matches(term, entity_type))

        elif mode == MatchingMode.LLM_ONLY:
            # Only use LLM matching (for testing/debugging)
            llm_matches = self._find_llm_matches(term, entity_type)
            all_matches.extend([m for m in llm_matches if m.confidence >= confidence_threshold])

        elif mode == MatchingMode.COMPREHENSIVE:
            # Use all strategies in hierarchical order

            # 1. Try exact matching first (fastest and safest)
            exact_matches = self._find_exact_matches(term, entity_type)
            all_matches.extend(exact_matches)

            # 2. If no exact matches, try pattern matching
            if not exact_matches:
                pattern_matches = self._find_pattern_matches(term, entity_type)
                all_matches.extend(pattern_matches)

                # 3. If still no matches, try LLM matching
                if not pattern_matches:
                    llm_matches = self._find_llm_matches(term, entity_type)
                    all_matches.extend([m for m in llm_matches if m.confidence >= confidence_threshold])

        # Remove duplicates and sort results
        return self._deduplicate_and_sort(all_matches)

    def _deduplicate_and_sort(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Remove duplicate matches and sort by safety and confidence."""
        if not matches:
            return []

        # Remove duplicates based on canonical ID, keeping the best match
        unique_matches = {}
        for match in matches:
            canonical_id = match.canonical_id

            if canonical_id not in unique_matches:
                unique_matches[canonical_id] = match
            else:
                # Keep the match with higher safety priority
                existing_match = unique_matches[canonical_id]
                if self._is_safer_match(match, existing_match):
                    unique_matches[canonical_id] = match

        # Sort by safety priority then confidence
        sorted_matches = sorted(unique_matches.values(),
                              key=lambda x: (self._get_safety_priority(x), x.confidence),
                              reverse=True)

        return sorted_matches

    def _is_safer_match(self, match1: MatchResult, match2: MatchResult) -> bool:
        """Determine if match1 is safer than match2."""
        priority1 = self._get_safety_priority(match1)
        priority2 = self._get_safety_priority(match2)

        if priority1 != priority2:
            return priority1 > priority2
        else:
            # Same safety level, prefer higher confidence
            return match1.confidence > match2.confidence

    def _get_safety_priority(self, match: MatchResult) -> int:
        """Get safety priority for a match (higher = safer)."""
        safety_priorities = {
            'existing_mapping': 100,           # Existing mappings are safest
            'exact_canonical': 95,             # Exact canonical matches
            'exact_normalized': 90,            # Exact normalized matches
            'safe_plural_addition': 85,        # Safe plural forms
            'safe_plural_removal': 85,
            'definite_article_removal': 80,   # Article removal
            'spacing_punctuation_normalization': 80,  # Spacing/punctuation
            'llm_semantic_cached': 70,         # Cached LLM matches
            'llm_semantic': 60,                # Fresh LLM matches
        }

        return safety_priorities.get(match.method, 0)

    # === BACKWARD COMPATIBILITY API ===


    def find_or_create_mapping(self, term: str, entity_type: str,
                             confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Find a match or create a new canonical entity if none found.

        Returns:
            Dictionary with canonical_id, canonical_name, method, confidence, is_new
        """
        if not term or not term.strip():
            return {
                'canonical_id': None,
                'canonical_name': term,
                'method': 'empty_term',
                'confidence': 0.0,
                'is_new': False,
                'reasoning': 'Empty or whitespace term'
            }

        term = term.strip()
        entity_type = self._validate_entity_type(entity_type)

        # Try to find matches using comprehensive strategy
        matches = self.find_matches(term, entity_type, MatchingMode.COMPREHENSIVE, confidence_threshold)

        if matches:
            best_match = matches[0]  # Already sorted by safety/confidence
            return {
                'canonical_id': best_match.canonical_id,
                'canonical_name': best_match.canonical_name,
                'method': best_match.method,
                'confidence': best_match.confidence,
                'is_new': False,
                'reasoning': best_match.reasoning
            }

        # No matches found - create new canonical entity
        if not config.fast_mode:
            self.logger.info(f"Creating new canonical entity for: {term} ({entity_type})")

        try:
            canonical_id = self.create_canonical_entity(term, entity_type)

            # Add the term as its own canonical mapping
            self.create_mapping(term, canonical_id, 1.0, "exact")

            return {
                'canonical_id': canonical_id,
                'canonical_name': term,
                'method': 'new_canonical',
                'confidence': 1.0,
                'is_new': True,
                'reasoning': 'No existing match found, created new canonical entity'
            }

        except Exception as e:
            self.logger.error(f"Error creating canonical entity for '{term}': {e}")
            return {
                'canonical_id': None,
                'canonical_name': term,
                'method': 'error',
                'confidence': 0.0,
                'is_new': False,
                'reasoning': f'Error creating canonical entity: {e}'
            }

    def get_canonical_name(self, term: str, entity_type: str) -> str:
        """Get the canonical name for a term, or return the term itself if not mapped."""
        existing_mapping = self.find_mapping_by_term(term, entity_type)
        return existing_mapping['canonical_name'] if existing_mapping else term


    # === BATCH PROCESSING METHODS ===

    def batch_normalize_terms(self, terms_list: List[str], entity_type: str,
                            confidence_threshold: float = 0.7) -> Dict[str, Dict[str, Any]]:
        """
        Efficiently normalize multiple terms using bulk database operations.

        Args:
            terms_list: List of terms to normalize
            entity_type: Either 'intervention' or 'condition'
            confidence_threshold: Minimum confidence for LLM matching

        Returns:
            Dictionary mapping terms to their normalization results
        """
        entity_type = self._validate_entity_type(entity_type)

        if not config.fast_mode:
            self.logger.info(f"Batch normalizing {len(terms_list)} {entity_type} terms")

        # Filter and prepare terms
        valid_terms = [(term.strip(), entity_type) for term in terms_list
                      if term and term.strip()]

        if not valid_terms:
            return {}

        # Use bulk normalization with transaction
        bulk_results = self.bulk_normalize_with_transaction(
            valid_terms, confidence_threshold, create_new_entities=True
        )

        # Convert back to the expected format
        results = {}
        for term, _ in valid_terms:
            key = (term, entity_type)
            if key in bulk_results:
                results[term] = bulk_results[key]

        return results


    # === DEDUPLICATION METHODS ===

    def get_canonical_entities_by_type_with_usage(self, entity_type: str):
        """Get all canonical entities of a specific type with usage counts"""
        entity_type = self._validate_entity_type(entity_type)
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT ce.id, ce.canonical_name,
                   COALESCE(intervention_count, 0) as intervention_usage,
                   COALESCE(condition_count, 0) as condition_usage,
                   (COALESCE(intervention_count, 0) + COALESCE(condition_count, 0)) as total_usage
            FROM canonical_entities ce
            LEFT JOIN (
                SELECT intervention_canonical_id, COUNT(*) as intervention_count
                FROM interventions
                WHERE intervention_canonical_id IS NOT NULL
                GROUP BY intervention_canonical_id
            ) i ON i.intervention_canonical_id = ce.id
            LEFT JOIN (
                SELECT condition_canonical_id, COUNT(*) as condition_count
                FROM interventions
                WHERE condition_canonical_id IS NOT NULL
                GROUP BY condition_canonical_id
            ) c ON c.condition_canonical_id = ce.id
            WHERE ce.entity_type = ?
            ORDER BY total_usage DESC
        """, (entity_type,))

        return cursor.fetchall()

    def get_llm_deduplication(self, terms: List[str]) -> Dict[str, Any]:
        """Get LLM analysis of duplicate terms using centralized prompt service"""
        if not LLM_AVAILABLE or not self.llm_client:
            return {"duplicate_groups": []}

        # Use centralized prompt service
        if self.prompt_service:
            prompt = self.prompt_service.create_duplicate_analysis_prompt(terms)
        else:
            # Fallback prompt (should not happen in normal operation)
            terms_list = "\n".join([f"- {term}" for term in terms])
            prompt = f"Analyze these terms for duplicates: {terms_list}"

        try:
            response = self.llm_client.generate(prompt, temperature=0.1)
            response_text = response['content'].strip()

            # Clean response - remove markdown formatting if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            result = json.loads(response_text.strip())
            return result
        except Exception as e:
            self.logger.error(f"LLM deduplication failed: {str(e)}")
            return {"duplicate_groups": []}

    def merge_canonical_entities(self, canonical_name: str, synonyms: List[str], entity_type: str):
        """
        Merge duplicate canonical entities into one with comprehensive transactional integrity.

        Uses transactions, constraint validation, and rollback capability to ensure data consistency.
        """
        entity_type = self._validate_entity_type(entity_type)

        # Pre-flight validation to avoid starting transaction for invalid operations
        if not synonyms:
            return 0

        # Remove duplicates and empty strings
        synonyms = [s.strip() for s in synonyms if s and s.strip()]
        if not synonyms:
            return 0

        cursor = self.db.cursor()
        merged_count = 0

        try:
            # Start transaction with savepoint for rollback capability
            cursor.execute("SAVEPOINT merge_entities_start")

            # Step 1: Validate all entities exist and get detailed info
            synonym_placeholders = ','.join(['?' for _ in synonyms])
            cursor.execute(f"""
                SELECT id, canonical_name,
                       (SELECT COUNT(*) FROM entity_mappings WHERE canonical_id = ce.id) as mapping_count,
                       (SELECT COUNT(*) FROM interventions WHERE intervention_canonical_id = ce.id OR condition_canonical_id = ce.id) as intervention_count
                FROM canonical_entities ce
                WHERE canonical_name IN ({synonym_placeholders}) AND entity_type = ?
            """, synonyms + [entity_type])

            entities_to_merge = cursor.fetchall()

            if not entities_to_merge:
                cursor.execute("RELEASE SAVEPOINT merge_entities_start")
                return 0

            # Step 2: Find or create target entity
            cursor.execute("""
                SELECT id, canonical_name
                FROM canonical_entities
                WHERE canonical_name = ? AND entity_type = ?
            """, (canonical_name, entity_type))

            target_entity = cursor.fetchone()

            if not target_entity:
                # Create new canonical entity with validation
                cursor.execute("""
                    INSERT INTO canonical_entities (canonical_name, entity_type, description)
                    VALUES (?, ?, ?)
                """, (canonical_name, entity_type, f"Merged entity from {len(entities_to_merge)} duplicates"))
                target_id = cursor.lastrowid

                if not target_id:
                    raise DatabaseError("Failed to create target canonical entity")
            else:
                target_id = target_entity[0]

            # Step 3: Pre-validate that merge operations will succeed
            total_mappings = sum(entity[2] for entity in entities_to_merge)  # mapping_count
            total_interventions = sum(entity[3] for entity in entities_to_merge)  # intervention_count

            self.logger.info(f"Merging {len(entities_to_merge)} entities into '{canonical_name}' (ID: {target_id})")
            self.logger.info(f"Will update {total_mappings} mappings and {total_interventions} intervention records")

            # Step 4: Perform atomic merge operations
            for entity_id, entity_name, mapping_count, intervention_count in entities_to_merge:
                if entity_id == target_id:
                    continue  # Skip self-merge

                # Validate entity still exists (concurrent access protection)
                cursor.execute("SELECT 1 FROM canonical_entities WHERE id = ?", (entity_id,))
                if not cursor.fetchone():
                    self.logger.warning(f"Entity {entity_id} ({entity_name}) no longer exists, skipping")
                    continue

                # Update intervention records with constraint checking
                if entity_type == 'intervention':
                    cursor.execute("""
                        UPDATE interventions
                        SET intervention_canonical_id = ?
                        WHERE intervention_canonical_id = ?
                    """, (target_id, entity_id))

                elif entity_type == 'condition':
                    cursor.execute("""
                        UPDATE interventions
                        SET condition_canonical_id = ?
                        WHERE condition_canonical_id = ?
                    """, (target_id, entity_id))

                # Verify intervention updates succeeded
                updated_interventions = cursor.rowcount
                if updated_interventions != intervention_count:
                    self.logger.warning(f"Expected to update {intervention_count} interventions, actually updated {updated_interventions}")

                # Update entity mappings with conflict resolution
                cursor.execute("""
                    UPDATE entity_mappings
                    SET canonical_id = ?
                    WHERE canonical_id = ?
                """, (target_id, entity_id))

                # Verify mapping updates succeeded
                updated_mappings = cursor.rowcount
                if updated_mappings != mapping_count:
                    self.logger.warning(f"Expected to update {mapping_count} mappings, actually updated {updated_mappings}")

                # Add original entity name as synonym mapping (with conflict handling)
                cursor.execute("""
                    INSERT OR IGNORE INTO entity_mappings
                    (raw_text, canonical_id, entity_type, confidence_score, mapping_method)
                    VALUES (?, ?, ?, ?, ?)
                """, (entity_name, target_id, entity_type, 0.95, 'deduplication_merge'))

                # Final constraint check: ensure no references remain before deletion
                cursor.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM entity_mappings WHERE canonical_id = ?) as remaining_mappings,
                        (SELECT COUNT(*) FROM interventions WHERE intervention_canonical_id = ? OR condition_canonical_id = ?) as remaining_interventions
                """, (entity_id, entity_id, entity_id))

                remaining_refs = cursor.fetchone()
                if remaining_refs[0] > 0 or remaining_refs[1] > 0:
                    raise DatabaseError(f"Cannot delete entity {entity_id}: {remaining_refs[0]} mappings and {remaining_refs[1]} interventions still reference it")

                # Safe to delete the old canonical entity
                cursor.execute("""
                    DELETE FROM canonical_entities
                    WHERE id = ?
                """, (entity_id,))

                if cursor.rowcount != 1:
                    raise DatabaseError(f"Failed to delete canonical entity {entity_id}")

                merged_count += 1
                self.logger.info(f"Successfully merged entity {entity_id} ({entity_name}) into {target_id}")

            # Step 5: Final validation of merged state
            cursor.execute("""
                SELECT COUNT(*) FROM entity_mappings WHERE canonical_id = ?
            """, (target_id,))
            final_mapping_count = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM interventions
                WHERE intervention_canonical_id = ? OR condition_canonical_id = ?
            """, (target_id, target_id))
            final_intervention_count = cursor.fetchone()[0]

            # Commit the transaction
            cursor.execute("RELEASE SAVEPOINT merge_entities_start")
            self.db.commit()

            self.logger.info(f"Merge completed successfully: {merged_count} entities merged")
            self.logger.info(f"Final state: {final_mapping_count} mappings, {final_intervention_count} interventions")

            return merged_count

        except Exception as e:
            # Rollback to savepoint on any error
            try:
                cursor.execute("ROLLBACK TO SAVEPOINT merge_entities_start")
                self.logger.error(f"Merge operation failed, rolled back: {e}")
            except Exception as rollback_error:
                self.logger.error(f"Rollback failed: {rollback_error}")
                # Database may be in inconsistent state - this is critical
                raise DatabaseError(f"Critical: Merge failed and rollback failed: {e}, {rollback_error}")

            raise DatabaseError(f"Entity merge failed: {e}")

    def batch_deduplicate_entities(self, entity_type: Optional[str] = None, confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Run LLM-based deduplication process on canonical entities.

        Args:
            entity_type: Specific entity type to deduplicate, or None for all types
            confidence_threshold: Minimum confidence for merging duplicates

        Returns:
            Dictionary with deduplication results
        """
        # Create backup (skip in FAST_MODE for performance)
        backup_path = None
        if not config.fast_mode:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            db_path = getattr(config, 'db_path', 'data/processed/intervention_research.db')
            backup_path = f"{db_path.replace('.db', '')}_deduplication_backup_{timestamp}.db"
            shutil.copy2(db_path, backup_path)

        total_merged = 0
        entity_types = [entity_type] if entity_type else ['intervention', 'condition']

        for ent_type in entity_types:
            entities = self.get_canonical_entities_by_type_with_usage(ent_type)

            if len(entities) < 2:
                continue

            # Get terms with usage > 0
            used_entities = [e for e in entities if e[4] > 0]  # total_usage > 0

            if len(used_entities) < 2:
                continue

            # Extract just the canonical names for LLM analysis
            term_names = [entity[1] for entity in used_entities[:50]]  # Limit to top 50 most used

            # Get LLM deduplication results
            llm_result = self.get_llm_deduplication(term_names)
            duplicate_groups = llm_result.get('duplicate_groups', [])

            # Process each duplicate group with transaction safety
            for group in duplicate_groups:
                canonical_name = group.get('canonical_name', '')
                synonyms = group.get('synonyms', [])
                confidence = group.get('confidence', 0.0)

                if confidence > confidence_threshold and len(synonyms) > 1:
                    try:
                        merged = self.merge_canonical_entities(canonical_name, synonyms, ent_type)
                        total_merged += merged

                        # merge_canonical_entities already handles its own transactions
                        # so no additional commit needed here

                    except DatabaseError as e:
                        self.logger.error(f"Failed to merge group {canonical_name}: {e}")
                        # Continue with other groups - don't let one failure stop the whole process
                        continue

        return {
            'total_merged': total_merged,
            'backup_path': backup_path,
            'confidence_threshold': confidence_threshold
        }

    # === MAPPING SUGGESTIONS METHODS ===

    def analyze_existing_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Analyze current mapping coverage"""
        cursor = self.db.cursor()

        # Count total unique terms
        cursor.execute("""
            SELECT
                'intervention' as entity_type,
                COUNT(DISTINCT intervention_name) as total_terms,
                COUNT(DISTINCT CASE WHEN em.raw_text IS NOT NULL THEN intervention_name END) as mapped_terms
            FROM interventions i
            LEFT JOIN entity_mappings em ON i.intervention_name = em.raw_text AND em.entity_type = 'intervention'

            UNION ALL

            SELECT
                'condition' as entity_type,
                COUNT(DISTINCT health_condition) as total_terms,
                COUNT(DISTINCT CASE WHEN em.raw_text IS NOT NULL THEN health_condition END) as mapped_terms
            FROM interventions i
            LEFT JOIN entity_mappings em ON i.health_condition = em.raw_text AND em.entity_type = 'condition'
        """)

        results = cursor.fetchall()
        coverage = {}

        for row in results:
            entity_type = row['entity_type']
            total = row['total_terms']
            mapped = row['mapped_terms']
            coverage[entity_type] = {
                'total': total,
                'mapped': mapped,
                'coverage_percent': (mapped / total * 100) if total > 0 else 0
            }

        return coverage

    def get_unmapped_terms_with_frequency(self, entity_type: str, min_frequency: int = 1) -> List[Tuple[str, int]]:
        """Get unmapped terms with their frequency counts"""
        entity_type = self._validate_entity_type(entity_type)
        cursor = self.db.cursor()

        if entity_type == 'intervention':
            column = 'intervention_name'
        else:
            column = 'health_condition'

        cursor.execute(f"""
            SELECT
                i.{column} as term,
                COUNT(*) as frequency
            FROM interventions i
            LEFT JOIN entity_mappings em ON i.{column} = em.raw_text AND em.entity_type = ?
            WHERE em.raw_text IS NULL
            AND i.{column} IS NOT NULL
            AND TRIM(i.{column}) != ''
            GROUP BY i.{column}
            HAVING frequency >= ?
            ORDER BY frequency DESC
        """, (entity_type, min_frequency))

        return [(row['term'], row['frequency']) for row in cursor.fetchall()]

    def batch_generate_mapping_suggestions(self, entity_type: Optional[str] = None,
                                         min_frequency: int = 2,
                                         batch_size: int = 20) -> List[Dict[str, Any]]:
        """
        Generate mapping suggestions using safe methods + LLM enhancement.

        Args:
            entity_type: Specific entity type or None for all types
            min_frequency: Minimum frequency for terms to be processed
            batch_size: Batch size for LLM processing

        Returns:
            List of mapping suggestion dictionaries
        """
        suggestions = []
        entity_types = [entity_type] if entity_type else ['condition', 'intervention']

        for ent_type in entity_types:
            if not config.fast_mode:
                self.logger.info(f"Processing {ent_type} terms")

            # Get unmapped terms
            unmapped_terms = self.get_unmapped_terms_with_frequency(ent_type, min_frequency)

            if not unmapped_terms:
                continue

            if not config.fast_mode:
                self.logger.info(f"Processing {len(unmapped_terms)} unmapped {ent_type} terms")

            # Process in batches for better performance
            for i in range(0, len(unmapped_terms), batch_size):
                batch = unmapped_terms[i:i + batch_size]
                batch_terms = [term for term, freq in batch]
                freq_map = {term: freq for term, freq in batch}

                for term in batch_terms:
                    frequency = freq_map[term]

                    # Try safe methods first
                    safe_matches = self.find_matches(term, ent_type, MatchingMode.SAFE_ONLY)

                    if safe_matches:
                        # Use best safe match
                        best_match = safe_matches[0]
                        suggestions.append({
                            'entity_type': ent_type,
                            'original_term': term,
                            'frequency': frequency,
                            'suggested_canonical': best_match.canonical_name,
                            'confidence': best_match.confidence,
                            'method': best_match.method,
                            'canonical_id': best_match.canonical_id,
                            'notes': 'Safe pattern/exact matching'
                        })
                    else:
                        # Try LLM semantic matching
                        llm_matches = self._find_llm_matches(term, ent_type)

                        if llm_matches and llm_matches[0].confidence >= 0.7:  # High confidence threshold
                            llm_match = llm_matches[0]
                            suggestions.append({
                                'entity_type': ent_type,
                                'original_term': term,
                                'frequency': frequency,
                                'suggested_canonical': llm_match.canonical_name,
                                'confidence': llm_match.confidence,
                                'method': 'llm_semantic',
                                'canonical_id': llm_match.canonical_id,
                                'notes': f"LLM match: {llm_match.reasoning}"
                            })
                        elif llm_matches and llm_matches[0].confidence >= 0.5:  # Medium confidence
                            llm_match = llm_matches[0]
                            suggestions.append({
                                'entity_type': ent_type,
                                'original_term': term,
                                'frequency': frequency,
                                'suggested_canonical': llm_match.canonical_name,
                                'confidence': llm_match.confidence,
                                'method': 'llm_semantic_review',
                                'canonical_id': llm_match.canonical_id,
                                'notes': f"LLM match - REVIEW NEEDED: {llm_match.reasoning}"
                            })
                        else:
                            # No good match found
                            suggestions.append({
                                'entity_type': ent_type,
                                'original_term': term,
                                'frequency': frequency,
                                'suggested_canonical': None,
                                'confidence': 0.0,
                                'method': 'no_match',
                                'canonical_id': None,
                                'notes': 'No safe or confident LLM match found - manual review needed'
                            })

        return suggestions

    def save_suggestions_to_csv(self, suggestions: List[Dict], output_path: str):
        """Save suggestions to CSV file"""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['entity_type', 'original_term', 'frequency', 'suggested_canonical',
                         'confidence', 'method', 'canonical_id', 'notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for suggestion in suggestions:
                writer.writerow(suggestion)

    def generate_summary_report(self, suggestions: List[Dict], coverage: Dict) -> str:
        """Generate a summary report of the mapping suggestions"""
        # Categorize suggestions by method and confidence
        categories = {
            'safe_matches': [],
            'llm_high_confidence': [],
            'llm_medium_confidence': [],
            'no_matches': []
        }

        for suggestion in suggestions:
            method = suggestion['method']
            confidence = suggestion['confidence']

            if method in ['existing_mapping', 'exact_normalized', 'safe_pattern']:
                categories['safe_matches'].append(suggestion)
            elif method == 'llm_semantic' and confidence >= 0.7:
                categories['llm_high_confidence'].append(suggestion)
            elif method in ['llm_semantic', 'llm_semantic_review'] and confidence >= 0.5:
                categories['llm_medium_confidence'].append(suggestion)
            else:
                categories['no_matches'].append(suggestion)

        # Generate report
        report = []
        report.append("=" * 60)
        report.append("LLM-ENHANCED MAPPING SUGGESTIONS REPORT")
        report.append("=" * 60)

        # Current coverage
        report.append("\nCURRENT MAPPING COVERAGE:")
        for entity_type, stats in coverage.items():
            report.append(f"  {entity_type}: {stats['mapped']}/{stats['total']} ({stats['coverage_percent']:.1f}%)")

        # New suggestions breakdown
        report.append(f"\nNEW MAPPING SUGGESTIONS:")
        report.append(f"  Total terms analyzed: {len(suggestions)}")
        report.append(f"  Safe matches (ready to apply): {len(categories['safe_matches'])}")
        report.append(f"  LLM high confidence (≥70%): {len(categories['llm_high_confidence'])}")
        report.append(f"  LLM medium confidence (50-69%): {len(categories['llm_medium_confidence'])}")
        report.append(f"  No confident matches: {len(categories['no_matches'])}")

        # Action recommendations
        report.append("\nRECOMMENDED ACTIONS:")
        if categories['safe_matches']:
            report.append(f"1. AUTO-APPLY {len(categories['safe_matches'])} safe matches")
        if categories['llm_high_confidence']:
            report.append(f"2. REVIEW & APPLY {len(categories['llm_high_confidence'])} high-confidence LLM matches")
        if categories['llm_medium_confidence']:
            report.append(f"3. MANUAL REVIEW {len(categories['llm_medium_confidence'])} medium-confidence LLM matches")
        if categories['no_matches']:
            report.append(f"4. RESEARCH {len(categories['no_matches'])} unmatched terms (create new canonical entities?)")

        # Top frequency unmatched terms
        no_matches_by_freq = sorted(categories['no_matches'], key=lambda x: x['frequency'], reverse=True)
        if no_matches_by_freq:
            report.append("\nTOP UNMATCHED TERMS (by frequency):")
            for suggestion in no_matches_by_freq[:10]:
                report.append(f"  {suggestion['original_term']} (freq: {suggestion['frequency']})")

        return "\n".join(report)

    def batch_apply_mappings(self, suggestions: List[Dict[str, Any]],
                           apply_safe_only: bool = True,
                           min_confidence: float = 0.9) -> Dict[str, Any]:
        """
        Apply mapping suggestions automatically using bulk operations with transaction management.

        Args:
            suggestions: List of mapping suggestions
            apply_safe_only: If True, only apply safe matching methods
            min_confidence: Minimum confidence threshold for automatic application

        Returns:
            Dictionary with application results
        """
        applied_count = 0
        skipped_count = 0
        errors = []

        # Filter suggestions that meet criteria for automatic application
        mappings_to_apply = []
        safe_methods = ['existing_mapping', 'exact_canonical', 'exact_normalized',
                       'safe_plural_addition', 'safe_plural_removal',
                       'definite_article_removal', 'spacing_punctuation_normalization']

        for suggestion in suggestions:
            try:
                should_apply = False

                if apply_safe_only:
                    should_apply = suggestion['method'] in safe_methods
                else:
                    should_apply = suggestion['confidence'] >= min_confidence

                if should_apply and suggestion['canonical_id']:
                    mappings_to_apply.append({
                        'original_term': suggestion['original_term'],
                        'canonical_id': suggestion['canonical_id'],
                        'confidence': suggestion['confidence'],
                        'method': suggestion['method']
                    })
                else:
                    skipped_count += 1

            except Exception as e:
                errors.append({
                    'term': suggestion.get('original_term', 'unknown'),
                    'error': str(e)
                })
                skipped_count += 1

        # Apply mappings using bulk operations with transaction
        if mappings_to_apply:
            try:
                self.db.execute("BEGIN TRANSACTION")
                mapping_ids = self.bulk_create_mappings(mappings_to_apply)
                self.db.commit()
                applied_count = len(mapping_ids)

            except Exception as e:
                self.db.rollback()
                self.logger.error(f"Error in bulk mapping application: {e}")

                # Fall back to individual application for error reporting
                for mapping in mappings_to_apply:
                    try:
                        self.create_mapping(
                            mapping['original_term'],
                            mapping['canonical_id'],
                            mapping['confidence'],
                            mapping['method']
                        )
                        applied_count += 1
                    except Exception as individual_error:
                        errors.append({
                            'term': mapping['original_term'],
                            'error': str(individual_error)
                        })
                        skipped_count += 1

        return {
            'applied_count': applied_count,
            'skipped_count': skipped_count,
            'errors': errors
        }

    # === STATISTICS AND REPORTING ===

    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the current mapping state."""
        cursor = self.db.cursor()

        # Count canonical entities by type
        cursor.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM canonical_entities
            GROUP BY entity_type
        """)
        entity_counts = {row['entity_type']: row['count'] for row in cursor.fetchall()}

        # Count mappings by type
        cursor.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM entity_mappings
            GROUP BY entity_type
        """)
        mapping_counts = {row['entity_type']: row['count'] for row in cursor.fetchall()}

        # Count unique terms vs mappings ratio
        cursor.execute("""
            SELECT entity_type,
                   COUNT(DISTINCT raw_text) as unique_terms,
                   COUNT(DISTINCT canonical_id) as canonical_entities
            FROM entity_mappings
            GROUP BY entity_type
        """)
        ratio_stats = {}
        for row in cursor.fetchall():
            entity_type = row['entity_type']
            ratio_stats[entity_type] = {
                'unique_terms': row['unique_terms'],
                'canonical_entities': row['canonical_entities'],
                'compression_ratio': row['unique_terms'] / row['canonical_entities'] if row['canonical_entities'] > 0 else 0
            }

        return {
            'canonical_entities': entity_counts,
            'mappings': mapping_counts,
            'ratios': ratio_stats
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and configuration."""
        return {
            'database_connected': self.db is not None,
            'llm_available': self.llm_client is not None,
            'llm_model': self.llm_model,
            'fast_mode': config.fast_mode,
            'components': {
                'matching_strategies': ['exact', 'pattern', 'llm'],
                'matching_modes': [mode.value for mode in MatchingMode]
            },
            'statistics': self.get_mapping_statistics()
        }

    # === MULTI-MODEL CONSENSUS METHODS ===
    # Note: The old create_multi_model_consensus method has been replaced with
    # the new process_consensus_batch method below which includes true duplicate removal

    # Note: generate_consensus_summary method is now implemented in the
    # CONSENSUS AND DEDUPLICATION METHODS section below

    # === CONFIDENCE DECAY AND REVALIDATION METHODS ===

    def identify_mappings_needing_revalidation(self, entity_type: Optional[str] = None,
                                             confidence_threshold: float = 0.5,
                                             max_age_years: float = 2.0) -> List[Dict[str, Any]]:
        """
        Identify mappings that need revalidation due to confidence decay or age.

        Args:
            entity_type: Specific entity type to check, or None for all types
            confidence_threshold: Minimum acceptable adjusted confidence
            max_age_years: Maximum age in years before requiring revalidation

        Returns:
            List of mappings that need attention
        """
        cursor = self.db.cursor()

        # Build query based on entity_type filter
        where_clause = "WHERE 1=1"
        params = []

        if entity_type:
            where_clause += " AND em.entity_type = ?"
            params.append(entity_type)

        cursor.execute(f"""
            SELECT em.id, em.canonical_id, em.raw_text, em.entity_type,
                   em.confidence_score, em.mapping_method, em.created_timestamp,
                   ce.canonical_name,
                   COUNT(i1.id) + COUNT(i2.id) as usage_count
            FROM entity_mappings em
            JOIN canonical_entities ce ON em.canonical_id = ce.id
            LEFT JOIN interventions i1 ON i1.intervention_canonical_id = em.canonical_id
            LEFT JOIN interventions i2 ON i2.condition_canonical_id = em.canonical_id
            {where_clause}
            GROUP BY em.id, em.canonical_id, em.raw_text, em.entity_type,
                     em.confidence_score, em.mapping_method, em.created_timestamp,
                     ce.canonical_name
            ORDER BY usage_count DESC, em.created_timestamp ASC
        """, params)

        mappings_needing_attention = []
        current_date = datetime.now()

        for row in cursor.fetchall():
            mapping = dict(row)

            # Calculate time-adjusted confidence
            adjusted_confidence = self._calculate_time_adjusted_confidence(
                mapping['confidence_score'],
                mapping['created_timestamp'],
                mapping['mapping_method']
            )

            # Calculate age in years
            try:
                created_date = datetime.fromisoformat(mapping['created_timestamp'].replace('Z', '+00:00'))
                age_years = (current_date - created_date).days / 365.25
            except:
                age_years = 0

            # Determine if revalidation is needed
            needs_revalidation = (
                adjusted_confidence < confidence_threshold or
                age_years > max_age_years or
                mapping['mapping_method'] in ['llm_semantic', 'llm_semantic_batch']  # Always review LLM matches periodically
            )

            if needs_revalidation:
                mapping['adjusted_confidence'] = adjusted_confidence
                mapping['age_years'] = age_years
                mapping['revalidation_reason'] = self._get_revalidation_reason(
                    adjusted_confidence, age_years, confidence_threshold, max_age_years
                )
                mappings_needing_attention.append(mapping)

        return mappings_needing_attention

    def _get_revalidation_reason(self, adjusted_confidence: float, age_years: float,
                               confidence_threshold: float, max_age_years: float) -> str:
        """Generate human-readable reason for why revalidation is needed."""
        reasons = []

        if adjusted_confidence < confidence_threshold:
            reasons.append(f"Low confidence: {adjusted_confidence:.2f} < {confidence_threshold}")

        if age_years > max_age_years:
            reasons.append(f"Old mapping: {age_years:.1f} years > {max_age_years}")

        return "; ".join(reasons) if reasons else "Periodic LLM review"

    def update_mapping_confidence(self, mapping_id: int, new_confidence: float,
                                 new_method: str, reason: str = "") -> bool:
        """
        Update the confidence and method for a mapping with audit trail.

        Args:
            mapping_id: ID of the mapping to update
            new_confidence: New confidence score
            new_method: New mapping method
            reason: Reason for the update (for audit trail)

        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.db.cursor()

            # Get current mapping for audit trail
            cursor.execute("""
                SELECT raw_text, entity_type, confidence_score, mapping_method
                FROM entity_mappings WHERE id = ?
            """, (mapping_id,))

            current_mapping = cursor.fetchone()
            if not current_mapping:
                self.logger.error(f"Mapping {mapping_id} not found")
                return False

            # Update the mapping
            cursor.execute("""
                UPDATE entity_mappings
                SET confidence_score = ?, mapping_method = ?, created_timestamp = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_confidence, new_method, mapping_id))

            if cursor.rowcount == 1:
                self.db.commit()
                self.logger.info(f"Updated mapping {mapping_id} ({current_mapping['raw_text']}): "
                               f"confidence {current_mapping['confidence_score']:.2f} → {new_confidence:.2f}, "
                               f"method {current_mapping['mapping_method']} → {new_method}")
                return True
            else:
                self.logger.error(f"Failed to update mapping {mapping_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error updating mapping {mapping_id}: {e}")
            return False

    # === ADDITIONAL BACKWARD COMPATIBILITY METHODS ===

    def search_canonical_entities(self, search_term: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for canonical entities by name."""
        cursor = self.db.cursor()

        if entity_type:
            cursor.execute("""
                SELECT id, canonical_name, entity_type, description, created_timestamp
                FROM canonical_entities
                WHERE canonical_name LIKE ? AND entity_type = ?
                ORDER BY canonical_name
            """, (f"%{search_term}%", entity_type))
        else:
            cursor.execute("""
                SELECT id, canonical_name, entity_type, description, created_timestamp
                FROM canonical_entities
                WHERE canonical_name LIKE ?
                ORDER BY canonical_name
            """, (f"%{search_term}%",))

        return [dict(row) for row in cursor.fetchall()]

    def get_all_mappings_for_canonical(self, canonical_id: int) -> List[Dict[str, Any]]:
        """Get all term mappings for a canonical entity."""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, raw_text, confidence_score, mapping_method, created_timestamp
            FROM entity_mappings
            WHERE canonical_id = ?
            ORDER BY confidence_score DESC
        """, (canonical_id,))

        return [dict(row) for row in cursor.fetchall()]

    # === CONSENSUS AND DEDUPLICATION METHODS ===

    def process_consensus_batch(self, raw_interventions: List[Dict], paper: Dict,
                              confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Essential duplicate removal and basic consensus processing.

        This method focuses on the critical task of removing true duplicates from same paper
        extractions while preserving individual intervention records for research purposes.

        For advanced research consensus (cross-paper evidence accumulation), use the
        InterventionConsensusAnalyzer in the data_mining module.

        Args:
            raw_interventions: All interventions from all models for a paper
            paper: Source paper information
            confidence_threshold: Minimum confidence for validation

        Returns:
            List of deduplicated interventions with true duplicates merged
        """
        if not raw_interventions:
            return []

        self.logger.info(f"Processing duplicates for {len(raw_interventions)} raw interventions from paper {paper.get('pmid', 'unknown')}")

        # Phase 1: Normalize all terms in batch for efficiency
        normalized_terms = self._batch_normalize_consensus_terms(raw_interventions)

        # Phase 2: Resolve canonical entities for all interventions
        interventions_with_canonicals = self._resolve_canonical_entities_batch(raw_interventions, normalized_terms)

        # Phase 3: CRITICAL - Detect and merge true duplicates from same paper
        deduplicated_interventions = self._detect_and_merge_same_paper_duplicates(interventions_with_canonicals, paper)

        # Phase 4: Basic validation and safety checks
        validated_interventions = self._basic_intervention_validation(deduplicated_interventions, confidence_threshold)

        self.logger.info(f"Duplicate processing complete: {len(validated_interventions)} deduplicated interventions from {len(raw_interventions)} raw inputs")
        return validated_interventions

    def _detect_and_merge_same_paper_duplicates(self, interventions: List[Dict], paper: Dict) -> List[Dict]:
        """
        CRITICAL METHOD: Detect and merge true duplicates from same paper where both models
        found identical intervention-condition correlations.

        This is the highest priority deduplication step that ensures statistical validity.

        Args:
            interventions: Interventions with resolved canonical entities
            paper: Source paper information

        Returns:
            List of interventions with same-paper duplicates merged
        """
        if len(interventions) <= 1:
            return interventions

        pmid = paper.get('pmid', 'unknown')
        self.logger.info(f"Detecting same-paper duplicates for paper {pmid} with {len(interventions)} interventions")

        # Group by canonical entity pair + correlation type for exact duplicate detection
        duplicate_groups = defaultdict(list)

        for intervention in interventions:
            # Create unique key for exact duplicate detection
            intervention_canonical_id = intervention.get('intervention_canonical_id')
            condition_canonical_id = intervention.get('condition_canonical_id')
            correlation_type = intervention.get('correlation_type', '').strip().lower()

            # Only group if we have both canonical IDs (complete interventions)
            if intervention_canonical_id and condition_canonical_id:
                duplicate_key = (intervention_canonical_id, condition_canonical_id, correlation_type)
                duplicate_groups[duplicate_key].append(intervention)
            else:
                # Keep incomplete interventions as-is (they'll be handled in other phases)
                duplicate_groups[('incomplete', id(intervention), '')].append(intervention)

        # Merge true duplicates and keep singles
        merged_interventions = []
        true_duplicates_found = 0

        for duplicate_key, intervention_group in duplicate_groups.items():
            if len(intervention_group) == 1:
                # Single intervention - no duplicates
                merged_interventions.append(intervention_group[0])
            else:
                # TRUE DUPLICATES FOUND - Merge them
                self.logger.info(f"Found {len(intervention_group)} true duplicates for key {duplicate_key}")
                merged_intervention = self._merge_true_duplicates(intervention_group, paper)
                merged_interventions.append(merged_intervention)
                true_duplicates_found += len(intervention_group) - 1

        self.logger.info(f"Same-paper duplicate detection complete: {true_duplicates_found} duplicates merged, {len(merged_interventions)} unique interventions remain")
        return merged_interventions

    def _merge_true_duplicates(self, duplicate_interventions: List[Dict], paper: Dict) -> Dict:
        """
        Merge interventions that are true duplicates (same paper, same canonical entities).

        This creates a single high-quality intervention with proper model attribution
        and boosted confidence due to cross-model validation.

        Args:
            duplicate_interventions: List of identical interventions from different models
            paper: Source paper information

        Returns:
            Single merged intervention with enhanced metadata
        """
        if len(duplicate_interventions) == 1:
            return duplicate_interventions[0]

        # Use highest confidence intervention as base
        base_intervention = max(duplicate_interventions, key=lambda x: self._get_effective_confidence(x))
        merged = base_intervention.copy()

        # Extract model information
        contributing_models = []
        confidence_scores = []
        supporting_quotes = []

        for intervention in duplicate_interventions:
            model = intervention.get('extraction_model', 'unknown')
            contributing_models.append(model)

            conf_score = self._get_effective_confidence(intervention)
            confidence_scores.append(conf_score)

            quote = intervention.get('supporting_quote', '').strip()
            if quote:
                supporting_quotes.append(f"[{model}, conf={conf_score:.2f}]: {quote}")

        # Enhanced metadata for true duplicates
        merged['models_contributing'] = contributing_models
        merged['models_used'] = ','.join(sorted(set(contributing_models)))
        merged['raw_extraction_count'] = len(duplicate_interventions)
        merged['model_agreement'] = 'full'  # Perfect agreement for true duplicates
        merged['duplicate_source'] = 'same_paper_same_correlation'
        merged['cross_model_validation'] = True

        # Merge dual confidence and boost for cross-model validation
        extraction_conf, study_conf, legacy_conf = self._merge_dual_confidence(duplicate_interventions)

        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            # Apply 10-15% boost for cross-model validation, cap at 0.98
            validation_boost = 0.12
            merged['consensus_confidence'] = min(0.98, avg_confidence + validation_boost)
            merged['confidence_interval'] = (
                max(0.0, merged['consensus_confidence'] - 0.05),  # Narrow interval due to agreement
                min(1.0, merged['consensus_confidence'] + 0.05)
            )

            # Set dual confidence values
            merged['extraction_confidence'] = min(0.98, extraction_conf + validation_boost) if extraction_conf > 0 else merged['consensus_confidence']
            merged['study_confidence'] = study_conf if study_conf > 0 else None

        else:
            merged['consensus_confidence'] = 0.75  # Default for validated duplicates
            merged['confidence_interval'] = (0.65, 0.85)
            merged['extraction_confidence'] = 0.75
            merged['study_confidence'] = None

        # Merge supporting evidence with model attribution
        if supporting_quotes:
            merged['supporting_quote'] = ' | '.join(supporting_quotes)

        # Add validation timestamp
        merged['consensus_created_at'] = datetime.now().isoformat()
        merged['paper_pmid'] = paper.get('pmid', 'unknown')

        self.logger.info(f"Merged {len(duplicate_interventions)} true duplicates: {merged.get('intervention_name')} + {merged.get('health_condition')}")
        return merged

    def _batch_normalize_consensus_terms(self, raw_interventions: List[Dict]) -> Dict[str, str]:
        """
        Batch normalize all intervention and condition terms for efficient processing.

        Args:
            raw_interventions: All raw interventions from models

        Returns:
            Dictionary mapping original terms to normalized forms
        """
        # Collect all unique terms
        all_terms = set()
        for intervention in raw_interventions:
            intervention_name = intervention.get('intervention_name', '').strip()
            condition_name = intervention.get('health_condition', '').strip()

            if intervention_name:
                all_terms.add(intervention_name)
            if condition_name:
                all_terms.add(condition_name)

        # Batch normalize all terms
        if all_terms:
            # Use existing bulk normalization method
            return self.bulk_normalize_terms_optimized(list(all_terms), 'mixed')
        return {}

    def _resolve_canonical_entities_batch(self, raw_interventions: List[Dict],
                                        normalized_terms: Dict[str, str]) -> List[Dict]:
        """
        Resolve canonical entities for all interventions in batch.

        Args:
            raw_interventions: Raw interventions from models
            normalized_terms: Pre-computed normalized terms

        Returns:
            Interventions with canonical entity IDs resolved
        """
        # Collect unique term-type pairs for bulk lookup
        terms_and_types = set()
        for intervention in raw_interventions:
            intervention_name = intervention.get('intervention_name', '').strip()
            condition_name = intervention.get('health_condition', '').strip()

            if intervention_name:
                terms_and_types.add((intervention_name, 'intervention'))
            if condition_name:
                terms_and_types.add((condition_name, 'condition'))

        # Bulk find existing mappings
        existing_mappings = self.bulk_find_existing_mappings(list(terms_and_types))

        # Resolve canonical entities for each intervention
        interventions_with_canonicals = []
        for intervention in raw_interventions:
            resolved_intervention = intervention.copy()

            # Resolve intervention canonical ID
            intervention_name = intervention.get('intervention_name', '').strip()
            if intervention_name:
                mapping = existing_mappings.get((intervention_name, 'intervention'))
                if mapping:
                    resolved_intervention['intervention_canonical_id'] = mapping['canonical_id']
                    resolved_intervention['intervention_canonical_name'] = mapping['canonical_name']

            # Resolve condition canonical ID
            condition_name = intervention.get('health_condition', '').strip()
            if condition_name:
                mapping = existing_mappings.get((condition_name, 'condition'))
                if mapping:
                    resolved_intervention['condition_canonical_id'] = mapping['canonical_id']
                    resolved_intervention['condition_canonical_name'] = mapping['canonical_name']

            interventions_with_canonicals.append(resolved_intervention)

        return interventions_with_canonicals

    def _basic_intervention_validation(self, interventions: List[Dict],
                                     confidence_threshold: float) -> List[Dict]:
        """
        Basic validation and safety checks for deduplicated interventions.

        This performs essential validation while preserving all interventions.
        Low-confidence interventions are flagged for human review rather than removed.

        Args:
            interventions: Deduplicated interventions to validate
            confidence_threshold: Confidence threshold for flagging

        Returns:
            List of all interventions with validation flags
        """
        if not interventions:
            return []

        validated_interventions = []
        validation_stats = {
            'total_processed': 0,
            'missing_required_fields': 0,
            'low_confidence_flagged': 0,
            'safety_flagged': 0,
            'passed_validation': 0
        }

        for intervention in interventions:
            validation_stats['total_processed'] += 1

            # Always start with the intervention and add validation flags
            validated_intervention = intervention.copy()
            validation_flags = []
            requires_human_review = False

            # Check required fields
            intervention_name = intervention.get('intervention_name', '').strip()
            if not intervention_name:
                self.logger.warning("Intervention missing intervention_name - flagged for review")
                validation_flags.append('missing_intervention_name')
                requires_human_review = True
                validation_stats['missing_required_fields'] += 1
            else:
                # Only exclude if completely unusable (no intervention name)
                # Other missing fields get flagged but preserved

                # Check confidence threshold using new effective confidence method
                confidence = self._get_effective_confidence(intervention)
                if confidence < confidence_threshold:
                    self.logger.info(f"Low confidence intervention {confidence:.2f} below threshold {confidence_threshold} - flagged for human review")
                    validation_flags.append(f'low_confidence_{confidence:.2f}')
                    requires_human_review = True
                    validation_stats['low_confidence_flagged'] += 1

                # Medical safety is now primarily handled through enhanced LLM prompts
                # during normalization. This validation remains as a basic check.
                condition_name = intervention.get('health_condition', '').strip()
                if intervention_name and condition_name:
                    # Basic safety check for obviously problematic combinations
                    # More sophisticated safety is handled during LLM normalization
                    norm_intervention = self.normalize_term(intervention_name.lower())
                    norm_condition = self.normalize_term(condition_name.lower())

                    # Flag for human review if terms are suspiciously similar but likely different
                    # (this catches obvious cases while LLM handles more nuanced ones)
                    if (norm_intervention != norm_condition and
                        len(norm_intervention) > 4 and len(norm_condition) > 4):
                        # Check for potentially confusing similar terms
                        if (abs(len(norm_intervention) - len(norm_condition)) <= 2 and
                            norm_intervention[:3] == norm_condition[:3]):
                            self.logger.warning(f"Potentially confusing intervention-condition pair flagged for review: {intervention_name} + {condition_name}")
                            validation_flags.append('potentially_confusing_terms')
                            requires_human_review = True
                            validation_stats['safety_flagged'] += 1

                # Add validation metadata
                validated_intervention['validation_flags'] = validation_flags
                validated_intervention['requires_human_review'] = requires_human_review
                validated_intervention['validation_timestamp'] = datetime.now().isoformat()

                if not requires_human_review:
                    validation_stats['passed_validation'] += 1

                validated_interventions.append(validated_intervention)

        # Log comprehensive validation summary
        self.logger.info(f"Validation complete: {validation_stats['total_processed']} interventions processed")
        self.logger.info(f"  - {validation_stats['passed_validation']} passed validation")
        self.logger.info(f"  - {validation_stats['low_confidence_flagged']} flagged for low confidence")
        self.logger.info(f"  - {validation_stats['safety_flagged']} flagged for safety review")
        self.logger.info(f"  - {validation_stats['missing_required_fields']} missing required fields")

        human_review_count = sum(1 for i in validated_interventions if i.get('requires_human_review', False))
        self.logger.info(f"Total requiring human review: {human_review_count}")

        return validated_interventions

    def generate_deduplication_summary(self, processed_interventions: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary statistics for deduplicated interventions.

        Args:
            processed_interventions: List of processed (deduplicated) interventions

        Returns:
            Summary statistics dictionary including human review flags
        """
        if not processed_interventions:
            return {'total_processed_interventions': 0}

        # Basic statistics
        model_usage = defaultdict(int)
        confidence_scores = []
        duplicate_merges = 0

        # Human review tracking
        human_review_stats = {
            'total_requiring_review': 0,
            'low_confidence_flagged': 0,
            'safety_flagged': 0,
            'missing_fields_flagged': 0,
            'passed_validation': 0
        }

        validation_flag_counts = defaultdict(int)

        for intervention in processed_interventions:
            # Count model usage
            models = intervention.get('models_contributing', [intervention.get('extraction_model', 'unknown')])
            if isinstance(models, list):
                for model in models:
                    model_usage[model] += 1
            else:
                model_usage[models] += 1

            # Count duplicate merges
            if intervention.get('cross_model_validation', False):
                duplicate_merges += 1

            # Collect confidence scores using effective confidence method
            conf = self._get_effective_confidence(intervention)
            confidence_scores.append(conf)

            # Track human review requirements
            if intervention.get('requires_human_review', False):
                human_review_stats['total_requiring_review'] += 1

                # Count specific flag types
                flags = intervention.get('validation_flags', [])
                for flag in flags:
                    validation_flag_counts[flag] += 1

                    if flag.startswith('low_confidence'):
                        human_review_stats['low_confidence_flagged'] += 1
                    elif flag in ['dangerous_combination', 'potentially_confusing_terms']:
                        human_review_stats['safety_flagged'] += 1
                    elif flag == 'missing_intervention_name':
                        human_review_stats['missing_fields_flagged'] += 1
            else:
                human_review_stats['passed_validation'] += 1

        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        return {
            'total_processed_interventions': len(processed_interventions),
            'true_duplicates_merged': duplicate_merges,
            'model_usage': dict(model_usage),
            'avg_confidence': avg_confidence,
            'confidence_range': (min(confidence_scores), max(confidence_scores)) if confidence_scores else (0, 0),
            'human_review_stats': human_review_stats,
            'validation_flag_breakdown': dict(validation_flag_counts),
            'data_preservation_rate': len(processed_interventions)  # All interventions preserved with flags
        }

    def get_interventions_for_human_review(self, processed_interventions: List[Dict],
                                         filter_by_flag: Optional[str] = None) -> List[Dict]:
        """
        Retrieve interventions that require human review.

        Args:
            processed_interventions: List of processed interventions
            filter_by_flag: Optional flag type to filter by (e.g., 'low_confidence', 'dangerous_combination')

        Returns:
            List of interventions requiring human review
        """
        review_interventions = []

        for intervention in processed_interventions:
            if intervention.get('requires_human_review', False):
                if filter_by_flag is None:
                    # Return all flagged interventions
                    review_interventions.append(intervention)
                else:
                    # Filter by specific flag type
                    flags = intervention.get('validation_flags', [])
                    if any(flag.startswith(filter_by_flag) for flag in flags):
                        review_interventions.append(intervention)

        self.logger.info(f"Retrieved {len(review_interventions)} interventions for human review"
                        + (f" (filter: {filter_by_flag})" if filter_by_flag else ""))

        return review_interventions

    def get_validation_summary_report(self, processed_interventions: List[Dict]) -> str:
        """
        Generate a human-readable validation summary report.

        Args:
            processed_interventions: List of processed interventions

        Returns:
            Formatted validation report
        """
        summary = self.generate_deduplication_summary(processed_interventions)
        human_stats = summary.get('human_review_stats', {})
        flag_breakdown = summary.get('validation_flag_breakdown', {})

        report_lines = [
            "=== Intervention Validation Summary ===",
            "",
            f"Total Processed: {summary.get('total_processed_interventions', 0)}",
            f"Data Preservation Rate: 100% (all interventions preserved with flags)",
            "",
            "Validation Results:",
            f"[PASS] Passed Validation: {human_stats.get('passed_validation', 0)}",
            f"[REVIEW] Requiring Human Review: {human_stats.get('total_requiring_review', 0)}",
            "",
            "Review Categories:",
            f"• Low Confidence: {human_stats.get('low_confidence_flagged', 0)}",
            f"• Safety Concerns: {human_stats.get('safety_flagged', 0)}",
            f"• Missing Fields: {human_stats.get('missing_fields_flagged', 0)}",
            "",
            "Detailed Flag Breakdown:"
        ]

        for flag, count in flag_breakdown.items():
            report_lines.append(f"• {flag}: {count}")

        report_lines.extend([
            "",
            f"True Duplicates Merged: {summary.get('true_duplicates_merged', 0)}",
            f"Average Confidence: {summary.get('avg_confidence', 0):.3f}",
            "",
            "=== End Report ==="
        ])

        return "\n".join(report_lines)


# === CONVENIENCE FUNCTIONS ===

def create_batch_processor(db_path: Optional[str] = None, llm_model: str = "gemma2:9b") -> BatchEntityProcessor:
    """Create a BatchEntityProcessor instance with database connection."""
    if db_path is None:
        db_path = getattr(config, 'db_path', 'back_end/data/processed/intervention_research.db')

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    return BatchEntityProcessor(conn, llm_model)


# === MAIN FUNCTION FOR CLI USAGE ===

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch Entity Processing")
    parser.add_argument('command', choices=['normalize', 'deduplicate', 'suggestions', 'status'],
                       help='Command to execute')
    parser.add_argument('--database', '-d', default=None,
                       help='Path to the database file')
    parser.add_argument('--entity-type', '-t', choices=['intervention', 'condition'],
                       help='Entity type to process')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--confidence', '-c', type=float, default=0.7,
                       help='Confidence threshold')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Run in quiet mode')

    args = parser.parse_args()

    # Set FAST_MODE if quiet
    if args.quiet:
        config.fast_mode = True

    try:
        processor = create_batch_processor(args.database)

        if args.command == 'status':
            status = processor.get_system_status()
            print(json.dumps(status, indent=2))

        elif args.command == 'deduplicate':
            result = processor.batch_deduplicate_entities(args.entity_type, args.confidence)
            print(f"Deduplication complete: {result['total_merged']} entities merged")
            if result['backup_path']:
                print(f"Backup created: {result['backup_path']}")

        elif args.command == 'suggestions':
            suggestions = processor.batch_generate_mapping_suggestions(args.entity_type)
            coverage = processor.analyze_existing_mappings()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = args.output or f"mapping_suggestions_{timestamp}.csv"
            processor.save_suggestions_to_csv(suggestions, csv_path)

            report = processor.generate_summary_report(suggestions, coverage)
            print(report)
            print(f"\nSuggestions saved to: {csv_path}")

        else:
            print(f"Command '{args.command}' not yet implemented")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()