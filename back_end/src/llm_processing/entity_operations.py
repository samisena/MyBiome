#!/usr/bin/env python3
"""
Entity Operations Module

This module contains database operations, LLM processing, and duplicate detection
functionality extracted from BatchEntityProcessor for better code organization.
"""

import sqlite3
import json
import re
import logging
import hashlib
from typing import Optional, List, Dict, Any, Union, Tuple
from collections import defaultdict, Counter
from datetime import datetime

# Import utilities
from .entity_utils import (
    EntityNormalizationError, DatabaseError, ValidationError, MatchingError, LLMError,
    EntityType, MatchingMode, MatchResult, ValidationResult,
    EntityValidator, EntityNormalizer, CacheManager, ConfidenceCalculator, JsonUtils,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE
)

# External dependencies
try:
    from back_end.src.data.api_clients import get_llm_client
    from back_end.src.data.utils import parse_json_safely
    from back_end.src.llm_processing.prompt_service import InterventionPromptService
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# === DATABASE OPERATIONS ===

class EntityRepository:
    """Centralized database operations for entity processing."""

    def __init__(self, db_connection: sqlite3.Connection):
        """Initialize repository with database connection."""
        self.db = db_connection
        self.db.row_factory = sqlite3.Row
        self.logger = logging.getLogger(__name__)

    # === PERFORMANCE OPTIMIZATION METHODS ===

    def ensure_performance_optimizations(self) -> None:
        """
        Ensure performance optimization tables and indexes exist.

        This adds the normalized_terms_cache table for O(1) pattern matching
        and optimizes LLM cache with hash-based keys.
        """
        with self.db as conn:
            cursor = conn.cursor()

            try:
                # Create basic required tables first
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS canonical_entities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        canonical_name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(canonical_name, entity_type)
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS entity_mappings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        raw_text TEXT NOT NULL,
                        canonical_id INTEGER NOT NULL,
                        entity_type TEXT NOT NULL,
                        confidence_score REAL,
                        mapping_method TEXT,
                        created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (canonical_id) REFERENCES canonical_entities(id),
                        UNIQUE(raw_text, entity_type, canonical_id)
                    )
                """)

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

                # Create LLM cache table if it doesn't exist and check for optimizations
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS llm_normalization_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        raw_text TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        candidates_offered TEXT,
                        llm_response TEXT,
                        match_result TEXT,
                        confidence_score REAL,
                        reasoning TEXT,
                        model_name TEXT,
                        cache_key TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(raw_text, entity_type, candidates_offered)
                    )
                """)

                # Check if cache_key column exists, add if not
                cursor.execute("PRAGMA table_info(llm_normalization_cache)")
                columns = [row[1] for row in cursor.fetchall()]

                if 'cache_key' not in columns:
                    cursor.execute("""
                        ALTER TABLE llm_normalization_cache
                        ADD COLUMN cache_key TEXT
                    """)
                    self.logger.info("Added cache_key column to LLM cache")

                # Create index for cache_key
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_llm_cache_key
                    ON llm_normalization_cache(cache_key)
                """)

                conn.commit()
                self.logger.info("Performance optimization tables and indexes ensured")
            except Exception as e:
                conn.rollback()
                raise DatabaseError(f"Failed to ensure performance optimizations: {e}")

    def get_or_compute_normalized_term(self, term: str, entity_type: str) -> str:
        """
        Get normalized term from cache or compute it.

        This provides O(1) lookup for normalized forms instead of O(N) computation.
        """
        with self.db as conn:
            cursor = conn.cursor()

            # First try cache lookup
            cursor.execute("""
                SELECT normalized_term
                FROM normalized_terms_cache
                WHERE original_term = ? AND entity_type = ?
            """, (term, entity_type))

            cached_result = cursor.fetchone()
            if cached_result:
                return cached_result[0]

            # Not in cache, compute and store
            normalized = EntityNormalizer.normalize_term(term)

            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO normalized_terms_cache
                    (original_term, normalized_term, entity_type)
                    VALUES (?, ?, ?)
                """, (term, normalized, entity_type))
                conn.commit()
            except Exception as e:
                # Log but don't fail - normalization still works
                self.logger.warning(f"Failed to cache normalized term: {e}")

            return normalized

    def bulk_populate_normalization_cache(self, entity_type: Optional[str] = None) -> int:
        """
        Bulk populate normalization cache from existing entity mappings.

        Args:
            entity_type: Specific entity type to populate, or None for all

        Returns:
            Number of terms cached
        """
        with self.db as conn:
            cursor = conn.cursor()

        # Query to get unique raw terms that aren't cached yet
        if entity_type:
            cursor.execute(f"""
                SELECT DISTINCT em.raw_text, em.entity_type
                FROM entity_mappings em
                LEFT JOIN normalized_terms_cache ntc
                    ON em.raw_text = ntc.original_term
                    AND em.entity_type = ntc.entity_type
                WHERE em.entity_type = ? AND ntc.original_term IS NULL
            """, (entity_type,))
        else:
            cursor.execute(f"""
                SELECT DISTINCT em.raw_text, em.entity_type
                FROM entity_mappings em
                LEFT JOIN normalized_terms_cache ntc
                    ON em.raw_text = ntc.original_term
                    AND em.entity_type = ntc.entity_type
                WHERE ntc.original_term IS NULL
            """)

        terms_to_cache = cursor.fetchall()
        populated_count = 0

        # Batch insert normalized terms
        cache_entries = []
        for row in terms_to_cache:
            raw_term = row[0]
            ent_type = row[1]
            normalized = EntityNormalizer.normalize_term(raw_term)
            cache_entries.append((raw_term, normalized, ent_type))

        if cache_entries:
            cursor.executemany("""
                INSERT OR IGNORE INTO normalized_terms_cache
                (original_term, normalized_term, entity_type)
                VALUES (?, ?, ?)
            """, cache_entries)

            populated_count = cursor.rowcount
            conn.commit()
            self.logger.info(f"Populated normalization cache with {populated_count} terms")

        return populated_count

    def bulk_normalize_terms_optimized(self, terms: List[str], entity_type: str) -> Dict[str, str]:
        """
        Efficiently normalize multiple terms using the pre-computed cache.

        This provides massive performance improvements for bulk operations by
        minimizing normalize_term() calls.
        """
        if not terms:
            return {}

        with self.db as conn:
            cursor = conn.cursor()
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

        # Step 3: Compute missing normalizations and cache them
        if terms_to_compute:
            cache_entries = []
            for term in terms_to_compute:
                normalized = EntityNormalizer.normalize_term(term)
                normalized_mapping[term] = normalized
                cache_entries.append((term, normalized, entity_type))

            # Batch insert new cache entries
                cursor.executemany("""
                    INSERT OR IGNORE INTO normalized_terms_cache
                    (original_term, normalized_term, entity_type)
                    VALUES (?, ?, ?)
                """, cache_entries)
                conn.commit()

            return normalized_mapping

    # === CANONICAL ENTITY OPERATIONS ===

    def find_canonical_by_id(self, canonical_id: int) -> Optional[Dict[str, Any]]:
        """Find canonical entity by ID."""
        with self.db as conn:
            cursor = conn.cursor()
        cursor.execute("""
            SELECT id, canonical_name, entity_type, description, created_at,
                   em.confidence_score, em.mapping_method, em.created_timestamp,
                   COUNT(em.id) as mapping_count
            FROM canonical_entities ce
            LEFT JOIN entity_mappings em ON ce.id = em.canonical_id
            WHERE ce.id = ?
            GROUP BY ce.id
        """, (canonical_id,))

        row = cursor.fetchone()
        if not row:
            return None

            return {
                'id': row[0], 'canonical_name': row[1], 'entity_type': row[2],
                'description': row[3], 'created_at': row[4],
                'confidence_score': row[5], 'mapping_method': row[6],
                'mapping_count': row[8]
            }

    def find_canonical_by_name(self, canonical_name: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Find canonical entity by name and type."""
        with self.db as conn:
            cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM canonical_entities
            WHERE canonical_name = ? AND entity_type = ?
        """, (canonical_name, entity_type))

        row = cursor.fetchone()
        return dict(row) if row else None

    def create_canonical_entity(self, canonical_name: str, entity_type: str,
                              description: str = None) -> int:
        """Create a new canonical entity."""
        try:
            # Validate inputs
            EntityValidator.validate_entity_type(entity_type)
            EntityValidator.validate_term(canonical_name)

            with self.db as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO canonical_entities (canonical_name, entity_type, description)
                    VALUES (?, ?, ?)
                """, (canonical_name, entity_type, description or f"Auto-created canonical entity"))

                canonical_id = cursor.lastrowid
                conn.commit()

                if not canonical_id:
                    raise DatabaseError("Failed to create canonical entity")

                return canonical_id

        except Exception as e:
            raise DatabaseError(f"Failed to create canonical entity: {e}")

    def find_mapping_by_term(self, term: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Find existing mapping for a term."""
        with self.db as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT em.id, em.raw_text, em.canonical_id, em.entity_type,
                       em.confidence_score, em.mapping_method, em.created_timestamp,
                       ce.canonical_name, ce.description
                FROM entity_mappings em
                JOIN canonical_entities ce ON em.canonical_id = ce.id
                WHERE em.raw_text = ? AND em.entity_type = ?
                ORDER BY em.confidence_score DESC
                LIMIT 1
            """, (term, entity_type))

            row = cursor.fetchone()
            if not row:
                return None

            mapping = {
                'id': row[0], 'raw_text': row[1], 'canonical_id': row[2],
                'entity_type': row[3], 'confidence_score': row[4],
                'mapping_method': row[5], 'created_timestamp': row[6],
                'canonical_name': row[7], 'description': row[8]
            }

            return mapping


    # === LLM CACHE OPERATIONS ===

    def save_llm_cache(self, raw_text: str, entity_type: str, candidates_offered: List[str],
                      llm_response: str, match_result: str, confidence_score: float,
                      reasoning: str, model_name: str, cache_key: str = None) -> None:
        """Save LLM matching result to cache."""
        if not cache_key:
            cache_key = CacheManager.create_cache_key(raw_text, entity_type, str(candidates_offered))

        with self.db as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO llm_normalization_cache
                    (raw_text, entity_type, candidates_offered, llm_response, match_result,
                     confidence_score, reasoning, model_name, cache_key)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (raw_text, entity_type, json.dumps(candidates_offered), llm_response,
                      match_result, confidence_score, reasoning, model_name, cache_key))
                conn.commit()

            except Exception as e:
                raise DatabaseError(f"Failed to save LLM cache: {e}")

    def get_llm_cache(self, raw_text: str, entity_type: str,
                     candidates_offered: List[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve LLM matching result from cache."""
        cache_key = CacheManager.create_cache_key(raw_text, entity_type, str(candidates_offered or []))

        with self.db as conn:
            cursor = conn.cursor()
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
                'match_result': result[0],
                'confidence_score': result[1],
                'reasoning': result[2],
                'llm_response': result[3],
                'created_at': result[4]
            }
        return None

    # === MAPPING OPERATIONS ===

    def create_mapping(self, raw_text: str, canonical_id: int, entity_type: str,
                      confidence_score: float, mapping_method: str) -> int:
        """Create a new entity mapping."""
        try:
            EntityValidator.validate_term(raw_text)
            EntityValidator.validate_entity_type(entity_type)
            EntityValidator.validate_confidence(confidence_score)

            with self.db as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO entity_mappings
                    (raw_text, canonical_id, entity_type, confidence_score, mapping_method)
                    VALUES (?, ?, ?, ?, ?)
                """, (raw_text, canonical_id, entity_type, confidence_score, mapping_method))

                mapping_id = cursor.lastrowid
                conn.commit()

                if not mapping_id:
                    raise DatabaseError("Failed to create mapping")

                return mapping_id

        except Exception as e:
            raise DatabaseError(f"Failed to create mapping: {e}")

    def get_mappings_by_canonical_id(self, canonical_id: int) -> List[Dict[str, Any]]:
        """Get all mappings for a canonical entity."""
        with self.db as conn:
            cursor = conn.cursor()
        cursor.execute("""
            SELECT em.id, em.raw_text, em.entity_type,
                   em.confidence_score, em.mapping_method, em.created_timestamp,
                   ce.canonical_name
            FROM entity_mappings em
            JOIN canonical_entities ce ON em.canonical_id = ce.id
            WHERE em.canonical_id = ?
            ORDER BY em.confidence_score DESC
        """, (canonical_id,))

        return [dict(row) for row in cursor.fetchall()]

# === LLM OPERATIONS ===

class LLMProcessor:
    """Handles all LLM-related operations for entity processing."""

    def __init__(self, repository: EntityRepository, llm_model: str = "gemma2:9b"):
        """Initialize LLM processor."""
        self.repository = repository
        self.llm_model = llm_model
        self.logger = logging.getLogger(__name__)

        # Initialize LLM client and prompt service if available
        if LLM_AVAILABLE:
            try:
                self.llm_client = get_llm_client(llm_model)
                self.prompt_service = InterventionPromptService()
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
                self.prompt_service = None
        else:
            self.llm_client = None
            self.prompt_service = None

    def find_llm_matches_batch(self, terms_and_types: List[Tuple[str, str]],
                              candidate_canonicals: List[str],
                              batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, List[MatchResult]]:
        """
        Find LLM matches for multiple terms efficiently using batch processing.

        Args:
            terms_and_types: List of (term, entity_type) tuples
            candidate_canonicals: List of canonical terms to match against
            batch_size: Number of terms to process in each LLM call

        Returns:
            Dictionary mapping terms to their match results
        """
        if not self.llm_client or not terms_and_types:
            return {}

        all_results = {}

        # Group terms by entity type for more efficient processing
        terms_by_type = defaultdict(list)
        for term, entity_type in terms_and_types:
            terms_by_type[entity_type].append(term)

        for entity_type, terms in terms_by_type.items():
            # Process in batches to avoid overwhelming the LLM
            for i in range(0, len(terms), batch_size):
                batch_terms = terms[i:i + batch_size]
                batch_results = self._execute_batch_llm_request(
                    batch_terms, entity_type, candidate_canonicals
                )
                all_results.update(batch_results)

        return all_results

    def _execute_batch_llm_request(self, terms: List[str], entity_type: str,
                                  candidate_canonicals: List[str]) -> Dict[str, List[MatchResult]]:
        """Execute a single LLM request for multiple terms using structured prompting."""
        if not terms:
            return {}

        # Build batch prompt
        if self.prompt_service:
            prompt = self.prompt_service.create_batch_entity_matching_prompt(
                terms, candidate_canonicals, entity_type
            )
        else:
            # Fallback prompt
            prompt = f"Match terms {terms} to canonical forms: {candidate_canonicals}"

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
                        self.repository.save_llm_cache(
                            term, entity_type, candidate_canonicals,
                            llm_content, match.canonical_name, match.confidence,
                            match.reasoning, self.llm_model
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to cache LLM result: {e}")

            return batch_results

        except Exception as e:
            self.logger.error(f"Batch LLM request failed: {e}")
            return {}

    def _parse_batch_llm_response(self, llm_content: str, terms: List[str], entity_type: str) -> Dict[str, List[MatchResult]]:
        """Parse batch LLM response and convert to MatchResult objects."""
        results = {}

        try:
            # Try to parse as JSON array
            parsed_response = JsonUtils.safe_json_loads(llm_content)
            if not isinstance(parsed_response, list):
                return results

            for item in parsed_response:
                if not isinstance(item, dict):
                    continue

                original_term = item.get('original_term')
                match_name = item.get('match')
                confidence = item.get('confidence', 0.0)
                reasoning = item.get('reasoning', '')

                if original_term and original_term in terms:
                    if match_name:
                        # Find canonical ID (simplified for now)
                        canonical_entity = self.repository.find_canonical_by_name(match_name, entity_type)
                        canonical_id = canonical_entity['id'] if canonical_entity else 0

                        match_result = MatchResult(
                            canonical_id=canonical_id,
                            canonical_name=match_name,
                            entity_type=entity_type,
                            confidence=confidence,
                            method='llm_batch',
                            reasoning=reasoning
                        )
                        results[original_term] = [match_result]
                    else:
                        results[original_term] = []

        except Exception as e:
            self.logger.error(f"Failed to parse batch LLM response: {e}")

        return results

    def get_llm_duplicate_analysis(self, terms: List[str]) -> Dict[str, Any]:
        """Get LLM analysis of duplicate terms using centralized prompt service."""
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

# === DUPLICATE DETECTION ===

class DuplicateDetector:
    """Handles duplicate detection and merging operations."""

    def __init__(self, repository: EntityRepository):
        """Initialize duplicate detector."""
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    def detect_same_paper_duplicates(self, interventions: List[Dict]) -> List[List[Dict]]:
        """
        Detect true duplicates from same paper where both models found identical correlations.

        This groups interventions by canonical intervention-condition pairs that represent
        the exact same correlation studied in the same paper.
        """
        # Group by canonical intervention-condition-correlation tuple
        correlation_groups = defaultdict(list)

        for intervention in interventions:
            # Create grouping key based on normalized canonical entities
            intervention_canonical = intervention.get('canonical_intervention_name', '').lower()
            condition_canonical = intervention.get('canonical_condition_name', '').lower()
            correlation_type = intervention.get('correlation_type', '').lower()

            # Only group if we have meaningful canonical names
            if intervention_canonical and condition_canonical:
                group_key = (intervention_canonical, condition_canonical, correlation_type)
                correlation_groups[group_key].append(intervention)

        # Return groups with multiple interventions (duplicates)
        duplicate_groups = [
            interventions for interventions in correlation_groups.values()
            if len(interventions) > 1
        ]

        return duplicate_groups

    def merge_duplicate_group(self, duplicate_interventions: List[Dict], paper: Dict) -> Dict:
        """
        Merge a group of duplicate interventions into a single consensus intervention.

        This preserves all information while creating a unified record with enhanced metadata.
        """
        if len(duplicate_interventions) == 1:
            return duplicate_interventions[0]

        # Use highest confidence intervention as base
        base_intervention = max(duplicate_interventions,
                               key=lambda x: ConfidenceCalculator.get_effective_confidence(x))
        merged = base_intervention.copy()

        # Extract model information
        contributing_models = []
        confidence_scores = []
        supporting_quotes = []

        for intervention in duplicate_interventions:
            model = intervention.get('extraction_model', 'unknown')
            contributing_models.append(model)

            conf_score = ConfidenceCalculator.get_effective_confidence(intervention)
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
        extraction_conf, study_conf, legacy_conf = ConfidenceCalculator.merge_dual_confidence(duplicate_interventions)

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
        else:
            merged['supporting_quote'] = merged.get('supporting_quote', '')

        # Add consensus metadata
        merged['consensus_metadata'] = {
            'created_at': datetime.now().isoformat(),
            'paper_pmid': paper.get('pmid', 'unknown'),
            'models_agreement_score': 1.0,  # Perfect agreement for true duplicates
            'evidence_strength': 'high',  # Cross-model validation provides high evidence
            'validation_method': 'cross_model_consensus'
        }

        return merged

    def batch_normalize_consensus_terms(self, raw_interventions: List[Dict]) -> Dict[str, str]:
        """Batch normalize all terms in interventions for efficiency."""
        # Extract unique terms that need normalization
        terms_to_normalize = set()
        entity_types = set()

        for intervention in raw_interventions:
            intervention_name = intervention.get('intervention_name', '').strip()
            condition_name = intervention.get('health_condition', '').strip()

            if intervention_name:
                terms_to_normalize.add(intervention_name)
                entity_types.add('intervention')
            if condition_name:
                terms_to_normalize.add(condition_name)
                entity_types.add('condition')

        # Batch normalize all terms
        normalized_terms = {}
        for entity_type in entity_types:
            type_terms = [term for term in terms_to_normalize]  # All terms for now
            type_normalized = self.repository.bulk_normalize_terms_optimized(type_terms, entity_type)
            normalized_terms.update(type_normalized)

        return normalized_terms

    def resolve_canonical_entities_batch(self, raw_interventions: List[Dict],
                                       normalized_terms: Dict[str, str]) -> List[Dict]:
        """Resolve canonical entities for all interventions in batch."""
        interventions_with_canonicals = []

        for intervention in raw_interventions.copy():
            # Resolve intervention canonical entity
            intervention_name = intervention.get('intervention_name', '').strip()
            if intervention_name and intervention_name in normalized_terms:
                canonical_intervention = self.repository.find_canonical_by_name(
                    normalized_terms[intervention_name], 'intervention'
                )
                if canonical_intervention:
                    intervention['canonical_intervention_name'] = canonical_intervention['canonical_name']
                    intervention['canonical_intervention_id'] = canonical_intervention['id']

            # Resolve condition canonical entity
            condition_name = intervention.get('health_condition', '').strip()
            if condition_name and condition_name in normalized_terms:
                canonical_condition = self.repository.find_canonical_by_name(
                    normalized_terms[condition_name], 'condition'
                )
                if canonical_condition:
                    intervention['canonical_condition_name'] = canonical_condition['canonical_name']
                    intervention['canonical_condition_id'] = canonical_condition['id']

            interventions_with_canonicals.append(intervention)

        return interventions_with_canonicals