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
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, Set

# External dependencies
try:
    from back_end.src.data.api_clients import get_llm_client
    from back_end.src.data.utils import parse_json_safely
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

# === EXCEPTIONS ===

class EntityNormalizationError(Exception):
    """Base exception for entity normalization errors."""
    pass

class DatabaseError(EntityNormalizationError):
    """Raised when database operations fail."""
    pass

class ValidationError(EntityNormalizationError):
    """Raised when input validation fails."""
    pass

class MatchingError(EntityNormalizationError):
    """Raised when matching operations fail."""
    pass

class LLMError(EntityNormalizationError):
    """Raised when LLM operations fail."""
    pass

class ConfigurationError(EntityNormalizationError):
    """Raised when system configuration is invalid."""
    pass

# === ENUMS ===

class EntityType(Enum):
    """Enum for valid entity types."""
    INTERVENTION = "intervention"
    CONDITION = "condition"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a value is a valid entity type."""
        return value in [member.value for member in cls]

class MatchingMode(Enum):
    """Different modes for matching entities."""
    SAFE_ONLY = "safe_only"
    COMPREHENSIVE = "comprehensive"
    LLM_ONLY = "llm_only"

# === MATCH RESULT CLASS ===

class MatchResult:
    """Standardized result from any matching strategy."""

    def __init__(self, canonical_id: int, canonical_name: str, entity_type: str,
                 confidence: float, method: str, reasoning: str = "",
                 metadata: Optional[Dict[str, Any]] = None):
        self.canonical_id = canonical_id
        self.canonical_name = canonical_name
        self.entity_type = entity_type
        self.confidence = confidence
        self.method = method
        self.reasoning = reasoning
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for consistency with existing code."""
        return {
            'id': self.canonical_id,
            'canonical_id': self.canonical_id,
            'canonical_name': self.canonical_name,
            'entity_type': self.entity_type,
            'confidence': self.confidence,
            'confidence_score': self.confidence,
            'match_method': self.method,
            'method': self.method,
            'reasoning': self.reasoning,
            'metadata': self.metadata
        }

# === MAIN BATCH ENTITY PROCESSOR CLASS ===

class BatchEntityProcessor:
    """
    Unified batch processor for entity normalization, deduplication, and mapping suggestions.

    This class consolidates all LLM processing functionality into efficient batch operations
    while preserving all sophisticated features from the original modular architecture.
    """

    # Known dangerous medical pairs that should never be matched
    DANGEROUS_PAIRS = [
        ('probiotics', 'prebiotics'),
        ('hyperglycemia', 'hypoglycemia'),
        ('hypertension', 'hypotension'),
        ('hyperthermia', 'hypothermia'),
        ('tachycardia', 'bradycardia'),
        ('hyponatremia', 'hypernatremia'),
        ('acidosis', 'alkalosis')
    ]

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

        # Initialize LLM client if available
        if LLM_AVAILABLE:
            try:
                self.llm_client = get_llm_client(llm_model)
            except Exception as e:
                logging.warning(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
        else:
            self.llm_client = None

        # Performance monitoring
        self.operation_counts = {}
        self.operation_times = {}

        # Configure logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for batch operations."""
        log_level = logging.ERROR if config.fast_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

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

    def _is_dangerous_match(self, term1: str, term2: str) -> bool:
        """Check if two terms represent a potentially dangerous match."""
        norm1 = self.normalize_term(term1.lower())
        norm2 = self.normalize_term(term2.lower())

        for dangerous1, dangerous2 in self.DANGEROUS_PAIRS:
            if ((norm1 == dangerous1 and norm2 == dangerous2) or
                (norm1 == dangerous2 and norm2 == dangerous1)):
                return True

        return False

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

    def find_mapping_by_term(self, term: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Find an existing mapping for a term."""
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
        return dict(result) if result else None

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

    def find_llm_cache(self, term: str, entity_type: str,
                      candidate_canonicals: Optional[List[str]], model_name: str) -> Optional[Dict[str, Any]]:
        """Check if we have a cached LLM decision for this term."""
        cursor = self.db.cursor()

        candidates_json = json.dumps(sorted(candidate_canonicals or []), sort_keys=True) if candidate_canonicals else None

        cursor.execute("""
            SELECT match_result, confidence_score, reasoning, llm_response, created_at
            FROM llm_normalization_cache
            WHERE input_term = ? AND entity_type = ? AND candidate_canonicals = ? AND model_name = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (term, entity_type, candidates_json, model_name))

        result = cursor.fetchone()
        if result:
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
        """Save LLM decision to cache."""
        cursor = self.db.cursor()

        candidates_json = json.dumps(sorted(candidate_canonicals or []), sort_keys=True) if candidate_canonicals else None

        cursor.execute("""
            INSERT OR REPLACE INTO llm_normalization_cache
            (input_term, entity_type, candidate_canonicals, llm_response, match_result,
             confidence_score, reasoning, model_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (term, entity_type, candidates_json, llm_response, match_result, confidence, reasoning, model_name))

        self.db.commit()

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
        """Find matches using safe pattern matching."""
        normalized_term = self.normalize_term(term)
        matches = []

        # Get all canonical entities and mappings for this entity type
        all_mappings = self.get_all_mappings_with_canonicals(entity_type)

        for mapping in all_mappings:
            canonical_name = self.normalize_term(mapping['canonical_name'])
            raw_text = self.normalize_term(mapping['raw_text'] or mapping['canonical_name'])

            # Check against canonical name
            match_result = self._check_pattern_match(normalized_term, canonical_name, mapping, 'canonical')
            if match_result:
                matches.append(match_result)

            # Check against mapped text (if different from canonical)
            if raw_text != canonical_name:
                match_result = self._check_pattern_match(normalized_term, raw_text, mapping, 'mapping')
                if match_result:
                    matches.append(match_result)

        # Remove duplicates based on canonical ID, keeping highest confidence
        unique_matches = {}
        for match in matches:
            canonical_id = match.canonical_id
            if canonical_id not in unique_matches or match.confidence > unique_matches[canonical_id].confidence:
                unique_matches[canonical_id] = match

        # Sort by confidence descending
        return sorted(unique_matches.values(), key=lambda x: x.confidence, reverse=True)

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
        """Find matches using LLM semantic understanding."""
        if not LLM_AVAILABLE or not self.llm_client:
            return []

        # Get candidate canonicals if not provided
        if candidate_canonicals is None:
            canonical_entities = self.get_all_canonical_entities(entity_type)
            candidate_canonicals = [entity['canonical_name'] for entity in canonical_entities]

        if not candidate_canonicals:
            return []

        # Check cache first
        cached_result = self.find_llm_cache(term, entity_type, candidate_canonicals, self.llm_model)
        if cached_result and cached_result['match_result']:
            canonical_entity = self.find_canonical_by_name(
                cached_result['match_result'], entity_type
            )
            if canonical_entity:
                return [MatchResult(
                    canonical_id=canonical_entity['id'],
                    canonical_name=canonical_entity['canonical_name'],
                    entity_type=entity_type,
                    confidence=cached_result['confidence_score'],
                    method='llm_semantic_cached',
                    reasoning=cached_result['reasoning']
                )]

        # Query LLM
        try:
            prompt = self._build_llm_prompt(term, candidate_canonicals, entity_type)
            response = self.llm_client.generate(prompt, temperature=0.1)  # Low temp for consistency

            # Parse response
            llm_content = response['content'].strip()
            parsed = self._parse_llm_response(llm_content)

            if not parsed:
                return []

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
                # Don't fail the match if caching fails
                self.logger.warning(f"Failed to cache LLM decision: {e}")

            # If we have a confident match, return it
            if match_name and confidence > 0.3:  # Minimum confidence threshold
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
            self.logger.error(f"Error in LLM matching for term '{term}': {e}")
            return []

        return []

    def _build_llm_prompt(self, term: str, candidate_canonicals: List[str], entity_type: str) -> str:
        """Build medical-aware LLM prompt for entity matching."""
        candidates_list = "\n".join([f"- {canonical}" for canonical in candidate_canonicals])

        prompt = f"""You are a medical terminology expert. Given the {entity_type} term '{term}', determine if it represents the same medical concept as any of these canonical terms:

{candidates_list}

IMPORTANT MEDICAL CONSIDERATIONS:
- Be very conservative - only match if you're confident they represent the SAME medical concept
- Different substances are different (probiotics != prebiotics)
- Opposite conditions are different (hypertension != hypotension)
- Similar-sounding but different medical terms should NOT match
- Consider synonyms, common names, and abbreviations that refer to the same concept

Respond with valid JSON only:
{{
    "match": "exact_canonical_name_from_list_above" or null,
    "confidence": 0.0-1.0,
    "reasoning": "brief medical explanation"
}}"""

        return prompt

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

    def normalize_entity(self, term: str, entity_type: str,
                        confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Normalize a term to its canonical form using all available methods.

        This is the main method for entity normalization that finds existing
        matches or creates new canonical entities as needed.
        """
        return self.find_or_create_mapping(term, entity_type, confidence_threshold)

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

    def find_safe_matches_only(self, term: str, entity_type: str) -> List[Dict[str, Any]]:
        """Find matches using only the safest methods for medical terms."""
        matches = self.find_matches(term, entity_type, mode=MatchingMode.SAFE_ONLY)

        # Convert to the old format for backward compatibility
        results = []
        for match in matches:
            result = match.to_dict()
            result['safety_level'] = 'safe'
            results.append(result)

        return results

    def find_comprehensive_matches(self, term: str, entity_type: str,
                                 use_llm: bool = True) -> List[Dict[str, Any]]:
        """Find matches using all available methods."""
        mode = MatchingMode.COMPREHENSIVE if use_llm else MatchingMode.SAFE_ONLY
        matches = self.find_matches(term, entity_type, mode)

        # Convert to old format for backward compatibility
        results = []
        for match in matches:
            result = match.to_dict()
            # Add safety level based on method
            if any(safe_method in match.method for safe_method in ['exact', 'pattern', 'plural', 'article', 'spacing']):
                result['safety_level'] = 'safe'
            else:
                result['safety_level'] = 'llm'
            results.append(result)

        return results

    # === BATCH PROCESSING METHODS ===

    def batch_normalize_terms(self, terms_list: List[str], entity_type: str,
                            confidence_threshold: float = 0.7) -> Dict[str, Dict[str, Any]]:
        """
        Efficiently normalize multiple terms.

        Args:
            terms_list: List of terms to normalize
            entity_type: Either 'intervention' or 'condition'
            confidence_threshold: Minimum confidence for LLM matching

        Returns:
            Dictionary mapping terms to their normalization results
        """
        results = {}
        entity_type = self._validate_entity_type(entity_type)

        if not config.fast_mode:
            self.logger.info(f"Batch normalizing {len(terms_list)} {entity_type} terms")

        for term in terms_list:
            if term and term.strip():
                results[term] = self.normalize_entity(term.strip(), entity_type, confidence_threshold)

        return results

    def batch_find_matches(self, terms_list: List[str], entity_type: str,
                          mode: MatchingMode = MatchingMode.COMPREHENSIVE) -> Dict[str, List[MatchResult]]:
        """
        Efficiently find matches for multiple terms.

        Args:
            terms_list: List of terms to find matches for
            entity_type: Either 'intervention' or 'condition'
            mode: Matching mode to use

        Returns:
            Dictionary mapping terms to their match results
        """
        results = {}
        entity_type = self._validate_entity_type(entity_type)

        for term in terms_list:
            if term and term.strip():
                matches = self.find_matches(term.strip(), entity_type, mode)
                if matches:
                    results[term] = matches

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
        """Get LLM analysis of duplicate terms"""
        if not LLM_AVAILABLE or not self.llm_client:
            return {"duplicate_groups": []}

        terms_list = "\n".join([f"- {term}" for term in terms])

        prompt = f"""Analyze these medical terms and identify which ones refer to the same concept.

Terms to analyze:
{terms_list}

Return ONLY valid JSON in this format:
{{
  "duplicate_groups": [
    {{
      "canonical_name": "most formal scientific name",
      "synonyms": ["term1", "term2", "term3"],
      "confidence": 0.95
    }}
  ]
}}

Rules:
- Each group must have confidence 0.0-1.0
- Use the most formal medical/scientific name as canonical_name
- Only group terms that are definitely the same concept"""

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
        """Merge duplicate canonical entities into one"""
        entity_type = self._validate_entity_type(entity_type)
        cursor = self.db.cursor()

        # Find the canonical entity to keep (prefer existing one with canonical_name)
        cursor.execute("""
            SELECT id, canonical_name
            FROM canonical_entities
            WHERE canonical_name = ? AND entity_type = ?
        """, (canonical_name, entity_type))

        target_entity = cursor.fetchone()

        if not target_entity:
            # Create new canonical entity
            cursor.execute("""
                INSERT INTO canonical_entities (canonical_name, entity_type, confidence_score)
                VALUES (?, ?, ?)
            """, (canonical_name, entity_type, 1.0))
            target_id = cursor.lastrowid
        else:
            target_id = target_entity[0]

        # Get all entities to merge
        synonym_placeholders = ','.join(['?' for _ in synonyms])
        cursor.execute(f"""
            SELECT id, canonical_name
            FROM canonical_entities
            WHERE canonical_name IN ({synonym_placeholders}) AND entity_type = ? AND id != ?
        """, synonyms + [entity_type, target_id])

        entities_to_merge = cursor.fetchall()

        if not entities_to_merge:
            return 0

        merged_count = 0

        for entity_id, entity_name in entities_to_merge:
            # Update intervention records
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

            # Update entity mappings
            cursor.execute("""
                UPDATE entity_mappings
                SET canonical_id = ?
                WHERE canonical_id = ?
            """, (target_id, entity_id))

            # Add synonym to entity_mappings if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO entity_mappings
                (raw_text, canonical_id, entity_type, confidence_score)
                VALUES (?, ?, ?, ?)
            """, (entity_name, target_id, entity_type, 0.95))

            # Delete the old canonical entity
            cursor.execute("""
                DELETE FROM canonical_entities
                WHERE id = ?
            """, (entity_id,))

            merged_count += 1

        return merged_count

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

            # Process each duplicate group
            for group in duplicate_groups:
                canonical_name = group.get('canonical_name', '')
                synonyms = group.get('synonyms', [])
                confidence = group.get('confidence', 0.0)

                if confidence > confidence_threshold and len(synonyms) > 1:
                    merged = self.merge_canonical_entities(canonical_name, synonyms, ent_type)
                    total_merged += merged

                    if merged > 0:
                        self.db.commit()

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
        Apply mapping suggestions automatically based on confidence criteria.

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

        for suggestion in suggestions:
            try:
                # Check if this suggestion meets criteria for automatic application
                should_apply = False

                if apply_safe_only:
                    # Only apply safe methods
                    safe_methods = ['existing_mapping', 'exact_canonical', 'exact_normalized',
                                  'safe_plural_addition', 'safe_plural_removal',
                                  'definite_article_removal', 'spacing_punctuation_normalization']
                    should_apply = suggestion['method'] in safe_methods
                else:
                    # Apply based on confidence threshold
                    should_apply = suggestion['confidence'] >= min_confidence

                if should_apply and suggestion['canonical_id']:
                    # Apply the mapping
                    self.create_mapping(
                        suggestion['original_term'],
                        suggestion['canonical_id'],
                        suggestion['confidence'],
                        suggestion['method']
                    )
                    applied_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                errors.append({
                    'term': suggestion['original_term'],
                    'error': str(e)
                })

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

    def create_multi_model_consensus(self, raw_interventions: List[Dict], paper: Dict) -> List[Dict]:
        """
        Create consensus interventions from multiple model extractions.

        This replaces the simple consensus logic from dual_model_analyzer with
        sophisticated normalization and medical-grade entity matching.

        Args:
            raw_interventions: All interventions from all models
            paper: Source paper information

        Returns:
            List of consensus interventions for database storage
        """
        if not raw_interventions:
            return []

        # Group interventions using sophisticated normalization
        grouped_interventions = self._group_interventions_by_similarity(raw_interventions)

        # Create consensus for each group
        consensus_interventions = []
        for group_key, intervention_group in grouped_interventions.items():
            consensus = self._create_consensus_intervention(intervention_group, paper)
            consensus_interventions.append(consensus)

        return consensus_interventions

    def _group_interventions_by_similarity(self, interventions: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group interventions by similarity using sophisticated normalization.

        Uses batch_entity_processor's advanced normalization instead of simple string matching.
        """
        grouped_interventions = {}

        for intervention in interventions:
            # Use sophisticated normalization for grouping key
            intervention_name = intervention.get('intervention_name', '').strip()
            health_condition = intervention.get('health_condition', '').strip()
            category = intervention.get('intervention_category', 'unknown')

            # For grouping purposes, use basic normalization to avoid creating new canonical entities
            # We just want to group similar terms, not persist new mappings
            if intervention_name:
                # Use the basic normalize_term method for grouping
                normalized_intervention_name = self.normalize_term(intervention_name)
                # Apply simple synonym mapping for common medical terms
                normalized_intervention_name = self._apply_simple_intervention_synonyms(normalized_intervention_name)
            else:
                normalized_intervention_name = ''

            if health_condition:
                # Use the basic normalize_term method for grouping
                normalized_condition_name = self.normalize_term(health_condition)
                # Apply simple synonym mapping for common medical terms
                normalized_condition_name = self._apply_simple_condition_synonyms(normalized_condition_name)
            else:
                normalized_condition_name = ''

            # Create sophisticated grouping key
            group_key = f"{normalized_intervention_name}|{normalized_condition_name}|{category}"

            if group_key not in grouped_interventions:
                grouped_interventions[group_key] = []
            grouped_interventions[group_key].append(intervention)

        return grouped_interventions

    def _apply_simple_intervention_synonyms(self, name: str) -> str:
        """Apply simple synonym mapping for common interventions."""
        if not name:
            return ""

        # Handle common variations (from original dual_model_analyzer)
        synonyms = {
            'probiotics': ['probiotic', 'probiotics', 'probiotic supplements'],
            'exercise': ['physical activity', 'exercise', 'physical exercise'],
            'meditation': ['mindfulness', 'meditation', 'mindfulness meditation'],
            'omega-3': ['omega 3', 'omega-3', 'fish oil', 'omega-3 fatty acids'],
            'vitamin d': ['vitamin d3', 'vitamin d', 'cholecalciferol'],
            'magnesium': ['magnesium supplement', 'magnesium', 'mg supplement']
        }

        # Find canonical form
        for canonical, variants in synonyms.items():
            if name in [v.lower() for v in variants]:
                return canonical

        return name

    def _apply_simple_condition_synonyms(self, condition: str) -> str:
        """Apply simple synonym mapping for common conditions."""
        if not condition:
            return ""

        # Handle common variations (from original dual_model_analyzer)
        synonyms = {
            'ibs': ['irritable bowel syndrome', 'ibs', 'irritable bowel'],
            'crohns disease': ['crohn\'s disease', 'crohns disease', 'crohn disease'],
            'depression': ['major depression', 'depression', 'depressive disorder'],
            'anxiety': ['anxiety disorder', 'anxiety', 'generalized anxiety'],
            'diabetes': ['type 2 diabetes', 'diabetes mellitus', 'diabetes']
        }

        # Find canonical form
        for canonical, variants in synonyms.items():
            if condition in [v.lower() for v in variants]:
                return canonical

        return condition

    def _create_consensus_intervention(self, intervention_group: List[Dict], paper: Dict) -> Dict:
        """
        Create a single consensus intervention from a group of similar interventions.

        Args:
            intervention_group: List of similar interventions from different models
            paper: Source paper information

        Returns:
            Consensus intervention dictionary for database storage
        """
        if len(intervention_group) == 1:
            # Single model result
            intervention = intervention_group[0].copy()
            intervention['consensus_confidence'] = 0.60
            intervention['model_agreement'] = 'single'
            intervention['models_contributing'] = [intervention.get('extraction_model', 'unknown')]
            return intervention

        # Multiple models found this intervention
        models_contributing = [i.get('extraction_model', 'unknown') for i in intervention_group]

        # Check for full agreement using medical criteria
        if self._check_model_agreement(intervention_group) == 'full':
            # Both models agree completely
            consensus = intervention_group[0].copy()  # Use first as base
            consensus['consensus_confidence'] = 0.95
            consensus['model_agreement'] = 'full'
            consensus['models_contributing'] = models_contributing

            # Average numerical values
            consensus['confidence_score'] = self._average_scores([i.get('confidence_score') for i in intervention_group])
            consensus['correlation_strength'] = self._average_scores([i.get('correlation_strength') for i in intervention_group])

        else:
            # Partial agreement - merge intelligently
            consensus = self._merge_intervention_group(intervention_group)
            consensus['consensus_confidence'] = 0.75
            consensus['model_agreement'] = 'partial'
            consensus['models_contributing'] = models_contributing

        # Add metadata for tracking
        consensus['raw_extraction_count'] = len(intervention_group)
        consensus['models_used'] = ','.join(sorted(models_contributing))

        return consensus

    def _check_model_agreement(self, interventions: List[Dict]) -> str:
        """
        Check the level of agreement between model extractions.

        Returns:
            'full', 'partial', or 'single'
        """
        if len(interventions) < 2:
            return 'single'

        first = interventions[0]

        for intervention in interventions[1:]:
            # Check key medical fields for agreement
            if (intervention.get('correlation_type') != first.get('correlation_type') or
                abs((intervention.get('confidence_score', 0) or 0) - (first.get('confidence_score', 0) or 0)) > 0.2 or
                abs((intervention.get('correlation_strength', 0) or 0) - (first.get('correlation_strength', 0) or 0)) > 0.2):
                return 'partial'

        return 'full'

    def _merge_intervention_group(self, interventions: List[Dict]) -> Dict:
        """
        Merge interventions using weighted averages and best medical evidence.
        """
        # Use the intervention with highest confidence as base
        base_intervention = max(interventions,
                              key=lambda x: x.get('confidence_score', 0) or 0)
        consensus = base_intervention.copy()

        # Average numerical scores
        consensus['confidence_score'] = self._average_scores([i.get('confidence_score') for i in interventions])
        consensus['correlation_strength'] = self._average_scores([i.get('correlation_strength') for i in interventions])

        # Use most common correlation type (medical consensus)
        correlation_types = [i.get('correlation_type') for i in interventions if i.get('correlation_type')]
        if correlation_types:
            consensus['correlation_type'] = max(set(correlation_types), key=correlation_types.count)

        # Combine supporting quotes for medical evidence
        quotes = [i.get('supporting_quote', '') for i in interventions if i.get('supporting_quote')]
        if quotes:
            consensus['supporting_quote'] = ' | '.join(quotes)

        return consensus

    def _average_scores(self, scores: List[Optional[float]]) -> Optional[float]:
        """Calculate average of numeric scores, handling None values."""
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            return None
        return sum(valid_scores) / len(valid_scores)

    def generate_consensus_summary(self, consensus_interventions: List[Dict]) -> Dict:
        """Generate summary statistics for consensus interventions."""
        if not consensus_interventions:
            return {}

        agreement_counts = {}
        model_usage = defaultdict(int)

        for intervention in consensus_interventions:
            agreement = intervention.get('model_agreement', 'unknown')
            agreement_counts[agreement] = agreement_counts.get(agreement, 0) + 1

            for model in intervention.get('models_contributing', []):
                model_usage[model] += 1

        return {
            'total_consensus_interventions': len(consensus_interventions),
            'agreement_breakdown': agreement_counts,
            'model_usage': dict(model_usage),
            'avg_consensus_confidence': self._average_scores([i.get('consensus_confidence') for i in consensus_interventions])
        }

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

    def add_term_mapping(self, original_term: str, canonical_id: int,
                        confidence: float, method: str) -> int:
        """Map a term to a canonical entity."""
        return self.create_mapping(original_term, canonical_id, confidence, method)

    def find_canonical_id(self, term: str, entity_type: str) -> Optional[int]:
        """Find the canonical ID for a term if it's already mapped."""
        existing_mapping = self.find_mapping_by_term(term, entity_type)
        return existing_mapping['canonical_id'] if existing_mapping else None

    def validate_entity_type(self, entity_type: str) -> bool:
        """Validate that an entity type is supported."""
        return EntityType.is_valid(entity_type)


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