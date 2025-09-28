"""
Entity Normalization System - Refactored Architecture

This module provides a clean, modular entity normalization system
with separated concerns and improved maintainability.

Classes:
    EntityNormalizer: High-level API for entity normalization operations
"""

import sqlite3
from typing import Optional, List, Dict, Any

from .entity_normalizer_repository import CanonicalRepository
from .entity_normalizer_engine import MatchingEngine, MatchingMode
from .entity_normalizer_matchers import MatchResult


class EntityNormalizer:
    """
    High-level API for entity normalization operations.

    This refactored class provides a clean interface while delegating
    all business logic to specialized components. It maintains backward
    compatibility with the existing API where possible.
    """

    def __init__(self, db_connection: sqlite3.Connection, llm_model: str = "gemma2:9b"):
        """
        Initialize the EntityNormalizer with a database connection.

        Args:
            db_connection: SQLite database connection object
            llm_model: LLM model name for semantic matching
        """
        # Initialize core components
        self.repository = CanonicalRepository(db_connection)
        self.matching_engine = MatchingEngine(self.repository, llm_model)

        # Store references for backward compatibility
        self.db = db_connection
        self.llm_model = llm_model

    # === HIGH-LEVEL PUBLIC API ===

    def normalize_entity(self, term: str, entity_type: str,
                        confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Normalize a term to its canonical form using all available methods.

        This is the main method for entity normalization that finds existing
        matches or creates new canonical entities as needed.

        Args:
            term: The term to normalize
            entity_type: Either 'intervention' or 'condition'
            confidence_threshold: Minimum confidence for LLM matching

        Returns:
            Dictionary with canonical_id, canonical_name, method, confidence, is_new
        """
        return self.matching_engine.find_or_create_mapping(term, entity_type, confidence_threshold)

    def find_matches(self, term: str, entity_type: str,
                    safe_only: bool = False,
                    confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find all possible matches for a term without creating new entities.

        Args:
            term: The term to find matches for
            entity_type: Either 'intervention' or 'condition'
            safe_only: If True, only use safe matching methods (no LLM)
            confidence_threshold: Minimum confidence for LLM matches

        Returns:
            List of match dictionaries with canonical entity information
        """
        mode = MatchingMode.SAFE_ONLY if safe_only else MatchingMode.COMPREHENSIVE

        matches = self.matching_engine.find_matches(term, entity_type, mode, confidence_threshold)
        return [match.to_dict() for match in matches]

    def get_canonical_name(self, term: str, entity_type: str) -> str:
        """
        Get the canonical name for a term, or return the term itself if not mapped.

        Args:
            term: The raw text term to normalize
            entity_type: Either 'intervention' or 'condition'

        Returns:
            The canonical name if term is mapped, otherwise the original term
        """
        existing_mapping = self.repository.find_mapping_by_term(term, entity_type)
        return existing_mapping['canonical_name'] if existing_mapping else term

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

        for term in terms_list:
            if term and term.strip():
                results[term] = self.normalize_entity(term.strip(), entity_type, confidence_threshold)

        return results

    # === REPOSITORY DELEGATION METHODS ===
    # These methods provide direct access to repository functionality

    def create_canonical_entity(self, canonical_name: str, entity_type: str,
                              scientific_name: Optional[str] = None) -> int:
        """
        Create a new canonical entity.

        Args:
            canonical_name: The canonical/normalized name
            entity_type: Either 'intervention' or 'condition'
            scientific_name: Optional scientific name for additional metadata

        Returns:
            The ID of the newly created canonical entity

        Raises:
            sqlite3.IntegrityError: If canonical_name already exists
        """
        return self.repository.create_canonical_entity(canonical_name, entity_type, scientific_name)

    def add_term_mapping(self, original_term: str, canonical_id: int,
                        confidence: float, method: str) -> int:
        """
        Map a term to a canonical entity.

        Args:
            original_term: The raw text term to map
            canonical_id: ID of the canonical entity to map to
            confidence: Confidence score (0.0 to 1.0)
            method: Mapping method (e.g., 'exact_match', 'pattern_match', 'manual')

        Returns:
            The ID of the newly created mapping

        Raises:
            ValueError: If canonical entity doesn't exist
            sqlite3.IntegrityError: If this exact mapping already exists
        """
        return self.repository.create_mapping(original_term, canonical_id, confidence, method)

    def search_canonical_entities(self, search_term: str,
                                entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for canonical entities by name.

        Args:
            search_term: Term to search for (supports partial matches)
            entity_type: Optional filter by entity type

        Returns:
            List of canonical entity dictionaries
        """
        return self.repository.search_canonical_entities(search_term, entity_type)

    def get_all_mappings_for_canonical(self, canonical_id: int) -> List[Dict[str, Any]]:
        """
        Get all term mappings for a canonical entity.

        Args:
            canonical_id: The canonical entity ID

        Returns:
            List of mapping dictionaries
        """
        return self.repository.get_mappings_for_canonical(canonical_id)

    def get_mapping_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the current mapping state.

        Returns:
            Dictionary with counts, ratios, and other statistics
        """
        return self.repository.get_mapping_statistics()

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the matching engine configuration.

        Returns:
            Dictionary with engine statistics and capabilities
        """
        return self.matching_engine.get_matching_statistics()

    # === BACKWARD COMPATIBILITY METHODS ===
    # These methods maintain compatibility with the old API

    def find_canonical_id(self, term: str, entity_type: str) -> Optional[int]:
        """
        Find the canonical ID for a term if it's already mapped.

        Args:
            term: The raw text term to look up
            entity_type: Either 'intervention' or 'condition'

        Returns:
            The canonical_id if term is mapped, None otherwise
        """
        existing_mapping = self.repository.find_mapping_by_term(term, entity_type)
        return existing_mapping['canonical_id'] if existing_mapping else None

    def find_safe_matches_only(self, term: str, entity_type: str) -> List[Dict[str, Any]]:
        """
        Find matches using only the safest methods for medical terms.

        This method maintains backward compatibility with the old API.

        Args:
            term: The term to search for
            entity_type: Either 'intervention' or 'condition'

        Returns:
            List of safe canonical entity matches
        """
        matches = self.matching_engine.find_safe_matches_only(term, entity_type)

        # Convert to the old format for backward compatibility
        results = []
        for match in matches:
            result = match.to_dict()
            # Ensure consistent field naming with old API
            if 'confidence' in result and 'confidence_score' not in result:
                result['confidence_score'] = result['confidence']
            result['safety_level'] = 'safe'
            results.append(result)

        return results

    def find_comprehensive_matches(self, term: str, entity_type: str,
                                 use_llm: bool = True) -> List[Dict[str, Any]]:
        """
        Find matches using all available methods.

        This method maintains backward compatibility with the old API.

        Args:
            term: The term to find matches for
            entity_type: Either 'intervention' or 'condition'
            use_llm: Whether to use LLM for semantic matching

        Returns:
            List of all matches found, sorted by confidence
        """
        mode = MatchingMode.COMPREHENSIVE if use_llm else MatchingMode.SAFE_ONLY
        matches = self.matching_engine.find_matches(term, entity_type, mode)

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

    def find_or_create_mapping(self, term: str, entity_type: str,
                             confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Find or create a mapping for a term using all available methods.

        This method maintains backward compatibility with the old API.

        Args:
            term: The term to normalize
            entity_type: 'intervention' or 'condition'
            confidence_threshold: Minimum confidence for LLM matching

        Returns:
            Dictionary with canonical_id, canonical_name, method, confidence, is_new
        """
        return self.matching_engine.find_or_create_mapping(term, entity_type, confidence_threshold)

    # === UTILITY METHODS ===

    def normalize_term(self, term: str) -> str:
        """
        Normalize a term using the standard normalization rules.

        Args:
            term: The raw text term to normalize

        Returns:
            The normalized term
        """
        from .entity_normalizer_matchers import MatchingStrategy
        return MatchingStrategy.normalize_term(term)

    def validate_entity_type(self, entity_type: str) -> bool:
        """
        Validate that an entity type is supported.

        Args:
            entity_type: The entity type to validate

        Returns:
            True if valid, False otherwise
        """
        return entity_type in ['intervention', 'condition']

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status and configuration.

        Returns:
            Dictionary with system status information
        """
        return {
            'database_connected': self.repository.db is not None,
            'llm_available': self.matching_engine.llm_matcher.llm_client is not None,
            'llm_model': self.llm_model,
            'components': {
                'repository': 'CanonicalRepository',
                'matching_engine': 'MatchingEngine',
                'strategies': ['ExactMatcher', 'PatternMatcher', 'LLMMatcher']
            },
            'statistics': {
                'engine_stats': self.get_engine_stats(),
                'mapping_stats': self.get_mapping_stats()
            }
        }