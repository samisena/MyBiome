"""
Database Repository for Entity Normalization

This module provides a clean abstraction layer for all database operations
related to entity normalization, separating data access from business logic.
"""

import sqlite3
import json
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime


class CanonicalRepository:
    """
    Repository class for managing canonical entities and mappings in the database.

    This class provides a clean interface for all database operations without
    any business logic, following the repository pattern.
    """

    def __init__(self, db_connection: sqlite3.Connection):
        """
        Initialize the repository with a database connection.

        Args:
            db_connection: SQLite database connection object
        """
        self.db = db_connection
        self.db.row_factory = sqlite3.Row  # Enable dict-like access to rows

    # === CANONICAL ENTITY OPERATIONS ===

    def find_canonical_by_id(self, canonical_id: int) -> Optional[Dict[str, Any]]:
        """
        Find a canonical entity by its ID.

        Args:
            canonical_id: The canonical entity ID

        Returns:
            Dictionary with canonical entity data or None if not found
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, canonical_name, entity_type, metadata, created_at
            FROM canonical_entities WHERE id = ?
        """, (canonical_id,))

        result = cursor.fetchone()
        return dict(result) if result else None

    def find_canonical_by_name(self, canonical_name: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Find a canonical entity by name and type.

        Args:
            canonical_name: The canonical name to search for
            entity_type: Either 'intervention' or 'condition'

        Returns:
            Dictionary with canonical entity data or None if not found
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, canonical_name, entity_type, metadata, created_at
            FROM canonical_entities
            WHERE canonical_name = ? AND entity_type = ?
        """, (canonical_name, entity_type))

        result = cursor.fetchone()
        return dict(result) if result else None

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
        cursor = self.db.cursor()

        # Prepare metadata JSON if scientific_name provided
        metadata = {}
        if scientific_name:
            metadata['scientific_name'] = scientific_name

        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO canonical_entities (canonical_name, entity_type, metadata)
            VALUES (?, ?, ?)
        """, (canonical_name, entity_type, metadata_json))

        self.db.commit()
        return cursor.lastrowid

    def search_canonical_entities(self, search_term: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for canonical entities by name.

        Args:
            search_term: Term to search for (supports partial matches)
            entity_type: Optional filter by entity type

        Returns:
            List of canonical entity dictionaries
        """
        cursor = self.db.cursor()

        if entity_type:
            cursor.execute("""
                SELECT id, canonical_name, entity_type, metadata, created_at
                FROM canonical_entities
                WHERE canonical_name LIKE ? AND entity_type = ?
                ORDER BY canonical_name
            """, (f"%{search_term}%", entity_type))
        else:
            cursor.execute("""
                SELECT id, canonical_name, entity_type, metadata, created_at
                FROM canonical_entities
                WHERE canonical_name LIKE ?
                ORDER BY canonical_name
            """, (f"%{search_term}%",))

        return [dict(row) for row in cursor.fetchall()]

    def get_all_canonical_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all canonical entities, optionally filtered by type.

        Args:
            entity_type: Optional filter by entity type

        Returns:
            List of canonical entity dictionaries
        """
        cursor = self.db.cursor()

        if entity_type:
            cursor.execute("""
                SELECT id, canonical_name, entity_type, metadata, created_at
                FROM canonical_entities
                WHERE entity_type = ?
                ORDER BY canonical_name
            """, (entity_type,))
        else:
            cursor.execute("""
                SELECT id, canonical_name, entity_type, metadata, created_at
                FROM canonical_entities
                ORDER BY canonical_name
            """)

        return [dict(row) for row in cursor.fetchall()]

    # === ENTITY MAPPING OPERATIONS ===

    def find_mapping_by_term(self, term: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """
        Find an existing mapping for a term.

        Args:
            term: The raw text term to look up
            entity_type: Either 'intervention' or 'condition'

        Returns:
            Dictionary with mapping data including canonical info, or None if not found
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT em.id, em.canonical_id, em.raw_text, em.entity_type,
                   em.confidence_score, em.mapping_method, em.created_at,
                   ce.canonical_name
            FROM entity_mappings em
            JOIN canonical_entities ce ON em.canonical_id = ce.id
            WHERE em.raw_text = ? AND em.entity_type = ?
        """, (term, entity_type))

        result = cursor.fetchone()
        return dict(result) if result else None

    def create_mapping(self, original_term: str, canonical_id: int,
                      confidence: float, method: str) -> int:
        """
        Create a new term mapping to a canonical entity.

        Args:
            original_term: The raw text term to map
            canonical_id: ID of the canonical entity to map to
            confidence: Confidence score (0.0 to 1.0)
            method: Mapping method (e.g., 'exact_match', 'pattern_match', 'llm_semantic')

        Returns:
            The ID of the newly created mapping

        Raises:
            ValueError: If canonical entity doesn't exist
            sqlite3.IntegrityError: If this exact mapping already exists
        """
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

    def get_mappings_for_canonical(self, canonical_id: int) -> List[Dict[str, Any]]:
        """
        Get all term mappings for a canonical entity.

        Args:
            canonical_id: The canonical entity ID

        Returns:
            List of mapping dictionaries
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT id, raw_text, confidence_score, mapping_method, created_at
            FROM entity_mappings
            WHERE canonical_id = ?
            ORDER BY confidence_score DESC
        """, (canonical_id,))

        return [dict(row) for row in cursor.fetchall()]

    def get_all_mappings_with_canonicals(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Get all mappings with their canonical entity information for an entity type.

        Args:
            entity_type: Either 'intervention' or 'condition'

        Returns:
            List of dictionaries with mapping and canonical data
        """
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
        """
        Check if we have a cached LLM decision for this term.

        Args:
            term: The input term
            entity_type: Either 'intervention' or 'condition'
            candidate_canonicals: List of candidate canonical names
            model_name: Name of the LLM model used

        Returns:
            Dictionary with cached LLM result or None if not found
        """
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
        """
        Save LLM decision to cache.

        Args:
            term: The input term
            entity_type: Either 'intervention' or 'condition'
            candidate_canonicals: List of candidate canonical names
            llm_response: Raw LLM response
            match_result: Matched canonical name or None
            confidence: Confidence score
            reasoning: LLM reasoning
            model_name: Name of the LLM model used

        Raises:
            Exception: If caching fails (should be handled gracefully by caller)
        """
        cursor = self.db.cursor()

        candidates_json = json.dumps(sorted(candidate_canonicals or []), sort_keys=True) if candidate_canonicals else None

        cursor.execute("""
            INSERT OR REPLACE INTO llm_normalization_cache
            (input_term, entity_type, candidate_canonicals, llm_response, match_result,
             confidence_score, reasoning, model_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (term, entity_type, candidates_json, llm_response, match_result, confidence, reasoning, model_name))

        self.db.commit()

    # === STATISTICS OPERATIONS ===

    def get_mapping_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the current mapping state.

        Returns:
            Dictionary with counts, ratios, and other statistics
        """
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