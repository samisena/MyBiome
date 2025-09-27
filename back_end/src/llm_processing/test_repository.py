"""
Tests for CanonicalRepository

This module tests all database operations in the repository layer.
"""

import sqlite3
import json
import pytest
from unittest.mock import Mock, patch
from .repository import CanonicalRepository


class TestCanonicalRepository:
    """Test suite for CanonicalRepository class."""

    @pytest.fixture
    def db_connection(self):
        """Create an in-memory database for testing."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row

        # Create test tables
        conn.execute("""
            CREATE TABLE canonical_entities (
                id INTEGER PRIMARY KEY,
                canonical_name TEXT UNIQUE NOT NULL,
                entity_type TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE entity_mappings (
                id INTEGER PRIMARY KEY,
                canonical_id INTEGER NOT NULL,
                raw_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                mapping_method TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (canonical_id) REFERENCES canonical_entities(id),
                UNIQUE(canonical_id, raw_text)
            )
        """)

        conn.execute("""
            CREATE TABLE llm_normalization_cache (
                id INTEGER PRIMARY KEY,
                input_term TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                candidate_canonicals TEXT,
                llm_response TEXT NOT NULL,
                match_result TEXT,
                confidence_score REAL NOT NULL,
                reasoning TEXT,
                model_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        return conn

    @pytest.fixture
    def repository(self, db_connection):
        """Create repository instance with test database."""
        return CanonicalRepository(db_connection)

    def test_create_canonical_entity(self, repository):
        """Test creating a new canonical entity."""
        canonical_id = repository.create_canonical_entity(
            canonical_name="ibuprofen",
            entity_type="intervention",
            scientific_name="2-(4-isobutylphenyl)propionic acid"
        )

        assert canonical_id is not None
        assert isinstance(canonical_id, int)
        assert canonical_id > 0

        # Verify the entity was created
        entity = repository.find_canonical_by_id(canonical_id)
        assert entity is not None
        assert entity['canonical_name'] == "ibuprofen"
        assert entity['entity_type'] == "intervention"

        # Verify metadata
        metadata = json.loads(entity['metadata'])
        assert metadata['scientific_name'] == "2-(4-isobutylphenyl)propionic acid"

    def test_create_canonical_entity_without_metadata(self, repository):
        """Test creating a canonical entity without metadata."""
        canonical_id = repository.create_canonical_entity(
            canonical_name="diabetes",
            entity_type="condition"
        )

        entity = repository.find_canonical_by_id(canonical_id)
        assert entity['metadata'] is None

    def test_create_duplicate_canonical_entity(self, repository):
        """Test that creating duplicate canonical entities raises an error."""
        repository.create_canonical_entity("ibuprofen", "intervention")

        with pytest.raises(sqlite3.IntegrityError):
            repository.create_canonical_entity("ibuprofen", "intervention")

    def test_find_canonical_by_name(self, repository):
        """Test finding canonical entity by name."""
        canonical_id = repository.create_canonical_entity("aspirin", "intervention")

        entity = repository.find_canonical_by_name("aspirin", "intervention")
        assert entity is not None
        assert entity['id'] == canonical_id
        assert entity['canonical_name'] == "aspirin"

        # Test case sensitivity
        entity = repository.find_canonical_by_name("Aspirin", "intervention")
        assert entity is None  # Should be case sensitive

        # Test wrong entity type
        entity = repository.find_canonical_by_name("aspirin", "condition")
        assert entity is None

    def test_search_canonical_entities(self, repository):
        """Test searching canonical entities."""
        # Create test entities
        repository.create_canonical_entity("diabetes type 1", "condition")
        repository.create_canonical_entity("diabetes type 2", "condition")
        repository.create_canonical_entity("diabetic neuropathy", "condition")
        repository.create_canonical_entity("insulin", "intervention")

        # Search with entity type filter
        results = repository.search_canonical_entities("diabetes", "condition")
        assert len(results) == 3
        assert all("diabetes" in result['canonical_name'] for result in results)

        # Search without entity type filter
        results = repository.search_canonical_entities("diabetes")
        assert len(results) == 3

        # Search with no matches
        results = repository.search_canonical_entities("nonexistent", "condition")
        assert len(results) == 0

    def test_create_mapping(self, repository):
        """Test creating a term mapping."""
        # Create canonical entity first
        canonical_id = repository.create_canonical_entity("ibuprofen", "intervention")

        # Create mapping
        mapping_id = repository.create_mapping(
            original_term="advil",
            canonical_id=canonical_id,
            confidence=0.95,
            method="manual"
        )

        assert mapping_id is not None
        assert isinstance(mapping_id, int)
        assert mapping_id > 0

        # Verify mapping exists
        mapping = repository.find_mapping_by_term("advil", "intervention")
        assert mapping is not None
        assert mapping['canonical_id'] == canonical_id
        assert mapping['canonical_name'] == "ibuprofen"
        assert mapping['confidence_score'] == 0.95

    def test_create_mapping_invalid_canonical_id(self, repository):
        """Test creating mapping with invalid canonical ID."""
        with pytest.raises(ValueError, match="Canonical entity with ID 999 not found"):
            repository.create_mapping("test", 999, 0.5, "test")

    def test_get_mappings_for_canonical(self, repository):
        """Test getting all mappings for a canonical entity."""
        canonical_id = repository.create_canonical_entity("ibuprofen", "intervention")

        # Create multiple mappings
        repository.create_mapping("advil", canonical_id, 0.95, "manual")
        repository.create_mapping("motrin", canonical_id, 0.90, "pattern")
        repository.create_mapping("ibuprofeno", canonical_id, 0.85, "manual")

        mappings = repository.get_mappings_for_canonical(canonical_id)
        assert len(mappings) == 3

        # Should be sorted by confidence descending
        confidences = [m['confidence_score'] for m in mappings]
        assert confidences == sorted(confidences, reverse=True)

    def test_llm_cache_operations(self, repository):
        """Test LLM caching functionality."""
        # Save cache entry
        repository.save_llm_cache(
            term="joint pain",
            entity_type="condition",
            candidate_canonicals=["arthritis", "osteoarthritis"],
            llm_response='{"match": "arthritis", "confidence": 0.8}',
            match_result="arthritis",
            confidence=0.8,
            reasoning="Joint pain is a common symptom of arthritis",
            model_name="test-model"
        )

        # Retrieve cache entry
        cached = repository.find_llm_cache(
            term="joint pain",
            entity_type="condition",
            candidate_canonicals=["arthritis", "osteoarthritis"],
            model_name="test-model"
        )

        assert cached is not None
        assert cached['match_result'] == "arthritis"
        assert cached['confidence_score'] == 0.8
        assert "arthritis" in cached['reasoning']

        # Test cache miss
        cached_miss = repository.find_llm_cache(
            term="different term",
            entity_type="condition",
            candidate_canonicals=["arthritis"],
            model_name="test-model"
        )
        assert cached_miss is None

    def test_get_mapping_statistics(self, repository):
        """Test getting mapping statistics."""
        # Create test data
        intervention_id = repository.create_canonical_entity("ibuprofen", "intervention")
        condition_id = repository.create_canonical_entity("arthritis", "condition")

        repository.create_mapping("advil", intervention_id, 0.95, "manual")
        repository.create_mapping("motrin", intervention_id, 0.90, "pattern")
        repository.create_mapping("joint pain", condition_id, 0.85, "manual")

        stats = repository.get_mapping_statistics()

        assert 'canonical_entities' in stats
        assert 'mappings' in stats
        assert 'ratios' in stats

        assert stats['canonical_entities']['intervention'] == 1
        assert stats['canonical_entities']['condition'] == 1
        assert stats['mappings']['intervention'] == 2
        assert stats['mappings']['condition'] == 1

        # Check compression ratios
        intervention_ratio = stats['ratios']['intervention']
        assert intervention_ratio['unique_terms'] == 2
        assert intervention_ratio['canonical_entities'] == 1
        assert intervention_ratio['compression_ratio'] == 2.0

    def test_get_all_mappings_with_canonicals(self, repository):
        """Test getting all mappings with canonical information."""
        # Create test data
        canonical_id = repository.create_canonical_entity("diabetes", "condition")
        repository.create_mapping("type 2 diabetes", canonical_id, 0.95, "manual")

        mappings = repository.get_all_mappings_with_canonicals("condition")
        assert len(mappings) >= 1

        found_mapping = next((m for m in mappings if m['canonical_name'] == "diabetes"), None)
        assert found_mapping is not None
        assert found_mapping['raw_text'] == "type 2 diabetes"


if __name__ == "__main__":
    pytest.main([__file__])