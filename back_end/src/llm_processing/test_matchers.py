"""
Tests for Matching Strategies

This module tests all matching strategies (ExactMatcher, PatternMatcher, LLMMatcher).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from .matchers import (
    MatchResult, MatchingStrategy, ExactMatcher, PatternMatcher, LLMMatcher
)
from .repository import CanonicalRepository


class TestMatchResult:
    """Test suite for MatchResult class."""

    def test_match_result_creation(self):
        """Test creating a MatchResult."""
        result = MatchResult(
            canonical_id=1,
            canonical_name="ibuprofen",
            entity_type="intervention",
            confidence=0.95,
            method="exact_match",
            reasoning="Perfect match",
            metadata={"scientific_name": "test"}
        )

        assert result.canonical_id == 1
        assert result.canonical_name == "ibuprofen"
        assert result.confidence == 0.95
        assert result.method == "exact_match"

    def test_match_result_to_dict(self):
        """Test converting MatchResult to dictionary."""
        result = MatchResult(
            canonical_id=1,
            canonical_name="ibuprofen",
            entity_type="intervention",
            confidence=0.95,
            method="exact_match",
            reasoning="Perfect match"
        )

        result_dict = result.to_dict()
        assert result_dict['id'] == 1
        assert result_dict['canonical_name'] == "ibuprofen"
        assert result_dict['confidence'] == 0.95
        assert result_dict['match_method'] == "exact_match"


class TestMatchingStrategy:
    """Test suite for base MatchingStrategy class."""

    def test_normalize_term(self):
        """Test term normalization."""
        # Test basic normalization
        assert MatchingStrategy.normalize_term("  Ibuprofen  ") == "ibuprofen"
        assert MatchingStrategy.normalize_term("TYPE-2 diabetes") == "type-2 diabetes"

        # Test punctuation removal
        assert MatchingStrategy.normalize_term("diabetes, type 2") == "diabetes type 2"
        assert MatchingStrategy.normalize_term("arthritis (rheumatoid)") == "arthritis"

        # Test multiple spaces
        assert MatchingStrategy.normalize_term("joint   pain") == "joint pain"


class TestExactMatcher:
    """Test suite for ExactMatcher class."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository for testing."""
        repo = Mock(spec=CanonicalRepository)
        return repo

    @pytest.fixture
    def exact_matcher(self, mock_repository):
        """Create ExactMatcher instance with mock repository."""
        return ExactMatcher(mock_repository)

    def test_exact_match_canonical_name(self, exact_matcher, mock_repository):
        """Test exact matching against canonical names."""
        # Setup mock data
        mock_repository.get_all_canonical_entities.return_value = [
            {
                'id': 1,
                'canonical_name': 'ibuprofen',
                'entity_type': 'intervention'
            }
        ]
        mock_repository.find_mapping_by_term.return_value = None
        mock_repository.get_all_mappings_with_canonicals.return_value = []

        # Test exact match
        matches = exact_matcher.find_matches("Ibuprofen", "intervention")
        assert len(matches) == 1
        assert matches[0].canonical_id == 1
        assert matches[0].canonical_name == "ibuprofen"
        assert matches[0].confidence == 1.0
        assert matches[0].method == "exact_canonical"

    def test_exact_match_existing_mapping(self, exact_matcher, mock_repository):
        """Test exact matching against existing mappings."""
        # Setup mock data
        mock_repository.get_all_canonical_entities.return_value = []
        mock_repository.find_mapping_by_term.return_value = {
            'canonical_id': 1,
            'canonical_name': 'ibuprofen',
            'raw_text': 'advil'
        }

        # Test exact match with existing mapping
        matches = exact_matcher.find_matches("advil", "intervention")
        assert len(matches) == 1
        assert matches[0].canonical_id == 1
        assert matches[0].canonical_name == "ibuprofen"
        assert matches[0].confidence == 1.0
        assert matches[0].method == "existing_mapping"

    def test_no_exact_match(self, exact_matcher, mock_repository):
        """Test when no exact match is found."""
        # Setup mock data for no matches
        mock_repository.get_all_canonical_entities.return_value = []
        mock_repository.find_mapping_by_term.return_value = None
        mock_repository.get_all_mappings_with_canonicals.return_value = []

        matches = exact_matcher.find_matches("nonexistent", "intervention")
        assert len(matches) == 0


class TestPatternMatcher:
    """Test suite for PatternMatcher class."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository for testing."""
        repo = Mock(spec=CanonicalRepository)
        return repo

    @pytest.fixture
    def pattern_matcher(self, mock_repository):
        """Create PatternMatcher instance with mock repository."""
        return PatternMatcher(mock_repository)

    def test_safe_pluralization_matching(self, pattern_matcher, mock_repository):
        """Test safe plural/singular matching."""
        # Setup mock data
        mock_repository.get_all_mappings_with_canonicals.return_value = [
            {
                'id': 1,
                'canonical_name': 'probiotic',
                'entity_type': 'intervention',
                'raw_text': 'probiotic'
            }
        ]

        # Test plural addition
        matches = pattern_matcher.find_matches("probiotics", "intervention")
        assert len(matches) == 1
        assert matches[0].method == "safe_plural_removal"
        assert matches[0].confidence == 0.95

        # Test plural removal (search for singular when canonical is plural)
        mock_repository.get_all_mappings_with_canonicals.return_value = [
            {
                'id': 1,
                'canonical_name': 'probiotics',
                'entity_type': 'intervention',
                'raw_text': 'probiotics'
            }
        ]

        matches = pattern_matcher.find_matches("probiotic", "intervention")
        assert len(matches) == 1
        assert matches[0].method == "safe_plural_addition"

    def test_definite_article_removal(self, pattern_matcher, mock_repository):
        """Test definite article removal matching."""
        mock_repository.get_all_mappings_with_canonicals.return_value = [
            {
                'id': 1,
                'canonical_name': 'mediterranean diet',
                'entity_type': 'intervention',
                'raw_text': 'mediterranean diet'
            }
        ]

        matches = pattern_matcher.find_matches("the mediterranean diet", "intervention")
        assert len(matches) == 1
        assert matches[0].method == "definite_article_removal"
        assert matches[0].confidence == 0.90

    def test_spacing_punctuation_normalization(self, pattern_matcher, mock_repository):
        """Test spacing and punctuation normalization matching."""
        mock_repository.get_all_mappings_with_canonicals.return_value = [
            {
                'id': 1,
                'canonical_name': 'omega-3',
                'entity_type': 'intervention',
                'raw_text': 'omega-3'
            }
        ]

        matches = pattern_matcher.find_matches("omega 3", "intervention")
        assert len(matches) == 1
        assert matches[0].method == "spacing_punctuation_normalization"
        assert matches[0].confidence == 0.90

    def test_unsafe_pluralization_rejection(self, pattern_matcher):
        """Test that unsafe pluralization candidates are rejected."""
        # These should not be considered safe pluralization candidates
        assert not pattern_matcher._is_safe_pluralization_candidate("prebiotics", "probiotics")
        assert not pattern_matcher._is_safe_pluralization_candidate("cat", "cats")  # Too short
        assert not pattern_matcher._is_safe_pluralization_candidate("data", "datum")  # Not simple +s

        # These should be safe
        assert pattern_matcher._is_safe_pluralization_candidate("probiotic", "probiotics")
        assert pattern_matcher._is_safe_pluralization_candidate("supplement", "supplements")

    def test_deduplication(self, pattern_matcher, mock_repository):
        """Test that duplicate matches are properly deduplicated."""
        # Setup mock data that would create duplicate matches
        mock_repository.get_all_mappings_with_canonicals.return_value = [
            {
                'id': 1,
                'canonical_name': 'omega-3',
                'entity_type': 'intervention',
                'raw_text': 'omega 3'
            },
            {
                'id': 1,  # Same canonical ID
                'canonical_name': 'omega-3',
                'entity_type': 'intervention',
                'raw_text': 'omega3'
            }
        ]

        matches = pattern_matcher.find_matches("omega3", "intervention")
        # Should only get one match despite multiple potential pattern matches
        assert len(matches) == 1


class TestLLMMatcher:
    """Test suite for LLMMatcher class."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository for testing."""
        repo = Mock(spec=CanonicalRepository)
        return repo

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock()
        return client

    def test_llm_matcher_without_client(self, mock_repository):
        """Test LLMMatcher when LLM client is not available."""
        with patch('llm_processing.matchers.LLM_AVAILABLE', False):
            matcher = LLMMatcher(mock_repository)
            matches = matcher.find_matches("joint pain", "condition")
            assert len(matches) == 0

    @patch('llm_processing.matchers.LLM_AVAILABLE', True)
    @patch('llm_processing.matchers.get_llm_client')
    def test_llm_matcher_cached_result(self, mock_get_client, mock_repository):
        """Test LLM matcher with cached result."""
        # Setup mocks
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        mock_repository.get_all_canonical_entities.return_value = [
            {'canonical_name': 'arthritis'}
        ]
        mock_repository.find_llm_cache.return_value = {
            'match_result': 'arthritis',
            'confidence_score': 0.8,
            'reasoning': 'Joint pain is a symptom of arthritis'
        }
        mock_repository.find_canonical_by_name.return_value = {
            'id': 1,
            'canonical_name': 'arthritis'
        }

        matcher = LLMMatcher(mock_repository)
        matches = matcher.find_matches("joint pain", "condition")

        assert len(matches) == 1
        assert matches[0].canonical_id == 1
        assert matches[0].method == "llm_semantic_cached"
        assert matches[0].confidence == 0.8

    @patch('llm_processing.matchers.LLM_AVAILABLE', True)
    @patch('llm_processing.matchers.get_llm_client')
    def test_llm_matcher_fresh_result(self, mock_get_client, mock_repository):
        """Test LLM matcher with fresh result."""
        # Setup mocks
        mock_client = Mock()
        mock_client.generate.return_value = {
            'content': '{"match": "arthritis", "confidence": 0.85, "reasoning": "Joint pain is common in arthritis"}'
        }
        mock_get_client.return_value = mock_client

        mock_repository.get_all_canonical_entities.return_value = [
            {'canonical_name': 'arthritis'}
        ]
        mock_repository.find_llm_cache.return_value = None  # No cache
        mock_repository.find_canonical_by_name.return_value = {
            'id': 1,
            'canonical_name': 'arthritis'
        }

        matcher = LLMMatcher(mock_repository)
        matches = matcher.find_matches("joint pain", "condition")

        assert len(matches) == 1
        assert matches[0].canonical_id == 1
        assert matches[0].method == "llm_semantic"
        assert matches[0].confidence == 0.85

        # Verify that result was cached
        mock_repository.save_llm_cache.assert_called_once()

    @patch('llm_processing.matchers.LLM_AVAILABLE', True)
    @patch('llm_processing.matchers.get_llm_client')
    def test_llm_matcher_low_confidence_rejected(self, mock_get_client, mock_repository):
        """Test that low confidence LLM results are rejected."""
        # Setup mocks
        mock_client = Mock()
        mock_client.generate.return_value = {
            'content': '{"match": "arthritis", "confidence": 0.2, "reasoning": "Very uncertain match"}'
        }
        mock_get_client.return_value = mock_client

        mock_repository.get_all_canonical_entities.return_value = [
            {'canonical_name': 'arthritis'}
        ]
        mock_repository.find_llm_cache.return_value = None

        matcher = LLMMatcher(mock_repository)
        matches = matcher.find_matches("joint pain", "condition")

        # Should reject low confidence matches
        assert len(matches) == 0

    def test_llm_prompt_building(self, mock_repository):
        """Test LLM prompt construction."""
        matcher = LLMMatcher(mock_repository)

        prompt = matcher._build_llm_prompt(
            "joint pain",
            ["arthritis", "osteoarthritis"],
            "condition"
        )

        assert "joint pain" in prompt
        assert "arthritis" in prompt
        assert "osteoarthritis" in prompt
        assert "condition" in prompt
        assert "medical terminology expert" in prompt
        assert "probiotics != prebiotics" in prompt  # Safety warnings

    def test_llm_response_parsing(self, mock_repository):
        """Test LLM response parsing."""
        matcher = LLMMatcher(mock_repository)

        # Test valid JSON
        valid_json = '{"match": "arthritis", "confidence": 0.8, "reasoning": "test"}'
        parsed = matcher._parse_llm_response(valid_json)
        assert parsed['match'] == "arthritis"
        assert parsed['confidence'] == 0.8

        # Test invalid JSON
        invalid_json = "This is not JSON"
        parsed = matcher._parse_llm_response(invalid_json)
        assert parsed is None


if __name__ == "__main__":
    pytest.main([__file__])