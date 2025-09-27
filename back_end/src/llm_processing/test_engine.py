"""
Tests for MatchingEngine

This module tests the MatchingEngine that orchestrates different matching strategies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from .engine import MatchingEngine, MatchingMode
from .repository import CanonicalRepository
from .matchers import MatchResult, ExactMatcher, PatternMatcher, LLMMatcher


class TestMatchingEngine:
    """Test suite for MatchingEngine class."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository for testing."""
        repo = Mock(spec=CanonicalRepository)
        return repo

    @pytest.fixture
    def mock_exact_matcher(self):
        """Create a mock ExactMatcher."""
        matcher = Mock(spec=ExactMatcher)
        return matcher

    @pytest.fixture
    def mock_pattern_matcher(self):
        """Create a mock PatternMatcher."""
        matcher = Mock(spec=PatternMatcher)
        return matcher

    @pytest.fixture
    def mock_llm_matcher(self):
        """Create a mock LLMMatcher."""
        matcher = Mock(spec=LLMMatcher)
        matcher.llm_client = Mock()  # Simulate LLM availability
        return matcher

    @pytest.fixture
    def matching_engine(self, mock_repository):
        """Create MatchingEngine instance with mocked components."""
        with patch('llm_processing.engine.ExactMatcher') as mock_exact_cls, \
             patch('llm_processing.engine.PatternMatcher') as mock_pattern_cls, \
             patch('llm_processing.engine.LLMMatcher') as mock_llm_cls:

            engine = MatchingEngine(mock_repository)

            # Replace the matchers with mocks
            engine.exact_matcher = Mock(spec=ExactMatcher)
            engine.pattern_matcher = Mock(spec=PatternMatcher)
            engine.llm_matcher = Mock(spec=LLMMatcher)
            engine.llm_matcher.llm_client = Mock()

            return engine

    def test_safe_only_mode_with_exact_match(self, matching_engine):
        """Test SAFE_ONLY mode when exact match is found."""
        # Setup exact match result
        exact_result = MatchResult(1, "ibuprofen", "intervention", 1.0, "exact_match")
        matching_engine.exact_matcher.find_matches.return_value = [exact_result]

        matches = matching_engine.find_matches("ibuprofen", "intervention", MatchingMode.SAFE_ONLY)

        assert len(matches) == 1
        assert matches[0].canonical_id == 1
        assert matches[0].method == "exact_match"

        # Verify exact matcher was called
        matching_engine.exact_matcher.find_matches.assert_called_once_with("ibuprofen", "intervention")
        # Verify pattern matcher was NOT called (since exact match found)
        matching_engine.pattern_matcher.find_matches.assert_not_called()

    def test_safe_only_mode_fallback_to_pattern(self, matching_engine):
        """Test SAFE_ONLY mode fallback to pattern matching."""
        # Setup no exact match, but pattern match exists
        matching_engine.exact_matcher.find_matches.return_value = []
        pattern_result = MatchResult(2, "probiotics", "intervention", 0.95, "plural_match")
        matching_engine.pattern_matcher.find_matches.return_value = [pattern_result]

        matches = matching_engine.find_matches("probiotic", "intervention", MatchingMode.SAFE_ONLY)

        assert len(matches) == 1
        assert matches[0].canonical_id == 2
        assert matches[0].method == "plural_match"

        # Verify both matchers were called
        matching_engine.exact_matcher.find_matches.assert_called_once()
        matching_engine.pattern_matcher.find_matches.assert_called_once()

    def test_llm_only_mode(self, matching_engine):
        """Test LLM_ONLY mode."""
        # Setup LLM match result
        llm_result = MatchResult(3, "arthritis", "condition", 0.85, "llm_semantic")
        matching_engine.llm_matcher.find_matches.return_value = [llm_result]

        matches = matching_engine.find_matches("joint pain", "condition", MatchingMode.LLM_ONLY)

        assert len(matches) == 1
        assert matches[0].canonical_id == 3
        assert matches[0].method == "llm_semantic"

        # Verify only LLM matcher was called
        matching_engine.llm_matcher.find_matches.assert_called_once_with("joint pain", "condition")
        matching_engine.exact_matcher.find_matches.assert_not_called()
        matching_engine.pattern_matcher.find_matches.assert_not_called()

    def test_llm_only_mode_low_confidence_filtered(self, matching_engine):
        """Test LLM_ONLY mode filters out low confidence results."""
        # Setup low confidence LLM result
        llm_result = MatchResult(3, "arthritis", "condition", 0.2, "llm_semantic")
        matching_engine.llm_matcher.find_matches.return_value = [llm_result]

        matches = matching_engine.find_matches(
            "joint pain", "condition", MatchingMode.LLM_ONLY, confidence_threshold=0.5
        )

        # Should be filtered out due to low confidence
        assert len(matches) == 0

    def test_comprehensive_mode_hierarchical_fallback(self, matching_engine):
        """Test COMPREHENSIVE mode with hierarchical fallback."""
        # Setup: no exact match, no pattern match, but LLM match exists
        matching_engine.exact_matcher.find_matches.return_value = []
        matching_engine.pattern_matcher.find_matches.return_value = []
        llm_result = MatchResult(4, "hypertension", "condition", 0.8, "llm_semantic")
        matching_engine.llm_matcher.find_matches.return_value = [llm_result]

        matches = matching_engine.find_matches("high blood pressure", "condition", MatchingMode.COMPREHENSIVE)

        assert len(matches) == 1
        assert matches[0].canonical_id == 4
        assert matches[0].method == "llm_semantic"

        # Verify all matchers were called in order
        matching_engine.exact_matcher.find_matches.assert_called_once()
        matching_engine.pattern_matcher.find_matches.assert_called_once()
        matching_engine.llm_matcher.find_matches.assert_called_once()

    def test_comprehensive_mode_stops_at_exact_match(self, matching_engine):
        """Test COMPREHENSIVE mode stops at exact match."""
        # Setup exact match
        exact_result = MatchResult(1, "ibuprofen", "intervention", 1.0, "exact_match")
        matching_engine.exact_matcher.find_matches.return_value = [exact_result]

        matches = matching_engine.find_matches("ibuprofen", "intervention", MatchingMode.COMPREHENSIVE)

        assert len(matches) == 1
        assert matches[0].method == "exact_match"

        # Verify only exact matcher was called
        matching_engine.exact_matcher.find_matches.assert_called_once()
        matching_engine.pattern_matcher.find_matches.assert_not_called()
        matching_engine.llm_matcher.find_matches.assert_not_called()

    def test_deduplication_and_sorting(self, matching_engine):
        """Test deduplication and sorting of results."""
        # Create multiple results for same canonical ID with different methods
        exact_result = MatchResult(1, "ibuprofen", "intervention", 1.0, "exact_match")
        pattern_result = MatchResult(1, "ibuprofen", "intervention", 0.95, "pattern_match")
        llm_result = MatchResult(2, "aspirin", "intervention", 0.8, "llm_semantic")

        # Mock all matchers to return results (simulating no early termination)
        matching_engine.exact_matcher.find_matches.return_value = []
        matching_engine.pattern_matcher.find_matches.return_value = []
        matching_engine.llm_matcher.find_matches.return_value = [exact_result, pattern_result, llm_result]

        matches = matching_engine.find_matches("test", "intervention", MatchingMode.LLM_ONLY)

        # Should deduplicate and keep the safer match (exact over pattern)
        assert len(matches) == 2  # One for each unique canonical ID
        canonical_ids = [m.canonical_id for m in matches]
        assert 1 in canonical_ids
        assert 2 in canonical_ids

    def test_find_or_create_mapping_existing_match(self, matching_engine, mock_repository):
        """Test find_or_create_mapping with existing match."""
        # Setup existing match
        match_result = MatchResult(1, "ibuprofen", "intervention", 0.95, "exact_match", "Perfect match")
        matching_engine.exact_matcher.find_matches.return_value = [match_result]
        matching_engine.pattern_matcher.find_matches.return_value = []
        matching_engine.llm_matcher.find_matches.return_value = []

        result = matching_engine.find_or_create_mapping("ibuprofen", "intervention")

        assert result['canonical_id'] == 1
        assert result['canonical_name'] == "ibuprofen"
        assert result['method'] == "exact_match"
        assert result['confidence'] == 0.95
        assert result['is_new'] is False
        assert result['reasoning'] == "Perfect match"

    def test_find_or_create_mapping_create_new(self, matching_engine, mock_repository):
        """Test find_or_create_mapping creates new canonical entity."""
        # Setup no matches found
        matching_engine.exact_matcher.find_matches.return_value = []
        matching_engine.pattern_matcher.find_matches.return_value = []
        matching_engine.llm_matcher.find_matches.return_value = []

        # Setup repository to create new entity
        mock_repository.create_canonical_entity.return_value = 5
        mock_repository.create_mapping.return_value = 10

        result = matching_engine.find_or_create_mapping("new_term", "intervention")

        assert result['canonical_id'] == 5
        assert result['canonical_name'] == "new_term"
        assert result['method'] == "new_canonical"
        assert result['confidence'] == 1.0
        assert result['is_new'] is True

        # Verify repository calls
        mock_repository.create_canonical_entity.assert_called_once_with("new_term", "intervention")
        mock_repository.create_mapping.assert_called_once_with("new_term", 5, 1.0, "exact_canonical")

    def test_find_or_create_mapping_empty_term(self, matching_engine):
        """Test find_or_create_mapping with empty term."""
        result = matching_engine.find_or_create_mapping("", "intervention")

        assert result['canonical_id'] is None
        assert result['method'] == "empty_term"
        assert result['confidence'] == 0.0
        assert result['is_new'] is False

    def test_find_or_create_mapping_creation_error(self, matching_engine, mock_repository):
        """Test find_or_create_mapping when creation fails."""
        # Setup no matches found
        matching_engine.exact_matcher.find_matches.return_value = []
        matching_engine.pattern_matcher.find_matches.return_value = []
        matching_engine.llm_matcher.find_matches.return_value = []

        # Setup repository to raise error
        mock_repository.create_canonical_entity.side_effect = Exception("Database error")

        result = matching_engine.find_or_create_mapping("test_term", "intervention")

        assert result['canonical_id'] is None
        assert result['method'] == "error"
        assert result['confidence'] == 0.0
        assert result['is_new'] is False
        assert "Database error" in result['reasoning']

    def test_batch_find_matches(self, matching_engine):
        """Test batch processing of multiple terms."""
        # Setup different results for different terms
        def side_effect(term, entity_type, mode):
            if term == "ibuprofen":
                return [MatchResult(1, "ibuprofen", "intervention", 1.0, "exact")]
            elif term == "aspirin":
                return [MatchResult(2, "aspirin", "intervention", 0.95, "pattern")]
            else:
                return []

        matching_engine.find_matches = Mock(side_effect=side_effect)

        results = matching_engine.batch_find_matches(
            ["ibuprofen", "aspirin", "nonexistent"],
            "intervention"
        )

        assert len(results) == 2  # Only terms with matches
        assert "ibuprofen" in results
        assert "aspirin" in results
        assert "nonexistent" not in results

        assert results["ibuprofen"][0].canonical_id == 1
        assert results["aspirin"][0].canonical_id == 2

    def test_get_matching_statistics(self, matching_engine, mock_repository):
        """Test getting matching engine statistics."""
        mock_repository.get_mapping_statistics.return_value = {
            'canonical_entities': {'intervention': 10},
            'mappings': {'intervention': 25}
        }

        stats = matching_engine.get_matching_statistics()

        assert 'strategies_available' in stats
        assert 'llm_model' in stats
        assert 'matching_modes' in stats
        assert 'repository_stats' in stats

        assert stats['strategies_available']['exact_matcher'] is True
        assert stats['strategies_available']['pattern_matcher'] is True
        assert stats['strategies_available']['llm_matcher'] is True

    def test_safety_priority_ordering(self, matching_engine):
        """Test that matches are ordered by safety priority."""
        # Create matches with different safety levels
        exact_match = MatchResult(1, "test", "intervention", 0.9, "exact_canonical")
        llm_match = MatchResult(2, "test2", "intervention", 0.95, "llm_semantic")  # Higher confidence but less safe
        pattern_match = MatchResult(3, "test3", "intervention", 0.85, "safe_plural_removal")

        matches = [llm_match, exact_match, pattern_match]  # Unsorted order

        sorted_matches = matching_engine._deduplicate_and_sort(matches)

        # Should be sorted by safety priority first, then confidence
        assert sorted_matches[0].method == "exact_canonical"  # Highest safety
        assert sorted_matches[1].method == "safe_plural_removal"  # Medium safety
        assert sorted_matches[2].method == "llm_semantic"  # Lowest safety (but still above threshold)


if __name__ == "__main__":
    pytest.main([__file__])