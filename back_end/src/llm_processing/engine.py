"""
Matching Engine for Entity Normalization

This module provides the orchestration layer that coordinates different
matching strategies and implements the fallback pattern from fast/safe
to slow/intelligent methods.
"""

from typing import List, Dict, Any, Optional, Set
from enum import Enum
from .repository import CanonicalRepository
from .matchers import MatchingStrategy, MatchResult, ExactMatcher, PatternMatcher, LLMMatcher


class MatchingMode(Enum):
    """
    Different modes for matching entities.

    SAFE_ONLY: Use only exact and pattern matching (no LLM)
    COMPREHENSIVE: Use all available strategies
    LLM_ONLY: Use only LLM matching (for testing/debugging)
    """
    SAFE_ONLY = "safe_only"
    COMPREHENSIVE = "comprehensive"
    LLM_ONLY = "llm_only"


class MatchingEngine:
    """
    Orchestrates different matching strategies to find entity matches.

    This class implements the hierarchical fallback pattern:
    1. Fast exact matching
    2. Safe pattern matching
    3. LLM semantic matching (if enabled)

    The engine prevents duplicate results and prioritizes safer matches.
    """

    def __init__(self, repository: CanonicalRepository, llm_model: str = "gemma2:9b"):
        """
        Initialize the matching engine with strategies.

        Args:
            repository: Database repository for entity operations
            llm_model: LLM model name for semantic matching
        """
        self.repository = repository

        # Initialize all available strategies
        self.exact_matcher = ExactMatcher(repository)
        self.pattern_matcher = PatternMatcher(repository)
        self.llm_matcher = LLMMatcher(repository, llm_model)

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

        all_matches = []

        if mode == MatchingMode.SAFE_ONLY:
            # Only use exact and pattern matching
            all_matches.extend(self.exact_matcher.find_matches(term, entity_type))
            if not all_matches:  # Only try pattern matching if no exact match
                all_matches.extend(self.pattern_matcher.find_matches(term, entity_type))

        elif mode == MatchingMode.LLM_ONLY:
            # Only use LLM matching (for testing/debugging)
            llm_matches = self.llm_matcher.find_matches(term, entity_type)
            all_matches.extend([m for m in llm_matches if m.confidence >= confidence_threshold])

        elif mode == MatchingMode.COMPREHENSIVE:
            # Use all strategies in hierarchical order

            # 1. Try exact matching first (fastest and safest)
            exact_matches = self.exact_matcher.find_matches(term, entity_type)
            all_matches.extend(exact_matches)

            # 2. If no exact matches, try pattern matching
            if not exact_matches:
                pattern_matches = self.pattern_matcher.find_matches(term, entity_type)
                all_matches.extend(pattern_matches)

                # 3. If still no matches, try LLM matching
                if not pattern_matches:
                    llm_matches = self.llm_matcher.find_matches(term, entity_type)
                    all_matches.extend([m for m in llm_matches if m.confidence >= confidence_threshold])

        # Remove duplicates and sort results
        return self._deduplicate_and_sort(all_matches)

    def find_safe_matches_only(self, term: str, entity_type: str) -> List[MatchResult]:
        """
        Find matches using only safe methods (exact + pattern matching).

        This is the recommended method for production medical term matching
        where false positives must be minimized.

        Args:
            term: The term to find matches for
            entity_type: Either 'intervention' or 'condition'

        Returns:
            List of safe MatchResult objects
        """
        return self.find_matches(term, entity_type, mode=MatchingMode.SAFE_ONLY)

    def find_comprehensive_matches(self, term: str, entity_type: str,
                                 confidence_threshold: float = 0.7) -> List[MatchResult]:
        """
        Find matches using all available strategies.

        Uses the hierarchical fallback pattern: exact → pattern → LLM.

        Args:
            term: The term to find matches for
            entity_type: Either 'intervention' or 'condition'
            confidence_threshold: Minimum confidence for LLM matches

        Returns:
            List of all MatchResult objects found
        """
        return self.find_matches(term, entity_type,
                               mode=MatchingMode.COMPREHENSIVE,
                               confidence_threshold=confidence_threshold)

    def find_or_create_mapping(self, term: str, entity_type: str,
                             confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Find a match or create a new canonical entity if none found.

        This is the main method for the extraction pipeline that ensures
        every term gets mapped to a canonical entity.

        Args:
            term: The term to normalize
            entity_type: Either 'intervention' or 'condition'
            confidence_threshold: Minimum confidence for LLM matching

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

        # Try to find matches using comprehensive strategy
        matches = self.find_comprehensive_matches(term, entity_type, confidence_threshold)

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
        print(f"Creating new canonical entity for: {term} ({entity_type})")

        try:
            canonical_id = self.repository.create_canonical_entity(term, entity_type)

            # Add the term as its own canonical mapping
            self.repository.create_mapping(term, canonical_id, 1.0, "exact_canonical")

            return {
                'canonical_id': canonical_id,
                'canonical_name': term,
                'method': 'new_canonical',
                'confidence': 1.0,
                'is_new': True,
                'reasoning': 'No existing match found, created new canonical entity'
            }

        except Exception as e:
            print(f"Error creating canonical entity for '{term}': {e}")
            return {
                'canonical_id': None,
                'canonical_name': term,
                'method': 'error',
                'confidence': 0.0,
                'is_new': False,
                'reasoning': f'Error creating canonical entity: {e}'
            }

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

        for term in terms_list:
            if term and term.strip():
                matches = self.find_matches(term.strip(), entity_type, mode)
                if matches:
                    results[term] = matches

        return results

    def get_matching_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the matching engine configuration and capabilities.

        Returns:
            Dictionary with engine statistics and configuration
        """
        return {
            'strategies_available': {
                'exact_matcher': True,
                'pattern_matcher': True,
                'llm_matcher': self.llm_matcher.llm_client is not None
            },
            'llm_model': self.llm_matcher.llm_model if self.llm_matcher.llm_client else None,
            'matching_modes': [mode.value for mode in MatchingMode],
            'repository_stats': self.repository.get_mapping_statistics()
        }

    def _deduplicate_and_sort(self, matches: List[MatchResult]) -> List[MatchResult]:
        """
        Remove duplicate matches and sort by safety and confidence.

        Prioritizes safer matching methods even if they have slightly lower confidence.

        Args:
            matches: List of MatchResult objects

        Returns:
            Deduplicated and sorted list of MatchResult objects
        """
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
        """
        Determine if match1 is safer than match2.

        Args:
            match1: First match to compare
            match2: Second match to compare

        Returns:
            True if match1 is safer than match2
        """
        priority1 = self._get_safety_priority(match1)
        priority2 = self._get_safety_priority(match2)

        if priority1 != priority2:
            return priority1 > priority2
        else:
            # Same safety level, prefer higher confidence
            return match1.confidence > match2.confidence

    def _get_safety_priority(self, match: MatchResult) -> int:
        """
        Get safety priority for a match (higher = safer).

        Args:
            match: MatchResult to evaluate

        Returns:
            Integer priority (higher = safer)
        """
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