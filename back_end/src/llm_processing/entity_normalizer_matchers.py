"""
Matching Strategies for Entity Normalization

This module implements different matching strategies using the strategy pattern,
allowing for modular and testable entity matching approaches.
"""

import re
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .repository import CanonicalRepository

# Import LLM functionality
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from data.api_clients import get_llm_client
    from data.utils import parse_json_safely
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class MatchResult:
    """
    Standardized result from any matching strategy.

    This class ensures consistent return types across all matchers.
    """

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
            'canonical_name': self.canonical_name,
            'entity_type': self.entity_type,
            'confidence': self.confidence,
            'match_method': self.method,
            'reasoning': self.reasoning,
            'metadata': self.metadata
        }


class MatchingStrategy(ABC):
    """
    Abstract base class for all matching strategies.

    This defines the common interface that all matchers must implement.
    """

    def __init__(self, repository: CanonicalRepository):
        self.repository = repository

    @abstractmethod
    def find_matches(self, term: str, entity_type: str) -> List[MatchResult]:
        """
        Find matches for a term using this strategy.

        Args:
            term: The term to find matches for
            entity_type: Either 'intervention' or 'condition'

        Returns:
            List of MatchResult objects, sorted by confidence descending
        """
        pass

    @staticmethod
    def normalize_term(term: str) -> str:
        """
        Normalize a term by lowercasing, stripping whitespace, and removing punctuation.

        This is a shared utility method used by multiple matchers.

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


class ExactMatcher(MatchingStrategy):
    """
    Strategy for exact normalized matching.

    This matcher finds entities where the normalized forms are identical,
    which is the safest and fastest matching method.
    """

    def find_matches(self, term: str, entity_type: str) -> List[MatchResult]:
        """
        Find exact normalized matches for a term.

        Args:
            term: The term to search for
            entity_type: Either 'intervention' or 'condition'

        Returns:
            List of MatchResult objects (at most one for exact matches)
        """
        normalized_term = self.normalize_term(term)

        # Check against canonical names
        canonical_entities = self.repository.get_all_canonical_entities(entity_type)
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
        existing_mapping = self.repository.find_mapping_by_term(term, entity_type)
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
        all_mappings = self.repository.get_all_mappings_with_canonicals(entity_type)
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


class PatternMatcher(MatchingStrategy):
    """
    Strategy for safe pattern-based matching.

    This matcher only applies very safe transformations that are unlikely
    to cause false positives in medical terminology.
    """

    def find_matches(self, term: str, entity_type: str) -> List[MatchResult]:
        """
        Find matches using safe pattern matching.

        Args:
            term: The term to search for
            entity_type: Either 'intervention' or 'condition'

        Returns:
            List of MatchResult objects, sorted by confidence descending
        """
        normalized_term = self.normalize_term(term)
        matches = []

        # Get all canonical entities and mappings for this entity type
        all_mappings = self.repository.get_all_mappings_with_canonicals(entity_type)

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
        """
        Check if term matches target using safe patterns.

        Args:
            term: Normalized term to match
            target: Normalized target to match against
            mapping: Database mapping record
            match_type: 'canonical' or 'mapping'

        Returns:
            MatchResult if match found, None otherwise
        """
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
        """
        Check if two terms are safe candidates for pluralization matching.

        Prevents dangerous matches like 'prebiotics'/'probiotics' by checking
        that the terms are very similar except for pluralization.

        Args:
            term1: First term
            term2: Second term

        Returns:
            True if safe to apply pluralization rules
        """
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


class LLMMatcher(MatchingStrategy):
    """
    Strategy for LLM-based semantic matching.

    This matcher uses an LLM to identify medical synonyms and related terms
    that pattern matching might miss, while being conservative to avoid
    dangerous false positives.
    """

    def __init__(self, repository: CanonicalRepository, llm_model: str = "gemma2:9b"):
        super().__init__(repository)
        self.llm_model = llm_model

        # Initialize LLM client if available
        if LLM_AVAILABLE:
            self.llm_client = get_llm_client(llm_model)
        else:
            self.llm_client = None

    def find_matches(self, term: str, entity_type: str,
                    candidate_canonicals: Optional[List[str]] = None) -> List[MatchResult]:
        """
        Find matches using LLM semantic understanding.

        Args:
            term: The term to find matches for
            entity_type: Either 'intervention' or 'condition'
            candidate_canonicals: Specific canonicals to test against (optional)

        Returns:
            List of MatchResult objects (at most one for LLM matches)
        """
        if not LLM_AVAILABLE or not self.llm_client:
            return []

        # Get candidate canonicals if not provided
        if candidate_canonicals is None:
            canonical_entities = self.repository.get_all_canonical_entities(entity_type)
            candidate_canonicals = [entity['canonical_name'] for entity in canonical_entities]

        if not candidate_canonicals:
            return []

        # Check cache first
        cached_result = self.repository.find_llm_cache(term, entity_type, candidate_canonicals, self.llm_model)
        if cached_result and cached_result['match_result']:
            canonical_entity = self.repository.find_canonical_by_name(
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
                self.repository.save_llm_cache(
                    term, entity_type, candidate_canonicals,
                    llm_content, match_name, confidence, reasoning, self.llm_model
                )
            except Exception as e:
                # Don't fail the match if caching fails
                print(f"Warning: Failed to cache LLM decision: {e}")

            # If we have a confident match, return it
            if match_name and confidence > 0.3:  # Minimum confidence threshold
                canonical_entity = self.repository.find_canonical_by_name(match_name, entity_type)
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
            print(f"Error in LLM matching for term '{term}': {e}")
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
                    print(f"Warning: Invalid LLM response format")
                    return None