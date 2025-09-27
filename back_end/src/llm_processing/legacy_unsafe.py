"""
LEGACY UNSAFE MATCHING METHODS

⚠️  WARNING: DANGEROUS FOR MEDICAL TERMINOLOGY ⚠️

This module contains deprecated string similarity matching methods that are
UNSAFE for medical terminology and can cause dangerous false positives.

These methods are preserved only for backward compatibility and research purposes.
DO NOT USE THESE METHODS IN PRODUCTION FOR MEDICAL DATA.

Known dangerous false positives:
- probiotics vs prebiotics (different substances)
- hyperglycemia vs hypoglycemia (opposite conditions)
- hypertension vs hypotension (opposite conditions)
- Any similar-sounding medical terms with different meanings

USE AT YOUR OWN RISK - YOU HAVE BEEN WARNED!
"""

import re
from typing import List, Dict, Any, Optional
from .repository import CanonicalRepository


class UnsafeStringMatcher:
    """
    DEPRECATED AND DANGEROUS string similarity matcher.

    ⚠️  WARNING: This class contains methods that are unsafe for medical terminology.
    These methods can cause dangerous false positives that could lead to incorrect
    medical term matching.

    This class is preserved only for:
    1. Backward compatibility with legacy code
    2. Research into why string similarity fails for medical terms
    3. Comparison studies between safe and unsafe methods

    DO NOT USE IN PRODUCTION FOR MEDICAL DATA!
    """

    def __init__(self, repository: CanonicalRepository):
        """
        Initialize the unsafe matcher.

        Args:
            repository: Database repository for entity operations
        """
        self.repository = repository

        # Log warning whenever this class is instantiated
        print("⚠️  WARNING: UnsafeStringMatcher instantiated - dangerous for medical terms!")

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

    def calculate_string_similarity(self, term1: str, term2: str) -> float:
        """
        DEPRECATED: String similarity matching is too dangerous for medical terms!

        ⚠️  WARNING: This method can cause dangerous false positives in medical terminology.

        Examples of dangerous false positives:
        - probiotics vs prebiotics (different substances)
        - hyperglycemia vs hypoglycemia (opposite conditions)
        - hypertension vs hypotension (opposite conditions)

        This method is kept only for backward compatibility and research purposes.
        DO NOT USE FOR MEDICAL TERM MATCHING!

        Args:
            term1: First term
            term2: Second term

        Returns:
            Similarity score between 0.0 and 1.0
        """
        print(f"⚠️  WARNING: Using dangerous similarity calculation for '{term1}' vs '{term2}'")

        # Normalize both terms
        norm1 = self.normalize_term(term1)
        norm2 = self.normalize_term(term2)

        if norm1 == norm2:
            return 1.0

        # For medical terms, we should be extremely conservative
        # Only return high similarity for very minor differences

        # Check for safe transformations only
        if self._is_safe_pluralization_candidate(norm1, norm2):
            return 0.98  # High but not perfect to indicate pattern match is better

        # Check for spacing/punctuation differences only
        norm1_no_spaces = re.sub(r'[-_\s]+', '', norm1)
        norm2_no_spaces = re.sub(r'[-_\s]+', '', norm2)

        if norm1_no_spaces == norm2_no_spaces and norm1 != norm2:
            return 0.97  # Spacing differences only

        # For all other cases, return low similarity to prevent dangerous matches
        return 0.0  # Conservative: no similarity matching for medical terms

    def find_by_similarity(self, term: str, entity_type: str, threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        HEAVILY RESTRICTED similarity matching for medical terms.

        ⚠️  WARNING: This method is extremely dangerous for medical terminology!
        It can cause false positives that could lead to incorrect medical decisions.

        This method is now extremely conservative to prevent dangerous
        false positives in medical data, but it should still not be used.

        For any matching needs, use the safe pattern matching or LLM verification instead!

        Args:
            term: The term to search for
            entity_type: Either 'intervention' or 'condition'
            threshold: Minimum similarity score (forced to be very high!)

        Returns:
            List of matching canonical entities (will be very few)
        """
        print(f"⚠️  WARNING: Using dangerous similarity matching for '{term}' - medical safety at risk!")

        matches = []

        # Force high threshold for medical safety
        safe_threshold = max(threshold, 0.95)

        # Get all canonical entities and their mappings for this entity type
        all_mappings = self.repository.get_all_mappings_with_canonicals(entity_type)

        for mapping in all_mappings:
            # Check similarity with canonical name
            canonical_similarity = self.calculate_string_similarity(term, mapping['canonical_name'])
            if canonical_similarity >= safe_threshold:

                # Additional safety check: flag for review if not identical
                needs_review = canonical_similarity < 1.0 and canonical_similarity > 0.9

                matches.append({
                    'id': mapping['id'],
                    'canonical_name': mapping['canonical_name'],
                    'entity_type': mapping['entity_type'],
                    'metadata': mapping.get('metadata'),
                    'similarity_score': canonical_similarity,
                    'match_method': 'DEPRECATED_UNSAFE_SIMILARITY',
                    'matched_text': mapping['canonical_name'],
                    'needs_llm_verification': needs_review,
                    'safety_note': '⚠️  DANGEROUS: High similarity but not identical - medical terms require safe matching',
                    'warning': 'THIS METHOD IS UNSAFE FOR MEDICAL TERMINOLOGY'
                })

            # Check similarity with mapped terms
            if mapping['raw_text']:
                mapping_similarity = self.calculate_string_similarity(term, mapping['raw_text'])
                if mapping_similarity >= safe_threshold:

                    needs_review = mapping_similarity < 1.0 and mapping_similarity > 0.9

                    matches.append({
                        'id': mapping['id'],
                        'canonical_name': mapping['canonical_name'],
                        'entity_type': mapping['entity_type'],
                        'metadata': mapping.get('metadata'),
                        'similarity_score': mapping_similarity,
                        'match_method': 'DEPRECATED_UNSAFE_SIMILARITY',
                        'matched_text': mapping['raw_text'],
                        'needs_llm_verification': needs_review,
                        'safety_note': '⚠️  DANGEROUS: High similarity but not identical - medical terms require safe matching',
                        'warning': 'THIS METHOD IS UNSAFE FOR MEDICAL TERMINOLOGY'
                    })

        # Remove duplicates based on canonical ID, keeping highest similarity
        best_matches = {}
        for match in matches:
            entity_id = match['id']
            if entity_id not in best_matches or match['similarity_score'] > best_matches[entity_id]['similarity_score']:
                best_matches[entity_id] = match

        # Sort by similarity score descending
        sorted_matches = sorted(best_matches.values(), key=lambda x: x['similarity_score'], reverse=True)

        if sorted_matches:
            print(f"⚠️  WARNING: Found {len(sorted_matches)} unsafe similarity matches - verify manually!")

        return sorted_matches

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


def create_unsafe_matcher(repository: CanonicalRepository) -> UnsafeStringMatcher:
    """
    Factory function to create an unsafe string matcher.

    ⚠️  WARNING: This creates a matcher with dangerous methods!

    Args:
        repository: Database repository for entity operations

    Returns:
        UnsafeStringMatcher instance
    """
    print("⚠️  WARNING: Creating unsafe string matcher - dangerous for medical terms!")
    print("⚠️  WARNING: Use safe pattern matching or LLM verification instead!")
    print("⚠️  WARNING: Known dangerous false positives: probiotics/prebiotics, hyper/hypotension, etc.")

    return UnsafeStringMatcher(repository)


# Explicit warning at module level
print("⚠️  WARNING: legacy_unsafe.py module imported - contains dangerous medical term matching methods!")
print("⚠️  WARNING: Use safe pattern matching or LLM verification instead!")

# Module-level constants for documentation
DANGEROUS_FALSE_POSITIVES = [
    "probiotics vs prebiotics (different substances)",
    "hyperglycemia vs hypoglycemia (opposite conditions)",
    "hypertension vs hypotension (opposite conditions)",
    "Any similar-sounding medical terms with different meanings"
]

SAFE_ALTERNATIVES = [
    "Use ExactMatcher for exact normalized matching",
    "Use PatternMatcher for safe pattern matching (plurals, spacing, etc.)",
    "Use LLMMatcher for semantic matching with medical domain awareness",
    "Use MatchingEngine.find_safe_matches_only() for production medical data"
]