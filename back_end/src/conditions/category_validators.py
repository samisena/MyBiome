"""
Category-specific validation system for condition data.
Handles specialized validation rules for health conditions.
"""

import re
from typing import Dict, List, Optional, Any, Set
from back_end.src.conditions.taxonomy import ConditionType, condition_taxonomy


class ConditionValidationError(Exception):
    """Custom exception for condition validation errors."""
    pass


class ConditionValidator:
    """Validates condition data based on category-specific rules."""

    def __init__(self):
        self.taxonomy = condition_taxonomy
        self.placeholder_patterns = self._build_placeholder_patterns()

    def _build_placeholder_patterns(self) -> Set[str]:
        """Build a set of placeholder patterns to reject."""
        return {
            '...', 'N/A', 'n/a', 'NA', 'na', 'null', 'NULL', 'none', 'None',
            'unknown', 'Unknown', 'UNKNOWN', 'placeholder', 'Placeholder', 'PLACEHOLDER',
            'TBD', 'tbd', 'TODO', 'todo', 'not specified', 'not available',
            'various', 'multiple', 'several', 'different', 'mixed',
            'disease', 'condition', 'disorder', 'syndrome'  # Too generic alone
        }

    def validate_condition_category(self, condition_name: str, category_str: str) -> ConditionType:
        """
        Validate condition category assignment.

        Args:
            condition_name: The name of the condition
            category_str: The category string to validate

        Returns:
            ConditionType enum value

        Raises:
            ConditionValidationError: If validation fails
        """
        if not category_str or not isinstance(category_str, str):
            raise ConditionValidationError("Condition category is required")

        category_clean = category_str.strip().lower()

        try:
            condition_type = ConditionType(category_clean)
        except ValueError:
            raise ConditionValidationError(
                f"Invalid condition category: '{category_str}'. "
                f"Must be one of: {[ct.value for ct in ConditionType]}"
            )

        # Additional validation: check if condition name makes sense for category
        self._validate_condition_for_category(condition_name, condition_type)

        return condition_type

    def validate_condition_name(self, name: str) -> str:
        """
        Validate condition name for quality and specificity.

        Args:
            name: Condition name to validate

        Returns:
            Cleaned condition name

        Raises:
            ConditionValidationError: If validation fails
        """
        if not name or not isinstance(name, str):
            raise ConditionValidationError("Condition name is required")

        name_clean = name.strip()

        # Check for placeholders
        if self._is_placeholder(name_clean):
            raise ConditionValidationError(
                f"Condition name appears to be a placeholder: '{name_clean}'"
            )

        # Minimum length check
        if len(name_clean) < 3:
            raise ConditionValidationError(
                f"Condition name too short: '{name_clean}'"
            )

        # Check for overly generic single-word conditions
        if len(name_clean.split()) == 1 and name_clean.lower() in self.placeholder_patterns:
            raise ConditionValidationError(
                f"Condition name too generic: '{name_clean}'"
            )

        return name_clean

    def _validate_condition_for_category(self, condition_name: str, category: ConditionType):
        """
        Apply category-specific validation rules to condition names.

        Args:
            condition_name: The condition name
            category: The assigned category

        Raises:
            ConditionValidationError: If condition doesn't match category
        """
        name_lower = condition_name.lower()

        # Get category definition
        category_def = self.taxonomy.get_category(category)

        # For now, this is a soft validation - we log warnings but don't reject
        # In the future, we could add stricter validation based on medical ontologies

        # Example: Check for obvious mismatches
        if category == ConditionType.CARDIAC:
            # Cardiac conditions should relate to heart/cardiovascular system
            cardiac_indicators = [
                'heart', 'cardiac', 'cardio', 'coronary', 'myocardial',
                'arrhythmia', 'hypertension', 'hypotension', 'atrial',
                'ventricular', 'vascular', 'angina', 'infarction'
            ]
            # This is a soft check - don't reject, just warn
            # We trust the LLM's classification

        elif category == ConditionType.INFECTIOUS:
            # Should not be in infectious if it's clearly parasitic
            if any(term in name_lower for term in ['malaria', 'parasite', 'helminth', 'worm']):
                raise ConditionValidationError(
                    f"Condition '{condition_name}' appears parasitic but categorized as infectious"
                )

        # Most validation is delegated to the LLM's judgment
        # We only reject obvious errors

    def _is_placeholder(self, text: str) -> bool:
        """Check if text appears to be a placeholder."""
        text_clean = text.strip().lower()

        # Direct placeholder matches
        if text_clean in self.placeholder_patterns:
            return True

        # Pattern-based checks
        placeholder_prefixes = ['unknown', 'placeholder', 'various', 'multiple', 'several']
        if any(text_clean.startswith(prefix) for prefix in placeholder_prefixes):
            words = text_clean.split()
            if len(words) == 1:  # Single word like "unknown" - reject
                return True

        # Check for ellipsis or dots
        if '...' in text or text_clean.replace('.', '') == '':
            return True

        # Reject if it's ONLY generic terms
        generic_only_terms = [
            'disease', 'condition', 'disorder', 'syndrome', 'illness',
            'problem', 'issue', 'complication'
        ]
        if text_clean in generic_only_terms:
            return True

        return False


# Global instance
condition_validator = ConditionValidator()
