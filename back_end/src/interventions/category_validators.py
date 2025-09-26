"""
Category-specific validation system for intervention data.
Handles specialized validation rules and data cleaning beyond basic validation.
"""

import re
from typing import Dict, List, Optional, Any, Set
from back_end.src.interventions.taxonomy import InterventionType, intervention_taxonomy
from back_end.src.data.validators import validation_manager, ValidationResult


class CategoryValidationError(Exception):
    """Custom exception for category-specific intervention validation errors."""
    pass


class CategorySpecificValidator:
    """Validates intervention data based on category-specific rules."""
    
    def __init__(self):
        self.taxonomy = intervention_taxonomy
        self.placeholder_patterns = self._build_placeholder_patterns()
    
    def _build_placeholder_patterns(self) -> Set[str]:
        """Build a set of placeholder patterns to reject."""
        return {
            '...', 'N/A', 'n/a', 'NA', 'na', 'null', 'NULL', 'none', 'None',
            'unknown', 'Unknown', 'UNKNOWN', 'placeholder', 'Placeholder', 'PLACEHOLDER',
            'TBD', 'tbd', 'TODO', 'todo', 'not specified', 'not available',
            'various', 'multiple', 'several', 'different', 'mixed'
            # Removed overly broad: 'intervention', 'treatment', 'therapy'
        }
    
    def validate_intervention(self, intervention_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a complete intervention record using enhanced validation.

        Args:
            intervention_data: Dictionary containing intervention information

        Returns:
            Validated and cleaned intervention dictionary

        Raises:
            CategoryValidationError: If validation fails
        """
        # Use new validation system first
        result = validation_manager.validate_intervention(intervention_data)

        if not result.is_valid:
            error_messages = [f"{issue.field}: {issue.message}" for issue in result.errors]
            raise CategoryValidationError(f"Intervention validation failed: {'; '.join(error_messages)}")

        # Continue with category-specific validation using cleaned data
        validated_data = result.cleaned_data.copy()

        # Log any warnings
        if result.warnings:
            validation_manager.log_validation_issues(result, "intervention validation")
        
        # Validate intervention category using cleaned data
        category = self._validate_category(validated_data['intervention_category'])

        # Validate intervention name
        intervention_name = self._validate_intervention_name(
            validated_data['intervention_name'], category
        )

        # Update validated data with category-specific validation
        validated_data.update({
            'intervention_category': category.value,
            'intervention_name': intervention_name
        })

        # Validate subcategory if provided
        if 'intervention_subcategory' in validated_data:
            validated_subcategory = self._validate_subcategory(
                validated_data['intervention_subcategory'], category
            )
            if validated_subcategory:
                validated_data['intervention_subcategory'] = validated_subcategory

        # Validate intervention details (category-specific)
        if 'intervention_details' in validated_data:
            validated_data['intervention_details'] = self._validate_intervention_details(
                validated_data['intervention_details'], category
            )

        # Additional category-specific validations can be added here

        return validated_data
    
    def _validate_category(self, category_str: str) -> InterventionType:
        """Validate intervention category."""
        if not category_str or not isinstance(category_str, str):
            raise CategoryValidationError("Intervention category is required")
        
        category_clean = category_str.strip().lower()
        
        try:
            return InterventionType(category_clean)
        except ValueError:
            raise CategoryValidationError(f"Invalid intervention category: '{category_str}'")
    
    def _validate_intervention_name(self, name: str, category: InterventionType) -> str:
        """Validate intervention name with category-specific rules."""
        if not name or not isinstance(name, str):
            raise CategoryValidationError("Intervention name is required")
        
        name_clean = name.strip()
        
        # Check for placeholders
        if self._is_placeholder(name_clean):
            raise CategoryValidationError(f"Intervention name appears to be a placeholder: '{name_clean}'")
        
        # Minimum length check
        if len(name_clean) < 3:
            raise CategoryValidationError(f"Intervention name too short: '{name_clean}'")
        
        # Category-specific validation
        self._validate_name_for_category(name_clean, category)
        
        return name_clean

    def _validate_subcategory(self, subcategory: str, category: InterventionType) -> Optional[str]:
        """Validate intervention subcategory against allowed values for the category."""
        if not subcategory or not isinstance(subcategory, str):
            return None

        subcategory_clean = subcategory.strip().lower()

        # Get allowed subcategories for this category
        category_def = self.taxonomy.get_category(category)
        allowed_subcategories = [sub.lower() for sub in category_def.subcategories]

        if subcategory_clean not in allowed_subcategories:
            raise CategoryValidationError(
                f"Invalid subcategory '{subcategory}' for {category.value}. "
                f"Allowed: {category_def.subcategories}"
            )

        return subcategory_clean

    def _validate_name_for_category(self, name: str, category: InterventionType):
        """Apply category-specific name validation rules."""
        name_lower = name.lower()
        
        if category == InterventionType.EXERCISE:
            # Expanded exercise-related terms (more permissive)
            exercise_indicators = [
                'exercise', 'training', 'activity', 'aerobic', 'resistance',
                'strength', 'cardio', 'yoga', 'pilates', 'walking', 'running',
                'cycling', 'swimming', 'workout', 'fitness', 'physical', 'movement',
                'therapy', 'rehabilitation', 'stretching', 'balance', 'mobility'
            ]
            if not any(indicator in name_lower for indicator in exercise_indicators):
                # Allow if it's a specific exercise name
                specific_exercises = [
                    'hiit', 'crossfit', 'zumba', 'tai chi', 'qigong', 'dance',
                    'boxing', 'martial arts', 'climbing', 'rowing', 'physiotherapy',
                    'aquatic', 'balance', 'stretching', 'flexibility'
                ]
                if not any(exercise in name_lower for exercise in specific_exercises):
                    raise CategoryValidationError(f"Exercise intervention name should contain exercise-related terms: '{name}'")
        
        elif category == InterventionType.SUPPLEMENT:
            # Should not be too generic - only reject single word generic terms
            generic_supplement_terms = [
                'supplement', 'supplementation', 'pill', 'capsule'
            ]
            if name_lower in generic_supplement_terms:
                raise CategoryValidationError(f"Supplement name too generic: '{name}'")
        
        elif category == InterventionType.MEDICATION:
            # Should not be too generic - only reject very generic single terms
            generic_med_terms = [
                'medication', 'drug', 'medicine', 'pill'
            ]
            if name_lower in generic_med_terms:
                raise CategoryValidationError(f"Medication name too generic: '{name}'")

        elif category == InterventionType.SURGERY:
            # Should not be too generic - only reject very basic terms
            generic_surgery_terms = [
                'surgery', 'operation'
            ]
            if name_lower in generic_surgery_terms:
                raise CategoryValidationError(f"Surgery name too generic: '{name}'")

        elif category == InterventionType.TEST:
            # Should not be too generic - only reject very basic terms
            generic_test_terms = [
                'test', 'testing'
            ]
            if name_lower in generic_test_terms:
                raise CategoryValidationError(f"Test name too generic: '{name}'")
    
    def _validate_health_condition(self, condition: str) -> str:
        """Validate health condition."""
        if not condition or not isinstance(condition, str):
            raise CategoryValidationError("Health condition is required")
        
        condition_clean = condition.strip()
        
        # Check for placeholders
        if self._is_placeholder(condition_clean):
            raise CategoryValidationError(f"Health condition appears to be a placeholder: '{condition_clean}'")
        
        # Minimum length check
        if len(condition_clean) < 3:
            raise CategoryValidationError(f"Health condition too short: '{condition_clean}'")
        
        return condition_clean
    
    def _validate_correlation_type(self, corr_type: str) -> str:
        """Validate correlation type."""
        if not corr_type or not isinstance(corr_type, str):
            raise CategoryValidationError("Correlation type is required")
        
        valid_types = ['positive', 'negative', 'neutral', 'inconclusive']
        corr_type_clean = corr_type.strip().lower()
        
        if corr_type_clean not in valid_types:
            raise CategoryValidationError(f"Invalid correlation type: '{corr_type}'. Must be one of: {valid_types}")
        
        return corr_type_clean
    
    def _validate_intervention_details(self, details: Dict[str, Any], 
                                     category: InterventionType) -> Dict[str, Any]:
        """Validate category-specific intervention details."""
        if not isinstance(details, dict):
            return {}
        
        category_def = self.taxonomy.get_category(category)
        validated_details = {}
        
        # Validate against category field definitions
        for field in category_def.get_all_fields():
            if field.name in details:
                value = details[field.name]
                validated_value = self._validate_detail_field(value, field)
                if validated_value is not None:
                    validated_details[field.name] = validated_value
        
        return validated_details
    
    def _validate_detail_field(self, value: Any, field_def) -> Any:
        """Validate a specific detail field."""
        if value is None:
            if field_def.required:
                raise CategoryValidationError(f"Required detail field '{field_def.name}' is missing")
            return None
        
        # Convert to appropriate type
        if field_def.data_type == "string":
            value_str = str(value).strip()
            if not value_str and field_def.required:
                raise CategoryValidationError(f"Required detail field '{field_def.name}' is empty")
            
            # Check validation rules
            if 'allowed_values' in field_def.validation_rules:
                if value_str.lower() not in [v.lower() for v in field_def.validation_rules['allowed_values']]:
                    raise CategoryValidationError(
                        f"Invalid value for '{field_def.name}': '{value_str}'. "
                        f"Allowed: {field_def.validation_rules['allowed_values']}"
                    )
            
            return value_str
        
        elif field_def.data_type == "number":
            try:
                return float(value)
            except (ValueError, TypeError):
                raise CategoryValidationError(f"Field '{field_def.name}' must be a number: '{value}'")
        
        elif field_def.data_type == "list":
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                # Try to parse comma-separated values
                return [item.strip() for item in value.split(',') if item.strip()]
            else:
                raise CategoryValidationError(f"Field '{field_def.name}' must be a list: '{value}'")
        
        elif field_def.data_type == "boolean":
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'y']
            else:
                return bool(value)
        
        return value
    
    def _validate_numeric_field(self, value: Any, field_name: str, 
                              min_val: float, max_val: float) -> float:
        """Validate numeric field with range checking."""
        try:
            num_value = float(value)
            if not min_val <= num_value <= max_val:
                raise CategoryValidationError(
                    f"{field_name} must be between {min_val} and {max_val}: {num_value}"
                )
            return num_value
        except (ValueError, TypeError):
            raise CategoryValidationError(f"{field_name} must be a number: '{value}'")
    
    def _validate_sample_size(self, value: Any) -> int:
        """Validate sample size field."""
        try:
            sample_size = int(value)
            if sample_size < 0:
                raise CategoryValidationError("Sample size must be non-negative")
            return sample_size
        except (ValueError, TypeError):
            raise CategoryValidationError(f"Sample size must be an integer: '{value}'")
    
    def _is_placeholder(self, text: str) -> bool:
        """Check if text appears to be a placeholder."""
        text_clean = text.strip().lower()
        
        # Direct placeholder matches
        if text_clean in self.placeholder_patterns:
            return True
        
        # Pattern-based checks
        placeholder_prefixes = ['unknown', 'placeholder', 'various', 'multiple', 'several']
        if any(text_clean.startswith(prefix) for prefix in placeholder_prefixes):
            return True
        
        # Check for ellipsis or dots
        if '...' in text or text_clean.replace('.', '') == '':
            return True
        
        return False


# Global instance
category_validator = CategorySpecificValidator()