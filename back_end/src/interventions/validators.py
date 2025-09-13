"""
Enhanced validation system for intervention data.
Handles category-specific validation rules and data cleaning.
"""

import re
from typing import Dict, List, Optional, Any, Set
from src.interventions.taxonomy import InterventionType, intervention_taxonomy
from src.data.validators import validation_manager, ValidationResult


class ValidationError(Exception):
    """Custom exception for intervention validation errors."""
    pass


class InterventionValidator:
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
            'various', 'multiple', 'several', 'different', 'mixed',
            'intervention', 'treatment', 'therapy'  # Too generic
        }
    
    def validate_intervention(self, intervention_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a complete intervention record using enhanced validation.
        
        Args:
            intervention_data: Dictionary containing intervention information
            
        Returns:
            Validated and cleaned intervention dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        # Use new validation system first
        result = validation_manager.validate_intervention(intervention_data)
        
        if not result.is_valid:
            error_messages = [f"{issue.field}: {issue.message}" for issue in result.errors]
            raise ValidationError(f"Intervention validation failed: {'; '.join(error_messages)}")
        
        # Continue with category-specific validation
        validated_data = result.cleaned_data.copy()
        
        # Log any warnings
        if result.warnings:
            validation_manager.log_validation_issues(result, "intervention validation")
        # Basic required fields
        required_base_fields = [
            'intervention_category', 'intervention_name', 'health_condition',
            'correlation_type', 'extraction_model'
        ]
        
        for field in required_base_fields:
            if field not in intervention_data or not intervention_data[field]:
                raise ValidationError(f"Missing required base field: {field}")
        
        # Validate intervention category
        category = self._validate_category(intervention_data['intervention_category'])
        
        # Validate intervention name
        intervention_name = self._validate_intervention_name(
            intervention_data['intervention_name'], category
        )
        
        # Validate health condition
        health_condition = self._validate_health_condition(
            intervention_data['health_condition']
        )
        
        # Validate correlation type
        correlation_type = self._validate_correlation_type(
            intervention_data['correlation_type']
        )
        
        # Build validated intervention
        validated = {
            'intervention_category': category.value,
            'intervention_name': intervention_name,
            'health_condition': health_condition,
            'correlation_type': correlation_type,
            'extraction_model': str(intervention_data['extraction_model']).strip()
        }
        
        # Validate intervention details (category-specific)
        if 'intervention_details' in intervention_data:
            validated['intervention_details'] = self._validate_intervention_details(
                intervention_data['intervention_details'], category
            )
        
        # Validate optional numeric fields
        for field in ['correlation_strength', 'confidence_score']:
            if field in intervention_data and intervention_data[field] is not None:
                validated[field] = self._validate_numeric_field(
                    intervention_data[field], field, 0.0, 1.0
                )
        
        # Validate sample size
        if 'sample_size' in intervention_data and intervention_data[field] is not None:
            validated['sample_size'] = self._validate_sample_size(
                intervention_data['sample_size']
            )
        
        # Validate text fields
        for field in ['supporting_quote', 'study_type', 'study_duration', 
                     'dosage', 'population_details']:
            if field in intervention_data and intervention_data[field]:
                validated[field] = str(intervention_data[field]).strip()
        
        return validated
    
    def _validate_category(self, category_str: str) -> InterventionType:
        """Validate intervention category."""
        if not category_str or not isinstance(category_str, str):
            raise ValidationError("Intervention category is required")
        
        category_clean = category_str.strip().lower()
        
        try:
            return InterventionType(category_clean)
        except ValueError:
            raise ValidationError(f"Invalid intervention category: '{category_str}'")
    
    def _validate_intervention_name(self, name: str, category: InterventionType) -> str:
        """Validate intervention name with category-specific rules."""
        if not name or not isinstance(name, str):
            raise ValidationError("Intervention name is required")
        
        name_clean = name.strip()
        
        # Check for placeholders
        if self._is_placeholder(name_clean):
            raise ValidationError(f"Intervention name appears to be a placeholder: '{name_clean}'")
        
        # Minimum length check
        if len(name_clean) < 3:
            raise ValidationError(f"Intervention name too short: '{name_clean}'")
        
        # Category-specific validation
        self._validate_name_for_category(name_clean, category)
        
        return name_clean
    
    def _validate_name_for_category(self, name: str, category: InterventionType):
        """Apply category-specific name validation rules."""
        name_lower = name.lower()
        
        if category == InterventionType.EXERCISE:
            # Should contain exercise-related terms
            exercise_indicators = [
                'exercise', 'training', 'activity', 'aerobic', 'resistance', 
                'strength', 'cardio', 'yoga', 'pilates', 'walking', 'running',
                'cycling', 'swimming', 'workout', 'fitness'
            ]
            if not any(indicator in name_lower for indicator in exercise_indicators):
                # Allow if it's a specific exercise name
                specific_exercises = [
                    'hiit', 'crossfit', 'zumba', 'tai chi', 'qigong', 'dance',
                    'boxing', 'martial arts', 'climbing', 'rowing'
                ]
                if not any(exercise in name_lower for exercise in specific_exercises):
                    raise ValidationError(f"Exercise intervention name should contain exercise-related terms: '{name}'")
        
        elif category == InterventionType.SUPPLEMENT:
            # Should not be too generic
            generic_supplement_terms = [
                'supplement', 'supplementation', 'nutraceutical', 'pill', 'capsule'
            ]
            if name_lower in generic_supplement_terms:
                raise ValidationError(f"Supplement name too generic: '{name}'")
        
        elif category == InterventionType.MEDICATION:
            # Should not be too generic
            generic_med_terms = [
                'medication', 'drug', 'medicine', 'pharmaceutical', 'pill'
            ]
            if name_lower in generic_med_terms:
                raise ValidationError(f"Medication name too generic: '{name}'")
    
    def _validate_health_condition(self, condition: str) -> str:
        """Validate health condition."""
        if not condition or not isinstance(condition, str):
            raise ValidationError("Health condition is required")
        
        condition_clean = condition.strip()
        
        # Check for placeholders
        if self._is_placeholder(condition_clean):
            raise ValidationError(f"Health condition appears to be a placeholder: '{condition_clean}'")
        
        # Minimum length check
        if len(condition_clean) < 3:
            raise ValidationError(f"Health condition too short: '{condition_clean}'")
        
        return condition_clean
    
    def _validate_correlation_type(self, corr_type: str) -> str:
        """Validate correlation type."""
        if not corr_type or not isinstance(corr_type, str):
            raise ValidationError("Correlation type is required")
        
        valid_types = ['positive', 'negative', 'neutral', 'inconclusive']
        corr_type_clean = corr_type.strip().lower()
        
        if corr_type_clean not in valid_types:
            raise ValidationError(f"Invalid correlation type: '{corr_type}'. Must be one of: {valid_types}")
        
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
                raise ValidationError(f"Required detail field '{field_def.name}' is missing")
            return None
        
        # Convert to appropriate type
        if field_def.data_type == "string":
            value_str = str(value).strip()
            if not value_str and field_def.required:
                raise ValidationError(f"Required detail field '{field_def.name}' is empty")
            
            # Check validation rules
            if 'allowed_values' in field_def.validation_rules:
                if value_str.lower() not in [v.lower() for v in field_def.validation_rules['allowed_values']]:
                    raise ValidationError(
                        f"Invalid value for '{field_def.name}': '{value_str}'. "
                        f"Allowed: {field_def.validation_rules['allowed_values']}"
                    )
            
            return value_str
        
        elif field_def.data_type == "number":
            try:
                return float(value)
            except (ValueError, TypeError):
                raise ValidationError(f"Field '{field_def.name}' must be a number: '{value}'")
        
        elif field_def.data_type == "list":
            if isinstance(value, list):
                return value
            elif isinstance(value, str):
                # Try to parse comma-separated values
                return [item.strip() for item in value.split(',') if item.strip()]
            else:
                raise ValidationError(f"Field '{field_def.name}' must be a list: '{value}'")
        
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
                raise ValidationError(
                    f"{field_name} must be between {min_val} and {max_val}: {num_value}"
                )
            return num_value
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a number: '{value}'")
    
    def _validate_sample_size(self, value: Any) -> int:
        """Validate sample size field."""
        try:
            sample_size = int(value)
            if sample_size < 0:
                raise ValidationError("Sample size must be non-negative")
            return sample_size
        except (ValueError, TypeError):
            raise ValidationError(f"Sample size must be an integer: '{value}'")
    
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
intervention_validator = InterventionValidator()