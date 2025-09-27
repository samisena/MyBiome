"""
Standardized Exception Handling and Validation

This module provides consistent error handling and data validation
for the entity normalization system.
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging

# Configure logging for better error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityNormalizationError(Exception):
    """Base exception for entity normalization errors."""
    pass


class DatabaseError(EntityNormalizationError):
    """Raised when database operations fail."""
    pass


class ValidationError(EntityNormalizationError):
    """Raised when input validation fails."""
    pass


class MatchingError(EntityNormalizationError):
    """Raised when matching operations fail."""
    pass


class LLMError(EntityNormalizationError):
    """Raised when LLM operations fail."""
    pass


class ConfigurationError(EntityNormalizationError):
    """Raised when system configuration is invalid."""
    pass


class EntityType(Enum):
    """Enum for valid entity types."""
    INTERVENTION = "intervention"
    CONDITION = "condition"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a value is a valid entity type."""
        return value in [member.value for member in cls]


class ValidationUtils:
    """Utilities for validating inputs and standardizing outputs."""

    @staticmethod
    def validate_entity_type(entity_type: str) -> str:
        """
        Validate and normalize entity type.

        Args:
            entity_type: The entity type to validate

        Returns:
            Validated entity type

        Raises:
            ValidationError: If entity type is invalid
        """
        if not entity_type or not isinstance(entity_type, str):
            raise ValidationError("Entity type must be a non-empty string")

        entity_type = entity_type.strip().lower()

        if not EntityType.is_valid(entity_type):
            valid_types = [member.value for member in EntityType]
            raise ValidationError(f"Invalid entity type '{entity_type}'. Must be one of: {valid_types}")

        return entity_type

    @staticmethod
    def validate_term(term: str) -> str:
        """
        Validate and normalize term input.

        Args:
            term: The term to validate

        Returns:
            Validated and stripped term

        Raises:
            ValidationError: If term is invalid
        """
        if not isinstance(term, str):
            raise ValidationError("Term must be a string")

        term = term.strip()

        if not term:
            raise ValidationError("Term cannot be empty or whitespace only")

        if len(term) > 1000:  # Reasonable limit
            raise ValidationError("Term is too long (max 1000 characters)")

        return term

    @staticmethod
    def validate_confidence(confidence: float) -> float:
        """
        Validate confidence score.

        Args:
            confidence: The confidence score to validate

        Returns:
            Validated confidence score

        Raises:
            ValidationError: If confidence is invalid
        """
        if not isinstance(confidence, (int, float)):
            raise ValidationError("Confidence must be a number")

        confidence = float(confidence)

        if not 0.0 <= confidence <= 1.0:
            raise ValidationError("Confidence must be between 0.0 and 1.0")

        return confidence

    @staticmethod
    def validate_canonical_id(canonical_id: int) -> int:
        """
        Validate canonical entity ID.

        Args:
            canonical_id: The ID to validate

        Returns:
            Validated ID

        Raises:
            ValidationError: If ID is invalid
        """
        if not isinstance(canonical_id, int):
            raise ValidationError("Canonical ID must be an integer")

        if canonical_id <= 0:
            raise ValidationError("Canonical ID must be positive")

        return canonical_id

    @staticmethod
    def standardize_match_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize a match result to ensure consistent field names and types.

        Args:
            result: Raw match result dictionary

        Returns:
            Standardized match result dictionary
        """
        if not isinstance(result, dict):
            raise ValidationError("Match result must be a dictionary")

        standardized = {}

        # Standardize ID field
        for id_field in ['id', 'canonical_id']:
            if id_field in result:
                standardized['canonical_id'] = ValidationUtils.validate_canonical_id(result[id_field])
                break
        else:
            raise ValidationError("Match result must contain 'id' or 'canonical_id'")

        # Standardize canonical name
        if 'canonical_name' not in result:
            raise ValidationError("Match result must contain 'canonical_name'")
        standardized['canonical_name'] = str(result['canonical_name'])

        # Standardize entity type
        if 'entity_type' in result:
            standardized['entity_type'] = ValidationUtils.validate_entity_type(result['entity_type'])

        # Standardize confidence (handle multiple field names)
        confidence = None
        for conf_field in ['confidence', 'confidence_score', 'similarity_score']:
            if conf_field in result:
                confidence = ValidationUtils.validate_confidence(result[conf_field])
                break

        if confidence is not None:
            standardized['confidence'] = confidence

        # Standardize method
        for method_field in ['match_method', 'method']:
            if method_field in result:
                standardized['method'] = str(result[method_field])
                break

        # Optional fields
        optional_fields = ['reasoning', 'metadata', 'cached', 'safety_level']
        for field in optional_fields:
            if field in result:
                standardized[field] = result[field]

        return standardized


class ErrorHandler:
    """Centralized error handling with proper logging."""

    @staticmethod
    def handle_database_error(operation: str, error: Exception) -> DatabaseError:
        """
        Handle database errors with proper logging.

        Args:
            operation: Description of the operation that failed
            error: The original exception

        Returns:
            DatabaseError with standardized message
        """
        error_msg = f"Database operation failed: {operation}. Error: {str(error)}"
        logger.error(error_msg)
        return DatabaseError(error_msg)

    @staticmethod
    def handle_llm_error(operation: str, error: Exception) -> LLMError:
        """
        Handle LLM errors with proper logging.

        Args:
            operation: Description of the operation that failed
            error: The original exception

        Returns:
            LLMError with standardized message
        """
        error_msg = f"LLM operation failed: {operation}. Error: {str(error)}"
        logger.error(error_msg)
        return LLMError(error_msg)

    @staticmethod
    def handle_matching_error(operation: str, error: Exception) -> MatchingError:
        """
        Handle matching errors with proper logging.

        Args:
            operation: Description of the operation that failed
            error: The original exception

        Returns:
            MatchingError with standardized message
        """
        error_msg = f"Matching operation failed: {operation}. Error: {str(error)}"
        logger.error(error_msg)
        return MatchingError(error_msg)

    @staticmethod
    def log_warning(message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning with optional details.

        Args:
            message: Warning message
            details: Optional additional details
        """
        if details:
            logger.warning(f"{message}. Details: {details}")
        else:
            logger.warning(message)

    @staticmethod
    def log_info(message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an info message with optional details.

        Args:
            message: Info message
            details: Optional additional details
        """
        if details:
            logger.info(f"{message}. Details: {details}")
        else:
            logger.info(message)


class ResultStandardizer:
    """Utilities for standardizing method return types."""

    @staticmethod
    def standardize_normalization_result(canonical_id: Optional[int], canonical_name: str,
                                       method: str, confidence: float,
                                       is_new: bool, reasoning: str = "") -> Dict[str, Any]:
        """
        Create a standardized normalization result.

        Args:
            canonical_id: ID of the canonical entity (None if failed)
            canonical_name: Name of the canonical entity
            method: Method used for normalization
            confidence: Confidence score (0.0 to 1.0)
            is_new: Whether a new canonical entity was created
            reasoning: Optional reasoning for the result

        Returns:
            Standardized normalization result dictionary
        """
        if canonical_id is not None:
            canonical_id = ValidationUtils.validate_canonical_id(canonical_id)

        confidence = ValidationUtils.validate_confidence(confidence)

        return {
            'canonical_id': canonical_id,
            'canonical_name': str(canonical_name),
            'method': str(method),
            'confidence': confidence,
            'is_new': bool(is_new),
            'reasoning': str(reasoning),
            'success': canonical_id is not None
        }

    @staticmethod
    def standardize_mapping_result(mapping_id: int, canonical_id: int,
                                 raw_text: str, confidence: float,
                                 method: str) -> Dict[str, Any]:
        """
        Create a standardized mapping result.

        Args:
            mapping_id: ID of the created mapping
            canonical_id: ID of the canonical entity
            raw_text: Original raw text that was mapped
            confidence: Confidence score (0.0 to 1.0)
            method: Method used for mapping

        Returns:
            Standardized mapping result dictionary
        """
        mapping_id = ValidationUtils.validate_canonical_id(mapping_id)
        canonical_id = ValidationUtils.validate_canonical_id(canonical_id)
        confidence = ValidationUtils.validate_confidence(confidence)

        return {
            'mapping_id': mapping_id,
            'canonical_id': canonical_id,
            'raw_text': str(raw_text),
            'confidence': confidence,
            'method': str(method),
            'success': True
        }

    @staticmethod
    def standardize_error_result(error_type: str, error_message: str,
                               operation: str) -> Dict[str, Any]:
        """
        Create a standardized error result.

        Args:
            error_type: Type of error that occurred
            error_message: Detailed error message
            operation: Operation that failed

        Returns:
            Standardized error result dictionary
        """
        return {
            'success': False,
            'error_type': str(error_type),
            'error_message': str(error_message),
            'operation': str(operation),
            'canonical_id': None,
            'canonical_name': None,
            'confidence': 0.0
        }