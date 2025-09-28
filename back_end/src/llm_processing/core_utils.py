"""
Core Utilities and Exception Handling for Entity Normalization

This module provides standardized exception handling, data validation,
and utility functions for the entity normalization system.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import logging
import json
import hashlib
from datetime import datetime

# Configure logging for better error reporting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# === EXCEPTIONS ===

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


# === ENUMS ===

class EntityType(Enum):
    """Enum for valid entity types."""
    INTERVENTION = "intervention"
    CONDITION = "condition"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a value is a valid entity type."""
        return value in [member.value for member in cls]


# === VALIDATION UTILITIES ===

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


# === ERROR HANDLING ===

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


# === RESULT STANDARDIZERS ===

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


# === CACHE UTILITIES ===

class CacheUtils:
    """Utilities for caching operations."""

    @staticmethod
    def generate_cache_key(*args, **kwargs) -> str:
        """
        Generate a consistent cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            MD5 hash of the arguments as cache key
        """
        # Create a consistent string representation
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())  # Sort for consistency
        }

        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    @staticmethod
    def is_cache_valid(cached_at: str, max_age_hours: int = 24) -> bool:
        """
        Check if a cached result is still valid.

        Args:
            cached_at: ISO timestamp when item was cached
            max_age_hours: Maximum age in hours

        Returns:
            True if cache is still valid
        """
        try:
            cache_time = datetime.fromisoformat(cached_at.replace('Z', '+00:00'))
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            return age_hours < max_age_hours
        except (ValueError, AttributeError):
            return False


# === MATCH RESULT CONVERTERS ===

class MatchResultConverter:
    """Utilities for converting between different match result formats."""

    @staticmethod
    def to_legacy_format(match_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a standardized match result to legacy format for backward compatibility.

        Args:
            match_result: Standardized match result

        Returns:
            Match result in legacy format
        """
        legacy_result = match_result.copy()

        # Add legacy field aliases
        if 'canonical_id' in legacy_result and 'id' not in legacy_result:
            legacy_result['id'] = legacy_result['canonical_id']

        if 'confidence' in legacy_result and 'confidence_score' not in legacy_result:
            legacy_result['confidence_score'] = legacy_result['confidence']

        if 'method' in legacy_result and 'match_method' not in legacy_result:
            legacy_result['match_method'] = legacy_result['method']

        return legacy_result

    @staticmethod
    def from_legacy_format(legacy_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a legacy match result to standardized format.

        Args:
            legacy_result: Match result in legacy format

        Returns:
            Standardized match result
        """
        return ValidationUtils.standardize_match_result(legacy_result)

    @staticmethod
    def batch_convert_to_legacy(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert a list of standardized results to legacy format.

        Args:
            results: List of standardized match results

        Returns:
            List of match results in legacy format
        """
        return [MatchResultConverter.to_legacy_format(result) for result in results]


# === TERM NORMALIZATION ===

class TermNormalizer:
    """Utilities for term normalization and validation."""

    @staticmethod
    def normalize_medical_term(term: str) -> str:
        """
        Normalize a medical term using safe, consistent rules.

        Args:
            term: Raw medical term

        Returns:
            Normalized term

        Raises:
            ValidationError: If term is invalid
        """
        term = ValidationUtils.validate_term(term)

        # Apply standard normalization
        import re

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

    @staticmethod
    def extract_scientific_name(term: str) -> tuple[str, Optional[str]]:
        """
        Extract scientific name from a term if present in parentheses.

        Args:
            term: Term that might contain scientific name

        Returns:
            Tuple of (main_term, scientific_name)
        """
        import re

        # Look for scientific name in parentheses
        match = re.search(r'^(.*?)\s*\(([^)]+)\)\s*$', term.strip())

        if match:
            main_term = match.group(1).strip()
            scientific_name = match.group(2).strip()

            # Basic validation for scientific name (genus species pattern)
            if re.match(r'^[A-Z][a-z]+\s+[a-z]+', scientific_name):
                return main_term, scientific_name

        return term, None

    @staticmethod
    def is_likely_abbreviation(term: str) -> bool:
        """
        Check if a term is likely an abbreviation.

        Args:
            term: Term to check

        Returns:
            True if term appears to be an abbreviation
        """
        # Simple heuristics for abbreviations
        if len(term) <= 5 and term.isupper():
            return True

        if '.' in term and len(term) <= 10:
            return True

        return False


# === CONFIGURATION VALIDATION ===

class ConfigurationValidator:
    """Utilities for validating system configuration."""

    @staticmethod
    def validate_llm_config(llm_model: str, llm_client: Any) -> bool:
        """
        Validate LLM configuration.

        Args:
            llm_model: LLM model name
            llm_client: LLM client instance

        Returns:
            True if configuration is valid

        Raises:
            ValidationError: If configuration is invalid
        """
        if not isinstance(llm_model, str) or not llm_model.strip():
            raise ValidationError("LLM model name must be a non-empty string")

        if llm_client is None:
            ErrorHandler.log_warning("LLM client is None - LLM functionality will be disabled")
            return False

        return True

    @staticmethod
    def validate_database_connection(db_connection: Any) -> bool:
        """
        Validate database connection.

        Args:
            db_connection: Database connection object

        Returns:
            True if connection is valid

        Raises:
            ValidationError: If connection is invalid
        """
        if db_connection is None:
            raise ValidationError("Database connection cannot be None")

        try:
            # Try to execute a simple query
            cursor = db_connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            return True
        except Exception as e:
            raise ValidationError(f"Database connection is not working: {e}")


# === PERFORMANCE MONITORING ===

class PerformanceMonitor:
    """Utilities for monitoring performance and collecting metrics."""

    def __init__(self):
        self.operation_counts = {}
        self.operation_times = {}

    def record_operation(self, operation_name: str, duration_seconds: float) -> None:
        """
        Record an operation for performance monitoring.

        Args:
            operation_name: Name of the operation
            duration_seconds: Duration in seconds
        """
        if operation_name not in self.operation_counts:
            self.operation_counts[operation_name] = 0
            self.operation_times[operation_name] = []

        self.operation_counts[operation_name] += 1
        self.operation_times[operation_name].append(duration_seconds)

        # Keep only last 100 measurements to prevent memory issues
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name] = self.operation_times[operation_name][-100:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        stats = {}

        for operation in self.operation_counts:
            times = self.operation_times[operation]
            if times:
                stats[operation] = {
                    'count': self.operation_counts[operation],
                    'avg_time_ms': sum(times) / len(times) * 1000,
                    'min_time_ms': min(times) * 1000,
                    'max_time_ms': max(times) * 1000,
                    'total_time_ms': sum(times) * 1000
                }

        return stats


# === SAFETY VALIDATION ===

class SafetyValidator:
    """Utilities for validating medical term safety."""

    # Known dangerous pairs that should never be matched
    DANGEROUS_PAIRS = [
        ('probiotics', 'prebiotics'),
        ('hyperglycemia', 'hypoglycemia'),
        ('hypertension', 'hypotension'),
        ('hyperthermia', 'hypothermia'),
        ('tachycardia', 'bradycardia'),
        ('hyponatremia', 'hypernatremia'),
        ('acidosis', 'alkalosis')
    ]

    @staticmethod
    def is_dangerous_match(term1: str, term2: str) -> bool:
        """
        Check if two terms represent a potentially dangerous match.

        Args:
            term1: First term
            term2: Second term

        Returns:
            True if this is a dangerous match that should be avoided
        """
        # Normalize terms for comparison
        norm1 = TermNormalizer.normalize_medical_term(term1.lower())
        norm2 = TermNormalizer.normalize_medical_term(term2.lower())

        # Check against known dangerous pairs
        for dangerous1, dangerous2 in SafetyValidator.DANGEROUS_PAIRS:
            if ((norm1 == dangerous1 and norm2 == dangerous2) or
                (norm1 == dangerous2 and norm2 == dangerous1)):
                return True

        return False

    @staticmethod
    def validate_match_safety(term: str, canonical_name: str, confidence: float) -> Dict[str, Any]:
        """
        Validate the safety of a proposed match.

        Args:
            term: Original term
            canonical_name: Proposed canonical match
            confidence: Confidence score

        Returns:
            Dictionary with safety assessment
        """
        is_dangerous = SafetyValidator.is_dangerous_match(term, canonical_name)

        if is_dangerous:
            ErrorHandler.log_warning(
                f"Dangerous match detected: '{term}' -> '{canonical_name}'",
                {'confidence': confidence}
            )

        return {
            'is_safe': not is_dangerous,
            'is_dangerous': is_dangerous,
            'requires_manual_review': is_dangerous or confidence < 0.8,
            'safety_score': 0.0 if is_dangerous else 1.0
        }


# === UTILITY DECORATORS ===

def timing_decorator(performance_monitor: PerformanceMonitor):
    """
    Decorator to automatically record operation timing.

    Args:
        performance_monitor: PerformanceMonitor instance

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                performance_monitor.record_operation(func.__name__, duration)
        return wrapper
    return decorator