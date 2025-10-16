#!/usr/bin/env python3
"""
Entity Processing Utilities

This module contains validation, exceptions, constants, and utility functions
used by the batch entity processor. Extracted for better code organization
and reusability.
"""

import re
import json
import hashlib
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass

# === EXCEPTIONS ===

class EntityNormalizationError(Exception):
    """Base exception for entity normalization errors."""
    def __init__(self, message: str, error_code: str = None, context: Dict = None):
        self.error_code = error_code or "GENERAL_ERROR"
        self.context = context or {}
        super().__init__(message)

class DatabaseError(EntityNormalizationError):
    """Database operation failed."""
    pass

class ValidationError(EntityNormalizationError):
    """Data validation failed."""
    pass

class MatchingError(EntityNormalizationError):
    """Entity matching failed."""
    pass

class LLMError(EntityNormalizationError):
    """LLM operation failed."""
    pass

class ConfigurationError(EntityNormalizationError):
    """Configuration is invalid."""
    pass

# === ENUMS ===

class EntityType(Enum):
    """Valid entity types for normalization."""
    INTERVENTION = "intervention"
    CONDITION = "condition"
    SIDE_EFFECT = "side_effect"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string is a valid entity type."""
        return value in [e.value for e in cls]

class MatchingMode(Enum):
    """Matching strategy modes."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"

# === DATA CLASSES ===

@dataclass
class MatchResult:
    """Result of an entity matching operation."""
    canonical_id: int
    canonical_name: str
    entity_type: str
    confidence: float
    method: str
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'canonical_id': self.canonical_id,
            'canonical_name': self.canonical_name,
            'entity_type': self.entity_type,
            'confidence': self.confidence,
            'method': self.method,
            'reasoning': self.reasoning
        }

@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Dict[str, Any] = None

# === VALIDATION FUNCTIONS ===

class EntityValidator:
    """Centralized validation for entity data."""

    # Valid entity types
    VALID_ENTITY_TYPES = {'intervention', 'condition', 'side_effect'}

    # Valid correlation types
    VALID_CORRELATION_TYPES = {'positive', 'negative', 'neutral', 'inconclusive'}

    # Valid delivery methods
    VALID_DELIVERY_METHODS = {
        'oral', 'injection', 'topical', 'inhalation', 'behavioral', 'digital',
        'surgical', 'intravenous', 'sublingual', 'rectal', 'transdermal',
        'acupuncture', 'nasal', 'nasal spray', 'intranasal', 'enema',
        'subcutaneous', 'intramuscular', 'needling', 'local application',
        'neuromodulation', 'electrical stimulation', 'subcutaneous injection',
        'acupuncture needles', 'blunt-tipped needles at non-acupoints',
        'educational', 'supervised', 'counseling', 'oral capsules or colonoscopy'
    }

    # Valid severity levels
    VALID_SEVERITY_LEVELS = {'mild', 'moderate', 'severe'}

    # Valid cost categories
    VALID_COST_CATEGORIES = {'low', 'medium', 'high'}

    @staticmethod
    def validate_entity_type(entity_type: str) -> str:
        """Validate and normalize entity type."""
        if not isinstance(entity_type, str):
            raise ValidationError("Entity type must be a string", "INVALID_TYPE")

        entity_type = entity_type.strip().lower()
        if not entity_type:
            raise ValidationError("Entity type must be a non-empty string", "EMPTY_ENTITY_TYPE")

        if entity_type not in EntityValidator.VALID_ENTITY_TYPES:
            valid_types = sorted(EntityValidator.VALID_ENTITY_TYPES)
            raise ValidationError(
                f"Invalid entity type '{entity_type}'. Must be one of: {valid_types}",
                "INVALID_ENTITY_TYPE",
                {"provided": entity_type, "valid_options": valid_types}
            )

        return entity_type

    @staticmethod
    def validate_term(term: str) -> str:
        """Validate and clean term string."""
        if not isinstance(term, str):
            raise ValidationError("Term must be a string", "INVALID_TERM_TYPE")

        term = term.strip()
        if not term:
            raise ValidationError("Term cannot be empty or whitespace only", "EMPTY_TERM")

        if len(term) > 1000:
            raise ValidationError("Term is too long (max 1000 characters)", "TERM_TOO_LONG")

        return term

    @staticmethod
    def validate_confidence(confidence: Union[int, float, None]) -> Optional[float]:
        """Validate confidence score."""
        if confidence is None:
            return None

        if not isinstance(confidence, (int, float)):
            raise ValidationError("Confidence must be a number", "INVALID_CONFIDENCE_TYPE")

        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValidationError(
                "Confidence must be between 0.0 and 1.0",
                "CONFIDENCE_OUT_OF_RANGE",
                {"provided": confidence, "valid_range": [0.0, 1.0]}
            )

        return confidence

    @staticmethod
    def validate_intervention(intervention: Dict[str, Any]) -> ValidationResult:
        """Validate intervention data structure."""
        errors = []
        warnings = []
        cleaned_data = intervention.copy()

        # Required fields
        required_fields = ['intervention_name', 'health_condition', 'correlation_type']
        for field in required_fields:
            if field not in intervention or not intervention[field]:
                errors.append(f"Missing required field: {field}")

        # Validate entity type if present
        if 'intervention_category' in intervention:
            try:
                cleaned_data['intervention_category'] = EntityValidator.validate_entity_type(
                    intervention['intervention_category']
                )
            except ValidationError as e:
                errors.append(f"Invalid intervention_category: {e}")

        # Validate confidence fields
        for conf_field in ['study_confidence']:
            if conf_field in intervention:
                try:
                    cleaned_data[conf_field] = EntityValidator.validate_confidence(
                        intervention[conf_field]
                    )
                except ValidationError as e:
                    errors.append(f"Invalid {conf_field}: {e}")

        # Validate correlation type
        if 'correlation_type' in intervention:
            if intervention['correlation_type'] not in EntityValidator.VALID_CORRELATION_TYPES:
                errors.append(f"Invalid correlation_type. Must be one of: {EntityValidator.VALID_CORRELATION_TYPES}")

        # Validate delivery method
        if 'delivery_method' in intervention and intervention['delivery_method']:
            if intervention['delivery_method'] not in EntityValidator.VALID_DELIVERY_METHODS:
                warnings.append(f"Unusual delivery_method: {intervention['delivery_method']}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_data if len(errors) == 0 else None
        )

# === NORMALIZATION UTILITIES ===

class EntityNormalizer:
    """Utilities for normalizing entity terms."""

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

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove punctuation and special characters except hyphens and spaces
        normalized = re.sub(r'[^\w\s\-]', '', normalized)

        # Remove extra hyphens and spaces
        normalized = re.sub(r'[-\s]+', ' ', normalized)

        return normalized.strip()

    @staticmethod
    def create_search_variants(term: str) -> Set[str]:
        """Create search variants for a term."""
        variants = {term.lower().strip()}

        # Remove punctuation variant
        no_punct = re.sub(r'[^\w\s]', '', term.lower())
        variants.add(no_punct.strip())

        # Hyphen to space variant
        hyphen_to_space = term.replace('-', ' ').lower().strip()
        variants.add(hyphen_to_space)

        # Space to hyphen variant
        space_to_hyphen = term.replace(' ', '-').lower().strip()
        variants.add(space_to_hyphen)

        # Remove common prefixes/suffixes
        prefixes = ['the ', 'a ', 'an ']
        suffixes = [' therapy', ' treatment', ' intervention']

        base_term = term.lower().strip()
        for prefix in prefixes:
            if base_term.startswith(prefix):
                variants.add(base_term[len(prefix):])

        for suffix in suffixes:
            if base_term.endswith(suffix):
                variants.add(base_term[:-len(suffix)])

        # Remove empty variants
        return {v for v in variants if v.strip()}

# === CACHING UTILITIES ===

class CacheManager:
    """Unified caching utilities."""

    @staticmethod
    def create_cache_key(term: str, entity_type: str, context: str = "") -> str:
        """Create a consistent cache key."""
        # Normalize components
        norm_term = EntityNormalizer.normalize_term(term)
        norm_type = entity_type.lower().strip()
        norm_context = context.lower().strip() if context else ""

        # Create composite key
        key_parts = [norm_term, norm_type, norm_context]
        key_string = "|".join(filter(None, key_parts))

        # Hash for consistent length and special character handling
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    @staticmethod
    def is_cache_key_valid(cache_key: str) -> bool:
        """Validate cache key format."""
        return (isinstance(cache_key, str) and
                len(cache_key) == 32 and  # MD5 hash length
                cache_key.isalnum())

# === CONFIDENCE UTILITIES ===

class ConfidenceCalculator:
    """Utilities for confidence score calculations."""

    @staticmethod
    def get_effective_confidence(intervention: Dict[str, Any]) -> float:
        """
        Get effective confidence from intervention data.

        Uses consensus_confidence as primary metric.
        """
        # Use consensus_confidence (from consensus processing)
        if 'consensus_confidence' in intervention:
            consensus_conf = intervention.get('consensus_confidence', 0)
            if consensus_conf is not None:
                return float(consensus_conf)

        # Fall back to study_confidence
        if 'study_confidence' in intervention:
            study_conf = intervention.get('study_confidence', 0)
            if study_conf is not None:
                return float(study_conf)

        return 0.0

    @staticmethod
    def merge_dual_confidence(interventions: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """
        Merge confidence values from multiple interventions.

        Returns:
            Tuple of (extraction_confidence=0.0, study_confidence, legacy_confidence=0.0)
            Note: extraction_confidence always returns 0.0 for backward compatibility
        """
        study_confidences = []

        for intervention in interventions:
            # Extract study_confidence
            study_conf = intervention.get('study_confidence')
            if study_conf is not None:
                study_confidences.append(float(study_conf))

        # Calculate merged study confidence (average)
        merged_study = sum(study_confidences) / len(study_confidences) if study_confidences else 0.0

        # Return (0.0, study_confidence, 0.0) for backward compatibility with existing code
        return 0.0, merged_study, 0.0

    @staticmethod
    def boost_confidence_for_agreement(base_confidence: float, agreement_count: int) -> float:
        """Apply confidence boost for multi-model agreement."""
        # Boost from multi-model agreement (up to 10% boost)
        boost_factor = 0.05 * (agreement_count - 1)
        boosted = base_confidence + boost_factor

        # Cap at 0.98 to maintain some uncertainty
        return min(0.98, boosted)

# === JSON UTILITIES ===

class JsonUtils:
    """Utilities for JSON processing."""

    @staticmethod
    def safe_json_loads(json_string: str) -> Optional[Any]:
        """Safely parse JSON string with error handling."""
        if not json_string or not isinstance(json_string, str):
            return None

        try:
            # Clean common markdown formatting
            cleaned = json_string.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def safe_json_dumps(obj: Any) -> str:
        """Safely serialize object to JSON."""
        try:
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return "{}"

# === STRING UTILITIES ===

class StringUtils:
    """String processing utilities."""

    @staticmethod
    def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate string to specified length."""
        if not text or len(text) <= max_length:
            return text

        return text[:max_length - len(suffix)] + suffix

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Clean and normalize whitespace in text."""
        if not text:
            return ""

        # Replace multiple whitespace with single space
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()

    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract numeric values from text."""
        if not text:
            return []

        # Find all numeric patterns including decimals
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) for n in numbers if n]

# === CONSTANTS ===

# Default confidence thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
HIGH_CONFIDENCE_THRESHOLD = 0.8
LOW_CONFIDENCE_THRESHOLD = 0.3

# Cache settings
DEFAULT_CACHE_TTL = 3600  # 1 hour
LONG_CACHE_TTL = 86400    # 24 hours

# Performance settings
DEFAULT_BATCH_SIZE = 100
MAX_BATCH_SIZE = 1000
MAX_TERM_LENGTH = 1000

# Database settings
DEFAULT_DB_TIMEOUT = 30
MAX_RETRY_ATTEMPTS = 3