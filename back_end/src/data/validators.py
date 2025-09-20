"""
Comprehensive input validation for the MyBiome system.
Provides validation for paper format, valid interventions, and API inputs.
"""

import re
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
from src.data.config import setup_logging

logger = setup_logging(__name__, 'validators.log')


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None


@dataclass 
class ValidationResult:
    """Result of validation process."""
    is_valid: bool
    issues: List[ValidationIssue]
    cleaned_data: Optional[Dict] = None
    
    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [issue for issue in self.issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]


class BaseValidator:
    """Base validator class with common validation methods."""
    
    def __init__(self):
        self.logger = logger
    
    def validate_required_fields(self, data: Dict, required_fields: List[str]) -> List[ValidationIssue]:
        """Validate that required fields are present and not empty."""
        issues = []
        
        for field in required_fields:
            if field not in data:
                issues.append(ValidationIssue(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    severity=ValidationSeverity.ERROR
                ))
            elif not data[field] or (isinstance(data[field], str) and not data[field].strip()):
                issues.append(ValidationIssue(
                    field=field,
                    message=f"Required field '{field}' is empty",
                    severity=ValidationSeverity.ERROR,
                    value=data[field]
                ))
        
        return issues
    
    def validate_string_length(self, value: str, field: str, min_length: int = None, 
                              max_length: int = None) -> List[ValidationIssue]:
        """Validate string length constraints."""
        issues = []
        
        if not isinstance(value, str):
            issues.append(ValidationIssue(
                field=field,
                message=f"Field '{field}' must be a string",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return issues
        
        length = len(value.strip())
        
        if min_length is not None and length < min_length:
            issues.append(ValidationIssue(
                field=field,
                message=f"Field '{field}' must be at least {min_length} characters (got {length})",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        if max_length is not None and length > max_length:
            issues.append(ValidationIssue(
                field=field,
                message=f"Field '{field}' must be at most {max_length} characters (got {length})",
                severity=ValidationSeverity.WARNING,
                value=value
            ))
        
        return issues
    
    def validate_enum_value(self, value: Any, field: str, valid_values: Union[Set, List]) -> List[ValidationIssue]:
        """Validate that value is in allowed set."""
        issues = []
        
        if value not in valid_values:
            issues.append(ValidationIssue(
                field=field,
                message=f"Field '{field}' has invalid value '{value}'. Must be one of: {list(valid_values)}",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return issues
    
    def validate_numeric_range(self, value: Union[int, float], field: str, 
                              min_value: Optional[float] = None, 
                              max_value: Optional[float] = None) -> List[ValidationIssue]:
        """Validate numeric range constraints."""
        issues = []
        
        if not isinstance(value, (int, float)):
            issues.append(ValidationIssue(
                field=field,
                message=f"Field '{field}' must be numeric",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return issues
        
        if min_value is not None and value < min_value:
            issues.append(ValidationIssue(
                field=field,
                message=f"Field '{field}' must be at least {min_value} (got {value})",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        if max_value is not None and value > max_value:
            issues.append(ValidationIssue(
                field=field,
                message=f"Field '{field}' must be at most {max_value} (got {value})",
                severity=ValidationSeverity.ERROR,
                value=value
            ))
        
        return issues


class PaperValidator(BaseValidator):
    """Validator for research paper data."""
    
    REQUIRED_FIELDS = ['pmid', 'title', 'abstract']
    PMID_PATTERN = re.compile(r'^\d{7,8}$')  # PubMed IDs are typically 7-8 digits
    
    def validate(self, paper_data: Dict) -> ValidationResult:
        """Validate paper data."""
        issues = []
        cleaned_data = paper_data.copy()
        
        # Required fields
        issues.extend(self.validate_required_fields(paper_data, self.REQUIRED_FIELDS))
        
        # PMID format validation
        if 'pmid' in paper_data:
            pmid = str(paper_data['pmid'])
            if not self.PMID_PATTERN.match(pmid):
                issues.append(ValidationIssue(
                    field='pmid',
                    message=f"PMID '{pmid}' has invalid format. Expected 7-8 digits.",
                    severity=ValidationSeverity.ERROR,
                    value=pmid
                ))
            else:
                cleaned_data['pmid'] = pmid  # Ensure string format
        
        # Title validation
        if 'title' in paper_data:
            issues.extend(self.validate_string_length(
                paper_data['title'], 'title', min_length=10, max_length=500
            ))
        
        # Abstract validation
        if 'abstract' in paper_data:
            issues.extend(self.validate_string_length(
                paper_data['abstract'], 'abstract', min_length=50, max_length=5000
            ))
        
        # Optional field validation
        if 'journal' in paper_data:
            issues.extend(self.validate_string_length(
                paper_data['journal'], 'journal', max_length=200
            ))
        
        # Publication date format
        if 'publication_date' in paper_data:
            date_value = paper_data['publication_date']
            if date_value and not re.match(r'^\d{4}(-\d{2})?(-\d{2})?$', str(date_value)):
                issues.append(ValidationIssue(
                    field='publication_date',
                    message=f"Publication date '{date_value}' has invalid format. Expected YYYY, YYYY-MM, or YYYY-MM-DD.",
                    severity=ValidationSeverity.WARNING,
                    value=date_value
                ))
        
        # DOI format validation
        if 'doi' in paper_data and paper_data['doi']:
            doi = paper_data['doi']
            if not re.match(r'^10\.\d+/.+', doi):
                issues.append(ValidationIssue(
                    field='doi',
                    message=f"DOI '{doi}' has invalid format.",
                    severity=ValidationSeverity.WARNING,
                    value=doi
                ))
        
        # Determine if validation passed
        is_valid = len([issue for issue in issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            cleaned_data=cleaned_data if is_valid else None
        )


class InterventionValidator(BaseValidator):
    """Validator for intervention data."""
    
    REQUIRED_FIELDS = ['intervention_category', 'intervention_name', 'health_condition', 'correlation_type']
    VALID_CATEGORIES = ['exercise', 'diet', 'supplement', 'medication', 'therapy', 'lifestyle', 'surgery', 'test', 'emerging']
    VALID_CORRELATION_TYPES = ['positive', 'negative', 'neutral', 'inconclusive']
    VALID_DELIVERY_METHODS = ['oral', 'injection', 'topical', 'inhalation', 'behavioral', 'digital', 'surgical', 'intravenous', 'sublingual', 'rectal', 'transdermal']
    VALID_SEVERITY_LEVELS = ['mild', 'moderate', 'severe']
    VALID_COST_CATEGORIES = ['low', 'medium', 'high']
    
    def validate(self, intervention_data: Dict) -> ValidationResult:
        """Validate intervention data."""
        issues = []
        cleaned_data = intervention_data.copy()
        
        # Required fields
        issues.extend(self.validate_required_fields(intervention_data, self.REQUIRED_FIELDS))
        
        # Category validation
        if 'intervention_category' in intervention_data:
            issues.extend(self.validate_enum_value(
                intervention_data['intervention_category'], 
                'intervention_category', 
                self.VALID_CATEGORIES
            ))
        
        # Intervention name validation
        if 'intervention_name' in intervention_data:
            name = intervention_data['intervention_name']
            issues.extend(self.validate_string_length(name, 'intervention_name', min_length=2, max_length=200))
            
            # Check for placeholder values
            placeholder_patterns = ['...', 'intervention', 'treatment', 'therapy', 'unknown', 'n/a']
            if any(placeholder in name.lower() for placeholder in placeholder_patterns):
                issues.append(ValidationIssue(
                    field='intervention_name',
                    message=f"Intervention name '{name}' appears to be a placeholder",
                    severity=ValidationSeverity.ERROR,
                    value=name
                ))
        
        # Health condition validation
        if 'health_condition' in intervention_data:
            issues.extend(self.validate_string_length(
                intervention_data['health_condition'], 'health_condition', min_length=2, max_length=200
            ))
        
        # Correlation type validation
        if 'correlation_type' in intervention_data:
            issues.extend(self.validate_enum_value(
                intervention_data['correlation_type'],
                'correlation_type',
                self.VALID_CORRELATION_TYPES
            ))
        
        # Numeric field validation
        for field, (min_val, max_val) in [
            ('correlation_strength', (0.0, 1.0)),
            ('confidence_score', (0.0, 1.0))
        ]:
            if field in intervention_data and intervention_data[field] is not None:
                issues.extend(self.validate_numeric_range(
                    intervention_data[field], field, min_val, max_val
                ))
        
        # Sample size validation
        if 'sample_size' in intervention_data and intervention_data['sample_size'] is not None:
            issues.extend(self.validate_numeric_range(
                intervention_data['sample_size'], 'sample_size', min_value=1
            ))
        
        # Paper ID validation (if present)
        if 'paper_id' in intervention_data:
            pmid = str(intervention_data['paper_id'])
            if not PaperValidator.PMID_PATTERN.match(pmid):
                issues.append(ValidationIssue(
                    field='paper_id',
                    message=f"Paper ID '{pmid}' has invalid PMID format",
                    severity=ValidationSeverity.ERROR,
                    value=pmid
                ))
        
        # Model name validation
        if 'extraction_model' in intervention_data:
            model = intervention_data['extraction_model']
            if not model or len(model.strip()) < 3:
                issues.append(ValidationIssue(
                    field='extraction_model',
                    message=f"Extraction model '{model}' is too short or empty",
                    severity=ValidationSeverity.WARNING,
                    value=model
                ))

        # Optional new fields validation
        if 'delivery_method' in intervention_data and intervention_data['delivery_method']:
            delivery_method = intervention_data['delivery_method'].lower() if isinstance(intervention_data['delivery_method'], str) else intervention_data['delivery_method']
            issues.extend(self.validate_enum_value(
                delivery_method,
                'delivery_method',
                self.VALID_DELIVERY_METHODS
            ))

        if 'severity' in intervention_data and intervention_data['severity']:
            severity = intervention_data['severity'].lower() if isinstance(intervention_data['severity'], str) else intervention_data['severity']
            issues.extend(self.validate_enum_value(
                severity,
                'severity',
                self.VALID_SEVERITY_LEVELS
            ))

        if 'cost_category' in intervention_data and intervention_data['cost_category']:
            cost_category = intervention_data['cost_category'].lower() if isinstance(intervention_data['cost_category'], str) else intervention_data['cost_category']
            issues.extend(self.validate_enum_value(
                cost_category,
                'cost_category',
                self.VALID_COST_CATEGORIES
            ))

        # Adverse effects validation (just check it's reasonable length)
        if 'adverse_effects' in intervention_data and intervention_data['adverse_effects']:
            issues.extend(self.validate_string_length(
                intervention_data['adverse_effects'], 'adverse_effects', max_length=1000
            ))
        
        # Determine if validation passed
        is_valid = len([issue for issue in issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            cleaned_data=cleaned_data if is_valid else None
        )


class APIInputValidator(BaseValidator):
    """Validator for API inputs and parameters."""
    
    def validate_search_query(self, query: str) -> ValidationResult:
        """Validate search query input."""
        issues = []
        
        if not query or not query.strip():
            issues.append(ValidationIssue(
                field='query',
                message="Search query cannot be empty",
                severity=ValidationSeverity.ERROR,
                value=query
            ))
        else:
            # Check length
            issues.extend(self.validate_string_length(query, 'query', min_length=3, max_length=1000))
            
            # Check for potentially problematic characters
            if re.search(r'[<>\'";]', query):
                issues.append(ValidationIssue(
                    field='query',
                    message="Query contains potentially unsafe characters",
                    severity=ValidationSeverity.WARNING,
                    value=query
                ))
        
        is_valid = len(self.get_errors(issues)) == 0
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            cleaned_data={'query': query.strip()} if is_valid else None
        )
    
    def validate_pagination_params(self, limit: Optional[int] = None, 
                                  offset: Optional[int] = None) -> ValidationResult:
        """Validate pagination parameters."""
        issues = []
        
        if limit is not None:
            issues.extend(self.validate_numeric_range(limit, 'limit', min_value=1, max_value=1000))
        
        if offset is not None:
            issues.extend(self.validate_numeric_range(offset, 'offset', min_value=0))
        
        is_valid = len(self.get_errors(issues)) == 0
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            cleaned_data={'limit': limit, 'offset': offset} if is_valid else None
        )
    
    def get_errors(self, issues: List[ValidationIssue]) -> List[ValidationIssue]:
        """Helper to get only error-level issues."""
        return [issue for issue in issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]


class ValidationManager:
    """Central manager for all validation operations."""
    
    def __init__(self):
        self.paper_validator = PaperValidator()
        self.intervention_validator = InterventionValidator()
        self.api_validator = APIInputValidator()
        self.logger = logger
    
    def validate_paper(self, paper_data: Dict) -> ValidationResult:
        """Validate paper data."""
        return self.paper_validator.validate(paper_data)
    
    def validate_intervention(self, intervention_data: Dict) -> ValidationResult:
        """Validate intervention data."""
        return self.intervention_validator.validate(intervention_data)
    
    def validate_api_input(self, input_type: str, **kwargs) -> ValidationResult:
        """Validate API inputs based on type."""
        if input_type == 'search_query':
            return self.api_validator.validate_search_query(kwargs.get('query', ''))
        elif input_type == 'pagination':
            return self.api_validator.validate_pagination_params(
                kwargs.get('limit'), kwargs.get('offset')
            )
        else:
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    field='input_type',
                    message=f"Unknown input type: {input_type}",
                    severity=ValidationSeverity.ERROR,
                    value=input_type
                )]
            )
    
    def log_validation_issues(self, result: ValidationResult, context: str = ""):
        """Log validation issues for monitoring."""
        if result.errors:
            error_messages = [f"{issue.field}: {issue.message}" for issue in result.errors]
            self.logger.error(f"Validation errors in {context}: {'; '.join(error_messages)}")
        
        if result.warnings:
            warning_messages = [f"{issue.field}: {issue.message}" for issue in result.warnings]
            self.logger.warning(f"Validation warnings in {context}: {'; '.join(warning_messages)}")


# Global validation manager instance
validation_manager = ValidationManager()