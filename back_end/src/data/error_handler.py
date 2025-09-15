"""
Centralized error handling and recovery for the MyBiome system.
Provides robust error handling with retry logic and circuit breaker patterns.
"""

import time
import requests
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from functools import wraps  #?Preserve original function name and docstrings when using decorators
from dataclasses import dataclass
from enum import Enum #? tool for creating enumerations (named constants)
import sys
from pathlib import Path

from src.data.config import setup_logging

logger = setup_logging(__name__, 'error_handler.log')

class ErrorSeverity(Enum):
    """We defined severity levels as Enumerations."""
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

#* Stores error metadata that occured during some operation
@dataclass
class ErrorContext:
    """ Context for Error Handling."""
    operation: str
    component: str
    paper_id: Optional[str] = None
    model_name: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    additional_info: Optional[Dict] = None

class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """
    Implementation of the Circuit breaker pattern for external services.
    The circuit breaker has three states:
        1. Closed - Normal operation; requests pass through
        2. Open - Service is failing; requests are blocked immediately
        3. Half-Open - Testing if service has recovered; allows one request through
    
    """
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """
        Args:
            failure_threshold: Number of failures before opening the circuit (default: 5)
            recovery_timeout: Seconds to wait before attempting recovery (default: 60)
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == 'open':
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise CircuitBreakerError(f"Circuit breaker open for {func.__name__}")
            else:
                self.state = 'half-open'
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RetryHandler:
    """Enhanced retry handler with exponential backoff and jitter."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 backoff_factor: float = 2.0, max_delay: float = 60.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.jitter = jitter
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)  # Add 50% jitter
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.max_retries:
            return False
        
        # Retry on specific exceptions
        retry_exceptions = (
            requests.exceptions.RequestException,
            ConnectionError,
            TimeoutError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        )
        
        # Don't retry on authentication errors
        no_retry_exceptions = (
            requests.exceptions.HTTPError,
        )
        
        if isinstance(exception, no_retry_exceptions):
            if hasattr(exception, 'response') and exception.response is not None:
                if exception.response.status_code in [401, 403, 404]:
                    return False
        
        return isinstance(exception, retry_exceptions)
    
class ErrorHandler:
    """Centralized error handler for all major components of the pipeline.
    That includes API errors and database connection errors."""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.retry_handler = RetryHandler()
        self.error_counts = {}
        self.logger = logger
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]
    
    def handle_api_error(self, error: Exception, context: ErrorContext) -> Tuple[bool, Optional[str]]:
        """
        Handle API-related errors with appropriate retry logic.
        
        Returns:
            Tuple of (should_retry, error_message)
        """
        error_message = str(error)
        
        # Rate limiting
        if isinstance(error, requests.exceptions.HTTPError):
            if hasattr(error, 'response') and error.response is not None:
                if error.response.status_code == 429:
                    retry_after = error.response.headers.get('Retry-After', '60')
                    self.logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(int(retry_after))
                    return True, "Rate limit exceeded, retrying after delay"
                
                elif error.response.status_code >= 500:
                    return True, f"Server error ({error.response.status_code}), retrying"
                
                elif error.response.status_code in [401, 403]:
                    return False, f"Authentication error ({error.response.status_code})"
        
        # Network errors
        if isinstance(error, (requests.exceptions.ConnectionError, 
                             requests.exceptions.Timeout,
                             ConnectionError, TimeoutError)):
            return True, f"Network error: {error_message}"
        
        # JSON parsing errors (usually from LLM responses)
        if isinstance(error, (ValueError, KeyError)) and ('json' in error_message.lower() or 
                                                         'parse' in error_message.lower()):
            return True, f"Response parsing error: {error_message}"
        
        # Default: don't retry unknown errors
        return False, f"Unhandled error: {error_message}"
    
    def handle_database_error(self, error: Exception, context: ErrorContext) -> Tuple[bool, Optional[str]]:
        """Handle database-related errors."""
        error_message = str(error)
        
        # SQLite lock errors
        if 'database is locked' in error_message.lower():
            self.logger.warning(f"Database locked for {context.operation}, retrying...")
            time.sleep(0.5)  # Short delay for lock release
            return True, "Database locked, retrying"
        
        # Connection errors
        if 'connection' in error_message.lower():
            return True, f"Database connection error: {error_message}"
        
        # Don't retry schema or constraint errors
        if any(keyword in error_message.lower() for keyword in ['constraint', 'schema', 'syntax']):
            return False, f"Database schema error: {error_message}"
        
        # Default: retry most database errors once
        if context.retry_count == 0:
            return True, f"Database error: {error_message}"
        
        return False, f"Persistent database error: {error_message}"
    
    def handle_validation_error(self, error: Exception, context: ErrorContext) -> Tuple[bool, Optional[str]]:
        """Handle validation errors."""
        error_message = str(error)
        
        # Log validation errors for analysis
        self.logger.warning(f"Validation error in {context.component}: {error_message}")
        
        # Track validation error frequency
        key = f"{context.component}:{context.operation}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Don't retry validation errors - they indicate data quality issues
        return False, f"Validation error: {error_message}"
    
    def execute_with_retry(self, func: Callable, context: ErrorContext, 
                          *args, **kwargs) -> Any:
        """
        Execute function with comprehensive error handling and retry logic.
        
        Args:
            func: Function to execute
            context: Error context information
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries are exhausted
        """
        last_error = None
        
        for attempt in range(context.max_retries + 1):
            try:
                context.retry_count = attempt
                
                # Use circuit breaker for external services
                if context.component in ['api', 'llm', 'pubmed']:
                    circuit_breaker = self.get_circuit_breaker(context.component)
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as error:
                last_error = error
                
                # Determine error handling strategy
                should_retry, error_msg = self._categorize_and_handle_error(error, context)
                
                if not should_retry or attempt >= context.max_retries:
                    self.logger.error(f"Operation {context.operation} failed permanently: {error_msg}")
                    raise error
                
                # Calculate retry delay
                delay = self.retry_handler.calculate_delay(attempt)
                
                self.logger.warning(f"Operation {context.operation} failed (attempt {attempt + 1}/{context.max_retries + 1}): {error_msg}. Retrying in {delay:.1f}s")
                
                time.sleep(delay)
        
        # Should never reach here, but just in case
        if last_error:
            raise last_error
    
    def _categorize_and_handle_error(self, error: Exception, context: ErrorContext) -> Tuple[bool, str]:
        """Categorize error and determine handling strategy."""
        
        # API/Network errors
        if isinstance(error, (requests.exceptions.RequestException, ConnectionError, TimeoutError)):
            return self.handle_api_error(error, context)
        
        # Database errors
        if any(keyword in str(error).lower() for keyword in ['database', 'sqlite', 'sql']):
            return self.handle_database_error(error, context)
        
        # Validation errors
        if any(keyword in type(error).__name__.lower() for keyword in ['validation', 'value', 'key']):
            return self.handle_validation_error(error, context)
        
        # LLM-specific errors
        if any(keyword in str(error).lower() for keyword in ['model', 'completion', 'token']):
            return True, f"LLM error: {str(error)}"
        
        # Default handling
        return False, f"Unknown error: {str(error)}"
    
    def log_error_summary(self) -> Dict[str, Any]:
        """Generate error summary report."""
        summary = {
            'circuit_breaker_states': {name: cb.state for name, cb in self.circuit_breakers.items()},
            'error_counts': dict(self.error_counts),
            'total_errors': sum(self.error_counts.values())
        }
        
        self.logger.info(f"Error summary: {summary}")
        return summary
    
def with_error_handling(operation: str, component: str, max_retries: int = 3):
    """
    Decorator for adding comprehensive error handling to functions.
    
    Args:
        operation: Description of the operation
        component: Component name (api, database, llm, etc.)
        max_retries: Maximum retry attempts
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=operation,
                component=component,
                max_retries=max_retries
            )
            
            return error_handler.execute_with_retry(func, context, *args, **kwargs)
        
        return wrapper
    return decorator


#* Global error handler instance
error_handler = ErrorHandler()

#* Convenience decorators for common use cases
def handle_api_errors(operation: str, max_retries: int = 3):
    """Decorator for API operations."""
    return with_error_handling(operation, 'api', max_retries)

def handle_database_errors(operation: str, max_retries: int = 2):
    """Decorator for database operations."""
    return with_error_handling(operation, 'database', max_retries)

def handle_llm_errors(operation: str, max_retries: int = 3):
    """Decorator for LLM operations."""
    return with_error_handling(operation, 'llm', max_retries)