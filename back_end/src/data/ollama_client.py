"""
Unified Ollama API Client

Centralized client for all Ollama LLM interactions with built-in:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Timeout handling
- Error recovery
- Temperature control
- Response validation

Created: October 16, 2025 (Round 2 Cleanup)
Purpose: Eliminate 5+ instances of duplicate LLM API call code
"""

import time
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

try:
    from .config import config, setup_logging
    from .constants import (
        LLM_MODEL_CURRENT,
        OLLAMA_API_URL,
        OLLAMA_TIMEOUT_SECONDS,
        OLLAMA_RETRY_DELAYS
    )
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.data.constants import (
        LLM_MODEL_CURRENT,
        OLLAMA_API_URL,
        OLLAMA_TIMEOUT_SECONDS,
        OLLAMA_RETRY_DELAYS
    )

logger = setup_logging(__name__, 'ollama_client.log')


class OllamaClientError(Exception):
    """Base exception for Ollama client errors."""
    pass


class OllamaTimeoutError(OllamaClientError):
    """Timeout waiting for Ollama response."""
    pass


class OllamaCircuitBreakerOpen(OllamaClientError):
    """Circuit breaker is open, preventing requests."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for Ollama API calls.

    Prevents cascading failures by temporarily blocking requests
    after repeated failures.
    """

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open

    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.state = 'closed'

    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will retry after {self.timeout_seconds}s"
            )

    def can_attempt(self) -> bool:
        """Check if request can be attempted."""
        if self.state == 'closed':
            return True

        if self.state == 'open':
            # Check if timeout has elapsed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.timeout_seconds:
                    self.state = 'half_open'
                    logger.info("Circuit breaker entering half-open state")
                    return True
            return False

        # half_open state - allow one attempt
        return True


class OllamaClient:
    """
    Unified client for Ollama LLM API interactions.

    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker for failure prevention
    - Configurable timeout and temperature
    - JSON mode support
    - Streaming support (future)

    Usage:
        client = OllamaClient(model="qwen3:14b", temperature=0.0)
        response = client.generate(prompt="Analyze this data...")
    """

    def __init__(
        self,
        model: str = LLM_MODEL_CURRENT,
        temperature: float = 0.0,
        api_url: str = OLLAMA_API_URL,
        timeout: int = OLLAMA_TIMEOUT_SECONDS,
        max_retries: int = 3,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """
        Initialize Ollama client.

        Args:
            model: LLM model name (default: qwen3:14b)
            temperature: Sampling temperature 0.0-1.0 (default: 0.0 for deterministic)
            api_url: Ollama API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            circuit_breaker: Optional custom circuit breaker
        """
        self.model = model
        self.temperature = temperature
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

        logger.info(f"Initialized OllamaClient: model={model}, temp={temperature}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature_override: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate completion from Ollama.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            json_mode: Enable JSON mode (forces valid JSON output)
            temperature_override: Override default temperature for this request
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            OllamaClientError: On API error
            OllamaTimeoutError: On timeout
            OllamaCircuitBreakerOpen: When circuit breaker is open
        """
        if not self.circuit_breaker.can_attempt():
            raise OllamaCircuitBreakerOpen(
                "Circuit breaker is open. Too many recent failures."
            )

        last_error = None
        retry_delays = OLLAMA_RETRY_DELAYS[:self.max_retries]

        for attempt in range(self.max_retries):
            try:
                response_text = self._make_request(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    json_mode=json_mode,
                    temperature=temperature_override or self.temperature,
                    max_tokens=max_tokens
                )

                self.circuit_breaker.record_success()
                return response_text

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Ollama timeout (attempt {attempt + 1}/{self.max_retries})")
                self.circuit_breaker.record_failure()

                if attempt < self.max_retries - 1:
                    delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(f"Ollama API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                self.circuit_breaker.record_failure()

                if attempt < self.max_retries - 1:
                    delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error during Ollama request: {e}")
                self.circuit_breaker.record_failure()
                raise OllamaClientError(f"Unexpected error: {e}") from e

        # All retries exhausted
        self.circuit_breaker.record_failure()

        if isinstance(last_error, requests.exceptions.Timeout):
            raise OllamaTimeoutError(
                f"Ollama request timed out after {self.max_retries} attempts"
            ) from last_error
        else:
            raise OllamaClientError(
                f"Ollama request failed after {self.max_retries} attempts: {last_error}"
            ) from last_error

    def _make_request(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_mode: bool,
        temperature: float,
        max_tokens: Optional[int]
    ) -> str:
        """Make actual HTTP request to Ollama API."""
        endpoint = f"{self.api_url}/api/generate"

        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': temperature
            }
        }

        if system_prompt:
            payload['system'] = system_prompt

        if json_mode:
            payload['format'] = 'json'

        if max_tokens:
            payload['options']['num_predict'] = max_tokens

        logger.debug(f"Ollama request: model={self.model}, temp={temperature}, json_mode={json_mode}")

        response = requests.post(
            endpoint,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()
        response_text = result.get('response', '').strip()

        if not response_text:
            raise OllamaClientError("Empty response from Ollama")

        logger.debug(f"Ollama response received: {len(response_text)} chars")
        return response_text

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        json_mode: bool = False
    ) -> List[str]:
        """
        Generate completions for multiple prompts.

        Note: Processes sequentially (Ollama doesn't support batch API).

        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt (shared)
            json_mode: Enable JSON mode for all

        Returns:
            List of generated responses
        """
        results = []

        for i, prompt in enumerate(prompts):
            logger.info(f"Processing batch item {i + 1}/{len(prompts)}")
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=json_mode
            )
            results.append(response)

        return results

    def is_available(self) -> bool:
        """
        Check if Ollama API is available.

        Returns:
            True if Ollama is responding
        """
        try:
            response = requests.get(
                f"{self.api_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        List available models in Ollama.

        Returns:
            List of model names
        """
        try:
            response = requests.get(
                f"{self.api_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()

            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []


# Global client instance (singleton pattern)
_default_client: Optional[OllamaClient] = None


def get_default_client() -> OllamaClient:
    """
    Get or create default Ollama client instance.

    Returns:
        Singleton OllamaClient instance
    """
    global _default_client

    if _default_client is None:
        _default_client = OllamaClient()
        logger.info("Created default OllamaClient instance")

    return _default_client


def generate_completion(
    prompt: str,
    model: str = LLM_MODEL_CURRENT,
    temperature: float = 0.0,
    json_mode: bool = False
) -> str:
    """
    Convenience function for one-off completions.

    Args:
        prompt: User prompt
        model: LLM model name
        temperature: Sampling temperature
        json_mode: Enable JSON mode

    Returns:
        Generated completion
    """
    client = OllamaClient(model=model, temperature=temperature)
    return client.generate(prompt=prompt, json_mode=json_mode)


if __name__ == "__main__":
    # Test the client
    import argparse

    parser = argparse.ArgumentParser(description="Test Ollama Client")
    parser.add_argument('--check', action='store_true', help='Check if Ollama is available')
    parser.add_argument('--models', action='store_true', help='List available models')
    parser.add_argument('--prompt', type=str, help='Test prompt')

    args = parser.parse_args()

    client = get_default_client()

    if args.check:
        available = client.is_available()
        print(f"Ollama available: {available}")

    elif args.models:
        models = client.list_models()
        print(f"Available models ({len(models)}):")
        for model in models:
            print(f"  - {model}")

    elif args.prompt:
        print(f"Generating completion for: {args.prompt[:50]}...")
        response = client.generate(args.prompt)
        print(f"\nResponse:\n{response}")

    else:
        parser.print_help()
