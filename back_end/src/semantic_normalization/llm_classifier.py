"""
LLM Classifier for Hierarchical Semantic Normalization

Uses qwen3:14b for:
1. Canonical group extraction (Layer 1)
2. Relationship type classification (EXACT_MATCH, VARIANT, etc.)

Includes ground truth examples (Scenarios 1-3) for few-shot learning.
"""

import os
import re
import json
import pickle
import logging
import requests
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Import prompt templates
from .prompts import (
    format_canonical_extraction_prompt,
    format_relationship_classification_prompt,
    CANONICAL_EXTRACTION_SCHEMA,
    RELATIONSHIP_CLASSIFICATION_SCHEMA
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMClassifier:
    """
    LLM-based classifier for canonical extraction and relationship classification.
    """

    def __init__(
        self,
        model: str = "qwen3:14b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: int = 60,
        max_retries: int = 3,
        strip_think_tags: bool = True,
        canonical_cache_path: Optional[str] = None,
        relationship_cache_path: Optional[str] = None
    ):
        """
        Initialize the LLM classifier.

        Args:
            model: Ollama model name
            base_url: Ollama server URL
            temperature: LLM temperature (0.0 for deterministic)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            strip_think_tags: Remove <think>...</think> tags from responses
            canonical_cache_path: Path to canonical extraction cache
            relationship_cache_path: Path to relationship classification cache
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.strip_think_tags = strip_think_tags

        # Caches
        self.canonical_cache_path = canonical_cache_path
        self.relationship_cache_path = relationship_cache_path
        self.canonical_cache: Dict[str, Dict] = {}
        self.relationship_cache: Dict[str, Dict] = {}

        # Load caches
        self._load_cache(self.canonical_cache_path, self.canonical_cache)
        self._load_cache(self.relationship_cache_path, self.relationship_cache)

        # Stats
        self.canonical_cache_hits = 0
        self.canonical_cache_misses = 0
        self.relationship_cache_hits = 0
        self.relationship_cache_misses = 0

        logger.info(f"LLMClassifier initialized with model: {model}")

    def _load_cache(self, cache_path: Optional[str], cache_dict: Dict):
        """Load cache from disk."""
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    loaded = pickle.load(f)
                    cache_dict.update(loaded)
                logger.info(f"Loaded {len(cache_dict)} items from {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")

    def _save_cache(self, cache_path: Optional[str], cache_dict: Dict):
        """Save cache to disk."""
        if not cache_path:
            return

        try:
            cache_dir = os.path.dirname(cache_path)
            os.makedirs(cache_dir, exist_ok=True)

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_dict, f)
            logger.debug(f"Saved {len(cache_dict)} items to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache {cache_path}: {e}")

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from LLM response."""
        if not self.strip_think_tags:
            return text

        # Remove think tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _call_llm(self, prompt: str) -> str:
        """
        Call Ollama LLM with retry logic.

        Args:
            prompt: Input prompt

        Returns:
            LLM response text
        """
        system_message = "Provide only the final JSON output without showing your reasoning process or using <think> tags. Start your response immediately with the { character."

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": system_message,
                        "temperature": self.temperature,
                        "stream": False
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()['response']
                result = self._strip_think_tags(result)

                return result

            except requests.exceptions.RequestException as e:
                logger.warning(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise

        raise RuntimeError(f"Failed to call LLM after {self.max_retries} attempts")

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response."""
        try:
            # Try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # Try to find any JSON object
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))

            raise ValueError(f"Could not parse JSON from response: {response}")

    def _normalize_intervention_name(self, name: str) -> str:
        """Normalize intervention name as fallback canonical."""
        # Lowercase
        normalized = name.lower()

        # Remove dosage patterns
        dosage_patterns = [
            r'\d+\s*(mg|g|mcg|µg|ug)',
            r'\d+\s*(IU|iu)',
            r'\d+\s*x?\s*10\^?\d+\s*(CFU|cfu)',
            r'\d+(\.\d+)?\s*(ml|mL)',
            r'\d+\s*(units?)',
        ]
        for pattern in dosage_patterns:
            normalized = re.sub(pattern, '', normalized)

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        return normalized

    def extract_canonical(self, intervention_name: str) -> Dict:
        """
        Extract canonical group for an intervention.

        Args:
            intervention_name: Original intervention name

        Returns:
            Dict with keys: canonical_group, reasoning, source
        """
        # Check cache
        if intervention_name in self.canonical_cache:
            self.canonical_cache_hits += 1
            return self.canonical_cache[intervention_name]

        self.canonical_cache_misses += 1

        try:
            # Generate prompt
            prompt = format_canonical_extraction_prompt(intervention_name)

            # Call LLM
            response = self._call_llm(prompt)

            # Parse response
            result = self._parse_json_response(response)

            # Validate response
            if 'canonical_group' not in result or not result['canonical_group']:
                raise ValueError("Missing or empty canonical_group in response")

            # Add metadata
            result['source'] = 'llm'
            result['model'] = self.model

            # Cache result
            self.canonical_cache[intervention_name] = result

            # Save cache periodically
            if len(self.canonical_cache) % 20 == 0:
                self._save_cache(self.canonical_cache_path, self.canonical_cache)

            return result

        except Exception as e:
            logger.error(f"Failed to extract canonical for '{intervention_name}': {e}")

            # Fallback to normalized name
            fallback_canonical = self._normalize_intervention_name(intervention_name)
            result = {
                'canonical_group': fallback_canonical,
                'reasoning': 'Fallback normalization (LLM failed)',
                'source': 'fallback',
                'model': None
            }

            # Cache fallback
            self.canonical_cache[intervention_name] = result
            return result

    def classify_relationship(
        self,
        intervention_1: str,
        intervention_2: str,
        similarity: float
    ) -> Dict:
        """
        Classify relationship between two interventions.

        Args:
            intervention_1: First intervention name
            intervention_2: Second intervention name
            similarity: Embedding similarity score (0.0-1.0)

        Returns:
            Dict with keys: relationship_type, layer_1_canonical, layer_2_same_variant, reasoning
        """
        # Create cache key (canonical ordering)
        cache_key = tuple(sorted([intervention_1, intervention_2])) + (round(similarity, 3),)

        # Check cache
        if cache_key in self.relationship_cache:
            self.relationship_cache_hits += 1
            return self.relationship_cache[cache_key]

        self.relationship_cache_misses += 1

        # Auto-classify based on similarity thresholds
        if similarity >= 0.95:
            # High similarity: likely EXACT_MATCH or DOSAGE_VARIANT
            # Let LLM decide
            pass
        elif similarity < 0.70:
            # Very low similarity: auto-classify as DIFFERENT
            result = {
                'relationship_type': 'DIFFERENT',
                'layer_1_canonical': None,
                'layer_2_same_variant': False,
                'reasoning': f'Low similarity ({similarity:.3f}) indicates unrelated interventions',
                'source': 'auto_threshold',
                'similarity': similarity
            }
            self.relationship_cache[cache_key] = result
            return result

        # LLM classification for intermediate similarity
        try:
            # Generate prompt
            prompt = format_relationship_classification_prompt(
                intervention_1,
                intervention_2,
                similarity
            )

            # Call LLM
            response = self._call_llm(prompt)

            # Parse response
            result = self._parse_json_response(response)

            # Validate response
            required_fields = ['relationship_type', 'layer_1_canonical', 'layer_2_same_variant', 'reasoning']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Validate relationship type
            valid_types = ['EXACT_MATCH', 'VARIANT', 'SUBTYPE', 'SAME_CATEGORY', 'DOSAGE_VARIANT', 'DIFFERENT']
            if result['relationship_type'] not in valid_types:
                raise ValueError(f"Invalid relationship type: {result['relationship_type']}")

            # Add metadata
            result['source'] = 'llm'
            result['model'] = self.model
            result['similarity'] = similarity

            # Cache result
            self.relationship_cache[cache_key] = result

            # Save cache periodically
            if len(self.relationship_cache) % 20 == 0:
                self._save_cache(self.relationship_cache_path, self.relationship_cache)

            return result

        except Exception as e:
            logger.error(f"Failed to classify relationship for '{intervention_1}' vs '{intervention_2}': {e}")

            # Fallback based on similarity ranges
            if similarity >= 0.85:
                rel_type = 'VARIANT'
            elif similarity >= 0.75:
                rel_type = 'SUBTYPE'
            elif similarity >= 0.70:
                rel_type = 'SAME_CATEGORY'
            else:
                rel_type = 'DIFFERENT'

            result = {
                'relationship_type': rel_type,
                'layer_1_canonical': None,
                'layer_2_same_variant': False,
                'reasoning': f'Fallback classification based on similarity ({similarity:.3f})',
                'source': 'fallback',
                'model': None,
                'similarity': similarity
            }

            # Cache fallback
            self.relationship_cache[cache_key] = result
            return result

    def get_stats(self) -> Dict:
        """Get classifier statistics."""
        canonical_total = self.canonical_cache_hits + self.canonical_cache_misses
        relationship_total = self.relationship_cache_hits + self.relationship_cache_misses

        return {
            'canonical_cache_size': len(self.canonical_cache),
            'canonical_cache_hits': self.canonical_cache_hits,
            'canonical_cache_misses': self.canonical_cache_misses,
            'canonical_hit_rate': self.canonical_cache_hits / canonical_total if canonical_total > 0 else 0.0,
            'relationship_cache_size': len(self.relationship_cache),
            'relationship_cache_hits': self.relationship_cache_hits,
            'relationship_cache_misses': self.relationship_cache_misses,
            'relationship_hit_rate': self.relationship_cache_hits / relationship_total if relationship_total > 0 else 0.0,
        }

    def save_caches_now(self):
        """Force save all caches to disk."""
        self._save_cache(self.canonical_cache_path, self.canonical_cache)
        self._save_cache(self.relationship_cache_path, self.relationship_cache)

    def __del__(self):
        """Cleanup: save caches on destruction."""
        if hasattr(self, 'canonical_cache'):
            self._save_cache(self.canonical_cache_path, self.canonical_cache)
        if hasattr(self, 'relationship_cache'):
            self._save_cache(self.relationship_cache_path, self.relationship_cache)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_llm_classifier(config_path: Optional[str] = None) -> LLMClassifier:
    """
    Load LLM classifier with configuration.

    Args:
        config_path: Path to YAML config file (optional)

    Returns:
        Configured LLMClassifier instance
    """
    # Default configuration
    default_config = {
        'model': 'qwen3:14b',
        'base_url': 'http://localhost:11434',
        'temperature': 0.0,
        'timeout': 60,
        'max_retries': 3,
        'strip_think_tags': True,
        'canonical_cache_path': 'c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/canonicals.pkl',
        'relationship_cache_path': 'c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/llm_decisions.pkl'
    }

    # Load config from file if provided
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            llm_config = config.get('llm', {})
            cache_config = config.get('cache', {})

            default_config['model'] = llm_config.get('model', default_config['model'])
            default_config['base_url'] = llm_config.get('base_url', default_config['base_url'])
            default_config['temperature'] = llm_config.get('temperature', default_config['temperature'])
            default_config['timeout'] = llm_config.get('timeout', default_config['timeout'])
            default_config['max_retries'] = llm_config.get('max_retries', default_config['max_retries'])
            default_config['strip_think_tags'] = llm_config.get('strip_think_tags', default_config['strip_think_tags'])
            default_config['canonical_cache_path'] = cache_config.get('canonical_cache_path', default_config['canonical_cache_path'])
            default_config['relationship_cache_path'] = cache_config.get('llm_cache_path', default_config['relationship_cache_path'])

    return LLMClassifier(**default_config)


if __name__ == "__main__":
    # Test the LLM classifier
    print("Testing LLMClassifier...")

    classifier = LLMClassifier(
        canonical_cache_path="c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/canonicals_test.pkl",
        relationship_cache_path="c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/relationships_test.pkl"
    )

    # Test canonical extraction
    print("\n=== Canonical Extraction Test ===")
    interventions = [
        "Lactobacillus reuteri DSM 17938",
        "Saccharomyces boulardii",
        "atorvastatin 20mg",
        "Cetuximab-β"
    ]

    for intervention in interventions:
        result = classifier.extract_canonical(intervention)
        print(f"\n{intervention}")
        print(f"  Canonical: {result['canonical_group']}")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Source: {result['source']}")

    # Test relationship classification
    print("\n=== Relationship Classification Test ===")
    pairs = [
        ("vitamin D", "cholecalciferol", 0.92),
        ("Lactobacillus reuteri", "Saccharomyces boulardii", 0.72),
        ("Cetuximab", "Cetuximab-β", 0.88),
        ("vitamin D", "chemotherapy", 0.15)
    ]

    for int1, int2, sim in pairs:
        result = classifier.classify_relationship(int1, int2, sim)
        print(f"\n{int1} vs {int2} (sim={sim:.2f})")
        print(f"  Type: {result['relationship_type']}")
        print(f"  Canonical: {result['layer_1_canonical']}")
        print(f"  Same variant: {result['layer_2_same_variant']}")
        print(f"  Reasoning: {result['reasoning']}")

    # Print stats
    stats = classifier.get_stats()
    print(f"\nClassifier stats: {stats}")
