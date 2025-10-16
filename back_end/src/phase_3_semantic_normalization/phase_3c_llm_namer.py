"""
Dynamic LLM Namer - Experimental Phase 3c with Dynamic Category Discovery

Allows LLM to discover categories dynamically with minimal constraints (4 examples only).
Tracks discovered categories with confidence scores for later consolidation.
"""

import json
import logging
import re
import time
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CategoryInfo:
    """Information about a discovered category."""
    category_name: str
    usage_count: int = 0
    confidences: List[str] = field(default_factory=list)
    example_clusters: List[str] = field(default_factory=list)
    is_novel: bool = False  # Not in original taxonomy


@dataclass
class ClusterData:
    """Cluster data for naming."""
    cluster_id: int
    entity_type: str
    member_entities: List[str]
    member_frequencies: Optional[List[int]] = None
    singleton: bool = False


@dataclass
class NamingResult:
    """Result of naming a single cluster."""
    cluster_id: int
    canonical_name: str
    category: str
    reasoning: str
    confidence: str  # HIGH, MEDIUM, LOW
    parent_cluster: Optional[str] = None
    raw_response: Optional[str] = None
    provenance: Optional[Dict] = None  # Tracking metadata


class LLMNamer:
    """
    LLM-based namer with dynamic category discovery.

    Features:
    - Minimal example categories (4 only)
    - Allows LLM to create new categories
    - Tracks discovered categories with usage stats
    - Enforces category naming standards
    - Saves raw responses for analysis
    """

    def __init__(
        self,
        model: str = "qwen3:14b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        timeout: int = 60,
        max_retries: int = 3,
        strip_think_tags: bool = True,
        max_members_shown: int = 10,
        include_frequency: bool = True,
        cache_path: Optional[str] = None,  # For compatibility (not used yet)
        example_categories: Optional[Dict[str, List[str]]] = None,
        forbidden_terms: Optional[List[str]] = None
    ):
        """
        Initialize dynamic LLM namer.

        Args:
            model: LLM model name
            base_url: Ollama API URL
            temperature: LLM temperature (0.0-1.0)
            max_tokens: Maximum response tokens
            timeout: API request timeout
            max_retries: Maximum retry attempts
            strip_think_tags: Remove <think> tags from qwen3
            max_members_shown: Maximum members in prompt
            include_frequency: Show paper frequency
            cache_path: Cache path (for compatibility, not used yet)
            example_categories: Dict of example categories per entity type
            forbidden_terms: Generic terms to reject
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout  # None = no timeout, respects config
        self.max_retries = max_retries
        self.strip_think_tags = strip_think_tags
        self.max_members_shown = max_members_shown
        self.include_frequency = include_frequency
        self.cache_path = cache_path  # For compatibility

        # Example categories (minimal - not exhaustive)
        self.example_categories = example_categories or {
            'intervention': ['supplement', 'exercise', 'medication', 'gene_therapy'],
            'condition': ['cardiac', 'neurological', 'endocrine', 'oncological'],
            'mechanism': []
        }

        # Forbidden generic terms
        self.forbidden_terms = forbidden_terms or [
            'treatment', 'intervention', 'therapy_general', 'supplements_general'
        ]

        # Track discovered categories
        self.discovered_categories: Dict[str, CategoryInfo] = {}

        # Caching
        self.cache: Dict[str, Dict] = {}  # cache_key -> naming result dict
        self.cache_hits = 0
        self.cache_misses = 0
        if self.cache_path:
            self._load_cache()

        # Statistics
        self.stats = {
            'names_generated': 0,
            'failures': 0,
            'categories_discovered': 0,
            'novel_categories': 0,
            'hit_rate': 0.0  # For compatibility
        }

        logger.info(f"LLMNamer initialized: model={model}, temperature={temperature}")
        if self.cache_path and self.cache:
            logger.info(f"Loaded {len(self.cache)} cached naming results from {self.cache_path}")

    def name_clusters(
        self,
        clusters: List[ClusterData],
        batch_size: int = 10
    ) -> List[NamingResult]:
        """
        Name all clusters using LLM with dynamic category discovery.

        Args:
            clusters: List of ClusterData objects
            batch_size: Clusters per LLM call

        Returns:
            List of NamingResult objects
        """
        all_results = []

        # Process in batches
        for i in range(0, len(clusters), batch_size):
            batch = clusters[i:i + batch_size]
            logger.info(f"Naming batch {i//batch_size + 1}/{(len(clusters) + batch_size - 1)//batch_size} ({len(batch)} clusters)")

            try:
                results = self._generate_names_batch(batch)
                all_results.extend(results)

                # Update statistics
                self.stats['names_generated'] += len(results)

            except Exception as e:
                logger.error(f"Batch {i//batch_size + 1} failed: {e}")
                self.stats['failures'] += len(batch)

        # Update category statistics
        self.stats['categories_discovered'] = len(self.discovered_categories)
        self.stats['novel_categories'] = sum(
            1 for cat in self.discovered_categories.values() if cat.is_novel
        )

        # Calculate cache hit rate
        total_requests = self.cache_hits + self.cache_misses
        self.stats['hit_rate'] = self.cache_hits / total_requests if total_requests > 0 else 0.0

        logger.info(f"Naming complete: {self.stats['names_generated']} named, {self.stats['failures']} failed")
        logger.info(f"Cache: {self.cache_hits} hits, {self.cache_misses} misses (hit rate: {self.stats['hit_rate']:.1%})")
        logger.info(f"Categories discovered: {self.stats['categories_discovered']} ({self.stats['novel_categories']} novel)")

        return all_results

    def _generate_names_batch(self, clusters: List[ClusterData]) -> List[NamingResult]:
        """
        Generate names for a batch of clusters using LLM (with caching).

        Args:
            clusters: List of ClusterData objects

        Returns:
            List of NamingResult objects
        """
        if not clusters:
            return []

        # Group by entity type
        entity_type = clusters[0].entity_type
        if not all(c.entity_type == entity_type for c in clusters):
            raise ValueError("All clusters in batch must have same entity_type")

        # Check cache for each cluster
        cached_results = []
        uncached_clusters = []

        for cluster in clusters:
            if self.cache_path:
                cache_key = self._get_cache_key(cluster)
                if cache_key in self.cache:
                    # Cache hit - reconstruct NamingResult from cached dict
                    cached_data = self.cache[cache_key]
                    result = NamingResult(
                        cluster_id=cluster.cluster_id,
                        canonical_name=cached_data['canonical_name'],
                        category=cached_data['category'],
                        reasoning=cached_data['reasoning'],
                        confidence=cached_data['confidence'],
                        parent_cluster=cached_data.get('parent_cluster'),
                        raw_response=cached_data.get('raw_response'),
                        provenance=cached_data.get('provenance')
                    )
                    cached_results.append(result)
                    self.cache_hits += 1

                    # Track category from cache
                    if result.category and result.category != "unknown":
                        self._track_category(result.category, entity_type, result.canonical_name, result.confidence)
                else:
                    uncached_clusters.append(cluster)
                    self.cache_misses += 1
            else:
                uncached_clusters.append(cluster)

        # If all clusters were cached, return immediately
        if not uncached_clusters:
            logger.info(f"All {len(clusters)} clusters found in cache")
            return cached_results

        # Generate names for uncached clusters
        logger.info(f"Naming {len(uncached_clusters)} uncached clusters ({len(cached_results)} from cache)")

        # Build prompt for uncached clusters only
        prompt = self._build_prompt(uncached_clusters, entity_type)

        # Call LLM with retry logic
        system_message = "Provide only the final JSON output without showing your reasoning process or using <think> tags. Start your response immediately with the [ character."

        for attempt in range(self.max_retries):
            try:
                # Use very large timeout if None (requests doesn't accept None)
                timeout_value = self.timeout if self.timeout is not None else 3600  # 1 hour max

                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": system_message,
                        "temperature": self.temperature,
                        "stream": False
                    },
                    timeout=timeout_value
                )
                response.raise_for_status()

                response_text = response.json()['response'].strip()

                # Strip think tags if present
                if self.strip_think_tags:
                    response_text = self._strip_think_tags(response_text)

                # Parse JSON response
                naming_data = self._parse_response(response_text)

                # Map to NamingResult objects (uncached clusters only)
                new_results = self._map_to_results(uncached_clusters, naming_data, entity_type, response_text)

                # Cache newly generated results
                if self.cache_path:
                    for i, result in enumerate(new_results):
                        cluster = uncached_clusters[i]
                        cache_key = self._get_cache_key(cluster)

                        # Convert NamingResult to dict for caching
                        self.cache[cache_key] = {
                            'canonical_name': result.canonical_name,
                            'category': result.category,
                            'reasoning': result.reasoning,
                            'confidence': result.confidence,
                            'parent_cluster': result.parent_cluster,
                            'raw_response': result.raw_response,
                            'provenance': result.provenance
                        }

                    # Save cache to disk
                    self._save_cache()

                # Combine cached and new results
                all_batch_results = cached_results + new_results

                logger.info(f"Successfully named {len(new_results)} clusters ({len(cached_results)} from cache, attempt {attempt + 1})")
                return all_batch_results

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning(f"Naming failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Naming failed after {self.max_retries} attempts: {e}")
                    raise

    def _build_prompt(self, clusters: List[ClusterData], entity_type: str) -> str:
        """
        Build entity-type-specific prompt with dynamic category discovery.

        Args:
            clusters: List of ClusterData objects
            entity_type: 'intervention', 'condition', or 'mechanism'

        Returns:
            Formatted prompt string
        """
        if entity_type == 'intervention':
            return self._build_intervention_prompt(clusters)
        elif entity_type == 'condition':
            return self._build_condition_prompt(clusters)
        elif entity_type == 'mechanism':
            return self._build_mechanism_prompt(clusters)
        else:
            raise ValueError(f"Unknown entity_type: {entity_type}")

    def _build_intervention_prompt(self, clusters: List[ClusterData]) -> str:
        """Build prompt for intervention naming with dynamic categories."""
        examples = self.example_categories['intervention']

        cluster_list = []
        for i, cluster in enumerate(clusters):
            members = self._format_members(cluster)
            cluster_list.append(f"{i+1}. {members}")

        clusters_text = "\n\n".join(cluster_list)

        prompt = f"""You are analyzing health intervention clusters to assign categories.

CATEGORY GUIDELINES:
- Use established categories when appropriate
- Create NEW categories for interventions that don't fit
- Categories should describe intervention TYPE (what it IS), not effect
- Format: lowercase with underscores (e.g., "gene_therapy")

EXAMPLE ESTABLISHED CATEGORIES:
- supplement: Nutritional supplements (vitamin D, omega-3, probiotics)
- exercise: Physical activity (running, resistance training, yoga)
- medication: Pharmaceutical drugs (metformin, statins, antibiotics)
- gene_therapy: Genetic interventions (CRISPR, CAR-T, stem cells)

CLUSTERS TO CATEGORIZE:
{clusters_text}

For each cluster, provide:
1. canonical_name: Most common/clear name (DO NOT include cluster numbers or prefixes like "Cluster X:")
2. category: Choose existing OR create new (lowercase_with_underscores)
3. reasoning: Brief explanation (1 sentence)
4. confidence: HIGH/MEDIUM/LOW

Return ONLY JSON array:
[
    {{"number": 1, "canonical_name": "vitamin D", "category": "supplement", "reasoning": "Nutritional supplement", "confidence": "HIGH"}},
    {{"number": 2, "canonical_name": "metformin", "category": "medication", "reasoning": "Pharmaceutical drug", "confidence": "HIGH"}},
    ...
]

No explanations. Just the JSON array."""

        return prompt

    def _build_condition_prompt(self, clusters: List[ClusterData]) -> str:
        """Build prompt for condition naming with dynamic categories."""
        examples = self.example_categories['condition']

        cluster_list = []
        for i, cluster in enumerate(clusters):
            members = self._format_members(cluster)
            cluster_list.append(f"{i+1}. {members}")

        clusters_text = "\n\n".join(cluster_list)

        prompt = f"""You are analyzing medical condition clusters to assign categories.

CATEGORY GUIDELINES:
- Use established medical specialties when appropriate
- Create NEW categories for novel or multisystem conditions
- Categories should reflect PRIMARY organ system or disease type
- Format: lowercase with underscores

EXAMPLE ESTABLISHED CATEGORIES:
- cardiac: Heart and blood vessels (heart failure, hypertension)
- neurological: Brain and nervous system (Alzheimer's, Parkinson's, stroke)
- endocrine: Hormones and metabolism (diabetes, thyroid disorders, PCOS)
- oncological: Cancers and malignancies (lung cancer, breast cancer, leukemia)

CLUSTERS TO CATEGORIZE:
{clusters_text}

For each cluster, provide:
1. canonical_name: Standard medical term (DO NOT include cluster numbers or prefixes like "Cluster X:")
2. category: Choose existing OR create new (lowercase_with_underscores)
3. reasoning: Brief explanation (1 sentence)
4. confidence: HIGH/MEDIUM/LOW
5. parent_condition: If this is a subtype, name the parent (otherwise null)

Return ONLY JSON array:
[
    {{"number": 1, "canonical_name": "type 2 diabetes", "category": "endocrine", "reasoning": "Metabolic disorder", "confidence": "HIGH", "parent_condition": null}},
    {{"number": 2, "canonical_name": "IBS-C", "category": "digestive", "reasoning": "Gastrointestinal disorder", "confidence": "HIGH", "parent_condition": "irritable bowel syndrome"}},
    ...
]

No explanations. Just the JSON array."""

        return prompt

    def _build_mechanism_prompt(self, clusters: List[ClusterData]) -> str:
        """Build prompt for mechanism naming (no fixed categories)."""
        cluster_list = []
        for i, cluster in enumerate(clusters):
            members = self._format_members(cluster, max_chars=200)
            cluster_list.append(f"{i+1}. {members}")

        clusters_text = "\n\n".join(cluster_list)

        prompt = f"""Analyze these mechanism descriptions and extract:
1. Canonical mechanism name (concise biological pathway/action, max 10 words - DO NOT include cluster numbers or prefixes like "Cluster X:")
2. Mechanism category (e.g., anti-inflammatory, neurotransmitter modulation, metabolic regulation)

CLUSTERS TO NAME:
{clusters_text}

Return ONLY JSON array:
[
    {{"number": 1, "canonical_name": "TNF-alpha pathway inhibition", "category": "anti-inflammatory", "reasoning": "Reduces inflammation", "confidence": "HIGH"}},
    {{"number": 2, "canonical_name": "serotonin reuptake inhibition", "category": "neurotransmitter_modulation", "reasoning": "Modulates serotonin", "confidence": "HIGH"}},
    ...
]

No explanations. Just the JSON array."""

        return prompt

    def _format_members(self, cluster: ClusterData, max_chars: Optional[int] = None) -> str:
        """Format cluster members for prompt."""
        members = cluster.member_entities[:self.max_members_shown]

        if self.include_frequency and cluster.member_frequencies:
            frequencies = cluster.member_frequencies[:self.max_members_shown]
            formatted = []
            for member, freq in zip(members, frequencies):
                if max_chars and len(member) > max_chars:
                    member = member[:max_chars] + "..."
                formatted.append(f"{member} ({freq} papers)")
        else:
            formatted = []
            for member in members:
                if max_chars and len(member) > max_chars:
                    member = member[:max_chars] + "..."
                formatted.append(member)

        members_text = ", ".join(formatted)

        if len(cluster.member_entities) > self.max_members_shown:
            members_text += f" (+ {len(cluster.member_entities) - self.max_members_shown} more)"

        return members_text

    def _strip_think_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from qwen3 output."""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _parse_response(self, response_text: str) -> List[Dict]:
        """Parse LLM JSON response."""
        # Remove markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        response_text = response_text.strip()

        try:
            results = json.loads(response_text)
            if not isinstance(results, list):
                raise ValueError("Response is not a JSON array")
            return results
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response_text[:500]}")
            raise ValueError(f"JSON parsing failed: {e}")

    def _map_to_results(
        self,
        clusters: List[ClusterData],
        naming_data: List[Dict],
        entity_type: str,
        raw_response: str
    ) -> List[NamingResult]:
        """
        Map LLM response to NamingResult objects and track discovered categories.

        Args:
            clusters: Original cluster data
            naming_data: Parsed JSON from LLM
            entity_type: Entity type
            raw_response: Raw LLM response text

        Returns:
            List of NamingResult objects
        """
        results = []

        for data in naming_data:
            number = data.get("number")
            canonical_name = data.get("canonical_name")
            category = data.get("category")
            reasoning = data.get("reasoning", "")
            confidence = data.get("confidence", "MEDIUM")
            parent = data.get("parent_condition") or data.get("parent")

            if not number or not canonical_name:
                logger.warning(f"Invalid naming data: {data}")
                continue

            if not (1 <= number <= len(clusters)):
                logger.warning(f"Invalid cluster number: {number}")
                continue

            cluster = clusters[number - 1]

            # Validate and normalize category
            if category:
                category = self._validate_category(category, entity_type, canonical_name)
                if category:
                    # Track discovered category
                    self._track_category(category, entity_type, canonical_name, confidence)

            result = NamingResult(
                cluster_id=cluster.cluster_id,
                canonical_name=canonical_name,
                category=category or "unknown",
                reasoning=reasoning,
                confidence=confidence.upper() if confidence else "MEDIUM",
                parent_cluster=parent,
                raw_response=raw_response,
                provenance={
                    'method': 'dynamic_llm',
                    'model': self.model,
                    'temperature': self.temperature
                }
            )

            results.append(result)

        return results

    def _validate_category(self, category: str, entity_type: str, cluster_name: str) -> Optional[str]:
        """
        Validate category name against standards.

        Args:
            category: Category name from LLM
            entity_type: Entity type
            cluster_name: Canonical name of cluster (for logging)

        Returns:
            Normalized category name or None if invalid
        """
        # Normalize to lowercase with underscores
        category = category.lower().strip().replace(" ", "_").replace("-", "_")

        # Check minimum length
        if len(category) < 3:
            logger.warning(f"Category too short: '{category}' for cluster '{cluster_name}'")
            return None

        # Check maximum length
        if len(category) > 50:
            logger.warning(f"Category too long: '{category}' for cluster '{cluster_name}'")
            return None

        # Check forbidden terms
        if category in self.forbidden_terms:
            logger.warning(f"Forbidden category: '{category}' for cluster '{cluster_name}'")
            return None

        # Check format (must be alphanumeric + underscores)
        if not re.match(r'^[a-z0-9_]+$', category):
            logger.warning(f"Invalid category format: '{category}' for cluster '{cluster_name}'")
            return None

        return category

    def _track_category(self, category: str, entity_type: str, example_cluster: str, confidence: str):
        """
        Track discovered category with usage statistics.

        Args:
            category: Category name
            entity_type: Entity type
            example_cluster: Example cluster name
            confidence: Confidence level
        """
        if category not in self.discovered_categories:
            # Check if novel (not in examples)
            is_novel = category not in self.example_categories.get(entity_type, [])

            self.discovered_categories[category] = CategoryInfo(
                category_name=category,
                usage_count=0,
                confidences=[],
                example_clusters=[],
                is_novel=is_novel
            )

        # Update statistics
        cat_info = self.discovered_categories[category]
        cat_info.usage_count += 1
        cat_info.confidences.append(confidence)
        if len(cat_info.example_clusters) < 5:  # Keep first 5 examples
            cat_info.example_clusters.append(example_cluster)

    def get_discovered_categories(self) -> Dict[str, CategoryInfo]:
        """Get all discovered categories with statistics."""
        return self.discovered_categories

    def get_stats(self) -> Dict:
        """Get naming statistics."""
        return self.stats.copy()

    def _get_cache_key(self, cluster: ClusterData) -> str:
        """
        Generate cache key for a cluster.

        Cache key based on cluster members + entity_type + temperature.
        If members change, cache key changes → cache miss → LLM runs again.

        Args:
            cluster: ClusterData object

        Returns:
            MD5 hash string
        """
        # Sort members for consistent hashing
        members_str = "|".join(sorted(cluster.member_entities))
        cache_string = f"{members_str}|{cluster.entity_type}|{self.temperature}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _load_cache(self):
        """Load naming cache from JSON file."""
        if not self.cache_path:
            return

        cache_file = Path(self.cache_path)
        if not cache_file.exists():
            logger.debug(f"No cache file found at {self.cache_path}")
            return

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
            logger.info(f"Loaded {len(self.cache)} cached results from {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache from {self.cache_path}: {e}")
            self.cache = {}

    def _save_cache(self):
        """Save naming cache to JSON file."""
        if not self.cache_path:
            return

        try:
            cache_file = Path(self.cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(self.cache)} cached results to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache to {self.cache_path}: {e}")
