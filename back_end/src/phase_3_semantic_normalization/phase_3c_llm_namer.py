"""
LLM Namer - Canonical naming using LLM (qwen3:14b) with temperature experimentation

Generates meaningful canonical names for clusters by analyzing member entities.
Supports temperature experimentation (0.0, 0.2, 0.3, 0.4) for quality/creativity trade-off.
"""

import json
import logging
import re
import time
import requests
from typing import Dict, List, Optional

from .phase_3c_base_namer import BaseNamer, ClusterData, NamingResult

logger = logging.getLogger(__name__)


class LLMNamer(BaseNamer):
    """
    LLM-based canonical namer using qwen3:14b via Ollama.

    Features:
    - Entity-type-specific prompts (interventions, conditions, mechanisms)
    - Temperature control (0.0=deterministic, 0.4=creative)
    - Batch processing (20 clusters per call)
    - JSON parsing with fallback
    - Retry logic (3 attempts)
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
        cache_path: Optional[str] = None,
        allowed_categories: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize LLM namer.

        Args:
            model: LLM model name (default: qwen3:14b)
            base_url: Ollama API URL
            temperature: LLM temperature (0.0-1.0)
            max_tokens: Maximum response tokens
            timeout: API request timeout in seconds
            max_retries: Maximum retry attempts
            strip_think_tags: Remove <think> tags from qwen3 output
            max_members_shown: Maximum members to include in prompt
            include_frequency: Show paper frequency in prompt
            cache_path: Path to cache naming results
            allowed_categories: Dict of allowed categories per entity type
        """
        super().__init__(f'llm_{model}', temperature, cache_path)

        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.strip_think_tags = strip_think_tags
        self.max_members_shown = max_members_shown
        self.include_frequency = include_frequency

        # Default allowed categories
        self.allowed_categories = allowed_categories or {
            'intervention': [
                'exercise', 'diet', 'supplement', 'medication', 'therapy',
                'lifestyle', 'surgery', 'test', 'device', 'procedure',
                'biologics', 'gene_therapy', 'emerging'
            ],
            'condition': [
                'cardiac', 'neurological', 'digestive', 'pulmonary', 'endocrine',
                'renal', 'oncological', 'rheumatological', 'psychiatric',
                'musculoskeletal', 'dermatological', 'infectious', 'immunological',
                'hematological', 'nutritional', 'toxicological', 'parasitic', 'other'
            ],
            'mechanism': []  # No fixed categories for mechanisms
        }

        logger.info(f"LLMNamer initialized: model={model}, temperature={temperature}")

    def _generate_names_batch(self, clusters: List[ClusterData]) -> List[NamingResult]:
        """
        Generate canonical names for a batch of clusters using LLM.

        Args:
            clusters: List of ClusterData objects

        Returns:
            List[NamingResult]: Naming results for each cluster
        """
        if not clusters:
            return []

        # Group by entity type (different prompts for each)
        entity_type = clusters[0].entity_type
        if not all(c.entity_type == entity_type for c in clusters):
            raise ValueError("All clusters in batch must have same entity_type")

        # Build prompt
        prompt = self._build_prompt(clusters, entity_type)

        # Call LLM with retry logic
        system_message = "Provide only the final JSON output without showing your reasoning process or using <think> tags. Start your response immediately with the [ character."

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

                response_text = response.json()['response'].strip()

                # Strip think tags if present
                if self.strip_think_tags:
                    response_text = self._strip_think_tags(response_text)

                # Parse JSON response
                naming_data = self._parse_response(response_text)

                # Map to NamingResult objects
                results = self._map_to_results(clusters, naming_data, entity_type)

                logger.info(f"Successfully named {len(results)} clusters (attempt {attempt + 1})")
                return results

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Naming failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Naming failed after {self.max_retries} attempts: {e}")
                    raise

    def _build_prompt(self, clusters: List[ClusterData], entity_type: str) -> str:
        """
        Build entity-type-specific prompt for naming clusters.

        Args:
            clusters: List of ClusterData objects
            entity_type: 'intervention', 'condition', or 'mechanism'

        Returns:
            str: Formatted prompt
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
        """Build prompt for intervention naming."""
        categories = self.allowed_categories['intervention']
        category_desc = ", ".join(categories)

        cluster_list = []
        for i, cluster in enumerate(clusters):
            members = self._format_members(cluster)
            cluster_list.append(f"{i+1}. Cluster {cluster.cluster_id}:\n   {members}")

        clusters_text = "\n\n".join(cluster_list)

        prompt = f"""Analyze these intervention name variants and provide:
1. Canonical name (most common/clear form)
2. Category (one of: {category_desc})

CLUSTERS TO NAME:
{clusters_text}

Return ONLY JSON array:
[
    {{"number": 1, "canonical_name": "vitamin D", "category": "supplement"}},
    {{"number": 2, "canonical_name": "metformin", "category": "medication"}},
    ...
]

No explanations. Just the JSON array."""

        return prompt

    def _build_condition_prompt(self, clusters: List[ClusterData]) -> str:
        """Build prompt for condition naming."""
        categories = self.allowed_categories['condition']
        category_desc = ", ".join(categories)

        cluster_list = []
        for i, cluster in enumerate(clusters):
            members = self._format_members(cluster)
            cluster_list.append(f"{i+1}. Cluster {cluster.cluster_id}:\n   {members}")

        clusters_text = "\n\n".join(cluster_list)

        prompt = f"""Analyze these medical condition name variants and provide:
1. Canonical name (standard medical term)
2. Category (one of: {category_desc})
3. Parent condition (if this is a subtype, otherwise null)

CLUSTERS TO NAME:
{clusters_text}

Return ONLY JSON array:
[
    {{"number": 1, "canonical_name": "irritable bowel syndrome", "category": "digestive", "parent": null}},
    {{"number": 2, "canonical_name": "IBS-C", "category": "digestive", "parent": "irritable bowel syndrome"}},
    ...
]

No explanations. Just the JSON array."""

        return prompt

    def _build_mechanism_prompt(self, clusters: List[ClusterData]) -> str:
        """Build prompt for mechanism naming."""
        cluster_list = []
        for i, cluster in enumerate(clusters):
            members = self._format_members(cluster, max_chars=200)  # Truncate long mechanisms
            cluster_list.append(f"{i+1}. Cluster {cluster.cluster_id}:\n   {members}")

        clusters_text = "\n\n".join(cluster_list)

        prompt = f"""Analyze these mechanism descriptions and extract:
1. Canonical mechanism name (concise biological pathway/action, max 10 words)
2. Mechanism category (e.g., anti-inflammatory, neurotransmitter modulation, metabolic regulation)

CLUSTERS TO NAME:
{clusters_text}

Return ONLY JSON array:
[
    {{"number": 1, "canonical_name": "TNF-alpha pathway inhibition", "category": "anti-inflammatory"}},
    {{"number": 2, "canonical_name": "serotonin reuptake inhibition", "category": "neurotransmitter modulation"}},
    ...
]

No explanations. Just the JSON array."""

        return prompt

    def _format_members(self, cluster: ClusterData, max_chars: Optional[int] = None) -> str:
        """
        Format cluster members for prompt.

        Args:
            cluster: ClusterData object
            max_chars: Maximum characters per member (for truncation)

        Returns:
            str: Formatted member list
        """
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
        """
        Parse LLM JSON response.

        Args:
            response_text: Raw LLM response

        Returns:
            List[Dict]: Parsed naming data

        Raises:
            ValueError: If JSON parsing fails
        """
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
        entity_type: str
    ) -> List[NamingResult]:
        """
        Map LLM response to NamingResult objects.

        Args:
            clusters: Original cluster data
            naming_data: Parsed JSON from LLM
            entity_type: Entity type

        Returns:
            List[NamingResult]
        """
        results = []

        for data in naming_data:
            number = data.get("number")
            canonical_name = data.get("canonical_name")
            category = data.get("category")
            parent = data.get("parent")

            if not number or not canonical_name:
                logger.warning(f"Invalid naming data: {data}")
                continue

            if not (1 <= number <= len(clusters)):
                logger.warning(f"Invalid cluster number: {number}")
                continue

            cluster = clusters[number - 1]

            # Validate category
            if entity_type != 'mechanism' and category:
                allowed = self.allowed_categories.get(entity_type, [])
                if category.lower() not in [c.lower() for c in allowed]:
                    logger.warning(f"Invalid category '{category}' for {entity_type}, setting to None")
                    category = None

            result = NamingResult(
                cluster_id=cluster.cluster_id,
                canonical_name=canonical_name,
                category=category,
                parent_cluster=parent,
                confidence=1.0,
                provenance={
                    'method': 'llm',
                    'model': self.model,
                    'temperature': self.temperature,
                    'members_shown': len(cluster.member_entities[:self.max_members_shown])
                }
            )

            results.append(result)

        return results
