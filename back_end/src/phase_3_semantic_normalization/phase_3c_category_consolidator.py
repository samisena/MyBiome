"""
Category Consolidator - Stage 3c.2

Consolidates synonymous or redundant categories discovered in Phase 3c.1.
Uses LLM to identify and merge similar categories while preserving semantic distinctions.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

from back_end.src.data.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class CategoryInfo:
    """Information about a discovered category."""
    category_name: str
    usage_count: int
    confidences: List[str]
    example_clusters: List[str]
    is_novel: bool


@dataclass
class ConsolidationMapping:
    """Mapping of old category name to new consolidated category."""
    old_category: str
    new_category: str
    reason: str
    affected_clusters: int


@dataclass
class ConsolidationResult:
    """Result of category consolidation."""
    original_count: int
    consolidated_count: int
    reduction_pct: float
    mappings: List[ConsolidationMapping]
    preserved_categories: List[str]


class CategoryConsolidator:
    """
    Consolidates synonymous categories using LLM.

    Features:
    - LLM-based synonym detection
    - Validation to prevent over-merging
    - Preservation of high-usage categories
    - Reporting of consolidation decisions
    """

    def __init__(
        self,
        model: str = "qwen3:14b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        similarity_threshold: float = 0.85,
        max_consolidation_ratio: float = 0.5,
        min_group_size: int = 2,
        timeout: int = 60,
        ollama_client: Optional[OllamaClient] = None
    ):
        """
        Initialize category consolidator.

        Args:
            model: LLM model name
            base_url: Ollama API URL (ignored if ollama_client provided)
            temperature: LLM temperature (keep low for consistency)
            similarity_threshold: Threshold for considering categories similar
            max_consolidation_ratio: Maximum % of categories to consolidate
            min_group_size: Minimum categories required to form consolidation group
            timeout: API request timeout
            ollama_client: Optional pre-configured OllamaClient instance
        """
        self.similarity_threshold = similarity_threshold
        self.max_consolidation_ratio = max_consolidation_ratio
        self.min_group_size = min_group_size

        # Use provided client or create new one
        self.llm_client = ollama_client or OllamaClient(
            model=model,
            temperature=temperature,
            api_url=base_url,
            timeout=timeout
        )

        logger.info(f"CategoryConsolidator initialized: threshold={similarity_threshold}")

    def consolidate_categories(
        self,
        discovered_categories: Dict[str, CategoryInfo],
        entity_type: str
    ) -> ConsolidationResult:
        """
        Consolidate discovered categories.

        Args:
            discovered_categories: Dict of category_name -> CategoryInfo
            entity_type: 'intervention', 'condition', or 'mechanism'

        Returns:
            ConsolidationResult with mappings
        """
        logger.info(f"Consolidating {len(discovered_categories)} categories for {entity_type}s...")

        original_count = len(discovered_categories)

        # Skip if too few categories
        if original_count < self.min_group_size:
            logger.info(f"Too few categories ({original_count}), skipping consolidation")
            return ConsolidationResult(
                original_count=original_count,
                consolidated_count=original_count,
                reduction_pct=0.0,
                mappings=[],
                preserved_categories=list(discovered_categories.keys())
            )

        # Build consolidation prompt
        prompt = self._build_consolidation_prompt(discovered_categories, entity_type)

        # Call LLM
        consolidation_map = self._call_llm_for_consolidation(prompt)

        # Validate consolidation
        validated_map = self._validate_consolidation(
            consolidation_map,
            discovered_categories
        )

        # Build consolidation mappings
        mappings = []
        for old_cat, new_cat in validated_map.items():
            if old_cat != new_cat:  # Only track actual changes
                mapping = ConsolidationMapping(
                    old_category=old_cat,
                    new_category=new_cat,
                    reason=f"Synonym consolidation",
                    affected_clusters=discovered_categories[old_cat].usage_count
                )
                mappings.append(mapping)

        # Calculate final categories
        final_categories = set()
        for cat_name in discovered_categories.keys():
            final_cat = validated_map.get(cat_name, cat_name)
            final_categories.add(final_cat)

        consolidated_count = len(final_categories)
        reduction_pct = ((original_count - consolidated_count) / original_count * 100) if original_count > 0 else 0

        result = ConsolidationResult(
            original_count=original_count,
            consolidated_count=consolidated_count,
            reduction_pct=reduction_pct,
            mappings=mappings,
            preserved_categories=sorted(final_categories)
        )

        logger.info(f"Consolidation complete: {original_count} → {consolidated_count} ({reduction_pct:.1f}% reduction)")
        logger.info(f"  Merged: {len(mappings)} categories")

        return result

    def _build_consolidation_prompt(
        self,
        discovered_categories: Dict[str, CategoryInfo],
        entity_type: str
    ) -> str:
        """
        Build LLM prompt for category consolidation.

        Args:
            discovered_categories: Dict of category_name -> CategoryInfo
            entity_type: Entity type

        Returns:
            Formatted prompt string
        """
        # Sort by usage count (most used first)
        sorted_categories = sorted(
            discovered_categories.items(),
            key=lambda x: x[1].usage_count,
            reverse=True
        )

        # Format categories with counts and examples
        category_lines = []
        for cat_name, cat_info in sorted_categories:
            examples = ", ".join(cat_info.example_clusters[:3])
            category_lines.append(f"- {cat_name} ({cat_info.usage_count} clusters): {examples}")

        categories_text = "\n".join(category_lines)

        prompt = f"""You are consolidating {entity_type} categories discovered from clustering.

DISCOVERED CATEGORIES (usage count and examples):
{categories_text}

TASK: Merge synonymous or redundant categories.

GUIDELINES:
- Merge spelling variations: "supplement" ← "supplements"
- Merge synonyms: "exercise" ← "physical_exercise", "physical_activity"
- Keep distinct categories separate: "supplement" ≠ "medication"
- Prefer simpler names: "supplement" over "nutritional_supplement"
- Preserve high-usage categories when possible
- Only merge categories that are TRUE synonyms

EXAMPLES:
- "supplement", "supplements" → supplement
- "exercise", "physical_exercise" → exercise
- "cardiac", "heart_disease", "cardiovascular" → cardiac
- "medication", "pharmaceutical" → medication

Return JSON mapping old → new (only include categories that should be merged):
{{
    "supplements": "supplement",
    "physical_exercise": "exercise",
    "cardiovascular": "cardiac",
    ...
}}

If no consolidation needed for a category, DO NOT include it in the output.
Return empty dict {{}} if no consolidation needed at all.

No explanations. Just the JSON object."""

        return prompt

    def _call_llm_for_consolidation(self, prompt: str) -> Dict[str, str]:
        """
        Call LLM API to get consolidation mappings.

        Args:
            prompt: Formatted prompt

        Returns:
            Dict mapping old_category -> new_category
        """
        try:
            # Use OllamaClient with JSON mode
            response_text = self.llm_client.generate(
                prompt=prompt,
                json_mode=True
            )

            # Parse JSON response
            consolidation_map = json.loads(response_text)

            if not isinstance(consolidation_map, dict):
                logger.warning("LLM response is not a dict, returning empty map")
                return {}

            logger.info(f"LLM proposed {len(consolidation_map)} consolidations")
            return consolidation_map

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Response was: {response_text[:200]}")
            return {}
        except Exception as e:
            logger.error(f"LLM API error during consolidation: {e}")
            return {}

    def _validate_consolidation(
        self,
        consolidation_map: Dict[str, str],
        discovered_categories: Dict[str, CategoryInfo]
    ) -> Dict[str, str]:
        """
        Validate and filter consolidation mappings.

        Args:
            consolidation_map: Proposed mappings from LLM
            discovered_categories: Original category data

        Returns:
            Validated consolidation map
        """
        validated = {}

        # Check consolidation ratio
        consolidation_ratio = len(consolidation_map) / len(discovered_categories)
        if consolidation_ratio > self.max_consolidation_ratio:
            logger.warning(
                f"Consolidation ratio too high ({consolidation_ratio:.1%} > {self.max_consolidation_ratio:.0%}), "
                f"rejecting consolidation"
            )
            return {}

        # Validate each mapping
        for old_cat, new_cat in consolidation_map.items():
            # Check both categories exist
            if old_cat not in discovered_categories:
                logger.warning(f"Unknown old category: {old_cat}")
                continue

            # Normalize new category name
            new_cat = new_cat.lower().strip().replace(" ", "_")

            # Check new category exists OR is being created by consolidation
            if new_cat not in discovered_categories and new_cat not in consolidation_map.values():
                logger.warning(f"Unknown new category: {new_cat}")
                continue

            # Don't allow self-loops
            if old_cat == new_cat:
                continue

            # Preserve high-usage categories (don't merge them INTO something else)
            old_usage = discovered_categories[old_cat].usage_count
            if new_cat in discovered_categories:
                new_usage = discovered_categories[new_cat].usage_count
                if old_usage > new_usage * 2:  # Old category is much more popular
                    logger.warning(
                        f"Rejecting merge of high-usage category {old_cat} ({old_usage}) "
                        f"into {new_cat} ({new_usage})"
                    )
                    continue

            validated[old_cat] = new_cat

        logger.info(f"Validated {len(validated)}/{len(consolidation_map)} consolidations")
        return validated

    def apply_consolidation(
        self,
        naming_results: List,
        consolidation_map: Dict[str, str]
    ) -> List:
        """
        Apply consolidation mappings to naming results.

        Args:
            naming_results: List of NamingResult objects
            consolidation_map: Dict mapping old_category -> new_category

        Returns:
            Updated naming results with consolidated categories
        """
        if not consolidation_map:
            return naming_results

        updated_results = []
        for result in naming_results:
            if result.category in consolidation_map:
                # Update category
                old_cat = result.category
                new_cat = consolidation_map[old_cat]
                result.category = new_cat
                logger.debug(f"Consolidated: {old_cat} → {new_cat} for cluster {result.canonical_name}")

            updated_results.append(result)

        return updated_results


if __name__ == "__main__":
    # Test consolidator
    logging.basicConfig(level=logging.INFO)

    # Sample discovered categories
    from phase_3c_dynamic_namer import CategoryInfo

    test_categories = {
        "supplement": CategoryInfo("supplement", 20, ["HIGH"]*20, ["vitamin D", "omega-3"], False),
        "supplements": CategoryInfo("supplements", 3, ["HIGH"]*3, ["probiotics"], True),
        "exercise": CategoryInfo("exercise", 15, ["HIGH"]*15, ["running", "yoga"], False),
        "physical_exercise": CategoryInfo("physical_exercise", 5, ["HIGH"]*5, ["resistance training"], True),
        "medication": CategoryInfo("medication", 18, ["HIGH"]*18, ["metformin", "statins"], False),
    }

    consolidator = CategoryConsolidator()
    result = consolidator.consolidate_categories(test_categories, "intervention")

    print(f"\nConsolidation Result:")
    print(f"  Original: {result.original_count}")
    print(f"  Consolidated: {result.consolidated_count}")
    print(f"  Reduction: {result.reduction_pct:.1f}%")
    print(f"\n  Mappings:")
    for mapping in result.mappings:
        print(f"    {mapping.old_category} → {mapping.new_category} ({mapping.affected_clusters} clusters)")
    print(f"\n  Final categories: {', '.join(result.preserved_categories)}")
