"""
Fixed Semantic Merger with proper error handling for JSON parsing issues.
"""

import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib

from src.data.config import config, setup_logging
from src.data.api_clients import get_llm_client
from src.data.utils import parse_json_safely

logger = setup_logging(__name__, 'semantic_merger.log')


@dataclass
class InterventionExtraction:
    """Represents a single intervention extraction from an LLM."""
    model_name: str
    intervention_name: str
    health_condition: str
    intervention_category: str
    correlation_type: str
    confidence_score: float
    correlation_strength: float
    supporting_quote: str
    raw_data: Dict[str, Any]


@dataclass
class MergeDecision:
    """Result of semantic merge analysis."""
    is_duplicate: bool
    canonical_name: str
    alternative_names: List[str]
    search_terms: List[str]
    semantic_confidence: float
    reasoning: str
    semantic_group_id: str
    merge_method: str  # 'exact_match', 'synonym_match', 'semantic_similarity'


@dataclass
class ValidationResult:
    """Result from merge decision validation."""
    agrees_with_merge: bool
    confidence: float
    reasoning: str
    alternative_reasoning: str


class SemanticMerger:
    """
    Enhanced LLM-based semantic merger with robust error handling.
    """

    def __init__(self, primary_model: str = "qwen2.5:14b",
                 validator_model: str = "gemma2:9b"):
        """Initialize with primary and validator models."""
        self.primary_model = primary_model
        self.validator_model = validator_model

        # Initialize LLM clients
        self.primary_client = get_llm_client(primary_model)
        self.validator_client = get_llm_client(validator_model)

        # Statistics tracking
        self.stats = {
            'total_comparisons': 0,
            'duplicates_found': 0,
            'json_parse_errors': 0,
            'llm_errors': 0
        }

        logger.info(f"Initialized SemanticMerger with primary_model={primary_model}, validator_model={validator_model}")

    def compare_interventions(self, extract1: InterventionExtraction,
                            extract2: InterventionExtraction) -> MergeDecision:
        """
        Compare two intervention extractions with robust error handling.
        """
        self.stats['total_comparisons'] += 1

        # Quick exact match check first
        if self._are_exact_matches(extract1, extract2):
            return self._create_exact_match_decision(extract1, extract2)

        # Check for obvious non-matches (different conditions)
        if not self._are_same_condition(extract1, extract2):
            return self._create_no_match_decision(extract1, extract2, "Different health conditions")

        # Use LLM for semantic comparison
        try:
            merge_decision = self._llm_semantic_comparison_safe(extract1, extract2)

            if merge_decision.is_duplicate:
                self.stats['duplicates_found'] += 1

            return merge_decision

        except Exception as e:
            logger.error(f"Error in semantic comparison: {e}")
            self.stats['llm_errors'] += 1
            # Fallback to conservative no-merge decision
            return self._create_no_match_decision(extract1, extract2, f"Error in analysis: {e}")

    def _llm_semantic_comparison_safe(self, extract1: InterventionExtraction,
                                    extract2: InterventionExtraction) -> MergeDecision:
        """LLM semantic comparison with robust error handling."""
        prompt = self._build_comparison_prompt(extract1, extract2)

        try:
            response = self.primary_client.generate(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.2
            )

            # Parse LLM response with proper error handling
            decision_data_list = parse_json_safely(response['content'])

            # Fixed: Check for empty list instead of None
            if not decision_data_list or not isinstance(decision_data_list, list) or len(decision_data_list) == 0:
                logger.warning("LLM returned empty or invalid JSON response")
                self.stats['json_parse_errors'] += 1
                return self._create_no_match_decision(extract1, extract2, "Invalid LLM response format")

            decision_data = decision_data_list[0]

            if not isinstance(decision_data, dict):
                logger.warning("LLM returned non-dict in JSON response")
                self.stats['json_parse_errors'] += 1
                return self._create_no_match_decision(extract1, extract2, "Invalid LLM response structure")

            return MergeDecision(
                is_duplicate=decision_data.get('is_duplicate', False),
                canonical_name=decision_data.get('canonical_name', extract1.intervention_name),
                alternative_names=decision_data.get('alternative_names', [extract1.intervention_name, extract2.intervention_name]),
                search_terms=decision_data.get('search_terms', []),
                semantic_confidence=float(decision_data.get('semantic_confidence', 0.5)),
                reasoning=decision_data.get('reasoning', 'No reasoning provided'),
                semantic_group_id=decision_data.get('semantic_group_id', self._generate_semantic_id(extract1, extract2)),
                merge_method=decision_data.get('merge_method', 'llm_analysis')
            )

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            self.stats['llm_errors'] += 1
            return self._create_no_match_decision(extract1, extract2, f"LLM generation failed: {e}")

    def validate_merge_decision_safe(self, decision: MergeDecision,
                                   extract1: InterventionExtraction,
                                   extract2: InterventionExtraction) -> ValidationResult:
        """Safe validation with robust error handling."""
        try:
            validation_prompt = self._build_validation_prompt(decision, extract1, extract2)

            response = self.validator_client.generate(
                prompt=validation_prompt,
                max_tokens=1000,
                temperature=0.1
            )

            validation_data_list = parse_json_safely(response['content'])

            # Fixed: Check for empty list instead of None
            if not validation_data_list or not isinstance(validation_data_list, list) or len(validation_data_list) == 0:
                logger.warning("Validator returned empty or invalid JSON response")
                self.stats['json_parse_errors'] += 1
                return ValidationResult(
                    agrees_with_merge=decision.is_duplicate,  # Default to original decision
                    confidence=0.5,
                    reasoning="Validator JSON parsing failed",
                    alternative_reasoning=""
                )

            validation_data = validation_data_list[0]

            if not isinstance(validation_data, dict):
                logger.warning("Validator returned non-dict in JSON response")
                self.stats['json_parse_errors'] += 1
                return ValidationResult(
                    agrees_with_merge=decision.is_duplicate,  # Default to original decision
                    confidence=0.5,
                    reasoning="Validator response structure invalid",
                    alternative_reasoning=""
                )

            return ValidationResult(
                agrees_with_merge=validation_data.get('agrees_with_merge', decision.is_duplicate),
                confidence=float(validation_data.get('confidence', 0.5)),
                reasoning=validation_data.get('reasoning', ''),
                alternative_reasoning=validation_data.get('alternative_reasoning', '')
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.stats['llm_errors'] += 1
            return ValidationResult(
                agrees_with_merge=decision.is_duplicate,  # Default to original decision
                confidence=0.5,
                reasoning=f"Validation error: {e}",
                alternative_reasoning=""
            )

    def _are_exact_matches(self, extract1: InterventionExtraction, extract2: InterventionExtraction) -> bool:
        """Check if two extractions are exact matches."""
        return (extract1.intervention_name.lower().strip() ==
                extract2.intervention_name.lower().strip())

    def _are_same_condition(self, extract1: InterventionExtraction, extract2: InterventionExtraction) -> bool:
        """Check if two extractions are for the same health condition."""
        cond1 = extract1.health_condition.lower().strip()
        cond2 = extract2.health_condition.lower().strip()

        # Allow some flexibility in condition matching
        return (cond1 == cond2 or
                cond1 in cond2 or
                cond2 in cond1 or
                self._conditions_are_related(cond1, cond2))

    def _conditions_are_related(self, cond1: str, cond2: str) -> bool:
        """Check if two conditions are related (e.g., IBS variants)."""
        # Define condition families
        ibs_variants = ['ibs', 'irritable bowel', 'ibs-d', 'ibs-c', 'ibs-m']
        gerd_variants = ['gerd', 'reflux', 'gastroesophageal', 'gastro-esophageal']

        # Check if both conditions belong to the same family
        for family in [ibs_variants, gerd_variants]:
            if any(variant in cond1 for variant in family) and any(variant in cond2 for variant in family):
                return True

        return False

    def _create_exact_match_decision(self, extract1: InterventionExtraction,
                                   extract2: InterventionExtraction) -> MergeDecision:
        """Create a merge decision for exact matches."""
        return MergeDecision(
            is_duplicate=True,
            canonical_name=extract1.intervention_name,
            alternative_names=[extract1.intervention_name, extract2.intervention_name],
            search_terms=[extract1.intervention_name.lower()],
            semantic_confidence=1.0,
            reasoning="Exact name match",
            semantic_group_id=self._generate_semantic_id(extract1, extract2),
            merge_method='exact_match'
        )

    def _create_no_match_decision(self, extract1: InterventionExtraction,
                                extract2: InterventionExtraction, reason: str) -> MergeDecision:
        """Create a no-merge decision."""
        return MergeDecision(
            is_duplicate=False,
            canonical_name=extract1.intervention_name,
            alternative_names=[extract1.intervention_name],
            search_terms=[extract1.intervention_name.lower()],
            semantic_confidence=0.0,
            reasoning=reason,
            semantic_group_id=self._generate_semantic_id(extract1, extract1),  # Individual ID
            merge_method='no_match'
        )

    def _generate_semantic_id(self, extract1: InterventionExtraction,
                            extract2: InterventionExtraction) -> str:
        """Generate a unique semantic group ID."""
        combined = f"{extract1.intervention_name}_{extract2.intervention_name}_{extract1.health_condition}"
        return f"sem_{hash(combined.lower())}"

    def _build_comparison_prompt(self, extract1: InterventionExtraction,
                               extract2: InterventionExtraction) -> str:
        """Build prompt for LLM semantic comparison."""
        return f"""
Compare these two medical interventions and determine if they are semantic duplicates:

EXTRACTION 1:
- Intervention: {extract1.intervention_name}
- Health Condition: {extract1.health_condition}
- Category: {extract1.intervention_category}
- Supporting Quote: {extract1.supporting_quote}

EXTRACTION 2:
- Intervention: {extract2.intervention_name}
- Health Condition: {extract2.health_condition}
- Category: {extract2.intervention_category}
- Supporting Quote: {extract2.supporting_quote}

Analyze if these interventions are:
1. Exact duplicates (same intervention, different naming)
2. Semantic duplicates (synonyms, abbreviations, alternate forms)
3. Different interventions entirely

Respond with valid JSON:
{{
    "is_duplicate": boolean,
    "canonical_name": "standardized name if duplicate",
    "alternative_names": ["list", "of", "alternative", "names"],
    "search_terms": ["search", "terms", "for", "matching"],
    "semantic_confidence": 0.0-1.0,
    "reasoning": "detailed explanation of decision",
    "semantic_group_id": "unique_group_id",
    "merge_method": "exact_match|synonym_match|semantic_similarity"
}}
"""

    def _build_validation_prompt(self, decision: MergeDecision,
                               extract1: InterventionExtraction,
                               extract2: InterventionExtraction) -> str:
        """Build validation prompt."""
        return f"""
Validate this semantic merge decision:

ORIGINAL DECISION: {decision.is_duplicate}
REASONING: {decision.reasoning}

INTERVENTIONS:
1. {extract1.intervention_name} ({extract1.health_condition})
2. {extract2.intervention_name} ({extract2.health_condition})

Do you agree with this merge decision? Respond with valid JSON:
{{
    "agrees_with_merge": boolean,
    "confidence": 0.0-1.0,
    "reasoning": "validation reasoning",
    "alternative_reasoning": "if you disagree, explain why"
}}
"""

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics."""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self.stats = {
            'total_comparisons': 0,
            'duplicates_found': 0,
            'json_parse_errors': 0,
            'llm_errors': 0
        }