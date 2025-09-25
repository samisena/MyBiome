"""
Semantic Merger LLM for identifying and merging duplicate interventions.
Uses advanced LLM reasoning to identify synonyms, variations, and semantically identical interventions.
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
    suggested_corrections: List[str]


class SemanticMerger:
    """
    Advanced semantic merger using LLM reasoning to identify duplicate interventions.
    Integrates with the dual-model pipeline to prevent duplicates during processing.
    """

    def __init__(self, primary_model: str = 'qwen2.5:14b', validator_model: str = 'gemma2:9b'):
        """
        Initialize semantic merger with primary and validation models.

        Args:
            primary_model: Model to use for primary semantic analysis
            validator_model: Model to use for validating merge decisions
        """
        self.primary_model = primary_model
        self.validator_model = validator_model
        self.primary_client = get_llm_client(primary_model)
        self.validator_client = get_llm_client(validator_model)

        # Cache for semantic groups to avoid reprocessing
        self.semantic_groups = {}

        # Statistics tracking
        self.stats = {
            'total_comparisons': 0,
            'duplicates_found': 0,
            'validation_agreements': 0,
            'validation_disagreements': 0,
            'human_reviews_needed': 0
        }

        logger.info(f"Initialized SemanticMerger with primary_model={primary_model}, validator_model={validator_model}")

    def compare_interventions(self, extract1: InterventionExtraction,
                            extract2: InterventionExtraction) -> MergeDecision:
        """
        Compare two intervention extractions to determine if they're duplicates.

        Args:
            extract1: First intervention extraction
            extract2: Second intervention extraction

        Returns:
            MergeDecision with merge analysis results
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
            merge_decision = self._llm_semantic_comparison(extract1, extract2)

            if merge_decision.is_duplicate:
                self.stats['duplicates_found'] += 1

            return merge_decision

        except Exception as e:
            logger.error(f"Error in semantic comparison: {e}")
            # Fallback to conservative no-merge decision
            return self._create_no_match_decision(extract1, extract2, f"Error in analysis: {e}")

    def validate_merge_decision(self, decision: MergeDecision,
                              extract1: InterventionExtraction,
                              extract2: InterventionExtraction) -> ValidationResult:
        """
        Use validation LLM to verify the merge decision.

        Args:
            decision: Original merge decision to validate
            extract1: First intervention extraction
            extract2: Second intervention extraction

        Returns:
            ValidationResult with validation analysis
        """
        try:
            validation_prompt = self._build_validation_prompt(decision, extract1, extract2)

            response = self.validator_client.generate(
                prompt=validation_prompt,
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent validation
            )

            validation_data_list = parse_json_safely(response['content'])

            # parse_json_safely returns List[Dict], so extract the first dict
            validation_data = None
            if validation_data_list and isinstance(validation_data_list, list) and len(validation_data_list) > 0:
                validation_data = validation_data_list[0]

            if validation_data and isinstance(validation_data, dict):
                result = ValidationResult(
                    agrees_with_merge=validation_data.get('agrees_with_merge', True),
                    confidence=validation_data.get('confidence', 0.5),
                    reasoning=validation_data.get('reasoning', ''),
                    suggested_corrections=validation_data.get('suggested_corrections', [])
                )

                if result.agrees_with_merge:
                    self.stats['validation_agreements'] += 1
                else:
                    self.stats['validation_disagreements'] += 1

                return result
            else:
                logger.warning("Failed to parse validation response")
                return ValidationResult(True, 0.5, "Validation parsing failed", [])

        except Exception as e:
            logger.error(f"Error in merge validation: {e}")
            return ValidationResult(True, 0.5, f"Validation error: {e}", [])

    def create_merged_intervention(self, extractions: List[InterventionExtraction],
                                 decision: MergeDecision) -> Dict[str, Any]:
        """
        Create a single merged intervention from multiple extractions.

        Args:
            extractions: List of intervention extractions to merge
            decision: Merge decision with canonical naming

        Returns:
            Merged intervention dictionary for database storage
        """
        if not extractions:
            raise ValueError("Cannot merge empty extractions list")

        # Use the highest confidence extraction as base
        base_extraction = max(extractions, key=lambda x: x.confidence_score)

        # Aggregate model information
        contributing_models = list(set(e.model_name for e in extractions))
        model_confidences = {e.model_name: e.confidence_score for e in extractions}

        # Calculate consensus confidence
        consensus_confidence = sum(e.confidence_score for e in extractions) / len(extractions)

        # Determine model agreement level
        agreement_level = self._calculate_agreement_level(extractions)

        # Create merged intervention
        merged = {
            # Basic intervention info (from base extraction)
            'intervention_name': base_extraction.intervention_name,
            'intervention_category': base_extraction.intervention_category,
            'health_condition': base_extraction.health_condition,
            'correlation_type': base_extraction.correlation_type,
            'confidence_score': base_extraction.confidence_score,
            'correlation_strength': base_extraction.correlation_strength,
            'supporting_quote': base_extraction.supporting_quote,

            # Enhanced semantic fields
            'canonical_name': decision.canonical_name,
            'alternative_names': decision.alternative_names,
            'search_terms': decision.search_terms,
            'semantic_group_id': decision.semantic_group_id,
            'semantic_confidence': decision.semantic_confidence,
            'merge_source': self.primary_model,

            # Consensus tracking
            'consensus_confidence': consensus_confidence,
            'model_agreement': agreement_level,
            'models_used': ','.join(contributing_models),
            'raw_extraction_count': len(extractions),
            'models_contributing': [
                {
                    'model': e.model_name,
                    'confidence': e.confidence_score,
                    'intervention_name': e.intervention_name,
                    'supporting_quote': e.supporting_quote[:200] + '...' if len(e.supporting_quote) > 200 else e.supporting_quote
                }
                for e in extractions
            ],

            # Merge decision log
            'merge_decision_log': {
                'timestamp': datetime.now().isoformat(),
                'decision_method': decision.merge_method,
                'reasoning': decision.reasoning,
                'primary_model': self.primary_model,
                'extractions_merged': len(extractions)
            }
        }

        # Copy other fields from base extraction
        if hasattr(base_extraction, 'raw_data') and base_extraction.raw_data:
            for key, value in base_extraction.raw_data.items():
                if key not in merged and value is not None:
                    merged[key] = value

        return merged

    def _safe_string_lower(self, value) -> str:
        """Safely convert any value to lowercase string."""
        if value is None:
            return ''
        if isinstance(value, list):
            return str(value[0] if value else '').lower()
        if isinstance(value, str):
            return value.lower()
        return str(value).lower()

    def _are_exact_matches(self, extract1: InterventionExtraction,
                          extract2: InterventionExtraction) -> bool:
        """Check if two extractions are exact matches."""
        name1 = self._safe_string_lower(extract1.intervention_name).strip()
        name2 = self._safe_string_lower(extract2.intervention_name).strip()
        cond1 = self._safe_string_lower(extract1.health_condition).strip()
        cond2 = self._safe_string_lower(extract2.health_condition).strip()
        cat1 = self._safe_string_lower(extract1.intervention_category)
        cat2 = self._safe_string_lower(extract2.intervention_category)

        return (name1 == name2 and cond1 == cond2 and cat1 == cat2)

    def _are_same_condition(self, extract1: InterventionExtraction,
                          extract2: InterventionExtraction) -> bool:
        """Check if two extractions are for the same health condition."""
        cond1 = self._safe_string_lower(extract1.health_condition).strip()
        cond2 = self._safe_string_lower(extract2.health_condition).strip()

        # Exact match
        if cond1 == cond2:
            return True

        # Check for common condition abbreviations/variations
        condition_synonyms = {
            'ibs': ['irritable bowel syndrome', 'irritable bowel'],
            'gerd': ['gastroesophageal reflux disease', 'gastro-esophageal reflux disease',
                    'gastroesophageal reflux disease (gerd)', 'gastro-esophageal reflux disease (gerd)',
                    'acid reflux', 'reflux disease', 'gerd'],
            'crohns': ["crohn's disease", 'crohns disease', 'crohn disease'],
            'diabetes': ['diabetes mellitus', 'type 2 diabetes', 'type 1 diabetes'],
            'depression': ['major depressive disorder', 'depressive disorder', 'clinical depression']
        }

        for key, synonyms in condition_synonyms.items():
            if (any(syn in cond1 for syn in synonyms) and any(syn in cond2 for syn in synonyms)):
                return True

        return False

    def _llm_semantic_comparison(self, extract1: InterventionExtraction,
                               extract2: InterventionExtraction) -> MergeDecision:
        """Use LLM to perform semantic comparison of interventions."""
        prompt = self._build_comparison_prompt(extract1, extract2)

        response = self.primary_client.generate(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.2  # Low temperature for consistent analysis
        )

        # Parse LLM response
        decision_data_list = parse_json_safely(response['content'])

        # parse_json_safely returns List[Dict], so extract the first dict
        decision_data = None
        if decision_data_list and isinstance(decision_data_list, list) and len(decision_data_list) > 0:
            decision_data = decision_data_list[0]

        if decision_data and isinstance(decision_data, dict):
            return MergeDecision(
                is_duplicate=decision_data.get('is_duplicate', False),
                canonical_name=decision_data.get('canonical_name', extract1.intervention_name),
                alternative_names=decision_data.get('alternative_names', []),
                search_terms=decision_data.get('search_terms', []),
                semantic_confidence=decision_data.get('confidence', 0.5),
                reasoning=decision_data.get('reasoning', ''),
                semantic_group_id=self._generate_semantic_group_id(decision_data.get('canonical_name', extract1.intervention_name), extract1.health_condition),
                merge_method='semantic_similarity'
            )
        else:
            logger.warning("Failed to parse LLM comparison response")
            return self._create_no_match_decision(extract1, extract2, "Failed to parse LLM response")

    def _build_comparison_prompt(self, extract1: InterventionExtraction,
                               extract2: InterventionExtraction) -> str:
        """Build prompt for LLM semantic comparison."""
        return f"""
You are a medical intervention semantic analyzer. Compare these two intervention extractions to determine if they represent the same medical intervention with different names.

EXTRACTION 1:
- Intervention: "{extract1.intervention_name}"
- Health Condition: "{extract1.health_condition}"
- Category: "{extract1.intervention_category}"
- Evidence Type: "{extract1.correlation_type}"
- Supporting Quote: "{extract1.supporting_quote[:300]}..."
- Model: {extract1.model_name}
- Confidence: {extract1.confidence_score}

EXTRACTION 2:
- Intervention: "{extract2.intervention_name}"
- Health Condition: "{extract2.health_condition}"
- Category: "{extract2.intervention_category}"
- Evidence Type: "{extract2.correlation_type}"
- Supporting Quote: "{extract2.supporting_quote[:300]}..."
- Model: {extract2.model_name}
- Confidence: {extract2.confidence_score}

ANALYSIS GUIDELINES:
1. Consider synonyms, abbreviations, brand names vs generic names
2. Look for different levels of specificity (e.g., "surgery" vs "laparoscopic surgery")
3. Consider mechanism of action similarity
4. Account for different naming conventions in medical literature
5. Be conservative - only merge if you're confident they're the same intervention

RESPOND IN JSON FORMAT:
{{
    "is_duplicate": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your decision",
    "canonical_name": "Best standardized name for this intervention",
    "alternative_names": ["list", "of", "alternative", "names", "found"],
    "search_terms": ["list", "of", "searchable", "terms", "including", "abbreviations"],
    "key_differences": ["any", "important", "differences", "noted"]
}}

Focus on semantic meaning rather than exact text matching. Two interventions are duplicates if they refer to the same medical treatment approach, even with different naming conventions.
"""

    def _build_validation_prompt(self, decision: MergeDecision,
                               extract1: InterventionExtraction,
                               extract2: InterventionExtraction) -> str:
        """Build prompt for validating merge decisions."""
        return f"""
You are a medical intervention validation expert. Review this merge decision made by another AI model.

ORIGINAL INTERVENTIONS:
1. "{extract1.intervention_name}" for {extract1.health_condition}
2. "{extract2.intervention_name}" for {extract2.health_condition}

MERGE DECISION:
- Decided: {"DUPLICATE" if decision.is_duplicate else "NOT DUPLICATE"}
- Canonical Name: "{decision.canonical_name}"
- Alternative Names: {decision.alternative_names}
- Confidence: {decision.semantic_confidence}
- Reasoning: "{decision.reasoning}"

YOUR TASK:
Validate whether this merge decision is correct. Consider:
1. Are these truly the same intervention?
2. Is the canonical name appropriate?
3. Are there any missed alternatives or incorrect groupings?
4. Could this cause confusion or data quality issues?

RESPOND IN JSON FORMAT:
{{
    "agrees_with_merge": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Your analysis of the merge decision",
    "suggested_corrections": ["any", "corrections", "you", "would", "recommend"],
    "risk_assessment": "low/medium/high risk if this merge is wrong"
}}

Be thorough in your validation - incorrect merges can damage data quality.
"""

    def _create_exact_match_decision(self, extract1: InterventionExtraction,
                                   extract2: InterventionExtraction) -> MergeDecision:
        """Create merge decision for exact matches."""
        return MergeDecision(
            is_duplicate=True,
            canonical_name=extract1.intervention_name,
            alternative_names=[extract1.intervention_name, extract2.intervention_name],
            search_terms=[self._safe_string_lower(extract1.intervention_name)],
            semantic_confidence=1.0,
            reasoning="Exact text match detected",
            semantic_group_id=self._generate_semantic_group_id(extract1.intervention_name, extract1.health_condition),
            merge_method='exact_match'
        )

    def _create_no_match_decision(self, extract1: InterventionExtraction,
                                extract2: InterventionExtraction,
                                reason: str) -> MergeDecision:
        """Create merge decision for non-matches."""
        return MergeDecision(
            is_duplicate=False,
            canonical_name=extract1.intervention_name,  # Keep original
            alternative_names=[],
            search_terms=[],
            semantic_confidence=0.0,
            reasoning=reason,
            semantic_group_id='',
            merge_method='no_match'
        )

    def _generate_semantic_group_id(self, canonical_name: str, condition: str) -> str:
        """Generate a unique ID for semantic groups."""
        # Create deterministic ID based on canonical name and condition
        content = f"{self._safe_string_lower(canonical_name).strip()}_{self._safe_string_lower(condition).strip()}"
        hash_obj = hashlib.md5(content.encode())
        return f"sem_{hash_obj.hexdigest()[:12]}"

    def _calculate_agreement_level(self, extractions: List[InterventionExtraction]) -> str:
        """Calculate the level of agreement between models."""
        if len(extractions) == 1:
            return 'single'

        # Check if all models found the same intervention name
        names = set(self._safe_string_lower(e.intervention_name).strip() for e in extractions)
        evidence_types = set(e.correlation_type for e in extractions)

        if len(names) == 1 and len(evidence_types) == 1:
            return 'full'
        elif len(names) == 1:
            return 'partial'  # Same intervention, different evidence assessment
        else:
            return 'conflict'  # Different interventions found

    def get_statistics(self) -> Dict[str, int]:
        """Get processing statistics."""
        return self.stats.copy()

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.stats = {key: 0 for key in self.stats}