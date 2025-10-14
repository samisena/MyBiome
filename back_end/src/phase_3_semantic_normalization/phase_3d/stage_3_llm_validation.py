"""
Stage 3: LLM Validation

Validate merge candidates using LLM (qwen3:14b) with auto-approval logic.
Classifies relationships and suggests canonical names for merged clusters.
"""

import json
import logging
import requests
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from .stage_2_candidate_generation import MergeCandidate
from .config import Phase3dConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class NameQualityScore:
    """Quality score for a canonical name."""
    score: int  # 0-100
    warnings: List[str]
    acceptable: bool


@dataclass
class DiversityCheck:
    """Diversity check result for cluster children."""
    warning: bool
    severity: str  # 'NONE', 'MODERATE', 'SEVERE'
    inter_child_similarity: float
    message: str


@dataclass
class LLMValidationResult:
    """Result of LLM validation for a merge candidate."""
    candidate: MergeCandidate
    relationship_type: str  # 'MERGE_IDENTICAL', 'CREATE_PARENT', 'DIFFERENT'
    llm_confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    suggested_parent_name: Optional[str]
    child_a_refined_name: Optional[str]
    child_b_refined_name: Optional[str]
    llm_reasoning: str
    name_quality: NameQualityScore
    diversity_check: DiversityCheck
    auto_approved: bool
    flagged_reason: Optional[str]


class LLMValidator:
    """
    Validates merge candidates using LLM.

    Auto-approves based on confidence and quality checks.
    """

    def __init__(self, config: Phase3dConfig = None):
        """
        Initialize LLM validator.

        Args:
            config: Configuration object (uses global if None)
        """
        self.config = config or get_config()
        self.llm_url = f"{self.config.llm_base_url}/api/generate"

        logger.info(f"Initialized LLMValidator with model: {self.config.llm_model}")

    def validate_candidates(
        self,
        candidates: List[MergeCandidate],
        embeddings: Dict[str, any] = None
    ) -> List[LLMValidationResult]:
        """
        Validate all candidates with LLM.

        Args:
            candidates: List of MergeCandidate objects
            embeddings: Optional dict of embeddings for diversity checks

        Returns:
            List of LLMValidationResult objects
        """
        logger.info(f"Validating {len(candidates)} candidates with LLM...")

        results = []
        approved_count = 0
        rejected_count = 0
        flagged_count = 0

        for i, candidate in enumerate(candidates, 1):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(candidates)}")

            try:
                result = self.validate_single_candidate(candidate, embeddings)
                results.append(result)

                if result.auto_approved:
                    approved_count += 1
                elif result.relationship_type == 'DIFFERENT':
                    rejected_count += 1
                else:
                    flagged_count += 1

            except Exception as e:
                logger.error(f"  Failed to validate candidate {candidate.cluster_a_id}-{candidate.cluster_b_id}: {e}")
                continue

        logger.info(f"Validation complete:")
        logger.info(f"  Approved: {approved_count}")
        logger.info(f"  Rejected: {rejected_count}")
        logger.info(f"  Flagged: {flagged_count}")

        return results

    def validate_single_candidate(
        self,
        candidate: MergeCandidate,
        embeddings: Dict[str, any] = None
    ) -> LLMValidationResult:
        """
        Validate a single merge candidate.

        Args:
            candidate: MergeCandidate object
            embeddings: Optional dict of embeddings

        Returns:
            LLMValidationResult object
        """
        # Call LLM
        llm_response = self._call_llm(candidate)

        # Parse response
        relationship_type = llm_response.get('relationship_type', 'DIFFERENT')
        llm_confidence = llm_response.get('confidence', 'LOW')
        suggested_parent_name = llm_response.get('suggested_parent_name')
        child_a_refined = llm_response.get('child_a_refined_name')
        child_b_refined = llm_response.get('child_b_refined_name')
        reasoning = llm_response.get('reasoning', '')

        # Quality checks
        name_quality = self._score_name_quality(suggested_parent_name or '')
        diversity_check = self._check_diversity(candidate, embeddings)

        # Auto-approval decision
        auto_approved, flagged_reason = self._should_auto_approve(
            relationship_type,
            llm_confidence,
            name_quality,
            diversity_check
        )

        return LLMValidationResult(
            candidate=candidate,
            relationship_type=relationship_type,
            llm_confidence=llm_confidence,
            suggested_parent_name=suggested_parent_name,
            child_a_refined_name=child_a_refined,
            child_b_refined_name=child_b_refined,
            llm_reasoning=reasoning,
            name_quality=name_quality,
            diversity_check=diversity_check,
            auto_approved=auto_approved,
            flagged_reason=flagged_reason
        )

    def _call_llm(self, candidate: MergeCandidate) -> Dict:
        """
        Call LLM API to classify relationship.

        Args:
            candidate: MergeCandidate object

        Returns:
            Dict with LLM response
        """
        prompt = self._build_prompt(candidate)

        payload = {
            "model": self.config.llm_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.3,  # Lower temperature for consistency
                "num_predict": 500
            }
        }

        try:
            response = requests.post(
                self.llm_url,
                json=payload,
                timeout=self.config.llm_timeout
            )
            response.raise_for_status()

            response_data = response.json()
            response_text = response_data.get('response', '{}')

            # Parse JSON response
            llm_output = json.loads(response_text)

            return llm_output

        except requests.exceptions.Timeout:
            logger.error(f"LLM timeout for candidate {candidate.cluster_a_id}-{candidate.cluster_b_id}")
            return {'relationship_type': 'DIFFERENT', 'confidence': 'LOW', 'reasoning': 'Timeout'}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return {'relationship_type': 'DIFFERENT', 'confidence': 'LOW', 'reasoning': 'Parse error'}

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return {'relationship_type': 'DIFFERENT', 'confidence': 'LOW', 'reasoning': f'Error: {e}'}

    def _build_prompt(self, candidate: MergeCandidate) -> str:
        """
        Build LLM prompt for relationship classification.

        Args:
            candidate: MergeCandidate object

        Returns:
            Formatted prompt string
        """
        cluster_a = candidate.cluster_a
        cluster_b = candidate.cluster_b

        # Sample members (max 5 per cluster)
        members_a = cluster_a.members[:5]
        members_b = cluster_b.members[:5]

        prompt = f"""You are a medical taxonomy expert building hierarchical cluster structures.
Your classifications will be AUTO-APPROVED, so be precise and conservative.

RELATIONSHIP TYPES:
1. MERGE_IDENTICAL: Same concept, different wording (merge into single cluster)
2. CREATE_PARENT: Related concepts, create common parent (preserve children)
3. DIFFERENT: Unrelated or too distinct, keep separate

PARENT NAME REQUIREMENTS:
- Be SPECIFIC (e.g., "Anti-Inflammatory Mechanisms" not "Mechanisms")
- Capture shared function (e.g., "Probiotic-Mediated Gut Modulation")
- Avoid generic terms alone: "interventions", "treatments", "supplements"
- Good examples: "Aerobic Exercise", "COX Enzyme Inhibition", "Lactobacillus Probiotics"
- Bad examples: "Medical Treatments", "Supplements", "Mechanisms"

CONFIDENCE LEVELS:
- HIGH: Very confident (>90% certain), clear semantic relationship
- MEDIUM: Somewhat confident (70-90%), related but boundary unclear
- LOW: Uncertain (<70%), may be false positive

TASK:
Cluster A (ID={cluster_a.cluster_id}):
  Canonical name: {cluster_a.canonical_name}
  Members ({len(cluster_a.members)} total, showing first 5):
{chr(10).join('    - ' + m for m in members_a)}

Cluster B (ID={cluster_b.cluster_id}):
  Canonical name: {cluster_b.canonical_name}
  Members ({len(cluster_b.members)} total, showing first 5):
{chr(10).join('    - ' + m for m in members_b)}

Similarity: {candidate.similarity:.3f}

Classify the relationship and suggest names.

IMPORTANT: If you're unsure or the parent name would be too generic, classify as DIFFERENT.

Respond in JSON format with these fields:
- relationship_type: "MERGE_IDENTICAL" | "CREATE_PARENT" | "DIFFERENT"
- suggested_parent_name: "specific parent name" (if merging)
- child_a_refined_name: "refined name for cluster A" (optional)
- child_b_refined_name: "refined name for cluster B" (optional)
- confidence: "HIGH" | "MEDIUM" | "LOW"
- reasoning: "brief explanation"
"""

        return prompt

    def _score_name_quality(self, name: str) -> NameQualityScore:
        """
        Score canonical name quality.

        Args:
            name: Canonical name to score

        Returns:
            NameQualityScore object
        """
        if not name:
            return NameQualityScore(score=0, warnings=['Empty name'], acceptable=False)

        warnings = []
        score = 100

        name_lower = name.lower()

        # Check for forbidden generic terms (alone)
        forbidden = self.config.forbidden_generic_terms
        if any(name_lower == term for term in forbidden):
            score -= 50
            warnings.append(f"Too generic: '{name}' needs qualifier")

        # Penalize very short names
        if len(name.split()) < 2:
            score -= 10
            warnings.append("Single-word name may be too broad")

        # Reward specific medical terms
        specific_terms = self.config.specific_term_indicators
        if any(term in name_lower for term in specific_terms):
            score += 10

        # Reward compound names (more specific)
        if len(name.split()) >= 3:
            score += 10

        # Penalize excessive length
        if len(name) > 80:
            score -= 5
            warnings.append("Name very long (>80 chars)")

        score = max(0, min(100, score))
        acceptable = score >= self.config.auto_approve_name_quality_min

        return NameQualityScore(
            score=score,
            warnings=warnings,
            acceptable=acceptable
        )

    def _check_diversity(
        self,
        candidate: MergeCandidate,
        embeddings: Dict[str, any] = None
    ) -> DiversityCheck:
        """
        Check diversity of children (are they too different?).

        Args:
            candidate: MergeCandidate object
            embeddings: Optional dict of embeddings

        Returns:
            DiversityCheck object
        """
        if embeddings is None:
            # No embeddings available - skip check
            return DiversityCheck(
                warning=False,
                severity='NONE',
                inter_child_similarity=0.0,
                message='Diversity check skipped (no embeddings)'
            )

        # Get member embeddings
        members_a = [embeddings.get(m) for m in candidate.cluster_a.members if m in embeddings]
        members_b = [embeddings.get(m) for m in candidate.cluster_b.members if m in embeddings]

        if not members_a or not members_b:
            return DiversityCheck(
                warning=False,
                severity='NONE',
                inter_child_similarity=0.0,
                message='Insufficient embeddings for diversity check'
            )

        # Compute average inter-child similarity
        import numpy as np
        from .validation_metrics import cosine_similarity

        inter_sims = []
        for emb_a in members_a[:5]:  # Sample up to 5
            for emb_b in members_b[:5]:
                sim = cosine_similarity(emb_a, emb_b)
                inter_sims.append(sim)

        avg_inter_sim = float(np.mean(inter_sims)) if inter_sims else 0.0

        # Determine severity
        if avg_inter_sim < self.config.diversity_severe_threshold:
            severity = 'SEVERE'
            warning = True
            message = f'Children have very low similarity ({avg_inter_sim:.2f}). Parent may be too broad.'
        elif avg_inter_sim < self.config.diversity_warning_threshold:
            severity = 'MODERATE'
            warning = True
            message = f'Children have moderate similarity ({avg_inter_sim:.2f}). Review parent scope.'
        else:
            severity = 'NONE'
            warning = False
            message = f'Children have good similarity ({avg_inter_sim:.2f}).'

        return DiversityCheck(
            warning=warning,
            severity=severity,
            inter_child_similarity=avg_inter_sim,
            message=message
        )

    def _should_auto_approve(
        self,
        relationship_type: str,
        llm_confidence: str,
        name_quality: NameQualityScore,
        diversity_check: DiversityCheck
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide if merge should be auto-approved.

        Args:
            relationship_type: LLM classification
            llm_confidence: LLM confidence level
            name_quality: Name quality score
            diversity_check: Diversity check result

        Returns:
            Tuple of (should_approve, flagged_reason)
        """
        # Never approve DIFFERENT
        if relationship_type == 'DIFFERENT':
            return False, 'LLM classified as DIFFERENT'

        # Check confidence
        if llm_confidence != self.config.auto_approve_confidence:
            return False, f'Low confidence ({llm_confidence})'

        # Check name quality
        if not name_quality.acceptable:
            return False, f'Poor name quality (score={name_quality.score})'

        # Check diversity
        if diversity_check.severity == 'SEVERE':
            return False, f'Severe diversity warning ({diversity_check.inter_child_similarity:.2f})'

        # All checks passed
        return True, None

    def get_approved_merges(self, results: List[LLMValidationResult]) -> List[LLMValidationResult]:
        """
        Filter to only auto-approved merges.

        Args:
            results: List of LLMValidationResult objects

        Returns:
            List of approved results
        """
        approved = [r for r in results if r.auto_approved]
        logger.info(f"Auto-approved merges: {len(approved)}/{len(results)}")
        return approved

    def save_flagged_merges(self, results: List[LLMValidationResult], output_path: str):
        """
        Save flagged merges to file for human review.

        Args:
            results: List of LLMValidationResult objects
            output_path: Path to output file
        """
        flagged = [r for r in results if not r.auto_approved and r.relationship_type != 'DIFFERENT']

        if not flagged:
            logger.info("No flagged merges to save")
            return

        data = {
            'total_flagged': len(flagged),
            'flagged_merges': [
                {
                    'cluster_a_id': r.candidate.cluster_a_id,
                    'cluster_a_name': r.candidate.cluster_a.canonical_name,
                    'cluster_b_id': r.candidate.cluster_b_id,
                    'cluster_b_name': r.candidate.cluster_b.canonical_name,
                    'similarity': float(r.candidate.similarity),
                    'relationship_type': r.relationship_type,
                    'llm_confidence': r.llm_confidence,
                    'suggested_parent_name': r.suggested_parent_name,
                    'name_quality_score': r.name_quality.score,
                    'diversity_severity': r.diversity_check.severity,
                    'flagged_reason': r.flagged_reason,
                    'llm_reasoning': r.llm_reasoning
                }
                for r in flagged
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(flagged)} flagged merges to: {output_path}")


if __name__ == "__main__":
    # Test LLM validator
    logging.basicConfig(level=logging.INFO)

    from .validation_metrics import Cluster
    import numpy as np

    # Create test candidate
    test_candidate = MergeCandidate(
        cluster_a_id=0,
        cluster_b_id=1,
        cluster_a=Cluster(0, "Aerobic Training", ["continuous aerobic training", "aerobic exercise"], None, 0),
        cluster_b=Cluster(1, "General Exercise", ["physical exercise", "exercise intervention"], None, 0),
        similarity=0.87,
        confidence_tier='MEDIUM'
    )

    validator = LLMValidator()

    print("Testing LLM validation...")
    result = validator.validate_single_candidate(test_candidate)

    print(f"\nResult:")
    print(f"  Relationship: {result.relationship_type}")
    print(f"  LLM Confidence: {result.llm_confidence}")
    print(f"  Suggested parent: {result.suggested_parent_name}")
    print(f"  Name quality: {result.name_quality.score}/100")
    print(f"  Auto-approved: {result.auto_approved}")
    if result.flagged_reason:
        print(f"  Flagged: {result.flagged_reason}")
    print(f"  Reasoning: {result.llm_reasoning}")
