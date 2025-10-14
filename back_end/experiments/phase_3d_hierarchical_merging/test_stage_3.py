"""
Test Suite for Stage 3: LLM Validation

Tests LLM validation logic, name quality scoring, and auto-approval.
Uses mock LLM responses to avoid API calls.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from stage_3_llm_validation import (
    LLMValidator,
    NameQualityScore,
    DiversityCheck,
    LLMValidationResult
)
from stage_2_candidate_generation import MergeCandidate
from validation_metrics import Cluster
from config import Phase3dConfig


class TestNameQualityScoring:
    """Test canonical name quality scoring."""

    def test_good_specific_name(self):
        """Test scoring of good specific name."""
        validator = LLMValidator()

        score = validator._score_name_quality("Anti-Inflammatory COX Inhibition")

        assert score.score >= 70
        assert score.acceptable == True
        assert len(score.warnings) == 0

    def test_generic_name_alone(self):
        """Test penalizing generic term alone."""
        validator = LLMValidator()

        score = validator._score_name_quality("supplement")

        assert score.score < 60
        assert score.acceptable == False
        assert any('generic' in w.lower() for w in score.warnings)

    def test_generic_with_qualifier(self):
        """Test accepting generic term with qualifier."""
        validator = LLMValidator()

        score = validator._score_name_quality("Probiotic Supplement")

        # Should be better than "supplement" alone
        assert score.score >= 60

    def test_single_word_name(self):
        """Test penalizing single-word names."""
        validator = LLMValidator()

        score = validator._score_name_quality("Exercise")

        # Should have warning about single word
        assert any('single-word' in w.lower() for w in score.warnings)

    def test_specific_medical_terms(self):
        """Test rewarding specific medical terms."""
        validator = LLMValidator()

        score_generic = validator._score_name_quality("Health Approach")
        score_specific = validator._score_name_quality("Glucose Regulation")

        # Specific should score higher
        assert score_specific.score > score_generic.score

    def test_empty_name(self):
        """Test handling empty name."""
        validator = LLMValidator()

        score = validator._score_name_quality("")

        assert score.score == 0
        assert score.acceptable == False
        assert 'Empty name' in score.warnings


class TestDiversityCheck:
    """Test diversity checking for children."""

    def test_similar_children(self):
        """Test children with high similarity (good)."""
        validator = LLMValidator()

        candidate = MergeCandidate(
            cluster_a_id=0,
            cluster_b_id=1,
            cluster_a=Cluster(0, "c0", ["m1", "m2"], None, 0),
            cluster_b=Cluster(1, "c1", ["m3", "m4"], None, 0),
            similarity=0.90,
            confidence_tier='HIGH'
        )

        embeddings = {
            'm1': np.array([0.9, 0.1, 0.0], dtype=np.float32),
            'm2': np.array([0.95, 0.05, 0.0], dtype=np.float32),
            'm3': np.array([0.92, 0.08, 0.0], dtype=np.float32),
            'm4': np.array([0.88, 0.12, 0.0], dtype=np.float32)
        }

        check = validator._check_diversity(candidate, embeddings)

        assert check.severity == 'NONE'
        assert check.warning == False
        assert check.inter_child_similarity > 0.70

    def test_different_children(self):
        """Test children with low similarity (warning)."""
        validator = LLMValidator()

        candidate = MergeCandidate(
            cluster_a_id=0,
            cluster_b_id=1,
            cluster_a=Cluster(0, "c0", ["m1"], None, 0),
            cluster_b=Cluster(1, "c1", ["m2"], None, 0),
            similarity=0.85,
            confidence_tier='MEDIUM'
        )

        # Very different embeddings
        embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Orthogonal
        }

        check = validator._check_diversity(candidate, embeddings)

        assert check.severity in ['MODERATE', 'SEVERE']
        assert check.warning == True
        assert check.inter_child_similarity < 0.40

    def test_no_embeddings(self):
        """Test handling when embeddings not available."""
        validator = LLMValidator()

        candidate = MergeCandidate(
            cluster_a_id=0,
            cluster_b_id=1,
            cluster_a=Cluster(0, "c0", ["m1"], None, 0),
            cluster_b=Cluster(1, "c1", ["m2"], None, 0),
            similarity=0.90,
            confidence_tier='HIGH'
        )

        check = validator._check_diversity(candidate, embeddings=None)

        assert check.severity == 'NONE'
        assert check.warning == False
        assert 'skipped' in check.message.lower()


class TestAutoApprovalLogic:
    """Test auto-approval decision logic."""

    def test_approve_all_good(self):
        """Test approval when all checks pass."""
        validator = LLMValidator()

        name_quality = NameQualityScore(score=80, warnings=[], acceptable=True)
        diversity_check = DiversityCheck(
            warning=False,
            severity='NONE',
            inter_child_similarity=0.75,
            message='Good'
        )

        should_approve, reason = validator._should_auto_approve(
            relationship_type='CREATE_PARENT',
            llm_confidence='HIGH',
            name_quality=name_quality,
            diversity_check=diversity_check
        )

        assert should_approve == True
        assert reason is None

    def test_reject_different(self):
        """Test rejection of DIFFERENT classification."""
        validator = LLMValidator()

        name_quality = NameQualityScore(score=80, warnings=[], acceptable=True)
        diversity_check = DiversityCheck(
            warning=False,
            severity='NONE',
            inter_child_similarity=0.75,
            message='Good'
        )

        should_approve, reason = validator._should_auto_approve(
            relationship_type='DIFFERENT',
            llm_confidence='HIGH',
            name_quality=name_quality,
            diversity_check=diversity_check
        )

        assert should_approve == False
        assert 'DIFFERENT' in reason

    def test_reject_low_confidence(self):
        """Test rejection of low confidence."""
        validator = LLMValidator()

        name_quality = NameQualityScore(score=80, warnings=[], acceptable=True)
        diversity_check = DiversityCheck(
            warning=False,
            severity='NONE',
            inter_child_similarity=0.75,
            message='Good'
        )

        should_approve, reason = validator._should_auto_approve(
            relationship_type='CREATE_PARENT',
            llm_confidence='MEDIUM',  # Not HIGH
            name_quality=name_quality,
            diversity_check=diversity_check
        )

        assert should_approve == False
        assert 'confidence' in reason.lower()

    def test_reject_poor_name_quality(self):
        """Test rejection of poor name quality."""
        validator = LLMValidator()

        name_quality = NameQualityScore(score=40, warnings=['Too generic'], acceptable=False)
        diversity_check = DiversityCheck(
            warning=False,
            severity='NONE',
            inter_child_similarity=0.75,
            message='Good'
        )

        should_approve, reason = validator._should_auto_approve(
            relationship_type='CREATE_PARENT',
            llm_confidence='HIGH',
            name_quality=name_quality,
            diversity_check=diversity_check
        )

        assert should_approve == False
        assert 'name quality' in reason.lower()

    def test_reject_severe_diversity(self):
        """Test rejection of severe diversity warning."""
        validator = LLMValidator()

        name_quality = NameQualityScore(score=80, warnings=[], acceptable=True)
        diversity_check = DiversityCheck(
            warning=True,
            severity='SEVERE',
            inter_child_similarity=0.25,
            message='Too different'
        )

        should_approve, reason = validator._should_auto_approve(
            relationship_type='CREATE_PARENT',
            llm_confidence='HIGH',
            name_quality=name_quality,
            diversity_check=diversity_check
        )

        assert should_approve == False
        assert 'diversity' in reason.lower()

    def test_accept_moderate_diversity(self):
        """Test acceptance of moderate diversity warning."""
        validator = LLMValidator()

        name_quality = NameQualityScore(score=80, warnings=[], acceptable=True)
        diversity_check = DiversityCheck(
            warning=True,
            severity='MODERATE',  # Not SEVERE
            inter_child_similarity=0.35,
            message='Moderate'
        )

        should_approve, reason = validator._should_auto_approve(
            relationship_type='CREATE_PARENT',
            llm_confidence='HIGH',
            name_quality=name_quality,
            diversity_check=diversity_check
        )

        # MODERATE diversity should be acceptable
        assert should_approve == True


class TestPromptBuilding:
    """Test LLM prompt construction."""

    def test_prompt_format(self):
        """Test prompt contains required elements."""
        validator = LLMValidator()

        candidate = MergeCandidate(
            cluster_a_id=0,
            cluster_b_id=1,
            cluster_a=Cluster(0, "Aerobic Training", ["aerobic exercise", "running"], None, 0),
            cluster_b=Cluster(1, "Resistance Training", ["weight lifting", "strength training"], None, 0),
            similarity=0.85,
            confidence_tier='MEDIUM'
        )

        prompt = validator._build_prompt(candidate)

        # Check key elements present
        assert "Cluster A" in prompt
        assert "Cluster B" in prompt
        assert "Aerobic Training" in prompt
        assert "Resistance Training" in prompt
        assert "MERGE_IDENTICAL" in prompt
        assert "CREATE_PARENT" in prompt
        assert "DIFFERENT" in prompt
        assert "0.850" in prompt  # Similarity

    def test_prompt_limits_members(self):
        """Test prompt limits member display to 5."""
        validator = LLMValidator()

        many_members = [f"member_{i}" for i in range(20)]
        candidate = MergeCandidate(
            cluster_a_id=0,
            cluster_b_id=1,
            cluster_a=Cluster(0, "Large Cluster", many_members, None, 0),
            cluster_b=Cluster(1, "Small Cluster", ["m1"], None, 0),
            similarity=0.90,
            confidence_tier='HIGH'
        )

        prompt = validator._build_prompt(candidate)

        # Should show "showing first 5" for cluster A
        assert "showing first 5" in prompt
        # Should not show member_10 (beyond first 5)
        assert "member_10" not in prompt


class TestFilteringResults:
    """Test filtering validation results."""

    def test_get_approved_only(self):
        """Test filtering to approved merges only."""
        validator = LLMValidator()

        # Create mock results
        from stage_3_llm_validation import LLMValidationResult

        approved_result = LLMValidationResult(
            candidate=None,
            relationship_type='CREATE_PARENT',
            llm_confidence='HIGH',
            suggested_parent_name='Parent',
            child_a_refined_name=None,
            child_b_refined_name=None,
            llm_reasoning='Good',
            name_quality=NameQualityScore(80, [], True),
            diversity_check=DiversityCheck(False, 'NONE', 0.75, 'Good'),
            auto_approved=True,
            flagged_reason=None
        )

        rejected_result = LLMValidationResult(
            candidate=None,
            relationship_type='DIFFERENT',
            llm_confidence='HIGH',
            suggested_parent_name=None,
            child_a_refined_name=None,
            child_b_refined_name=None,
            llm_reasoning='Different',
            name_quality=NameQualityScore(0, [], False),
            diversity_check=DiversityCheck(False, 'NONE', 0.0, ''),
            auto_approved=False,
            flagged_reason='DIFFERENT'
        )

        results = [approved_result, rejected_result]

        approved = validator.get_approved_merges(results)

        assert len(approved) == 1
        assert approved[0].auto_approved == True


def run_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING STAGE 3 TESTS")
    print("="*60)

    test_classes = [
        TestNameQualityScoring(),
        TestDiversityCheck(),
        TestAutoApprovalLogic(),
        TestPromptBuilding(),
        TestFilteringResults()
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")

        test_methods = [m for m in dir(test_class) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_class, method_name)
                method()
                print(f"  [PASS] {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  [FAIL] {method_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*60)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("="*60)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
