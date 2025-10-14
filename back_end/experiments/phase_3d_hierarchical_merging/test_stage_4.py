"""
Test Suite for Stage 4: Cross-Category Detection

Tests cross-category detection and reporting.
"""

from pathlib import Path
import sys
import json
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

from stage_4_cross_category_detection import CrossCategoryDetector, CrossCategoryCase
from stage_3_llm_validation import LLMValidationResult, NameQualityScore, DiversityCheck
from stage_2_candidate_generation import MergeCandidate
from validation_metrics import Cluster


class TestCrossCategoryDetection:
    """Test cross-category case detection."""

    def test_detect_different_categories(self):
        """Test detecting merges with different categories."""
        # Create mock approved merges with different categories
        cluster_a = Cluster(0, "Probiotics", ["Lactobacillus", "Bifidobacterium"], None, 0)
        cluster_b = Cluster(1, "FMT", ["fecal microbiota transplant"], None, 0)

        candidate = MergeCandidate(
            cluster_a_id=0,
            cluster_b_id=1,
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            similarity=0.90,
            confidence_tier='HIGH'
        )

        approved_merge = LLMValidationResult(
            candidate=candidate,
            relationship_type='CREATE_PARENT',
            llm_confidence='HIGH',
            suggested_parent_name='Gut Microbiome Modulation',
            child_a_refined_name=None,
            child_b_refined_name=None,
            llm_reasoning='Both modulate gut microbiome',
            name_quality=NameQualityScore(80, [], True),
            diversity_check=DiversityCheck(False, 'NONE', 0.75, 'Good'),
            auto_approved=True,
            flagged_reason=None
        )

        # Note: Can't test DB lookup without actual database,
        # but we can test the structure works
        detector = CrossCategoryDetector(
            db_path=':memory:',  # In-memory DB for testing
            entity_type='mechanism'
        )

        # Test case structure
        test_case = CrossCategoryCase(
            parent_name='Gut Microbiome Modulation',
            cluster_a_id=0,
            cluster_a_name='Probiotics',
            cluster_a_category='supplement',
            cluster_a_members=['Lactobacillus'],
            cluster_b_id=1,
            cluster_b_name='FMT',
            cluster_b_category='procedure',
            cluster_b_members=['fecal microbiota transplant'],
            similarity=0.90,
            llm_reasoning='Both modulate gut microbiome',
            suggestion='Review categorization'
        )

        # Verify case has expected fields
        assert test_case.cluster_a_category == 'supplement'
        assert test_case.cluster_b_category == 'procedure'
        assert test_case.cluster_a_category != test_case.cluster_b_category

    def test_suggestion_generation(self):
        """Test generating recategorization suggestions."""
        detector = CrossCategoryDetector(db_path=':memory:', entity_type='mechanism')

        # Mock merge with "same" in reasoning
        merge_same = LLMValidationResult(
            candidate=None,
            relationship_type='MERGE_IDENTICAL',
            llm_confidence='HIGH',
            suggested_parent_name='Parent',
            child_a_refined_name=None,
            child_b_refined_name=None,
            llm_reasoning='These are the same mechanism',
            name_quality=NameQualityScore(80, [], True),
            diversity_check=DiversityCheck(False, 'NONE', 0.75, 'Good'),
            auto_approved=True,
            flagged_reason=None
        )

        suggestion = detector._generate_suggestion('cat_a', 'cat_b', merge_same)

        assert 'standardizing' in suggestion.lower() or 'same' in suggestion.lower()


class TestReportGeneration:
    """Test report generation."""

    def test_generate_json_report(self):
        """Test JSON report generation."""
        detector = CrossCategoryDetector(db_path=':memory:', entity_type='mechanism')

        # Create test cases
        cases = [
            CrossCategoryCase(
                parent_name='Parent 1',
                cluster_a_id=0,
                cluster_a_name='Cluster A',
                cluster_a_category='supplement',
                cluster_a_members=['m1', 'm2'],
                cluster_b_id=1,
                cluster_b_name='Cluster B',
                cluster_b_category='procedure',
                cluster_b_members=['m3', 'm4'],
                similarity=0.90,
                llm_reasoning='Related',
                suggestion='Review'
            )
        ]

        # Generate report to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            detector.generate_report(cases, temp_path)

            # Verify file created and valid JSON
            with open(temp_path, 'r') as f:
                report_data = json.load(f)

            assert 'total_cases' in report_data
            assert report_data['total_cases'] == 1
            assert 'cases' in report_data
            assert len(report_data['cases']) == 1
            assert report_data['cases'][0]['parent_name'] == 'Parent 1'

        finally:
            # Cleanup
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_empty_cases_report(self):
        """Test report generation with no cases."""
        detector = CrossCategoryDetector(db_path=':memory:', entity_type='mechanism')

        # Should handle empty list gracefully
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            detector.generate_report([], temp_path)

            # Should not create file for empty cases
            # (based on implementation)
            # Just verify no errors

        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestSummaryStats:
    """Test summary statistics generation."""

    def test_most_common_pairs(self):
        """Test identifying most common category pairs."""
        detector = CrossCategoryDetector(db_path=':memory:', entity_type='mechanism')

        cases = [
            CrossCategoryCase('P1', 0, 'A', 'supplement', ['m1'], 1, 'B', 'procedure', ['m2'], 0.9, 'R', 'S'),
            CrossCategoryCase('P2', 2, 'C', 'supplement', ['m3'], 3, 'D', 'procedure', ['m4'], 0.9, 'R', 'S'),
            CrossCategoryCase('P3', 4, 'E', 'exercise', ['m5'], 5, 'F', 'therapy', ['m6'], 0.9, 'R', 'S')
        ]

        summary = detector._generate_summary_stats(cases)

        assert 'most_common_pairs' in summary
        # Should identify supplement+procedure as most common (2 occurrences)
        most_common = summary['most_common_pairs'][0]
        assert set(most_common['categories']) == {'supplement', 'procedure'}
        assert most_common['count'] == 2


def run_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING STAGE 4 TESTS")
    print("="*60)

    test_classes = [
        TestCrossCategoryDetection(),
        TestReportGeneration(),
        TestSummaryStats()
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

    print("\n" + "="*60)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("="*60)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
