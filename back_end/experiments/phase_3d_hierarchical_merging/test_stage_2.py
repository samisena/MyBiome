"""
Test Suite for Stage 2: Candidate Generation

Tests candidate finding, filtering, and ranking.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from stage_2_candidate_generation import (
    generate_merge_candidates,
    get_confidence_tier,
    filter_candidates_by_tier,
    filter_candidates_by_size_imbalance,
    remove_duplicate_pairs,
    group_candidates_by_cluster
)
from validation_metrics import Cluster


class TestConfidenceTiers:
    """Test confidence tier classification."""

    def test_high_tier(self):
        """Test HIGH tier (>= 0.90)."""
        assert get_confidence_tier(0.95) == 'HIGH'
        assert get_confidence_tier(0.90) == 'HIGH'

    def test_medium_tier(self):
        """Test MEDIUM tier (0.85-0.90)."""
        assert get_confidence_tier(0.89) == 'MEDIUM'
        assert get_confidence_tier(0.85) == 'MEDIUM'

    def test_low_tier(self):
        """Test LOW tier (< 0.85)."""
        assert get_confidence_tier(0.84) == 'LOW'
        assert get_confidence_tier(0.70) == 'LOW'


class TestCandidateGeneration:
    """Test candidate generation."""

    def test_basic_generation(self):
        """Test generating candidates from clusters."""
        clusters = [
            Cluster(0, "c0", ["m1"], None, 0),
            Cluster(1, "c1", ["m2"], None, 0),
            Cluster(2, "c2", ["m3"], None, 0)
        ]

        # Make 0 and 1 similar, 2 different
        centroids = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.95, 0.05, 0.0], dtype=np.float32),  # Very similar
            2: np.array([0.0, 1.0, 0.0], dtype=np.float32)  # Different
        }

        candidates = generate_merge_candidates(clusters, centroids, similarity_threshold=0.85)

        # Should find 0-1 pair (similarity ~0.95)
        assert len(candidates) >= 1
        assert candidates[0].cluster_a_id in [0, 1]
        assert candidates[0].cluster_b_id in [0, 1]
        assert candidates[0].similarity >= 0.85

    def test_threshold_filtering(self):
        """Test similarity threshold filtering."""
        clusters = [
            Cluster(0, "c0", ["m1"], None, 0),
            Cluster(1, "c1", ["m2"], None, 0)
        ]

        centroids = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.80, 0.60, 0.0], dtype=np.float32)  # Sim ~0.80
        }

        # With threshold 0.85, should find nothing
        high_threshold = generate_merge_candidates(clusters, centroids, similarity_threshold=0.85)
        assert len(high_threshold) == 0

        # With threshold 0.75, should find the pair
        low_threshold = generate_merge_candidates(clusters, centroids, similarity_threshold=0.75)
        assert len(low_threshold) == 1

    def test_sorted_by_similarity(self):
        """Test candidates are sorted by similarity (descending)."""
        clusters = [
            Cluster(0, "c0", ["m1"], None, 0),
            Cluster(1, "c1", ["m2"], None, 0),
            Cluster(2, "c2", ["m3"], None, 0)
        ]

        centroids = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.95, 0.05, 0.0], dtype=np.float32),  # Sim ~0.95
            2: np.array([0.90, 0.10, 0.0], dtype=np.float32)  # Sim ~0.90
        }

        candidates = generate_merge_candidates(clusters, centroids, similarity_threshold=0.85)

        # Should have 2-3 candidates, sorted descending
        assert len(candidates) >= 2
        for i in range(len(candidates) - 1):
            assert candidates[i].similarity >= candidates[i+1].similarity

    def test_max_candidates_limit(self):
        """Test max_candidates parameter limits results."""
        clusters = [Cluster(i, f"c{i}", [f"m{i}"], None, 0) for i in range(10)]

        # Make all similar
        centroids = {i: np.array([1.0, 0.0, 0.0], dtype=np.float32) for i in range(10)}

        # Generate with limit
        candidates = generate_merge_candidates(
            clusters, centroids,
            similarity_threshold=0.90,
            max_candidates=5
        )

        assert len(candidates) == 5


class TestFilteringByTier:
    """Test tier-based filtering."""

    def test_filter_high_only(self):
        """Test filtering to HIGH tier only."""
        clusters = [
            Cluster(0, "c0", ["m1"], None, 0),
            Cluster(1, "c1", ["m2"], None, 0),
            Cluster(2, "c2", ["m3"], None, 0),
            Cluster(3, "c3", ["m4"], None, 0)
        ]

        centroids = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.95, 0.05, 0.0], dtype=np.float32),  # HIGH
            2: np.array([0.88, 0.12, 0.0], dtype=np.float32),  # MEDIUM
            3: np.array([0.80, 0.20, 0.0], dtype=np.float32)  # LOW
        }

        all_candidates = generate_merge_candidates(clusters, centroids, similarity_threshold=0.75)

        high_only = filter_candidates_by_tier(all_candidates, min_tier='HIGH')

        # Should only keep HIGH tier candidates
        assert all(c.confidence_tier == 'HIGH' for c in high_only)

    def test_filter_medium_and_above(self):
        """Test filtering to MEDIUM and HIGH tiers."""
        clusters = [Cluster(i, f"c{i}", [f"m{i}"], None, 0) for i in range(4)]

        centroids = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.95, 0.05, 0.0], dtype=np.float32),  # HIGH
            2: np.array([0.88, 0.12, 0.0], dtype=np.float32),  # MEDIUM
            3: np.array([0.80, 0.20, 0.0], dtype=np.float32)  # LOW
        }

        all_candidates = generate_merge_candidates(clusters, centroids, similarity_threshold=0.75)

        medium_and_above = filter_candidates_by_tier(all_candidates, min_tier='MEDIUM')

        # Should keep HIGH and MEDIUM, exclude LOW
        assert all(c.confidence_tier in ['HIGH', 'MEDIUM'] for c in medium_and_above)


class TestSizeImbalanceFiltering:
    """Test size imbalance filtering."""

    def test_filter_large_imbalance(self):
        """Test filtering out large size imbalances."""
        clusters = [
            Cluster(0, "c0", ["m1"], None, 0),  # Size 1
            Cluster(1, "c1", [f"m{i}" for i in range(2, 52)], None, 0)  # Size 50
        ]

        centroids = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.95, 0.05, 0.0], dtype=np.float32)
        }

        candidates = generate_merge_candidates(clusters, centroids, similarity_threshold=0.85)

        # Size ratio = 50/1 = 50
        # Filter with max_size_ratio=10 should remove this pair
        filtered = filter_candidates_by_size_imbalance(candidates, max_size_ratio=10.0)

        assert len(filtered) == 0

    def test_keep_balanced_sizes(self):
        """Test keeping balanced size pairs."""
        clusters = [
            Cluster(0, "c0", ["m1", "m2", "m3"], None, 0),  # Size 3
            Cluster(1, "c1", ["m4", "m5"], None, 0)  # Size 2
        ]

        centroids = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.95, 0.05, 0.0], dtype=np.float32)
        }

        candidates = generate_merge_candidates(clusters, centroids, similarity_threshold=0.85)

        # Size ratio = 3/2 = 1.5, should be kept
        filtered = filter_candidates_by_size_imbalance(candidates, max_size_ratio=10.0)

        assert len(filtered) == len(candidates)


class TestDuplicateRemoval:
    """Test duplicate pair removal."""

    def test_remove_duplicates(self):
        """Test removing duplicate pairs."""
        # This test manually creates candidates (not via generation)
        # since generation shouldn't create duplicates in the first place

        cluster_0 = Cluster(0, "c0", ["m1"], None, 0)
        cluster_1 = Cluster(1, "c1", ["m2"], None, 0)

        from stage_2_candidate_generation import MergeCandidate

        candidates = [
            MergeCandidate(0, 1, cluster_0, cluster_1, 0.90, 'HIGH'),
            MergeCandidate(1, 0, cluster_1, cluster_0, 0.90, 'HIGH'),  # Duplicate (reversed)
        ]

        deduplicated = remove_duplicate_pairs(candidates)

        assert len(deduplicated) == 1


class TestCandidateGrouping:
    """Test grouping candidates by cluster."""

    def test_group_by_cluster(self):
        """Test grouping candidates by involved clusters."""
        clusters = [
            Cluster(0, "c0", ["m1"], None, 0),
            Cluster(1, "c1", ["m2"], None, 0),
            Cluster(2, "c2", ["m3"], None, 0)
        ]

        # Make 0-1 and 0-2 similar (0 is connected to both)
        centroids = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.95, 0.05, 0.0], dtype=np.float32),
            2: np.array([0.90, 0.10, 0.0], dtype=np.float32)
        }

        candidates = generate_merge_candidates(clusters, centroids, similarity_threshold=0.85)

        groups = group_candidates_by_cluster(candidates)

        # Cluster 0 should appear in 2 candidates (0-1, 0-2)
        assert 0 in groups
        assert len(groups[0]) >= 2


def run_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING STAGE 2 TESTS")
    print("="*60)

    test_classes = [
        TestConfidenceTiers(),
        TestCandidateGeneration(),
        TestFilteringByTier(),
        TestSizeImbalanceFiltering(),
        TestDuplicateRemoval(),
        TestCandidateGrouping()
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
