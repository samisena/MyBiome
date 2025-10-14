"""
Test Suite for Stage 1: Centroid Computation

Tests centroid computation, normalization, and validation.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from stage_1_centroid_computation import (
    compute_centroids,
    compute_single_centroid,
    compute_weighted_centroid,
    validate_centroids,
    compute_centroids_with_details
)
from validation_metrics import Cluster


class TestBasicCentroid:
    """Test basic centroid computation."""

    def test_two_vectors_orthogonal(self):
        """Test centroid of two orthogonal vectors."""
        cluster = Cluster(0, "test", ["m1", "m2"], None, 0)
        embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([0.0, 1.0, 0.0], dtype=np.float32)
        }

        centroid = compute_single_centroid(cluster, embeddings, embedding_dim=3)

        # Should be normalized average of [1,0,0] and [0,1,0]
        # Average: [0.5, 0.5, 0.0]
        # Normalized: [0.707, 0.707, 0.0]
        expected = np.array([0.707, 0.707, 0.0])
        assert np.allclose(centroid, expected, atol=0.01)

    def test_three_vectors_same_direction(self):
        """Test centroid of three vectors in same direction."""
        cluster = Cluster(0, "test", ["m1", "m2", "m3"], None, 0)
        embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([2.0, 0.0, 0.0], dtype=np.float32),
            'm3': np.array([3.0, 0.0, 0.0], dtype=np.float32)
        }

        centroid = compute_single_centroid(cluster, embeddings, embedding_dim=3)

        # Average: [2.0, 0.0, 0.0]
        # Normalized: [1.0, 0.0, 0.0]
        expected = np.array([1.0, 0.0, 0.0])
        assert np.allclose(centroid, expected, atol=0.01)

    def test_singleton_cluster(self):
        """Test centroid of single-member cluster."""
        cluster = Cluster(0, "test", ["m1"], None, 0)
        embeddings = {
            'm1': np.array([0.6, 0.8, 0.0], dtype=np.float32)
        }

        centroid = compute_single_centroid(cluster, embeddings, embedding_dim=3)

        # Should be normalized version of [0.6, 0.8, 0.0]
        expected = np.array([0.6, 0.8, 0.0])
        assert np.allclose(centroid, expected, atol=0.01)


class TestMissingEmbeddings:
    """Test handling of missing embeddings."""

    def test_all_missing(self):
        """Test cluster with all missing embeddings."""
        cluster = Cluster(0, "test", ["m1", "m2"], None, 0)
        embeddings = {}  # No embeddings

        centroid = compute_single_centroid(cluster, embeddings, embedding_dim=3)

        # Should return zero vector
        expected = np.array([0.0, 0.0, 0.0])
        assert np.allclose(centroid, expected, atol=0.01)

    def test_partial_missing(self):
        """Test cluster with some missing embeddings."""
        cluster = Cluster(0, "test", ["m1", "m2", "m3"], None, 0)
        embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            # m2 missing
            'm3': np.array([0.0, 1.0, 0.0], dtype=np.float32)
        }

        centroid = compute_single_centroid(cluster, embeddings, embedding_dim=3)

        # Should use only m1 and m3
        # Average: [0.5, 0.5, 0.0]
        # Normalized: [0.707, 0.707, 0.0]
        expected = np.array([0.707, 0.707, 0.0])
        assert np.allclose(centroid, expected, atol=0.01)


class TestNormalization:
    """Test normalization behavior."""

    def test_with_normalization(self):
        """Test centroid with normalization enabled."""
        cluster = Cluster(0, "test", ["m1", "m2"], None, 0)
        embeddings = {
            'm1': np.array([3.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([0.0, 4.0, 0.0], dtype=np.float32)
        }

        centroid = compute_single_centroid(cluster, embeddings, embedding_dim=3, normalize=True)

        # Should have norm ~1.0
        norm = np.linalg.norm(centroid)
        assert abs(norm - 1.0) < 0.01

    def test_without_normalization(self):
        """Test centroid with normalization disabled."""
        cluster = Cluster(0, "test", ["m1", "m2"], None, 0)
        embeddings = {
            'm1': np.array([3.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([0.0, 4.0, 0.0], dtype=np.float32)
        }

        centroid = compute_single_centroid(cluster, embeddings, embedding_dim=3, normalize=False)

        # Should be unnormalized average: [1.5, 2.0, 0.0]
        expected = np.array([1.5, 2.0, 0.0])
        assert np.allclose(centroid, expected, atol=0.01)


class TestWeightedCentroid:
    """Test weighted centroid computation."""

    def test_equal_weights(self):
        """Test weighted centroid with equal weights."""
        cluster = Cluster(0, "test", ["m1", "m2"], None, 0)
        embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([0.0, 1.0, 0.0], dtype=np.float32)
        }
        weights = {'m1': 1.0, 'm2': 1.0}

        centroid = compute_weighted_centroid(cluster, embeddings, weights, embedding_dim=3)

        # Should be same as unweighted
        expected = np.array([0.707, 0.707, 0.0])
        assert np.allclose(centroid, expected, atol=0.01)

    def test_different_weights(self):
        """Test weighted centroid with different weights."""
        cluster = Cluster(0, "test", ["m1", "m2"], None, 0)
        embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([0.0, 1.0, 0.0], dtype=np.float32)
        }
        weights = {'m1': 3.0, 'm2': 1.0}  # m1 weighted 3x more

        centroid = compute_weighted_centroid(cluster, embeddings, weights, embedding_dim=3)

        # Weighted average: 0.75*[1,0,0] + 0.25*[0,1,0] = [0.75, 0.25, 0.0]
        # Normalized: [0.948, 0.316, 0.0]
        expected = np.array([0.948, 0.316, 0.0])
        assert np.allclose(centroid, expected, atol=0.01)

    def test_default_weights(self):
        """Test weighted centroid with missing weights (default to 1.0)."""
        cluster = Cluster(0, "test", ["m1", "m2"], None, 0)
        embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([0.0, 1.0, 0.0], dtype=np.float32)
        }
        weights = {'m1': 2.0}  # m2 not specified, defaults to 1.0

        centroid = compute_weighted_centroid(cluster, embeddings, weights, embedding_dim=3)

        # Weighted average: 0.667*[1,0,0] + 0.333*[0,1,0]
        # Should be between [0.707, 0.707, 0] (equal) and [0.948, 0.316, 0] (3:1)
        assert centroid[0] > centroid[1]  # m1 should dominate


class TestBatchComputation:
    """Test computing centroids for multiple clusters."""

    def test_multiple_clusters(self):
        """Test computing centroids for multiple clusters."""
        clusters = [
            Cluster(0, "c0", ["m1", "m2"], None, 0),
            Cluster(1, "c1", ["m3"], None, 0),
            Cluster(2, "c2", ["m4", "m5", "m6"], None, 0)
        ]

        embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([0.9, 0.1, 0.0], dtype=np.float32),
            'm3': np.array([0.0, 1.0, 0.0], dtype=np.float32),
            'm4': np.array([0.0, 0.0, 1.0], dtype=np.float32),
            'm5': np.array([0.0, 0.1, 0.9], dtype=np.float32),
            'm6': np.array([0.1, 0.0, 0.9], dtype=np.float32)
        }

        centroids = compute_centroids(clusters, embeddings, embedding_dim=3)

        assert len(centroids) == 3
        assert 0 in centroids
        assert 1 in centroids
        assert 2 in centroids

        # Each should be normalized
        for centroid in centroids.values():
            norm = np.linalg.norm(centroid)
            assert abs(norm - 1.0) < 0.01 or abs(norm - 0.0) < 0.01  # Either normalized or zero


class TestValidation:
    """Test centroid validation."""

    def test_valid_centroids(self):
        """Test validation of valid centroids."""
        centroids = {
            0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            1: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            2: np.array([0.707, 0.707, 0.0], dtype=np.float32)
        }

        validation = validate_centroids(centroids, expected_dim=3)

        assert validation['valid'] == True
        assert len(validation['issues']) == 0

    def test_wrong_dimension(self):
        """Test validation catches wrong dimension."""
        centroids = {
            0: np.array([1.0, 0.0], dtype=np.float32),  # Wrong dimension
            1: np.array([0.0, 1.0, 0.0], dtype=np.float32)
        }

        validation = validate_centroids(centroids, expected_dim=3)

        assert validation['valid'] == False
        assert len(validation['issues']) > 0
        assert 'Wrong dimension' in validation['issues'][0]

    def test_nan_values(self):
        """Test validation catches NaN values."""
        centroids = {
            0: np.array([np.nan, 0.0, 0.0], dtype=np.float32)
        }

        validation = validate_centroids(centroids, expected_dim=3)

        assert validation['valid'] == False
        assert any('NaN' in issue for issue in validation['issues'])

    def test_unnormalized_centroids(self):
        """Test validation catches unnormalized centroids."""
        centroids = {
            0: np.array([5.0, 0.0, 0.0], dtype=np.float32)  # Norm = 5.0, not normalized
        }

        validation = validate_centroids(centroids, expected_dim=3)

        assert validation['valid'] == False
        assert any('Not normalized' in issue for issue in validation['issues'])


class TestDetailedComputation:
    """Test detailed computation with diagnostics."""

    def test_with_details(self):
        """Test detailed computation returns correct info."""
        clusters = [
            Cluster(0, "c0", ["m1", "m2", "m3"], None, 0),  # All present
            Cluster(1, "c1", ["m4", "m5"], None, 0),  # 1 missing
            Cluster(2, "c2", ["m6"], None, 0)  # All missing
        ]

        embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([0.9, 0.1, 0.0], dtype=np.float32),
            'm3': np.array([0.8, 0.2, 0.0], dtype=np.float32),
            'm4': np.array([0.0, 1.0, 0.0], dtype=np.float32)
            # m5, m6 missing
        }

        results = compute_centroids_with_details(clusters, embeddings, embedding_dim=3)

        assert len(results) == 3

        # Cluster 0: no missing
        assert results[0].cluster_id == 0
        assert results[0].member_count == 3
        assert results[0].missing_embeddings == 0
        assert results[0].normalization_applied == True

        # Cluster 1: 1 missing
        assert results[1].cluster_id == 1
        assert results[1].member_count == 2
        assert results[1].missing_embeddings == 1

        # Cluster 2: all missing (zero vector)
        assert results[2].cluster_id == 2
        assert results[2].member_count == 1
        assert results[2].missing_embeddings == 1
        assert results[2].normalization_applied == False
        assert np.allclose(results[2].centroid, np.zeros(3))


def run_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING STAGE 1 TESTS")
    print("="*60)

    test_classes = [
        TestBasicCentroid(),
        TestMissingEmbeddings(),
        TestNormalization(),
        TestWeightedCentroid(),
        TestBatchComputation(),
        TestValidation(),
        TestDetailedComputation()
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
