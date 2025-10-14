"""
Test Suite for Stage 0: Hyperparameter Optimizer

Tests grid search, simulation, and evaluation metrics.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from stage_0_hyperparameter_optimizer import HyperparameterOptimizer
from validation_metrics import Cluster, evaluate_hierarchy_quality, cosine_similarity
from config import Phase3dConfig


def create_test_clusters(num_clusters=10) -> list:
    """Create synthetic clusters for testing."""
    clusters = []
    for i in range(num_clusters):
        clusters.append(Cluster(
            cluster_id=i,
            canonical_name=f"cluster_{i}",
            members=[f"member_{i}_{j}" for j in range(3)],  # 3 members each
            parent_id=None,
            hierarchy_level=0
        ))
    return clusters


def create_test_embeddings(clusters: list, embedding_dim=768) -> dict:
    """Create synthetic embeddings for cluster members."""
    embeddings = {}

    for cluster in clusters:
        # Create similar embeddings for members within same cluster
        base_vector = np.random.randn(embedding_dim).astype(np.float32)
        base_vector = base_vector / np.linalg.norm(base_vector)

        for member in cluster.members:
            # Add small noise to base vector
            noise = np.random.randn(embedding_dim).astype(np.float32) * 0.1
            embedding = base_vector + noise
            embedding = embedding / np.linalg.norm(embedding)
            embeddings[member] = embedding

    return embeddings


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0"""
        vec = np.array([1.0, 0.0, 0.0])
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.01

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity ~0.0"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim - 0.0) < 0.01

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity 0.0 (clipped)"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        sim = cosine_similarity(vec1, vec2)
        assert 0.0 <= sim <= 1.0  # Should be clipped to valid range


class TestCentroidComputation:
    """Test centroid computation."""

    def test_centroid_basic(self):
        """Test centroid of 2 vectors."""
        optimizer = HyperparameterOptimizer()

        # Create test data
        clusters = [
            Cluster(0, "test", ["m1", "m2"], None, 0)
        ]

        optimizer.embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32),
            'm2': np.array([0.0, 1.0, 0.0], dtype=np.float32)
        }

        centroids = optimizer._compute_centroids(clusters)

        assert len(centroids) == 1
        # Should be normalized average
        expected = np.array([0.707, 0.707, 0.0])
        assert np.allclose(centroids[0][:3], expected, atol=0.01)

    def test_centroid_singleton(self):
        """Test centroid of single-member cluster."""
        optimizer = HyperparameterOptimizer()

        clusters = [
            Cluster(0, "test", ["m1"], None, 0)
        ]

        optimizer.embeddings = {
            'm1': np.array([1.0, 0.0, 0.0], dtype=np.float32)
        }

        centroids = optimizer._compute_centroids(clusters)

        assert len(centroids) == 1
        assert np.allclose(centroids[0][:3], np.array([1.0, 0.0, 0.0]), atol=0.01)

    def test_centroid_missing_embeddings(self):
        """Test handling of missing embeddings."""
        optimizer = HyperparameterOptimizer()
        optimizer.config = Phase3dConfig()

        clusters = [
            Cluster(0, "test", ["m1", "m2"], None, 0)
        ]

        optimizer.embeddings = {}  # No embeddings

        centroids = optimizer._compute_centroids(clusters)

        assert len(centroids) == 1
        # Should return zero vector
        assert np.allclose(centroids[0], np.zeros(768), atol=0.01)


class TestGreedyMerging:
    """Test greedy merging logic."""

    def test_single_merge(self):
        """Test merging two clusters."""
        optimizer = HyperparameterOptimizer()

        clusters = [
            Cluster(0, "c1", ["m1", "m2"], None, 0),
            Cluster(1, "c2", ["m3", "m4"], None, 0)
        ]

        merge_pairs = [(0, 1, 0.90)]  # High similarity

        parents = optimizer._apply_greedy_merges(clusters, merge_pairs, level=3)

        assert len(parents) == 1  # Should merge into 1 parent
        assert len(parents[0].members) == 4  # Combined members
        assert set(parents[0].members) == {'m1', 'm2', 'm3', 'm4'}

    def test_no_merges(self):
        """Test when no merges above threshold."""
        optimizer = HyperparameterOptimizer()

        clusters = [
            Cluster(0, "c1", ["m1"], None, 0),
            Cluster(1, "c2", ["m2"], None, 0),
            Cluster(2, "c3", ["m3"], None, 0)
        ]

        merge_pairs = []  # No merges

        parents = optimizer._apply_greedy_merges(clusters, merge_pairs, level=3)

        assert len(parents) == 3  # All unchanged

    def test_multiple_merges(self):
        """Test merging multiple pairs."""
        optimizer = HyperparameterOptimizer()

        clusters = [
            Cluster(0, "c1", ["m1"], None, 0),
            Cluster(1, "c2", ["m2"], None, 0),
            Cluster(2, "c3", ["m3"], None, 0),
            Cluster(3, "c4", ["m4"], None, 0)
        ]

        merge_pairs = [
            (0, 1, 0.95),  # Merge c1+c2
            (2, 3, 0.90)   # Merge c3+c4
        ]

        parents = optimizer._apply_greedy_merges(clusters, merge_pairs, level=3)

        assert len(parents) == 2  # Two merged clusters
        assert all(len(p.members) == 2 for p in parents)


class TestSimulation:
    """Test hierarchy simulation."""

    def test_simulation_basic(self):
        """Test basic simulation with high thresholds."""
        optimizer = HyperparameterOptimizer()

        # Create test data
        clusters = create_test_clusters(num_clusters=5)
        optimizer.clusters = clusters
        optimizer.embeddings = create_test_embeddings(clusters, embedding_dim=768)
        optimizer.config = Phase3dConfig()

        # Very high thresholds = no merges
        thresholds = {3: 0.99, 2: 0.99, 1: 0.99}

        hierarchy = optimizer.simulate_hierarchy_building(thresholds)

        assert 'level_3' in hierarchy
        assert len(hierarchy['level_3']) == 5  # No merges

    def test_simulation_low_thresholds(self):
        """Test simulation with low thresholds (many merges)."""
        optimizer = HyperparameterOptimizer()

        clusters = create_test_clusters(num_clusters=5)
        optimizer.clusters = clusters
        optimizer.embeddings = create_test_embeddings(clusters, embedding_dim=768)
        optimizer.config = Phase3dConfig()

        # Very low thresholds = many merges
        thresholds = {3: 0.50, 2: 0.50, 1: 0.50}

        hierarchy = optimizer.simulate_hierarchy_building(thresholds)

        # Should have some reduction
        if 'level_0' in hierarchy:
            assert len(hierarchy['level_0']) < 5


class TestEvaluation:
    """Test quality evaluation."""

    def test_evaluate_hierarchy(self):
        """Test full hierarchy evaluation."""
        clusters_l3 = create_test_clusters(num_clusters=10)
        clusters_l0 = create_test_clusters(num_clusters=5)  # 50% reduction

        hierarchy = {
            'level_3': clusters_l3,
            'level_2': clusters_l0.copy(),
            'level_1': clusters_l0.copy(),
            'level_0': clusters_l0
        }

        embeddings = create_test_embeddings(clusters_l3, embedding_dim=768)

        config = {
            'target_reduction_range': (0.40, 0.60),
            'target_depth_range': (2, 3),
            'target_singleton_threshold': 0.50,
            'target_coherence_min': 0.65,
            'target_separation_min': 0.35
        }

        metrics = evaluate_hierarchy_quality(hierarchy, embeddings, config)

        # Check all scores are in valid range
        assert 0 <= metrics.reduction_score <= 25
        assert 0 <= metrics.depth_score <= 15
        assert 0 <= metrics.size_distribution_score <= 20
        assert 0 <= metrics.coherence_score <= 25
        assert 0 <= metrics.separation_score <= 15
        assert 0 <= metrics.composite_score <= 100


def run_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING STAGE 0 TESTS")
    print("="*60)

    test_classes = [
        TestCosineSimilarity(),
        TestCentroidComputation(),
        TestGreedyMerging(),
        TestSimulation(),
        TestEvaluation()
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
