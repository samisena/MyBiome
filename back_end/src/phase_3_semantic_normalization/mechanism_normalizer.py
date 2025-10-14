"""
Mechanism Semantic Normalization & Clustering

Implements data-driven mechanism taxonomy using embedding-based clustering (HDBSCAN).
Discovers emergent categories from mechanism texts, supports incremental updates,
and enables mechanism-based intervention discovery.

Key Features:
- Embedding-based clustering with nomic-embed-text (768-dim)
- HDBSCAN for density-based cluster discovery
- Hierarchical 2-level taxonomy (parent mechanisms → sub-mechanisms)
- Incremental mechanism assignment for continuous learning
- Re-clustering triggers based on data growth and quality metrics
- Comprehensive validation with quick feedback loops
- Cross-entity tracking (mechanism → condition, intervention → mechanism)

Architecture:
- Reuses 70% of existing semantic normalization infrastructure
- Integrates with Phase 3.6 of batch_medical_rotation pipeline
- Supports continuous mode with iteration tracking

Usage:
    from mechanism_normalizer import MechanismNormalizer

    normalizer = MechanismNormalizer(db_path=config.db_path)

    # Initial clustering
    result = normalizer.cluster_all_mechanisms()

    # Incremental assignment
    cluster_id = normalizer.assign_mechanism("gut microbiome modulation")

    # Re-clustering (if needed)
    if normalizer.should_recluster():
        result = normalizer.recluster_all_mechanisms()
"""

import os
import re
import json
import pickle
import logging
import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

# Third-party imports
try:
    import hdbscan
    from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("hdbscan or sklearn not available - install with: pip install hdbscan scikit-learn")

# Import existing semantic normalization components
from .phase_3_embedding_engine import EmbeddingEngine
from .phase_3_llm_classifier import LLMClassifier
from .phase_3_hierarchy_manager import HierarchyManager
from .mechanism_canonical_extractor import MechanismCanonicalExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClusterMetrics:
    """Validation metrics for clustering quality."""
    silhouette_score: float
    davies_bouldin_index: float
    num_clusters: int
    singleton_count: int
    singleton_percentage: float
    avg_cluster_size: float
    manual_coherence: Optional[float] = None

    def passes_thresholds(self) -> bool:
        """Check if metrics pass quality thresholds."""
        checks = {
            'silhouette': self.silhouette_score > 0.35,
            'singleton_pct': self.singleton_percentage < 0.20,
            'num_clusters': 10 <= self.num_clusters <= 40
        }
        return all(checks.values())

    def summary(self) -> str:
        """Generate 1-line validation summary."""
        status = "PASS" if self.passes_thresholds() else "WARN"
        return f"{status}: {self.num_clusters} clusters, sil={self.silhouette_score:.2f}, singleton={self.singleton_percentage:.1%}"


@dataclass
class ClusteringResult:
    """Result of clustering operation."""
    success: bool
    num_clusters: int
    num_mechanisms: int
    metrics: ClusterMetrics
    cluster_assignments: Dict[str, int]  # mechanism_text -> cluster_id
    canonical_names: Dict[int, str]  # cluster_id -> canonical_name
    cluster_members: Dict[int, List[str]]  # cluster_id -> [mechanism_texts]
    hierarchies: Dict[int, List[int]] = field(default_factory=dict)  # parent_id -> [child_ids]
    elapsed_time_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class IncrementalAssignmentResult:
    """Result of incremental mechanism assignment."""
    mechanism_text: str
    cluster_id: Optional[int]
    assignment_type: str  # 'primary', 'multi_label', 'singleton'
    similarity_score: float
    multi_label_clusters: Optional[List[int]] = None


class ValidationCheckpoint:
    """Quick validation with 1-2 line summaries."""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = datetime.now()

    def validate_and_report(self, metrics: Dict[str, float], thresholds: Dict[str, float]) -> str:
        """Compute pass/fail and generate summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        # Check thresholds
        passed = all(metrics[k] >= v for k, v in thresholds.items() if k in metrics)
        status = "✓ PASS" if passed else "✗ FAIL"

        # Generate 1-line summary
        metric_str = ", ".join([f"{k}={v:.2f}" for k, v in metrics.items()])
        summary = f"{status} | {self.step_name} | {elapsed:.1f}s | {metric_str}"

        logger.info(summary)
        return status


class MechanismNormalizer:
    """
    Main orchestrator for mechanism semantic normalization and clustering.

    Implements Approach 2 (Embedding + Clustering) with:
    - HDBSCAN density-based clustering
    - Hierarchical 2-level taxonomy
    - Incremental updates with re-clustering triggers
    - Comprehensive validation checkpoints
    """

    def __init__(
        self,
        db_path: str,
        cache_dir: Optional[str] = None,
        # HDBSCAN parameters
        min_cluster_size: int = 5,
        min_samples: int = 3,
        cluster_selection_epsilon: float = 0.0,
        # Incremental assignment thresholds
        primary_threshold: float = 0.75,
        secondary_threshold: float = 0.60,
        # Re-clustering triggers
        recluster_iteration_interval: int = 5,
        recluster_singleton_threshold: float = 0.10,
        recluster_growth_threshold: float = 0.30
    ):
        """
        Initialize mechanism normalizer.

        Args:
            db_path: Path to intervention_research.db
            cache_dir: Cache directory for embeddings and LLM decisions
            min_cluster_size: HDBSCAN minimum cluster size
            min_samples: HDBSCAN minimum samples
            cluster_selection_epsilon: HDBSCAN cluster selection epsilon
            primary_threshold: Similarity threshold for single cluster assignment
            secondary_threshold: Similarity threshold for multi-label assignment
            recluster_iteration_interval: Re-cluster every N iterations
            recluster_singleton_threshold: Re-cluster if singleton rate > threshold
            recluster_growth_threshold: Re-cluster if data growth > threshold
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("hdbscan and scikit-learn required - install with: pip install hdbscan scikit-learn")

        self.db_path = db_path

        # Setup cache directory
        if cache_dir is None:
            from .config import CACHE_DIR
            cache_dir = CACHE_DIR
        else:
            cache_dir = Path(cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components (reuse existing infrastructure)
        self.embedding_engine = EmbeddingEngine(
            cache_path=str(cache_dir / "mechanism_embeddings.pkl")
        )
        self.llm_classifier = LLMClassifier(
            canonical_cache_path=str(cache_dir / "mechanism_canonicals.pkl"),
            relationship_cache_path=str(cache_dir / "mechanism_relationships.pkl")
        )
        self.canonical_extractor = MechanismCanonicalExtractor(
            llm_classifier=self.llm_classifier,
            cache_dir=cache_dir
        )

        # HDBSCAN parameters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

        # Incremental assignment thresholds
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold

        # Re-clustering triggers
        self.recluster_iteration_interval = recluster_iteration_interval
        self.recluster_singleton_threshold = recluster_singleton_threshold
        self.recluster_growth_threshold = recluster_growth_threshold

        # State
        self.current_iteration = 0
        self.last_cluster_size = 0
        self.cluster_centroids: Dict[int, np.ndarray] = {}

        # Session file for resumability
        self.session_file = cache_dir / "mechanism_normalizer_session.pkl"
        self._load_session()

        logger.info("MechanismNormalizer initialized")
        logger.info(f"  HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={cluster_selection_epsilon}")
        logger.info(f"  Thresholds: primary={primary_threshold}, secondary={secondary_threshold}")

    def _load_session(self):
        """Load session state from file."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'rb') as f:
                    state = pickle.load(f)
                self.current_iteration = state.get('current_iteration', 0)
                self.last_cluster_size = state.get('last_cluster_size', 0)
                self.cluster_centroids = state.get('cluster_centroids', {})
                logger.info(f"Loaded session: iteration={self.current_iteration}, last_size={self.last_cluster_size}")
            except Exception as e:
                logger.warning(f"Failed to load session: {e}")

    def _save_session(self):
        """Save session state to file."""
        try:
            state = {
                'current_iteration': self.current_iteration,
                'last_cluster_size': self.last_cluster_size,
                'cluster_centroids': self.cluster_centroids,
                'last_save': datetime.now().isoformat()
            }
            with open(self.session_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def load_mechanisms_from_db(self) -> List[str]:
        """
        Load unique mechanism texts from interventions table.

        Returns:
            List of unique mechanism texts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if interventions table exists and has mechanism column
        cursor.execute("""
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type='table' AND name='interventions'
        """)

        if cursor.fetchone()[0] == 0:
            logger.warning("interventions table does not exist yet")
            conn.close()
            return []

        # Check if mechanism column exists
        cursor.execute("PRAGMA table_info(interventions)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'mechanism' not in columns:
            logger.warning("mechanism column does not exist in interventions table")
            conn.close()
            return []

        # Load mechanisms
        cursor.execute("""
            SELECT DISTINCT mechanism
            FROM interventions
            WHERE mechanism IS NOT NULL
              AND mechanism != ''
            ORDER BY mechanism
        """)

        mechanisms = [row[0] for row in cursor.fetchall()]
        conn.close()

        logger.info(f"Loaded {len(mechanisms)} unique mechanisms from database")
        return mechanisms

    def _normalize_mechanism_text(self, text: str) -> str:
        """Normalize mechanism text (lowercase, strip whitespace)."""
        return ' '.join(text.lower().strip().split())

    def _compute_cluster_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> ClusterMetrics:
        """
        Compute clustering quality metrics.

        Args:
            embeddings: Mechanism embeddings (N × 768)
            labels: Cluster labels (N,)

        Returns:
            ClusterMetrics object
        """
        # Filter out noise points (-1) for metrics computation
        valid_mask = labels != -1
        valid_embeddings = embeddings[valid_mask]
        valid_labels = labels[valid_mask]

        # Compute metrics
        if len(valid_embeddings) > 1 and len(set(valid_labels)) > 1:
            silhouette = silhouette_score(valid_embeddings, valid_labels)
            davies_bouldin = davies_bouldin_score(valid_embeddings, valid_labels)
        else:
            silhouette = 0.0
            davies_bouldin = float('inf')

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        singleton_count = sum(labels == -1)
        singleton_pct = singleton_count / len(labels) if len(labels) > 0 else 0.0

        # Average cluster size
        cluster_sizes = [sum(labels == i) for i in set(labels) if i != -1]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0

        return ClusterMetrics(
            silhouette_score=silhouette,
            davies_bouldin_index=davies_bouldin,
            num_clusters=num_clusters,
            singleton_count=singleton_count,
            singleton_percentage=singleton_pct,
            avg_cluster_size=avg_cluster_size
        )

    def _run_hdbscan(
        self,
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
        """
        Run HDBSCAN clustering on embeddings.

        Args:
            embeddings: Mechanism embeddings (N × 768)

        Returns:
            Tuple of (labels, clusterer)
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        labels = clusterer.fit_predict(embeddings)

        return labels, clusterer

    def _extract_canonical_names(
        self,
        cluster_members: Dict[int, List[str]]
    ) -> Dict[int, str]:
        """
        Extract canonical names for clusters using LLM.

        Args:
            cluster_members: Dict mapping cluster_id to list of member mechanisms

        Returns:
            Dict mapping cluster_id to canonical_name
        """
        checkpoint = ValidationCheckpoint("Canonical Name Extraction")

        # Use canonical extractor (LLM-based)
        extraction_results = self.canonical_extractor.extract_batch(cluster_members)

        # Convert results to simple dict
        canonical_names = {
            cluster_id: result.canonical_name
            for cluster_id, result in extraction_results.items()
        }

        # Validation checkpoint
        metrics = {
            'clusters_named': len(canonical_names),
        }
        thresholds = {'clusters_named': 1}
        checkpoint.validate_and_report(metrics, thresholds)

        logger.info(f"Extracted {len(canonical_names)} canonical names using LLM")

        return canonical_names

    def _compute_cluster_centroids(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        mechanisms: List[str]
    ) -> Dict[int, np.ndarray]:
        """
        Compute cluster centroids for incremental assignment.

        Args:
            embeddings: Mechanism embeddings (N × 768)
            labels: Cluster labels (N,)
            mechanisms: Mechanism texts (for indexing)

        Returns:
            Dict mapping cluster_id to centroid vector
        """
        centroids = {}

        for cluster_id in set(labels):
            if cluster_id == -1:
                continue  # Skip noise

            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            centroid = cluster_embeddings.mean(axis=0)
            centroids[cluster_id] = centroid

        logger.info(f"Computed {len(centroids)} cluster centroids")
        return centroids

    def cluster_all_mechanisms(
        self,
        mechanisms: Optional[List[str]] = None
    ) -> ClusteringResult:
        """
        Run initial clustering on all mechanisms.

        Args:
            mechanisms: Optional list of mechanism texts (loads from DB if None)

        Returns:
            ClusteringResult with metrics and assignments
        """
        start_time = datetime.now()

        # Load mechanisms
        if mechanisms is None:
            mechanisms = self.load_mechanisms_from_db()

        if not mechanisms:
            return ClusteringResult(
                success=False,
                num_clusters=0,
                num_mechanisms=0,
                metrics=ClusterMetrics(0, 0, 0, 0, 0, 0),
                cluster_assignments={},
                canonical_names={},
                cluster_members={},
                error="No mechanisms found in database"
            )

        logger.info(f"Clustering {len(mechanisms)} mechanisms...")

        # Step 1: Generate embeddings
        checkpoint = ValidationCheckpoint("Embedding Generation")
        embeddings_list = self.embedding_engine.generate_embeddings_batch(mechanisms)
        embeddings = np.array(embeddings_list)

        checkpoint.validate_and_report(
            {'embeddings_generated': len(embeddings)},
            {'embeddings_generated': 1}
        )

        # Step 2: Run HDBSCAN clustering
        checkpoint = ValidationCheckpoint("HDBSCAN Clustering")
        labels, clusterer = self._run_hdbscan(embeddings)

        # Compute metrics
        metrics = self._compute_cluster_metrics(embeddings, labels)
        logger.info(metrics.summary())

        checkpoint.validate_and_report(
            {
                'silhouette': metrics.silhouette_score,
                'num_clusters': metrics.num_clusters,
                'singleton_pct': metrics.singleton_percentage
            },
            {'silhouette': 0.35, 'num_clusters': 10, 'singleton_pct': 0.0}
        )

        # Step 3: Build cluster assignments and members
        cluster_assignments = {mech: int(label) for mech, label in zip(mechanisms, labels)}

        cluster_members = defaultdict(list)
        for mech, label in zip(mechanisms, labels):
            cluster_members[int(label)].append(mech)

        # Step 4: Extract canonical names
        canonical_names = self._extract_canonical_names(dict(cluster_members))

        # Step 5: Compute cluster centroids for incremental assignment
        self.cluster_centroids = self._compute_cluster_centroids(embeddings, labels, mechanisms)

        # Update state
        self.last_cluster_size = len(mechanisms)
        self._save_session()

        elapsed_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Clustering completed in {elapsed_time:.1f}s")
        logger.info(f"  Mechanisms: {len(mechanisms)}")
        logger.info(f"  Clusters: {metrics.num_clusters}")
        logger.info(f"  Singletons: {metrics.singleton_count} ({metrics.singleton_percentage:.1%})")
        logger.info(f"  Silhouette: {metrics.silhouette_score:.3f}")

        return ClusteringResult(
            success=True,
            num_clusters=metrics.num_clusters,
            num_mechanisms=len(mechanisms),
            metrics=metrics,
            cluster_assignments=cluster_assignments,
            canonical_names=canonical_names,
            cluster_members=dict(cluster_members),
            elapsed_time_seconds=elapsed_time
        )

    def assign_mechanism(
        self,
        mechanism_text: str
    ) -> IncrementalAssignmentResult:
        """
        Assign new mechanism to existing clusters (incremental).

        Args:
            mechanism_text: New mechanism to assign

        Returns:
            IncrementalAssignmentResult with assignment details
        """
        # Generate embedding
        embedding = self.embedding_engine.generate_embedding(mechanism_text)

        if not self.cluster_centroids:
            # No clusters yet, create singleton
            return IncrementalAssignmentResult(
                mechanism_text=mechanism_text,
                cluster_id=-1,
                assignment_type='singleton',
                similarity_score=0.0
            )

        # Find nearest cluster centroid
        max_similarity = 0.0
        nearest_cluster = -1

        for cluster_id, centroid in self.cluster_centroids.items():
            similarity = self.embedding_engine.cosine_similarity(embedding, centroid)
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_cluster = cluster_id

        # Determine assignment type based on thresholds
        if max_similarity >= self.primary_threshold:
            # Primary assignment (single cluster)
            return IncrementalAssignmentResult(
                mechanism_text=mechanism_text,
                cluster_id=nearest_cluster,
                assignment_type='primary',
                similarity_score=max_similarity
            )

        elif max_similarity >= self.secondary_threshold:
            # Multi-label candidate (find top 3 clusters)
            similarities = []
            for cluster_id, centroid in self.cluster_centroids.items():
                similarity = self.embedding_engine.cosine_similarity(embedding, centroid)
                if similarity >= self.secondary_threshold:
                    similarities.append((cluster_id, similarity))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_clusters = [cid for cid, sim in similarities[:3]]

            return IncrementalAssignmentResult(
                mechanism_text=mechanism_text,
                cluster_id=top_clusters[0] if top_clusters else -1,
                assignment_type='multi_label',
                similarity_score=max_similarity,
                multi_label_clusters=top_clusters
            )

        else:
            # Singleton (low similarity)
            return IncrementalAssignmentResult(
                mechanism_text=mechanism_text,
                cluster_id=-1,
                assignment_type='singleton',
                similarity_score=max_similarity
            )

    def should_recluster(self) -> Tuple[bool, List[str]]:
        """
        Determine if full re-clustering is needed.

        Returns:
            Tuple of (should_recluster, reasons)
        """
        reasons = []

        # Condition 1: Iteration interval
        if self.current_iteration > 0 and self.current_iteration % self.recluster_iteration_interval == 0:
            reasons.append(f"scheduled_interval (every {self.recluster_iteration_interval} iterations)")

        # Condition 2: Singleton rate (requires DB query)
        # TODO: Implement singleton rate check

        # Condition 3: Data growth
        current_size = len(self.load_mechanisms_from_db())
        if self.last_cluster_size > 0:
            growth_rate = (current_size - self.last_cluster_size) / self.last_cluster_size
            if growth_rate > self.recluster_growth_threshold:
                reasons.append(f"data_growth ({growth_rate:.1%} > {self.recluster_growth_threshold:.0%})")

        should_trigger = len(reasons) > 0

        if should_trigger:
            logger.info(f"Re-clustering triggered: {', '.join(reasons)}")

        return should_trigger, reasons

    def recluster_all_mechanisms(
        self,
        previous_labels: Optional[np.ndarray] = None
    ) -> ClusteringResult:
        """
        Full re-clustering with ARI validation.

        Args:
            previous_labels: Optional previous cluster labels for ARI computation

        Returns:
            ClusteringResult with new clustering and ARI validation
        """
        logger.info("Running full re-clustering...")

        # Run clustering
        result = self.cluster_all_mechanisms()

        if not result.success:
            return result

        # Compute ARI if previous labels provided
        if previous_labels is not None:
            new_labels = np.array([result.cluster_assignments[mech]
                                   for mech in result.cluster_assignments.keys()])
            ari = adjusted_rand_score(previous_labels, new_labels)

            logger.info(f"Adjusted Rand Index: {ari:.3f}")

            if ari < 0.70:
                logger.warning("Major cluster shift detected (ARI < 0.70) - flagging for manual review")

        return result


def main():
    """Command-line interface for mechanism normalizer."""
    import argparse

    parser = argparse.ArgumentParser(description="Mechanism Semantic Normalization & Clustering")
    parser.add_argument('--db-path', required=True, help='Path to intervention_research.db')
    parser.add_argument('--cache-dir', help='Cache directory (default: auto)')
    parser.add_argument('--action', choices=['cluster', 'assign', 'recluster'], default='cluster',
                       help='Action to perform')
    parser.add_argument('--mechanism', help='Mechanism text for assignment (--action=assign)')

    args = parser.parse_args()

    normalizer = MechanismNormalizer(db_path=args.db_path, cache_dir=args.cache_dir)

    if args.action == 'cluster':
        result = normalizer.cluster_all_mechanisms()
        print(f"\n{'='*60}")
        print("CLUSTERING COMPLETE")
        print(f"{'='*60}")
        print(f"Success: {result.success}")
        print(f"Mechanisms: {result.num_mechanisms}")
        print(f"Clusters: {result.num_clusters}")
        print(f"Silhouette: {result.metrics.silhouette_score:.3f}")
        print(f"Singletons: {result.metrics.singleton_count} ({result.metrics.singleton_percentage:.1%})")
        print(f"Time: {result.elapsed_time_seconds:.1f}s")

    elif args.action == 'assign':
        if not args.mechanism:
            print("Error: --mechanism required for assignment")
            return

        result = normalizer.assign_mechanism(args.mechanism)
        print(f"\nMechanism: {result.mechanism_text}")
        print(f"Cluster ID: {result.cluster_id}")
        print(f"Assignment Type: {result.assignment_type}")
        print(f"Similarity: {result.similarity_score:.3f}")
        if result.multi_label_clusters:
            print(f"Multi-label clusters: {result.multi_label_clusters}")

    elif args.action == 'recluster':
        result = normalizer.recluster_all_mechanisms()
        print(f"\nRe-clustering complete")
        print(f"Clusters: {result.num_clusters}")
        print(f"Silhouette: {result.metrics.silhouette_score:.3f}")


if __name__ == "__main__":
    main()
