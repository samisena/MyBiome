"""
Phase 3d Hierarchical Cluster Merging Orchestrator

Coordinates the full Phase 3d pipeline:
1. Load clusters from Phase 3b/3c
2. Load embeddings from Phase 3a cache
3. Compute centroids (Stage 1)
4. Generate merge candidates (Stage 2)
5. LLM validation (Stage 3)
6. Apply merges and create hierarchy (Stage 4-5)

Integrates with Phase 3a/3b/3c clustering-first pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .config import config, Phase3dConfig
from .embedding_loader import load_embeddings_for_clusters, get_embedding_cache_info
from .stage_1_centroid_computation import compute_centroids
from .validation_metrics import Cluster

logger = logging.getLogger(__name__)


@dataclass
class Phase3dResults:
    """Results from Phase 3d hierarchical merging."""
    entity_type: str
    initial_clusters: int
    final_clusters: int
    reduction_pct: float
    hierarchy_depth: int
    merges_applied: int
    duration_seconds: float


class Phase3dOrchestrator:
    """
    Orchestrator for Phase 3d hierarchical cluster merging.

    Loads clusters from Phase 3b/3c and embeddings from Phase 3a,
    then applies hierarchical merging to create parent-child relationships.
    """

    def __init__(self, config_override: Optional[Phase3dConfig] = None):
        """
        Initialize Phase 3d orchestrator.

        Args:
            config_override: Optional config override (uses global config if None)
        """
        self.config = config_override or config
        logger.info(f"Phase3dOrchestrator initialized: entity_type={self.config.entity_type}")

    def run(
        self,
        entity_type: str,
        naming_results: Optional[Dict] = None,
        cluster_labels: Optional[np.ndarray] = None,
        entity_names: Optional[List[str]] = None
    ) -> Phase3dResults:
        """
        Run full Phase 3d pipeline for entity type.

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'
            naming_results: Dict of cluster_id -> NamingResult from Phase 3c
            cluster_labels: Cluster labels from Phase 3b (parallel to entity_names)
            entity_names: Entity names from Phase 3a

        Returns:
            Phase3dResults with merging statistics
        """
        start_time = time.time()

        logger.info("="*60)
        logger.info(f"PHASE 3d HIERARCHICAL MERGING: {entity_type.upper()}S")
        logger.info("="*60)

        # Step 1: Check embeddings exist
        logger.info("[Step 1] Checking Phase 3a embedding cache...")
        cache_info = get_embedding_cache_info(entity_type)

        if not cache_info['exists']:
            raise FileNotFoundError(
                f"Phase 3a embeddings not found for {entity_type}s. "
                f"Please run Phase 3a first: {cache_info['path']}"
            )

        logger.info(f"  Found: {cache_info['count']} embeddings ({cache_info['dimension']}D)")

        # Step 2: Load clusters from Phase 3b/3c
        logger.info("[Step 2] Loading clusters from Phase 3b/3c...")
        clusters = self._load_clusters_from_phase3bc(
            entity_type,
            naming_results=naming_results,
            cluster_labels=cluster_labels,
            entity_names=entity_names
        )
        logger.info(f"  Loaded: {len(clusters)} clusters")

        # Step 3: Load embeddings
        logger.info("[Step 3] Loading embeddings for cluster members...")
        embeddings = load_embeddings_for_clusters(
            clusters=clusters,
            entity_type=entity_type
        )
        logger.info(f"  Loaded: {len(embeddings)} member embeddings")

        # Step 4: Compute centroids
        logger.info("[Step 4] Computing cluster centroids...")
        centroids = compute_centroids(
            clusters=clusters,
            embeddings=embeddings,
            embedding_dim=self.config.embedding_dimension,
            normalize=True
        )
        logger.info(f"  Computed: {len(centroids)} centroids")

        # Step 5: Generate merge candidates
        logger.info("[Step 5] Generating merge candidates...")
        from .stage_2_candidate_generation import generate_merge_candidates

        candidates = generate_merge_candidates(
            clusters=clusters,
            centroids=centroids,
            similarity_threshold=self.config.phase3d.get('candidate_generation', {}).get('similarity_threshold', 0.6),
            max_candidates=self.config.phase3d.get('candidate_generation', {}).get('max_candidates', 1000)
        )
        logger.info(f"  Generated: {len(candidates)} merge candidates")

        # Step 6: LLM validation
        logger.info("[Step 6] Validating merge candidates with LLM...")
        from .stage_3_llm_validation import LLMValidator

        validator = LLMValidator(self.config)
        validation_results = validator.validate_candidates(candidates, embeddings)
        approved_merges = validator.get_approved_merges(validation_results)
        logger.info(f"  Auto-approved: {len(approved_merges)}/{len(validation_results)} merges")

        # Step 7: Apply approved merges
        logger.info("[Step 7] Applying approved merges to database...")
        from .stage_5_merge_application import MergeApplicator

        applicator = MergeApplicator(db_path=str(self.config.db_path), entity_type=entity_type)
        application_result = applicator.apply_merges(
            approved_merges=approved_merges,
            target_level=2,
            create_backup=True
        )

        # Calculate final results
        duration = time.time() - start_time

        results = Phase3dResults(
            entity_type=entity_type,
            initial_clusters=len(clusters),
            final_clusters=application_result.parents_created + len(clusters),
            reduction_pct=(1 - (application_result.parents_created + len(clusters)) / len(clusters)) * 100 if len(clusters) > 0 else 0.0,
            hierarchy_depth=2,  # Currently creates parent-child only
            merges_applied=application_result.merges_identical + application_result.merges_parent_child,
            duration_seconds=duration
        )

        logger.info("="*60)
        logger.info(f"PHASE 3d COMPLETE: {entity_type.upper()}S")
        logger.info("="*60)
        logger.info(f"  Initial clusters: {results.initial_clusters}")
        logger.info(f"  Final clusters: {results.final_clusters}")
        logger.info(f"  Reduction: {results.reduction_pct:.1f}%")
        logger.info(f"  Duration: {results.duration_seconds:.1f}s")

        return results

    def _load_clusters_from_phase3bc(
        self,
        entity_type: str,
        naming_results: Optional[Dict] = None,
        cluster_labels: Optional[np.ndarray] = None,
        entity_names: Optional[List[str]] = None
    ) -> List[Cluster]:
        """
        Load clusters from Phase 3b/3c results.

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'
            naming_results: Dict of cluster_id -> NamingResult from Phase 3c
            cluster_labels: Cluster labels from Phase 3b (parallel to entity_names)
            entity_names: Entity names from Phase 3a

        Returns:
            List of Cluster objects
        """
        if naming_results is None or cluster_labels is None or entity_names is None:
            logger.warning("_load_clusters_from_phase3bc: Missing Phase 3b/3c data, returning empty list")
            return []

        clusters = []

        # Group entities by cluster_id
        cluster_members = {}
        for i, (entity_name, cluster_id) in enumerate(zip(entity_names, cluster_labels)):
            if cluster_id not in cluster_members:
                cluster_members[cluster_id] = []
            cluster_members[cluster_id].append(entity_name)

        # Create Cluster objects
        for cluster_id, members in cluster_members.items():
            naming_result = naming_results.get(cluster_id)

            if naming_result is None:
                logger.warning(f"No naming result for cluster {cluster_id}, skipping")
                continue

            cluster = Cluster(
                cluster_id=cluster_id,
                canonical_name=naming_result.canonical_name,
                category=naming_result.category,
                members=members,
                confidence=naming_result.confidence if hasattr(naming_result, 'confidence') else 'MEDIUM'
            )
            clusters.append(cluster)

        logger.info(f"Loaded {len(clusters)} clusters with {sum(len(c.members) for c in clusters)} total members")

        return clusters


def main():
    """CLI entry point for Phase 3d orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3d Hierarchical Cluster Merging")

    parser.add_argument(
        '--entity-type',
        type=str,
        required=True,
        choices=['intervention', 'condition', 'mechanism'],
        help='Entity type to process'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Override config file path'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run orchestrator
    orchestrator = Phase3dOrchestrator()

    try:
        results = orchestrator.run(args.entity_type)
        print(f"\n✓ Phase 3d complete for {args.entity_type}s")
        print(f"  Clusters: {results.initial_clusters} → {results.final_clusters} ({results.reduction_pct:.1f}% reduction)")
        print(f"  Duration: {results.duration_seconds:.1f}s")

    except Exception as e:
        logger.error(f"Phase 3d failed: {e}", exc_info=True)
        print(f"\n✗ Phase 3d failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
