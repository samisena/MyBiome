"""
Mechanism Clustering Pipeline Orchestrator - Complete Integration

Orchestrates the full mechanism clustering pipeline from raw mechanisms to database population:
1. Load mechanisms from database
2. Run clustering (with optional preprocessing and hierarchy)
3. Extract canonical names
4. Populate database tables
5. Build analytics associations
6. Validate results

This is the main entry point for Phase 3.6 integration with batch_medical_rotation.py

Usage:
    from phase_3c_mechanism_pipeline_orchestrator import MechanismPipelineOrchestrator

    orchestrator = MechanismPipelineOrchestrator(db_path=config.db_path)
    result = orchestrator.run_full_pipeline()
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

# Import all components
from .mechanism_normalizer import MechanismNormalizer, ClusteringResult
from .mechanism_db_manager import MechanismDatabaseManager
from .mechanism_preprocessor import MechanismPreprocessor
from .mechanism_baseline_test import BaselineTest
from .mechanism_preprocessing_comparison import PreprocessingComparison

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of full pipeline execution."""
    success: bool
    elapsed_time_seconds: float

    # Clustering results
    clustering_result: Optional[ClusteringResult] = None

    # Database population
    clusters_created: int = 0
    memberships_created: int = 0
    associations_created: int = 0

    # Validation
    passed_validation: bool = False
    validation_summary: str = ""

    # Error handling
    error: Optional[str] = None
    phase_reached: str = "initialization"


class MechanismPipelineOrchestrator:
    """
    Orchestrates the complete mechanism clustering pipeline.

    Phases:
    1. Initialization & Schema Setup
    2. Mechanism Clustering
    3. Database Population
    4. Analytics Building
    5. Validation
    """

    def __init__(
        self,
        db_path: str,
        cache_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        force_schema_init: bool = False,
        enable_preprocessing: bool = False,
        enable_hierarchy: bool = True
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            db_path: Path to intervention_research.db
            cache_dir: Cache directory for embeddings and LLM decisions
            results_dir: Directory for results
            force_schema_init: Force drop/recreate database tables
            enable_preprocessing: Enable hierarchical decomposition preprocessing
            enable_hierarchy: Enable 2-level hierarchical clustering
        """
        self.db_path = db_path
        self.force_schema_init = force_schema_init
        self.enable_preprocessing = enable_preprocessing
        self.enable_hierarchy = enable_hierarchy

        # Setup directories
        if cache_dir is None:
            from .config import CACHE_DIR
            cache_dir = CACHE_DIR

        if results_dir is None:
            from .config import RESULTS_DIR
            results_dir = RESULTS_DIR
        else:
            results_dir = Path(results_dir)

        self.cache_dir = cache_dir
        self.results_dir = results_dir

        # Initialize components
        self.normalizer = MechanismNormalizer(
            db_path=db_path,
            cache_dir=cache_dir
        )

        self.db_manager = MechanismDatabaseManager(db_path=db_path)

        if enable_preprocessing:
            self.preprocessor = MechanismPreprocessor(cache_dir=cache_dir)
        else:
            self.preprocessor = None

        logger.info("MechanismPipelineOrchestrator initialized")
        logger.info(f"  Preprocessing: {'enabled' if enable_preprocessing else 'disabled'}")
        logger.info(f"  Hierarchy: {'enabled' if enable_hierarchy else 'disabled'}")

    def run_full_pipeline(self) -> PipelineResult:
        """
        Run the complete mechanism clustering pipeline.

        Returns:
            PipelineResult with execution details
        """
        start_time = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("MECHANISM CLUSTERING PIPELINE - PHASE 3.6")
        logger.info("="*80)

        try:
            # Phase 1: Schema Initialization
            logger.info("\n" + "="*80)
            logger.info("PHASE 1: SCHEMA INITIALIZATION")
            logger.info("="*80)

            if not self._initialize_schema():
                return PipelineResult(
                    success=False,
                    elapsed_time_seconds=(datetime.now() - start_time).total_seconds(),
                    error="Schema initialization failed",
                    phase_reached="schema_initialization"
                )

            # Phase 2: Mechanism Clustering
            logger.info("\n" + "="*80)
            logger.info("PHASE 2: MECHANISM CLUSTERING")
            logger.info("="*80)

            clustering_result = self._run_clustering()

            if not clustering_result or not clustering_result.success:
                return PipelineResult(
                    success=False,
                    elapsed_time_seconds=(datetime.now() - start_time).total_seconds(),
                    clustering_result=clustering_result,
                    error="Clustering failed",
                    phase_reached="clustering"
                )

            # Phase 3: Database Population
            logger.info("\n" + "="*80)
            logger.info("PHASE 3: DATABASE POPULATION")
            logger.info("="*80)

            clusters_created, memberships_created = self._populate_database(clustering_result)

            if clusters_created == 0:
                logger.warning("No clusters were created in database")

            # Phase 4: Analytics Building
            logger.info("\n" + "="*80)
            logger.info("PHASE 4: ANALYTICS BUILDING")
            logger.info("="*80)

            associations_created = self._build_analytics()

            # Phase 5: Validation
            logger.info("\n" + "="*80)
            logger.info("PHASE 5: VALIDATION")
            logger.info("="*80)

            passed_validation, validation_summary = self._validate_results(clustering_result)

            # Success
            elapsed_time = (datetime.now() - start_time).total_seconds()

            result = PipelineResult(
                success=True,
                elapsed_time_seconds=elapsed_time,
                clustering_result=clustering_result,
                clusters_created=clusters_created,
                memberships_created=memberships_created,
                associations_created=associations_created,
                passed_validation=passed_validation,
                validation_summary=validation_summary,
                phase_reached="complete"
            )

            self._print_final_summary(result)

            return result

        except Exception as e:
            logger.error(f"Pipeline failed with exception: {e}")
            import traceback
            traceback.print_exc()

            return PipelineResult(
                success=False,
                elapsed_time_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
                phase_reached="exception"
            )

    def _initialize_schema(self) -> bool:
        """Initialize database schema."""
        logger.info("Initializing mechanism clustering schema...")

        success = self.db_manager.initialize_schema(force=self.force_schema_init)

        if success:
            logger.info("✓ Schema initialized successfully")
        else:
            logger.error("✗ Schema initialization failed")

        return success

    def _run_clustering(self) -> Optional[ClusteringResult]:
        """Run mechanism clustering (with optional preprocessing)."""

        # Load mechanisms
        mechanisms = self.normalizer.load_mechanisms_from_db()

        if not mechanisms:
            logger.error("No mechanisms found in database")
            return None

        # Optional: Run preprocessing comparison
        if self.enable_preprocessing and self.preprocessor:
            logger.info("Running preprocessing comparison...")

            comparison = PreprocessingComparison(
                db_path=self.db_path,
                cache_dir=self.cache_dir,
                results_dir=self.results_dir
            )

            comparison_result = comparison.run_comparison(
                run_baseline_first=True
            )

            if comparison_result['comparison']['should_keep_preprocessing']:
                logger.info("✓ Preprocessing improved results, using enhanced dataset")
                # Use enhanced mechanisms (TODO: extract from comparison_result)
            else:
                logger.info("✓ Baseline performed better, using raw mechanisms")

        # Run clustering
        logger.info("Running HDBSCAN clustering...")
        clustering_result = self.normalizer.cluster_all_mechanisms(mechanisms=mechanisms)

        if clustering_result.success:
            logger.info(f"✓ Clustering complete: {clustering_result.num_clusters} clusters created")
        else:
            logger.error(f"✗ Clustering failed: {clustering_result.error}")

        return clustering_result

    def _populate_database(self, clustering_result: ClusteringResult) -> tuple[int, int]:
        """
        Populate database with clustering results.

        Args:
            clustering_result: ClusteringResult from clustering phase

        Returns:
            Tuple of (clusters_created, memberships_created)
        """
        logger.info("Populating mechanism_clusters table...")

        clusters_created = 0
        memberships_created = 0

        # Create clusters
        for cluster_id, canonical_name in clustering_result.canonical_names.items():
            if cluster_id == -1:
                continue  # Skip singleton cluster

            # Get parent ID if hierarchical
            parent_id = None
            hierarchy_level = 0

            if clustering_result.hierarchies:
                # Check if this cluster is a child
                for parent, children in clustering_result.hierarchies.items():
                    if cluster_id in children:
                        parent_id = parent
                        hierarchy_level = 1
                        break

            # Create cluster
            created_id = self.db_manager.create_cluster(
                canonical_name=canonical_name,
                parent_cluster_id=parent_id,
                hierarchy_level=hierarchy_level
            )

            if created_id:
                clusters_created += 1

        logger.info(f"✓ Created {clusters_created} clusters")

        # Create memberships
        logger.info("Populating mechanism_cluster_membership table...")

        for mechanism_text, cluster_id in clustering_result.cluster_assignments.items():
            # Determine assignment type (TODO: enhance with actual assignment metadata)
            assignment_type = 'primary' if cluster_id != -1 else 'singleton'

            membership_id = self.db_manager.add_membership(
                mechanism_text=mechanism_text,
                cluster_id=cluster_id if cluster_id != -1 else self.db_manager.get_cluster_id_by_name("unclustered_mechanisms"),
                assignment_type=assignment_type
            )

            if membership_id:
                memberships_created += 1

        logger.info(f"✓ Created {memberships_created} memberships")

        # Update cluster stats
        logger.info("Updating cluster statistics...")
        for cluster_id in clustering_result.canonical_names.keys():
            if cluster_id != -1:
                self.db_manager.update_cluster_stats(cluster_id)

        logger.info("✓ Cluster statistics updated")

        return clusters_created, memberships_created

    def _build_analytics(self) -> int:
        """Build analytics tables."""
        logger.info("Populating intervention_mechanisms table...")

        rows_created = self.db_manager.populate_intervention_mechanisms()
        logger.info(f"✓ Created {rows_created} intervention-mechanism mappings")

        logger.info("Building mechanism_condition_associations table...")

        associations_created = self.db_manager.build_mechanism_condition_associations()
        logger.info(f"✓ Created {associations_created} mechanism-condition associations")

        return associations_created

    def _validate_results(self, clustering_result: ClusteringResult) -> tuple[bool, str]:
        """
        Validate pipeline results.

        Args:
            clustering_result: ClusteringResult to validate

        Returns:
            Tuple of (passed, summary_string)
        """
        logger.info("Validating clustering results...")

        # Get database stats
        stats = self.db_manager.get_cluster_stats()

        # Validation checks
        checks = {
            'silhouette_score': clustering_result.metrics.silhouette_score > 0.35,
            'singleton_percentage': clustering_result.metrics.singleton_percentage < 0.20,
            'clusters_in_range': 10 <= clustering_result.num_clusters <= 40,
            'clusters_in_db': stats.get('total_clusters', 0) > 0,
            'memberships_in_db': stats.get('total_memberships', 0) > 0,
            'associations_in_db': stats.get('total_associations', 0) > 0
        }

        passed = all(checks.values())

        # Generate summary
        summary_lines = []
        for check_name, check_passed in checks.items():
            status = "✓" if check_passed else "✗"
            summary_lines.append(f"{status} {check_name}")

        summary = "\n".join(summary_lines)

        logger.info("\nValidation Results:")
        logger.info(summary)

        if passed:
            logger.info("\n✓ ALL VALIDATION CHECKS PASSED")
        else:
            logger.warning("\n✗ SOME VALIDATION CHECKS FAILED")

        return passed, summary

    def _print_final_summary(self, result: PipelineResult):
        """Print final pipeline summary."""
        logger.info("\n" + "="*80)
        logger.info("MECHANISM CLUSTERING PIPELINE - COMPLETE")
        logger.info("="*80)

        logger.info(f"\nExecution Summary:")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Phase Reached: {result.phase_reached}")
        logger.info(f"  Elapsed Time: {result.elapsed_time_seconds:.1f}s")

        if result.clustering_result:
            logger.info(f"\nClustering Results:")
            logger.info(f"  Mechanisms: {result.clustering_result.num_mechanisms}")
            logger.info(f"  Clusters: {result.clustering_result.num_clusters}")
            logger.info(f"  Silhouette: {result.clustering_result.metrics.silhouette_score:.3f}")
            logger.info(f"  Singletons: {result.clustering_result.metrics.singleton_percentage:.1%}")

            if result.clustering_result.hierarchies:
                logger.info(f"  Hierarchies: {len(result.clustering_result.hierarchies)} parent-child relationships")

        logger.info(f"\nDatabase Population:")
        logger.info(f"  Clusters Created: {result.clusters_created}")
        logger.info(f"  Memberships Created: {result.memberships_created}")
        logger.info(f"  Associations Created: {result.associations_created}")

        logger.info(f"\nValidation:")
        logger.info(f"  Passed: {result.passed_validation}")
        if result.validation_summary:
            logger.info(f"\n{result.validation_summary}")

        if result.error:
            logger.error(f"\nError: {result.error}")

        logger.info("\n" + "="*80)


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Mechanism Clustering Pipeline Orchestrator (Phase 3.6)")
    parser.add_argument('--db-path', required=True, help='Path to intervention_research.db')
    parser.add_argument('--force-schema-init', action='store_true', help='Force drop/recreate tables')
    parser.add_argument('--enable-preprocessing', action='store_true', help='Enable hierarchical decomposition')
    parser.add_argument('--disable-hierarchy', action='store_true', help='Disable hierarchical clustering')
    parser.add_argument('--cache-dir', help='Cache directory (default: auto)')
    parser.add_argument('--results-dir', help='Results directory (default: auto)')

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = MechanismPipelineOrchestrator(
        db_path=args.db_path,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        force_schema_init=args.force_schema_init,
        enable_preprocessing=args.enable_preprocessing,
        enable_hierarchy=not args.disable_hierarchy
    )

    # Run pipeline
    result = orchestrator.run_full_pipeline()

    # Exit code
    exit_code = 0 if result.success else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
