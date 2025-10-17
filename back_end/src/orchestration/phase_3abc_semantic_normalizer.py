"""
Phase 3a/3b/3c Semantic Normalization Orchestrator (NEW - Clustering-First Architecture)

Runs unified Phase 3 pipeline with clustering-first approach:
- Phase 3a: Semantic Embedding (mxbai-embed-large, 1024-dim)
- Phase 3b: Clustering (Hierarchical with threshold=0.7 + Singleton Handler)
- Phase 3c: LLM Canonical Naming (qwen3:14b with temperature=0.0)

This is the NEW implementation that replaces the old naming-first approach.

Usage:
    # Run Phase 3a/3b/3c on all entity types (interventions, conditions, mechanisms)
    python -m back_end.src.orchestration.phase_3abc_semantic_normalizer --all

    # Run on specific entity type
    python -m back_end.src.orchestration.phase_3abc_semantic_normalizer --entity-type intervention

    # Check status
    python -m back_end.src.orchestration.phase_3abc_semantic_normalizer --status
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from back_end.src.data.config import config
from back_end.src.phase_3_semantic_normalization.phase_3_orchestrator import (
    UnifiedPhase3Orchestrator,
    EntityResults
)

# Setup logging
logging.basicConfig(
    level=logging.INFO if not config.fast_mode else logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase3ABCOrchestrator:
    """
    Orchestrator wrapper for unified Phase 3 pipeline.

    Provides a simple interface for running the clustering-first pipeline
    on interventions, conditions, and mechanisms.
    """

    def __init__(self, db_path: str = None, config_path: str = None):
        """Initialize orchestrator."""
        self.db_path = db_path or str(config.db_path)

        # Default config path
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'phase_3_semantic_normalization' / 'phase_3_config.yaml'

        self.config_path = Path(config_path)

        # Initialize unified orchestrator
        self.orchestrator = UnifiedPhase3Orchestrator(
            db_path=self.db_path,
            config_path=str(self.config_path)
        )

        logger.info(f"Initialized Phase3ABCOrchestrator with DB: {self.db_path}")

    def run_all_entity_types(self) -> Dict:
        """
        Run Phase 3a/3b/3c on all entity types.

        Returns:
            Dictionary with results for each entity type
        """
        print("\n" + "="*80)
        print("RUNNING UNIFIED PHASE 3 PIPELINE (CLUSTERING-FIRST)")
        print("="*80)
        print("\nPhase 3a: Semantic Embedding (mxbai-embed-large, 1024-dim)")
        print("Phase 3b: Clustering (Hierarchical threshold=0.7 + Singleton Handler)")
        print("Phase 3c: LLM Canonical Naming (qwen3:14b, temperature=0.0)")
        print("\n" + "="*80)

        results = {}

        # Process each entity type
        for entity_type in ['intervention', 'condition', 'mechanism']:
            print(f"\n\n{'='*80}")
            print(f"PROCESSING: {entity_type.upper()}S")
            print(f"{'='*80}\n")

            try:
                entity_results = self.orchestrator.run_pipeline(
                    entity_type=entity_type,
                    force_reembed=False,  # Use cached embeddings if available
                    force_recluster=False  # Use cached clusters if available
                )

                results[entity_type] = {
                    'success': True,
                    'embedding_duration': entity_results.embedding_duration_seconds,
                    'embeddings_generated': entity_results.embeddings_generated,
                    'clustering_duration': entity_results.clustering_duration_seconds,
                    'clusters_created': entity_results.clusters_created,
                    'singletons': entity_results.singletons_assigned,
                    'naming_duration': entity_results.naming_duration_seconds,
                    'entities_named': entity_results.entities_named,
                    'total_duration': entity_results.total_duration_seconds
                }

                print(f"\n[SUCCESS] {entity_type.capitalize()} processing complete!")
                print(f"  - Embeddings: {entity_results.embeddings_generated} entities")
                print(f"  - Clusters: {entity_results.clusters_created} (+ {entity_results.singletons_assigned} singletons)")
                print(f"  - Named: {entity_results.entities_named} canonical groups")
                print(f"  - Total time: {entity_results.total_duration_seconds:.1f}s")

            except Exception as e:
                logger.error(f"Error processing {entity_type}s: {e}", exc_info=True)
                results[entity_type] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"\n[ERROR] Failed to process {entity_type}s: {e}")

        # Summary
        print("\n\n" + "="*80)
        print("PHASE 3 PIPELINE COMPLETE")
        print("="*80)

        successful = sum(1 for r in results.values() if r.get('success', False))
        print(f"\nEntity types processed: {successful}/{len(results)}")

        for entity_type, result in results.items():
            if result.get('success'):
                print(f"\n{entity_type.capitalize()}s:")
                print(f"  - Clusters: {result['clusters_created']} + {result['singletons']} singletons")
                print(f"  - Named: {result['entities_named']} canonical groups")
                print(f"  - Time: {result['total_duration']:.1f}s")
            else:
                print(f"\n{entity_type.capitalize()}s: FAILED - {result.get('error', 'Unknown error')}")

        print("\n" + "="*80 + "\n")

        return results

    def run_single_entity_type(self, entity_type: str) -> EntityResults:
        """
        Run Phase 3a/3b/3c on a single entity type.

        Note: The orchestrator runs ALL entity types, so we extract just the requested one.

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'

        Returns:
            EntityResults object with processing statistics
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING: {entity_type.upper()}S")
        print(f"{'='*80}\n")

        try:
            # Run the full pipeline (processes all entity types)
            pipeline_results = self.orchestrator.run()

            if not pipeline_results['success']:
                raise Exception(pipeline_results.get('error', 'Unknown error'))

            # Extract results for requested entity type
            entity_type_plural = f"{entity_type}s"
            results = pipeline_results['results'][entity_type_plural]

            print(f"\n[SUCCESS] {entity_type.capitalize()} processing complete!")
            print(f"  - Embeddings: {results.embeddings_generated} entities")
            print(f"  - Clusters: {results.num_clusters} ({results.num_natural_clusters} natural + {results.num_singleton_clusters} singletons)")
            print(f"  - Named: {results.names_generated} canonical groups")
            print(f"  - Embedding time: {results.embedding_duration_seconds:.1f}s")
            print(f"  - Clustering time: {results.clustering_duration_seconds:.1f}s")
            print(f"  - Naming time: {results.naming_duration_seconds:.1f}s")

            return results

        except Exception as e:
            logger.error(f"Error processing {entity_type}s: {e}", exc_info=True)
            print(f"\n[ERROR] Failed to process {entity_type}s: {e}")
            raise

    def display_status(self):
        """Display Phase 3 pipeline status."""
        print("\n" + "="*80)
        print("PHASE 3ABC PIPELINE STATUS")
        print("="*80)
        print("\nClustering-First Architecture:")
        print("  - Phase 3a: Semantic Embedding (mxbai-embed-large)")
        print("  - Phase 3b: Clustering (Hierarchical + Singleton Handler)")
        print("  - Phase 3c: LLM Naming (qwen3:14b)")
        print("\n" + "="*80)

        # Check cache directories
        cache_dir = Path(__file__).parent.parent / 'phase_3_semantic_normalization' / 'cache'

        if cache_dir.exists():
            print(f"\nCache directory: {cache_dir}")

            # Count cached files
            embedding_caches = list(cache_dir.glob("*_embeddings.pkl"))
            cluster_caches = list(cache_dir.glob("*_clusters.pkl"))
            naming_caches = list(cache_dir.glob("*_naming.json"))

            print(f"  - Embedding caches: {len(embedding_caches)}")
            print(f"  - Cluster caches: {len(cluster_caches)}")
            print(f"  - Naming caches: {len(naming_caches)}")
        else:
            print("\nNo cache directory found (pipeline has not been run yet)")

        print("\n" + "="*80 + "\n")


def main():
    """CLI entry point for Phase 3a/3b/3c orchestrator."""
    parser = argparse.ArgumentParser(
        description="Unified Phase 3 pipeline (clustering-first architecture)"
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run pipeline on all entity types (interventions, conditions, mechanisms)'
    )

    parser.add_argument(
        '--entity-type',
        type=str,
        choices=['intervention', 'condition', 'mechanism'],
        help='Run pipeline on specific entity type'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show pipeline status'
    )

    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Override database path'
    )

    parser.add_argument(
        '--config-path',
        type=str,
        default=None,
        help='Override config YAML path'
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = Phase3ABCOrchestrator(
        db_path=args.db_path,
        config_path=args.config_path
    )

    # Handle commands
    if args.status:
        orchestrator.display_status()

    elif args.all:
        results = orchestrator.run_all_entity_types()

        # Exit with error code if any entity type failed
        failed = sum(1 for r in results.values() if not r.get('success', False))
        sys.exit(1 if failed > 0 else 0)

    elif args.entity_type:
        try:
            orchestrator.run_single_entity_type(args.entity_type)
            sys.exit(0)
        except Exception:
            sys.exit(1)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Run on all entity types")
        print("  python -m back_end.src.orchestration.phase_3abc_semantic_normalizer --all")
        print()
        print("  # Run on interventions only")
        print("  python -m back_end.src.orchestration.phase_3abc_semantic_normalizer --entity-type intervention")
        print()
        print("  # Check status")
        print("  python -m back_end.src.orchestration.phase_3abc_semantic_normalizer --status")


if __name__ == "__main__":
    main()
