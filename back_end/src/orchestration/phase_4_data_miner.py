#!/usr/bin/env python3
"""
Phase 4 Data Mining Orchestrator

Coordinates Phase 4a (Knowledge Graph Construction) and Phase 4b (Bayesian Scoring)
using canonical groups from Phase 3 for cleaner analytics and better statistical power.

Pipeline Flow:
    Phase 3 (Semantic Normalization) →
    Phase 4a (Knowledge Graph) →
    Phase 4b (Bayesian Scoring) →
    Frontend Export

Features:
- Integrates with Phase 3 canonical groups
- Pools evidence across cluster members
- Cleaner knowledge graph (538 nodes vs. 716 duplicates)
- Better statistical power (pooled evidence)
- Database persistence
- YAML configuration support
- Session tracking

Usage:
    # Run complete Phase 4 pipeline
    python -m back_end.src.orchestration.phase_4_data_miner

    # Run Phase 4a only
    python -m back_end.src.orchestration.phase_4_data_miner --phase-4a-only

    # Run Phase 4b only
    python -m back_end.src.orchestration.phase_4_data_miner --phase-4b-only

    # Check status
    python -m back_end.src.orchestration.phase_4_data_miner --status
"""

import sys
import argparse
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    from ..data.config import config, setup_logging
    from ..phase_1_data_collection.database_manager import database_manager
    from ..phase_4_data_mining.phase_4a_knowledge_graph import MedicalKnowledgeGraph
    from ..phase_4_data_mining.phase_4b_bayesian_scorer import BayesianEvidenceScorer
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config, setup_logging
    from back_end.src.phase_1_data_collection.database_manager import database_manager
    from back_end.src.phase_4_data_mining.phase_4a_knowledge_graph import MedicalKnowledgeGraph
    from back_end.src.phase_4_data_mining.phase_4b_bayesian_scorer import BayesianEvidenceScorer

logger = setup_logging(__name__, 'phase_4_data_miner.log')


@dataclass
class Phase4Results:
    """Results from Phase 4 execution."""
    success: bool
    phase_4a_completed: bool = False
    phase_4b_completed: bool = False

    # Phase 4a statistics
    canonical_groups_processed: int = 0
    knowledge_graph_nodes: int = 0
    knowledge_graph_edges: int = 0
    phase_4a_duration_seconds: float = 0.0

    # Phase 4b statistics
    bayesian_scores_generated: int = 0
    high_confidence_scores: int = 0
    phase_4b_duration_seconds: float = 0.0

    # Overall statistics
    total_duration_seconds: float = 0.0
    error: Optional[str] = None


class Phase4DataMiningOrchestrator:
    """
    Phase 4 orchestrator - coordinates knowledge graph construction and Bayesian scoring.

    Consumes Phase 3 canonical groups to build cleaner, more powerful analytics.
    """

    def __init__(self, db_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize Phase 4 orchestrator.

        Args:
            db_path: Path to intervention_research.db (defaults to config.db_path)
            config_path: Path to phase_4_config.yaml (optional)
        """
        self.db_path = Path(db_path) if db_path else Path(config.db_path)
        self.config_path = config_path

        # Load Phase 4 configuration if provided
        self.phase4_config = self._load_config() if config_path else None

        # Initialize components (lazy loaded)
        self.knowledge_graph = None
        self.bayesian_scorer = None

        logger.info("Phase 4 Data Mining Orchestrator initialized")
        logger.info(f"Database: {self.db_path}")

    def _load_config(self) -> Dict:
        """Load Phase 4 configuration from YAML."""
        import yaml

        try:
            with open(self.config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            logger.info(f"Loaded Phase 4 configuration from {self.config_path}")
            return cfg
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {}

    def run(self, force: bool = False) -> Phase4Results:
        """
        Run complete Phase 4 pipeline (4a + 4b).

        Args:
            force: Force rebuild even if results exist

        Returns:
            Phase4Results with execution statistics
        """
        logger.info("="*60)
        logger.info("PHASE 4: DATA MINING PIPELINE")
        logger.info("="*60)
        logger.info("Phase 4a: Knowledge Graph Construction")
        logger.info("Phase 4b: Bayesian Evidence Scoring")
        logger.info("="*60)

        start_time = time.time()
        results = Phase4Results(success=False)

        try:
            # Phase 4a: Knowledge Graph Construction
            phase_4a_result = self._run_phase_4a(force=force)
            results.phase_4a_completed = phase_4a_result['success']
            results.canonical_groups_processed = phase_4a_result.get('canonical_groups_processed', 0)
            results.knowledge_graph_nodes = phase_4a_result.get('nodes_created', 0)
            results.knowledge_graph_edges = phase_4a_result.get('edges_created', 0)
            results.phase_4a_duration_seconds = phase_4a_result.get('duration_seconds', 0.0)

            if not phase_4a_result['success']:
                results.error = f"Phase 4a failed: {phase_4a_result.get('error', 'Unknown error')}"
                return results

            # Phase 4b: Bayesian Scoring
            phase_4b_result = self._run_phase_4b(force=force)
            results.phase_4b_completed = phase_4b_result['success']
            results.bayesian_scores_generated = phase_4b_result.get('scores_generated', 0)
            results.high_confidence_scores = phase_4b_result.get('high_confidence_count', 0)
            results.phase_4b_duration_seconds = phase_4b_result.get('duration_seconds', 0.0)

            if not phase_4b_result['success']:
                results.error = f"Phase 4b failed: {phase_4b_result.get('error', 'Unknown error')}"
                return results

            # Success!
            results.success = True
            results.total_duration_seconds = time.time() - start_time

            logger.info("\n" + "="*60)
            logger.info("PHASE 4 COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Total duration: {results.total_duration_seconds:.1f}s")
            logger.info(f"Knowledge graph nodes: {results.knowledge_graph_nodes}")
            logger.info(f"Knowledge graph edges: {results.knowledge_graph_edges}")
            logger.info(f"Bayesian scores: {results.bayesian_scores_generated}")
            logger.info(f"High confidence scores: {results.high_confidence_scores}")

            return results

        except Exception as e:
            logger.error(f"Phase 4 pipeline failed: {e}")
            logger.error(traceback.format_exc())
            results.error = str(e)
            results.total_duration_seconds = time.time() - start_time
            return results

    def _run_phase_4a(self, force: bool = False) -> Dict[str, Any]:
        """
        Run Phase 4a: Knowledge Graph Construction.

        Builds multi-edge bidirectional graph from Phase 3 canonical groups.

        Args:
            force: Force rebuild even if graph exists

        Returns:
            Dictionary with Phase 4a results
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 4A: KNOWLEDGE GRAPH CONSTRUCTION")
        logger.info("="*60)

        start_time = time.time()

        try:
            # Initialize knowledge graph
            self.knowledge_graph = MedicalKnowledgeGraph(save_to_database=True)

            # Build from Phase 3 canonical groups
            logger.info("Building knowledge graph from Phase 3 canonical groups...")
            build_result = self.knowledge_graph.build_from_phase3_clusters(
                db_path=str(self.db_path)
            )

            duration = time.time() - start_time

            result = {
                'success': True,
                'canonical_groups_processed': build_result['canonical_groups_processed'],
                'nodes_created': build_result['nodes_created'],
                'edges_created': build_result['edges_created'],
                'duration_seconds': duration,
                'avg_edges_per_group': build_result.get('avg_edges_per_group', 0.0)
            }

            logger.info(f"[SUCCESS] Phase 4a completed in {duration:.1f}s")
            logger.info(f"  Canonical groups: {result['canonical_groups_processed']}")
            logger.info(f"  Nodes: {result['nodes_created']}")
            logger.info(f"  Edges: {result['edges_created']}")

            return result

        except Exception as e:
            logger.error(f"Phase 4a failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }

    def _run_phase_4b(self, force: bool = False) -> Dict[str, Any]:
        """
        Run Phase 4b: Bayesian Evidence Scoring.

        Scores canonical groups using Bayesian statistics with pooled evidence.

        Args:
            force: Force rebuild even if scores exist

        Returns:
            Dictionary with Phase 4b results
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 4B: BAYESIAN EVIDENCE SCORING")
        logger.info("="*60)

        start_time = time.time()

        try:
            # Initialize Bayesian scorer
            self.bayesian_scorer = BayesianEvidenceScorer(
                alpha_prior=1.0,
                beta_prior=1.0,
                save_to_database=True,
                analysis_model="bayesian_phase4b_v1"
            )

            # Ensure knowledge graph is available
            if not self.knowledge_graph:
                logger.warning("Knowledge graph not available, attempting to rebuild...")
                phase_4a_result = self._run_phase_4a(force=False)
                if not phase_4a_result['success']:
                    raise RuntimeError("Phase 4a must complete before Phase 4b")

            # Score all canonical groups
            logger.info("Scoring all canonical groups from Phase 3...")
            all_scores = self.bayesian_scorer.score_all_canonical_groups(
                knowledge_graph=self.knowledge_graph,
                db_path=str(self.db_path),
                min_evidence=1
            )

            # Calculate statistics
            total_scores = sum(len(scores) for scores in all_scores.values())
            high_confidence_scores = sum(
                1 for group_scores in all_scores.values()
                for score in group_scores.values()
                if score.get('confidence', 0) > 0.7
            )

            duration = time.time() - start_time

            result = {
                'success': True,
                'scores_generated': total_scores,
                'canonical_groups_scored': len(all_scores),
                'high_confidence_count': high_confidence_scores,
                'duration_seconds': duration,
                'avg_score_per_group': total_scores / len(all_scores) if all_scores else 0
            }

            logger.info(f"[SUCCESS] Phase 4b completed in {duration:.1f}s")
            logger.info(f"  Canonical groups scored: {result['canonical_groups_scored']}")
            logger.info(f"  Total scores: {result['scores_generated']}")
            logger.info(f"  High confidence (>0.7): {result['high_confidence_count']}")

            return result

        except Exception as e:
            logger.error(f"Phase 4b failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }

    def get_status(self) -> Dict[str, Any]:
        """
        Get current Phase 4 status.

        Returns:
            Dictionary with Phase 4 statistics
        """
        try:
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                # Count knowledge graph nodes
                cursor.execute("SELECT COUNT(*) FROM knowledge_graph_nodes")
                kg_nodes = cursor.fetchone()[0]

                # Count knowledge graph edges
                cursor.execute("SELECT COUNT(*) FROM knowledge_graph_edges")
                kg_edges = cursor.fetchone()[0]

                # Count Bayesian scores
                cursor.execute("SELECT COUNT(*) FROM bayesian_scores")
                bayesian_scores = cursor.fetchone()[0]

                # Count high confidence scores
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM bayesian_scores
                    WHERE confidence_adjusted_score > 0.7
                """)
                high_confidence = cursor.fetchone()[0]

                return {
                    'phase_4a': {
                        'knowledge_graph_nodes': kg_nodes,
                        'knowledge_graph_edges': kg_edges,
                        'completed': kg_nodes > 0
                    },
                    'phase_4b': {
                        'bayesian_scores': bayesian_scores,
                        'high_confidence_scores': high_confidence,
                        'completed': bayesian_scores > 0
                    },
                    'pipeline_completed': kg_nodes > 0 and bayesian_scores > 0
                }

        except Exception as e:
            logger.error(f"Error getting Phase 4 status: {e}")
            return {
                'error': str(e),
                'phase_4a': {'completed': False},
                'phase_4b': {'completed': False},
                'pipeline_completed': False
            }


def main():
    """Command line interface for Phase 4 Data Mining."""
    parser = argparse.ArgumentParser(
        description="Phase 4 Data Mining Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete Phase 4 pipeline
  python phase_4_data_miner.py

  # Run Phase 4a only
  python phase_4_data_miner.py --phase-4a-only

  # Run Phase 4b only
  python phase_4_data_miner.py --phase-4b-only

  # Check status
  python phase_4_data_miner.py --status

  # Force rebuild
  python phase_4_data_miner.py --force
        """
    )

    parser.add_argument('--db-path', type=str, help='Path to intervention_research.db')
    parser.add_argument('--config', type=str, help='Path to phase_4_config.yaml')
    parser.add_argument('--phase-4a-only', action='store_true', help='Run Phase 4a only')
    parser.add_argument('--phase-4b-only', action='store_true', help='Run Phase 4b only')
    parser.add_argument('--force', action='store_true', help='Force rebuild')
    parser.add_argument('--status', action='store_true', help='Show Phase 4 status')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    try:
        # Initialize orchestrator
        orchestrator = Phase4DataMiningOrchestrator(
            db_path=args.db_path,
            config_path=args.config
        )

        # Handle status check
        if args.status:
            status = orchestrator.get_status()
            print(json.dumps(status, indent=2))
            return

        # Run Phase 4
        if args.phase_4a_only:
            logger.info("Running Phase 4a only...")
            result = orchestrator._run_phase_4a(force=args.force)
            print(json.dumps(result, indent=2))
        elif args.phase_4b_only:
            logger.info("Running Phase 4b only...")
            result = orchestrator._run_phase_4b(force=args.force)
            print(json.dumps(result, indent=2))
        else:
            # Run complete pipeline
            results = orchestrator.run(force=args.force)

            # Print summary
            if results.success:
                print("\n[SUCCESS] Phase 4 completed successfully")
                print(f"Total duration: {results.total_duration_seconds:.1f}s")
                print(f"\nPhase 4a (Knowledge Graph):")
                print(f"  Canonical groups: {results.canonical_groups_processed}")
                print(f"  Nodes: {results.knowledge_graph_nodes}")
                print(f"  Edges: {results.knowledge_graph_edges}")
                print(f"  Duration: {results.phase_4a_duration_seconds:.1f}s")
                print(f"\nPhase 4b (Bayesian Scoring):")
                print(f"  Scores generated: {results.bayesian_scores_generated}")
                print(f"  High confidence: {results.high_confidence_scores}")
                print(f"  Duration: {results.phase_4b_duration_seconds:.1f}s")
            else:
                print(f"\n[FAILED] Phase 4 failed: {results.error}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Phase 4 interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Phase 4 failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
