"""
Phase execution logic for batch pipeline.

Handles running each pipeline phase with proper error handling and session updates.
"""

from typing import Dict, Any, Optional
from back_end.src.data.config import config, setup_logging
from .batch_session import BatchSession
from .batch_config import BatchPhase

logger = setup_logging(__name__, 'batch_medical_rotation.log')


class PhaseRunner:
    """Executes pipeline phases and updates session state."""

    def __init__(self):
        """Initialize phase runner with lazy-loaded components."""
        # Lazy load components (initialized on first use)
        self._paper_collector = None
        self._llm_processor = None
        self._dedup_integrator = None

    @property
    def paper_collector(self):
        """Lazy load paper collector."""
        if self._paper_collector is None:
            from .phase_1_paper_collector import RotationPaperCollector
            self._paper_collector = RotationPaperCollector()
        return self._paper_collector

    @property
    def llm_processor(self):
        """Lazy load LLM processor."""
        if self._llm_processor is None:
            from .phase_2_llm_processor import RotationLLMProcessor
            self._llm_processor = RotationLLMProcessor()
        return self._llm_processor

    @property
    def dedup_integrator(self):
        """Lazy load semantic grouping integrator."""
        if self._dedup_integrator is None:
            from .rotation_semantic_grouping_integrator import RotationSemanticGroupingIntegrator
            self._dedup_integrator = RotationSemanticGroupingIntegrator()
        return self._dedup_integrator

    def run_collection_phase(self, session: BatchSession) -> Dict[str, Any]:
        """Run Phase 1: Batch collection."""
        logger.info(f"Collecting {session.papers_per_condition} papers for all 60 conditions...")

        try:
            collection_result = self.paper_collector.collect_all_conditions_batch(
                papers_per_condition=session.papers_per_condition,
                min_year=2015
            )

            session.collection_result = {
                'total_conditions': collection_result.total_conditions,
                'successful_conditions': collection_result.successful_conditions,
                'failed_conditions': collection_result.failed_conditions,
                'total_papers_collected': collection_result.total_papers_collected,
                'collection_time_seconds': collection_result.total_collection_time_seconds,
                'success_rate': (collection_result.successful_conditions / collection_result.total_conditions) * 100,
                'quality_gate_passed': collection_result.success
            }

            session.total_papers_collected = collection_result.total_papers_collected

            if not collection_result.success:
                logger.error(f"Collection phase failed: {collection_result.error}")
                return {
                    'success': False,
                    'error': f'Collection quality gate failed: {collection_result.error}',
                    'phase': 'collection'
                }

            logger.info("[SUCCESS] Collection phase completed successfully")
            logger.info(f"  Papers collected: {collection_result.total_papers_collected}")
            logger.info(f"  Success rate: {(collection_result.successful_conditions / collection_result.total_conditions) * 100:.1f}%")

            return {'success': True, 'result': session.collection_result}

        except Exception as e:
            logger.error(f"Collection phase failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'collection'}

    def run_processing_phase(self, session: BatchSession) -> Dict[str, Any]:
        """Run Phase 2: Batch LLM processing."""
        logger.info("Processing all collected papers with LLM...")

        try:
            processing_result = self.llm_processor.process_all_papers_batch()

            session.processing_result = {
                'total_papers_found': processing_result.get('total_papers_found', 0),
                'papers_processed': processing_result.get('papers_processed', 0),
                'papers_failed': processing_result.get('papers_failed', 0),
                'interventions_extracted': processing_result.get('interventions_extracted', 0),
                'processing_time_seconds': processing_result.get('processing_time_seconds', 0),
                'success_rate': processing_result.get('success_rate', 0),
                'model_statistics': processing_result.get('model_statistics', {}),
                'interventions_by_category': processing_result.get('interventions_by_category', {}),
                'failed_papers_count': len(processing_result.get('failed_papers', []))
            }

            session.total_papers_processed = processing_result.get('papers_processed', 0)
            session.total_interventions_extracted = processing_result.get('interventions_extracted', 0)

            if not processing_result['success']:
                logger.error(f"Processing phase failed: {processing_result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': f"Processing failed: {processing_result.get('error', 'Unknown error')}",
                    'phase': 'processing'
                }

            logger.info("[SUCCESS] Processing phase completed successfully")
            logger.info(f"  Papers processed: {session.total_papers_processed}")
            logger.info(f"  Interventions extracted: {session.total_interventions_extracted}")
            logger.info(f"  Success rate: {processing_result.get('success_rate', 0):.1f}%")

            return {'success': True, 'result': session.processing_result}

        except Exception as e:
            logger.error(f"Processing phase failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'processing'}

    def run_semantic_normalization_phase(self, session: BatchSession) -> Dict[str, Any]:
        """Run Phase 3: Semantic normalization."""
        logger.info("Running Phase 3: Semantic normalization...")

        try:
            # Step 1: Normalize interventions
            logger.info("  Step 1: Normalizing interventions...")
            grouping_result = self.dedup_integrator.group_all_data_semantically_batch()

            total_interventions_processed = grouping_result.get('interventions_processed', 0)
            intervention_groups_created = grouping_result.get('canonical_entities_created', 0)

            # Step 2: Normalize conditions
            logger.info("  Step 2: Normalizing condition entities...")
            from .phase_3_semantic_normalizer import SemanticNormalizationOrchestrator
            condition_orchestrator = SemanticNormalizationOrchestrator(db_path=str(config.db_path))
            condition_result = condition_orchestrator.normalize_all_condition_entities(batch_size=50, force=True)

            total_conditions_processed = condition_result.get('processed', 0)
            condition_groups_created = condition_result.get('canonical_groups', 0)
            total_canonical_groups = intervention_groups_created + condition_groups_created

            session.semantic_normalization_result = {
                'total_interventions_processed': total_interventions_processed,
                'intervention_groups_created': intervention_groups_created,
                'total_conditions_processed': total_conditions_processed,
                'condition_groups_created': condition_groups_created,
                'canonical_groups_created': total_canonical_groups,
                'interventions_grouped': grouping_result.get('total_merged', 0),
                'condition_relationships': condition_result.get('relationships', 0),
                'processing_time_seconds': grouping_result.get('processing_time_seconds', 0),
                'semantic_groups_found': grouping_result.get('duplicate_groups_found', 0),
                'method': grouping_result.get('method', 'llm_semantic_grouping')
            }

            session.total_canonical_groups_created = total_canonical_groups

            if not grouping_result['success']:
                logger.error(f"Intervention semantic normalization failed: {grouping_result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': f"Semantic normalization failed: {grouping_result.get('error', 'Unknown error')}",
                    'phase': 'semantic_normalization'
                }

            if condition_result.get('errors', 0) > 0 and condition_result.get('processed', 0) == 0:
                logger.error(f"Condition semantic normalization failed")
                return {
                    'success': False,
                    'error': f"Condition normalization failed: {condition_result.get('error', 'Unknown error')}",
                    'phase': 'semantic_normalization'
                }

            logger.info("[SUCCESS] Semantic normalization completed successfully")
            logger.info(f"  Interventions analyzed: {total_interventions_processed}")
            logger.info(f"  Intervention groups created: {intervention_groups_created}")
            logger.info(f"  Conditions analyzed: {total_conditions_processed}")
            logger.info(f"  Condition groups created: {condition_groups_created}")
            logger.info(f"  Total canonical groups: {total_canonical_groups}")

            return {'success': True, 'result': session.semantic_normalization_result}

        except Exception as e:
            logger.error(f"Semantic normalization failed: {e}")
            return {'success': False, 'error': str(e), 'phase': 'semantic_normalization'}

    def run_data_mining_phase(self, session: BatchSession) -> Dict[str, Any]:
        """Run Phase 4: Data mining (knowledge graph + Bayesian scoring)."""
        logger.info("Running Phase 4: Data Mining...")

        try:
            from .phase_4_data_miner import Phase4DataMiningOrchestrator

            phase4_orchestrator = Phase4DataMiningOrchestrator(
                db_path=str(config.db_path),
                config_path=str(config.data_root / "phase_4_data_mining" / "phase_4_config.yaml")
            )

            mining_result = phase4_orchestrator.run(force=False)

            session.data_mining_result = {
                'phase_4a_completed': mining_result.phase_4a_completed,
                'phase_4b_completed': mining_result.phase_4b_completed,
                'canonical_groups_processed': mining_result.canonical_groups_processed,
                'knowledge_graph_nodes': mining_result.knowledge_graph_nodes,
                'knowledge_graph_edges': mining_result.knowledge_graph_edges,
                'bayesian_scores_generated': mining_result.bayesian_scores_generated,
                'high_confidence_scores': mining_result.high_confidence_scores,
                'phase_4a_duration_seconds': mining_result.phase_4a_duration_seconds,
                'phase_4b_duration_seconds': mining_result.phase_4b_duration_seconds,
                'total_duration_seconds': mining_result.total_duration_seconds
            }

            session.total_knowledge_graph_nodes = mining_result.knowledge_graph_nodes
            session.total_knowledge_graph_edges = mining_result.knowledge_graph_edges
            session.total_bayesian_scores = mining_result.bayesian_scores_generated

            if not mining_result.success:
                logger.error(f"Data mining failed: {mining_result.error}")
                return {
                    'success': False,
                    'error': f"Data mining failed: {mining_result.error}",
                    'phase': 'data_mining'
                }

            logger.info("[SUCCESS] Data mining completed successfully")
            logger.info(f"  Knowledge graph nodes: {mining_result.knowledge_graph_nodes}")
            logger.info(f"  Knowledge graph edges: {mining_result.knowledge_graph_edges}")
            logger.info(f"  Bayesian scores: {mining_result.bayesian_scores_generated}")

            return {'success': True, 'result': session.data_mining_result}

        except Exception as e:
            logger.error(f"Data mining failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e), 'phase': 'data_mining'}

    def run_frontend_export_phase(self, session: BatchSession) -> Dict[str, Any]:
        """Run Phase 5: Frontend data export."""
        logger.info("Running Phase 5: Frontend data export...")

        try:
            from .phase_5_frontend_updater import Phase5FrontendExportOrchestrator

            phase5_orchestrator = Phase5FrontendExportOrchestrator(
                db_path=str(config.db_path),
                config_path=None
            )

            export_results = phase5_orchestrator.run()

            session.frontend_export_result = {
                'table_view_completed': export_results.table_view_completed,
                'network_viz_completed': export_results.network_viz_completed,
                'files_exported': export_results.files_exported,
                'table_view_size_mb': export_results.table_view_size_mb,
                'network_viz_size_mb': export_results.network_viz_size_mb,
                'total_interventions': export_results.total_interventions,
                'total_nodes': export_results.total_nodes,
                'total_edges': export_results.total_edges,
                'validation_passed': export_results.validation_passed,
                'validation_warnings_count': len(export_results.validation_warnings) if export_results.validation_warnings else 0
            }

            session.total_files_exported = export_results.files_exported

            if not export_results.success:
                logger.error(f"Frontend export failed: {export_results.error}")
                return {
                    'success': False,
                    'error': f"Frontend export failed: {export_results.error}",
                    'phase': 'frontend_export'
                }

            logger.info("[SUCCESS] Frontend export completed successfully")
            logger.info(f"  Files exported: {export_results.files_exported}")
            logger.info(f"  Table view: {export_results.table_view_size_mb:.2f} MB")
            logger.info(f"  Network viz: {export_results.network_viz_size_mb:.2f} MB")

            return {'success': True, 'result': session.frontend_export_result}

        except Exception as e:
            logger.error(f"Frontend export failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e), 'phase': 'frontend_export'}
