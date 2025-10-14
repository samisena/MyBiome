#!/usr/bin/env python3
"""
Data Mining Orchestrator - Comprehensive analytics pipeline for medical research data

This script orchestrates the complete data mining workflow including:
- Medical knowledge graph construction
- Bayesian evidence scoring
- Treatment recommendation generation
- Research gap identification
- Innovation tracking
- Pattern discovery
- Correlation analysis

Features:
- Modular pipeline with configurable stages
- Progress tracking and session persistence
- Error handling and retry logic
- Parallel processing where applicable
- Comprehensive logging and reporting

Usage:
    # Run complete data mining pipeline
    python data_mining_orchestrator.py --all

    # Run specific analysis modules
    python data_mining_orchestrator.py --knowledge-graph --bayesian-scoring

    # Custom configuration
    python data_mining_orchestrator.py --config custom_config.json

    # Resume previous session
    python data_mining_orchestrator.py --resume

Examples:
    # Full pipeline for all conditions
    python data_mining_orchestrator.py --all --output-dir results/

    # Quick analysis for specific conditions
    python data_mining_orchestrator.py --conditions "ibs,gerd" --knowledge-graph --recommendations

    # Research gap analysis only
    python data_mining_orchestrator.py --research-gaps --innovation-tracking
"""

import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from back_end.src.data.config import config, setup_logging
    from back_end.src.data.repositories import repository_manager
    from back_end.src.phase_1_data_collection.database_manager import database_manager

    # Data mining modules
    from back_end.src.data_mining.bayesian_scorer import BayesianEvidenceScorer
    from back_end.src.data_mining.medical_knowledge_graph import MedicalKnowledgeGraph
    from back_end.src.data_mining.treatment_recommendation_engine import TreatmentRecommendationEngine
    from back_end.src.data_mining.research_gaps import ResearchGapIdentification
    from back_end.src.data_mining.innovation_tracking_system import InnovationTrackingSystem
    from back_end.src.data_mining.biological_patterns import BiologicalPatternDiscovery
    from back_end.src.data_mining.condition_similarity_mapping import ConditionSimilarityMapper
    from back_end.src.data_mining.power_combinations import PowerCombinationAnalysis
    from back_end.src.data_mining.failed_interventions import FailedInterventionCatalog
    from back_end.src.data_mining.correlation_consistency_checker import CorrelationConsistencyChecker

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)

logger = setup_logging(__name__, 'data_mining_orchestrator.log')


@dataclass
class DataMiningConfig:
    """Configuration for data mining orchestrator."""
    # Pipeline stages
    build_knowledge_graph: bool = True
    run_bayesian_scoring: bool = True
    generate_recommendations: bool = True
    analyze_research_gaps: bool = True
    track_innovations: bool = True
    analyze_patterns: bool = True
    map_similarities: bool = True
    analyze_combinations: bool = True
    analyze_failed_interventions: bool = True
    check_correlations: bool = True

    # Data filtering
    conditions: List[str] = None
    min_evidence_count: int = 3
    min_year: int = 2010
    confidence_threshold: float = 0.7

    # Processing options
    parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 100

    # Output options
    output_dir: Path = Path("data_mining_results")
    save_intermediate: bool = True
    export_formats: List[str] = None  # ['json', 'csv', 'xlsx']

    # Session management
    session_file: Path = Path("data_mining_session.json")
    auto_save_interval: int = 300  # seconds

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.export_formats is None:
            self.export_formats = ['json']
        self.output_dir = Path(self.output_dir)


@dataclass
class StageProgress:
    """Progress tracking for individual pipeline stages."""
    stage_name: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    records_processed: int = 0
    total_records: int = 0
    error_message: Optional[str] = None

    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def progress_percent(self) -> float:
        if self.total_records > 0:
            return (self.records_processed / self.total_records) * 100
        return 0.0


class DataMiningOrchestrator:
    """
    Orchestrates comprehensive data mining operations for medical research data.
    """

    def __init__(self, config: DataMiningConfig):
        self.config = config
        self.session_start = datetime.now()

        # Pipeline stage tracking
        self.stages: Dict[str, StageProgress] = {}
        self.results: Dict[str, Any] = {}

        # Initialize components
        self.knowledge_graph = None
        self.bayesian_scorer = None
        self.recommendation_engine = None
        self.research_gap_analyzer = None
        self.innovation_tracker = None
        self.pattern_analyzer = None
        self.similarity_mapper = None
        self.combination_analyzer = None
        self.failed_intervention_analyzer = None
        self.correlation_checker = None

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup auto-save
        self._setup_auto_save()

    def _setup_auto_save(self):
        """Setup automatic session saving."""
        def auto_save():
            while True:
                time.sleep(self.config.auto_save_interval)
                try:
                    self._save_session()
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")

        auto_save_thread = threading.Thread(target=auto_save, daemon=True)
        auto_save_thread.start()

    def _save_session(self):
        """Save current session state."""
        session_data = {
            'config': asdict(self.config),
            'session_start': self.session_start.isoformat(),
            'stages': {name: asdict(stage) for name, stage in self.stages.items()},
            'results_summary': self._get_results_summary()
        }

        with open(self.config.session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

    def _load_session(self) -> bool:
        """Load previous session state."""
        try:
            if not self.config.session_file.exists():
                return False

            with open(self.config.session_file, 'r') as f:
                session_data = json.load(f)

            # Restore stage progress
            for name, stage_data in session_data.get('stages', {}).items():
                stage = StageProgress(**stage_data)
                # Convert string timestamps back to datetime
                if stage.start_time:
                    stage.start_time = datetime.fromisoformat(stage.start_time)
                if stage.end_time:
                    stage.end_time = datetime.fromisoformat(stage.end_time)
                self.stages[name] = stage

            if not config.fast_mode:
                logger.info(f"Resumed session from {session_data.get('session_start')}")
            return True

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False

    def _get_results_summary(self) -> Dict[str, Any]:
        """Get summary of current results."""
        summary = {}
        for stage_name, result in self.results.items():
            if isinstance(result, dict):
                summary[stage_name] = {
                    'type': type(result).__name__,
                    'keys': list(result.keys()) if hasattr(result, 'keys') else None,
                    'size': len(result) if hasattr(result, '__len__') else None
                }
            else:
                summary[stage_name] = {
                    'type': type(result).__name__,
                    'size': len(result) if hasattr(result, '__len__') else None
                }
        return summary

    def _init_stage(self, stage_name: str, total_records: int = 0) -> StageProgress:
        """Initialize a new pipeline stage."""
        stage = StageProgress(
            stage_name=stage_name,
            status="running",
            start_time=datetime.now(),
            total_records=total_records
        )
        self.stages[stage_name] = stage
        if not config.fast_mode:
            logger.info(f"Starting stage: {stage_name}")
        return stage

    def _complete_stage(self, stage_name: str, success: bool = True, error: str = None):
        """Mark a pipeline stage as completed."""
        if stage_name in self.stages:
            stage = self.stages[stage_name]
            stage.end_time = datetime.now()
            stage.status = "completed" if success else "failed"
            if error:
                stage.error_message = error

            if success:
                if not config.fast_mode:
                    logger.info(f"Completed stage: {stage_name} in {stage.duration}")
            else:
                logger.error(f"Failed stage: {stage_name} - {error}")

    def run_complete_pipeline(self):
        """Run the complete data mining pipeline."""
        if not config.fast_mode:
            logger.info("Starting complete data mining pipeline")

        try:
            # Stage 1: Build Knowledge Graph
            if self.config.build_knowledge_graph:
                self._build_knowledge_graph()

            # Stage 2: Bayesian Scoring
            if self.config.run_bayesian_scoring and self.knowledge_graph:
                self._run_bayesian_scoring()

            # Stage 3: Generate Recommendations
            if self.config.generate_recommendations:
                self._generate_recommendations()

            # Stage 4: Analyze Research Gaps
            if self.config.analyze_research_gaps:
                self._analyze_research_gaps()

            # Stage 5: Track Innovations
            if self.config.track_innovations:
                self._track_innovations()

            # Stage 6: Pattern Analysis
            if self.config.analyze_patterns:
                self._analyze_patterns()

            # Stage 7: Similarity Mapping
            if self.config.map_similarities:
                self._map_similarities()

            # Stage 8: Combination Analysis
            if self.config.analyze_combinations:
                self._analyze_combinations()

            # Stage 9: Failed Intervention Analysis
            if self.config.analyze_failed_interventions:
                self._analyze_failed_interventions()

            # Stage 10: Correlation Consistency Check
            if self.config.check_correlations:
                self._check_correlations()

            # Generate final report
            self._generate_final_report()

            if not config.fast_mode:
                logger.info("Data mining pipeline completed successfully")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            raise

    def _build_knowledge_graph(self):
        """Build medical knowledge graph from database."""
        stage = self._init_stage("knowledge_graph")

        try:
            if not config.fast_mode:
                logger.info("Building medical knowledge graph...")
            self.knowledge_graph = MedicalKnowledgeGraph(save_to_database=True)

            # Get all intervention data from database
            with database_manager.get_connection() as conn:
                if self.config.conditions:
                    # Filter by specific conditions
                    placeholders = ','.join(['?' for _ in self.config.conditions])
                    query = f"""
                    SELECT DISTINCT p.pmid, p.title, substr(p.publication_date, 1, 4),
                           i.intervention_name, i.health_condition, i.evidence_type,
                           i.confidence_score, i.sample_size, i.study_design
                    FROM papers p
                    JOIN interventions i ON p.id = i.paper_id
                    WHERE i.health_condition IN ({placeholders})
                    AND substr(p.publication_date, 1, 4) >= ?
                    ORDER BY CAST(substr(p.publication_date, 1, 4) AS INTEGER) DESC
                    """
                    params = list(self.config.conditions) + [self.config.min_year]
                else:
                    # Get all data
                    query = """
                    SELECT DISTINCT p.pmid, p.title, substr(p.publication_date, 1, 4),
                           i.intervention_name, i.health_condition, i.evidence_type,
                           i.confidence_score, i.sample_size, i.study_design
                    FROM papers p
                    JOIN interventions i ON p.id = i.paper_id
                    WHERE CAST(substr(p.publication_date, 1, 4) AS INTEGER) >= ?
                    ORDER BY CAST(substr(p.publication_date, 1, 4) AS INTEGER) DESC
                    """
                    params = [self.config.min_year]

                cursor = conn.execute(query, params)
                records = cursor.fetchall()
                stage.total_records = len(records)

                # Process records in batches
                for i, record in enumerate(records):
                    pmid, title, year, intervention, condition, evidence_type, confidence, sample_size, study_design = record

                    # Add to knowledge graph
                    self.knowledge_graph.add_intervention_evidence(
                        study_id=str(pmid),
                        title=title,
                        intervention_name=intervention,
                        condition=condition,
                        evidence_type=evidence_type,
                        confidence=confidence,
                        sample_size=sample_size or 0,
                        study_design=study_design or "unknown",
                        publication_year=year or 0
                    )

                    stage.records_processed = i + 1

                    if i % self.config.batch_size == 0:
                        if not config.fast_mode:
                            logger.info(f"Processed {i+1}/{len(records)} records")

            # Save knowledge graph
            if self.config.save_intermediate:
                output_file = self.config.output_dir / "knowledge_graph.json"
                self.knowledge_graph.save_to_file(output_file)
                if not config.fast_mode:
                    logger.info(f"Saved knowledge graph to {output_file}")

            self.results['knowledge_graph'] = self.knowledge_graph
            self._complete_stage("knowledge_graph", success=True)

        except Exception as e:
            self._complete_stage("knowledge_graph", success=False, error=str(e))
            raise

    def _run_bayesian_scoring(self):
        """Run Bayesian evidence scoring."""
        stage = self._init_stage("bayesian_scoring")

        try:
            if not config.fast_mode:
                logger.info("Running Bayesian evidence scoring...")
            self.bayesian_scorer = BayesianEvidenceScorer(save_to_database=True)

            # Get all unique intervention-condition pairs
            pairs = self.knowledge_graph.get_all_intervention_condition_pairs()
            stage.total_records = len(pairs)

            scores = {}
            for i, (intervention, condition) in enumerate(pairs):
                # Score this intervention-condition pair
                score_result = self.bayesian_scorer.score_intervention(
                    intervention, condition, knowledge_graph=self.knowledge_graph
                )

                pair_key = f"{intervention}|{condition}"
                scores[pair_key] = score_result

                stage.records_processed = i + 1

                if i % 100 == 0 and not config.fast_mode:
                    logger.info(f"Scored {i+1}/{len(pairs)} intervention-condition pairs")

            # Save scoring results
            if self.config.save_intermediate:
                output_file = self.config.output_dir / "bayesian_scores.json"
                with open(output_file, 'w') as f:
                    json.dump(scores, f, indent=2)
                if not config.fast_mode:
                    logger.info(f"Saved Bayesian scores to {output_file}")

            self.results['bayesian_scores'] = scores
            self._complete_stage("bayesian_scoring", success=True)

        except Exception as e:
            self._complete_stage("bayesian_scoring", success=False, error=str(e))
            raise

    def _generate_recommendations(self):
        """Generate treatment recommendations."""
        stage = self._init_stage("recommendations")

        try:
            logger.info("Generating treatment recommendations...")
            self.recommendation_engine = TreatmentRecommendationEngine(save_to_database=True)

            # Get recommendations for each condition
            if self.config.conditions:
                conditions_to_process = self.config.conditions
            else:
                conditions_to_process = self.knowledge_graph.get_all_conditions()

            stage.total_records = len(conditions_to_process)
            recommendations = {}

            for i, condition in enumerate(conditions_to_process):
                condition_recs = self.recommendation_engine.get_recommendations_for_condition(
                    condition,
                    min_confidence=self.config.confidence_threshold
                )
                recommendations[condition] = condition_recs

                stage.records_processed = i + 1
                logger.info(f"Generated recommendations for {condition}")

            # Save recommendations
            if self.config.save_intermediate:
                output_file = self.config.output_dir / "treatment_recommendations.json"
                with open(output_file, 'w') as f:
                    json.dump(recommendations, f, indent=2, default=str)
                if not config.fast_mode:
                    logger.info(f"Saved recommendations to {output_file}")

            self.results['recommendations'] = recommendations
            self._complete_stage("recommendations", success=True)

        except Exception as e:
            self._complete_stage("recommendations", success=False, error=str(e))
            raise

    def _analyze_research_gaps(self):
        """Analyze research gaps."""
        stage = self._init_stage("research_gaps")

        try:
            logger.info("Analyzing research gaps...")
            self.research_gap_analyzer = ResearchGapIdentification(self.knowledge_graph)

            # Identify research gaps
            gaps = self.research_gap_analyzer.identify_comprehensive_gaps(
                min_evidence_threshold=self.config.min_evidence_count
            )

            stage.records_processed = len(gaps)
            stage.total_records = len(gaps)

            # Save research gaps
            if self.config.save_intermediate:
                output_file = self.config.output_dir / "research_gaps.json"
                with open(output_file, 'w') as f:
                    json.dump(gaps, f, indent=2, default=str)
                if not config.fast_mode:
                    logger.info(f"Saved research gaps to {output_file}")

            self.results['research_gaps'] = gaps
            self._complete_stage("research_gaps", success=True)

        except Exception as e:
            self._complete_stage("research_gaps", success=False, error=str(e))
            raise

    def _track_innovations(self):
        """Track emerging innovations."""
        stage = self._init_stage("innovation_tracking")

        try:
            logger.info("Tracking innovations...")
            self.innovation_tracker = InnovationTrackingSystem(self.knowledge_graph)

            # Identify emerging treatments
            innovations = self.innovation_tracker.identify_emerging_treatments()

            stage.records_processed = len(innovations)
            stage.total_records = len(innovations)

            # Save innovation tracking results
            if self.config.save_intermediate:
                output_file = self.config.output_dir / "innovation_tracking.json"
                with open(output_file, 'w') as f:
                    json.dump(innovations, f, indent=2, default=str)
                if not config.fast_mode:
                    logger.info(f"Saved innovation tracking to {output_file}")

            self.results['innovations'] = innovations
            self._complete_stage("innovation_tracking", success=True)

        except Exception as e:
            self._complete_stage("innovation_tracking", success=False, error=str(e))
            raise

    def _analyze_patterns(self):
        """Analyze biological patterns."""
        stage = self._init_stage("pattern_analysis")

        try:
            logger.info("Analyzing biological patterns...")
            self.pattern_analyzer = BiologicalPatternDiscovery(self.knowledge_graph)

            # Run pattern analysis
            patterns = self.pattern_analyzer.analyze_all_patterns()

            stage.records_processed = len(patterns)
            stage.total_records = len(patterns)

            # Save pattern analysis
            if self.config.save_intermediate:
                output_file = self.config.output_dir / "biological_patterns.json"
                with open(output_file, 'w') as f:
                    json.dump(patterns, f, indent=2, default=str)
                if not config.fast_mode:
                    logger.info(f"Saved biological patterns to {output_file}")

            self.results['patterns'] = patterns
            self._complete_stage("pattern_analysis", success=True)

        except Exception as e:
            self._complete_stage("pattern_analysis", success=False, error=str(e))
            raise

    def _map_similarities(self):
        """Map condition similarities."""
        stage = self._init_stage("similarity_mapping")

        try:
            logger.info("Mapping condition similarities...")
            self.similarity_mapper = ConditionSimilarityMapper(self.knowledge_graph)

            # Calculate similarity matrix
            similarity_matrix = self.similarity_mapper.calculate_similarity_matrix()

            stage.records_processed = len(similarity_matrix)
            stage.total_records = len(similarity_matrix)

            # Save similarity mapping
            if self.config.save_intermediate:
                output_file = self.config.output_dir / "condition_similarities.json"
                with open(output_file, 'w') as f:
                    json.dump(similarity_matrix, f, indent=2, default=str)
                if not config.fast_mode:
                    logger.info(f"Saved condition similarities to {output_file}")

            self.results['similarities'] = similarity_matrix
            self._complete_stage("similarity_mapping", success=True)

        except Exception as e:
            self._complete_stage("similarity_mapping", success=False, error=str(e))
            raise

    def _analyze_combinations(self):
        """Analyze intervention combinations."""
        stage = self._init_stage("combination_analysis")

        try:
            logger.info("Analyzing intervention combinations...")
            self.combination_analyzer = PowerCombinationAnalyzer(self.knowledge_graph)

            # Find powerful combinations
            combinations = self.combination_analyzer.find_powerful_combinations()

            stage.records_processed = len(combinations)
            stage.total_records = len(combinations)

            # Save combination analysis
            if self.config.save_intermediate:
                output_file = self.config.output_dir / "power_combinations.json"
                with open(output_file, 'w') as f:
                    json.dump(combinations, f, indent=2, default=str)
                if not config.fast_mode:
                    logger.info(f"Saved power combinations to {output_file}")

            self.results['combinations'] = combinations
            self._complete_stage("combination_analysis", success=True)

        except Exception as e:
            self._complete_stage("combination_analysis", success=False, error=str(e))
            raise

    def _analyze_failed_interventions(self):
        """Analyze failed interventions."""
        stage = self._init_stage("failed_intervention_analysis")

        try:
            logger.info("Analyzing failed interventions...")
            self.failed_intervention_analyzer = FailedInterventionAnalyzer(self.knowledge_graph)

            # Analyze failed interventions
            failed_analysis = self.failed_intervention_analyzer.analyze_failures()

            stage.records_processed = len(failed_analysis)
            stage.total_records = len(failed_analysis)

            # Save failed intervention analysis
            if self.config.save_intermediate:
                output_file = self.config.output_dir / "failed_interventions.json"
                with open(output_file, 'w') as f:
                    json.dump(failed_analysis, f, indent=2, default=str)
                if not config.fast_mode:
                    logger.info(f"Saved failed intervention analysis to {output_file}")

            self.results['failed_interventions'] = failed_analysis
            self._complete_stage("failed_intervention_analysis", success=True)

        except Exception as e:
            self._complete_stage("failed_intervention_analysis", success=False, error=str(e))
            raise

    def _check_correlations(self):
        """Check correlation consistency."""
        stage = self._init_stage("correlation_check")

        try:
            logger.info("Checking correlation consistency...")
            self.correlation_checker = CorrelationConsistencyChecker(self.knowledge_graph)

            # Run consistency checks
            consistency_report = self.correlation_checker.run_comprehensive_check()

            stage.records_processed = 1
            stage.total_records = 1

            # Save correlation check
            if self.config.save_intermediate:
                output_file = self.config.output_dir / "correlation_consistency.json"
                with open(output_file, 'w') as f:
                    json.dump(consistency_report, f, indent=2, default=str)
                if not config.fast_mode:
                    logger.info(f"Saved correlation consistency check to {output_file}")

            self.results['correlation_consistency'] = consistency_report
            self._complete_stage("correlation_check", success=True)

        except Exception as e:
            self._complete_stage("correlation_check", success=False, error=str(e))
            raise

    def _generate_final_report(self):
        """Generate comprehensive final report."""
        logger.info("Generating final report...")

        # Calculate overall statistics
        total_duration = datetime.now() - self.session_start
        completed_stages = [s for s in self.stages.values() if s.status == "completed"]
        failed_stages = [s for s in self.stages.values() if s.status == "failed"]

        report = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration': str(total_duration),
                'completed_stages': len(completed_stages),
                'failed_stages': len(failed_stages),
                'success_rate': len(completed_stages) / len(self.stages) * 100 if self.stages else 0
            },
            'configuration': asdict(self.config),
            'stage_summary': {
                name: {
                    'status': stage.status,
                    'duration': str(stage.duration) if stage.duration else None,
                    'records_processed': stage.records_processed,
                    'progress_percent': stage.progress_percent,
                    'error_message': stage.error_message
                }
                for name, stage in self.stages.items()
            },
            'results_summary': self._get_results_summary(),
            'key_findings': self._extract_key_findings()
        }

        # Save final report
        report_file = self.config.output_dir / f"data_mining_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Final report saved to {report_file}")
        logger.info(f"Data mining completed in {total_duration}")

        return report

    def _extract_key_findings(self) -> Dict[str, Any]:
        """Extract key findings from all analyses."""
        findings = {}

        # Knowledge graph insights
        if self.knowledge_graph:
            findings['knowledge_graph'] = {
                'total_interventions': len(self.knowledge_graph.get_all_interventions()),
                'total_conditions': len(self.knowledge_graph.get_all_conditions()),
                'total_edges': self.knowledge_graph.get_total_edge_count()
            }

        # Top recommendations
        if 'recommendations' in self.results:
            findings['top_recommendations'] = self._get_top_recommendations()

        # Research gaps
        if 'research_gaps' in self.results:
            findings['critical_research_gaps'] = self._get_critical_gaps()

        # Emerging innovations
        if 'innovations' in self.results:
            findings['emerging_innovations'] = self._get_top_innovations()

        return findings

    def _get_top_recommendations(self) -> List[Dict[str, Any]]:
        """Get top treatment recommendations across all conditions."""
        all_recs = []
        for condition, recs in self.results.get('recommendations', {}).items():
            for rec in recs[:5]:  # Top 5 per condition
                rec['condition'] = condition
                all_recs.append(rec)

        # Sort by confidence score
        all_recs.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return all_recs[:20]  # Top 20 overall

    def _get_critical_gaps(self) -> List[Dict[str, Any]]:
        """Get most critical research gaps."""
        gaps = self.results.get('research_gaps', {})
        if isinstance(gaps, dict):
            gap_list = []
            for gap_type, gap_data in gaps.items():
                if isinstance(gap_data, list):
                    gap_list.extend(gap_data)
            return gap_list[:10]  # Top 10 gaps
        return []

    def _get_top_innovations(self) -> List[Dict[str, Any]]:
        """Get top emerging innovations."""
        innovations = self.results.get('innovations', [])
        if isinstance(innovations, list):
            return innovations[:10]  # Top 10 innovations
        return []


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data Mining Orchestrator for Medical Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Pipeline control
    parser.add_argument('--all', action='store_true',
                       help='Run complete data mining pipeline')
    parser.add_argument('--knowledge-graph', action='store_true',
                       help='Build medical knowledge graph')
    parser.add_argument('--bayesian-scoring', action='store_true',
                       help='Run Bayesian evidence scoring')
    parser.add_argument('--recommendations', action='store_true',
                       help='Generate treatment recommendations')
    parser.add_argument('--research-gaps', action='store_true',
                       help='Analyze research gaps')
    parser.add_argument('--innovation-tracking', action='store_true',
                       help='Track emerging innovations')
    parser.add_argument('--pattern-analysis', action='store_true',
                       help='Analyze biological patterns')
    parser.add_argument('--similarity-mapping', action='store_true',
                       help='Map condition similarities')
    parser.add_argument('--combination-analysis', action='store_true',
                       help='Analyze intervention combinations')
    parser.add_argument('--failed-interventions', action='store_true',
                       help='Analyze failed interventions')
    parser.add_argument('--correlation-check', action='store_true',
                       help='Check correlation consistency')

    # Data filtering
    parser.add_argument('--conditions', type=str,
                       help='Comma-separated list of conditions to analyze')
    parser.add_argument('--min-evidence', type=int, default=3,
                       help='Minimum evidence count threshold')
    parser.add_argument('--min-year', type=int, default=2010,
                       help='Minimum publication year')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Confidence threshold for recommendations')

    # Processing options
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel processing')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of worker threads')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')

    # Output options
    parser.add_argument('--output-dir', type=str, default='data_mining_results',
                       help='Output directory for results')
    parser.add_argument('--export-formats', type=str, default='json',
                       help='Export formats (comma-separated): json,csv,xlsx')
    parser.add_argument('--no-intermediate', action='store_true',
                       help='Skip saving intermediate results')

    # Session management
    parser.add_argument('--resume', action='store_true',
                       help='Resume previous session')
    parser.add_argument('--session-file', type=str, default='data_mining_session.json',
                       help='Session state file')

    # Utility options
    parser.add_argument('--status', action='store_true',
                       help='Show current pipeline status')
    parser.add_argument('--config', type=str,
                       help='Load configuration from JSON file')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    try:
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            config = DataMiningConfig(**config_data)
        else:
            # Build configuration from arguments
            config = DataMiningConfig(
                build_knowledge_graph=args.all or args.knowledge_graph,
                run_bayesian_scoring=args.all or args.bayesian_scoring,
                generate_recommendations=args.all or args.recommendations,
                analyze_research_gaps=args.all or args.research_gaps,
                track_innovations=args.all or args.innovation_tracking,
                analyze_patterns=args.all or args.pattern_analysis,
                map_similarities=args.all or args.similarity_mapping,
                analyze_combinations=args.all or args.combination_analysis,
                analyze_failed_interventions=args.all or args.failed_interventions,
                check_correlations=args.all or args.correlation_check,

                conditions=args.conditions.split(',') if args.conditions else [],
                min_evidence_count=args.min_evidence,
                min_year=args.min_year,
                confidence_threshold=args.confidence_threshold,

                parallel_processing=args.parallel,
                max_workers=args.max_workers,
                batch_size=args.batch_size,

                output_dir=Path(args.output_dir),
                export_formats=args.export_formats.split(','),
                save_intermediate=not args.no_intermediate,

                session_file=Path(args.session_file)
            )

        # Initialize orchestrator
        orchestrator = DataMiningOrchestrator(config)

        # Handle different run modes
        if args.status:
            # Show status and exit
            if orchestrator._load_session():
                print("Current pipeline status:")
                for name, stage in orchestrator.stages.items():
                    print(f"  {name}: {stage.status} ({stage.progress_percent:.1f}%)")
            else:
                print("No active session found")
            return

        # Resume previous session if requested
        if args.resume:
            orchestrator._load_session()

        # Run pipeline
        if any([args.all, args.knowledge_graph, args.bayesian_scoring,
               args.recommendations, args.research_gaps, args.innovation_tracking,
               args.pattern_analysis, args.similarity_mapping, args.combination_analysis,
               args.failed_interventions, args.correlation_check]):
            orchestrator.run_complete_pipeline()
        else:
            print("No pipeline stages specified. Use --help for options.")

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\nPipeline interrupted. Session state saved.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()