#!/usr/bin/env python3
"""
Orchestration pipeline for dual-model extraction and consensus building.

This module coordinates the sequential execution of:
1. dual_model_analyzer - Raw intervention extraction from both models
2. batch_entity_processor - Sophisticated consensus building and normalization

Usage:
    from back_end.src.orchestration.dual_model_consensus_pipeline import process_papers_with_consensus

    results = process_papers_with_consensus(papers, batch_size=3)
"""

import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# Set up logging
logger = logging.getLogger(__name__)

def process_papers_with_consensus(papers: List[Dict],
                                 batch_size: int = 3,
                                 save_to_db: bool = True,
                                 consensus_confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Process papers through dual-model extraction and consensus building pipeline.

    Args:
        papers: List of paper dictionaries with pmid, title, abstract
        batch_size: Number of papers to process in each batch
        save_to_db: Whether to save results to database
        consensus_confidence_threshold: Minimum confidence for consensus building

    Returns:
        Dictionary with processing results and statistics
    """

    try:
        # Import here to avoid circular dependencies
        from ..llm_processing.dual_model_analyzer import DualModelAnalyzer
        from ..llm_processing.batch_entity_processor import create_batch_processor
        from ..data_collection.database_manager import database_manager

        logger.info(f"Starting dual-model consensus pipeline for {len(papers)} papers")

        # Initialize components
        dual_analyzer = DualModelAnalyzer()

        # Phase 1: Raw Extraction
        logger.info("Phase 1: Running dual-model raw extraction...")

        extraction_results = dual_analyzer.process_papers_batch(
            papers=papers,
            batch_size=batch_size,
            save_to_db=save_to_db
        )

        logger.info(f"Raw extraction complete: {extraction_results.get('total_interventions', 0)} raw interventions")

        # Phase 2: Consensus Building (if we have results)
        if extraction_results.get('total_interventions', 0) > 0 and save_to_db:
            logger.info("Phase 2: Building consensus from raw extractions...")

            consensus_results = build_consensus_for_papers(
                papers=papers,
                confidence_threshold=consensus_confidence_threshold
            )

            logger.info(f"Consensus building complete: {consensus_results.get('total_consensus_interventions', 0)} consensus interventions")

            # Combine results
            combined_results = {
                'total_papers_processed': extraction_results.get('total_processed', 0),
                'raw_extraction_results': extraction_results,
                'consensus_results': consensus_results,
                'pipeline_summary': _generate_pipeline_summary(extraction_results, consensus_results)
            }

        else:
            # No consensus building needed/possible
            combined_results = {
                'total_papers_processed': extraction_results.get('total_processed', 0),
                'raw_extraction_results': extraction_results,
                'consensus_results': None,
                'pipeline_summary': _generate_pipeline_summary(extraction_results, None)
            }

        logger.info("Dual-model consensus pipeline completed successfully")
        return combined_results

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def build_consensus_for_papers(papers: List[Dict],
                              confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Build consensus for raw interventions from the specified papers.

    Args:
        papers: List of papers to build consensus for
        confidence_threshold: Minimum confidence for consensus building

    Returns:
        Dictionary with consensus building results
    """

    try:
        from ..llm_processing.batch_entity_processor import create_batch_processor
        from ..data_collection.database_manager import database_manager

        # Get database connection and create processor
        with database_manager.get_connection() as conn:
            processor = create_batch_processor()

            total_consensus_interventions = 0
            total_papers_processed = 0
            consensus_summaries = []

            for paper in papers:
                pmid = paper.get('pmid')

                # Get raw interventions for this paper
                raw_interventions = _get_raw_interventions_for_paper(conn, pmid)

                if raw_interventions:
                    # Build consensus for this paper using new unified method
                    consensus_interventions = processor.process_consensus_batch(
                        raw_interventions, paper, confidence_threshold
                    )

                    if consensus_interventions:
                        # Save consensus interventions
                        _save_consensus_interventions(conn, consensus_interventions, pmid)

                        # Generate summary for this paper
                        paper_summary = processor.generate_deduplication_summary(consensus_interventions)
                        consensus_summaries.append({
                            'pmid': pmid,
                            'summary': paper_summary
                        })

                        total_consensus_interventions += len(consensus_interventions)

                    total_papers_processed += 1

                    logger.debug(f"Consensus built for paper {pmid}: {len(consensus_interventions)} interventions")

            # Generate overall summary
            overall_summary = _generate_consensus_summary(consensus_summaries)

            return {
                'total_papers_processed': total_papers_processed,
                'total_consensus_interventions': total_consensus_interventions,
                'paper_summaries': consensus_summaries,
                'overall_summary': overall_summary
            }

    except Exception as e:
        logger.error(f"Consensus building failed: {e}")
        raise

def _get_raw_interventions_for_paper(conn, pmid: str) -> List[Dict]:
    """Get raw interventions for a specific paper from database."""

    cursor = conn.cursor()
    cursor.execute("""
        SELECT *
        FROM interventions
        WHERE paper_id = ? AND consensus_processed = 0
        ORDER BY extraction_model, id
    """, (pmid,))

    rows = cursor.fetchall()
    return [dict(row) for row in rows]

def _save_consensus_interventions(conn, consensus_interventions: List[Dict], pmid: str):
    """Save consensus interventions to database and mark originals as processed."""

    cursor = conn.cursor()

    # Save each consensus intervention
    for intervention in consensus_interventions:
        intervention['paper_id'] = pmid
        intervention['consensus_processed'] = True
        intervention['extraction_model'] = 'consensus'  # Mark as consensus result

        # Insert consensus intervention
        # Note: This would need to be adapted to your specific database schema
        cursor.execute("""
            INSERT INTO interventions
            (paper_id, intervention_name, health_condition, intervention_category,
             correlation_type, confidence_score, correlation_strength,
             extraction_confidence, study_confidence,
             extraction_model, consensus_processed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            intervention.get('paper_id'),
            intervention.get('intervention_name'),
            intervention.get('health_condition'),
            intervention.get('intervention_category'),
            intervention.get('correlation_type'),
            intervention.get('confidence_score'),
            intervention.get('correlation_strength'),
            intervention.get('extraction_confidence'),
            intervention.get('study_confidence'),
            intervention.get('extraction_model'),
            intervention.get('consensus_processed')
        ))

    # Mark original raw interventions as processed
    cursor.execute("""
        UPDATE interventions
        SET consensus_processed = 1
        WHERE paper_id = ? AND extraction_model != 'consensus'
    """, (pmid,))

    conn.commit()

def _generate_pipeline_summary(extraction_results: Dict, consensus_results: Optional[Dict]) -> Dict:
    """Generate summary statistics for the entire pipeline."""

    summary = {
        'extraction_phase': {
            'papers_processed': extraction_results.get('total_processed', 0),
            'raw_interventions': extraction_results.get('total_interventions', 0),
            'failed_papers': len(extraction_results.get('failed_papers', [])),
            'model_statistics': extraction_results.get('model_statistics', {})
        }
    }

    if consensus_results:
        summary['consensus_phase'] = {
            'papers_processed': consensus_results.get('total_papers_processed', 0),
            'consensus_interventions': consensus_results.get('total_consensus_interventions', 0),
            'compression_ratio': None
        }

        # Calculate compression ratio (raw -> consensus)
        raw_count = extraction_results.get('total_interventions', 0)
        consensus_count = consensus_results.get('total_consensus_interventions', 0)
        if raw_count > 0:
            summary['consensus_phase']['compression_ratio'] = consensus_count / raw_count
    else:
        summary['consensus_phase'] = None

    return summary

def _generate_consensus_summary(paper_summaries: List[Dict]) -> Dict:
    """Generate overall consensus summary from individual paper summaries."""

    if not paper_summaries:
        return {}

    # Aggregate statistics across all papers
    total_interventions = sum(s['summary'].get('total_consensus_interventions', 0) for s in paper_summaries)

    # Aggregate agreement breakdowns
    agreement_totals = {}
    model_usage_totals = {}

    for paper_summary in paper_summaries:
        summary = paper_summary['summary']

        # Aggregate agreement breakdown
        for agreement_type, count in summary.get('agreement_breakdown', {}).items():
            agreement_totals[agreement_type] = agreement_totals.get(agreement_type, 0) + count

        # Aggregate model usage
        for model, count in summary.get('model_usage', {}).items():
            model_usage_totals[model] = model_usage_totals.get(model, 0) + count

    return {
        'total_consensus_interventions': total_interventions,
        'total_papers': len(paper_summaries),
        'agreement_breakdown': agreement_totals,
        'model_usage': model_usage_totals
    }

# Convenience function for backward compatibility
def process_papers_dual_model(papers: List[Dict], **kwargs) -> Dict[str, Any]:
    """Backward compatibility wrapper for existing code."""
    return process_papers_with_consensus(papers, **kwargs)