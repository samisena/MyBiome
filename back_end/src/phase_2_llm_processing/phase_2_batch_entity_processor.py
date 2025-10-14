#!/usr/bin/env python3
"""
Unified Batch Entity Processor - Refactored

This module provides the main orchestration for entity processing operations,
delegating to specialized operation classes for better code organization.

Optimized for batch processing while preserving all sophisticated features.
"""

import sqlite3
import json
import re
import os
import sys
import csv
import logging
import argparse
import traceback
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, Set
from datetime import datetime
from collections import defaultdict, Counter

# Local imports
from .phase_2_entity_utils import (
    EntityNormalizationError, DatabaseError, ValidationError, MatchingError, LLMError, ConfigurationError,
    EntityType, MatchingMode, MatchResult, ValidationResult,
    EntityValidator, ConfidenceCalculator,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_BATCH_SIZE
)
from .phase_2_entity_operations import EntityRepository, LLMProcessor, SemanticGrouper

# External dependencies
try:
    from back_end.src.phase_1_data_collection.database_manager import database_manager
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Configuration
try:
    from back_end.src.data.config import config
except ImportError:
    # Fallback configuration
    class SimpleConfig:
        fast_mode = os.getenv('FAST_MODE', '1').lower() in ('1', 'true', 'yes')
        data_root = 'data'
        db_path = 'data/processed/intervention_research.db'
    config = SimpleConfig()

# Consensus functionality is now integrated directly into BatchEntityProcessor
CONSENSUS_AVAILABLE = True

#* === MAIN BATCH ENTITY PROCESSOR CLASS ===

class BatchEntityProcessor:
    """
    Unified batch processor for entity normalization, semantic grouping, and mapping suggestions.

    This class orchestrates entity processing operations by delegating to specialized
    operation classes for better code organization and maintainability.
    """

    def __init__(self, db_connection: sqlite3.Connection, llm_model: str = "qwen2.5:14b"):
        """
        Initialize the BatchEntityProcessor with a database connection.

        Args:
            db_connection: SQLite database connection object
            llm_model: LLM model name for semantic matching
        """
        # Store references
        self.db = db_connection
        self.db.row_factory = sqlite3.Row  # Enable dict-like access to rows
        self.llm_model = llm_model

        # Initialize operation modules
        self.repository = EntityRepository(db_connection)
        self.llm_processor = LLMProcessor(self.repository, llm_model)
        # Attach llm_processor to repository so SemanticGrouper can access it
        self.repository.llm_processor = self.llm_processor
        self.semantic_grouper = SemanticGrouper(self.repository)

        # Performance monitoring
        self.operation_counts = {}
        self.operation_times = {}

        # Configure logging
        self._setup_logging()

        # Initialize performance optimizations
        self.repository.ensure_performance_optimizations()

    def _setup_logging(self):
        """Set up logging for batch operations."""
        log_level = logging.ERROR if config.fast_mode else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    #? === DELEGATION METHODS ===
    # These methods delegate to the appropriate operation classes

    def get_or_compute_normalized_term(self, term: str, entity_type: str) -> str:
        """Get normalized term from cache or compute it."""
        return self.repository.get_or_compute_normalized_term(term, entity_type)

    def bulk_normalize_terms_optimized(self, terms: List[str], entity_type: str) -> Dict[str, str]:
        """Efficiently normalize multiple terms using the pre-computed cache."""
        return self.repository.bulk_normalize_terms_optimized(terms, entity_type)

    def find_canonical_by_id(self, canonical_id: int) -> Optional[Dict[str, Any]]:
        """Find canonical entity by ID."""
        return self.repository.find_canonical_by_id(canonical_id)

    def find_canonical_by_name(self, canonical_name: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Find canonical entity by name and type."""
        return self.repository.find_canonical_by_name(canonical_name, entity_type)

    def create_canonical_entity(self, canonical_name: str, entity_type: str, description: str = None) -> int:
        """Create a new canonical entity."""
        return self.repository.create_canonical_entity(canonical_name, entity_type, description)

    def find_mapping_by_term(self, term: str, entity_type: str, apply_confidence_decay: bool = True) -> Optional[Dict[str, Any]]:
        """Find existing mapping for a term."""
        return self.repository.find_mapping_by_term(term, entity_type, apply_confidence_decay)

    def create_mapping(self, raw_text: str, canonical_id: int, entity_type: str,
                      confidence_score: float, mapping_method: str) -> int:
        """Create a new entity mapping."""
        return self.repository.create_mapping(raw_text, canonical_id, entity_type, confidence_score, mapping_method)

    def get_llm_duplicate_analysis(self, terms: List[str]) -> Dict[str, Any]:
        """Get LLM analysis of duplicate terms."""
        return self.llm_processor.get_llm_duplicate_analysis(terms)

    # === VALIDATION METHODS ===

    def _validate_entity_type(self, entity_type: str) -> str:
        """Validate and normalize entity type."""
        return EntityValidator.validate_entity_type(entity_type)

    def _validate_term(self, term: str) -> str:
        """Validate and clean term string."""
        return EntityValidator.validate_term(term)

    def _validate_confidence(self, confidence: float) -> float:
        """Validate confidence score."""
        return EntityValidator.validate_confidence(confidence)

    def _get_effective_confidence(self, intervention: Dict[str, Any]) -> float:
        """Get effective confidence from intervention data."""
        return ConfidenceCalculator.get_effective_confidence(intervention)

    def _merge_dual_confidence(self, interventions: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """Merge confidence values from multiple interventions."""
        return ConfidenceCalculator.merge_dual_confidence(interventions)

    # === CORE CONSENSUS PROCESSING ===
    # NOTE: Phase 2 consensus processing REMOVED with single-model architecture
    # Single-model extraction (qwen2.5:14b only) eliminates same-paper duplicates
    # Phase 3 (batch_deduplicate_entities) still performs cross-paper canonical merging

    # DEPRECATED: No longer needed with single-model extraction
    # def process_consensus_batch(self, raw_interventions: List[Dict], paper: Dict,
    #                           confidence_threshold: float = 0.5) -> List[Dict]:
    #     """
    #     [DEPRECATED - Single Model Architecture]
    #     This method was used for merging duplicates from dual-model extraction.
    #     With qwen2.5:14b-only extraction, no same-paper duplicates are created.
    #     """
    #     pass

    # DEPRECATED: No longer needed with single-model extraction
    # def _detect_and_merge_same_paper_duplicates(self, interventions: List[Dict], paper: Dict) -> List[Dict]:
    #     """
    #     [DEPRECATED - Single Model Architecture]
    #     Dual-model extraction created same-paper duplicates requiring merge.
    #     Single-model extraction creates no duplicates to merge.
    #     """
    #     pass

    # DEPRECATED: Validation moved to SingleModelAnalyzer._validate_and_enhance_interventions()
    # def _basic_intervention_validation(self, interventions: List[Dict],
    #                                  confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> List[Dict]:
    #     """
    #     [DEPRECATED - Single Model Architecture]
    #     Validation now occurs directly in single_model_analyzer.py during extraction.
    #     """
    #     pass

    #! === DEDUPLICATION SUMMARY ===

    def batch_group_entities_semantically(self) -> Dict[str, Any]:
        """
        Comprehensive LLM-based semantic grouping of ALL interventions in the database.

        This performs real semantic grouping using the sophisticated SemanticGrouper
        and LLM analysis to merge semantically equivalent interventions across papers.

        Returns:
            Comprehensive semantic grouping result with statistics
        """
        start_time = datetime.now()
        self.logger.info("Starting comprehensive LLM-based semantic grouping")

        try:
            # Get all interventions that could have duplicates
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT i.id, i.intervention_name, i.health_condition, i.paper_id,
                       i.correlation_type, i.correlation_strength,
                       i.extraction_confidence, i.study_confidence,
                       i.extraction_model, i.verification_model,
                       i.intervention_canonical_id, i.condition_canonical_id,
                       i.normalized, i.consensus_confidence, i.sample_size,
                       i.study_type, i.supporting_quote
                FROM interventions i
                ORDER BY i.paper_id, i.intervention_name
            """)
            all_interventions = [dict(row) for row in cursor.fetchall()]

            total_interventions = len(all_interventions)
            self.logger.info(f"Analyzing {total_interventions} interventions for semantic_grouping")

            if total_interventions == 0:
                return {
                    'total_merged': 0,
                    'interventions_processed': 0,
                    'duplicate_groups_found': 0,
                    'processing_time_seconds': 0,
                    'method': 'llm_comprehensive_semantic_grouping',
                    'message': 'No interventions found for semantic_grouping'
                }

            # Single-model architecture (qwen2.5:14b only) means no same-paper duplicates
            # Only Phase 2 (cross-paper semantic semantic_grouping) is needed
            total_merged = 0
            total_duplicate_groups = 0

            # Cross-paper semantic semantic_grouping using LLM analysis
            self.logger.info("Cross-paper semantic semantic_grouping (Phase 3 canonical entity merging)")

            # Get unique intervention names for LLM analysis
            unique_interventions = {}
            for intervention in all_interventions:
                name = intervention['intervention_name'].lower().strip()
                if name not in unique_interventions:
                    unique_interventions[name] = []
                unique_interventions[name].append(intervention)

            # Find potential cross-paper duplicates using LLM
            intervention_names = list(unique_interventions.keys())
            if len(intervention_names) > 1:
                try:
                    # Process ALL intervention names in batches for comprehensive LLM analysis
                    # No artificial limits - true comprehensive semantic semantic_grouping
                    self.logger.info(f"Starting comprehensive LLM analysis of {len(intervention_names)} unique interventions...")

                    batch_size = 20  # Reasonable batch size for LLM processing
                    cross_paper_merged = 0

                    for i in range(0, len(intervention_names), batch_size):
                        batch_names = intervention_names[i:i + batch_size]
                        batch_start_time = datetime.now()

                        self.logger.info(f"Processing LLM batch {i//batch_size + 1}/{(len(intervention_names)-1)//batch_size + 1}: {len(batch_names)} interventions")

                        # Use LLM to identify semantic duplicates in this batch
                        llm_analysis = self.get_llm_duplicate_analysis(batch_names)

                        # Process LLM-identified duplicates for this batch
                        batch_merged = self._process_llm_duplicate_analysis(llm_analysis, unique_interventions)
                        cross_paper_merged += batch_merged

                        batch_time = (datetime.now() - batch_start_time).total_seconds()
                        self.logger.info(f"Batch completed in {batch_time:.1f}s, merged {batch_merged} interventions")

                    total_merged += cross_paper_merged
                    self.logger.info(f"Comprehensive LLM analysis completed: {cross_paper_merged} cross-paper duplicates merged")

                except Exception as e:
                    self.logger.warning(f"Comprehensive cross-paper LLM analysis failed: {e}")

            processing_time = (datetime.now() - start_time).total_seconds()

            result = {
                'total_merged': total_merged,
                'interventions_processed': total_interventions,
                'duplicate_groups_found': total_duplicate_groups,
                'processing_time_seconds': processing_time,
                'method': 'llm_comprehensive_semantic_grouping',
                'phases_completed': ['cross_paper_semantic_analysis']
            }

            self.logger.info(f"Comprehensive LLM semantic_grouping completed: "
                           f"{total_merged} interventions merged from {total_duplicate_groups} duplicate groups "
                           f"in {processing_time:.1f}s")

            return result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Comprehensive LLM semantic_grouping failed: {e}")
            self.logger.error(traceback.format_exc())

            return {
                'total_merged': 0,
                'interventions_processed': 0,
                'duplicate_groups_found': 0,
                'processing_time_seconds': processing_time,
                'method': 'llm_comprehensive_semantic_grouping',
                'error': str(e)
            }

    def _get_effective_confidence(self, intervention: Dict) -> float:
        """Get effective confidence score for an intervention."""
        return (
            intervention.get('consensus_confidence') or
            intervention.get('extraction_confidence') or
            intervention.get('study_confidence') or
            0.5
        )

    def _link_interventions_to_canonical(self, intervention_group: List[Dict], canonical_name: str) -> None:
        """
        Link all interventions in a semantic group to a canonical entity.

        PRESERVES all original intervention rows - no deletions.
        Creates canonical entity and entity mappings for transparency.

        Args:
            intervention_group: List of semantically equivalent interventions
            canonical_name: The selected canonical name for this group
        """
        try:
            cursor = self.db.cursor()

            # Step 1: Create or find canonical entity for intervention
            canonical_entity = self.find_canonical_by_name(canonical_name, 'intervention')
            if canonical_entity:
                canonical_id = canonical_entity['id']
                self.logger.info(f"Using existing canonical entity: {canonical_name} (id={canonical_id})")
            else:
                canonical_id = self.create_canonical_entity(
                    canonical_name=canonical_name,
                    entity_type='intervention',
                    description=f"Canonical entity for {len(intervention_group)} semantic variants"
                )
                self.logger.info(f"Created canonical entity: {canonical_name} (id={canonical_id})")

            # Step 2: Create entity mappings for all variants
            unique_names = set()
            for intervention in intervention_group:
                variant_name = intervention.get('intervention_name', '').strip()
                if variant_name and variant_name not in unique_names:
                    unique_names.add(variant_name)

                    # Check if mapping already exists
                    existing_mapping = self.repository.find_mapping_by_term(variant_name, 'intervention')
                    if not existing_mapping:
                        # Calculate confidence based on similarity to canonical name
                        confidence = 1.0 if variant_name.lower() == canonical_name.lower() else 0.9

                        self.create_mapping(
                            raw_text=variant_name,
                            canonical_id=canonical_id,
                            entity_type='intervention',
                            confidence_score=confidence,
                            mapping_method='llm_semantic_grouping'
                        )
                        self.logger.debug(f"Created mapping: '{variant_name}' → canonical_id={canonical_id}")

            # Step 3: Update ALL interventions to link to canonical entity (NO DELETIONS)
            intervention_ids = [intervention['id'] for intervention in intervention_group]

            # Calculate merged confidence for this group
            confidences = [self._get_effective_confidence(i) for i in intervention_group]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.9
            consensus_confidence = min(0.98, avg_confidence + 0.1)  # Boost for cross-paper validation

            for intervention_id in intervention_ids:
                cursor.execute("""
                    UPDATE interventions
                    SET intervention_canonical_id = ?,
                        normalized = 1,
                        consensus_confidence = ?
                    WHERE id = ?
                """, (canonical_id, consensus_confidence, intervention_id))

            self.db.commit()

            self.logger.info(f"Linked {len(intervention_ids)} interventions to canonical entity '{canonical_name}' "
                           f"(preserved all {len(intervention_ids)} rows)")

        except Exception as e:
            self.logger.error(f"Failed to link interventions to canonical entity: {e}")
            self.db.rollback()
            raise

    def _process_llm_duplicate_analysis(self, llm_analysis: Dict, unique_interventions: Dict) -> int:
        """
        Process LLM duplicate analysis and link interventions to canonical entities.

        NO DELETIONS - preserves all intervention rows and links them to canonical entities.

        Returns:
            Number of interventions linked to canonical entities (not deleted count)
        """
        linked_count = 0

        try:
            # Extract duplicate groups from LLM analysis
            if 'duplicate_groups' in llm_analysis:
                for group in llm_analysis['duplicate_groups']:
                    if len(group) > 1:
                        # Collect all interventions for this duplicate group
                        all_group_interventions = []
                        for intervention_name in group:
                            if intervention_name.lower() in unique_interventions:
                                all_group_interventions.extend(unique_interventions[intervention_name.lower()])

                        if len(all_group_interventions) > 1:
                            # Select canonical name using SemanticGrouper
                            canonical_name = self.semantic_grouper.select_canonical_name(
                                all_group_interventions,
                                group  # Pass the LLM-identified group names
                            )

                            # Link ALL interventions to canonical entity (preserves all rows)
                            self._link_interventions_to_canonical(all_group_interventions, canonical_name)

                            # Count interventions linked (not deleted!)
                            linked_count += len(all_group_interventions)

                            self.logger.info(f"Linked {len(all_group_interventions)} interventions to canonical '{canonical_name}'")

        except Exception as e:
            self.logger.warning(f"Error processing LLM duplicate analysis: {e}")
            import traceback
            self.logger.warning(traceback.format_exc())

        return linked_count

    def generate_semantic_grouping_summary(self, processed_interventions: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive summary of semantic_grouping process.

        Returns:
            Summary statistics dictionary including human review flags
        """
        if not processed_interventions:
            return {'total_processed_interventions': 0}

        # Basic statistics
        model_usage = defaultdict(int)
        confidence_scores = []
        duplicate_merges = 0

        # Human review tracking
        human_review_stats = {
            'total_requiring_review': 0,
            'low_confidence_flagged': 0,
            'safety_flagged': 0,
            'missing_fields_flagged': 0,
            'passed_validation': 0
        }

        validation_flag_counts = defaultdict(int)

        for intervention in processed_interventions:
            # Count model usage
            models = intervention.get('models_contributing', [intervention.get('extraction_model', 'unknown')])
            if isinstance(models, list):
                for model in models:
                    model_usage[model] += 1
            else:
                model_usage[models] += 1

            # Count duplicate merges
            if intervention.get('cross_model_validation', False):
                duplicate_merges += 1

            # Collect confidence scores using effective confidence method
            conf = self._get_effective_confidence(intervention)
            confidence_scores.append(conf)

            # Track human review requirements
            if intervention.get('requires_human_review', False):
                human_review_stats['total_requiring_review'] += 1

                # Count specific flag types
                flags = intervention.get('validation_flags', [])
                for flag in flags:
                    validation_flag_counts[flag] += 1

                    if flag.startswith('low_confidence'):
                        human_review_stats['low_confidence_flagged'] += 1
                    elif flag in ['dangerous_combination', 'potentially_confusing_terms']:
                        human_review_stats['safety_flagged'] += 1
                    elif flag == 'missing_intervention_name':
                        human_review_stats['missing_fields_flagged'] += 1
            else:
                human_review_stats['passed_validation'] += 1

        # Calculate aggregated statistics
        total_interventions = len(processed_interventions)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        return {
            'total_consensus_interventions': total_interventions,
            'model_usage': dict(model_usage),
            'duplicate_merges': duplicate_merges,
            'average_confidence': round(avg_confidence, 3),
            'confidence_range': (min(confidence_scores), max(confidence_scores)) if confidence_scores else (0, 0),
            'human_review_required': human_review_stats,
            'validation_flag_breakdown': dict(validation_flag_counts)
        }

    # === SYSTEM STATUS AND UTILITIES ===

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        cursor = self.db.cursor()

        # Count entities and mappings
        cursor.execute("SELECT COUNT(*) FROM canonical_entities")
        total_entities = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM entity_mappings")
        total_mappings = cursor.fetchone()[0]

        # Count by entity type
        cursor.execute("SELECT entity_type, COUNT(*) FROM canonical_entities GROUP BY entity_type")
        entities_by_type = dict(cursor.fetchall())

        # Cache statistics
        cursor.execute("SELECT COUNT(*) FROM normalized_terms_cache")
        cache_entries = cursor.fetchone()[0]

        return {
            'database_path': str(self.db),
            'total_canonical_entities': total_entities,
            'total_entity_mappings': total_mappings,
            'entities_by_type': entities_by_type,
            'cache_entries': cache_entries,
            'llm_model': self.llm_model,
            'llm_available': LLM_AVAILABLE,
            'operation_counts': dict(self.operation_counts),
            'fast_mode': config.fast_mode
        }

# === FACTORY FUNCTIONS ===

def create_batch_processor(db_path: Optional[str] = None, llm_model: str = "gemma2:9b") -> BatchEntityProcessor:
    """Create a BatchEntityProcessor instance with database connection."""
    if db_path is None:
        db_path = getattr(config, 'db_path', 'back_end/data/processed/intervention_research.db')

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    return BatchEntityProcessor(conn, llm_model)

# === COMMAND LINE INTERFACE ===

def main():
    """Command line interface for batch entity processing operations."""
    parser = argparse.ArgumentParser(description="Batch Entity Processor CLI")

    parser.add_argument('command', choices=['status', 'deduplicate', 'suggestions'],
                       help='Command to execute')
    parser.add_argument('--database', '-d', type=str,
                       help='Database path (optional, uses config default)')
    parser.add_argument('--entity-type', '-e', type=str, choices=['intervention', 'condition', 'side_effect'],
                       help='Entity type to process')
    parser.add_argument('--confidence', '-c', type=float, default=0.7,
                       help='Confidence threshold for processing')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path for results')
    parser.add_argument('--fast', action='store_true',
                       help='Enable fast mode (less logging)')

    args = parser.parse_args()

    if args.fast:
        config.fast_mode = True

    try:
        processor = create_batch_processor(args.database)

        if args.command == 'status':
            status = processor.get_system_status()
            print(json.dumps(status, indent=2))

        else:
            print(f"Command '{args.command}' not yet implemented")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()