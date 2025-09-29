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
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, Set
from datetime import datetime
from collections import defaultdict, Counter

# Local imports
from .entity_utils import (
    EntityNormalizationError, DatabaseError, ValidationError, MatchingError, LLMError, ConfigurationError,
    EntityType, MatchingMode, MatchResult, ValidationResult,
    EntityValidator, ConfidenceCalculator,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_BATCH_SIZE
)
from .entity_operations import EntityRepository, LLMProcessor, DuplicateDetector

# External dependencies
try:
    from back_end.src.data_collection.database_manager import database_manager
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
    Unified batch processor for entity normalization, deduplication, and mapping suggestions.

    This class orchestrates entity processing operations by delegating to specialized
    operation classes for better code organization and maintainability.
    """

    def __init__(self, db_connection: sqlite3.Connection, llm_model: str = "gemma2:9b"):
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
        self.duplicate_detector = DuplicateDetector(self.repository)

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

    def process_consensus_batch(self, raw_interventions: List[Dict], paper: Dict,
                              confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Essential duplicate removal and basic consensus processing.

        This method focuses on the critical task of removing true duplicates from same paper
        extractions while preserving individual intervention records for research purposes.

        Args:
            raw_interventions: All interventions from all models for a paper
            paper: Source paper information
            confidence_threshold: Minimum confidence for validation

        Returns:
            List of deduplicated interventions with true duplicates merged
        """
        if not raw_interventions:
            return []

        self.logger.info(f"Processing duplicates for {len(raw_interventions)} raw interventions from paper {paper.get('pmid', 'unknown')}")

        # Phase 1: Normalize all terms in batch for efficiency
        normalized_terms = self.duplicate_detector.batch_normalize_consensus_terms(raw_interventions)

        # Phase 2: Resolve canonical entities for all interventions
        interventions_with_canonicals = self.duplicate_detector.resolve_canonical_entities_batch(raw_interventions, normalized_terms)

        # Phase 3: CRITICAL - Detect and merge true duplicates from same paper
        deduplicated_interventions = self._detect_and_merge_same_paper_duplicates(interventions_with_canonicals, paper)

        # Phase 4: Basic validation and safety checks
        validated_interventions = self._basic_intervention_validation(deduplicated_interventions, confidence_threshold)

        self.logger.info(f"Duplicate processing complete: {len(validated_interventions)} deduplicated interventions from {len(raw_interventions)} raw inputs")
        return validated_interventions

    def _detect_and_merge_same_paper_duplicates(self, interventions: List[Dict], paper: Dict) -> List[Dict]:
        """
        CRITICAL METHOD: Detect and merge true duplicates from same paper where both models
        found identical intervention-condition correlations.
        """
        if not interventions:
            return []

        # Detect duplicate groups
        duplicate_groups = self.duplicate_detector.detect_same_paper_duplicates(interventions)

        # Process each group
        deduplicated_interventions = []
        processed_interventions = set()

        for group in duplicate_groups:
            if len(group) > 1:
                # Merge the duplicate group
                merged_intervention = self.duplicate_detector.merge_duplicate_group(group, paper)
                deduplicated_interventions.append(merged_intervention)

                # Mark all interventions in this group as processed
                for intervention in group:
                    processed_interventions.add(id(intervention))
            else:
                # Single intervention, add as-is if not already processed
                intervention = group[0]
                if id(intervention) not in processed_interventions:
                    deduplicated_interventions.append(intervention)
                    processed_interventions.add(id(intervention))

        # Add any interventions that weren't in any duplicate group
        for intervention in interventions:
            if id(intervention) not in processed_interventions:
                deduplicated_interventions.append(intervention)

        return deduplicated_interventions

    def _basic_intervention_validation(self, interventions: List[Dict],
                                     confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> List[Dict]:
        """
        Basic intervention validation with human review flagging instead of filtering.

        This preserves data while flagging interventions that need human review.
        """
        validated_interventions = []
        validation_stats = {
            'total_processed': 0,
            'passed_validation': 0,
            'low_confidence_flagged': 0,
            'missing_required_fields': 0,
            'requires_human_review': 0
        }

        for intervention in interventions:
            validation_stats['total_processed'] += 1
            validation_flags = []
            requires_human_review = False

            # Validate required fields
            intervention_name = intervention.get('intervention_name', '').strip()
            if not intervention_name:
                validation_flags.append('missing_intervention_name')
                requires_human_review = True
                validation_stats['missing_required_fields'] += 1
            else:
                # Check confidence threshold using effective confidence method
                confidence = self._get_effective_confidence(intervention)
                if confidence < confidence_threshold:
                    self.logger.info(f"Low confidence intervention {confidence:.2f} below threshold {confidence_threshold} - flagged for human review")
                    validation_flags.append(f'low_confidence_{confidence:.2f}')
                    requires_human_review = True
                    validation_stats['low_confidence_flagged'] += 1

            # Add validation metadata
            intervention['validation_flags'] = validation_flags
            intervention['requires_human_review'] = requires_human_review

            if not requires_human_review:
                validation_stats['passed_validation'] += 1
            else:
                validation_stats['requires_human_review'] += 1

            # Always add to results (no filtering, just flagging)
            validated_interventions.append(intervention)

        self.logger.info(f"Validation complete: {validation_stats['passed_validation']} passed, {validation_stats['requires_human_review']} flagged for review")
        return validated_interventions

    #! === DEDUPLICATION SUMMARY ===

    def generate_deduplication_summary(self, processed_interventions: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive summary of deduplication process.

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