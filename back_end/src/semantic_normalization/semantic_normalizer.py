"""
Semantic Normalizer Wrapper
Provides a simpler interface for the orchestrator to use the MainNormalizer.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path

from .normalizer import MainNormalizer

logger = logging.getLogger(__name__)


class SemanticNormalizer:
    """
    Simplified wrapper around MainNormalizer for orchestration.

    Provides a condition-based interface that the orchestrator expects.
    """

    def __init__(self, db_path: str):
        """
        Initialize semantic normalizer.

        Args:
            db_path: Path to intervention_research.db (with semantic_hierarchy tables)
        """
        self.db_path = db_path
        self.normalizer = MainNormalizer(db_path=db_path)

    def normalize_interventions(
        self,
        interventions: List[str],
        entity_type: str = 'intervention',
        source_table: str = 'interventions',
        batch_size: int = 50
    ) -> Dict:
        """
        Normalize a list of interventions.

        Args:
            interventions: List of intervention names to normalize
            entity_type: Type of entity (default: 'intervention')
            source_table: Source table name (default: 'interventions')
            batch_size: Batch size for saving state

        Returns:
            Dictionary with normalization statistics
        """
        logger.info(f"Normalizing {len(interventions)} interventions...")

        # Process each intervention
        total_processed = 0
        canonical_groups_created = 0
        relationships_created = 0
        errors = 0

        # Get all intervention names for similarity search
        all_intervention_names = interventions

        for i, intervention_name in enumerate(interventions):
            try:
                # Create fake intervention dict (MainNormalizer expects dict format)
                intervention = {
                    'intervention_name': intervention_name,
                    'intervention_category': None,  # Could be populated from DB
                    'health_condition': None,
                    'frequency': 1
                }

                result = self.normalizer.process_intervention(
                    intervention=intervention,
                    all_intervention_names=all_intervention_names,
                    top_k_similar=5
                )

                if result:
                    total_processed += 1
                    if result.get('relationships_created', 0) > 0:
                        relationships_created += result['relationships_created']

                # Save state periodically
                if (i + 1) % batch_size == 0:
                    self.normalizer._save_session_state()
                    self.normalizer.embedding_engine.save_cache_now()
                    self.normalizer.llm_classifier.save_caches_now()
                    logger.info(f"Processed {i + 1}/{len(interventions)} interventions")

            except Exception as e:
                logger.error(f"Error processing '{intervention_name}': {e}")
                errors += 1

        # Final save
        self.normalizer._save_session_state()
        self.normalizer.embedding_engine.save_cache_now()
        self.normalizer.llm_classifier.save_caches_now()

        # Count canonical groups created
        canonical_groups_created = self.normalizer.stats.get('canonical_groups_created', 0)

        return {
            'total_processed': total_processed,
            'canonical_groups_created': canonical_groups_created,
            'relationships_created': relationships_created,
            'errors': errors
        }
