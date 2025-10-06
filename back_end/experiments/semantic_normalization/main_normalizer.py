"""
Main Normalizer - Orchestrates Hierarchical Semantic Normalization Pipeline

Processes interventions from the database:
1. Load interventions
2. Generate embeddings
3. Extract canonicals via LLM
4. Find similar interventions and classify relationships
5. Assign hierarchical layers
6. Populate database tables

Supports resumable sessions and progress tracking.
"""

import os
import json
import pickle
import sqlite3
import logging
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Import local modules
from embedding_engine import EmbeddingEngine
from llm_classifier import LLMClassifier
from hierarchy_manager import HierarchyManager, initialize_database_schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MainNormalizer:
    """
    Main orchestrator for hierarchical semantic normalization.
    """

    def __init__(
        self,
        source_db_path: str,
        target_db_path: str,
        config_path: Optional[str] = None,
        session_file: Optional[str] = None
    ):
        """
        Initialize the main normalizer.

        Args:
            source_db_path: Path to source database (intervention_research.db)
            target_db_path: Path to target database (hierarchical schema)
            config_path: Path to YAML config file (optional)
            session_file: Path to session state file for resumability
        """
        self.source_db_path = source_db_path
        self.target_db_path = target_db_path
        self.config_path = config_path
        self.session_file = session_file or "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/normalizer_session.pkl"

        # Initialize components
        self.embedding_engine = self._load_embedding_engine()
        self.llm_classifier = self._load_llm_classifier()
        self.hierarchy_manager = HierarchyManager(target_db_path)

        # Session state
        self.session_state = self._load_session_state()
        self.processed_interventions = set(self.session_state.get('processed', []))

        # Stats
        self.stats = {
            'total_interventions': 0,
            'processed': len(self.processed_interventions),
            'skipped': 0,
            'errors': 0,
            'relationships_created': 0,
            'canonical_groups_created': 0
        }

        logger.info("MainNormalizer initialized")

    def _load_embedding_engine(self) -> EmbeddingEngine:
        """Load embedding engine with config."""
        cache_path = "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/embeddings.pkl"

        if self.config_path and os.path.exists(self.config_path):
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                emb_config = config.get('embedding', {})
                cache_path = config.get('cache', {}).get('embedding_cache_path', cache_path)

                return EmbeddingEngine(
                    model=emb_config.get('model', 'nomic-embed-text'),
                    base_url="http://localhost:11434",
                    cache_path=cache_path,
                    batch_size=emb_config.get('batch_size', 32)
                )

        return EmbeddingEngine(cache_path=cache_path)

    def _load_llm_classifier(self) -> LLMClassifier:
        """Load LLM classifier with config."""
        canonical_cache = "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/canonicals.pkl"
        relationship_cache = "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/cache/llm_decisions.pkl"

        if self.config_path and os.path.exists(self.config_path):
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                llm_config = config.get('llm', {})
                cache_config = config.get('cache', {})

                return LLMClassifier(
                    model=llm_config.get('model', 'qwen3:14b'),
                    base_url=llm_config.get('base_url', 'http://localhost:11434'),
                    temperature=llm_config.get('temperature', 0.0),
                    timeout=llm_config.get('timeout', 60),
                    max_retries=llm_config.get('max_retries', 3),
                    strip_think_tags=llm_config.get('strip_think_tags', True),
                    canonical_cache_path=cache_config.get('canonical_cache_path', canonical_cache),
                    relationship_cache_path=cache_config.get('llm_cache_path', relationship_cache)
                )

        return LLMClassifier(
            canonical_cache_path=canonical_cache,
            relationship_cache_path=relationship_cache
        )

    def _load_session_state(self) -> Dict:
        """Load session state from file."""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'rb') as f:
                    state = pickle.load(f)
                logger.info(f"Loaded session state: {len(state.get('processed', []))} interventions processed")
                return state
            except Exception as e:
                logger.warning(f"Failed to load session state: {e}")

        return {'processed': [], 'start_time': datetime.now().isoformat()}

    def _save_session_state(self):
        """Save session state to file."""
        try:
            self.session_state['processed'] = list(self.processed_interventions)
            self.session_state['last_save'] = datetime.now().isoformat()
            self.session_state['stats'] = self.stats

            os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
            with open(self.session_file, 'wb') as f:
                pickle.dump(self.session_state, f)

            logger.debug("Session state saved")
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")

    def load_interventions(self) -> List[Dict]:
        """
        Load interventions from source database.

        Returns:
            List of intervention dicts
        """
        conn = sqlite3.connect(self.source_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
        SELECT
            intervention_name,
            intervention_category,
            health_condition,
            COUNT(*) as frequency
        FROM interventions
        WHERE intervention_name IS NOT NULL
        GROUP BY intervention_name, intervention_category, health_condition
        ORDER BY frequency DESC
        """

        cursor.execute(query)
        interventions = [dict(row) for row in cursor.fetchall()]

        conn.close()

        logger.info(f"Loaded {len(interventions)} intervention records")
        return interventions

    def process_intervention(
        self,
        intervention: Dict,
        all_intervention_names: List[str],
        top_k_similar: int = 5
    ) -> Optional[Dict]:
        """
        Process a single intervention through the normalization pipeline.

        Args:
            intervention: Intervention dict
            all_intervention_names: List of all intervention names for similarity search
            top_k_similar: Number of similar interventions to find

        Returns:
            Processing result dict
        """
        intervention_name = intervention['intervention_name']

        # Skip if already processed
        if intervention_name in self.processed_interventions:
            self.stats['skipped'] += 1
            return None

        try:
            # Step 1: Extract canonical group
            canonical_result = self.llm_classifier.extract_canonical(intervention_name)
            canonical_group = canonical_result['canonical_group']

            # Step 2: Generate embedding
            embedding = self.embedding_engine.generate_embedding(intervention_name)
            embedding_bytes = embedding.tobytes()

            # Step 3: Find similar interventions
            similar_interventions = self.embedding_engine.find_similar(
                query_text=intervention_name,
                candidate_texts=all_intervention_names,
                top_k=top_k_similar,
                min_similarity=0.70
            )

            # Step 4: Extract Layer 2 variant and Layer 3 detail
            layer_2_variant = intervention_name.lower()  # Default: normalized name
            layer_3_detail = self.hierarchy_manager.extract_dosage(intervention_name)

            # Step 5: Create or get canonical group
            canonical_id = self.hierarchy_manager.get_or_create_canonical_group(
                canonical_name=canonical_group,
                entity_type='intervention',
                layer_0_category=intervention.get('intervention_category'),
                description=canonical_result.get('reasoning')
            )

            if canonical_id and canonical_group not in [cg for cg in self.session_state.get('canonical_groups', [])]:
                self.stats['canonical_groups_created'] += 1
                self.session_state.setdefault('canonical_groups', []).append(canonical_group)

            # Step 6: Create semantic entity
            entity_id = self.hierarchy_manager.create_semantic_entity(
                entity_name=intervention_name,
                entity_type='intervention',
                layer_0_category=intervention.get('intervention_category'),
                layer_1_canonical=canonical_group,
                layer_2_variant=layer_2_variant,
                layer_3_detail=layer_3_detail,
                relationship_type=None,  # Will be set when creating relationships
                aggregation_rule=None,
                embedding_vector=embedding_bytes,
                embedding_model='nomic-embed-text',
                source_table='interventions',
                source_ids=None  # Could be populated later
            )

            # Step 7: Process similar interventions and create relationships
            relationships_created = 0
            for similar_name, similarity in similar_interventions:
                # Classify relationship
                rel_result = self.llm_classifier.classify_relationship(
                    intervention_name,
                    similar_name,
                    similarity
                )

                # Get or create similar entity (if not already processed)
                similar_entity = self.hierarchy_manager.get_entity_by_name(similar_name, 'intervention')

                if similar_entity:
                    # Create relationship
                    self.hierarchy_manager.create_entity_relationship(
                        entity_1_id=entity_id,
                        entity_2_id=similar_entity['id'],
                        relationship_type=rel_result['relationship_type'],
                        relationship_confidence=0.85,  # Could be calculated
                        source='llm_inference',
                        labeled_by=self.llm_classifier.model,
                        similarity_score=similarity
                    )
                    relationships_created += 1

            self.stats['relationships_created'] += relationships_created

            # Step 8: Update canonical group stats
            self.hierarchy_manager.update_canonical_group_stats(canonical_group, 'intervention')

            # Mark as processed
            self.processed_interventions.add(intervention_name)
            self.stats['processed'] += 1

            return {
                'intervention_name': intervention_name,
                'entity_id': entity_id,
                'canonical_group': canonical_group,
                'layer_2_variant': layer_2_variant,
                'layer_3_detail': layer_3_detail,
                'relationships_created': relationships_created
            }

        except Exception as e:
            logger.error(f"Error processing '{intervention_name}': {e}")
            self.stats['errors'] += 1
            return None

    def run(self, batch_size: int = 50, log_interval: int = 10):
        """
        Run the normalization pipeline.

        Args:
            batch_size: Save state every N interventions
            log_interval: Log progress every N interventions
        """
        logger.info("Starting normalization pipeline...")

        # Load interventions
        interventions = self.load_interventions()
        self.stats['total_interventions'] = len(interventions)

        # Get unique intervention names
        all_intervention_names = [i['intervention_name'] for i in interventions]

        # Process interventions with progress bar
        with tqdm(total=len(interventions), desc="Normalizing interventions") as pbar:
            for i, intervention in enumerate(interventions):
                # Process
                result = self.process_intervention(
                    intervention,
                    all_intervention_names,
                    top_k_similar=5
                )

                # Update progress
                pbar.update(1)

                # Log progress
                if (i + 1) % log_interval == 0:
                    logger.info(f"Processed {i + 1}/{len(interventions)} interventions")

                # Save state periodically
                if (i + 1) % batch_size == 0:
                    self._save_session_state()
                    self.embedding_engine.save_cache_now()
                    self.llm_classifier.save_caches_now()

        # Final save
        self._save_session_state()
        self.embedding_engine.save_cache_now()
        self.llm_classifier.save_caches_now()

        # Print final stats
        self._print_final_stats()

    def _print_final_stats(self):
        """Print final statistics."""
        print("\n" + "="*80)
        print("NORMALIZATION PIPELINE COMPLETE")
        print("="*80)

        print(f"\nTotal interventions: {self.stats['total_interventions']}")
        print(f"Successfully processed: {self.stats['processed']}")
        print(f"Skipped (already processed): {self.stats['skipped']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Canonical groups created: {self.stats['canonical_groups_created']}")
        print(f"Relationships created: {self.stats['relationships_created']}")

        # Embedding engine stats
        emb_stats = self.embedding_engine.get_stats()
        print(f"\nEmbedding engine:")
        print(f"  Cache size: {emb_stats['cache_size']}")
        print(f"  Hit rate: {emb_stats['hit_rate']:.2%}")

        # LLM classifier stats
        llm_stats = self.llm_classifier.get_stats()
        print(f"\nLLM classifier:")
        print(f"  Canonical cache: {llm_stats['canonical_cache_size']} (hit rate: {llm_stats['canonical_hit_rate']:.2%})")
        print(f"  Relationship cache: {llm_stats['relationship_cache_size']} (hit rate: {llm_stats['relationship_hit_rate']:.2%})")

        # Hierarchy stats
        hierarchy_stats = self.hierarchy_manager.get_hierarchy_stats()
        print(f"\nHierarchy database:")
        print(f"  Total entities: {hierarchy_stats['total_entities']}")
        print(f"  Canonical groups: {hierarchy_stats['total_canonical_groups']}")
        print(f"  Relationships: {hierarchy_stats['total_relationships']}")

        if hierarchy_stats.get('relationships_by_type'):
            print(f"\n  Relationships by type:")
            for rel_type, count in hierarchy_stats['relationships_by_type'].items():
                print(f"    - {rel_type}: {count}")

        print("\n" + "="*80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical Semantic Normalization Pipeline")
    parser.add_argument(
        '--source-db',
        default='c:/Users/samis/Desktop/MyBiome/back_end/data/processed/intervention_research.db',
        help='Source database path'
    )
    parser.add_argument(
        '--target-db',
        default='c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/hierarchical_normalization.db',
        help='Target database path'
    )
    parser.add_argument(
        '--config',
        default='c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/config/config_phase2.yaml',
        help='Config YAML path'
    )
    parser.add_argument(
        '--init-schema',
        action='store_true',
        help='Initialize database schema before running'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Save state every N interventions'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous session'
    )

    args = parser.parse_args()

    # Initialize schema if requested
    if args.init_schema:
        logger.info("Initializing database schema...")
        initialize_database_schema(args.target_db)

    # Create normalizer
    normalizer = MainNormalizer(
        source_db_path=args.source_db,
        target_db_path=args.target_db,
        config_path=args.config if os.path.exists(args.config) else None
    )

    # Run pipeline
    normalizer.run(batch_size=args.batch_size)


if __name__ == "__main__":
    main()
