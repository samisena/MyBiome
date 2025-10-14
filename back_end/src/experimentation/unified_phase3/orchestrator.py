"""
Unified Phase 3 Orchestrator - Main Pipeline Coordinator

Coordinates all three phases of the unified semantic clustering pipeline:
1. Phase 3a: Semantic Embedding (for interventions, conditions, mechanisms)
2. Phase 3b: Clustering (HDBSCAN or Hierarchical + Singleton Handler)
3. Phase 3c: LLM Canonical Naming (qwen3:14b with temperature control)

Supports configuration via YAML, session persistence, and experiment tracking.
"""

import json
import sqlite3
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np

from .embedders import InterventionEmbedder, ConditionEmbedder, MechanismEmbedder
from .clusterers import HDBSCANClusterer, HierarchicalClusterer, SingletonHandler
from .namers import LLMNamer, ClusterData

logger = logging.getLogger(__name__)


@dataclass
class EntityResults:
    """Results for a single entity type."""
    entity_type: str

    # Phase 3a: Embedding
    embedding_duration_seconds: float
    embeddings_generated: int
    embedding_cache_hit_rate: float

    # Phase 3b: Clustering
    clustering_duration_seconds: float
    num_clusters: int
    num_natural_clusters: int
    num_singleton_clusters: int
    num_noise_points: int
    assignment_rate: float
    silhouette_score: Optional[float]
    davies_bouldin_score: Optional[float]
    cluster_size_stats: Dict

    # Phase 3c: Naming
    naming_duration_seconds: float
    names_generated: int
    naming_failures: int
    naming_cache_hit_rate: float

    # Data
    entity_names: List[str]
    embeddings: np.ndarray
    cluster_labels: np.ndarray
    naming_results: Dict


class UnifiedPhase3Orchestrator:
    """
    Main orchestrator for unified Phase 3 pipeline.

    Runs all three phases for interventions, conditions, and mechanisms.
    Tracks results and saves to experiment database.
    """

    def __init__(
        self,
        config_path: str,
        db_path: str,
        experiment_db_path: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize orchestrator.

        Args:
            config_path: Path to YAML configuration file
            db_path: Path to intervention_research.db
            experiment_db_path: Path to experiment database (optional)
            cache_dir: Cache directory for embeddings/clusters/naming (optional)
        """
        self.config_path = Path(config_path)
        self.db_path = Path(db_path)

        # Load configuration
        self.config = self._load_config()

        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(self.config['cache']['base_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup experiment database
        if experiment_db_path:
            self.experiment_db_path = Path(experiment_db_path)
        else:
            self.experiment_db_path = Path(self.config['database']['experiment_db_path'])

        # Initialize experiment database if needed
        self._initialize_experiment_db()

        # Experiment metadata
        self.experiment_id = None
        self.experiment_name = self.config['experiment']['name']
        self.start_time = None

        logger.info(f"UnifiedPhase3Orchestrator initialized: experiment={self.experiment_name}")

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _load_config(self) -> Dict:
        """Load YAML configuration."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Handle _base reference
        if '_base' in config:
            base_path = self.config_path.parent / config['_base']
            with open(base_path, 'r') as f:
                base_config = yaml.safe_load(f)
            # Deep merge configs (config overrides base)
            config = self._deep_merge(base_config, config)

        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    def _initialize_experiment_db(self):
        """Initialize experiment database with schema."""
        if not self.experiment_db_path.exists():
            logger.info(f"Creating experiment database: {self.experiment_db_path}")
            schema_path = Path(__file__).parent / "experiment_schema.sql"

            conn = sqlite3.connect(self.experiment_db_path)
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            conn.executescript(schema_sql)
            conn.commit()
            conn.close()

            logger.info("Experiment database initialized")

    def run(self) -> Dict[str, Any]:
        """
        Run complete unified Phase 3 pipeline.

        Returns:
            Dict with experiment results
        """
        self.start_time = time.time()

        logger.info("="*60)
        logger.info("UNIFIED PHASE 3 PIPELINE STARTING")
        logger.info("="*60)
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Config: {self.config_path}")

        # Create experiment record
        self.experiment_id = self._create_experiment_record()

        try:
            # Run pipeline for each entity type
            results = {}

            # Process interventions
            logger.info("\n" + "="*60)
            logger.info("PROCESSING INTERVENTIONS")
            logger.info("="*60)
            results['interventions'] = self._process_entity_type('intervention')
            # Save incrementally after interventions complete
            self._save_entity_results('intervention', results['interventions'])

            # Process conditions
            logger.info("\n" + "="*60)
            logger.info("PROCESSING CONDITIONS")
            logger.info("="*60)
            results['conditions'] = self._process_entity_type('condition')
            # Save incrementally after conditions complete
            self._save_entity_results('condition', results['conditions'])

            # Process mechanisms
            logger.info("\n" + "="*60)
            logger.info("PROCESSING MECHANISMS")
            logger.info("="*60)
            results['mechanisms'] = self._process_entity_type('mechanism')
            # Save incrementally after mechanisms complete
            self._save_entity_results('mechanism', results['mechanisms'])

            # Update experiment status
            duration = time.time() - self.start_time
            self._update_experiment_status('completed', duration)

            logger.info("\n" + "="*60)
            logger.info("UNIFIED PHASE 3 PIPELINE COMPLETED")
            logger.info("="*60)
            logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
            logger.info(f"Experiment ID: {self.experiment_id}")

            return {
                'success': True,
                'experiment_id': self.experiment_id,
                'experiment_name': self.experiment_name,
                'duration_seconds': duration,
                'results': results
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            duration = time.time() - self.start_time
            self._update_experiment_status('failed', duration)
            self._log_experiment_error(str(e))

            return {
                'success': False,
                'experiment_id': self.experiment_id,
                'error': str(e),
                'duration_seconds': duration
            }

    def _process_entity_type(self, entity_type: str) -> EntityResults:
        """
        Process a single entity type through all three phases.

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'

        Returns:
            EntityResults with all metrics
        """
        # Phase 3a: Embedding
        embeddings, entity_names, embedding_stats = self._run_phase3a(entity_type)

        # Phase 3b: Clustering
        cluster_labels, clustering_stats = self._run_phase3b(entity_type, embeddings)

        # Phase 3c: Naming
        naming_results, naming_stats = self._run_phase3c(
            entity_type, entity_names, cluster_labels
        )

        # Combine results
        results = EntityResults(
            entity_type=entity_type,
            # Embedding
            embedding_duration_seconds=embedding_stats['duration'],
            embeddings_generated=embedding_stats['embeddings_generated'],
            embedding_cache_hit_rate=embedding_stats['cache_hit_rate'],
            # Clustering
            clustering_duration_seconds=clustering_stats['duration'],
            num_clusters=clustering_stats['num_clusters'],
            num_natural_clusters=clustering_stats['num_natural_clusters'],
            num_singleton_clusters=clustering_stats['num_singleton_clusters'],
            num_noise_points=clustering_stats['num_noise_points'],
            assignment_rate=clustering_stats['assignment_rate'],
            silhouette_score=clustering_stats.get('silhouette_score'),
            davies_bouldin_score=clustering_stats.get('davies_bouldin_score'),
            cluster_size_stats=clustering_stats['cluster_size_stats'],
            # Naming
            naming_duration_seconds=naming_stats['duration'],
            names_generated=naming_stats['names_generated'],
            naming_failures=naming_stats['naming_failures'],
            naming_cache_hit_rate=naming_stats['cache_hit_rate'],
            # Data
            entity_names=entity_names,
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            naming_results=naming_results
        )

        return results

    def _run_phase3a(self, entity_type: str) -> Tuple[np.ndarray, List[str], Dict]:
        """
        Phase 3a: Semantic Embedding.

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'

        Returns:
            Tuple of (embeddings, entity_names, stats)
        """
        logger.info(f"[Phase 3a] Generating embeddings for {entity_type}s...")

        start_time = time.time()

        # Get configuration
        config = self.config['embedding'][f"{entity_type}s"]
        cache_path = self.cache_dir / f"embeddings_{entity_type}_{config['model']}.pkl"

        # Create embedder
        if entity_type == 'intervention':
            embedder = InterventionEmbedder(
                model=config['model'],
                dimension=config['dimension'],
                batch_size=config['batch_size'],
                cache_path=str(cache_path),
                normalization=config['normalization'],
                include_context=config.get('include_context', False)
            )
            embeddings, entity_names = embedder.embed_interventions_from_db(str(self.db_path))

        elif entity_type == 'condition':
            embedder = ConditionEmbedder(
                model=config['model'],
                dimension=config['dimension'],
                batch_size=config['batch_size'],
                cache_path=str(cache_path),
                normalization=config['normalization'],
                include_context=config.get('include_context', False)
            )
            embeddings, entity_names = embedder.embed_conditions_from_db(str(self.db_path))

        elif entity_type == 'mechanism':
            embedder = MechanismEmbedder(
                model=config['model'],
                dimension=config['dimension'],
                batch_size=config['batch_size'],
                cache_path=str(cache_path),
                normalization=config['normalization'],
                include_context=config.get('include_context', False)
            )
            embeddings, entity_names = embedder.embed_mechanisms_from_db(str(self.db_path))

        else:
            raise ValueError(f"Unknown entity_type: {entity_type}")

        duration = time.time() - start_time
        embedder_stats = embedder.get_stats()

        stats = {
            'duration': duration,
            'embeddings_generated': embedder_stats['total_embeddings_generated'],
            'cache_hit_rate': embedder_stats['hit_rate']
        }

        logger.info(f"[Phase 3a] Generated {len(embeddings)} embeddings in {duration:.1f}s")
        logger.info(f"[Phase 3a] Cache hit rate: {stats['cache_hit_rate']:.2%}")

        return embeddings, entity_names, stats

    def _run_phase3b(self, entity_type: str, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Phase 3b: Clustering.

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'
            embeddings: Embedding vectors

        Returns:
            Tuple of (cluster_labels, stats)
        """
        logger.info(f"[Phase 3b] Clustering {entity_type}s...")

        start_time = time.time()

        # Get configuration
        config = self.config['clustering'][f"{entity_type}s"]
        cache_path = self.cache_dir / f"clusters_{entity_type}_{config['algorithm']}.pkl"

        # Create clusterer
        if config['algorithm'] == 'hdbscan':
            clusterer = HDBSCANClusterer(
                min_cluster_size=config['min_cluster_size'],
                min_samples=config['min_samples'],
                cluster_selection_epsilon=config['cluster_selection_epsilon'],
                metric=config['metric'],
                cache_path=str(cache_path)
            )
        elif config['algorithm'] == 'hierarchical':
            clusterer = HierarchicalClusterer(
                linkage=config['linkage'],
                distance_threshold=config.get('distance_threshold'),
                n_clusters=config.get('n_clusters'),
                metric=config['metric'],
                cache_path=str(cache_path)
            )
        else:
            raise ValueError(f"Unknown clustering algorithm: {config['algorithm']}")

        # Cluster
        cluster_labels, metadata = clusterer.cluster(embeddings)

        # Apply singleton handler for 100% assignment
        handler = SingletonHandler()
        final_labels, singleton_metadata = handler.process_labels(cluster_labels)

        duration = time.time() - start_time

        stats = {
            'duration': duration,
            'num_clusters': singleton_metadata['total_clusters'],
            'num_natural_clusters': singleton_metadata['original_clusters'],
            'num_singleton_clusters': singleton_metadata['singletons_created'],
            'num_noise_points': singleton_metadata.get('original_noise_count', 0),
            'assignment_rate': singleton_metadata['assignment_rate'],
            'silhouette_score': metadata.get('silhouette_score'),
            'davies_bouldin_score': metadata.get('davies_bouldin_score'),
            'cluster_size_stats': metadata.get('cluster_size_stats', {})
        }

        logger.info(f"[Phase 3b] Created {stats['num_clusters']} clusters in {duration:.1f}s")
        logger.info(f"[Phase 3b] Natural: {stats['num_natural_clusters']}, Singletons: {stats['num_singleton_clusters']}")
        logger.info(f"[Phase 3b] Assignment rate: {stats['assignment_rate']:.0%}")

        return final_labels, stats

    def _run_phase3c(
        self,
        entity_type: str,
        entity_names: List[str],
        cluster_labels: np.ndarray
    ) -> Tuple[Dict, Dict]:
        """
        Phase 3c: LLM Canonical Naming.

        Args:
            entity_type: 'intervention', 'condition', or 'mechanism'
            entity_names: List of entity names
            cluster_labels: Cluster assignments

        Returns:
            Tuple of (naming_results, stats)
        """
        logger.info(f"[Phase 3c] Naming {entity_type} clusters...")

        start_time = time.time()

        # Get configuration
        naming_config = self.config['naming']
        temperature = naming_config['llm']['temperature']
        cache_path = self.cache_dir / f"naming_{entity_type}_temp{temperature:.1f}.json"

        # Create namer
        namer = LLMNamer(
            model=naming_config['llm']['model'],
            base_url=naming_config['llm']['base_url'],
            temperature=temperature,
            max_tokens=naming_config['llm']['max_tokens'],
            timeout=naming_config['llm']['timeout'],
            max_retries=naming_config['llm']['max_retries'],
            strip_think_tags=naming_config['llm']['strip_think_tags'],
            max_members_shown=naming_config['prompts']['max_members_shown'],
            include_frequency=naming_config['prompts']['include_frequency'],
            cache_path=str(cache_path),
            allowed_categories=naming_config['validation']['allowed_categories']
        )

        # Build cluster data
        clusters = self._build_cluster_data(entity_type, entity_names, cluster_labels)

        # Limit to 100 clusters for faster experimentation
        clusters_limited = clusters[:100]
        logger.info(f"Limiting naming to first 100 clusters (out of {len(clusters)} total)")

        # Name clusters
        naming_results = namer.name_clusters(
            clusters_limited,
            batch_size=naming_config['prompts']['batch_size']
        )

        duration = time.time() - start_time
        namer_stats = namer.get_stats()

        stats = {
            'duration': duration,
            'names_generated': namer_stats['names_generated'],
            'naming_failures': namer_stats['failures'],
            'cache_hit_rate': namer_stats['hit_rate']
        }

        logger.info(f"[Phase 3c] Named {len(naming_results)} clusters in {duration:.1f}s")
        logger.info(f"[Phase 3c] Failures: {stats['naming_failures']}, Cache hit rate: {stats['cache_hit_rate']:.2%}")

        return naming_results, stats

    def _build_cluster_data(
        self,
        entity_type: str,
        entity_names: List[str],
        cluster_labels: np.ndarray
    ) -> List[ClusterData]:
        """
        Build ClusterData objects from cluster assignments.

        Args:
            entity_type: Entity type
            entity_names: List of entity names
            cluster_labels: Cluster assignments

        Returns:
            List of ClusterData objects
        """
        clusters_dict = {}

        # Group entities by cluster
        for name, label in zip(entity_names, cluster_labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(name)

        # Create ClusterData objects
        clusters = []
        for cluster_id, members in clusters_dict.items():
            is_singleton = len(members) == 1

            clusters.append(ClusterData(
                cluster_id=int(cluster_id),
                entity_type=entity_type,
                member_entities=members,
                member_frequencies=None,  # Could be populated from DB
                singleton=is_singleton
            ))

        return clusters

    def _create_experiment_record(self) -> int:
        """Create experiment record in database."""
        conn = sqlite3.connect(self.experiment_db_path)
        cursor = conn.cursor()

        # Check if experiment already exists
        cursor.execute("SELECT experiment_id FROM experiments WHERE experiment_name = ?",
                      (self.experiment_name,))
        existing = cursor.fetchone()

        if existing:
            # Update existing experiment
            experiment_id = existing[0]
            cursor.execute("""
                UPDATE experiments
                SET description = ?, config_path = ?,
                    embedding_model = ?, clustering_algorithm = ?, naming_temperature = ?,
                    embedding_hyperparameters = ?, clustering_hyperparameters = ?, naming_hyperparameters = ?,
                    status = ?, started_at = ?, tags = ?
                WHERE experiment_id = ?
            """, (
                self.config['experiment'].get('description', ''),
                str(self.config_path),
                self.config['embedding']['interventions']['model'],
                self.config['clustering']['interventions']['algorithm'],
                self.config['naming']['llm']['temperature'],
                json.dumps(self.config['embedding']),
                json.dumps(self.config['clustering']),
                json.dumps(self.config['naming']),
                'running',
                datetime.now().isoformat(),
                json.dumps(self.config['experiment'].get('tags', [])),
                experiment_id
            ))
        else:
            # Create new experiment
            cursor.execute("""
                INSERT INTO experiments (
                    experiment_name, description, config_path,
                    embedding_model, clustering_algorithm, naming_temperature,
                    embedding_hyperparameters, clustering_hyperparameters, naming_hyperparameters,
                    status, started_at, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.experiment_name,
                self.config['experiment'].get('description', ''),
                str(self.config_path),
                self.config['embedding']['interventions']['model'],
                self.config['clustering']['interventions']['algorithm'],
                self.config['naming']['llm']['temperature'],
                json.dumps(self.config['embedding']),
                json.dumps(self.config['clustering']),
                json.dumps(self.config['naming']),
                'running',
                datetime.now().isoformat(),
                json.dumps(self.config['experiment'].get('tags', []))
            ))
            experiment_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return experiment_id

    def _update_experiment_status(self, status: str, duration: float):
        """Update experiment status."""
        conn = sqlite3.connect(self.experiment_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE experiments
            SET status = ?, completed_at = ?, duration_seconds = ?
            WHERE experiment_id = ?
        """, (status, datetime.now().isoformat(), duration, self.experiment_id))

        conn.commit()
        conn.close()

    def _save_entity_results(self, entity_type: str, entity_results: EntityResults):
        """
        Save results for a single entity type to database (incremental save).

        Args:
            entity_type: Entity type name ('intervention', 'condition', 'mechanism')
            entity_results: EntityResults object for this entity type
        """
        conn = sqlite3.connect(self.experiment_db_path)
        cursor = conn.cursor()

        try:
            # Save experiment_results
            cursor.execute("""
                INSERT INTO experiment_results (
                    experiment_id, entity_type,
                    embedding_duration_seconds, embeddings_generated, embedding_cache_hit_rate,
                    clustering_duration_seconds, num_clusters, num_natural_clusters,
                    num_singleton_clusters, num_noise_points, assignment_rate,
                    silhouette_score, davies_bouldin_score,
                    min_cluster_size, max_cluster_size, mean_cluster_size, median_cluster_size,
                    naming_duration_seconds, names_generated, naming_failures, naming_cache_hit_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.experiment_id, entity_type,
                entity_results.embedding_duration_seconds,
                entity_results.embeddings_generated,
                entity_results.embedding_cache_hit_rate,
                entity_results.clustering_duration_seconds,
                entity_results.num_clusters,
                entity_results.num_natural_clusters,
                entity_results.num_singleton_clusters,
                entity_results.num_noise_points,
                entity_results.assignment_rate,
                entity_results.silhouette_score,
                entity_results.davies_bouldin_score,
                entity_results.cluster_size_stats.get('min'),
                entity_results.cluster_size_stats.get('max'),
                entity_results.cluster_size_stats.get('mean'),
                entity_results.cluster_size_stats.get('median'),
                entity_results.naming_duration_seconds,
                entity_results.names_generated,
                entity_results.naming_failures,
                entity_results.naming_cache_hit_rate
            ))

            # Save cluster_details
            for cluster_id, naming_result in entity_results.naming_results.items():
                # Find cluster members
                member_mask = entity_results.cluster_labels == cluster_id
                members = [entity_results.entity_names[i] for i in range(len(member_mask)) if member_mask[i]]

                cursor.execute("""
                    INSERT INTO cluster_details (
                        experiment_id, entity_type, cluster_id,
                        canonical_name, category, parent_cluster,
                        member_count, is_singleton,
                        member_entities, confidence,
                        naming_method, naming_model, naming_temperature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.experiment_id, entity_type, cluster_id,
                    naming_result.canonical_name,
                    naming_result.category,
                    naming_result.parent_cluster,
                    len(members),
                    len(members) == 1,
                    json.dumps(members),
                    naming_result.confidence,
                    naming_result.provenance.get('method') if naming_result.provenance else None,
                    naming_result.provenance.get('model') if naming_result.provenance else None,
                    naming_result.provenance.get('temperature') if naming_result.provenance else None
                ))

            conn.commit()
            logger.info(f"Saved {entity_type} results to database (ID: {self.experiment_id})")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to save {entity_type} results: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def _save_experiment_results(self, results: Dict[str, EntityResults]):
        """
        Save detailed experiment results to database (batch save - DEPRECATED).

        This method is kept for backward compatibility but is no longer used
        by the main pipeline, which uses incremental saves via _save_entity_results.
        """
        conn = sqlite3.connect(self.experiment_db_path)
        cursor = conn.cursor()

        for entity_type, entity_results in results.items():
            # Save experiment_results
            cursor.execute("""
                INSERT INTO experiment_results (
                    experiment_id, entity_type,
                    embedding_duration_seconds, embeddings_generated, embedding_cache_hit_rate,
                    clustering_duration_seconds, num_clusters, num_natural_clusters,
                    num_singleton_clusters, num_noise_points, assignment_rate,
                    silhouette_score, davies_bouldin_score,
                    min_cluster_size, max_cluster_size, mean_cluster_size, median_cluster_size,
                    naming_duration_seconds, names_generated, naming_failures, naming_cache_hit_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.experiment_id, entity_type,
                entity_results.embedding_duration_seconds,
                entity_results.embeddings_generated,
                entity_results.embedding_cache_hit_rate,
                entity_results.clustering_duration_seconds,
                entity_results.num_clusters,
                entity_results.num_natural_clusters,
                entity_results.num_singleton_clusters,
                entity_results.num_noise_points,
                entity_results.assignment_rate,
                entity_results.silhouette_score,
                entity_results.davies_bouldin_score,
                entity_results.cluster_size_stats.get('min'),
                entity_results.cluster_size_stats.get('max'),
                entity_results.cluster_size_stats.get('mean'),
                entity_results.cluster_size_stats.get('median'),
                entity_results.naming_duration_seconds,
                entity_results.names_generated,
                entity_results.naming_failures,
                entity_results.naming_cache_hit_rate
            ))

            # Save cluster_details
            for cluster_id, naming_result in entity_results.naming_results.items():
                # Find cluster members
                member_mask = entity_results.cluster_labels == cluster_id
                members = [entity_results.entity_names[i] for i in range(len(member_mask)) if member_mask[i]]

                cursor.execute("""
                    INSERT INTO cluster_details (
                        experiment_id, entity_type, cluster_id,
                        canonical_name, category, parent_cluster,
                        member_count, is_singleton,
                        member_entities, confidence,
                        naming_method, naming_model, naming_temperature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.experiment_id, entity_type, cluster_id,
                    naming_result.canonical_name,
                    naming_result.category,
                    naming_result.parent_cluster,
                    len(members),
                    len(members) == 1,
                    json.dumps(members),
                    naming_result.confidence,
                    naming_result.provenance.get('method') if naming_result.provenance else None,
                    naming_result.provenance.get('model') if naming_result.provenance else None,
                    naming_result.provenance.get('temperature') if naming_result.provenance else None
                ))

        conn.commit()
        conn.close()

        logger.info(f"Saved experiment results to database (ID: {self.experiment_id})")

    def _log_experiment_error(self, error_message: str):
        """Log error to experiment_logs table."""
        conn = sqlite3.connect(self.experiment_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO experiment_logs (experiment_id, log_level, phase, message)
            VALUES (?, ?, ?, ?)
        """, (self.experiment_id, 'ERROR', 'pipeline', error_message))

        conn.commit()
        conn.close()
