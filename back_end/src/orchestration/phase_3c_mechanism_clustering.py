"""
Rotation Mechanism Clustering - LEGACY FILE (Phase 3c superseded by phase_3c_llm_namer.py)

⚠️  DEPRECATED: This file appears to be legacy mechanism clustering code from an older architecture.
    Current Phase 3c uses: phase_3c_llm_namer.py for LLM-based canonical naming.

    If this file is still in use, it should be migrated to use:
    - mxbai-embed-large (1024-dim) instead of nomic-embed-text (768-dim)
    - Current Phase 3 architecture patterns

ORIGINAL PURPOSE:
Integrates mechanism semantic clustering into the batch medical rotation pipeline.
Runs after Phase 3.5 (Group Categorization) to cluster all extracted mechanisms.

Architecture:
1. Load mechanism texts from interventions table
2. Generate/load semantic embeddings (mxbai-embed-large via Ollama) ← UPDATED
3. Run HDBSCAN clustering
4. Create singleton clusters for unassigned mechanisms (100% assignment)
5. Extract canonical names via LLM (qwen3:14b)
6. Populate database tables

Output:
- mechanism_clusters: Cluster metadata with canonical names
- mechanism_cluster_membership: Mechanism-to-cluster assignments
- intervention_mechanisms: Junction table linking interventions
- mechanism_condition_associations: Analytics for which mechanisms work for which conditions

Performance:
- First run: ~25 minutes (embedding generation for 666 mechanisms)
- Subsequent runs: <5 seconds (uses cached embeddings)
- Incremental updates: Only embed new mechanisms
"""

import time
import logging
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


class RotationMechanismClusterer:
    """
    Orchestrates mechanism clustering for pipeline integration.

    Guarantees 100% mechanism assignment (no mechanism ignored).
    """

    def __init__(self, db_path: str, cache_dir: Optional[str] = None):
        """
        Initialize mechanism clusterer.

        Args:
            db_path: Path to intervention_research.db
            cache_dir: Directory for caching embeddings
        """
        self.db_path = db_path
        self.cache_dir = Path(cache_dir) if cache_dir else Path("back_end/data/semantic_normalization_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_cache_path = self.cache_dir / "mechanism_embeddings_nomic.json"

        # Import dependencies here to avoid circular imports
        try:
            import sys
            semantic_norm_dir = Path(__file__).parent.parent / "semantic_normalization"
            sys.path.insert(0, str(semantic_norm_dir))

            # Import complete assignment clustering
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "complete_assignment_clustering",
                semantic_norm_dir / "complete_assignment_test.py"
            )
            self.clustering_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.clustering_module)

            logger.info("Mechanism clusterer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize mechanism clusterer: {e}")
            raise

    def run(self, force: bool = False) -> Dict[str, Any]:
        """
        Run complete mechanism clustering pipeline.

        Args:
            force: Force re-clustering even if results exist

        Returns:
            Dictionary with clustering results and statistics
        """
        logger.info("="*60)
        logger.info("PHASE 3.6: MECHANISM CLUSTERING")
        logger.info("="*60)

        start_time = time.time()

        try:
            # Step 1: Load mechanisms from database
            logger.info("[1/6] Loading mechanisms from database...")
            mechanisms = self._load_mechanisms_from_db()
            logger.info(f"  Loaded {len(mechanisms)} unique mechanisms")

            if len(mechanisms) == 0:
                logger.warning("No mechanisms found in database")
                return {
                    'success': True,
                    'mechanisms_found': 0,
                    'clusters_created': 0,
                    'message': 'No mechanisms to cluster'
                }

            # Step 2: Generate or load embeddings
            logger.info("[2/6] Generating/loading semantic embeddings...")
            embeddings = self._get_or_generate_embeddings(mechanisms, force=force)
            logger.info(f"  Embeddings ready: {embeddings.shape}")

            # Step 3: Run HDBSCAN clustering
            logger.info("[3/6] Running HDBSCAN clustering...")
            cluster_labels = self._run_clustering(embeddings)

            num_hdbscan_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            num_unassigned = list(cluster_labels).count(-1)
            logger.info(f"  HDBSCAN discovered: {num_hdbscan_clusters} clusters")
            logger.info(f"  Unassigned mechanisms: {num_unassigned}")

            # Step 4: Create singleton clusters (100% assignment)
            logger.info("[4/6] Creating singleton clusters for unassigned mechanisms...")
            cluster_labels = self._create_singleton_clusters(cluster_labels)

            total_clusters = len(set(cluster_labels))
            logger.info(f"  Total clusters: {total_clusters} (100% assignment)")

            # Step 5: Populate database
            logger.info("[5/6] Populating database tables...")
            db_result = self._populate_database(mechanisms, cluster_labels, embeddings)
            logger.info(f"  Clusters created: {db_result['clusters_created']}")
            logger.info(f"  Memberships created: {db_result['memberships_created']}")

            # Step 6: Build analytics
            logger.info("[6/6] Building mechanism-condition associations...")
            associations = self._build_analytics()
            logger.info(f"  Associations created: {associations}")

            elapsed_time = time.time() - start_time

            # Compute statistics
            cluster_sizes = self._compute_cluster_statistics(cluster_labels)

            result = {
                'success': True,
                'elapsed_time_seconds': elapsed_time,
                'mechanisms_processed': len(mechanisms),
                'clusters_created': total_clusters,
                'hdbscan_clusters': num_hdbscan_clusters,
                'singleton_clusters': num_unassigned,
                'assignment_rate': 100.0,  # Guaranteed
                'database_updates': {
                    'clusters_created': db_result['clusters_created'],
                    'memberships_created': db_result['memberships_created'],
                    'associations_created': associations
                },
                'cluster_statistics': cluster_sizes,
                'embeddings_cached': self.embeddings_cache_path.exists()
            }

            logger.info("[SUCCESS] Mechanism clustering completed")
            logger.info(f"  Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
            logger.info(f"  Mechanisms: {len(mechanisms)}")
            logger.info(f"  Clusters: {total_clusters}")
            logger.info(f"  Assignment: 100%")

            return result

        except Exception as e:
            logger.error(f"Mechanism clustering failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'elapsed_time_seconds': time.time() - start_time
            }

    def _load_mechanisms_from_db(self) -> List[str]:
        """Load unique mechanism texts from interventions table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT mechanism
            FROM interventions
            WHERE mechanism IS NOT NULL
              AND mechanism != ''
              AND mechanism != 'N/A'
            ORDER BY mechanism
        """)

        mechanisms = [row[0] for row in cursor.fetchall()]
        conn.close()

        return mechanisms

    def _get_or_generate_embeddings(self, mechanisms: List[str], force: bool = False):
        """Get embeddings from cache or generate via Ollama."""
        import numpy as np

        # Check cache
        if not force and self.embeddings_cache_path.exists():
            logger.info("  Loading embeddings from cache...")
            with open(self.embeddings_cache_path, 'r') as f:
                cache_data = json.load(f)

            cached_mechanisms = cache_data['mechanisms']

            # Verify cache matches current mechanisms
            if cached_mechanisms == mechanisms:
                logger.info("  Cache hit - using cached embeddings")
                embeddings = np.array(cache_data['embeddings'], dtype=np.float32)
                return embeddings
            else:
                logger.info("  Cache miss - mechanisms changed, regenerating...")

        # Generate embeddings via Ollama
        logger.info("  Generating embeddings via Ollama (this may take ~25 minutes)...")
        embeddings = self._generate_embeddings_ollama(mechanisms)

        # Save to cache
        logger.info("  Saving embeddings to cache...")
        cache_data = {
            'mechanisms': mechanisms,
            'embeddings': embeddings.tolist(),
            'model': 'mxbai-embed-large',  # Updated Oct 16, 2025
            'dimension': 1024,  # Updated Oct 16, 2025
            'timestamp': datetime.now().isoformat()
        }

        with open(self.embeddings_cache_path, 'w') as f:
            json.dump(cache_data, f)

        logger.info("  Embeddings cached for future runs")

        return embeddings

    def _generate_embeddings_ollama(self, mechanisms: List[str]):
        """Generate embeddings using Ollama API."""
        import numpy as np
        import requests

        OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
        EMBEDDING_MODEL = "mxbai-embed-large"  # Updated Oct 16, 2025
        EMBEDDING_DIM = 1024  # Updated Oct 16, 2025
        BATCH_SIZE = 10

        embeddings = []
        failed_count = 0

        total_batches = (len(mechanisms) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(0, len(mechanisms), BATCH_SIZE):
            batch = mechanisms[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1

            if batch_num % 10 == 0 or batch_num == 1:
                logger.info(f"    Processing batch {batch_num}/{total_batches}...")

            for text in batch:
                try:
                    response = requests.post(
                        OLLAMA_API_URL,
                        json={"model": EMBEDDING_MODEL, "prompt": text},
                        timeout=30
                    )

                    if response.status_code == 200:
                        embedding = response.json().get('embedding', [])
                        if len(embedding) == EMBEDDING_DIM:
                            embeddings.append(embedding)
                        else:
                            embeddings.append([0.0] * EMBEDDING_DIM)
                            failed_count += 1
                    else:
                        embeddings.append([0.0] * EMBEDDING_DIM)
                        failed_count += 1

                except Exception as e:
                    logger.warning(f"    Failed to embed: {text[:50]}... - {e}")
                    embeddings.append([0.0] * EMBEDDING_DIM)
                    failed_count += 1

            time.sleep(0.1)  # Rate limiting

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-9)

        if failed_count > 0:
            logger.warning(f"  {failed_count} embeddings failed (using zero vectors)")

        return embeddings_array

    def _run_clustering(self, embeddings):
        """Run HDBSCAN clustering."""
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,  # HDBSCAN minimum
            min_samples=1,       # Most permissive
            cluster_selection_epsilon=0.0,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        return cluster_labels

    def _create_singleton_clusters(self, cluster_labels):
        """Create singleton clusters for unassigned mechanisms."""
        import numpy as np

        unassigned_mask = cluster_labels == -1
        num_unassigned = np.sum(unassigned_mask)

        if num_unassigned == 0:
            return cluster_labels

        next_cluster_id = max(cluster_labels) + 1

        for i, is_unassigned in enumerate(unassigned_mask):
            if is_unassigned:
                cluster_labels[i] = next_cluster_id
                next_cluster_id += 1

        return cluster_labels

    def _populate_database(self, mechanisms: List[str], cluster_labels, embeddings) -> Dict[str, int]:
        """Populate database tables with clustering results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Initialize schema if needed
        self._initialize_schema(cursor)

        # Clear existing data (if force re-clustering)
        cursor.execute("DELETE FROM mechanism_cluster_membership")
        cursor.execute("DELETE FROM mechanism_clusters")
        cursor.execute("DELETE FROM intervention_mechanisms")
        cursor.execute("DELETE FROM mechanism_condition_associations")

        # Group mechanisms by cluster
        cluster_dict = {}
        for mechanism, label in zip(mechanisms, cluster_labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(mechanism)

        # Create cluster records with temporary names
        clusters_created = 0
        for cluster_id, members in cluster_dict.items():
            assignment_type = 'singleton' if len(members) == 1 else 'hdbscan'

            # Temporary canonical name (will be improved with LLM extraction later)
            if len(members) == 1:
                canonical_name = members[0][:100]  # Truncate to 100 chars
            else:
                # Use first member as temporary name
                canonical_name = f"Cluster {cluster_id}: {members[0][:80]}"

            cursor.execute("""
                INSERT OR REPLACE INTO mechanism_clusters (cluster_id, canonical_name, member_count, hierarchy_level)
                VALUES (?, ?, ?, 0)
            """, (int(cluster_id), canonical_name, len(members)))

            clusters_created += 1

        # Create membership records
        memberships_created = 0
        for mechanism, label in zip(mechanisms, cluster_labels):
            assignment_type = 'singleton' if len(cluster_dict[label]) == 1 else 'hdbscan'

            cursor.execute("""
                INSERT OR REPLACE INTO mechanism_cluster_membership (mechanism_text, cluster_id, assignment_type)
                VALUES (?, ?, ?)
            """, (mechanism, int(label), assignment_type))

            memberships_created += 1

        conn.commit()
        conn.close()

        return {
            'clusters_created': clusters_created,
            'memberships_created': memberships_created
        }

    def _initialize_schema(self, cursor):
        """Initialize mechanism clustering database schema."""

        # mechanism_clusters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mechanism_clusters (
                cluster_id INTEGER PRIMARY KEY,
                canonical_name TEXT NOT NULL UNIQUE,
                parent_cluster_id INTEGER,
                hierarchy_level INTEGER DEFAULT 0,
                member_count INTEGER,
                avg_silhouette REAL,
                creation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # mechanism_cluster_membership table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mechanism_cluster_membership (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mechanism_text TEXT NOT NULL,
                cluster_id INTEGER NOT NULL,
                assignment_type TEXT CHECK(assignment_type IN ('hdbscan', 'singleton')),
                similarity_score REAL,
                embedding_vector BLOB,
                FOREIGN KEY (cluster_id) REFERENCES mechanism_clusters(cluster_id),
                UNIQUE(mechanism_text, cluster_id)
            )
        """)

        # intervention_mechanisms junction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS intervention_mechanisms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intervention_id INTEGER NOT NULL,
                mechanism_text TEXT NOT NULL,
                cluster_id INTEGER,
                health_condition TEXT,
                outcome_type TEXT,  -- UPDATED: was correlation_strength (removed Oct 16, 2025)
                FOREIGN KEY (intervention_id) REFERENCES interventions(id),
                FOREIGN KEY (cluster_id) REFERENCES mechanism_clusters(cluster_id)
            )
        """)

        # mechanism_condition_associations analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mechanism_condition_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                health_condition TEXT NOT NULL,
                intervention_count INTEGER DEFAULT 0,
                -- avg_correlation_strength REAL,  -- REMOVED: correlation_strength field removed Oct 16, 2025
                FOREIGN KEY (cluster_id) REFERENCES mechanism_clusters(cluster_id),
                UNIQUE(cluster_id, health_condition)
            )
        """)

    def _build_analytics(self) -> int:
        """Build mechanism-condition association analytics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Clear existing associations
        cursor.execute("DELETE FROM mechanism_condition_associations")

        # Populate intervention_mechanisms junction table
        cursor.execute("""
            INSERT OR REPLACE INTO intervention_mechanisms (
                intervention_id, mechanism_text, cluster_id, health_condition, outcome_type
            )
            SELECT
                i.id,
                i.mechanism,
                mcm.cluster_id,
                i.health_condition,
                i.outcome_type
            FROM interventions i
            INNER JOIN mechanism_cluster_membership mcm ON i.mechanism = mcm.mechanism_text
            WHERE i.mechanism IS NOT NULL
              AND i.mechanism != ''
              AND i.mechanism != 'N/A'
        """)

        # Build mechanism-condition associations
        cursor.execute("""
            INSERT INTO mechanism_condition_associations (
                cluster_id, health_condition, intervention_count
            )
            SELECT
                cluster_id,
                health_condition,
                COUNT(*) as intervention_count
            FROM intervention_mechanisms
            WHERE health_condition IS NOT NULL
            GROUP BY cluster_id, health_condition
        """)

        associations_created = cursor.rowcount

        conn.commit()
        conn.close()

        return associations_created

    def _compute_cluster_statistics(self, cluster_labels) -> Dict[str, Any]:
        """Compute cluster size distribution statistics."""
        cluster_sizes = {}
        for label in cluster_labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

        sizes = list(cluster_sizes.values())

        return {
            'total_clusters': len(sizes),
            'average_size': sum(sizes) / len(sizes) if sizes else 0,
            'median_size': sorted(sizes)[len(sizes) // 2] if sizes else 0,
            'largest_cluster': max(sizes) if sizes else 0,
            'smallest_cluster': min(sizes) if sizes else 0,
            'singletons': sum(1 for s in sizes if s == 1),
            'singleton_percentage': (sum(1 for s in sizes if s == 1) / len(sizes) * 100) if sizes else 0
        }


if __name__ == "__main__":
    # Test the clusterer
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent))
    from back_end.src.data.config import config

    clusterer = RotationMechanismClusterer(
        db_path=str(config.db_path),
        cache_dir="back_end/data/semantic_normalization_cache"
    )

    result = clusterer.run(force=False)

    print("\n" + "="*60)
    print("MECHANISM CLUSTERING TEST COMPLETE")
    print("="*60)
    print(json.dumps(result, indent=2))
