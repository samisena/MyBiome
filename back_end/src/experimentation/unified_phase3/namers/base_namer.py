"""
Base Namer Abstract Class

Defines the interface for all naming engines in the unified Phase 3 pipeline.
Supports batch processing, temperature experimentation, and provenance tracking.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ClusterData:
    """Data class for cluster information."""
    cluster_id: int
    entity_type: str  # 'intervention', 'condition', 'mechanism'
    member_entities: List[str]
    member_frequencies: Optional[List[int]] = None
    singleton: bool = False


@dataclass
class NamingResult:
    """Data class for naming result."""
    cluster_id: int
    canonical_name: str
    category: Optional[str] = None
    parent_cluster: Optional[str] = None
    confidence: float = 1.0
    provenance: Optional[Dict] = None


class BaseNamer(ABC):
    """
    Abstract base class for all naming engines.

    Subclasses must implement:
    - _generate_names_batch(clusters: List[ClusterData]) -> List[NamingResult]
    """

    def __init__(
        self,
        namer_name: str,
        temperature: float = 0.0,
        cache_path: Optional[str] = None
    ):
        """
        Initialize the base namer.

        Args:
            namer_name: Name of naming engine (e.g., 'llm_qwen3')
            temperature: LLM temperature for creativity (0.0-1.0)
            cache_path: Path to cache naming results
        """
        self.namer_name = namer_name
        self.temperature = temperature
        self.cache_path = Path(cache_path) if cache_path else None

        # Cache: {cluster_hash: NamingResult}
        self.cache: Dict[str, NamingResult] = {}
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

        # Statistics
        self.stats = {
            'names_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'failures': 0
        }

        logger.info(f"Initialized {self.__class__.__name__}: namer={namer_name}, temperature={temperature}")

    @abstractmethod
    def _generate_names_batch(self, clusters: List[ClusterData]) -> List[NamingResult]:
        """
        Generate canonical names for a batch of clusters.

        This method must be implemented by subclasses.

        Args:
            clusters: List of ClusterData objects

        Returns:
            List[NamingResult]: Naming results for each cluster
        """
        pass

    def name_clusters(
        self,
        clusters: List[ClusterData],
        batch_size: int = 20
    ) -> Dict[int, NamingResult]:
        """
        Name clusters with batching and caching support.

        Args:
            clusters: List of ClusterData objects
            batch_size: Number of clusters per batch

        Returns:
            Dict mapping cluster_id to NamingResult
        """
        results = {}
        clusters_to_name = []
        cluster_indices = []

        # Check cache
        for i, cluster in enumerate(clusters):
            cluster_hash = self._hash_cluster(cluster)
            if cluster_hash in self.cache:
                results[cluster.cluster_id] = self.cache[cluster_hash]
                self.stats['cache_hits'] += 1
            else:
                clusters_to_name.append(cluster)
                cluster_indices.append(i)
                self.stats['cache_misses'] += 1

        # Generate missing names in batches
        if clusters_to_name:
            logger.info(f"Naming {len(clusters_to_name)} clusters in batches of {batch_size}")

            for i in range(0, len(clusters_to_name), batch_size):
                batch = clusters_to_name[i:i + batch_size]

                try:
                    batch_results = self._generate_names_batch(batch)

                    # Cache and store results
                    for cluster, naming_result in zip(batch, batch_results):
                        cluster_hash = self._hash_cluster(cluster)
                        self.cache[cluster_hash] = naming_result
                        results[cluster.cluster_id] = naming_result
                        self.stats['names_generated'] += 1

                except Exception as e:
                    logger.error(f"Failed to name batch: {e}")
                    self.stats['failures'] += len(batch)

                    # Create fallback names
                    for cluster in batch:
                        fallback_result = self._create_fallback_name(cluster)
                        results[cluster.cluster_id] = fallback_result

            # Save cache periodically
            if len(clusters_to_name) >= 50:
                self.save_cache()

        return results

    def _create_fallback_name(self, cluster: ClusterData) -> NamingResult:
        """
        Create fallback name when LLM fails.

        Args:
            cluster: ClusterData object

        Returns:
            NamingResult with fallback name
        """
        if cluster.singleton:
            fallback_name = cluster.member_entities[0]
        else:
            # Use most common member as fallback
            if cluster.member_frequencies:
                most_common_idx = cluster.member_frequencies.index(max(cluster.member_frequencies))
                fallback_name = cluster.member_entities[most_common_idx]
            else:
                fallback_name = cluster.member_entities[0]

        return NamingResult(
            cluster_id=cluster.cluster_id,
            canonical_name=fallback_name,
            category=None,
            confidence=0.5,
            provenance={'method': 'fallback'}
        )

    def _hash_cluster(self, cluster: ClusterData) -> str:
        """Generate hash for cluster (for caching)."""
        import hashlib
        cluster_str = f"{cluster.entity_type}_{sorted(cluster.member_entities)}"
        return hashlib.sha256(cluster_str.encode('utf-8')).hexdigest()

    def _load_cache(self):
        """Load cache from disk."""
        try:
            with open(self.cache_path, 'r') as f:
                cache_data = json.load(f)
                self.cache = {
                    k: NamingResult(**v)
                    for k, v in cache_data.items()
                }
            logger.info(f"Loaded {len(self.cache)} cached naming results from {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self.cache = {}

    def save_cache(self):
        """Save cache to disk."""
        if not self.cache_path:
            return

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                k: {
                    'cluster_id': v.cluster_id,
                    'canonical_name': v.canonical_name,
                    'category': v.category,
                    'parent_cluster': v.parent_cluster,
                    'confidence': v.confidence,
                    'provenance': v.provenance
                }
                for k, v in self.cache.items()
            }
            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Saved {len(self.cache)} naming results to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_stats(self) -> Dict:
        """Get naming statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0.0

        return {
            'namer': self.namer_name,
            'temperature': self.temperature,
            'cache_size': len(self.cache),
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': hit_rate,
            'names_generated': self.stats['names_generated'],
            'failures': self.stats['failures'],
            'failure_rate': self.stats['failures'] / total_requests if total_requests > 0 else 0.0
        }

    def __del__(self):
        """Cleanup: save cache on destruction."""
        if hasattr(self, 'cache') and self.cache_path:
            self.save_cache()
