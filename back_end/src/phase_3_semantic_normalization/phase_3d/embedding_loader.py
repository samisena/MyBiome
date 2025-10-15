"""
Embedding Loader for Phase 3d

Loads embeddings from Phase 3a cache for use in Phase 3d hierarchical merging.
This ensures consistency between Phase 3a/3b/3c and Phase 3d by using the same embeddings.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .config import config

logger = logging.getLogger(__name__)


def load_embeddings_from_phase3a_cache(
    entity_type: str,
    cache_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load embeddings from Phase 3a cache.

    Args:
        entity_type: 'intervention', 'condition', or 'mechanism'
        cache_dir: Override cache directory (default: uses config)

    Returns:
        Dict mapping entity names to embedding vectors (1024-dim for mxbai-embed-large)

    Raises:
        FileNotFoundError: If cache file doesn't exist
        ValueError: If embeddings have wrong dimension
    """
    if cache_dir is None:
        cache_dir = config.cache_dir

    cache_path = Path(cache_dir) / f"embeddings_{entity_type}_mxbai-embed-large.pkl"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Phase 3a embedding cache not found: {cache_path}\n"
            f"Please run Phase 3a first to generate embeddings."
        )

    logger.info(f"Loading embeddings from Phase 3a cache: {cache_path}")

    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)

    embeddings = cache_data.get('embeddings', {})

    if not embeddings:
        raise ValueError(f"No embeddings found in cache file: {cache_path}")

    # Validate dimensions
    first_embedding = next(iter(embeddings.values()))
    embedding_dim = len(first_embedding)

    if embedding_dim != config.embedding_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: cache has {embedding_dim}D, "
            f"config expects {config.embedding_dimension}D (mxbai-embed-large)"
        )

    logger.info(f"Loaded {len(embeddings)} embeddings ({embedding_dim}D)")

    return embeddings


def load_embeddings_for_clusters(
    clusters: List,
    entity_type: str,
    cache_dir: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load embeddings for cluster members from Phase 3a cache.

    Args:
        clusters: List of Cluster objects (must have .members attribute)
        entity_type: 'intervention', 'condition', or 'mechanism'
        cache_dir: Override cache directory

    Returns:
        Dict mapping member names to embedding vectors

    Raises:
        FileNotFoundError: If cache doesn't exist
    """
    # Load all embeddings from cache
    all_embeddings = load_embeddings_from_phase3a_cache(entity_type, cache_dir)

    # Collect all unique member names from clusters
    member_names = set()
    for cluster in clusters:
        member_names.update(cluster.members)

    # Filter to only members we need
    cluster_embeddings = {}
    missing_members = []

    for member in member_names:
        if member in all_embeddings:
            cluster_embeddings[member] = all_embeddings[member]
        else:
            missing_members.append(member)

    logger.info(f"Loaded embeddings for {len(cluster_embeddings)}/{len(member_names)} cluster members")

    if missing_members:
        logger.warning(
            f"Missing embeddings for {len(missing_members)} members: "
            f"{', '.join(missing_members[:5])}{'...' if len(missing_members) > 5 else ''}"
        )

    return cluster_embeddings


def check_embedding_cache_exists(entity_type: str, cache_dir: Optional[str] = None) -> bool:
    """
    Check if Phase 3a embedding cache exists for entity type.

    Args:
        entity_type: 'intervention', 'condition', or 'mechanism'
        cache_dir: Override cache directory

    Returns:
        True if cache exists, False otherwise
    """
    if cache_dir is None:
        cache_dir = config.cache_dir

    cache_path = Path(cache_dir) / f"embeddings_{entity_type}_mxbai-embed-large.pkl"
    return cache_path.exists()


def get_embedding_cache_info(entity_type: str, cache_dir: Optional[str] = None) -> Dict:
    """
    Get information about Phase 3a embedding cache.

    Args:
        entity_type: 'intervention', 'condition', or 'mechanism'
        cache_dir: Override cache directory

    Returns:
        Dict with cache information (exists, path, count, dimension)
    """
    if cache_dir is None:
        cache_dir = config.cache_dir

    cache_path = Path(cache_dir) / f"embeddings_{entity_type}_mxbai-embed-large.pkl"

    if not cache_path.exists():
        return {
            'exists': False,
            'path': str(cache_path),
            'count': 0,
            'dimension': None
        }

    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        embeddings = cache_data.get('embeddings', {})
        first_embedding = next(iter(embeddings.values())) if embeddings else None
        dimension = len(first_embedding) if first_embedding is not None else None

        return {
            'exists': True,
            'path': str(cache_path),
            'count': len(embeddings),
            'dimension': dimension
        }

    except Exception as e:
        logger.error(f"Error reading cache: {e}")
        return {
            'exists': True,
            'path': str(cache_path),
            'count': None,
            'dimension': None,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test the embedding loader
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("TESTING EMBEDDING LOADER")
    print("="*60)

    for entity_type in ['intervention', 'condition', 'mechanism']:
        print(f"\n{entity_type.capitalize()}s:")

        # Check if cache exists
        info = get_embedding_cache_info(entity_type)

        if info['exists']:
            print(f"  Cache: {info['path']}")
            print(f"  Count: {info['count']}")
            print(f"  Dimension: {info['dimension']}")

            # Try loading
            try:
                embeddings = load_embeddings_from_phase3a_cache(entity_type)
                print(f"  Loaded: {len(embeddings)} embeddings")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  Cache not found: {info['path']}")

    print("\n" + "="*60)
