"""
Clusterers Module - Clustering Algorithms for Entity Grouping

Provides implementations for clustering embeddings using various algorithms
(HDBSCAN, Hierarchical) with 100% assignment guarantee via SingletonHandler.
"""

from .base_clusterer import BaseClusterer
from .hdbscan_clusterer import HDBSCANClusterer
from .hierarchical_clusterer import HierarchicalClusterer
from .singleton_handler import SingletonHandler

__all__ = [
    'BaseClusterer',
    'HDBSCANClusterer',
    'HierarchicalClusterer',
    'SingletonHandler'
]
