"""
Common graph utilities for medical knowledge graph operations.
Centralizes graph traversal, edge aggregation, and evidence extraction.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source: str
    target: str
    weight: float
    edge_type: str
    metadata: Dict = None


@dataclass
class GraphPath:
    """Represents a path through the graph."""
    nodes: List[str]
    edges: List[GraphEdge]
    total_weight: float
    path_type: str


class GraphTraversal:
    """Utilities for traversing knowledge graphs."""

    @staticmethod
    def get_neighbors(
        node: str,
        edges: Dict[str, List[GraphEdge]],
        edge_type: Optional[str] = None,
        direction: str = 'outgoing'
    ) -> List[Tuple[str, float]]:
        """
        Get neighbors of a node with weights.

        Args:
            node: Source node
            edges: Edge dictionary
            edge_type: Optional filter by edge type
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of (neighbor, weight) tuples
        """
        neighbors = []

        if direction in ['outgoing', 'both']:
            if node in edges:
                for edge in edges[node]:
                    if edge_type is None or edge.edge_type == edge_type:
                        neighbors.append((edge.target, edge.weight))

        if direction in ['incoming', 'both']:
            # Search for incoming edges
            for source, edge_list in edges.items():
                if source != node:
                    for edge in edge_list:
                        if edge.target == node:
                            if edge_type is None or edge.edge_type == edge_type:
                                neighbors.append((source, edge.weight))

        return neighbors

    @staticmethod
    def find_paths(
        start: str,
        end: str,
        edges: Dict[str, List[GraphEdge]],
        max_length: int = 3,
        min_weight: float = 0.1
    ) -> List[GraphPath]:
        """
        Find paths between two nodes.

        Args:
            start: Starting node
            end: Target node
            edges: Edge dictionary
            max_length: Maximum path length
            min_weight: Minimum edge weight to consider

        Returns:
            List of GraphPath objects
        """
        paths = []
        visited = set()

        def dfs(current: str, target: str, path: List[str], path_edges: List[GraphEdge], depth: int):
            if depth > max_length:
                return

            if current == target and len(path) > 1:
                total_weight = sum(e.weight for e in path_edges)
                paths.append(GraphPath(
                    nodes=path.copy(),
                    edges=path_edges.copy(),
                    total_weight=total_weight,
                    path_type='direct' if len(path) == 2 else 'indirect'
                ))
                return

            visited.add(current)

            if current in edges:
                for edge in edges[current]:
                    if edge.weight >= min_weight and edge.target not in visited:
                        path.append(edge.target)
                        path_edges.append(edge)
                        dfs(edge.target, target, path, path_edges, depth + 1)
                        path.pop()
                        path_edges.pop()

            visited.remove(current)

        dfs(start, end, [start], [], 0)
        return sorted(paths, key=lambda p: p.total_weight, reverse=True)

    @staticmethod
    def get_connected_components(
        nodes: Set[str],
        edges: Dict[str, List[GraphEdge]]
    ) -> List[Set[str]]:
        """
        Find connected components in the graph.

        Args:
            nodes: Set of all nodes
            edges: Edge dictionary

        Returns:
            List of connected component sets
        """
        visited = set()
        components = []

        def dfs(node: str, component: Set[str]):
            visited.add(node)
            component.add(node)

            if node in edges:
                for edge in edges[node]:
                    if edge.target not in visited:
                        dfs(edge.target, component)

        for node in nodes:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)

        return components


class EdgeAggregation:
    """Utilities for aggregating edge evidence."""

    @staticmethod
    def aggregate_parallel_edges(
        edges: List[GraphEdge],
        method: str = 'weighted_mean'
    ) -> float:
        """
        Aggregate multiple parallel edges into single score.

        Args:
            edges: List of parallel edges
            method: Aggregation method

        Returns:
            Aggregated score
        """
        if not edges:
            return 0.0

        weights = [e.weight for e in edges]

        if method == 'weighted_mean':
            # Weighted by evidence count if available
            if all(e.metadata and 'evidence_count' in e.metadata for e in edges):
                counts = [e.metadata['evidence_count'] for e in edges]
                return np.average(weights, weights=counts)
            else:
                return np.mean(weights)

        elif method == 'max':
            return max(weights)

        elif method == 'sum':
            return min(1.0, sum(weights))  # Cap at 1.0

        elif method == 'harmonic_mean':
            if all(w > 0 for w in weights):
                return len(weights) / sum(1/w for w in weights)
            else:
                return 0.0

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    @staticmethod
    def combine_edge_types(
        edge_groups: Dict[str, List[GraphEdge]],
        type_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Combine edges of different types.

        Args:
            edge_groups: Edges grouped by type
            type_weights: Weights for each edge type

        Returns:
            Combined score
        """
        if not edge_groups:
            return 0.0

        if type_weights is None:
            type_weights = {
                'direct': 1.0,
                'mechanism': 0.7,
                'similar': 0.5,
                'indirect': 0.3
            }

        total_score = 0.0
        total_weight = 0.0

        for edge_type, edges in edge_groups.items():
            if edges:
                type_score = EdgeAggregation.aggregate_parallel_edges(edges)
                weight = type_weights.get(edge_type, 0.5)
                total_score += type_score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def calculate_edge_confidence(
        edge: GraphEdge,
        global_statistics: Optional[Dict] = None
    ) -> float:
        """
        Calculate confidence score for an edge.

        Args:
            edge: Graph edge
            global_statistics: Optional global graph statistics

        Returns:
            Confidence score [0, 1]
        """
        confidence = edge.weight  # Base confidence is the weight

        if edge.metadata:
            # Boost for multiple sources
            if 'source_count' in edge.metadata:
                source_factor = min(1.0, np.log1p(edge.metadata['source_count']) / np.log(10))
                confidence *= (0.7 + 0.3 * source_factor)

            # Boost for recent evidence
            if 'recency_score' in edge.metadata:
                confidence *= (0.8 + 0.2 * edge.metadata['recency_score'])

            # Penalty for controversial evidence
            if 'controversy_score' in edge.metadata:
                confidence *= (1.0 - 0.3 * edge.metadata['controversy_score'])

        # Global statistics adjustments
        if global_statistics:
            if 'avg_weight' in global_statistics:
                # Relative to average
                relative_strength = edge.weight / global_statistics['avg_weight']
                confidence *= min(1.5, relative_strength)

        return min(1.0, confidence)


class EvidenceExtraction:
    """Utilities for extracting evidence from graph structures."""

    @staticmethod
    def extract_intervention_evidence(
        intervention: str,
        condition: str,
        forward_edges: Dict[str, List[GraphEdge]],
        reverse_edges: Dict[str, List[GraphEdge]]
    ) -> Dict[str, Any]:
        """
        Extract all evidence for an intervention-condition pair.

        Args:
            intervention: Intervention name
            condition: Condition name
            forward_edges: Forward edge dictionary
            reverse_edges: Reverse edge dictionary

        Returns:
            Dictionary of evidence types and values
        """
        evidence = {
            'direct_edges': [],
            'mechanism_edges': [],
            'similar_condition_edges': [],
            'total_evidence_count': 0,
            'unique_sources': set(),
            'effectiveness_scores': []
        }

        # Direct edges
        if intervention in forward_edges:
            for edge in forward_edges[intervention]:
                if edge.target == condition:
                    evidence['direct_edges'].append(edge)
                    evidence['effectiveness_scores'].append(edge.weight)
                    if edge.metadata and 'source' in edge.metadata:
                        evidence['unique_sources'].add(edge.metadata['source'])

        # Reverse lookup for condition
        if condition in reverse_edges:
            for edge in reverse_edges[condition]:
                if edge.source == intervention:
                    if edge not in evidence['direct_edges']:
                        evidence['direct_edges'].append(edge)

        # Calculate summary statistics
        evidence['total_evidence_count'] = len(evidence['direct_edges'])
        evidence['avg_effectiveness'] = (
            np.mean(evidence['effectiveness_scores'])
            if evidence['effectiveness_scores'] else 0.0
        )
        evidence['source_diversity'] = len(evidence['unique_sources'])

        return evidence

    @staticmethod
    def extract_mechanism_evidence(
        source: str,
        target: str,
        mechanism_edges: Dict[str, Dict[str, List[GraphEdge]]]
    ) -> List[Dict]:
        """
        Extract mechanism-based evidence.

        Args:
            source: Source node
            target: Target node
            mechanism_edges: Edges organized by mechanism

        Returns:
            List of mechanism evidence
        """
        mechanism_evidence = []

        for mechanism, edges in mechanism_edges.items():
            if source in edges:
                for edge in edges[source]:
                    if edge.target == target or edge.edge_type == 'mechanism':
                        mechanism_evidence.append({
                            'mechanism': mechanism,
                            'strength': edge.weight,
                            'edge': edge
                        })

        return mechanism_evidence

    @staticmethod
    def get_evidence_summary(
        node: str,
        edges: Dict[str, List[GraphEdge]],
        evidence_type: str = 'all'
    ) -> Dict[str, float]:
        """
        Get summary of evidence for a node.

        Args:
            node: Target node
            edges: Edge dictionary
            evidence_type: Type of evidence to summarize

        Returns:
            Summary statistics
        """
        summary = {
            'total_edges': 0,
            'avg_weight': 0.0,
            'max_weight': 0.0,
            'min_weight': 1.0,
            'unique_sources': 0
        }

        all_edges = []
        sources = set()

        # Collect relevant edges
        if evidence_type in ['all', 'outgoing']:
            if node in edges:
                all_edges.extend(edges[node])
                sources.update(e.source for e in edges[node] if hasattr(e, 'source'))

        if evidence_type in ['all', 'incoming']:
            for source, edge_list in edges.items():
                for edge in edge_list:
                    if edge.target == node:
                        all_edges.append(edge)
                        sources.add(source)

        # Calculate summary
        if all_edges:
            weights = [e.weight for e in all_edges]
            summary['total_edges'] = len(all_edges)
            summary['avg_weight'] = np.mean(weights)
            summary['max_weight'] = max(weights)
            summary['min_weight'] = min(weights)
            summary['unique_sources'] = len(sources)

        return summary


class GraphMetrics:
    """Utilities for calculating graph-level metrics."""

    @staticmethod
    def calculate_centrality(
        node: str,
        edges: Dict[str, List[GraphEdge]],
        metric: str = 'degree'
    ) -> float:
        """
        Calculate centrality metric for a node.

        Args:
            node: Target node
            edges: Edge dictionary
            metric: Centrality metric type

        Returns:
            Centrality score
        """
        if metric == 'degree':
            # Simple degree centrality
            out_degree = len(edges.get(node, []))
            in_degree = sum(
                1 for edge_list in edges.values()
                for edge in edge_list
                if edge.target == node
            )
            return float(out_degree + in_degree)

        elif metric == 'weighted_degree':
            # Weighted by edge strengths
            out_weight = sum(e.weight for e in edges.get(node, []))
            in_weight = sum(
                edge.weight
                for edge_list in edges.values()
                for edge in edge_list
                if edge.target == node
            )
            return float(out_weight + in_weight)

        else:
            raise ValueError(f"Unknown centrality metric: {metric}")

    @staticmethod
    def calculate_clustering_coefficient(
        node: str,
        edges: Dict[str, List[GraphEdge]]
    ) -> float:
        """
        Calculate local clustering coefficient.

        Args:
            node: Target node
            edges: Edge dictionary

        Returns:
            Clustering coefficient [0, 1]
        """
        # Get neighbors
        neighbors = set()
        if node in edges:
            neighbors.update(e.target for e in edges[node])

        if len(neighbors) < 2:
            return 0.0

        # Count edges between neighbors
        neighbor_edges = 0
        for n1 in neighbors:
            if n1 in edges:
                for edge in edges[n1]:
                    if edge.target in neighbors:
                        neighbor_edges += 1

        # Calculate coefficient
        possible_edges = len(neighbors) * (len(neighbors) - 1)
        return neighbor_edges / possible_edges if possible_edges > 0 else 0.0