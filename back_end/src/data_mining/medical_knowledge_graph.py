"""
Multi-Edge Graph Construction for Medical Knowledge.

Creates a comprehensive graph structure that preserves all evidence types
and metadata, enabling bidirectional queries for conditions and interventions.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
from datetime import datetime

try:
    from back_end.src.phase_1_data_collection.data_mining_repository import (
        DataMiningRepository,
        KnowledgeGraphNode,
        KnowledgeGraphEdge
    )
    from back_end.src.data.config import setup_logging
except ImportError as e:
    print(f"Warning: Could not import database components: {e}")
    DataMiningRepository = None
    KnowledgeGraphNode = None
    KnowledgeGraphEdge = None

logger = setup_logging(__name__, 'knowledge_graph.log') if 'setup_logging' in globals() else None


@dataclass
class StudyEvidence:
    """Individual study evidence with complete metadata."""
    study_id: str
    title: str
    evidence_type: str  # 'positive', 'negative', 'neutral', 'unsure'
    weight: float
    confidence: float
    sample_size: int
    study_design: str = "unknown"  # RCT, observational, meta-analysis, etc.
    publication_year: int = 0
    journal: str = ""
    doi: str = ""
    effect_size: Optional[float] = None
    p_value: Optional[float] = None

    def __post_init__(self):
        """Validate evidence data on creation."""
        if self.evidence_type not in ['positive', 'negative', 'neutral', 'unsure']:
            raise ValueError(f"Invalid evidence_type: {self.evidence_type}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1: {self.confidence}")


@dataclass
class GraphEdge:
    """Single edge in the medical knowledge graph."""
    source: str  # condition or intervention
    target: str  # condition or intervention
    evidence: StudyEvidence
    edge_type: str  # 'treats', 'causes', 'correlates', 'prevents'

    @property
    def weight(self) -> float:
        """Get the weight of this edge."""
        return self.evidence.weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary for serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'evidence': {
                'study_id': self.evidence.study_id,
                'title': self.evidence.title,
                'evidence_type': self.evidence.evidence_type,
                'weight': self.evidence.weight,
                'confidence': self.evidence.confidence,
                'sample_size': self.evidence.sample_size,
                'study_design': self.evidence.study_design,
                'publication_year': self.evidence.publication_year,
                'journal': self.evidence.journal,
                'doi': self.evidence.doi,
                'effect_size': self.evidence.effect_size,
                'p_value': self.evidence.p_value
            }
        }


class MedicalKnowledgeGraph:
    """
    Multi-edge bidirectional graph for medical knowledge.

    Key features:
    - Multiple edges between same nodes (preserves all studies)
    - Bidirectional queries (what treats X? what does Y treat?)
    - Negative evidence preservation
    - Complete metadata preservation
    - Fundamental intervention detection
    """

    # Weight mapping for different evidence types
    WEIGHT_MAP = {
        'positive': 1.0,   # Treatment works
        'negative': -1.0,  # Treatment doesn't work (important signal!)
        'neutral': 0.0,    # No effect
        'unsure': 0.3      # Slight positive signal
    }

    def __init__(self, save_to_database: bool = True):
        """Initialize empty medical knowledge graph."""
        # Multi-edge storage: source -> target -> list of edges
        self.forward_edges: Dict[str, Dict[str, List[GraphEdge]]] = defaultdict(lambda: defaultdict(list))
        self.reverse_edges: Dict[str, Dict[str, List[GraphEdge]]] = defaultdict(lambda: defaultdict(list))

        # Node metadata
        self.nodes: Dict[str, Dict[str, Any]] = {}

        # Database integration
        self.save_to_database = save_to_database
        self.repository = None
        if save_to_database and DataMiningRepository:
            try:
                self.repository = DataMiningRepository()
            except Exception as e:
                if logger:
                    logger.warning(f"Could not initialize database repository: {e}")
                self.save_to_database = False

        # Edge type tracking
        self.edge_types: Set[str] = set()

        # Statistics
        self.stats = {
            'total_edges': 0,
            'total_nodes': 0,
            'evidence_types': defaultdict(int),
            'edge_types': defaultdict(int)
        }

    def add_node(self, node_id: str, node_type: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add a node (condition or intervention) to the graph.

        Args:
            node_id: Unique identifier for the node
            node_type: 'condition' or 'intervention'
            metadata: Additional node metadata
        """
        if metadata is None:
            metadata = {}

        self.nodes[node_id] = {
            'type': node_type,
            'added_at': datetime.now().isoformat(),
            **metadata
        }
        self.stats['total_nodes'] = len(self.nodes)

        # Save to database if enabled
        if self.save_to_database and self.repository and KnowledgeGraphNode:
            try:
                self._save_node_to_database(node_id, node_type, metadata)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save node to database: {e}")

    def add_evidence(
        self,
        source: str,
        target: str,
        evidence: StudyEvidence,
        edge_type: str = 'treats'
    ) -> None:
        """
        Add evidence edge to the graph.

        Args:
            source: Source node (usually intervention)
            target: Target node (usually condition)
            evidence: Study evidence data
            edge_type: Type of relationship
        """
        # Calculate weight based on evidence type
        evidence.weight = self.WEIGHT_MAP[evidence.evidence_type]

        # Create the edge
        edge = GraphEdge(source=source, target=target, evidence=evidence, edge_type=edge_type)

        # Add to forward edges (source -> target)
        self.forward_edges[source][target].append(edge)

        # Add to reverse edges (target -> source)
        self.reverse_edges[target][source].append(edge)

        # Update tracking
        self.edge_types.add(edge_type)
        self.stats['total_edges'] += 1
        self.stats['evidence_types'][evidence.evidence_type] += 1
        self.stats['edge_types'][edge_type] += 1

        # Ensure nodes exist
        if source not in self.nodes:
            self.add_node(source, 'intervention')
        if target not in self.nodes:
            self.add_node(target, 'condition')

        # Save to database if enabled
        if self.save_to_database and self.repository and KnowledgeGraphEdge:
            try:
                self._save_edge_to_database(edge)
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to save edge to database: {e}")

    def query_treatments_for_condition(
        self,
        condition: str,
        min_confidence: float = 0.0,
        evidence_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query 1: "What treats this condition?"

        Args:
            condition: Condition to find treatments for
            min_confidence: Minimum confidence threshold
            evidence_types: Filter by evidence types

        Returns:
            List of treatments with aggregated evidence
        """
        if evidence_types is None:
            evidence_types = ['positive', 'negative', 'neutral', 'unsure']

        treatments = defaultdict(list)

        # Get all edges pointing TO this condition (reverse edges)
        for intervention, edges in self.reverse_edges[condition].items():
            for edge in edges:
                if (edge.evidence.confidence >= min_confidence and
                    edge.evidence.evidence_type in evidence_types):
                    treatments[intervention].append(edge)

        # Aggregate evidence for each treatment
        results = []
        for intervention, edges in treatments.items():
            result = self._aggregate_intervention_evidence(intervention, condition, edges)
            results.append(result)

        # Sort by aggregate score (confidence-weighted)
        results.sort(key=lambda x: x['aggregate_score'], reverse=True)
        return results

    def query_conditions_for_intervention(
        self,
        intervention: str,
        min_confidence: float = 0.0,
        evidence_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query 2: "What conditions does this intervention help?"

        Args:
            intervention: Intervention to find applications for
            min_confidence: Minimum confidence threshold
            evidence_types: Filter by evidence types

        Returns:
            List of conditions with aggregated evidence
        """
        if evidence_types is None:
            evidence_types = ['positive', 'negative', 'neutral', 'unsure']

        conditions = defaultdict(list)

        # Get all edges starting FROM this intervention (forward edges)
        for condition, edges in self.forward_edges[intervention].items():
            for edge in edges:
                if (edge.evidence.confidence >= min_confidence and
                    edge.evidence.evidence_type in evidence_types):
                    conditions[condition].append(edge)

        # Aggregate evidence for each condition
        results = []
        for condition, edges in conditions.items():
            result = self._aggregate_condition_evidence(intervention, condition, edges)
            results.append(result)

        # Sort by aggregate score
        results.sort(key=lambda x: x['aggregate_score'], reverse=True)
        return results

    def detect_fundamental_interventions(self, min_conditions: int = 3) -> List[Dict[str, Any]]:
        """
        Detect "fundamental" interventions that help many conditions.

        Args:
            min_conditions: Minimum number of conditions to be considered fundamental

        Returns:
            List of fundamental interventions ranked by breadth and effectiveness
        """
        fundamentals = []

        for intervention in self.forward_edges:
            # Count positive conditions (aggregate score > 0)
            positive_conditions = 0
            total_score = 0
            condition_count = len(self.forward_edges[intervention])

            for condition, edges in self.forward_edges[intervention].items():
                agg_evidence = self._aggregate_condition_evidence(intervention, condition, edges)
                if agg_evidence['aggregate_score'] > 0:
                    positive_conditions += 1
                    total_score += agg_evidence['aggregate_score']

            if positive_conditions >= min_conditions:
                fundamentals.append({
                    'intervention': intervention,
                    'positive_conditions': positive_conditions,
                    'total_conditions': condition_count,
                    'avg_effectiveness': total_score / positive_conditions if positive_conditions > 0 else 0,
                    'breadth_score': positive_conditions * (total_score / positive_conditions) if positive_conditions > 0 else 0
                })

        # Sort by breadth score (conditions × effectiveness)
        fundamentals.sort(key=lambda x: x['breadth_score'], reverse=True)
        return fundamentals

    def _aggregate_intervention_evidence(
        self,
        intervention: str,
        condition: str,
        edges: List[GraphEdge]
    ) -> Dict[str, Any]:
        """Aggregate evidence for an intervention treating a condition."""
        if not edges:
            return {'intervention': intervention, 'condition': condition, 'aggregate_score': 0, 'edge_count': 0}

        total_weight = sum(edge.weight * edge.evidence.confidence for edge in edges)
        total_confidence = sum(edge.evidence.confidence for edge in edges)
        avg_confidence = total_confidence / len(edges)

        # Confidence-weighted aggregate score
        aggregate_score = total_weight / len(edges) if edges else 0
        weighted_score = aggregate_score * avg_confidence

        evidence_breakdown = defaultdict(int)
        for edge in edges:
            evidence_breakdown[edge.evidence.evidence_type] += 1

        return {
            'intervention': intervention,
            'condition': condition,
            'aggregate_score': round(weighted_score, 3),
            'raw_score': round(aggregate_score, 3),
            'avg_confidence': round(avg_confidence, 3),
            'edge_count': len(edges),
            'evidence_breakdown': dict(evidence_breakdown),
            'strongest_evidence': max(edges, key=lambda e: e.evidence.confidence).to_dict(),
            'recommendation': self._generate_recommendation(weighted_score, avg_confidence, len(edges))
        }

    def _aggregate_condition_evidence(
        self,
        intervention: str,
        condition: str,
        edges: List[GraphEdge]
    ) -> Dict[str, Any]:
        """Aggregate evidence for a condition being treated by an intervention."""
        return self._aggregate_intervention_evidence(intervention, condition, edges)

    def _generate_recommendation(self, score: float, confidence: float, evidence_count: int) -> str:
        """Generate human-readable recommendation."""
        if score > 0.5 and confidence > 0.7 and evidence_count >= 3:
            return "Strongly recommended - robust positive evidence"
        elif score > 0.3 and confidence > 0.5:
            return "Recommended - moderate positive evidence"
        elif score < -0.3:
            return "Not recommended - evidence suggests ineffective"
        elif evidence_count < 2:
            return "Insufficient evidence - more research needed"
        else:
            return "Neutral - mixed or weak evidence"

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        return {
            'nodes': self.stats['total_nodes'],
            'edges': self.stats['total_edges'],
            'evidence_types': dict(self.stats['evidence_types']),
            'edge_types': dict(self.stats['edge_types']),
            'avg_edges_per_node': self.stats['total_edges'] / max(self.stats['total_nodes'], 1),
            'node_types': {
                'conditions': len([n for n in self.nodes.values() if n.get('type') == 'condition']),
                'interventions': len([n for n in self.nodes.values() if n.get('type') == 'intervention'])
            }
        }

    @property
    def backward_edges(self):
        """
        Alias for reverse_edges to maintain API compatibility.

        Returns reverse_edges which maps: condition -> intervention -> List[GraphEdge]
        This enables queries like "what treatments exist for condition X?"
        """
        return self.reverse_edges

    def export_to_json(self, filepath: str) -> None:
        """Export graph to JSON format."""
        export_data = {
            'nodes': self.nodes,
            'edges': []
        }

        # Collect all edges
        for source, targets in self.forward_edges.items():
            for target, edges in targets.items():
                for edge in edges:
                    export_data['edges'].append(edge.to_dict())

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def _save_node_to_database(self, node_id: str, node_type: str, metadata: Dict[str, Any]) -> None:
        """Save a knowledge graph node to the database."""
        try:
            db_node = KnowledgeGraphNode(
                node_id=node_id,
                node_type=node_type,
                node_name=metadata.get('name', node_id),
                node_data=metadata,
                created_at=datetime.now()
            )
            self.repository.save_knowledge_graph_node(db_node)

            if logger:
                logger.debug(f"Saved node to database: {node_id}")

        except Exception as e:
            if logger:
                logger.error(f"Error saving node to database: {e}")
            raise

    def _save_edge_to_database(self, edge: GraphEdge) -> None:
        """Save a knowledge graph edge to the database."""
        try:
            # Generate unique edge ID
            edge_id = f"{edge.source}_{edge.target}_{edge.evidence.study_id}_{edge.edge_type}"

            db_edge = KnowledgeGraphEdge(
                edge_id=edge_id,
                source_node_id=edge.source,
                target_node_id=edge.target,
                edge_type=edge.edge_type,
                edge_weight=edge.evidence.weight,
                evidence_type=edge.evidence.evidence_type,
                confidence=edge.evidence.confidence,
                study_id=edge.evidence.study_id,
                study_title=edge.evidence.title,
                sample_size=edge.evidence.sample_size,
                study_design=edge.evidence.study_design,
                publication_year=edge.evidence.publication_year,
                journal=edge.evidence.journal,
                doi=edge.evidence.doi,
                effect_size=edge.evidence.effect_size,
                p_value=edge.evidence.p_value,
                generation_model="knowledge_graph_v1",
                generation_version="1.0",
                created_at=datetime.now()
            )
            self.repository.save_knowledge_graph_edge(db_edge)

            if logger:
                logger.debug(f"Saved edge to database: {edge_id}")

        except Exception as e:
            if logger:
                logger.error(f"Error saving edge to database: {e}")
            raise