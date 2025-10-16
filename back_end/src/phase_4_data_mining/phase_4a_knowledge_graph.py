"""
Phase 4a: Knowledge Graph Construction for Medical Knowledge.

Builds a multi-edge bidirectional graph from Phase 3 canonical groups,
enabling evidence pooling and cleaner analytics.

Key Features:
- Integrates with Phase 3 clustering (canonical groups)
- Pools evidence across cluster members for better statistical power
- Multi-edge preservation (all studies retained)
- Bidirectional queries (what treats X? what does Y treat?)
- Negative evidence preservation
- Complete metadata preservation
- Fundamental intervention detection

Migration from standalone data_mining/medical_knowledge_graph.py to Phase 4a.
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
    evidence_type: str  # 'improves', 'worsens', 'no_effect', 'inconclusive'
    weight: float
    confidence: float
    sample_size: int
    # Mechanism fields (edge label - Phase 3c integration)
    mechanism_cluster_id: Optional[int] = None
    mechanism_canonical_name: str = ""  # Mechanism IS the edge label
    mechanism_raw_text: Optional[str] = None
    mechanism_similarity: Optional[float] = None
    # Study metadata
    study_design: str = "unknown"  # RCT, observational, meta-analysis, etc.
    publication_year: int = 0
    journal: str = ""
    doi: str = ""
    effect_size: Optional[float] = None
    p_value: Optional[float] = None

    def __post_init__(self):
        """Validate evidence data on creation."""
        if self.evidence_type not in ['improves', 'worsens', 'no_effect', 'inconclusive']:
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
                # Mechanism fields
                'mechanism_cluster_id': self.evidence.mechanism_cluster_id,
                'mechanism_canonical_name': self.evidence.mechanism_canonical_name,
                'mechanism_raw_text': self.evidence.mechanism_raw_text,
                'mechanism_similarity': self.evidence.mechanism_similarity,
                # Study metadata
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

    # Weight mapping for different evidence types (health impact semantics)
    WEIGHT_MAP = {
        'improves': 1.0,      # Treatment improves patient health
        'worsens': -1.0,      # Treatment worsens patient health (important signal!)
        'no_effect': 0.0,     # No measurable impact
        'inconclusive': 0.3   # Unclear, slight positive signal
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
            evidence_types = ['improves', 'worsens', 'no_effect', 'inconclusive']

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
            evidence_types = ['improves', 'worsens', 'no_effect', 'inconclusive']

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

    # =================================================================
    # MECHANISM-BASED QUERIES (Phase 3c Integration)
    # =================================================================

    def query_mechanisms_for_pair(
        self,
        intervention: str,
        condition: str,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Query all mechanisms linking a specific intervention to a condition.

        Args:
            intervention: Intervention (canonical name)
            condition: Condition (canonical name)
            min_confidence: Minimum confidence threshold

        Returns:
            List of mechanisms with evidence details
        """
        mechanisms = defaultdict(list)

        # Get all edges between this intervention-condition pair
        if intervention in self.forward_edges and condition in self.forward_edges[intervention]:
            for edge in self.forward_edges[intervention][condition]:
                if (edge.evidence.confidence >= min_confidence and
                    edge.evidence.mechanism_canonical_name):
                    mechanisms[edge.evidence.mechanism_canonical_name].append(edge)

        # Aggregate evidence for each mechanism
        results = []
        for mechanism_name, edges in mechanisms.items():
            total_weight = sum(e.weight * e.evidence.confidence for e in edges)
            avg_confidence = sum(e.evidence.confidence for e in edges) / len(edges)

            evidence_breakdown = defaultdict(int)
            for edge in edges:
                evidence_breakdown[edge.evidence.evidence_type] += 1

            results.append({
                'mechanism': mechanism_name,
                'intervention': intervention,
                'condition': condition,
                'edge_count': len(edges),
                'avg_confidence': round(avg_confidence, 3),
                'aggregate_score': round(total_weight / len(edges) if edges else 0, 3),
                'evidence_breakdown': dict(evidence_breakdown),
                'studies': [e.evidence.study_id for e in edges]
            })

        # Sort by aggregate score
        results.sort(key=lambda x: x['aggregate_score'], reverse=True)
        return results

    def query_interventions_by_mechanism(
        self,
        mechanism: str,
        condition: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Query interventions that work via a specific mechanism.

        Args:
            mechanism: Mechanism (canonical name)
            condition: Optional condition filter
            min_confidence: Minimum confidence threshold

        Returns:
            List of interventions using this mechanism
        """
        interventions = defaultdict(lambda: defaultdict(list))

        # Search all edges for this mechanism
        for intervention in self.forward_edges:
            for cond, edges in self.forward_edges[intervention].items():
                # Skip if condition filter doesn't match
                if condition and cond != condition:
                    continue

                for edge in edges:
                    if (edge.evidence.confidence >= min_confidence and
                        edge.evidence.mechanism_canonical_name == mechanism):
                        interventions[intervention][cond].append(edge)

        # Aggregate results
        results = []
        for intervention, conditions in interventions.items():
            for cond, edges in conditions.items():
                total_weight = sum(e.weight * e.evidence.confidence for e in edges)
                avg_confidence = sum(e.evidence.confidence for e in edges) / len(edges)

                results.append({
                    'intervention': intervention,
                    'condition': cond,
                    'mechanism': mechanism,
                    'edge_count': len(edges),
                    'avg_confidence': round(avg_confidence, 3),
                    'aggregate_score': round(total_weight / len(edges) if edges else 0, 3),
                    'studies': [e.evidence.study_id for e in edges]
                })

        # Sort by aggregate score
        results.sort(key=lambda x: x['aggregate_score'], reverse=True)
        return results

    def query_conditions_by_mechanism(
        self,
        mechanism: str,
        intervention: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Query conditions affected by a specific mechanism.

        Args:
            mechanism: Mechanism (canonical name)
            intervention: Optional intervention filter
            min_confidence: Minimum confidence threshold

        Returns:
            List of conditions affected by this mechanism
        """
        conditions = defaultdict(lambda: defaultdict(list))

        # Search all edges for this mechanism
        for interv in self.forward_edges:
            # Skip if intervention filter doesn't match
            if intervention and interv != intervention:
                continue

            for condition, edges in self.forward_edges[interv].items():
                for edge in edges:
                    if (edge.evidence.confidence >= min_confidence and
                        edge.evidence.mechanism_canonical_name == mechanism):
                        conditions[condition][interv].append(edge)

        # Aggregate results
        results = []
        for condition, interventions in conditions.items():
            for interv, edges in interventions.items():
                total_weight = sum(e.weight * e.evidence.confidence for e in edges)
                avg_confidence = sum(e.evidence.confidence for e in edges) / len(edges)

                results.append({
                    'condition': condition,
                    'intervention': interv,
                    'mechanism': mechanism,
                    'edge_count': len(edges),
                    'avg_confidence': round(avg_confidence, 3),
                    'aggregate_score': round(total_weight / len(edges) if edges else 0, 3),
                    'studies': [e.evidence.study_id for e in edges]
                })

        # Sort by aggregate score
        results.sort(key=lambda x: x['aggregate_score'], reverse=True)
        return results

    def query_mechanisms_for_condition(
        self,
        condition: str,
        min_confidence: float = 0.0,
        evidence_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query all mechanisms that affect a specific condition.

        Args:
            condition: Condition (canonical name)
            min_confidence: Minimum confidence threshold
            evidence_types: Filter by evidence types

        Returns:
            List of mechanisms with aggregated evidence
        """
        if evidence_types is None:
            evidence_types = ['improves', 'worsens', 'no_effect', 'inconclusive']

        mechanisms = defaultdict(list)

        # Get all edges pointing to this condition
        if condition in self.reverse_edges:
            for intervention, edges in self.reverse_edges[condition].items():
                for edge in edges:
                    if (edge.evidence.confidence >= min_confidence and
                        edge.evidence.evidence_type in evidence_types and
                        edge.evidence.mechanism_canonical_name):
                        mechanisms[edge.evidence.mechanism_canonical_name].append(edge)

        # Aggregate evidence for each mechanism
        results = []
        for mechanism, edges in mechanisms.items():
            total_weight = sum(e.weight * e.evidence.confidence for e in edges)
            avg_confidence = sum(e.evidence.confidence for e in edges) / len(edges)

            evidence_breakdown = defaultdict(int)
            interventions_using = set()
            for edge in edges:
                evidence_breakdown[edge.evidence.evidence_type] += 1
                interventions_using.add(edge.source)

            results.append({
                'mechanism': mechanism,
                'condition': condition,
                'edge_count': len(edges),
                'interventions_count': len(interventions_using),
                'interventions': list(interventions_using),
                'avg_confidence': round(avg_confidence, 3),
                'aggregate_score': round(total_weight / len(edges) if edges else 0, 3),
                'evidence_breakdown': dict(evidence_breakdown)
            })

        # Sort by aggregate score
        results.sort(key=lambda x: x['aggregate_score'], reverse=True)
        return results

    def query_mechanisms_for_intervention(
        self,
        intervention: str,
        min_confidence: float = 0.0,
        evidence_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query all mechanisms used by a specific intervention.

        Args:
            intervention: Intervention (canonical name)
            min_confidence: Minimum confidence threshold
            evidence_types: Filter by evidence types

        Returns:
            List of mechanisms with aggregated evidence
        """
        if evidence_types is None:
            evidence_types = ['improves', 'worsens', 'no_effect', 'inconclusive']

        mechanisms = defaultdict(list)

        # Get all edges starting from this intervention
        if intervention in self.forward_edges:
            for condition, edges in self.forward_edges[intervention].items():
                for edge in edges:
                    if (edge.evidence.confidence >= min_confidence and
                        edge.evidence.evidence_type in evidence_types and
                        edge.evidence.mechanism_canonical_name):
                        mechanisms[edge.evidence.mechanism_canonical_name].append(edge)

        # Aggregate evidence for each mechanism
        results = []
        for mechanism, edges in mechanisms.items():
            total_weight = sum(e.weight * e.evidence.confidence for e in edges)
            avg_confidence = sum(e.evidence.confidence for e in edges) / len(edges)

            evidence_breakdown = defaultdict(int)
            conditions_treated = set()
            for edge in edges:
                evidence_breakdown[edge.evidence.evidence_type] += 1
                conditions_treated.add(edge.target)

            results.append({
                'mechanism': mechanism,
                'intervention': intervention,
                'edge_count': len(edges),
                'conditions_count': len(conditions_treated),
                'conditions': list(conditions_treated),
                'avg_confidence': round(avg_confidence, 3),
                'aggregate_score': round(total_weight / len(edges) if edges else 0, 3),
                'evidence_breakdown': dict(evidence_breakdown)
            })

        # Sort by aggregate score
        results.sort(key=lambda x: x['aggregate_score'], reverse=True)
        return results

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
            # Generate unique edge ID (include mechanism for uniqueness)
            edge_id = f"{edge.source}_{edge.target}_{edge.evidence.study_id}_{edge.edge_type}"
            if edge.evidence.mechanism_canonical_name:
                edge_id += f"_{edge.evidence.mechanism_canonical_name}"

            db_edge = KnowledgeGraphEdge(
                edge_id=edge_id,
                source_node_id=edge.source,
                target_node_id=edge.target,
                edge_type=edge.edge_type,
                edge_weight=edge.evidence.weight,
                evidence_type=edge.evidence.evidence_type,
                confidence=edge.evidence.confidence,
                # Mechanism fields (Phase 3c integration)
                mechanism_cluster_id=edge.evidence.mechanism_cluster_id,
                mechanism_canonical_name=edge.evidence.mechanism_canonical_name,
                mechanism_text_raw=edge.evidence.mechanism_raw_text,
                mechanism_similarity_score=edge.evidence.mechanism_similarity,
                # Study metadata
                study_id=edge.evidence.study_id,
                study_title=edge.evidence.title,
                sample_size=edge.evidence.sample_size,
                study_design=edge.evidence.study_design,
                publication_year=edge.evidence.publication_year,
                journal=edge.evidence.journal,
                doi=edge.evidence.doi,
                effect_size=edge.evidence.effect_size,
                p_value=edge.evidence.p_value,
                generation_model="knowledge_graph_v2",  # Updated version
                generation_version="2.0",
                created_at=datetime.now()
            )
            self.repository.save_knowledge_graph_edge(db_edge)

            if logger:
                logger.debug(f"Saved edge to database: {edge_id}")

        except Exception as e:
            if logger:
                logger.error(f"Error saving edge to database: {e}")
            raise

    # =================================================================
    # PHASE 3 INTEGRATION - Canonical Group Support
    # =================================================================

    def build_from_phase3_clusters(self, db_path: str) -> Dict[str, Any]:
        """
        Build knowledge graph from Phase 3 canonical groups (TRIPLE CANONICAL ARCHITECTURE).

        Uses canonical groups for ALL three entity types:
        - Interventions (source nodes)
        - Conditions (target nodes)
        - Mechanisms (edge labels)

        This creates a cleaner graph with pooled evidence and enables mechanism-based queries.

        Args:
            db_path: Path to intervention_research.db

        Returns:
            Dictionary with build statistics
        """
        import sqlite3

        if logger:
            logger.info("Building knowledge graph from Phase 3 canonical groups (triple canonical)...")

        start_time = datetime.now()
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # ===================================================================
            # STEP 1: Load ALL canonical groups (interventions, conditions, mechanisms)
            # ===================================================================
            cursor.execute("""
                SELECT DISTINCT cg.id, cg.canonical_name, cg.entity_type, cg.layer_0_category
                FROM canonical_groups cg
                WHERE cg.entity_type IN ('intervention', 'condition', 'mechanism')
                ORDER BY cg.entity_type, cg.canonical_name
            """)
            all_canonical_groups = cursor.fetchall()

            # Separate by entity type
            intervention_groups = [g for g in all_canonical_groups if g['entity_type'] == 'intervention']
            condition_groups = [g for g in all_canonical_groups if g['entity_type'] == 'condition']
            mechanism_groups = [g for g in all_canonical_groups if g['entity_type'] == 'mechanism']

            if logger:
                logger.info(f"Found {len(intervention_groups)} intervention groups, "
                          f"{len(condition_groups)} condition groups, "
                          f"{len(mechanism_groups)} mechanism groups")

            # ===================================================================
            # STEP 2: Build condition canonical lookup map
            # ===================================================================
            condition_canonical_map = {}
            for cond_group in condition_groups:
                cursor.execute("""
                    SELECT entity_name FROM semantic_hierarchy
                    WHERE layer_1_canonical = ? AND entity_type = 'condition'
                """, (cond_group['canonical_name'],))
                for row in cursor.fetchall():
                    condition_canonical_map[row['entity_name']] = cond_group['canonical_name']

            if logger:
                logger.info(f"Built condition canonical map with {len(condition_canonical_map)} entries")

            # ===================================================================
            # STEP 3: Process each intervention canonical group
            # ===================================================================
            interventions_processed = 0
            edges_created = 0
            edges_skipped_no_mechanism = 0

            for group in intervention_groups:
                canonical_id = group['id']
                canonical_name = group['canonical_name']
                category = group['layer_0_category']

                # Get all raw intervention names in this cluster
                cursor.execute("""
                    SELECT DISTINCT entity_name
                    FROM semantic_hierarchy
                    WHERE layer_1_canonical = ? AND entity_type = 'intervention'
                """, (canonical_name,))
                cluster_members = [row['entity_name'] for row in cursor.fetchall()]

                if not cluster_members:
                    continue

                # Get all interventions WITH mechanism data (joined with mechanism clusters)
                placeholders = ','.join(['?' for _ in cluster_members])
                cursor.execute(f"""
                    SELECT DISTINCT
                        p.pmid,
                        p.title,
                        substr(p.publication_date, 1, 4) as year,
                        i.intervention_name,
                        i.health_condition,
                        i.mechanism,
                        i.outcome_type as evidence_type,
                        i.study_confidence as confidence,
                        i.sample_size,
                        i.study_type as study_design,
                        p.journal,
                        p.doi,
                        mcm.cluster_id as mechanism_cluster_id,
                        mc.canonical_name as mechanism_canonical_name,
                        mcm.similarity_score as mechanism_similarity
                    FROM interventions i
                    JOIN papers p ON i.paper_id = p.pmid
                    LEFT JOIN mechanism_cluster_membership mcm ON i.mechanism = mcm.mechanism_text
                    LEFT JOIN mechanism_clusters mc ON mcm.cluster_id = mc.cluster_id
                    WHERE i.intervention_name IN ({placeholders})
                """, cluster_members)

                evidence_rows = cursor.fetchall()

                if not evidence_rows:
                    continue

                # Add intervention canonical group node
                self.add_node(
                    node_id=canonical_name,
                    node_type='intervention',
                    metadata={
                        'name': canonical_name,
                        'canonical_group_id': canonical_id,
                        'category': category,
                        'cluster_size': len(cluster_members),
                        'cluster_members': cluster_members,
                        'evidence_count': len(evidence_rows)
                    }
                )

                # Create edges for each piece of evidence
                for row in evidence_rows:
                    # Map raw condition to canonical
                    raw_condition = row['health_condition']
                    canonical_condition = condition_canonical_map.get(raw_condition, raw_condition)

                    # Skip edges without mechanisms (mechanisms ARE the edge labels)
                    if not row['mechanism_canonical_name']:
                        edges_skipped_no_mechanism += 1
                        if logger and edges_skipped_no_mechanism == 1:
                            logger.debug(f"Skipping edges with no mechanism (will count total at end)")
                        continue

                    # Map outcome_type to internal evidence_type (backward compatibility)
                    evidence_type_map = {
                        # New health-impact values
                        'improves': 'improves',
                        'worsens': 'worsens',
                        'no_effect': 'no_effect',
                        'inconclusive': 'inconclusive',
                        # Legacy values (if database has old data)
                        'positive': 'improves',
                        'negative': 'worsens',
                        'neutral': 'no_effect',
                        'no_correlation': 'no_effect',
                        'positive_correlation': 'improves',
                        'negative_correlation': 'worsens'
                    }
                    evidence_type = evidence_type_map.get(
                        row['evidence_type'].lower() if row['evidence_type'] else 'no_effect',
                        'no_effect'
                    )

                    # Create evidence object WITH mechanism data
                    evidence = StudyEvidence(
                        study_id=str(row['pmid']),
                        title=row['title'] or "Untitled",
                        evidence_type=evidence_type,
                        weight=0.0,  # Will be set by add_evidence
                        confidence=min(float(row['confidence'] or 0.5), 1.0),
                        sample_size=int(row['sample_size'] or 0),
                        # Mechanism fields (edge label)
                        mechanism_cluster_id=row['mechanism_cluster_id'],
                        mechanism_canonical_name=row['mechanism_canonical_name'],
                        mechanism_raw_text=row['mechanism'],
                        mechanism_similarity=row['mechanism_similarity'],
                        # Study metadata
                        study_design=row['study_design'] or "unknown",
                        publication_year=int(row['year']) if row['year'] else 0,
                        journal=row['journal'] or "",
                        doi=row['doi'] or ""
                    )

                    # Add edge (intervention canonical -> condition canonical, via mechanism)
                    self.add_evidence(
                        source=canonical_name,  # Intervention canonical
                        target=canonical_condition,  # Condition canonical
                        evidence=evidence,  # Contains mechanism data
                        edge_type='treats'  # Keep for legacy compatibility
                    )
                    edges_created += 1

                interventions_processed += 1

                if interventions_processed % 50 == 0 and logger:
                    logger.info(f"Processed {interventions_processed}/{len(intervention_groups)} intervention groups...")

            duration = (datetime.now() - start_time).total_seconds()

            result = {
                'success': True,
                'canonical_groups_processed': interventions_processed,  # For backward compatibility with orchestrator
                'intervention_groups_processed': interventions_processed,
                'condition_groups_loaded': len(condition_groups),
                'mechanism_groups_loaded': len(mechanism_groups),
                'nodes_created': len(self.nodes),
                'edges_created': edges_created,
                'edges_skipped_no_mechanism': edges_skipped_no_mechanism,
                'duration_seconds': duration,
                'avg_edges_per_group': edges_created / interventions_processed if interventions_processed > 0 else 0
            }

            if logger:
                logger.info(f"Knowledge graph built successfully (triple canonical architecture):")
                logger.info(f"  Intervention groups: {interventions_processed}")
                logger.info(f"  Condition groups: {len(condition_groups)}")
                logger.info(f"  Mechanism groups: {len(mechanism_groups)}")
                logger.info(f"  Nodes: {len(self.nodes)}")
                logger.info(f"  Edges: {edges_created}")
                logger.info(f"  Edges skipped (no mechanism): {edges_skipped_no_mechanism}")
                logger.info(f"  Duration: {duration:.1f}s")

            return result

        except Exception as e:
            if logger:
                logger.error(f"Failed to build knowledge graph from Phase 3 clusters: {e}")
            raise
        finally:
            conn.close()

    def get_canonical_group_for_intervention(self, intervention_name: str, db_path: str) -> Optional[str]:
        """
        Look up canonical group name for a raw intervention name.

        Args:
            intervention_name: Raw intervention name
            db_path: Path to intervention_research.db

        Returns:
            Canonical group name if found, None otherwise
        """
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT layer_1_canonical
                FROM semantic_hierarchy
                WHERE entity_name = ? AND entity_type = 'intervention'
                LIMIT 1
            """, (intervention_name,))

            row = cursor.fetchone()
            return row['layer_1_canonical'] if row else None

        finally:
            conn.close()

    def add_intervention_evidence(
        self,
        study_id: str,
        title: str,
        intervention_name: str,
        condition: str,
        evidence_type: str,
        confidence: float,
        sample_size: int = 0,
        study_design: str = "unknown",
        publication_year: int = 0,
        journal: str = "",
        doi: str = "",
        effect_size: Optional[float] = None,
        p_value: Optional[float] = None
    ) -> None:
        """
        Add intervention evidence to the graph.

        Convenience method that creates StudyEvidence and adds to graph.
        Used by data_mining_orchestrator.py for backward compatibility.

        Args:
            study_id: Study identifier (e.g., PMID)
            title: Paper title
            intervention_name: Intervention name (can be raw or canonical)
            condition: Health condition
            evidence_type: Type of evidence ('improves', 'worsens', 'no_effect', 'inconclusive')
            confidence: Confidence score (0-1)
            sample_size: Study sample size
            study_design: Type of study (RCT, observational, etc.)
            publication_year: Year published
            journal: Journal name
            doi: DOI identifier
            effect_size: Effect size if available
            p_value: P-value if available
        """
        evidence = StudyEvidence(
            study_id=study_id,
            title=title,
            evidence_type=evidence_type,
            weight=0.0,  # Will be set by add_evidence
            confidence=confidence,
            sample_size=sample_size,
            study_design=study_design,
            publication_year=publication_year,
            journal=journal,
            doi=doi,
            effect_size=effect_size,
            p_value=p_value
        )

        self.add_evidence(
            source=intervention_name,
            target=condition,
            evidence=evidence,
            edge_type='treats'
        )

    def get_all_interventions(self) -> List[str]:
        """Get list of all intervention nodes."""
        return [node_id for node_id, data in self.nodes.items() if data.get('type') == 'intervention']

    def get_all_conditions(self) -> List[str]:
        """Get list of all condition nodes."""
        return [node_id for node_id, data in self.nodes.items() if data.get('type') == 'condition']

    def get_all_intervention_condition_pairs(self) -> List[Tuple[str, str]]:
        """Get all intervention-condition pairs that have edges."""
        pairs = []
        for intervention in self.forward_edges:
            for condition in self.forward_edges[intervention]:
                pairs.append((intervention, condition))
        return pairs

    def save_to_file(self, filepath: str) -> None:
        """Save graph to JSON file (alias for export_to_json)."""
        self.export_to_json(filepath)

    def get_total_edge_count(self) -> int:
        """Get total number of edges in the graph."""
        return self.stats['total_edges']