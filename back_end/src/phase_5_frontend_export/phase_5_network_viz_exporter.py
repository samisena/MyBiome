"""
Phase 5 Network Visualization Exporter

Exports Phase 4a knowledge graph data to D3.js-compatible JSON format.
Refactored from back_end/src/utils/export_network_visualization_data.py
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

from back_end.src.data.config import setup_logging
from .phase_5_base_exporter import BaseExporter

logger = setup_logging(__name__, 'phase_5_network_viz.log')


class NetworkVizExporter(BaseExporter):
    """
    Export knowledge graph for D3.js network visualization.

    Reads from Phase 4a knowledge graph tables:
    - knowledge_graph_nodes
    - knowledge_graph_edges
    """

    def __init__(self, db_path: str = None, config_path: str = None):
        super().__init__(db_path=db_path, config_path=config_path, export_type="network_viz")

    def extract_data(self) -> Dict[str, Any]:
        """Extract knowledge graph nodes and edges."""
        conn = self.get_database_connection()
        cursor = conn.cursor()

        try:
            # Load all nodes
            cursor.execute("""
                SELECT
                    node_id,
                    node_type,
                    node_name,
                    node_data
                FROM knowledge_graph_nodes
                ORDER BY node_type, node_name
            """)
            node_rows = cursor.fetchall()

            # Load all edges
            cursor.execute("""
                SELECT
                    source_node_id,
                    target_node_id,
                    edge_type,
                    edge_weight,
                    evidence_type,
                    confidence,
                    study_id,
                    mechanism_canonical_name,
                    mechanism_text_raw
                FROM knowledge_graph_edges
                ORDER BY source_node_id, target_node_id
            """)
            edge_rows = cursor.fetchall()

            return {
                'node_rows': node_rows,
                'edge_rows': edge_rows
            }

        finally:
            conn.close()

    def transform_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform knowledge graph to D3.js format."""
        nodes = []
        links = []
        node_types_count = defaultdict(int)
        categories_count = defaultdict(int)
        evidence_types_count = defaultdict(int)
        mechanisms_set = set()

        # Process nodes
        for row in raw_data['node_rows']:
            node_id = row['node_id']
            node_type = row['node_type']
            node_name = row['node_name']
            node_data = self._parse_node_data(row['node_data'])

            node = {
                'id': node_id,
                'name': node_name,
                'type': node_type
            }

            # Add intervention-specific metadata
            if node_type == 'intervention':
                node['category'] = node_data.get('category', 'unknown')
                node['cluster_size'] = node_data.get('cluster_size', 1)
                node['evidence_count'] = node_data.get('evidence_count', 0)
                categories_count[node['category']] += 1

            nodes.append(node)
            node_types_count[node_type] += 1

        # Process edges
        for row in raw_data['edge_rows']:
            link = {
                'source': row['source_node_id'],
                'target': row['target_node_id'],
                'edge_type': row['edge_type'],
                'effect': row['evidence_type'],
                'confidence': float(row['confidence']) if row['confidence'] else 0.5,
                'study_id': row['study_id']
            }

            # Add mechanism
            mechanism = row['mechanism_canonical_name']
            if mechanism:
                link['mechanism'] = mechanism
                mechanisms_set.add(mechanism)
            elif row['mechanism_text_raw']:
                link['mechanism'] = row['mechanism_text_raw']
                mechanisms_set.add(row['mechanism_text_raw'])
            else:
                link['mechanism'] = 'Unknown mechanism'

            links.append(link)
            evidence_types_count[row['evidence_type']] += 1

        # Calculate statistics
        all_categories = sorted(categories_count.keys())
        confidence_values = [link['confidence'] for link in links] if links else [0]

        metadata = {
            'export_date': datetime.now().isoformat(),
            'database_path': str(self.db_path),
            'node_count': len(nodes),
            'edge_count': len(links),
            'node_types': dict(node_types_count),
            'evidence_types': dict(evidence_types_count),
            'categories': all_categories,
            'category_counts': dict(categories_count),
            'mechanism_count': len(mechanisms_set),
            'confidence_stats': {
                'min': min(confidence_values),
                'max': max(confidence_values),
                'avg': sum(confidence_values) / len(confidence_values) if confidence_values else 0
            }
        }

        return {
            'nodes': nodes,
            'links': links,
            'metadata': metadata
        }

    def _parse_node_data(self, node_data_str: str) -> Dict[str, Any]:
        """Parse node_data JSON string."""
        if not node_data_str:
            return {}
        try:
            return json.loads(node_data_str)
        except json.JSONDecodeError:
            return {}

    def _get_output_path(self) -> Path:
        """Get output path for network visualization JSON."""
        return self.resolve_output_path('network_viz')

    def _count_records(self, data: Dict[str, Any]) -> int:
        """Count total nodes and edges."""
        return len(data.get('node_rows', [])) + len(data.get('edge_rows', []))
