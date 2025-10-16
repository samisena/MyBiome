"""
Export Phase 4a knowledge graph data to JSON format for D3.js network visualization.

Reads from knowledge_graph_nodes and knowledge_graph_edges tables and exports
to frontend/data/network_graph.json

⚠️  DEPRECATION WARNING (October 16, 2025):
    This script is now DEPRECATED in favor of Phase 5 automated frontend export.

    RECOMMENDED: Use Phase 5 instead:
      python -m back_end.src.orchestration.phase_5_frontend_updater

    Phase 5 Benefits:
      - Atomic file writes (no corrupted JSON)
      - Automatic backups (.bak files)
      - Post-export validation
      - Session tracking
      - Integrated into main pipeline (auto-runs after Phase 4b)

    This legacy script is kept for backward compatibility with manual exports only.
    New development should use Phase 5.

Usage:
    python -m back_end.src.utils.export_network_visualization_data
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DB_PATH = PROJECT_ROOT / "back_end" / "data" / "processed" / "intervention_research.db"
OUTPUT_PATH = PROJECT_ROOT / "frontend_network_viz_experiment" / "data" / "network_graph.json"


def parse_node_data(node_data_str: str) -> Dict[str, Any]:
    """Parse node_data JSON string, return dict or empty dict if invalid."""
    if not node_data_str:
        return {}
    try:
        return json.loads(node_data_str)
    except json.JSONDecodeError:
        return {}


def export_network_data() -> Dict[str, Any]:
    """
    Export knowledge graph from database to D3.js-compatible JSON format.

    Returns:
        Dictionary with 'nodes', 'links', and 'metadata' keys
    """
    print(f"Connecting to database: {DB_PATH}")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ===================================================================
    # STEP 1: Load all nodes
    # ===================================================================
    print("Loading nodes from knowledge_graph_nodes...")

    cursor.execute("""
        SELECT
            node_id,
            node_type,
            node_name,
            node_data
        FROM knowledge_graph_nodes
        ORDER BY node_type, node_name
    """)

    nodes = []
    node_types_count = defaultdict(int)
    categories_count = defaultdict(int)

    for row in cursor.fetchall():
        node_id = row['node_id']
        node_type = row['node_type']
        node_name = row['node_name']
        node_data = parse_node_data(row['node_data'])

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

    print(f"  Loaded {len(nodes)} nodes:")
    for node_type, count in node_types_count.items():
        print(f"    {node_type}: {count}")

    # ===================================================================
    # STEP 2: Load all edges
    # ===================================================================
    print("Loading edges from knowledge_graph_edges...")

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

    links = []
    evidence_types_count = defaultdict(int)
    mechanisms_set = set()

    for row in cursor.fetchall():
        link = {
            'source': row['source_node_id'],
            'target': row['target_node_id'],
            'edge_type': row['edge_type'],
            'effect': row['evidence_type'],  # 'positive', 'negative', 'neutral'
            'confidence': float(row['confidence']) if row['confidence'] else 0.5,
            'study_id': row['study_id']
        }

        # Add mechanism if available
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

    print(f"  Loaded {len(links)} edges:")
    for evidence_type, count in evidence_types_count.items():
        print(f"    {evidence_type}: {count}")

    # ===================================================================
    # STEP 3: Calculate statistics
    # ===================================================================
    print("Calculating statistics...")

    # Get all unique categories
    all_categories = sorted(categories_count.keys())

    metadata = {
        'export_date': datetime.now().isoformat(),
        'database_path': str(DB_PATH),
        'node_count': len(nodes),
        'edge_count': len(links),
        'node_types': dict(node_types_count),
        'evidence_types': dict(evidence_types_count),
        'categories': all_categories,
        'category_counts': dict(categories_count),
        'mechanism_count': len(mechanisms_set),
        'confidence_stats': {
            'min': min([link['confidence'] for link in links]) if links else 0,
            'max': max([link['confidence'] for link in links]) if links else 0,
            'avg': sum([link['confidence'] for link in links]) / len(links) if links else 0
        }
    }

    conn.close()

    # ===================================================================
    # STEP 4: Return data structure
    # ===================================================================
    return {
        'nodes': nodes,
        'links': links,
        'metadata': metadata
    }


def main():
    """Main export function."""
    print("=" * 60)
    print("PHASE 4A KNOWLEDGE GRAPH -> D3.JS NETWORK VISUALIZATION")
    print("=" * 60)
    print()

    try:
        # Export data
        data = export_network_data()

        # Ensure output directory exists
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON file
        print()
        print(f"Writing to: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        file_size_kb = OUTPUT_PATH.stat().st_size / 1024
        print(f"  File size: {file_size_kb:.1f} KB")

        # Print summary
        print()
        print("=" * 60)
        print("EXPORT SUMMARY")
        print("=" * 60)
        print(f"Nodes: {data['metadata']['node_count']}")
        print(f"  Interventions: {data['metadata']['node_types'].get('intervention', 0)}")
        print(f"  Conditions: {data['metadata']['node_types'].get('condition', 0)}")
        print()
        print(f"Edges: {data['metadata']['edge_count']}")
        print(f"  Positive: {data['metadata']['evidence_types'].get('positive', 0)}")
        print(f"  Negative: {data['metadata']['evidence_types'].get('negative', 0)}")
        print(f"  Neutral: {data['metadata']['evidence_types'].get('neutral', 0)}")
        print()
        print(f"Unique mechanisms: {data['metadata']['mechanism_count']}")
        print()
        print(f"Categories ({len(data['metadata']['categories'])}):")
        for category in data['metadata']['categories']:
            count = data['metadata']['category_counts'].get(category, 0)
            print(f"  {category}: {count}")
        print()
        print(f"Confidence range: {data['metadata']['confidence_stats']['min']:.2f} - {data['metadata']['confidence_stats']['max']:.2f}")
        print(f"Average confidence: {data['metadata']['confidence_stats']['avg']:.2f}")
        print()
        print("=" * 60)
        print("EXPORT COMPLETE")
        print("=" * 60)
        print()
        print(f"Next step: Open frontend_network_viz_experiment/index.html in browser")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
