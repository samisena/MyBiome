"""
Hierarchy Manager for Hierarchical Semantic Normalization

Manages hierarchical layer assignment and aggregation rules for interventions.
Populates database tables: semantic_hierarchy, entity_relationships, canonical_groups.
"""

import re
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HierarchyManager:
    """
    Manage hierarchical layer assignment and database operations.
    """

    # Dosage extraction patterns
    DOSAGE_PATTERNS = [
        r'\d+\s*(mg|g|mcg|µg|ug)',
        r'\d+\s*(IU|iu)',
        r'\d+\s*x?\s*10\^?\d+\s*(CFU|cfu)',
        r'\d+(\.\d+)?\s*(ml|mL)',
        r'\d+\s*(units?)',
    ]

    # Aggregation rules per relationship type (layer-based taxonomy)
    AGGREGATION_RULES = {
        'EXACT_MATCH': {
            'share_layer_0': True,
            'share_layer_1': True,
            'share_layer_2': True,
            'share_layer_3': True,
            'rule': 'merge_completely'
        },
        'DOSAGE_VARIANT': {
            'share_layer_0': True,
            'share_layer_1': True,
            'share_layer_2': True,
            'share_layer_3': False,
            'rule': 'share_layers_0_1_2'
        },
        'SAME_CATEGORY_TYPE_VARIANT': {
            'share_layer_0': True,
            'share_layer_1': True,
            'share_layer_2': False,
            'share_layer_3': False,
            'rule': 'share_layers_0_1'
        },
        'SAME_CATEGORY': {
            'share_layer_0': True,
            'share_layer_1': False,
            'share_layer_2': False,
            'share_layer_3': False,
            'rule': 'share_layer_0_only'
        },
        'DIFFERENT': {
            'share_layer_0': False,
            'share_layer_1': False,
            'share_layer_2': False,
            'share_layer_3': False,
            'rule': 'no_relationship'
        }
    }

    def __init__(self, db_path: str):
        """
        Initialize the hierarchy manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        logger.info(f"HierarchyManager initialized with database: {db_path}")

    def extract_dosage(self, intervention_name: str) -> Optional[str]:
        """
        Extract dosage/detail information from intervention name.

        Args:
            intervention_name: Full intervention name

        Returns:
            Dosage string if found, else None
        """
        for pattern in self.DOSAGE_PATTERNS:
            match = re.search(pattern, intervention_name, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def create_semantic_entity(
        self,
        entity_name: str,
        entity_type: str,
        layer_0_category: Optional[str],
        layer_1_canonical: str,
        layer_2_variant: str,
        layer_3_detail: Optional[str],
        relationship_type: Optional[str],
        aggregation_rule: Optional[str],
        embedding_vector: Optional[bytes] = None,
        embedding_model: Optional[str] = None,
        source_table: str = 'interventions',
        source_ids: Optional[str] = None
    ) -> int:
        """
        Create a semantic hierarchy entity in the database.

        Args:
            entity_name: Original entity name
            entity_type: 'intervention' or 'condition'
            layer_0_category: Taxonomy category
            layer_1_canonical: Canonical group name
            layer_2_variant: Specific variant name
            layer_3_detail: Dosage/detail string
            relationship_type: Type of relationship to parent
            aggregation_rule: Aggregation rule code
            embedding_vector: Serialized embedding (optional)
            embedding_model: Model used for embedding (optional)
            source_table: Source database table
            source_ids: JSON array of source IDs

        Returns:
            ID of created entity
        """
        query = """
        INSERT INTO semantic_hierarchy (
            entity_name, entity_type, layer_0_category, layer_1_canonical,
            layer_2_variant, layer_3_detail, relationship_type, aggregation_rule,
            embedding_vector, embedding_model, embedding_dimension,
            source_table, source_ids, occurrence_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(entity_name, entity_type, layer_2_variant)
        DO UPDATE SET
            occurrence_count = occurrence_count + 1,
            updated_at = CURRENT_TIMESTAMP
        """

        embedding_dim = 768 if embedding_vector else None

        self.cursor.execute(query, (
            entity_name, entity_type, layer_0_category, layer_1_canonical,
            layer_2_variant, layer_3_detail, relationship_type, aggregation_rule,
            embedding_vector, embedding_model, embedding_dim,
            source_table, source_ids
        ))
        self.conn.commit()

        return self.cursor.lastrowid

    def get_or_create_canonical_group(
        self,
        canonical_name: str,
        entity_type: str,
        layer_0_category: Optional[str] = None,
        description: Optional[str] = None
    ) -> int:
        """
        Get existing canonical group or create new one.

        Args:
            canonical_name: Canonical group name
            entity_type: 'intervention' or 'condition'
            layer_0_category: Taxonomy category
            description: Group description

        Returns:
            ID of canonical group
        """
        # Check if exists
        self.cursor.execute(
            "SELECT id FROM canonical_groups WHERE canonical_name = ? AND entity_type = ?",
            (canonical_name, entity_type)
        )
        row = self.cursor.fetchone()

        if row:
            return row['id']

        # Create new
        query = """
        INSERT INTO canonical_groups (
            canonical_name, entity_type, layer_0_category, description, member_count, total_paper_count
        ) VALUES (?, ?, ?, ?, 0, 0)
        """

        self.cursor.execute(query, (canonical_name, entity_type, layer_0_category, description))
        self.conn.commit()

        return self.cursor.lastrowid

    def create_entity_relationship(
        self,
        entity_1_id: int,
        entity_2_id: int,
        relationship_type: str,
        relationship_confidence: float,
        source: str,
        labeled_by: Optional[str] = None,
        similarity_score: Optional[float] = None
    ) -> int:
        """
        Create relationship between two entities.

        Args:
            entity_1_id: First entity ID
            entity_2_id: Second entity ID
            relationship_type: Relationship type code
            relationship_confidence: Confidence score (0.0-1.0)
            source: Source of relationship ('llm_inference', 'manual_labeling', etc.)
            labeled_by: Who/what labeled this
            similarity_score: Embedding similarity (optional)

        Returns:
            ID of created relationship
        """
        # Ensure canonical ordering (smaller ID first)
        if entity_1_id > entity_2_id:
            entity_1_id, entity_2_id = entity_2_id, entity_1_id

        # Get aggregation rules
        rules = self.AGGREGATION_RULES.get(relationship_type, {})
        share_layer_1 = rules.get('share_layer_1', False)
        share_layer_2 = rules.get('share_layer_2', False)

        query = """
        INSERT OR IGNORE INTO entity_relationships (
            entity_1_id, entity_2_id, relationship_type, relationship_confidence,
            source, labeled_by, share_layer_1, share_layer_2, similarity_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        self.cursor.execute(query, (
            entity_1_id, entity_2_id, relationship_type, relationship_confidence,
            source, labeled_by, share_layer_1, share_layer_2, similarity_score
        ))
        self.conn.commit()

        return self.cursor.lastrowid

    def update_canonical_group_stats(self, canonical_name: str, entity_type: str):
        """
        Update member count and paper count for a canonical group.

        Args:
            canonical_name: Canonical group name
            entity_type: 'intervention' or 'condition'
        """
        # Count members
        self.cursor.execute("""
            SELECT
                COUNT(DISTINCT layer_2_variant) as member_count,
                SUM(occurrence_count) as total_papers
            FROM semantic_hierarchy
            WHERE layer_1_canonical = ? AND entity_type = ?
        """, (canonical_name, entity_type))

        row = self.cursor.fetchone()
        member_count = row['member_count'] or 0
        total_papers = row['total_papers'] or 0

        # Update canonical group
        self.cursor.execute("""
            UPDATE canonical_groups
            SET member_count = ?, total_paper_count = ?, updated_at = CURRENT_TIMESTAMP
            WHERE canonical_name = ? AND entity_type = ?
        """, (member_count, total_papers, canonical_name, entity_type))

        self.conn.commit()

    def get_entity_by_name(self, entity_name: str, entity_type: str) -> Optional[Dict]:
        """
        Get semantic hierarchy entity by name.

        Args:
            entity_name: Entity name to search
            entity_type: 'intervention' or 'condition'

        Returns:
            Entity dict if found, else None
        """
        self.cursor.execute("""
            SELECT * FROM semantic_hierarchy
            WHERE entity_name = ? AND entity_type = ?
        """, (entity_name, entity_type))

        row = self.cursor.fetchone()
        return dict(row) if row else None

    def get_canonical_group_members(self, canonical_name: str, entity_type: str) -> List[Dict]:
        """
        Get all members of a canonical group.

        Args:
            canonical_name: Canonical group name
            entity_type: 'intervention' or 'condition'

        Returns:
            List of entity dicts
        """
        self.cursor.execute("""
            SELECT * FROM semantic_hierarchy
            WHERE layer_1_canonical = ? AND entity_type = ?
            ORDER BY occurrence_count DESC
        """, (canonical_name, entity_type))

        return [dict(row) for row in self.cursor.fetchall()]

    def find_related_entities(
        self,
        entity_id: int,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Find entities related to a given entity.

        Args:
            entity_id: Source entity ID
            relationship_types: Filter by relationship types (optional)

        Returns:
            List of related entity dicts with relationship info
        """
        query = """
        SELECT
            sh.*,
            er.relationship_type,
            er.relationship_confidence,
            er.similarity_score
        FROM entity_relationships er
        JOIN semantic_hierarchy sh ON (
            (er.entity_1_id = ? AND sh.id = er.entity_2_id) OR
            (er.entity_2_id = ? AND sh.id = er.entity_1_id)
        )
        """

        params = [entity_id, entity_id]

        if relationship_types:
            placeholders = ','.join(['?'] * len(relationship_types))
            query += f" WHERE er.relationship_type IN ({placeholders})"
            params.extend(relationship_types)

        self.cursor.execute(query, params)

        return [dict(row) for row in self.cursor.fetchall()]

    def get_hierarchy_stats(self) -> Dict:
        """Get hierarchy statistics."""
        stats = {}

        # Total entities
        self.cursor.execute("SELECT COUNT(*) as count FROM semantic_hierarchy")
        stats['total_entities'] = self.cursor.fetchone()['count']

        # Entities by type
        self.cursor.execute("""
            SELECT entity_type, COUNT(*) as count
            FROM semantic_hierarchy
            GROUP BY entity_type
        """)
        stats['by_type'] = {row['entity_type']: row['count'] for row in self.cursor.fetchall()}

        # Canonical groups
        self.cursor.execute("SELECT COUNT(*) as count FROM canonical_groups")
        stats['total_canonical_groups'] = self.cursor.fetchone()['count']

        # Relationships
        self.cursor.execute("SELECT COUNT(*) as count FROM entity_relationships")
        stats['total_relationships'] = self.cursor.fetchone()['count']

        # Relationships by type
        self.cursor.execute("""
            SELECT relationship_type, COUNT(*) as count
            FROM entity_relationships
            GROUP BY relationship_type
        """)
        stats['relationships_by_type'] = {row['relationship_type']: row['count'] for row in self.cursor.fetchall()}

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __del__(self):
        """Cleanup: close connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def initialize_database_schema(db_path: str):
    """
    Initialize database schema from HIERARCHICAL_SCHEMA.sql.

    Args:
        db_path: Path to SQLite database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Read schema file
    schema_path = "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/HIERARCHICAL_SCHEMA.sql"

    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()

    # Split and execute statements
    statements = schema_sql.split(';')

    for statement in statements:
        statement = statement.strip()
        if statement and not statement.startswith('--'):
            try:
                cursor.execute(statement)
            except sqlite3.OperationalError as e:
                # Skip errors for already existing tables
                if 'already exists' not in str(e):
                    logger.warning(f"Schema execution warning: {e}")

    conn.commit()
    conn.close()

    logger.info(f"Database schema initialized at: {db_path}")


if __name__ == "__main__":
    # Test the hierarchy manager
    print("Testing HierarchyManager...")

    db_path = "c:/Users/samis/Desktop/MyBiome/back_end/experiments/semantic_normalization/data/test_hierarchy.db"

    # Initialize schema
    initialize_database_schema(db_path)

    manager = HierarchyManager(db_path)

    # Test dosage extraction
    print("\n=== Dosage Extraction Test ===")
    test_names = [
        "metformin 500mg",
        "Lactobacillus reuteri 10^9 CFU",
        "vitamin D 1000 IU",
        "atorvastatin"
    ]

    for name in test_names:
        dosage = manager.extract_dosage(name)
        print(f"{name} -> Dosage: {dosage or 'None'}")

    # Test entity creation
    print("\n=== Entity Creation Test ===")

    # Create canonical group
    canonical_id = manager.get_or_create_canonical_group(
        canonical_name="probiotics",
        entity_type="intervention",
        layer_0_category="supplement",
        description="Beneficial bacterial supplements"
    )
    print(f"Created canonical group 'probiotics' with ID: {canonical_id}")

    # Create entities
    entity_1_id = manager.create_semantic_entity(
        entity_name="Lactobacillus reuteri DSM 17938",
        entity_type="intervention",
        layer_0_category="supplement",
        layer_1_canonical="probiotics",
        layer_2_variant="L. reuteri",
        layer_3_detail="DSM 17938",
        relationship_type="SAME_CATEGORY_TYPE_VARIANT",
        aggregation_rule="share_layers_0_1"
    )
    print(f"Created entity 'L. reuteri' with ID: {entity_1_id}")

    entity_2_id = manager.create_semantic_entity(
        entity_name="Saccharomyces boulardii",
        entity_type="intervention",
        layer_0_category="supplement",
        layer_1_canonical="probiotics",
        layer_2_variant="S. boulardii",
        layer_3_detail=None,
        relationship_type="SAME_CATEGORY_TYPE_VARIANT",
        aggregation_rule="share_layers_0_1"
    )
    print(f"Created entity 'S. boulardii' with ID: {entity_2_id}")

    # Create relationship
    rel_id = manager.create_entity_relationship(
        entity_1_id=entity_1_id,
        entity_2_id=entity_2_id,
        relationship_type="SAME_CATEGORY_TYPE_VARIANT",
        relationship_confidence=0.85,
        source="llm_inference",
        labeled_by="qwen3:14b",
        similarity_score=0.72
    )
    print(f"Created relationship with ID: {rel_id}")

    # Update canonical stats
    manager.update_canonical_group_stats("probiotics", "intervention")
    print("Updated canonical group stats")

    # Get stats
    stats = manager.get_hierarchy_stats()
    print(f"\nHierarchy stats: {stats}")

    manager.close()
