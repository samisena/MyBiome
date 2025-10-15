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

    # Update canonical stats
    manager.update_canonical_group_stats("probiotics", "intervention")
    print("Updated canonical group stats")

    # Get stats
    stats = manager.get_hierarchy_stats()
    print(f"\nHierarchy stats: {stats}")

    manager.close()
