#!/usr/bin/env python3
"""
Test script for EntityNormalizer class
"""

import sqlite3
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from entity_normalizer import EntityNormalizer


def test_entity_normalizer():
    """Test basic EntityNormalizer functionality"""

    # Connect to the database
    db_path = "data/processed/intervention_research.db"
    if not os.path.exists(db_path):
        print(f"[FAIL] Database not found at {db_path}")
        return False

    try:
        conn = sqlite3.connect(db_path)
        print(f"[PASS] Connected to database: {db_path}")

        # Initialize the normalizer
        normalizer = EntityNormalizer(conn)
        print("[PASS] EntityNormalizer instantiated successfully")

        # Test 1: Check that we can search for non-existent term (should return None)
        canonical_id = normalizer.find_canonical_id("non_existent_term", "intervention")
        if canonical_id is None:
            print("[PASS] find_canonical_id returns None for non-existent term")
        else:
            print(f"[FAIL] Expected None, got {canonical_id}")
            return False

        # Test 2: Create a test canonical entity
        try:
            test_canonical_id = normalizer.create_canonical_entity(
                canonical_name="test_probiotics",
                entity_type="intervention",
                scientific_name="Lactobacillus test strain"
            )
            print(f"[PASS] Created canonical entity with ID: {test_canonical_id}")
        except Exception as e:
            print(f"[FAIL] Failed to create canonical entity: {e}")
            return False

        # Test 3: Add a term mapping
        try:
            mapping_id = normalizer.add_term_mapping(
                original_term="probiotics test",
                canonical_id=test_canonical_id,
                confidence=1.0,
                method="manual_test"
            )
            print(f"[PASS] Created term mapping with ID: {mapping_id}")
        except Exception as e:
            print(f"[FAIL] Failed to create term mapping: {e}")
            return False

        # Test 4: Find the canonical ID we just created
        found_id = normalizer.find_canonical_id("probiotics test", "intervention")
        if found_id == test_canonical_id:
            print(f"[PASS] find_canonical_id found correct ID: {found_id}")
        else:
            print(f"[FAIL] Expected {test_canonical_id}, got {found_id}")
            return False

        # Test 5: Get canonical name
        canonical_name = normalizer.get_canonical_name("probiotics test", "intervention")
        if canonical_name == "test_probiotics":
            print(f"[PASS] get_canonical_name returned: {canonical_name}")
        else:
            print(f"[FAIL] Expected 'test_probiotics', got '{canonical_name}'")
            return False

        # Test 6: Get canonical name for unmapped term (should return original)
        original_name = normalizer.get_canonical_name("unmapped_term", "intervention")
        if original_name == "unmapped_term":
            print("[PASS] get_canonical_name returns original term when not mapped")
        else:
            print(f"[FAIL] Expected 'unmapped_term', got '{original_name}'")
            return False

        # Test 7: Get mapping statistics
        try:
            stats = normalizer.get_mapping_stats()
            print(f"[PASS] Got mapping stats: {stats}")
        except Exception as e:
            print(f"[FAIL] Failed to get mapping stats: {e}")
            return False

        # Clean up test data
        try:
            conn.execute("DELETE FROM entity_mappings WHERE canonical_id = ?", (test_canonical_id,))
            conn.execute("DELETE FROM canonical_entities WHERE id = ?", (test_canonical_id,))
            conn.commit()
            print("[PASS] Cleaned up test data")
        except Exception as e:
            print(f"[WARN] Warning: Failed to clean up test data: {e}")

        conn.close()
        print("\n[SUCCESS] All tests passed! EntityNormalizer is working correctly.")
        return True

    except Exception as e:
        print(f"[FAIL] Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = test_entity_normalizer()
    sys.exit(0 if success else 1)