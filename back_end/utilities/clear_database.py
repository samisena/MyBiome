#!/usr/bin/env python3
"""
Script to clear the pubmed_research.db database.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

# Add back_end directory for imports
back_end_dir = Path(__file__).parent
sys.path.insert(0, str(back_end_dir))

try:
    from src.paper_collection.database_manager import database_manager
    from src.data.config import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)


def clear_database():
    """Clear all entries from the database."""
    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Clear interventions first (due to foreign key constraints)
            cursor.execute("DELETE FROM interventions")
            interventions_deleted = cursor.rowcount

            cursor.execute("DELETE FROM intervention_extractions")
            extractions_deleted = cursor.rowcount

            cursor.execute("DELETE FROM intervention_consensus")
            consensus_deleted = cursor.rowcount

            # Clear papers
            cursor.execute("DELETE FROM papers")
            papers_deleted = cursor.rowcount

            # Reset auto-increment counters
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='interventions'")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='intervention_extractions'")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='intervention_consensus'")

            conn.commit()

            print(f"Database cleared:")
            print(f"  - {papers_deleted} papers deleted")
            print(f"  - {interventions_deleted} interventions deleted")
            print(f"  - {extractions_deleted} extractions deleted")
            print(f"  - {consensus_deleted} consensus records deleted")

            return True

    except Exception as e:
        print(f"Error clearing database: {e}")
        return False


def main():
    """Main function to clear database."""
    print("=== MyBiome Database Clear ===")

    # Setup logging
    logger = setup_logging(__name__, 'clear_database.log')

    # Ask for confirmation
    response = input("\nAre you sure you want to clear the entire database? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Operation cancelled.")
        return False

    # Clear database
    print("\nClearing database...")
    if not clear_database():
        print("Failed to clear database.")
        return False

    print("\n=== Database cleared successfully! ===")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)