"""
Test Script - Small Dataset Validation

Tests the Unified Phase 3 pipeline on a small subset of data:
- 10 interventions
- 10 conditions
- 20 mechanisms

Validates that all components work together correctly before running on full dataset.
"""

import sys
import sqlite3
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experimentation.unified_phase3.orchestrator import UnifiedPhase3Orchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_database(source_db: str, test_db: str):
    """
    Create a small test database with subset of data.

    Args:
        source_db: Path to full intervention_research.db
        test_db: Path to create test database
    """
    logger.info(f"Creating test database: {test_db}")

    # Delete existing test database
    import os
    if os.path.exists(test_db):
        os.remove(test_db)
        logger.info(f"Deleted existing test database")

    # Connect to source and test databases
    source_conn = sqlite3.connect(source_db)
    test_conn = sqlite3.connect(test_db)

    # Copy schema (skip internal tables)
    source_schema = source_conn.execute("""
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """).fetchall()
    for (sql,) in source_schema:
        if sql:  # Skip None
            test_conn.execute(sql)

    # Copy small subset of data
    logger.info("Copying subset of interventions...")

    # Get 10 unique interventions with mechanisms
    source_cursor = source_conn.cursor()
    source_cursor.execute("""
        SELECT DISTINCT intervention_name
        FROM interventions
        WHERE mechanism IS NOT NULL AND mechanism != '' AND mechanism != 'N/A'
        LIMIT 10
    """)
    intervention_names = [row[0] for row in source_cursor.fetchall()]
    logger.info(f"Selected interventions: {intervention_names}")

    # Get 10 unique conditions
    source_cursor.execute("""
        SELECT DISTINCT health_condition
        FROM interventions
        WHERE health_condition IS NOT NULL AND health_condition != ''
        LIMIT 10
    """)
    condition_names = [row[0] for row in source_cursor.fetchall()]
    logger.info(f"Selected conditions: {condition_names}")

    # Copy interventions matching selected names
    source_cursor.execute(f"""
        SELECT * FROM interventions
        WHERE intervention_name IN ({','.join('?'*len(intervention_names))})
        LIMIT 50
    """, intervention_names)

    columns = [desc[0] for desc in source_cursor.description]
    rows = source_cursor.fetchall()

    if rows:
        placeholders = ','.join('?' * len(columns))
        test_conn.executemany(f"INSERT INTO interventions VALUES ({placeholders})", rows)
        logger.info(f"Copied {len(rows)} intervention records")

    test_conn.commit()

    # Verify counts
    test_cursor = test_conn.cursor()
    test_cursor.execute("SELECT COUNT(DISTINCT intervention_name) FROM interventions")
    num_interventions = test_cursor.fetchone()[0]

    test_cursor.execute("SELECT COUNT(DISTINCT health_condition) FROM interventions")
    num_conditions = test_cursor.fetchone()[0]

    test_cursor.execute("SELECT COUNT(DISTINCT mechanism) FROM interventions WHERE mechanism IS NOT NULL AND mechanism != 'N/A'")
    num_mechanisms = test_cursor.fetchone()[0]

    logger.info(f"Test database created:")
    logger.info(f"  - {num_interventions} unique interventions")
    logger.info(f"  - {num_conditions} unique conditions")
    logger.info(f"  - {num_mechanisms} unique mechanisms")

    source_conn.close()
    test_conn.close()

    return num_interventions, num_conditions, num_mechanisms


def run_test_experiment(test_db: str, config_path: str):
    """
    Run test experiment on small dataset.

    Args:
        test_db: Path to test database
        config_path: Path to configuration file

    Returns:
        Dict with test results
    """
    logger.info("\n" + "="*60)
    logger.info("RUNNING TEST EXPERIMENT")
    logger.info("="*60)

    # Create orchestrator
    orchestrator = UnifiedPhase3Orchestrator(
        config_path=config_path,
        db_path=test_db,
        cache_dir="back_end/data/test_cache"
    )

    # Run pipeline
    result = orchestrator.run()

    # Print results
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS")
    logger.info("="*60)

    if result['success']:
        logger.info("[SUCCESS] Test experiment completed")
        logger.info(f"Duration: {result['duration_seconds']:.1f}s")

        for entity_type in ['interventions', 'conditions', 'mechanisms']:
            entity_results = result['results'].get(entity_type)
            if entity_results:
                logger.info(f"\n{entity_type.upper()}:")
                logger.info(f"  - Entities: {len(entity_results.entity_names)}")
                logger.info(f"  - Clusters: {entity_results.num_clusters}")
                logger.info(f"  - Natural clusters: {entity_results.num_natural_clusters}")
                logger.info(f"  - Singletons: {entity_results.num_singleton_clusters}")
                logger.info(f"  - Assignment rate: {entity_results.assignment_rate:.0%}")
                logger.info(f"  - Silhouette score: {entity_results.silhouette_score:.3f}" if entity_results.silhouette_score else "  - Silhouette score: N/A")
                logger.info(f"  - Naming failures: {entity_results.naming_failures}")
                logger.info(f"  - Sample clusters:")

                # Show 3 sample cluster names
                for i, (cluster_id, naming_result) in enumerate(list(entity_results.naming_results.items())[:3]):
                    logger.info(f"    {cluster_id}: {naming_result.canonical_name} ({naming_result.category})")
    else:
        logger.error(f"[FAILED] Test experiment failed: {result.get('error')}")

    return result


def validate_results(result: dict) -> bool:
    """
    Validate test results meet expected criteria.

    Args:
        result: Test experiment result

    Returns:
        True if validation passes
    """
    logger.info("\n" + "="*60)
    logger.info("VALIDATING RESULTS")
    logger.info("="*60)

    if not result['success']:
        logger.error("FAIL: Experiment did not complete successfully")
        return False

    validation_passed = True

    for entity_type in ['interventions', 'conditions', 'mechanisms']:
        entity_results = result['results'].get(entity_type)
        if not entity_results:
            logger.error(f"FAIL: No results for {entity_type}")
            validation_passed = False
            continue

        logger.info(f"\nValidating {entity_type}...")

        # Check 100% assignment
        if entity_results.assignment_rate != 1.0:
            logger.error(f"FAIL: Assignment rate is {entity_results.assignment_rate:.0%}, expected 100%")
            validation_passed = False
        else:
            logger.info(f"PASS: 100% assignment rate")

        # Check clusters were created
        if entity_results.num_clusters == 0:
            logger.error(f"FAIL: No clusters created")
            validation_passed = False
        else:
            logger.info(f"PASS: Created {entity_results.num_clusters} clusters")

        # Check naming succeeded
        if entity_results.naming_failures > entity_results.num_clusters * 0.1:  # Allow 10% failure
            logger.warning(f"WARNING: {entity_results.naming_failures} naming failures (>{10}% of clusters)")
        else:
            logger.info(f"PASS: Naming mostly successful ({entity_results.naming_failures} failures)")

        # Check embeddings exist (either generated or loaded from cache)
        if len(entity_results.embeddings) == 0:
            logger.error(f"FAIL: No embeddings available")
            validation_passed = False
        else:
            if entity_results.embeddings_generated > 0:
                logger.info(f"PASS: Generated {entity_results.embeddings_generated} embeddings")
            else:
                logger.info(f"PASS: Loaded {len(entity_results.embeddings)} embeddings from cache")

    if validation_passed:
        logger.info("\n" + "="*60)
        logger.info("[SUCCESS] ALL VALIDATIONS PASSED")
        logger.info("="*60)
    else:
        logger.error("\n" + "="*60)
        logger.error("[FAILED] SOME VALIDATIONS FAILED")
        logger.error("="*60)

    return validation_passed


def main():
    """Main test script."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Unified Phase 3 on small dataset")
    parser.add_argument('--source-db', required=True, help='Path to source intervention_research.db')
    parser.add_argument('--test-db', default='back_end/data/test_intervention_research.db',
                       help='Path to create test database')
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--skip-create', action='store_true', help='Skip test database creation')

    args = parser.parse_args()

    # Create test database
    if not args.skip_create:
        create_test_database(args.source_db, args.test_db)
    else:
        logger.info(f"Using existing test database: {args.test_db}")

    # Run test experiment
    result = run_test_experiment(args.test_db, args.config)

    # Validate results
    validation_passed = validate_results(result)

    # Exit with appropriate code
    sys.exit(0 if validation_passed else 1)


if __name__ == "__main__":
    main()
