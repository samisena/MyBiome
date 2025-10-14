"""
Validation Module for Group-Based Categorization

Validates that:
1. All interventions have categories (100% coverage)
2. Groups are semantically pure (don't span multiple categories)
3. Results agree with existing categorizations
"""

import sqlite3
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_category_coverage(db_path: str) -> Dict:
    """
    Ensure 100% of interventions have categories.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dict with coverage statistics
    """
    logger.info("Validating category coverage")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Total interventions
    cursor.execute("SELECT COUNT(DISTINCT intervention_name) FROM interventions")
    total_interventions = cursor.fetchone()[0]

    # Categorized interventions
    cursor.execute("""
        SELECT COUNT(DISTINCT intervention_name)
        FROM interventions
        WHERE intervention_category IS NOT NULL AND intervention_category != ''
    """)
    categorized = cursor.fetchone()[0]

    # Interventions with groups
    cursor.execute("""
        SELECT COUNT(DISTINCT i.intervention_name)
        FROM interventions i
        JOIN semantic_hierarchy sh ON i.intervention_name = sh.entity_name
        WHERE sh.entity_type = 'intervention'
    """)
    grouped = cursor.fetchone()[0]

    # Orphan interventions (not in any group)
    cursor.execute("""
        SELECT COUNT(DISTINCT intervention_name)
        FROM interventions
        WHERE intervention_name NOT IN (
            SELECT entity_name FROM semantic_hierarchy WHERE entity_type = 'intervention'
        )
    """)
    orphans = cursor.fetchone()[0]

    # Uncategorized interventions
    cursor.execute("""
        SELECT intervention_name
        FROM interventions
        WHERE intervention_category IS NULL OR intervention_category = ''
    """)
    uncategorized = [row[0] for row in cursor.fetchall()]

    conn.close()

    coverage_rate = categorized / total_interventions if total_interventions > 0 else 0
    orphan_rate = orphans / total_interventions if total_interventions > 0 else 0

    result = {
        'total_interventions': total_interventions,
        'categorized': categorized,
        'uncategorized_count': len(uncategorized),
        'uncategorized_names': uncategorized[:20],  # First 20 for inspection
        'grouped': grouped,
        'orphans': orphans,
        'coverage_rate': coverage_rate,
        'orphan_rate': orphan_rate,
        'passed': len(uncategorized) == 0
    }

    logger.info(f"Coverage: {categorized}/{total_interventions} ({coverage_rate*100:.1f}%)")
    logger.info(f"Grouped: {grouped}, Orphans: {orphans} ({orphan_rate*100:.1f}%)")

    if result['passed']:
        logger.info("✓ Coverage validation PASSED: 100% interventions categorized")
    else:
        logger.warning(f"✗ Coverage validation FAILED: {len(uncategorized)} interventions uncategorized")

    return result


def validate_group_purity(db_path: str, show_examples: int = 5) -> Dict:
    """
    Check if groups span multiple categories (potential categorization errors).

    Args:
        db_path: Path to SQLite database
        show_examples: Number of example mixed groups to show

    Returns:
        Dict with purity statistics
    """
    logger.info("Validating group purity")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Find groups where members have different original categories
    # (This helps identify potential miscategorizations)
    query = """
    SELECT
        sh.layer_1_canonical AS canonical_name,
        cg.layer_0_category AS group_category,
        COUNT(DISTINCT sh.entity_name) AS member_count,
        GROUP_CONCAT(DISTINCT sh.layer_0_category) AS member_categories
    FROM semantic_hierarchy sh
    LEFT JOIN canonical_groups cg ON sh.layer_1_canonical = cg.canonical_name
    WHERE sh.entity_type = 'intervention'
    GROUP BY sh.layer_1_canonical, cg.layer_0_category
    """

    cursor.execute(query)
    groups = [dict(row) for row in cursor.fetchall()]

    # Analyze purity
    pure_groups = []
    mixed_groups = []

    for group in groups:
        member_cats = group['member_categories']
        if member_cats:
            unique_cats = set(cat.strip() for cat in member_cats.split(',') if cat.strip())
            if len(unique_cats) > 1:
                mixed_groups.append({
                    'canonical_name': group['canonical_name'],
                    'group_category': group['group_category'],
                    'member_categories': list(unique_cats),
                    'member_count': group['member_count']
                })
            elif len(unique_cats) == 1:
                pure_groups.append(group)

    purity_rate = len(pure_groups) / len(groups) if groups else 0

    conn.close()

    result = {
        'total_groups': len(groups),
        'pure_groups': len(pure_groups),
        'mixed_groups': len(mixed_groups),
        'purity_rate': purity_rate,
        'mixed_examples': mixed_groups[:show_examples],
        'passed': len(mixed_groups) < len(groups) * 0.1  # <10% mixed acceptable
    }

    logger.info(f"Purity: {len(pure_groups)}/{len(groups)} groups pure ({purity_rate*100:.1f}%)")
    logger.info(f"Mixed groups: {len(mixed_groups)} ({len(mixed_groups)/len(groups)*100:.1f}%)")

    if mixed_groups:
        logger.info(f"\nExample mixed groups (first {show_examples}):")
        for mg in mixed_groups[:show_examples]:
            logger.info(f"  - {mg['canonical_name']}: group={mg['group_category']}, members={mg['member_categories']}")

    if result['passed']:
        logger.info("✓ Purity validation PASSED: <10% mixed groups")
    else:
        logger.warning(f"✗ Purity validation FAILED: {len(mixed_groups)} mixed groups")

    return result


def compare_with_existing(db_path: str, show_disagreements: int = 10) -> Dict:
    """
    Compare group-based categories with existing individual categories.

    Args:
        db_path: Path to SQLite database
        show_disagreements: Number of disagreement examples to show

    Returns:
        Dict with comparison statistics
    """
    logger.info("Comparing with existing categorizations")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # For interventions that had original categories (from Phase 2.5),
    # compare with new group-based categories
    query = """
    SELECT
        i.intervention_name,
        i.intervention_category AS current_category,
        sh.layer_0_category AS original_hierarchy_category,
        cg.layer_0_category AS group_category,
        sh.layer_1_canonical AS canonical_name
    FROM interventions i
    JOIN semantic_hierarchy sh ON i.intervention_name = sh.entity_name
    JOIN canonical_groups cg ON sh.layer_1_canonical = cg.canonical_name
    WHERE sh.entity_type = 'intervention'
    AND i.intervention_category IS NOT NULL
    AND i.intervention_category != ''
    """

    cursor.execute(query)
    rows = [dict(row) for row in cursor.fetchall()]

    if not rows:
        logger.info("No existing categorizations to compare")
        return {
            'total_compared': 0,
            'agreements': 0,
            'disagreements': 0,
            'agreement_rate': None,
            'passed': True
        }

    agreements = []
    disagreements = []

    for row in rows:
        current = row['current_category']
        group = row['group_category']

        if current == group:
            agreements.append(row)
        else:
            disagreements.append(row)

    agreement_rate = len(agreements) / len(rows) if rows else 0

    conn.close()

    result = {
        'total_compared': len(rows),
        'agreements': len(agreements),
        'disagreements': len(disagreements),
        'agreement_rate': agreement_rate,
        'disagreement_examples': disagreements[:show_disagreements],
        'passed': agreement_rate >= 0.95  # >95% agreement required
    }

    logger.info(f"Comparison: {len(agreements)}/{len(rows)} agreements ({agreement_rate*100:.1f}%)")
    logger.info(f"Disagreements: {len(disagreements)}")

    if disagreements:
        logger.info(f"\nExample disagreements (first {show_disagreements}):")
        for d in disagreements[:show_disagreements]:
            logger.info(f"  - {d['intervention_name']}: current={d['current_category']}, group={d['group_category']} (group: {d['canonical_name']})")

    if result['passed']:
        logger.info("✓ Comparison PASSED: >95% agreement")
    else:
        logger.warning(f"✗ Comparison FAILED: {agreement_rate*100:.1f}% agreement (target: >95%)")

    return result


def validate_all(db_path: str) -> Dict:
    """
    Run all validation checks.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dict with all validation results
    """
    logger.info("=" * 60)
    logger.info("Running all validation checks")
    logger.info("=" * 60)

    results = {
        'coverage': validate_category_coverage(db_path),
        'purity': validate_group_purity(db_path),
        'comparison': compare_with_existing(db_path)
    }

    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)

    all_passed = all(results[key]['passed'] for key in results if 'passed' in results[key])

    for check, result in results.items():
        status = "✓ PASSED" if result.get('passed', False) else "✗ FAILED"
        logger.info(f"{check.upper()}: {status}")

    if all_passed:
        logger.info("\n✓ ALL VALIDATION CHECKS PASSED")
    else:
        logger.warning("\n✗ SOME VALIDATION CHECKS FAILED")

    results['all_passed'] = all_passed

    return results


if __name__ == "__main__":
    # Test validation
    from back_end.src.data.config import config

    results = validate_all(config.db_path)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Coverage: {results['coverage']['coverage_rate']*100:.1f}% ({results['coverage']['passed'] and 'PASSED' or 'FAILED'})")
    print(f"Purity: {results['purity']['purity_rate']*100:.1f}% ({results['purity']['passed'] and 'PASSED' or 'FAILED'})")
    if results['comparison']['agreement_rate'] is not None:
        print(f"Agreement: {results['comparison']['agreement_rate']*100:.1f}% ({results['comparison']['passed'] and 'PASSED' or 'FAILED'})")
    print(f"\nOverall: {results['all_passed'] and 'PASSED' or 'FAILED'}")
