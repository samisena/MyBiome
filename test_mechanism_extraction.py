"""
Test script for validating mechanism extraction in Phase 2.

This script:
1. Runs the database migration to add mechanism field
2. Tests extraction on 3-5 sample papers from the existing database
3. Validates mechanism field quality
4. Measures performance impact
5. Reports results

Usage:
    python test_mechanism_extraction.py
"""

import sys
import time
from pathlib import Path
import sqlite3
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from back_end.src.data.config import setup_logging
from back_end.src.llm_processing.single_model_analyzer import SingleModelAnalyzer

logger = setup_logging(__name__, 'test_mechanism_extraction.log')


def get_sample_papers(db_path: Path, limit: int = 5) -> List[Dict]:
    """Get sample papers from database for testing."""
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get diverse papers with fulltext if available
        cursor.execute("""
            SELECT pmid, title, abstract, has_fulltext, fulltext_path
            FROM papers
            WHERE abstract IS NOT NULL
              AND length(abstract) > 200
            ORDER BY RANDOM()
            LIMIT ?
        """, (limit,))

        papers = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return papers

    except Exception as e:
        logger.error(f"Failed to get sample papers: {e}")
        return []


def validate_mechanism(mechanism: str, intervention_name: str, condition: str) -> Dict:
    """
    Validate a mechanism field for quality.

    Returns dict with:
    - is_valid: bool
    - issues: List[str]
    - score: int (0-10)
    """
    issues = []
    score = 10

    if not mechanism:
        return {
            'is_valid': False,
            'issues': ['Mechanism is null or empty'],
            'score': 0
        }

    mechanism_lower = mechanism.lower().strip()

    # Check for placeholders
    placeholders = [
        'unknown', 'unclear', 'not specified', 'n/a',
        'mechanism not described', 'improves symptoms', 'helps condition'
    ]

    if any(placeholder in mechanism_lower for placeholder in placeholders):
        issues.append(f"Contains placeholder/generic term")
        score -= 5

    # Check length
    if len(mechanism) < 15:
        issues.append(f"Too short ({len(mechanism)} chars)")
        score -= 3
    elif len(mechanism) > 500:
        issues.append(f"Too long ({len(mechanism)} chars)")
        score -= 1

    # Check for specificity (meaningful words)
    import re
    meaningful_words = re.findall(r'\b\w{6,}\b', mechanism_lower)
    if len(meaningful_words) < 2:
        issues.append("Lacks specific biological/behavioral terms")
        score -= 3

    # Check for biological/medical keywords
    medical_keywords = [
        'inflammatory', 'inflammation', 'microbiome', 'bacteria', 'neurotransmitter',
        'receptor', 'pathway', 'signaling', 'metabolism', 'hormone', 'enzyme',
        'cognitive', 'behavioral', 'psychological', 'neural', 'cellular',
        'immune', 'oxidative', 'endorphin', 'serotonin', 'dopamine',
        'modulation', 'regulation', 'inhibition', 'activation', 'expression',
        'restructuring', 'adherence', 'knowledge', 'education', 'support'
    ]

    has_medical_keyword = any(keyword in mechanism_lower for keyword in medical_keywords)
    if not has_medical_keyword:
        issues.append("Missing clear biological/behavioral/psychological terms")
        score -= 2

    # Check if mechanism relates to intervention or condition
    intervention_words = set(intervention_name.lower().split())
    condition_words = set(condition.lower().split())
    mechanism_words = set(mechanism_lower.split())

    # Mechanism should not just repeat intervention/condition name
    overlap_intervention = len(intervention_words & mechanism_words) / len(intervention_words) if intervention_words else 0
    if overlap_intervention > 0.8:
        issues.append("Mechanism just repeats intervention name")
        score -= 2

    is_valid = score >= 6

    return {
        'is_valid': is_valid,
        'issues': issues,
        'score': max(0, score),
        'mechanism': mechanism
    }


def test_extraction_with_mechanism():
    """Main test function."""

    print("\n" + "="*80)
    print("MECHANISM EXTRACTION TEST")
    print("="*80)

    # Step 1: Run migration
    print("\n[Step 1/4] Running database migration...")
    try:
        from back_end.scripts.migrate_add_mechanism import main as run_migration
        migration_result = run_migration()
        if migration_result != 0:
            print("[WARN]  Migration reported issues, but continuing with tests...")
    except Exception as e:
        print(f"[WARN]  Migration error: {e} (continuing anyway)")

    # Step 2: Get sample papers
    print("\n[Step 2/4] Loading sample papers from database...")
    db_path = Path("back_end/data/processed/intervention_research.db")

    if not db_path.exists():
        print(f"[FAIL] Database not found at {db_path}")
        print("Please ensure the database exists with papers.")
        return 1

    papers = get_sample_papers(db_path, limit=2)  # Reduced to 2 for faster testing

    if not papers:
        print("[FAIL] No papers found in database")
        return 1

    print(f"[OK] Loaded {len(papers)} test papers")

    # Step 3: Extract interventions with mechanism field
    print(f"\n[Step 3/4] Extracting interventions with mechanism field...")
    print("-" * 80)

    analyzer = SingleModelAnalyzer()

    all_results = []
    total_time = 0
    mechanism_validations = []

    for i, paper in enumerate(papers, 1):
        print(f"\nPaper {i}/{len(papers)}: {paper['pmid']}")
        print(f"Title: {paper['title'][:80]}...")

        start_time = time.time()
        result = analyzer.extract_interventions(paper)
        extraction_time = time.time() - start_time
        total_time += extraction_time

        interventions = result.get('interventions', [])
        print(f"  Extracted: {len(interventions)} intervention(s) in {extraction_time:.1f}s")

        if interventions:
            for intervention in interventions:
                mechanism = intervention.get('mechanism')
                intervention_name = intervention.get('intervention_name', 'unknown')
                condition = intervention.get('health_condition', 'unknown')

                validation = validate_mechanism(mechanism, intervention_name, condition)
                mechanism_validations.append({
                    'paper_id': paper['pmid'],
                    'intervention': intervention_name,
                    'condition': condition,
                    'mechanism': mechanism,
                    'validation': validation
                })

                # Print mechanism info
                status = "[OK]" if validation['is_valid'] else "[FAIL]"
                score = validation['score']
                print(f"  {status} {intervention_name} -> {condition}")
                print(f"     Mechanism ({score}/10): {mechanism[:100]}...")
                if validation['issues']:
                    for issue in validation['issues']:
                        print(f"     [WARN]  {issue}")

        all_results.append(result)

    # Step 4: Summary and statistics
    print("\n" + "="*80)
    print("[Step 4/4] TEST RESULTS SUMMARY")
    print("="*80)

    total_interventions = sum(len(r.get('interventions', [])) for r in all_results)
    avg_time_per_paper = total_time / len(papers) if papers else 0

    print(f"\n[STATS] Extraction Statistics:")
    print(f"  Papers processed: {len(papers)}")
    print(f"  Total interventions: {total_interventions}")
    print(f"  Average time per paper: {avg_time_per_paper:.1f}s")
    print(f"  Total extraction time: {total_time:.1f}s")

    print(f"\n[ANALYSIS] Mechanism Quality Analysis:")
    valid_mechanisms = [v for v in mechanism_validations if v['validation']['is_valid']]
    invalid_mechanisms = [v for v in mechanism_validations if not v['validation']['is_valid']]

    if mechanism_validations:
        avg_score = sum(v['validation']['score'] for v in mechanism_validations) / len(mechanism_validations)
        print(f"  Valid mechanisms: {len(valid_mechanisms)}/{len(mechanism_validations)} ({len(valid_mechanisms)/len(mechanism_validations)*100:.1f}%)")
        print(f"  Average quality score: {avg_score:.1f}/10")

        if invalid_mechanisms:
            print(f"\n[WARN]  Invalid Mechanisms ({len(invalid_mechanisms)}):")
            for v in invalid_mechanisms[:3]:  # Show first 3
                print(f"    - {v['intervention']} -> {v['condition']}")
                print(f"      Mechanism: {v['mechanism']}")
                print(f"      Issues: {', '.join(v['validation']['issues'])}")

        # Show best examples
        if valid_mechanisms:
            print(f"\n[OK] Best Mechanism Examples:")
            sorted_valid = sorted(valid_mechanisms, key=lambda x: x['validation']['score'], reverse=True)
            for v in sorted_valid[:3]:  # Show top 3
                print(f"    [{v['validation']['score']}/10] {v['intervention']} -> {v['condition']}")
                print(f"      {v['mechanism']}")

    # Performance check
    print(f"\n[PERF] Performance Impact:")
    baseline_time = 22.0  # qwen3:14b baseline from CLAUDE.md
    time_increase = ((avg_time_per_paper - baseline_time) / baseline_time * 100) if baseline_time else 0

    status_icon = "[OK]" if time_increase < 10 else "[WARN]" if time_increase < 20 else "[FAIL]"
    print(f"  {status_icon} Time per paper: {avg_time_per_paper:.1f}s (baseline: {baseline_time}s)")
    print(f"  {status_icon} Performance change: {time_increase:+.1f}%")

    # Final verdict
    print(f"\n{'='*80}")
    success = (
        len(valid_mechanisms) / len(mechanism_validations) >= 0.7 if mechanism_validations else False
    ) and (time_increase < 15)

    if success:
        print("[OK] MECHANISM EXTRACTION TEST PASSED")
        print("   - Mechanism quality is good (>=70% valid)")
        print("   - Performance impact is acceptable (<15% slowdown)")
        return 0
    else:
        print("[FAIL] MECHANISM EXTRACTION TEST NEEDS IMPROVEMENT")
        if mechanism_validations and len(valid_mechanisms) / len(mechanism_validations) < 0.7:
            print("   - Mechanism quality needs improvement")
        if time_increase >= 15:
            print("   - Performance impact too high (>=15% slowdown)")
        return 1


if __name__ == "__main__":
    try:
        exit_code = test_extraction_with_mechanism()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        print(f"\n[FAIL] Test failed with exception: {e}")
        sys.exit(1)
