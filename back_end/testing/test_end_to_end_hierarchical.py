"""
End-to-end test for hierarchical prompt through full pipeline.
Tests: LLM extraction → flattening → validation → database insertion.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from back_end.src.data.config import setup_logging
from back_end.src.phase_1_data_collection.database_manager import database_manager
from back_end.src.phase_2_llm_processing.phase_2_single_model_analyzer import SingleModelAnalyzer

logger = setup_logging(__name__, 'test_end_to_end.log')


def main():
    """Run end-to-end test on 2 papers."""
    print("\n" + "="*80)
    print("END-TO-END HIERARCHICAL PROMPT TEST")
    print("="*80 + "\n")

    # Get 2 papers for processing
    print("Fetching 2 papers for processing from database...")
    papers = database_manager.get_papers_for_processing(extraction_model='qwen3:14b', limit=2)

    if len(papers) < 2:
        print(f"[WARNING] Only found {len(papers)} unprocessed papers")
        if len(papers) == 0:
            print("[ERROR] No unprocessed papers available. Test cannot proceed.")
            return

    print(f"Found {len(papers)} paper(s) to test\n")

    # Initialize analyzer
    print("Initializing single-model analyzer (qwen3:14b)...")
    analyzer = SingleModelAnalyzer()
    print("Analyzer ready\n")

    # Process each paper
    results = []
    for i, paper in enumerate(papers, 1):
        print(f"\n{'='*80}")
        print(f"PAPER {i}/{len(papers)}")
        print(f"{'='*80}")
        print(f"PMID: {paper['pmid']}")
        print(f"Title: {paper['title'][:80]}...")
        print()

        # Extract interventions
        print("Step 1: LLM extraction (hierarchical format)...")
        result = analyzer.analyze_paper(paper)

        if result.get('error'):
            print(f"[ERROR] Extraction failed: {result['error']}")
            results.append({'pmid': paper['pmid'], 'success': False, 'error': result['error']})
            continue

        interventions = result.get('interventions', [])
        print(f"[OK] Extracted {len(interventions)} intervention(s)")

        # Check for new fields
        if interventions:
            sample = interventions[0]
            new_fields = {
                'study_focus': sample.get('study_focus'),
                'measured_metrics': sample.get('measured_metrics'),
                'findings': sample.get('findings'),
                'study_location': sample.get('study_location'),
                'publisher': sample.get('publisher')
            }

            print("\nStep 2: Verify new fields present...")
            for field, value in new_fields.items():
                if value is not None:
                    display_val = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)
                    print(f"  {field}: {display_val}")
                else:
                    print(f"  {field}: [NULL]")

        # Store interventions
        print("\nStep 3: Database insertion...")
        inserted = 0
        failed = 0

        for intervention in interventions:
            success = database_manager.insert_intervention(intervention)
            if success:
                inserted += 1
            else:
                failed += 1

        print(f"[OK] Inserted: {inserted}, Failed: {failed}")

        # Verify in database
        print("\nStep 4: Verification...")
        db_interventions = database_manager.get_interventions_for_paper(paper['pmid'])
        print(f"[OK] Found {len(db_interventions)} intervention(s) in database")

        # Check if new fields were saved
        if db_interventions:
            sample_db = db_interventions[0]
            has_new_fields = any([
                sample_db.get('study_focus'),
                sample_db.get('measured_metrics'),
                sample_db.get('findings'),
                sample_db.get('study_location'),
                sample_db.get('publisher')
            ])

            if has_new_fields:
                print("[SUCCESS] New hierarchical fields saved to database!")
            else:
                print("[WARNING] New fields not found in database - check serialization")

        results.append({
            'pmid': paper['pmid'],
            'success': True,
            'extracted': len(interventions),
            'inserted': inserted
        })

    # Summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    successful = sum(1 for r in results if r['success'])
    total_extracted = sum(r.get('extracted', 0) for r in results if r['success'])
    total_inserted = sum(r.get('inserted', 0) for r in results if r['success'])

    print(f"Papers processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Total interventions extracted: {total_extracted}")
    print(f"Total interventions inserted: {total_inserted}")
    print()

    if successful == len(results) and total_inserted > 0:
        print("[SUCCESS] End-to-end test PASSED!")
        print("New hierarchical prompt is working correctly through full pipeline.")
    else:
        print("[FAIL] End-to-end test encountered issues")
        print("Check logs for details.")


if __name__ == '__main__':
    main()
