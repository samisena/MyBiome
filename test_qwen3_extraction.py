#!/usr/bin/env python3
"""Test script to verify qwen3:14b extraction on a small number of papers."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from back_end.src.data_collection.database_manager import DatabaseManager
from back_end.src.llm_processing.single_model_analyzer import SingleModelAnalyzer

def test_qwen3_extraction():
    """Test qwen3:14b extraction on recent diabetes papers."""

    db = DatabaseManager()

    # Get 3 most recent diabetes papers
    papers = db.get_papers_by_condition("diabetes", limit=3)

    if not papers:
        print("ERROR: No diabetes papers found in database")
        return

    print(f"Testing qwen3:14b extraction on {len(papers)} papers...")
    print(f"Paper type: {type(papers[0]) if papers else 'N/A'}")
    if papers and hasattr(papers[0], 'keys'):
        print(f"Paper keys: {list(papers[0].keys())[:5]}...")
    print("="*60)

    analyzer = SingleModelAnalyzer()

    for i, paper in enumerate(papers, 1):
        paper_id = paper.get('id') or paper.get('pmid')
        title = paper['title']

        print(f"\nPaper {i}/{len(papers)}: PMID={paper_id}")
        print(f"Title: {title[:80]}...")

        try:
            # Extract interventions - pass paper dict directly
            result = analyzer.extract_interventions(paper)

            interventions = result.get('interventions', [])
            print(f"[OK] Extracted {len(interventions)} interventions")

            # Show ALL intervention details with full data
            for idx, interv in enumerate(interventions, 1):
                print(f"\n  [{idx}] Intervention: {interv.get('intervention_name', 'N/A')}")
                print(f"      Full intervention data:")
                for key, val in interv.items():
                    if val and key not in ['paper_id', 'extraction_model']:  # Skip IDs
                        print(f"        {key}: {val}")

        except Exception as e:
            print(f"[ERROR] {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Test completed!")

if __name__ == "__main__":
    test_qwen3_extraction()
