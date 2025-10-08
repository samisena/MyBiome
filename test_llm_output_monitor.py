"""
Test script to process a few papers and monitor LLM output for <think> tags.
"""
import sys
import time
from pathlib import Path

# Add back_end to path
sys.path.append(str(Path(__file__).parent))

from back_end.src.data.config import config, setup_logging
from back_end.src.llm_processing.single_model_analyzer import SingleModelAnalyzer
from back_end.src.data.repositories import repository_manager

logger = setup_logging(__name__, 'test_monitor.log')


def test_processing_with_monitoring(num_papers=3):
    """Process a few papers and monitor the output."""

    print(f"\n{'='*60}")
    print(f"Testing LLM Processing with Output Monitoring")
    print(f"{'='*60}\n")

    # Get unprocessed papers
    papers = repository_manager.papers.get_unprocessed_papers(extraction_model='qwen3:14b', limit=num_papers)

    if not papers:
        print("No unprocessed papers found. Looking for any papers...")
        # Get any papers
        conn = repository_manager.papers.conn
        cursor = conn.cursor()
        cursor.execute("SELECT pmid, title FROM papers LIMIT ?", (num_papers,))
        papers = [{"pmid": row[0], "title": row[1]} for row in cursor.fetchall()]

    print(f"Found {len(papers)} papers to test\n")

    if not papers:
        print("ERROR: No papers in database!")
        return

    # Initialize analyzer
    analyzer = SingleModelAnalyzer()

    # Patch the analyzer to capture raw LLM output
    original_generate = analyzer.model_config['client'].generate

    captured_responses = []

    def monitored_generate(*args, **kwargs):
        """Wrapper to capture and log LLM responses."""
        response = original_generate(*args, **kwargs)
        content = response.get('content', '')
        captured_responses.append(content)

        # Log response details
        has_think = '<think>' in content
        think_count = content.count('<think>')
        content_length = len(content)

        print(f"\n{'='*60}")
        print(f"LLM Response Captured:")
        print(f"  Length: {content_length} chars")
        print(f"  Has <think> tags: {has_think}")
        if has_think:
            print(f"  Number of <think> tags: {think_count}")
            # Show first think tag
            start = content.find('<think>')
            end = content.find('</think>', start) + 8
            print(f"\n  First <think> block preview:")
            print(f"  {content[start:end][:200]}...")
        print(f"{'='*60}\n")

        return response

    # Apply monitoring wrapper
    analyzer.model_config['client'].generate = monitored_generate

    # Process papers
    total_time = 0
    results = []

    for i, paper in enumerate(papers, 1):
        pmid = paper['pmid']
        print(f"\n{'='*60}")
        print(f"Processing Paper {i}/{len(papers)}: PMID {pmid}")
        print(f"Title: {paper.get('title', 'N/A')[:80]}...")
        print(f"{'='*60}")

        start = time.time()
        result = analyzer.extract_interventions(paper)
        elapsed = time.time() - start
        total_time += elapsed

        results.append(result)

        print(f"\n  ✓ Completed in {elapsed:.1f}s")
        print(f"  Interventions extracted: {result.get('total_interventions', 0)}")
        if result.get('error'):
            print(f"  ERROR: {result['error']}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Test Complete!")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  Papers processed: {len(papers)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average time per paper: {total_time/len(papers):.1f}s")
    print(f"  Total interventions: {sum(r.get('total_interventions', 0) for r in results)}")

    # Analyze captured responses
    print(f"\nLLM Response Analysis:")
    print(f"  Responses captured: {len(captured_responses)}")
    responses_with_think = sum(1 for r in captured_responses if '<think>' in r)
    print(f"  Responses with <think> tags: {responses_with_think}/{len(captured_responses)}")

    if responses_with_think > 0:
        print(f"\n  WARNING: Model is generating <think> tags despite instructions!")
        print(f"  System message should suppress this.")
    else:
        print(f"\n  ✓ Good! No <think> tags found in responses.")

    avg_length = sum(len(r) for r in captured_responses) / len(captured_responses) if captured_responses else 0
    print(f"  Average response length: {avg_length:.0f} chars")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    try:
        test_processing_with_monitoring(num_papers=3)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
