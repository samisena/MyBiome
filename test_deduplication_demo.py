#!/usr/bin/env python3
"""
Demonstration of the dual LLM approach and deduplication.
Collects a few papers, processes them with both models, then shows deduplication.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.data_collection.pubmed_collector import PubMedCollector
from back_end.src.data_collection.database_manager import database_manager
from back_end.src.llm_processing.dual_model_analyzer import DualModelAnalyzer
from back_end.src.llm_processing.batch_entity_processor import create_batch_processor

def main():
    print("=" * 60)
    print("DUAL LLM AND DEDUPLICATION DEMONSTRATION")
    print("=" * 60)

    # Step 1: Collect a few papers
    print("\n1. COLLECTING PAPERS...")
    collector = PubMedCollector()

    # Search for diabetes papers
    condition = "type 2 diabetes"
    pmids = collector.search_papers(f'{condition} intervention', min_year=2023, max_results=2)
    print(f"Found {len(pmids)} papers: {pmids}")

    if pmids:
        # Fetch metadata
        metadata_file = collector.fetch_papers_metadata(pmids)
        if metadata_file:
            # Parse and save papers (parse_metadata_file automatically saves to database)
            papers = collector.parser.parse_metadata_file(str(metadata_file))
            print(f"Parsed and saved {len(papers)} papers to database")

    # Step 2: Process with dual LLM
    print("\n2. PROCESSING WITH DUAL LLM (gemma2:9b and qwen2.5:14b)...")
    analyzer = DualModelAnalyzer()

    # Get unprocessed papers
    unprocessed = analyzer.get_unprocessed_papers()
    print(f"Found {len(unprocessed)} unprocessed papers")

    if unprocessed:
        # Process papers with both models
        results = analyzer.process_papers_batch(unprocessed[:2], save_to_db=True, batch_size=1)
        print(f"\nProcessing Results:")
        print(f"  Papers processed: {results['successful_papers']}")
        print(f"  Total interventions extracted: {results['total_interventions']}")
        print(f"  Model statistics: {results['model_statistics']}")

    # Step 3: Check interventions BEFORE deduplication
    print("\n3. CHECKING INTERVENTIONS BEFORE DEDUPLICATION...")
    with database_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, intervention_name, health_condition, extraction_model, paper_id
            FROM interventions
            ORDER BY paper_id, intervention_name
        """)
        interventions_before = cursor.fetchall()

        print(f"Total interventions before deduplication: {len(interventions_before)}")
        for i, (id, name, condition, model, paper_id) in enumerate(interventions_before, 1):
            print(f"  {i}. [{model}] {name} (for {condition}) - Paper: {paper_id}")

    # Step 4: Run deduplication
    print("\n4. RUNNING DEDUPLICATION...")
    processor = create_batch_processor()

    if hasattr(processor, 'batch_deduplicate_entities'):
        dedup_result = processor.batch_deduplicate_entities()
        print(f"\nDeduplication Results:")
        print(f"  Total interventions processed: {dedup_result.get('interventions_processed', 0)}")
        print(f"  Duplicates merged: {dedup_result.get('total_merged', 0)}")
        print(f"  Duplicate groups found: {dedup_result.get('duplicate_groups_found', 0)}")
        print(f"  Papers processed: {dedup_result.get('papers_processed', 0)}")
    else:
        print("Batch deduplication method not available")

    # Step 5: Check interventions AFTER deduplication
    print("\n5. CHECKING INTERVENTIONS AFTER DEDUPLICATION...")
    with database_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, intervention_name, health_condition, extraction_model, paper_id,
                   intervention_canonical_id, normalized
            FROM interventions
            WHERE id NOT IN (
                SELECT duplicate_id FROM intervention_duplicates WHERE duplicate_id IS NOT NULL
            )
            ORDER BY paper_id, intervention_name
        """)
        interventions_after = cursor.fetchall()

        print(f"Total interventions after deduplication: {len(interventions_after)}")
        for i, row in enumerate(interventions_after, 1):
            print(f"  {i}. [{row[3]}] {row[1]} (for {row[2]}) - Paper: {row[4]}")
            if row[5]:  # canonical_id
                print(f"     -> Canonical ID: {row[5]}, Normalized: {row[6]}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {len(interventions_before)} interventions → {len(interventions_after)} after deduplication")
    print(f"Duplicates removed: {len(interventions_before) - len(interventions_after)}")
    print("=" * 60)

if __name__ == "__main__":
    main()