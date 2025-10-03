"""
Quick test script to verify condition_category extraction works.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from back_end.src.data_collection.database_manager import database_manager
from back_end.src.llm_processing.single_model_analyzer import SingleModelAnalyzer
from back_end.src.data.repositories import repository_manager

def test_condition_category_extraction():
    """Test that condition_category is being extracted and saved."""

    print("Testing condition_category extraction...")
    print("=" * 60)

    # Initialize analyzer
    analyzer = SingleModelAnalyzer()

    # Get 3 unprocessed papers
    papers = analyzer.get_unprocessed_papers(limit=3)

    if not papers:
        print("No unprocessed papers found. Fetching any 3 papers...")
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM papers LIMIT 3')
            columns = [desc[0] for desc in cursor.description]
            papers = [dict(zip(columns, row)) for row in cursor.fetchall()]

    print(f"\nProcessing {len(papers)} papers...\n")

    # Process papers
    results = analyzer.process_papers_batch(papers, save_to_db=True, batch_size=3)

    print(f"\nProcessing complete!")
    print(f"Total papers processed: {results['summary']['total_papers']}")
    print(f"Total interventions extracted: {results['summary']['total_interventions']}")
    print(f"Success rate: {results['summary']['success_rate']:.1%}")

    # Check if condition_category was extracted
    print("\n" + "=" * 60)
    print("Checking for condition_category in extracted interventions...")
    print("=" * 60)

    with database_manager.get_connection() as conn:
        cursor = conn.cursor()

        # Get recently added interventions
        cursor.execute('''
            SELECT
                intervention_name,
                health_condition,
                condition_category,
                intervention_category,
                paper_id
            FROM interventions
            WHERE extraction_model = 'qwen2.5:14b'
            ORDER BY extraction_timestamp DESC
            LIMIT 10
        ''')

        interventions = cursor.fetchall()

        if interventions:
            print(f"\nFound {len(interventions)} recent interventions:")
            print("\n")

            with_category = 0
            without_category = 0

            for i, row in enumerate(interventions, 1):
                intervention_name, health_condition, condition_category, intervention_category, paper_id = row

                print(f"{i}. Paper {paper_id}")
                print(f"   Intervention: {intervention_name} ({intervention_category})")
                print(f"   Condition: {health_condition}")
                print(f"   Condition Category: {condition_category or 'MISSING'}")

                if condition_category:
                    with_category += 1
                    print(f"   ✓ Condition category present")
                else:
                    without_category += 1
                    print(f"   ✗ Condition category missing")
                print()

            print("=" * 60)
            print(f"Summary:")
            print(f"  With condition_category: {with_category}")
            print(f"  Without condition_category: {without_category}")
            print(f"  Success rate: {with_category}/{len(interventions)} = {with_category/len(interventions):.1%}")

            if with_category == len(interventions):
                print("\n✓ SUCCESS: All interventions have condition_category!")
                return True
            elif with_category > 0:
                print("\n⚠ PARTIAL: Some interventions have condition_category")
                return False
            else:
                print("\n✗ FAILURE: No interventions have condition_category")
                return False
        else:
            print("No interventions found in database")
            return False

if __name__ == "__main__":
    success = test_condition_category_extraction()
    sys.exit(0 if success else 1)
