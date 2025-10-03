"""Simple test for condition_category extraction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from back_end.src.llm_processing.single_model_analyzer import SingleModelAnalyzer
from back_end.src.data_collection.database_manager import database_manager

print("=" * 70)
print("Testing condition_category extraction after fix")
print("=" * 70)

# Get papers to process
analyzer = SingleModelAnalyzer()
papers = analyzer.get_unprocessed_papers(limit=3)

if not papers:
    print("\nNo unprocessed papers. Getting any 3 papers...")
    with database_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM papers WHERE pmid NOT IN (SELECT DISTINCT paper_id FROM interventions WHERE extraction_model = "qwen2.5:14b") LIMIT 3')
        columns = [desc[0] for desc in cursor.description]
        papers = [dict(zip(columns, row)) for row in cursor.fetchall()]

print(f"\nProcessing {len(papers)} papers...")
print(f"Papers: {[p['pmid'] for p in papers]}\n")

# Process
results = analyzer.process_papers_batch(papers, save_to_db=True, batch_size=3)

print(f"\n" + "=" * 70)
print("Checking database for condition_category...")
print("=" * 70)

# Check results
with database_manager.get_connection() as conn:
    cursor = conn.cursor()

    # Get latest interventions
    cursor.execute('''
        SELECT
            intervention_name,
            health_condition,
            condition_category,
            paper_id
        FROM interventions
        WHERE extraction_model = 'qwen2.5:14b'
        ORDER BY extraction_timestamp DESC
        LIMIT 10
    ''')

    interventions = cursor.fetchall()

    print(f"\nLast 10 interventions extracted:\n")

    with_cat = 0
    without_cat = 0

    for i, row in enumerate(interventions, 1):
        intervention_name, health_condition, condition_category, paper_id = row

        status = "✓" if condition_category else "✗"
        cat_display = condition_category or "NULL"

        print(f"{i}. {status} [{paper_id}] {intervention_name}")
        print(f"   Condition: {health_condition}")
        print(f"   Category: {cat_display}")
        print()

        if condition_category:
            with_cat += 1
        else:
            without_cat += 1

    # Overall stats
    cursor.execute('SELECT COUNT(*) FROM interventions WHERE condition_category IS NOT NULL')
    total_with_cat = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM interventions')
    total = cursor.fetchone()[0]

    print("=" * 70)
    print(f"OVERALL STATS:")
    print(f"  Total interventions in DB: {total}")
    print(f"  With condition_category: {total_with_cat}")
    print(f"  Without condition_category: {total - total_with_cat}")
    print()
    print(f"RECENT EXTRACTION:")
    print(f"  With condition_category: {with_cat}/{len(interventions)}")
    print(f"  Without condition_category: {without_cat}/{len(interventions)}")
    print("=" * 70)

    if with_cat == len(interventions) and with_cat > 0:
        print("\n✓✓✓ SUCCESS! All recent interventions have condition_category! ✓✓✓")
        sys.exit(0)
    elif with_cat > 0:
        print("\n⚠ PARTIAL: Some interventions have condition_category")
        sys.exit(1)
    else:
        print("\n✗✗✗ FAILURE: No interventions have condition_category ✗✗✗")
        sys.exit(1)
