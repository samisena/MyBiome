"""Test the unprocessed papers query."""
import sqlite3

conn = sqlite3.connect('back_end/data/processed/intervention_research.db')
cursor = conn.cursor()

# Test the exact query used by dual_model_analyzer
model_names = ['gemma2:9b', 'qwen2.5:14b']
placeholders = ','.join(['?' for _ in model_names])

query = f'''
    SELECT DISTINCT p.*
    FROM papers p
    WHERE p.abstract IS NOT NULL
      AND p.abstract != ''
      AND (p.processing_status IS NULL OR p.processing_status != 'failed')
      AND p.pmid NOT IN (
          SELECT DISTINCT paper_id
          FROM interventions
          WHERE extraction_model IN ({placeholders})
      )
    ORDER BY
        COALESCE(p.influence_score, 0) DESC,
        COALESCE(p.citation_count, 0) DESC,
        p.publication_date DESC
'''

cursor.execute(query, model_names)
columns = [desc[0] for desc in cursor.description]
papers = [dict(zip(columns, row)) for row in cursor.fetchall()]

print(f"Unprocessed papers found: {len(papers)}")
print()

for paper in papers[:5]:
    print(f"  {paper['pmid']}: {paper['title'][:50]}...")

if not papers:
    print("No unprocessed papers - checking why:")
    print()

    # Check total papers
    cursor.execute("SELECT COUNT(*) FROM papers WHERE pmid NOT LIKE 'TEST_%'")
    total = cursor.fetchone()[0]
    print(f"  Total real papers: {total}")

    # Check papers with abstracts
    cursor.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND abstract != '' AND pmid NOT LIKE 'TEST_%'")
    with_abstract = cursor.fetchone()[0]
    print(f"  Papers with abstracts: {with_abstract}")

    # Check interventions
    cursor.execute("SELECT COUNT(DISTINCT paper_id) FROM interventions WHERE extraction_model IN ('gemma2:9b', 'qwen2.5:14b')")
    already_processed = cursor.fetchone()[0]
    print(f"  Papers already processed by models: {already_processed}")

    # List interventions
    cursor.execute("SELECT paper_id, extraction_model, COUNT(*) FROM interventions GROUP BY paper_id, extraction_model")
    print(f"\\n  Interventions in database:")
    for row in cursor.fetchall():
        print(f"    Paper {row[0]} by {row[1]}: {row[2]} interventions")

conn.close()
