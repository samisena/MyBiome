#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('back_end/data/processed/intervention_research.db')
cursor = conn.cursor()

# Get a few failed papers to understand why
cursor.execute('''
    SELECT pmid, title, abstract
    FROM papers
    WHERE processing_status = "failed"
    LIMIT 3
''')

print("=" * 60)
print("FAILED PAPERS SAMPLE")
print("=" * 60)

for row in cursor.fetchall():
    pmid, title, abstract = row
    print(f"\nPMID: {pmid}")
    print(f"Title: {title[:100]}...")
    print(f"Abstract length: {len(abstract) if abstract else 0} chars")
    print(f"Has abstract: {'YES' if abstract and len(abstract) > 100 else 'NO'}")
    print("-" * 60)

conn.close()
