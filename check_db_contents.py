#!/usr/bin/env python3
"""Check database contents."""

import sqlite3

conn = sqlite3.connect('back_end/data/processed/intervention_research.db')
cursor = conn.cursor()

# Check papers
cursor.execute('SELECT COUNT(*) FROM papers')
print(f'Total papers: {cursor.fetchone()[0]}')

cursor.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND abstract != ''")
print(f'Papers with abstracts: {cursor.fetchone()[0]}')

cursor.execute("SELECT COUNT(*) FROM papers WHERE processing_status = 'pending'")
print(f'Papers pending: {cursor.fetchone()[0]}')

cursor.execute("SELECT COUNT(*) FROM papers WHERE processing_status = 'processed'")
print(f'Papers processed: {cursor.fetchone()[0]}')

# Check interventions
cursor.execute('SELECT COUNT(*) FROM interventions')
print(f'Total interventions: {cursor.fetchone()[0]}')

# Show some papers
cursor.execute("SELECT pmid, title, processing_status FROM papers LIMIT 5")
papers = cursor.fetchall()
print("\nSample papers:")
for pmid, title, status in papers:
    print(f"  {pmid}: {title[:50]}... [{status}]")

conn.close()