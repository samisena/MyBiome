import sqlite3

conn = sqlite3.connect('back_end/data/processed/intervention_research.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM papers')
count = cursor.fetchone()[0]
print(f'Papers in database: {count}')

if count > 0:
    cursor.execute('SELECT pmid, SUBSTR(title,1,50) FROM papers ORDER BY created_at DESC LIMIT 10')
    papers = cursor.fetchall()
    print('\nLatest papers:')
    for p in papers:
        print(f'  {p[0]}: {p[1]}...')

conn.close()
