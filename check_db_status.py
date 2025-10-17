#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('back_end/data/processed/intervention_research.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM interventions')
total_interventions = cursor.fetchone()[0]
print(f'Total interventions: {total_interventions}')

cursor.execute('SELECT COUNT(*) FROM papers WHERE processing_status = "processed"')
processed = cursor.fetchone()[0]
print(f'Processed papers: {processed}')

cursor.execute('SELECT COUNT(*) FROM papers WHERE processing_status = "failed"')
failed = cursor.fetchone()[0]
print(f'Failed papers: {failed}')

cursor.execute('SELECT COUNT(*) FROM papers WHERE processing_status IS NULL OR processing_status = "pending"')
pending = cursor.fetchone()[0]
print(f'Pending papers: {pending}')

conn.close()
