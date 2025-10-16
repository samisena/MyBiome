"""Query confidence levels from the actual database."""
import sqlite3
from pathlib import Path

# Try to find the actual database
possible_paths = [
    Path(r"C:\Users\samis\Desktop\MyBiome\back_end\data\processed\intervention_research.db"),
    Path(r"C:\Users\samis\Desktop\MyBiome\back_end\src\data\intervention_research.db"),
    Path(r"C:\Users\samis\Desktop\MyBiome\back_end\data\intervention_research.db"),
]

db_path = None
for path in possible_paths:
    if path.exists():
        db_path = path
        print(f"Found database: {path}")
        break

if not db_path:
    print("ERROR: Could not find database file")
    exit(1)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Check what tables exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print(f"\nTables in database: {tables}")

# If interventions table exists, check its schema
if 'interventions' in tables:
    cursor.execute("PRAGMA table_info(interventions)")
    columns = cursor.fetchall()
    print(f"\nColumns in 'interventions' table:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

    # Get count
    cursor.execute("SELECT COUNT(*) FROM interventions")
    count = cursor.fetchone()[0]
    print(f"\nTotal interventions: {count}")

    # Try to query confidence levels
    try:
        cursor.execute("""
            SELECT extraction_confidence, COUNT(*)
            FROM interventions
            WHERE extraction_confidence IS NOT NULL
            GROUP BY extraction_confidence
        """)
        confidence_counts = cursor.fetchall()
        print(f"\nConfidence level distribution:")
        for row in confidence_counts:
            print(f"  {row[0]}: {row[1]}")
    except sqlite3.OperationalError as e:
        print(f"Error querying confidence: {e}")

# Check papers table
if 'papers' in tables:
    cursor.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    print(f"\nTotal papers: {count}")

conn.close()