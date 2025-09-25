#!/usr/bin/env python3
"""
Debug version to identify where the migration is hanging.
"""

import sys
import sqlite3
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

print("DEBUG: Starting debug migration")

# Test database connection
print("DEBUG: Testing database connection...")
db_path = "data/processed/intervention_research.db"
try:
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM interventions WHERE (canonical_name IS NULL OR canonical_name = '')")
    result = cursor.fetchone()[0]
    conn.close()
    print(f"DEBUG: Found {result} interventions needing semantic processing")
except Exception as e:
    print(f"DEBUG: Database error: {e}")
    sys.exit(1)

# Test semantic merger import
print("DEBUG: Testing semantic merger import...")
try:
    from llm.semantic_merger import SemanticMerger, InterventionExtraction
    print("DEBUG: Successfully imported SemanticMerger")
except Exception as e:
    print(f"DEBUG: Import error: {e}")
    sys.exit(1)

# Test semantic merger initialization
print("DEBUG: Testing semantic merger initialization...")
try:
    merger = SemanticMerger(
        primary_model="qwen2.5:14b",
        validator_model="gemma2:9b"
    )
    print("DEBUG: Successfully initialized SemanticMerger")
except Exception as e:
    print(f"DEBUG: Initialization error: {e}")
    sys.exit(1)

# Test getting interventions
print("DEBUG: Testing intervention retrieval...")
try:
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, intervention_name, health_condition
        FROM interventions
        WHERE (canonical_name IS NULL OR canonical_name = '')
        ORDER BY health_condition
        LIMIT 5
    """)
    sample_interventions = [dict(row) for row in cursor.fetchall()]
    conn.close()

    print(f"DEBUG: Retrieved {len(sample_interventions)} sample interventions:")
    for interv in sample_interventions:
        print(f"  - {interv['intervention_name']} ({interv['health_condition']})")

except Exception as e:
    print(f"DEBUG: Retrieval error: {e}")
    sys.exit(1)

print("DEBUG: All systems operational - ready for full migration")