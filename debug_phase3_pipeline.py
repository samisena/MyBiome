#!/usr/bin/env python3
"""
Diagnostic to test Phase 3 pipeline execution with 3 test interventions.
This will show exactly where the pipeline is spending time.
"""

import sys
import time
import sqlite3
from pathlib import Path

# Add back_end to path
sys.path.insert(0, str(Path(__file__).parent))

from back_end.src.phase_3_semantic_normalization.phase_3_orchestrator import UnifiedPhase3Orchestrator

print("="*80)
print("PHASE 3 PIPELINE EXECUTION TEST")
print("="*80)

# Setup paths
db_path = Path(__file__).parent / 'back_end' / 'data' / 'processed' / 'intervention_research.db'
config_path = Path(__file__).parent / 'back_end' / 'src' / 'phase_3_semantic_normalization' / 'phase_3_config.yaml'

# Step 1: Insert test data
print("\n[Step 1] Inserting 3 test interventions...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Clean up any existing test data first
cursor.execute("DELETE FROM papers WHERE pmid = 'TEST_DEBUG_99999'")
cursor.execute("DELETE FROM semantic_hierarchy WHERE entity_name IN ('test_aspirin', 'test_ibuprofen', 'test_acetaminophen')")
conn.commit()

# Insert test paper
cursor.execute("""
    INSERT INTO papers (pmid, title, abstract, publication_date, processing_status, llm_processed)
    VALUES (?, ?, ?, ?, ?, ?)
""", ('TEST_DEBUG_99999', 'Test Paper', 'Test abstract', '2025-01-01', 'processed', 1))

test_pmid = 'TEST_DEBUG_99999'

# Insert test interventions
test_interventions = [
    ('test_aspirin', 'headache', 'reduces inflammation'),
    ('test_ibuprofen', 'arthritis', 'reduces inflammation'),
    ('test_acetaminophen', 'pain', 'inhibits prostaglandin synthesis')
]

for name, condition, mechanism in test_interventions:
    cursor.execute("""
        INSERT INTO interventions (
            paper_id, intervention_name, health_condition, mechanism,
            outcome_type, intervention_details, extraction_model
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (test_pmid, name, condition, mechanism, 'improves', '100mg daily', 'test'))
    print(f"  Inserted: {name}")

conn.commit()
conn.close()

# Step 2: Initialize orchestrator
print("\n[Step 2] Initializing orchestrator...")
start = time.time()
orchestrator = UnifiedPhase3Orchestrator(
    db_path=str(db_path),
    config_path=str(config_path)
)
print(f"  Initialized in {time.time()-start:.2f}s")

# Step 3: Run pipeline with detailed timing
print("\n[Step 3] Running Phase 3 pipeline...")
print("  This will process interventions only (not all entities)")
print("  Watching for slowdowns...")

overall_start = time.time()

try:
    print("\n  [3.1] Starting run_pipeline()...")
    phase_start = time.time()

    results = orchestrator.run_pipeline(
        entity_type='intervention',
        force_reembed=False,
        force_recluster=False
    )

    phase_duration = time.time() - phase_start

    print(f"\n  [3.2] Pipeline completed in {phase_duration:.1f}s")

    # Show breakdown
    print(f"\n  Phase Breakdown:")
    print(f"    Phase 3a (Embedding):  {results.embedding_duration_seconds:.1f}s")
    print(f"    Phase 3b (Clustering): {results.clustering_duration_seconds:.1f}s")
    print(f"    Phase 3c (Naming):     {results.naming_duration_seconds:.1f}s")

    print(f"\n  Results:")
    print(f"    Embeddings generated: {results.embeddings_generated}")
    print(f"    Clusters created: {results.num_clusters}")
    print(f"    Names generated: {results.names_generated}")

    total_duration = time.time() - overall_start
    print(f"\n[SUCCESS] Total test duration: {total_duration:.1f}s")

except Exception as e:
    print(f"\n[FAIL] Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Cleanup
    print("\n[Step 4] Cleaning up test data...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM papers WHERE pmid = 'TEST_DEBUG_99999'")
    cursor.execute("DELETE FROM semantic_hierarchy WHERE entity_name IN ('test_aspirin', 'test_ibuprofen', 'test_acetaminophen')")
    conn.commit()
    conn.close()
    print("  Cleanup complete")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
