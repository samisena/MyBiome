#!/usr/bin/env python3
"""
Simple diagnostic to test Phase 3 orchestrator initialization.
"""

import sys
import time
from pathlib import Path

# Add back_end to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("PHASE 3 ORCHESTRATOR INITIALIZATION TEST")
print("="*80)

# Step 1: Import test
print("\n[Step 1] Importing UnifiedPhase3Orchestrator...")
try:
    from back_end.src.phase_3_semantic_normalization.phase_3_orchestrator import UnifiedPhase3Orchestrator
    print("[OK] Import successful")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Step 2: Path setup
print("\n[Step 2] Setting up paths...")
db_path = Path(__file__).parent / 'back_end' / 'data' / 'processed' / 'intervention_research.db'
config_path = Path(__file__).parent / 'back_end' / 'src' / 'phase_3_semantic_normalization' / 'phase_3_config.yaml'

if not db_path.exists():
    print(f"[FAIL] Database not found: {db_path}")
    sys.exit(1)
print(f"[OK] Database found: {db_path}")

if not config_path.exists():
    print(f"[FAIL] Config not found: {config_path}")
    sys.exit(1)
print(f"[OK] Config found: {config_path}")

# Step 3: Initialization test
print("\n[Step 3] Initializing orchestrator...")
print("This should take <2 seconds if working correctly...")
start_time = time.time()

try:
    orchestrator = UnifiedPhase3Orchestrator(
        db_path=str(db_path),
        config_path=str(config_path)
    )
    duration = time.time() - start_time
    print(f"[OK] Orchestrator initialized in {duration:.2f}s")

    # Step 4: Method check
    print("\n[Step 4] Verifying run_pipeline() method exists...")
    if hasattr(orchestrator, 'run_pipeline'):
        print("[OK] run_pipeline() method found")
    else:
        print("[FAIL] run_pipeline() method NOT FOUND")
        sys.exit(1)

    print("\n" + "="*80)
    print("SUCCESS: All diagnostic checks passed")
    print("="*80)

except Exception as e:
    duration = time.time() - start_time
    print(f"[FAIL] Initialization failed after {duration:.2f}s")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
