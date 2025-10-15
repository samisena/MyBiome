"""
Quick Orchestrator Test

Tests the Phase 3abc orchestrator with a small dataset to verify it works correctly.
This test does NOT run the full pipeline (too slow) - it just verifies:
1. Orchestrator can be instantiated
2. Config file can be loaded
3. Basic functionality works

Run from MyBiome directory:
    python -m back_end.src.phase_3_semantic_normalization.test_orchestrator
"""

import sys
from pathlib import Path

def test_orchestrator():
    """Test orchestrator instantiation and config loading."""

    print("Testing Orchestrator Instantiation...")
    print("="*60)

    try:
        from back_end.src.phase_3_semantic_normalization.phase_3abc_orchestrator import (
            UnifiedPhase3Orchestrator,
            EntityResults
        )
        print("[PASS] Imported UnifiedPhase3Orchestrator")
    except Exception as e:
        print(f"[FAIL] Failed to import: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check config file exists
    config_path = Path(__file__).parent / 'phase_3_config.yaml'
    if not config_path.exists():
        print(f"[FAIL] Config file not found: {config_path}")
        return False
    print(f"[PASS] Config file exists: {config_path}")

    # Check database exists
    db_path = Path(__file__).parent.parent.parent.parent / 'data' / 'processed' / 'intervention_research.db'
    if not db_path.exists():
        print(f"[FAIL] Database not found: {db_path}")
        return False
    print(f"[PASS] Database exists: {db_path}")

    # Try to instantiate orchestrator
    try:
        orchestrator = UnifiedPhase3Orchestrator(
            db_path=str(db_path),
            config_path=str(config_path)
        )
        print("[PASS] Orchestrator instantiated successfully")
    except Exception as e:
        print(f"[FAIL] Failed to instantiate orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check config loaded correctly
    try:
        assert orchestrator.config is not None, "Config is None"
        assert 'embedding' in orchestrator.config, "embedding section missing from config"
        assert 'clustering' in orchestrator.config, "clustering section missing from config"
        assert 'naming' in orchestrator.config, "naming section missing from config"
        print("[PASS] Config loaded correctly")
    except AssertionError as e:
        print(f"[FAIL] Config validation failed: {e}")
        return False

    # Verify cache directories
    try:
        cache_base = Path(orchestrator.config['cache']['base_dir'])
        print(f"[INFO] Cache base directory: {cache_base}")
        if cache_base.exists():
            print("[PASS] Cache directory exists")
        else:
            print("[INFO] Cache directory will be created on first run")
    except Exception as e:
        print(f"[WARN] Could not verify cache directory: {e}")

    # Success
    print("\n" + "="*60)
    print("ALL ORCHESTRATOR TESTS PASSED")
    print("="*60)
    print("\nNote: This test only verifies instantiation and config loading.")
    print("To test the full pipeline, run:")
    print("  python -m back_end.src.orchestration.phase_3abc_semantic_normalizer --entity-type intervention")
    print("\n" + "="*60)

    return True


if __name__ == "__main__":
    success = test_orchestrator()
    sys.exit(0 if success else 1)
