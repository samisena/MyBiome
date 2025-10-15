"""
Quick import test for new Phase 3a/3b/3c files.

Tests that all imports work correctly after migration.
Run from MyBiome directory: python -m back_end.src.phase_3_semantic_normalization.test_imports
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all new Phase 3 files can be imported."""

    print("Testing Phase 3a imports (Embedders)...")
    try:
        from back_end.src.phase_3_semantic_normalization.phase_3a_base_embedder import BaseEmbedder
        from back_end.src.phase_3_semantic_normalization.phase_3a_intervention_embedder import InterventionEmbedder
        from back_end.src.phase_3_semantic_normalization.phase_3a_condition_embedder import ConditionEmbedder
        from back_end.src.phase_3_semantic_normalization.phase_3a_mechanism_embedder import MechanismEmbedder
        print("[PASS] Phase 3a imports successful")
    except Exception as e:
        print(f"[FAIL] Phase 3a imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nTesting Phase 3b imports (Clusterers)...")
    try:
        from back_end.src.phase_3_semantic_normalization.phase_3b_base_clusterer import BaseClusterer
        from back_end.src.phase_3_semantic_normalization.phase_3b_hierarchical_clusterer import HierarchicalClusterer
        from back_end.src.phase_3_semantic_normalization.phase_3b_hdbscan_clusterer import HDBSCANClusterer
        from back_end.src.phase_3_semantic_normalization.phase_3b_singleton_handler import SingletonHandler
        print("[PASS] Phase 3b imports successful")
    except Exception as e:
        print(f"[FAIL] Phase 3b imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nTesting Phase 3c imports (Namers)...")
    try:
        from back_end.src.phase_3_semantic_normalization.phase_3c_base_namer import BaseNamer, ClusterData, NamingResult
        from back_end.src.phase_3_semantic_normalization.phase_3c_llm_namer import LLMNamer
        print("[PASS] Phase 3c imports successful")
    except Exception as e:
        print(f"[FAIL] Phase 3c imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nTesting Orchestrator import...")
    try:
        from back_end.src.phase_3_semantic_normalization.phase_3abc_orchestrator import UnifiedPhase3Orchestrator, EntityResults
        print("[PASS] Orchestrator import successful")
    except Exception as e:
        print(f"[FAIL] Orchestrator import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("ALL IMPORT TESTS PASSED")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
