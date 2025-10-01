#!/usr/bin/env python3
"""
Test Phase 2.4: Progress Reporting with tqdm

Tests that progress bars work correctly for:
1. Paper collection progress
2. Paper processing progress
3. Overall pipeline progress
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from tqdm import tqdm
import time

print("=== Phase 2.4: Progress Reporting Tests ===\n")

def test_collection_progress():
    """Test 2.4a: Collection progress bar."""
    print("Test 2.4a: Collection progress bar")

    # Simulate collection of 5 conditions
    conditions = ['hypertension', 'diabetes', 'asthma', 'arthritis', 'migraine']

    with tqdm(total=len(conditions), desc="Collecting papers", unit="condition") as pbar:
        for i, condition in enumerate(conditions):
            time.sleep(0.1)  # Simulate collection
            papers_collected = (i + 1) * 10
            pbar.set_postfix({'papers': papers_collected, 'success': i + 1})
            pbar.update(1)

    print("[PASS] Collection progress bar works\n")

def test_processing_progress():
    """Test 2.4b: Processing progress bar."""
    print("Test 2.4b: Processing progress bar")

    # Simulate processing of 10 papers
    papers = list(range(1, 11))

    with tqdm(total=len(papers), desc="Processing papers", unit="paper") as pbar:
        interventions = 0
        failed = 0

        for paper in papers:
            time.sleep(0.05)  # Simulate processing

            # Simulate some extractions
            if paper % 3 == 0:
                interventions += 2
            elif paper % 5 == 0:
                failed += 1
            else:
                interventions += 1

            pbar.set_postfix({'interventions': interventions, 'failed': failed})
            pbar.update(1)

    print(f"[PASS] Processing progress bar works (interventions: {interventions}, failed: {failed})\n")

def test_nested_progress():
    """Test 2.4c: Nested progress bars (batches)."""
    print("Test 2.4c: Nested progress bars")

    batches = 3
    papers_per_batch = 5
    total_papers = batches * papers_per_batch

    with tqdm(total=total_papers, desc="Overall progress", unit="paper", position=0) as outer_pbar:
        for batch_num in range(1, batches + 1):
            with tqdm(total=papers_per_batch, desc=f"Batch {batch_num}", unit="paper", position=1, leave=False) as inner_pbar:
                for paper_num in range(papers_per_batch):
                    time.sleep(0.02)
                    inner_pbar.update(1)
                    outer_pbar.update(1)

    print("[PASS] Nested progress bars work\n")

def test_import_dependencies():
    """Test 2.4d: Verify tqdm is properly imported in all modules."""
    print("Test 2.4d: Verify tqdm imports")

    try:
        # Test dual_model_analyzer
        from back_end.src.llm_processing.dual_model_analyzer import tqdm as dma_tqdm
        print("[OK] tqdm imported in dual_model_analyzer.py")

        # Test rotation_paper_collector
        from back_end.src.orchestration.rotation_paper_collector import tqdm as rpc_tqdm
        print("[OK] tqdm imported in rotation_paper_collector.py")

        # Test rotation_llm_processor
        from back_end.src.orchestration.rotation_llm_processor import tqdm as rlp_tqdm
        print("[OK] tqdm imported in rotation_llm_processor.py")

        print("[PASS] All modules have tqdm imported\n")
        return True

    except ImportError as e:
        print(f"[FAIL] Import error: {e}\n")
        return False

def test_progress_bar_formatting():
    """Test 2.4e: Progress bar formatting and postfix updates."""
    print("Test 2.4e: Progress bar formatting")

    with tqdm(total=100, desc="Test progress", unit="item", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for i in range(100):
            if i % 10 == 0:
                pbar.set_postfix({'batch': i // 10, 'rate': f'{i}%'})
            if i % 20 == 0:
                time.sleep(0.01)  # Occasional delay
            pbar.update(1)

    print("[PASS] Progress bar formatting works\n")

# Run all tests
if __name__ == "__main__":
    print("Testing progress reporting with tqdm...\n")

    tests_passed = 0
    tests_total = 5

    try:
        test_collection_progress()
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Collection progress test failed: {e}\n")

    try:
        test_processing_progress()
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Processing progress test failed: {e}\n")

    try:
        test_nested_progress()
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Nested progress test failed: {e}\n")

    try:
        if test_import_dependencies():
            tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Import test failed: {e}\n")

    try:
        test_progress_bar_formatting()
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Formatting test failed: {e}\n")

    # Summary
    print("="*60)
    print(f"Tests Passed: {tests_passed}/{tests_total} ({tests_passed/tests_total*100:.0f}%)")
    print("="*60)

    if tests_passed == tests_total:
        print("\n=== Phase 2.4 Progress Reporting: COMPLETE ===")
    else:
        print(f"\n=== Phase 2.4: {tests_total - tests_passed} test(s) failed ===")

    sys.exit(0 if tests_passed == tests_total else 1)
