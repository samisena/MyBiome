#!/usr/bin/env python3
"""
Test script for the new batch medical rotation pipeline.
Demonstrates the optimized workflow with simplified phases.
"""

import sys
import time
from pathlib import Path

# Add the back_end directory to sys.path for imports
sys.path.append(str(Path(__file__).parent / "back_end"))

def test_batch_collection():
    """Test the batch collection phase."""
    print("="*60)
    print("TESTING BATCH COLLECTION")
    print("="*60)

    try:
        from back_end.src.orchestration.rotation_paper_collector import collect_all_conditions_batch

        print("Testing batch collection for 2 papers per condition...")
        result = collect_all_conditions_batch(papers_per_condition=2, min_year=2020)

        print(f"✓ Collection completed")
        print(f"  Total conditions: {result.total_conditions}")
        print(f"  Successful conditions: {result.successful_conditions}")
        print(f"  Failed conditions: {result.failed_conditions}")
        print(f"  Total papers collected: {result.total_papers_collected}")
        print(f"  Collection time: {result.total_collection_time_seconds:.1f}s")
        print(f"  Quality gate: {'PASSED' if result.success else 'FAILED'}")

        if result.error:
            print(f"  Error: {result.error}")

        return result.success

    except Exception as e:
        print(f"✗ Collection test failed: {e}")
        return False

def test_batch_processing():
    """Test the batch processing phase."""
    print("\n" + "="*60)
    print("TESTING BATCH PROCESSING")
    print("="*60)

    try:
        from back_end.src.orchestration.rotation_llm_processor import RotationLLMProcessor

        processor = RotationLLMProcessor()
        print("Testing batch LLM processing...")

        result = processor.process_all_papers_batch()

        print(f"✓ Processing completed")
        print(f"  Papers found: {result.get('total_papers_found', 0)}")
        print(f"  Papers processed: {result.get('papers_processed', 0)}")
        print(f"  Papers failed: {result.get('papers_failed', 0)}")
        print(f"  Interventions extracted: {result.get('interventions_extracted', 0)}")
        print(f"  Processing time: {result.get('processing_time_seconds', 0):.1f}s")
        print(f"  Success rate: {result.get('success_rate', 0):.1f}%")

        if not result['success']:
            print(f"  Error: {result.get('error', 'Unknown error')}")

        return result['success']

    except Exception as e:
        print(f"✗ Processing test failed: {e}")
        return False

def test_batch_deduplication():
    """Test the batch deduplication phase."""
    print("\n" + "="*60)
    print("TESTING BATCH DEDUPLICATION")
    print("="*60)

    try:
        from back_end.src.orchestration.rotation_deduplication_integrator import deduplicate_all_data_batch

        print("Testing global batch deduplication...")
        result = deduplicate_all_data_batch()

        print(f"✓ Deduplication completed")
        print(f"  Interventions processed: {result.get('total_interventions_processed', 0)}")
        print(f"  Deduplicated interventions: {result.get('deduplicated_interventions', 0)}")
        print(f"  Entities merged: {result.get('entities_merged', 0)}")
        print(f"  Deduplication rate: {result.get('deduplication_rate', 0):.1f}%")
        print(f"  Processing time: {result.get('processing_time_seconds', 0):.1f}s")

        if not result['success']:
            print(f"  Error: {result.get('error', 'Unknown error')}")

        return result['success']

    except Exception as e:
        print(f"✗ Deduplication test failed: {e}")
        return False

def test_complete_batch_pipeline():
    """Test the complete batch pipeline."""
    print("\n" + "="*60)
    print("TESTING COMPLETE BATCH PIPELINE")
    print("="*60)

    try:
        from back_end.src.orchestration.batch_medical_rotation import BatchMedicalRotationPipeline

        pipeline = BatchMedicalRotationPipeline()
        print("Testing complete batch pipeline with 2 papers per condition...")

        result = pipeline.run_batch_pipeline(papers_per_condition=2)

        print(f"✓ Complete pipeline completed")
        print(f"  Success: {result['success']}")
        print(f"  Session: {result.get('session_id', 'unknown')}")
        print(f"  Iteration: {result.get('iteration_completed', 0)}")
        print(f"  Total time: {result.get('total_time_seconds', 0):.1f}s")

        if 'statistics' in result:
            stats = result['statistics']
            print(f"  Papers collected: {stats.get('papers_collected', 0)}")
            print(f"  Papers processed: {stats.get('papers_processed', 0)}")
            print(f"  Interventions extracted: {stats.get('interventions_extracted', 0)}")
            print(f"  Duplicates removed: {stats.get('duplicates_removed', 0)}")

        if not result['success']:
            print(f"  Error: {result.get('error', 'Unknown error')}")

        return result['success']

    except Exception as e:
        print(f"✗ Complete pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("BATCH MEDICAL ROTATION PIPELINE TESTS")
    print("="*60)
    print("Testing the new optimized batch-oriented pipeline")
    print("This tests each phase individually and then the complete workflow")

    all_tests_passed = True

    # Test individual phases
    print("\n📋 PHASE TESTING")
    collection_success = test_batch_collection()
    processing_success = test_batch_processing()
    deduplication_success = test_batch_deduplication()

    # Test complete pipeline
    print("\n🔄 INTEGRATION TESTING")
    pipeline_success = test_complete_batch_pipeline()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Batch Collection: {'✓ PASS' if collection_success else '✗ FAIL'}")
    print(f"Batch Processing: {'✓ PASS' if processing_success else '✗ FAIL'}")
    print(f"Batch Deduplication: {'✓ PASS' if deduplication_success else '✗ FAIL'}")
    print(f"Complete Pipeline: {'✓ PASS' if pipeline_success else '✗ FAIL'}")

    all_tests_passed = all([collection_success, processing_success, deduplication_success, pipeline_success])

    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")

    if all_tests_passed:
        print("\n🎉 The batch medical rotation pipeline is ready for production!")
        print("Key improvements achieved:")
        print("• Simplified 3-phase architecture (collection → processing → deduplication)")
        print("• Parallel collection for faster paper gathering")
        print("• Sequential dual-LLM processing optimized for 8GB VRAM")
        print("• Global deduplication across all conditions")
        print("• Simple session management with phase-level recovery")
        print("• Quality gates between phases")
    else:
        print("\n⚠️  Some tests failed. Please check the logs for details.")

    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())