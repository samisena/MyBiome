#!/usr/bin/env python3
"""
Test the corrected consensus logic for different scenarios.
"""

import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.llm.consensus_analyzer import MultiLLMConsensusAnalyzer, ExtractionResult

def test_no_correlations_consensus():
    """Test that when both models find no correlations, needs_review=False."""
    
    analyzer = MultiLLMConsensusAnalyzer()
    
    # Simulate both models finding no correlations (successful extraction, empty results)
    extractions = [
        ExtractionResult(
            model_name="gemma2:9b",
            correlations=[],  # No correlations found
            extraction_time=10.0,
            token_usage={'input': 100, 'output': 50, 'total': 150},
            error=None  # No error - successful extraction
        ),
        ExtractionResult(
            model_name="qwen2.5:14b", 
            correlations=[],  # No correlations found
            extraction_time=15.0,
            token_usage={'input': 120, 'output': 60, 'total': 180},
            error=None  # No error - successful extraction
        )
    ]
    
    result = analyzer.analyze_consensus(extractions)
    
    print("=== TEST: Both models find no correlations ===")
    print(f"Consensus status: {result.consensus_status}")
    print(f"Needs review: {result.needs_review}")
    print(f"Review reasons: {result.review_reasons}")
    print(f"Expected: needs_review=False (both models agree: no correlations)")
    
    # This should NOT need review - both models successfully agreed on no correlations
    expected_result = result.needs_review == False and result.consensus_status == 'no_correlations_consensus'
    print(f"PASS: {expected_result}")
    
    return expected_result

def test_model_failure_case():
    """Test that when models fail to run, needs_review=True."""
    
    analyzer = MultiLLMConsensusAnalyzer()
    
    # Simulate one model succeeding (no correlations) and one failing
    extractions = [
        ExtractionResult(
            model_name="gemma2:9b",
            correlations=[],  # No correlations found
            extraction_time=10.0,
            token_usage={'input': 100, 'output': 50, 'total': 150},
            error=None  # No error - successful extraction
        ),
        ExtractionResult(
            model_name="qwen2.5:14b",
            correlations=[],  # No correlations (but doesn't matter due to error)
            extraction_time=5.0,
            token_usage={'input': 0, 'output': 0, 'total': 0},
            error="API timeout"  # Error occurred
        )
    ]
    
    result = analyzer.analyze_consensus(extractions)
    
    print("\n=== TEST: One model fails, one succeeds with no correlations ===")
    print(f"Consensus status: {result.consensus_status}")
    print(f"Needs review: {result.needs_review}")
    print(f"Review reasons: {result.review_reasons}")
    print(f"Expected: needs_review=True (model failure)")
    
    # This SHOULD need review - one model failed
    expected_result = result.needs_review == True and result.consensus_status == 'extraction_failures'
    print(f"PASS: {expected_result}")
    
    return expected_result

def test_disagreement_case():
    """Test that when models disagree on correlations, needs_review=True."""
    
    analyzer = MultiLLMConsensusAnalyzer()
    
    # Simulate models finding different correlations
    extractions = [
        ExtractionResult(
            model_name="gemma2:9b",
            correlations=[
                {
                    'paper_id': 'test123',
                    'probiotic_strain': 'Lactobacillus acidophilus',
                    'health_condition': 'digestive health',
                    'correlation_type': 'positive',
                    'extraction_model': 'gemma2:9b'
                }
            ],
            extraction_time=10.0,
            token_usage={'input': 100, 'output': 50, 'total': 150},
            error=None
        ),
        ExtractionResult(
            model_name="qwen2.5:14b",
            correlations=[],  # Different result - found no correlations
            extraction_time=15.0,
            token_usage={'input': 120, 'output': 60, 'total': 180},
            error=None
        )
    ]
    
    result = analyzer.analyze_consensus(extractions)
    
    print("\n=== TEST: Models disagree (one finds correlation, other doesn't) ===")
    print(f"Consensus status: {result.consensus_status}")
    print(f"Needs review: {result.needs_review}")
    print(f"Review reasons: {result.review_reasons}")
    print(f"Model-specific findings: {len(result.model_specific_correlations['gemma2:9b'])} vs {len(result.model_specific_correlations['qwen2.5:14b'])}")
    print(f"Expected: needs_review=True (disagreement)")
    
    # This SHOULD need review - models disagree
    expected_result = result.needs_review == True
    print(f"PASS: {expected_result}")
    
    return expected_result

def test_full_agreement_case():
    """Test that when models agree on correlations, needs_review=False."""
    
    analyzer = MultiLLMConsensusAnalyzer()
    
    # Simulate both models finding the same correlation
    correlation_data = {
        'paper_id': 'test123',
        'probiotic_strain': 'Lactobacillus acidophilus',
        'health_condition': 'digestive health', 
        'correlation_type': 'positive',
        'correlation_strength': 0.8,
        'confidence_score': 0.9
    }
    
    extractions = [
        ExtractionResult(
            model_name="gemma2:9b",
            correlations=[{**correlation_data, 'extraction_model': 'gemma2:9b'}],
            extraction_time=10.0,
            token_usage={'input': 100, 'output': 50, 'total': 150},
            error=None
        ),
        ExtractionResult(
            model_name="qwen2.5:14b",
            correlations=[{**correlation_data, 'extraction_model': 'qwen2.5:14b'}],
            extraction_time=15.0,
            token_usage={'input': 120, 'output': 60, 'total': 180},
            error=None
        )
    ]
    
    result = analyzer.analyze_consensus(extractions)
    
    print("\n=== TEST: Both models agree on same correlation ===")
    print(f"Consensus status: {result.consensus_status}")
    print(f"Needs review: {result.needs_review}")
    print(f"Agreed correlations: {len(result.agreed_correlations)}")
    print(f"Conflicts: {len(result.conflicting_correlations)}")
    print(f"Expected: needs_review=False (full agreement)")
    
    # This should NOT need review - both models agree
    expected_result = result.needs_review == False and result.consensus_status == 'full_agreement'
    print(f"PASS: {expected_result}")
    
    return expected_result

def main():
    """Run all consensus logic tests."""
    
    print("=" * 70)
    print("TESTING CORRECTED CONSENSUS LOGIC")
    print("=" * 70)
    
    # Run all test scenarios
    test1 = test_no_correlations_consensus()
    test2 = test_model_failure_case()
    test3 = test_disagreement_case()
    test4 = test_full_agreement_case()
    
    all_passed = all([test1, test2, test3, test4])
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  Both models, no correlations: {'PASS' if test1 else 'FAIL'}")
    print(f"  Model failure case: {'PASS' if test2 else 'FAIL'}")
    print(f"  Models disagree: {'PASS' if test3 else 'FAIL'}")
    print(f"  Models agree on correlations: {'PASS' if test4 else 'FAIL'}")
    print("=" * 70)
    
    if all_passed:
        print("ALL TESTS PASSED - CONSENSUS LOGIC FIXED!")
    else:
        print("SOME TESTS FAILED - LOGIC NEEDS MORE WORK")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)