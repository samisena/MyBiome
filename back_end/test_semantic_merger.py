"""
Test script for the semantic merger functionality.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import SemanticMerger, InterventionExtraction

def test_semantic_merger():
    """Test the semantic merger with sample interventions."""
    print("=== Testing Semantic Merger ===\n")

    # Initialize the semantic merger
    print("Initializing semantic merger...")
    merger = SemanticMerger()

    # Create test extractions that should be duplicates (GERD example)
    extract1 = InterventionExtraction(
        model_name='gemma2:9b',
        intervention_name='proton pump inhibitors',
        health_condition='gastroesophageal reflux disease (GERD)',
        intervention_category='medication',
        correlation_type='positive',
        confidence_score=0.85,
        correlation_strength=0.78,
        supporting_quote='Proton pump inhibitors significantly reduced GERD symptoms in clinical trials.',
        raw_data={
            'intervention_name': 'proton pump inhibitors',
            'health_condition': 'gastroesophageal reflux disease (GERD)',
            'intervention_category': 'medication',
            'correlation_type': 'positive',
            'confidence_score': 0.85,
            'correlation_strength': 0.78,
            'supporting_quote': 'Proton pump inhibitors significantly reduced GERD symptoms in clinical trials.',
            'extraction_model': 'gemma2:9b'
        }
    )

    extract2 = InterventionExtraction(
        model_name='qwen2.5:14b',
        intervention_name='PPIs',
        health_condition='GERD',
        intervention_category='medication',
        correlation_type='positive',
        confidence_score=0.92,
        correlation_strength=0.81,
        supporting_quote='PPIs demonstrated superior acid suppression compared to H2 blockers.',
        raw_data={
            'intervention_name': 'PPIs',
            'health_condition': 'GERD',
            'intervention_category': 'medication',
            'correlation_type': 'positive',
            'confidence_score': 0.92,
            'correlation_strength': 0.81,
            'supporting_quote': 'PPIs demonstrated superior acid suppression compared to H2 blockers.',
            'extraction_model': 'qwen2.5:14b'
        }
    )

    # Create test extractions that should NOT be duplicates
    extract3 = InterventionExtraction(
        model_name='gemma2:9b',
        intervention_name='dietary modification',
        health_condition='GERD',
        intervention_category='lifestyle',
        correlation_type='positive',
        confidence_score=0.75,
        correlation_strength=0.65,
        supporting_quote='Dietary modifications including avoiding acidic foods helped reduce GERD symptoms.',
        raw_data={
            'intervention_name': 'dietary modification',
            'health_condition': 'GERD',
            'intervention_category': 'lifestyle',
            'correlation_type': 'positive',
            'confidence_score': 0.75,
            'correlation_strength': 0.65,
            'supporting_quote': 'Dietary modifications including avoiding acidic foods helped reduce GERD symptoms.',
            'extraction_model': 'gemma2:9b'
        }
    )

    print("Testing extraction 1 vs extraction 2 (should be duplicates):")
    print(f"  Extract 1: {extract1.intervention_name} for {extract1.health_condition}")
    print(f"  Extract 2: {extract2.intervention_name} for {extract2.health_condition}")

    # Test exact match first
    print("\n1. Testing condition matching...")
    same_condition = merger._are_same_condition(extract1, extract2)
    print(f"   Same condition detected: {same_condition}")

    exact_match = merger._are_exact_matches(extract1, extract2)
    print(f"   Exact match detected: {exact_match}")

    # Test semantic comparison (this will make LLM calls)
    print("\n2. Testing semantic comparison...")
    try:
        merge_decision = merger.compare_interventions(extract1, extract2)
        print(f"   Merge decision: {'DUPLICATE' if merge_decision.is_duplicate else 'NOT DUPLICATE'}")
        print(f"   Canonical name: {merge_decision.canonical_name}")
        print(f"   Alternative names: {merge_decision.alternative_names}")
        print(f"   Confidence: {merge_decision.semantic_confidence}")
        print(f"   Reasoning: {merge_decision.reasoning}")

        # Test validation
        print("\n3. Testing merge validation...")
        validation = merger.validate_merge_decision(merge_decision, extract1, extract2)
        print(f"   Validation agrees: {validation.agrees_with_merge}")
        print(f"   Validation confidence: {validation.confidence}")
        print(f"   Validation reasoning: {validation.reasoning}")

        # Test merged intervention creation
        if merge_decision.is_duplicate and validation.agrees_with_merge:
            print("\n4. Testing merged intervention creation...")
            merged = merger.create_merged_intervention([extract1, extract2], merge_decision)
            print(f"   Merged intervention name: {merged.get('intervention_name')}")
            print(f"   Canonical name: {merged.get('canonical_name')}")
            print(f"   Alternative names: {merged.get('alternative_names')}")
            print(f"   Model agreement: {merged.get('model_agreement')}")
            print(f"   Consensus confidence: {merged.get('consensus_confidence')}")

    except Exception as e:
        print(f"   Error in LLM testing: {e}")
        print("   This is expected if LLM models are not available locally")

    print("\n5. Testing different interventions (should NOT merge):")
    print(f"   Extract 1: {extract1.intervention_name}")
    print(f"   Extract 3: {extract3.intervention_name}")

    try:
        merge_decision_different = merger.compare_interventions(extract1, extract3)
        print(f"   Merge decision: {'DUPLICATE' if merge_decision_different.is_duplicate else 'NOT DUPLICATE'}")
        print(f"   Reasoning: {merge_decision_different.reasoning}")
    except Exception as e:
        print(f"   Error in different intervention test: {e}")

    # Print statistics
    print("\n=== Semantic Merger Statistics ===")
    stats = merger.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== Test Completed ===")

if __name__ == "__main__":
    test_semantic_merger()