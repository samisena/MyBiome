#!/usr/bin/env python3
"""
Test the actual LLM semantic merger to show how the second LLM layer works.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import SemanticMerger, InterventionExtraction

def test_llm_semantic_merger():
    """Test the actual LLM semantic merger with real GERD PPI examples."""
    print("=== Testing LLM Semantic Merger ===")
    print("Models: primary_model='qwen2.5:14b', validator_model='gemma2:9b'")
    print()

    # Initialize the semantic merger with actual LLM models
    print("Initializing SemanticMerger...")
    merger = SemanticMerger(
        primary_model='qwen2.5:14b',    # Primary analysis model
        validator_model='gemma2:9b'     # Validation model
    )

    # Create the PPI interventions that are causing duplicates
    ppi1 = InterventionExtraction(
        model_name='gemma2:9b',
        intervention_name='proton pump inhibitors',
        health_condition='GERD',
        intervention_category='medication',
        correlation_type='positive',
        confidence_score=0.85,
        correlation_strength=0.78,
        supporting_quote='Proton pump inhibitors significantly reduced GERD symptoms in clinical trials.',
        raw_data={'paper_id': 'paper_123'}
    )

    ppi2 = InterventionExtraction(
        model_name='qwen2.5:14b',
        intervention_name='proton pump inhibitors (PPI)',
        health_condition='gastro-esophageal reflux disease (GERD)',
        intervention_category='medication',
        correlation_type='positive',
        confidence_score=0.92,
        correlation_strength=0.81,
        supporting_quote='PPIs demonstrated superior acid suppression compared to H2 blockers.',
        raw_data={'paper_id': 'paper_456'}
    )

    ppi3 = InterventionExtraction(
        model_name='gemma2:9b',
        intervention_name='proton pump inhibitors (PPIs)',
        health_condition='reflux esophagitis (RE)',
        intervention_category='medication',
        correlation_type='positive',
        confidence_score=0.88,
        correlation_strength=0.75,
        supporting_quote='Proton pump inhibitors (PPIs) showed significant efficacy in treating reflux esophagitis.',
        raw_data={'paper_id': 'paper_789'}
    )

    print("Testing PPI interventions:")
    print(f"1. '{ppi1.intervention_name}' (condition: {ppi1.health_condition})")
    print(f"2. '{ppi2.intervention_name}' (condition: {ppi2.health_condition})")
    print(f"3. '{ppi3.intervention_name}' (condition: {ppi3.health_condition})")
    print()

    # Test semantic comparison between PPI variants
    print("=== LLM Semantic Analysis ===")

    try:
        print("Comparing PPI #1 vs PPI #2...")
        decision12 = merger.compare_interventions(ppi1, ppi2)

        print(f"Result: {'DUPLICATE' if decision12.is_duplicate else 'NOT DUPLICATE'}")
        print(f"Confidence: {decision12.semantic_confidence}")
        print(f"Canonical Name: {decision12.canonical_name}")
        print(f"Alternative Names: {decision12.alternative_names}")
        print(f"Reasoning: {decision12.reasoning}")
        print(f"Method: {decision12.merge_method}")
        print()

        if decision12.is_duplicate:
            print("Validating merge decision with second model...")
            validation12 = merger.validate_merge_decision(decision12, ppi1, ppi2)
            print(f"Validation agrees: {validation12.agrees_with_merge}")
            print(f"Validation confidence: {validation12.confidence}")
            print(f"Validation reasoning: {validation12.reasoning}")
            print()

        print("Comparing PPI #1 vs PPI #3...")
        decision13 = merger.compare_interventions(ppi1, ppi3)

        print(f"Result: {'DUPLICATE' if decision13.is_duplicate else 'NOT DUPLICATE'}")
        print(f"Confidence: {decision13.semantic_confidence}")
        print(f"Canonical Name: {decision13.canonical_name}")
        print(f"Alternative Names: {decision13.alternative_names}")
        print(f"Reasoning: {decision13.reasoning}")
        print()

        # Create merged intervention if duplicates found
        if decision12.is_duplicate:
            print("Creating merged intervention...")
            merged = merger.create_merged_intervention([ppi1, ppi2, ppi3], decision12)
            print("Merged intervention:")
            print(f"  Canonical Name: {merged['canonical_name']}")
            print(f"  Alternative Names: {merged['alternative_names']}")
            print(f"  Search Terms: {merged['search_terms']}")
            print(f"  Semantic Group ID: {merged['semantic_group_id']}")
            print(f"  Model Agreement: {merged['model_agreement']}")
            print(f"  Models Used: {merged['models_used']}")
            print(f"  Consensus Confidence: {merged['consensus_confidence']}")

    except Exception as e:
        print(f"Error in LLM processing: {e}")
        print("This might be because LLM models are not available locally.")
        print()
        print("=== How the LLM Semantic Merger Works ===")
        print()
        print("1. PRIMARY MODEL (qwen2.5:14b):")
        print("   - Receives detailed prompt with both interventions")
        print("   - Analyzes semantic similarity considering:")
        print("     * Synonyms and abbreviations (PPI = proton pump inhibitors)")
        print("     * Different levels of specificity")
        print("     * Mechanism of action similarity")
        print("     * Medical literature naming conventions")
        print("   - Returns JSON with merge decision and reasoning")
        print()
        print("2. VALIDATOR MODEL (gemma2:9b):")
        print("   - Reviews the primary model's decision")
        print("   - Provides independent validation")
        print("   - Flags potential errors or risky merges")
        print("   - Suggests corrections if needed")
        print()
        print("3. MERGE CREATION:")
        print("   - Selects best intervention as canonical")
        print("   - Collects all variations as alternative names")
        print("   - Generates comprehensive search terms")
        print("   - Creates semantic group ID for future matching")
        print()
        print("Expected outcome for PPIs:")
        print("- Should identify all 3 as duplicates")
        print("- Canonical: 'proton pump inhibitors'")
        print("- Alternatives: ['proton pump inhibitors', 'proton pump inhibitors (PPI)', 'proton pump inhibitors (PPIs)', 'PPIs', 'PPI']")
        print("- Single semantic group ID for all variants")

    # Print statistics
    print("\n=== Processing Statistics ===")
    stats = merger.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_llm_semantic_merger()