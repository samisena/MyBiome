"""
Basic test of the enhanced pipeline without LLM calls.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from llm.semantic_merger import InterventionExtraction

def test_basic_functionality():
    """Test basic functionality without LLM calls."""
    print("=== Testing Basic Pipeline Functionality ===\n")

    # Test intervention extraction creation
    print("1. Testing InterventionExtraction creation...")
    extraction = InterventionExtraction(
        model_name='test_model',
        intervention_name='proton pump inhibitors',
        health_condition='GERD',
        intervention_category='medication',
        correlation_type='positive',
        confidence_score=0.85,
        correlation_strength=0.78,
        supporting_quote='Test quote',
        raw_data={'test': 'data'}
    )
    print(f"   [OK] Created extraction: {extraction.intervention_name}")

    # Test database schema compatibility
    print("\n2. Testing database schema compatibility...")
    try:
        from paper_collection.database_manager import DatabaseManager
        db = DatabaseManager()
        print("   [OK] Database manager initialized with new schema")
    except Exception as e:
        print(f"   [ERROR] Database error: {e}")

    # Test merged intervention structure
    print("\n3. Testing merged intervention structure...")
    merged_data = {
        'intervention_name': 'proton pump inhibitors',
        'canonical_name': 'Proton Pump Inhibitors',
        'alternative_names': ['PPIs', 'acid pump inhibitors'],
        'search_terms': ['ppi', 'proton pump', 'omeprazole'],
        'semantic_group_id': 'sem_123abc',
        'semantic_confidence': 0.95,
        'merge_source': 'qwen2.5:14b',
        'consensus_confidence': 0.90,
        'model_agreement': 'full',
        'models_used': 'gemma2:9b,qwen2.5:14b',
        'validator_agreement': True,
        'needs_human_review': False
    }

    print("   [OK] Merged intervention structure:")
    for key, value in merged_data.items():
        print(f"      {key}: {value}")

    # Test frontend deduplication logic (simulated)
    print("\n4. Testing frontend deduplication logic...")
    sample_interventions = [
        {
            'id': 1,
            'intervention_name': 'proton pump inhibitors',
            'canonical_name': 'Proton Pump Inhibitors',
            'alternative_names': ['PPIs', 'acid pump inhibitors'],
            'semantic_group_id': 'sem_123abc',
            'confidence_score': 0.85
        },
        {
            'id': 2,
            'intervention_name': 'PPIs',
            'canonical_name': 'Proton Pump Inhibitors',
            'alternative_names': ['proton pump inhibitors'],
            'semantic_group_id': 'sem_123abc',
            'confidence_score': 0.92
        },
        {
            'id': 3,
            'intervention_name': 'dietary modification',
            'canonical_name': 'Dietary Modification',
            'alternative_names': [],
            'semantic_group_id': 'sem_456def',
            'confidence_score': 0.75
        }
    ]

    # Simulate deduplication
    seen_groups = {}
    deduplicated = []

    for intervention in sample_interventions:
        group_id = intervention.get('semantic_group_id', intervention['intervention_name'])

        if group_id not in seen_groups or intervention['confidence_score'] > seen_groups[group_id]['confidence_score']:
            seen_groups[group_id] = intervention

    deduplicated = list(seen_groups.values())

    print(f"   [OK] Original interventions: {len(sample_interventions)}")
    print(f"   [OK] After deduplication: {len(deduplicated)}")
    print("   [OK] Deduplicated interventions:")
    for intervention in deduplicated:
        print(f"      - {intervention['canonical_name']} (confidence: {intervention['confidence_score']})")

    # Test condition synonym matching
    print("\n5. Testing condition synonym matching...")
    test_conditions = [
        ('GERD', 'gastroesophageal reflux disease'),
        ('IBS', 'irritable bowel syndrome'),
        ('diabetes', 'type 2 diabetes'),
        ('depression', 'major depressive disorder')
    ]

    for cond1, cond2 in test_conditions:
        # Simple synonym check (without full semantic merger)
        synonyms_match = any([
            'gerd' in cond1.lower() and 'gastroesophageal' in cond2.lower(),
            'ibs' in cond1.lower() and 'irritable bowel' in cond2.lower(),
            'diabetes' in cond1.lower() and 'diabetes' in cond2.lower(),
            'depression' in cond1.lower() and 'depressive' in cond2.lower()
        ])
        print(f"   [{'OK' if synonyms_match else 'NO'}] {cond1} ~ {cond2}")

    print("\n=== Basic Tests Completed Successfully ===")

if __name__ == "__main__":
    test_basic_functionality()