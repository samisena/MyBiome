#!/usr/bin/env python3
"""
Test script for the new intervention system.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    # Test imports
    print("Testing imports...")
    from src.interventions.taxonomy import intervention_taxonomy, InterventionType
    from src.interventions.validators import intervention_validator
    from src.interventions.search_terms import search_terms
    from src.paper_collection.database_manager import database_manager
    print("[OK] All imports successful")
    
    # Test taxonomy
    print("\nTesting intervention taxonomy...")
    categories = intervention_taxonomy.get_all_categories()
    print(f"[OK] Found {len(categories)} intervention categories:")
    for cat_type, cat_def in categories.items():
        print(f"  - {cat_type.value}: {cat_def.display_name}")
        print(f"    Subcategories: {len(cat_def.subcategories)}")
    
    # Test search terms
    print("\nTesting search terms...")
    for cat_type in InterventionType:
        terms = search_terms.get_terms_for_category(cat_type)
        print(f"[OK] {cat_type.value}: {len(terms)} search terms")
    
    # Test database setup
    print("\nTesting database setup...")
    stats = database_manager.get_database_stats()
    print(f"[OK] Database connected successfully")
    print(f"   Total papers: {stats.get('total_papers', 0)}")
    print(f"   Total interventions: {stats.get('total_interventions', 0)}")
    print(f"   Database file: {database_manager.db_path}")
    
    # Test validator
    print("\nTesting intervention validator...")
    test_intervention = {
        'intervention_category': 'exercise',
        'intervention_name': 'aerobic exercise',
        'health_condition': 'depression',
        'correlation_type': 'positive',
        'extraction_model': 'test_model',
        'paper_id': '12345678',
        'intervention_details': {
            'exercise_type': 'cycling',
            'duration': '30 minutes',
            'frequency': '3x per week',
            'intensity': 'moderate'
        }
    }
    
    try:
        validated = intervention_validator.validate_intervention(test_intervention)
        print("[OK] Intervention validation successful")
        print(f"   Validated category: {validated['intervention_category']}")
        print(f"   Validated name: {validated['intervention_name']}")
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
    
    print("\n[SUCCESS] All tests passed! Intervention system is working correctly.")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    sys.exit(1)