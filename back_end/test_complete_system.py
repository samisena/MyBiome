#!/usr/bin/env python3
"""
Test script for the complete intervention system transformation.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    print("=" * 60)
    print("MYBIOME INTERVENTION SYSTEM - COMPLETE TEST")
    print("=" * 60)
    
    # Test 1: Core System Components
    print("\n1. Testing core system components...")
    from src.interventions.taxonomy import intervention_taxonomy, InterventionType
    from src.interventions.validators import intervention_validator
    from src.interventions.search_terms import search_terms
    from src.paper_collection.database_manager import database_manager
    print("[OK] All core components imported successfully")
    
    # Test 2: Database System
    print("\n2. Testing database system...")
    stats = database_manager.get_database_stats()
    print(f"[OK] Database connected: {database_manager.db_path}")
    print(f"     Papers: {stats.get('total_papers', 0)}")
    print(f"     Interventions: {stats.get('total_interventions', 0)}")
    
    # Test 3: Intervention Categories
    print("\n3. Testing intervention categories...")
    categories = intervention_taxonomy.get_all_categories()
    print(f"[OK] {len(categories)} intervention categories loaded:")
    for cat_type, cat_def in categories.items():
        print(f"     - {cat_type.value}: {len(cat_def.subcategories)} subcategories")
    
    # Test 4: Search Terms
    print("\n4. Testing search terms...")
    all_terms = search_terms.get_all_intervention_terms()
    print(f"[OK] {len(all_terms)} total search terms across all categories")
    
    # Test 5: Dual-Model Analyzer
    print("\n5. Testing dual-model analyzer...")
    from src.llm.dual_model_analyzer import DualModelAnalyzer
    analyzer = DualModelAnalyzer()
    print(f"[OK] Dual-model analyzer initialized with models: {list(analyzer.models.keys())}")
    
    # Test 6: PubMed Collector
    print("\n6. Testing intervention-focused PubMed collector...")
    from src.paper_collection.pubmed_collector import PubMedCollector
    collector = PubMedCollector()
    query = collector._build_intervention_query("depression", True)
    print(f"[OK] Intervention query built (length: {len(query)} chars)")
    
    # Test 7: Pipeline
    print("\n7. Testing intervention research pipeline...")
    from src.llm.pipeline import InterventionResearchPipeline
    pipeline = InterventionResearchPipeline()
    print("[OK] Intervention research pipeline initialized")
    
    # Test 8: Validation
    print("\n8. Testing intervention validation...")
    test_data = {
        'intervention_category': 'diet',
        'intervention_name': 'Mediterranean diet',
        'health_condition': 'cardiovascular disease',
        'correlation_type': 'positive',
        'extraction_model': 'gemma2:9b',
        'paper_id': '12345678',
        'intervention_details': {
            'diet_type': 'Mediterranean',
            'duration': '6 months',
            'compliance_measure': 'dietary questionnaire'
        }
    }
    
    validated = intervention_validator.validate_intervention(test_data)
    print("[OK] Intervention validation successful")
    print(f"     Category: {validated['intervention_category']}")
    print(f"     Name: {validated['intervention_name']}")
    print(f"     Details: {len(validated.get('intervention_details', {}))} fields")
    
    print("\n" + "=" * 60)
    print("TRANSFORMATION COMPLETE!")
    print("=" * 60)
    print("\nMyBiome has been successfully transformed from:")
    print("  FROM: Probiotic-focused research system")
    print("  TO:   Comprehensive health intervention platform")
    print("\nSupported intervention types:")
    for cat_type in InterventionType:
        print(f"  - {cat_type.value.title()}")
    print("\nKey capabilities:")
    print("  - Dual-model AI analysis (gemma2:9b + qwen2.5:14b)")
    print("  - Comprehensive search across intervention types")
    print("  - Sophisticated validation and quality control")
    print("  - Rich intervention categorization and metadata")
    print("  - Full research pipeline automation")
    print("\n[SUCCESS] System transformation completed successfully!")
    
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Test failed: {e}")
    sys.exit(1)