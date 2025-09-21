#!/usr/bin/env python3
"""
Test script for enhanced intervention extraction pipeline.
Tests paper collection and intervention extraction with new optional fields.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

try:
    from src.llm.pipeline import InterventionResearchPipeline
    from src.paper_collection.database_manager import database_manager
    from src.export_to_json import export_minimal_dataset, export_correlations_data
    from src.data.config import setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the back_end directory")
    sys.exit(1)

def test_database_schema():
    """Test that database schema includes new optional fields."""
    print("=== Testing Database Schema ===")

    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check if new columns exist in interventions table
            cursor.execute("PRAGMA table_info(interventions)")
            columns = [row[1] for row in cursor.fetchall()]

            required_new_columns = ['delivery_method', 'severity', 'adverse_effects', 'cost_category']
            missing_columns = [col for col in required_new_columns if col not in columns]

            if missing_columns:
                print(f"[FAIL] Missing columns: {missing_columns}")
                return False
            else:
                print("[PASS] All new columns present in database schema")
                print(f"   Found columns: {required_new_columns}")
                return True

    except Exception as e:
        print(f"[FAIL] Database schema test failed: {e}")
        return False

def test_paper_collection():
    """Test paper collection with a small sample."""
    print("\n=== Testing Paper Collection ===")

    try:
        pipeline = InterventionResearchPipeline()

        # Test with a small set of health conditions
        test_conditions = ['depression', 'anxiety']

        print(f"Collecting papers for conditions: {test_conditions}")

        results = pipeline.collect_research_data(
            conditions=test_conditions,
            max_papers_per_condition=3,  # Small test
            include_fulltext=False  # Skip fulltext for faster testing
        )

        if results.get('total_papers_collected', 0) > 0:
            print(f"✅ Successfully collected {results['total_papers_collected']} papers")
            print(f"   Conditions processed: {results['successful_conditions']}/{results['conditions_processed']}")
            return True
        else:
            print("❌ No papers collected")
            return False

    except Exception as e:
        print(f"❌ Paper collection test failed: {e}")
        return False

def test_intervention_extraction():
    """Test intervention extraction with enhanced fields."""
    print("\n=== Testing Intervention Extraction ===")

    try:
        pipeline = InterventionResearchPipeline()

        # Process a small number of papers
        print("Running intervention extraction on available papers...")

        results = pipeline.analyze_interventions(
            limit_papers=2,  # Very small test
            batch_size=1     # Process one at a time for easier debugging
        )

        if results.get('interventions_extracted', 0) > 0:
            print(f"✅ Successfully extracted {results['interventions_extracted']} interventions")
            print(f"   Papers processed: {results['papers_processed']}")
            print(f"   Success rate: {results['success_rate']:.1f}%")

            # Show category breakdown
            categories = results.get('interventions_by_category', {})
            if categories:
                print("   Interventions by category:")
                for category, count in categories.items():
                    if count > 0:
                        print(f"     {category}: {count}")

            return True
        else:
            print("❌ No interventions extracted")
            print(f"   Papers processed: {results.get('papers_processed', 0)}")
            return False

    except Exception as e:
        print(f"❌ Intervention extraction test failed: {e}")
        return False

def test_database_content():
    """Test that extracted data includes new optional fields."""
    print("\n=== Testing Database Content ===")

    try:
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check if we have any interventions with new fields
            cursor.execute('''
                SELECT COUNT(*) as total,
                       COUNT(delivery_method) as has_delivery_method,
                       COUNT(severity) as has_severity,
                       COUNT(adverse_effects) as has_adverse_effects,
                       COUNT(cost_category) as has_cost_category
                FROM interventions
            ''')

            row = cursor.fetchone()
            total = row[0]

            if total == 0:
                print("❌ No interventions found in database")
                return False

            print(f"✅ Found {total} interventions in database")
            print(f"   With delivery_method: {row[1]}")
            print(f"   With severity: {row[2]}")
            print(f"   With adverse_effects: {row[3]}")
            print(f"   With cost_category: {row[4]}")

            # Show a sample intervention with new fields
            cursor.execute('''
                SELECT intervention_name, health_condition, correlation_type,
                       delivery_method, severity, adverse_effects, cost_category
                FROM interventions
                WHERE (delivery_method IS NOT NULL OR severity IS NOT NULL OR
                       adverse_effects IS NOT NULL OR cost_category IS NOT NULL)
                LIMIT 1
            ''')

            sample = cursor.fetchone()
            if sample:
                print("   Sample intervention with enhanced fields:")
                print(f"     {sample[0]} → {sample[1]} ({sample[2]})")
                if sample[3]: print(f"     Delivery: {sample[3]}")
                if sample[4]: print(f"     Severity: {sample[4]}")
                if sample[5]: print(f"     Adverse effects: {sample[5]}")
                if sample[6]: print(f"     Cost: {sample[6]}")

            return True

    except Exception as e:
        print(f"❌ Database content test failed: {e}")
        return False

def test_export_functions():
    """Test export functions with enhanced data."""
    print("\n=== Testing Export Functions ===")

    try:
        # Test minimal export
        print("Testing minimal dataset export...")
        minimal_data = export_minimal_dataset()

        if minimal_data:
            print(f"✅ Minimal export: {len(minimal_data)} records")
            if minimal_data:
                sample = minimal_data[0]
                print(f"   Sample: {sample}")
        else:
            print("⚠️  Minimal export returned no data")

        # Test enhanced export
        print("Testing enhanced dataset export...")
        enhanced_data = export_correlations_data()

        if enhanced_data.get('interventions'):
            interventions = enhanced_data['interventions']
            print(f"✅ Enhanced export: {len(interventions)} interventions")

            # Check if enhanced fields are present
            sample = interventions[0] if interventions else {}
            enhanced_fields = ['delivery_method', 'severity', 'adverse_effects', 'cost_category']
            present_fields = [field for field in enhanced_fields if sample.get(field)]

            print(f"   Enhanced fields present: {present_fields}")

            # Show stats
            stats = enhanced_data.get('summary_stats', {})
            print(f"   Total interventions: {stats.get('total_interventions', 0)}")
            print(f"   Unique conditions: {stats.get('unique_conditions', 0)}")
            print(f"   Positive correlations: {stats.get('positive_correlations', 0)}")
        else:
            print("⚠️  Enhanced export returned no interventions")

        return True

    except Exception as e:
        print(f"❌ Export functions test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Enhanced Intervention Extraction Pipeline Test")
    print("=" * 50)

    # Setup logging
    logger = setup_logging(__name__, 'pipeline_test.log')

    tests = [
        ("Database Schema", test_database_schema),
        ("Paper Collection", test_paper_collection),
        ("Intervention Extraction", test_intervention_extraction),
        ("Database Content", test_database_content),
        ("Export Functions", test_export_functions)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced pipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)