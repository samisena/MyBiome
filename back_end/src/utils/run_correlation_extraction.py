#!/usr/bin/env python3
"""
Script to run intervention extraction on papers in the database.
"""

import sys
from pathlib import Path

try:
    from back_end.src.data_collection.database_manager import database_manager
    from back_end.src.llm_processing.dual_model_analyzer import DualModelAnalyzer
    from back_end.src.data.config import config, setup_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the MyBiome directory")
    sys.exit(1)

def run_intervention_extraction(limit: int = None, batch_size: int = 3):
    """Run intervention extraction on pending papers using dual-model approach."""
    try:
        print("=== Running Intervention Extraction (Dual-Model) ===")
        
        # Initialize dual-model analyzer
        analyzer = DualModelAnalyzer()
        
        # Get papers that need processing
        papers_to_process = analyzer.get_unprocessed_papers(limit)
        
        print(f"Found {len(papers_to_process)} papers to process")
        
        if not papers_to_process:
            print("No papers need processing!")
            return True
        
        # Process papers in batches
        results = analyzer.process_unprocessed_papers(
            limit=limit,
            batch_size=batch_size
        )
        
        print("\nIntervention Extraction Results:")
        print(f"  Papers processed: {results.get('successful_papers', 0)}/{results.get('total_papers', 0)}")
        print(f"  Papers failed: {len(results.get('failed_papers', []))}")
        print(f"  Total interventions found: {results.get('total_interventions', 0)}")
        
        # Show interventions by category
        categories = results.get('interventions_by_category', {})
        if categories:
            print("  Interventions by category:")
            for category, count in categories.items():
                if count > 0:
                    print(f"    {category}: {count}")
        
        # Show model statistics
        model_stats = results.get('model_statistics', {})
        if model_stats:
            print("  Model statistics:")
            for model, stats in model_stats.items():
                print(f"    {model}: {stats.get('interventions', 0)} interventions from {stats.get('papers', 0)} papers")
        
        
        return results.get('successful_papers', 0) > 0
        
    except Exception as e:
        print(f"Error during intervention extraction: {e}")
        return False

def show_final_stats():
    """Show final database statistics."""
    print("\n=== Final Database Statistics ===")
    
    stats = database_manager.get_database_stats()
    print(f"Total papers: {stats.get('total_papers', 0)}")
    print(f"Total interventions: {stats.get('total_interventions', 0)}")
    print(f"Papers with fulltext: {stats.get('papers_with_fulltext', 0)}")
    
    print("\nProcessing Status:")
    for status, count in stats.get('processing_status', {}).items():
        print(f"  {status}: {count}")
    
    print("\nValidation Status:")
    for status, count in stats.get('validation_status', {}).items():
        print(f"  {status}: {count}")
    
    # Show intervention categories if any exist
    categories = stats.get('intervention_categories', {})
    if categories:
        print("\nInterventions by Category:")
        for category, count in categories.items():
            print(f"  {category}: {count}")
    
    # Show some example interventions if any exist
    if stats.get('total_interventions', 0) > 0:
        print("\nSample Interventions:")
        with database_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT intervention_category, intervention_name, health_condition, 
                       correlation_type, correlation_strength, confidence_score
                FROM interventions 
                ORDER BY confidence_score DESC 
                LIMIT 5
            ''')
            
            for row in cursor.fetchall():
                strength = f", strength: {row[4]:.2f}" if row[4] is not None else ""
                confidence = f", confidence: {row[5]:.2f}" if row[5] is not None else ""
                print(f"  • {row[0]}: {row[1]} → {row[2]} ({row[3]}{strength}{confidence})")

def main():
    """Main function to run intervention extraction."""
    print("MyBiome Intervention Extraction (Dual-Model)")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(__name__, 'intervention_extraction.log')
    
    try:
        # Run intervention extraction on all pending papers
        success = run_intervention_extraction(
            limit=None,  # Process all papers
            batch_size=3  # Process in small batches for dual models
        )
        
        if success:
            print("\n[SUCCESS] Intervention extraction completed successfully!")
        else:
            print("\n[FAILED] Intervention extraction failed or no interventions found")
        
        # Show final statistics
        show_final_stats()
        
        return success
        
    except Exception as e:
        logger.error(f"Intervention extraction failed: {e}")
        print(f"[ERROR] Intervention extraction failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)