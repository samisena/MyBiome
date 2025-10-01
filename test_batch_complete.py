#!/usr/bin/env python3
"""
Test the complete batch_medical_rotation pipeline with just 3 conditions.
This demonstrates all 3 phases: collection, dual LLM processing, and deduplication.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.orchestration.batch_medical_rotation import BatchMedicalRotationPipeline
from back_end.src.data_collection.database_manager import database_manager
from back_end.src.data.config import config

def main():
    print("=" * 60)
    print("BATCH MEDICAL ROTATION - COMPLETE PIPELINE TEST")
    print("=" * 60)

    # Temporarily override config to use only 3 conditions for testing
    original_specialties = config.medical_specialties
    config.medical_specialties = {
        'cardiology': ['hypertension'],
        'endocrinology': ['diabetes'],
        'neurology': ['alzheimers']
    }

    try:
        # Create pipeline
        pipeline = BatchMedicalRotationPipeline()

        print("\nRunning pipeline for 3 conditions with 1 paper each...")
        print("Conditions: hypertension, diabetes, alzheimers")

        # Run the complete pipeline
        result = pipeline.run_batch_pipeline(papers_per_condition=1, resume=False)

        if result['success']:
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"Session: {result['session_id']}")
            print(f"Total time: {result['total_time_seconds']:.1f} seconds")
            print("\nStatistics:")
            for key, value in result['statistics'].items():
                print(f"  {key}: {value}")

            # Check final database state
            with database_manager.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM papers")
                paper_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM interventions")
                intervention_count = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(DISTINCT intervention_canonical_id) FROM interventions WHERE intervention_canonical_id IS NOT NULL")
                canonical_count = cursor.fetchone()[0] or 0

                print("\nDatabase Summary:")
                print(f"  Papers: {paper_count}")
                print(f"  Interventions: {intervention_count}")
                print(f"  Canonical entities: {canonical_count}")

                # Show duplicate merging examples
                cursor.execute("""
                    SELECT intervention_name, COUNT(*) as count,
                           GROUP_CONCAT(extraction_model) as models
                    FROM interventions
                    GROUP BY intervention_name
                    HAVING count > 1
                    LIMIT 5
                """)
                duplicates = cursor.fetchall()

                if duplicates:
                    print("\nDuplicate Interventions (from dual LLM):")
                    for name, count, models in duplicates:
                        print(f"  '{name}' - {count} extractions by: {models}")

        else:
            print(f"\nPipeline failed: {result.get('error', 'Unknown error')}")

    finally:
        # Restore original config
        config.medical_specialties = original_specialties
        print("\n(Config restored to original 60 conditions)")

if __name__ == "__main__":
    main()