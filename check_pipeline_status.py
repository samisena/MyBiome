#!/usr/bin/env python3
"""
Quick status checker for the batch medical rotation pipeline.
Shows current progress, phases completed, and whether the pipeline is looping.
"""

import json
import os
from datetime import datetime

def check_pipeline_status():
    session_file = "back_end/data/batch_session.json"

    if not os.path.exists(session_file):
        print("No active session found.")
        return

    with open(session_file, 'r') as f:
        session = json.load(f)

    print("=" * 60)
    print("BATCH MEDICAL ROTATION PIPELINE STATUS")
    print("=" * 60)

    # Session info
    print(f"\nSession ID: {session['session_id']}")
    print(f"Started: {session['start_time']}")
    print(f"Iteration: {session['iteration_number']}")
    print(f"Papers per condition: {session['papers_per_condition']}")

    # Current phase
    print(f"\nCurrent Phase: {session['current_phase'].upper()}")

    # Phase completion status
    print("\nPhase Completion:")
    print(f"  Collection: {'DONE' if session['collection_completed'] else 'IN PROGRESS'}")
    print(f"  Processing: {'DONE' if session['processing_completed'] else 'IN PROGRESS'}")
    print(f"  Deduplication: {'DONE' if session['deduplication_completed'] else 'IN PROGRESS'}")

    # Stats
    print(f"\nTotal Stats:")
    print(f"  Papers collected: {session['total_papers_collected']}")
    print(f"  Papers processed: {session['total_papers_processed']}")
    print(f"  Interventions extracted: {session['total_interventions_extracted']}")

    # Collection result
    if session.get('collection_result'):
        result = session['collection_result']
        print(f"\nCollection Phase:")
        print(f"  Conditions: {result['successful_conditions']}/{result['total_conditions']}")
        print(f"  Papers: {result['total_papers_collected']}")
        print(f"  Success rate: {result['success_rate']:.1f}%")
        print(f"  Time: {result['collection_time_seconds']:.1f}s")

    # Processing result
    if session.get('processing_result'):
        result = session['processing_result']
        print(f"\nProcessing Phase:")
        print(f"  Papers processed: {result['papers_processed']}/{result['total_papers_found']}")
        print(f"  Interventions: {result['interventions_extracted']}")
        print(f"  Success rate: {result['success_rate']:.1f}%")
        print(f"  Time: {result['processing_time_seconds']:.1f}s ({result['processing_time_seconds']/60:.1f} min)")

        if result.get('interventions_by_category'):
            print(f"\n  Categories:")
            for cat, count in sorted(result['interventions_by_category'].items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"    {cat}: {count}")

    # Deduplication result
    if session.get('deduplication_result') and session['deduplication_result'].get('phases_completed'):
        result = session['deduplication_result']
        print(f"\nDeduplication Phase:")
        print(f"  Method: {result.get('method', 'N/A')}")
        print(f"  Phases completed: {', '.join(result['phases_completed'])}")
        print(f"  Time: {result.get('processing_time_seconds', 0):.1f}s")

    # Check if looping
    print("\n" + "=" * 60)
    iteration = session['iteration_number']
    if iteration > 1:
        print(f"LOOPING DETECTED: Pipeline on iteration {iteration}")
        print("The pipeline has restarted and is collecting papers again.")
    else:
        print("Pipeline running first iteration (not looping yet)")

    print("=" * 60)

if __name__ == "__main__":
    check_pipeline_status()