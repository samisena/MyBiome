#!/usr/bin/env python3
"""Test LLM processing directly."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from back_end.src.orchestration.rotation_llm_processor import RotationLLMProcessor

processor = RotationLLMProcessor()

# Get unprocessed papers
papers = processor._get_all_unprocessed_papers()
print(f"Found {len(papers)} unprocessed papers")

if papers:
    print(f"First paper: {papers[0].get('pmid')} - {papers[0].get('title', '')[:50]}...")

# Check thermal status
thermal = processor.get_thermal_status()
print(f"\nThermal status:")
print(f"  GPU temp: {thermal.get('gpu_temp')}°C")
print(f"  Is safe: {thermal.get('is_safe')}")

# Try processing just 1 paper
if papers:
    print("\nTrying to process 1 paper...")
    from back_end.src.llm_processing.dual_model_analyzer import DualModelAnalyzer

    analyzer = DualModelAnalyzer()
    result = analyzer.process_single_paper(papers[0])
    print(f"Processing result: {result.get('total_interventions')} interventions extracted")