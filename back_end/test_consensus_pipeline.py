#!/usr/bin/env python3
"""
Test script for the new multi-LLM consensus pipeline.
This script demonstrates how to use the updated pipeline with gemma2:9b and qwen2.5:14b.
"""

import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.llm.pipeline import EnhancedResearchPipeline
from src.data.config import config

def main():
    """Test the consensus pipeline with a small batch of papers."""
    
    print("=" * 60)
    print("TESTING MULTI-LLM CONSENSUS PIPELINE")
    print("=" * 60)
    
    # Display current configuration
    print(f"Models configured:")
    for i, model_config in enumerate(config.multi_llm.models, 1):
        print(f"  {i}. {model_config.model_name}")
    
    print(f"Consensus threshold: {config.multi_llm.consensus_threshold}")
    print(f"Conflict resolution: {config.multi_llm.conflict_resolution}")
    
    # Initialize pipeline with consensus analysis
    pipeline = EnhancedResearchPipeline(use_consensus=True)
    
    # Test with a small batch of papers (limit to 3 for testing)
    print(f"\nTesting consensus analysis with 3 papers...")
    
    try:
        # Just analyze existing papers without collecting new ones
        analysis_results = pipeline.analyze_correlations(
            limit_papers=3  # Small test batch
        )
        
        print(f"\n=== Test Results ===")
        print(f"Analysis type: {analysis_results.get('analysis_type')}")
        print(f"Papers processed: {analysis_results.get('papers_processed', 0)}")
        
        if analysis_results.get('analysis_type') == 'consensus':
            print(f"Agreed correlations: {analysis_results.get('agreed_correlations', 0)}")
            print(f"Conflicts: {analysis_results.get('conflicts', 0)}")
            print(f"Papers needing review: {analysis_results.get('papers_needing_review', 0)}")
            
            # Show token usage per model
            token_usage = analysis_results.get('token_usage', {})
            if token_usage:
                print(f"\nToken usage by model:")
                for model, usage in token_usage.items():
                    if isinstance(usage, dict):
                        print(f"  {model}: {usage.get('total_tokens', 0):,} tokens")
        
        print(f"Success rate: {analysis_results.get('success_rate', 0):.1f}%")
        print(f"Stage duration: {analysis_results.get('stage_duration', 0):.1f}s")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return 1
    
    print(f"\n=== Test completed successfully! ===")
    print(f"\nTo use the consensus pipeline in your scripts:")
    print(f"```python")
    print(f"from src.llm.pipeline import EnhancedResearchPipeline")
    print(f"")
    print(f"# Use consensus analysis (default)")
    print(f"pipeline = EnhancedResearchPipeline(use_consensus=True)")
    print(f"")
    print(f"# Or use single LLM (backward compatibility)")
    print(f"pipeline = EnhancedResearchPipeline(use_consensus=False)")
    print(f"```")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)