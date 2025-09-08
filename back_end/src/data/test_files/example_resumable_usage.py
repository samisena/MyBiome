#!/usr/bin/env python3
"""
Example usage script for the resumable pipeline.
Shows different ways to run the pipeline with various configurations.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.data.test_files.resumable_pipeline import ResumablePipeline


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    pipeline = ResumablePipeline(
        condition="IBS",
        max_papers=10,
        primary_model="deepseek-llm:7b-chat",
        secondary_model="llama3.1:8b"
    )
    
    success = pipeline.run()
    print(f"Pipeline completed successfully: {success}")
    return success


def example_custom_models():
    """Example with different model configuration"""
    print("\n=== Custom Models Example ===")
    
    pipeline = ResumablePipeline(
        condition="Crohn's disease",
        max_papers=25,
        primary_model="llama3.1:8b",
        secondary_model="deepseek-llm:7b-chat",
        ollama_url="http://localhost:11434/v1"
    )
    
    success = pipeline.run()
    print(f"Pipeline completed successfully: {success}")
    return success


def example_large_study():
    """Example for a large study that might need to be resumed"""
    print("\n=== Large Study Example (Resumable) ===")
    
    pipeline = ResumablePipeline(
        condition="inflammatory bowel disease",
        max_papers=100,
        primary_model="deepseek-llm:7b-chat",
        secondary_model="llama3.1:8b",
        state_file="large_study_ibd_state.json"
    )
    
    success = pipeline.run()
    print(f"Pipeline completed successfully: {success}")
    return success


def main():
    """Run examples"""
    print("Resumable Pipeline Usage Examples")
    print("=" * 50)
    
    # You can uncomment and run different examples
    
    # Basic usage
    # example_basic_usage()
    
    # Custom models
    # example_custom_models()
    
    # Large study (most likely to need resuming)
    # example_large_study()
    
    print("\nTo run the pipeline from command line, use:")
    print("python resumable_pipeline.py 'IBS' 50 --primary-model 'deepseek-llm:7b-chat' --secondary-model 'llama3.1:8b'")
    print("\nTo resume a stopped pipeline, just run the same command again.")
    print("To start fresh, add --clean flag.")


if __name__ == "__main__":
    main()