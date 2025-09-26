#!/usr/bin/env python3
"""
Non-interactive script to reprocess abstracts.
"""

import sys
from pathlib import Path

from back_end.src.utils.analyze_reviews import AbstractReprocessor

def main():
    """Run the reprocessor without user input."""
    print("MyBiome Abstract Reprocessor")
    print("============================")
    print("Re-processing XML files with fixed abstract parser")
    print()
    
    reprocessor = AbstractReprocessor()
    
    # Show current state
    print("Current abstract state:")
    reprocessor.check_sample_improvements()
    
    print("\nStarting reprocessing...")
    
    # Run reprocessing
    reprocessor.reprocess_all_files()
    
    # Show final state
    print("\nFinal abstract state:")
    reprocessor.check_sample_improvements()

if __name__ == "__main__":
    main()