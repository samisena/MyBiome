#!/usr/bin/env python3
"""
Launcher script for the correlation review tool.
Run this from the back_end directory.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.action_functions.review_correlations import main

if __name__ == "__main__":
    main()