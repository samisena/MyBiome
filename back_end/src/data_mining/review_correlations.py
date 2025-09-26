#!/usr/bin/env python3
"""
Launcher script for the correlation review tool.
Run this from the back_end directory.
"""

import sys
from pathlib import Path

from back_end.src.utils.review_correlations import main

if __name__ == "__main__":
    main()