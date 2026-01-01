#!/usr/bin/env python3
"""Test script for constant medium (smoke) rendering on GPU."""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from scenes import cornell_smoke

if __name__ == "__main__":
    print("Testing constant medium (smoke) on Taichi GPU renderer...")
    cornell_smoke()
    print("\nTest complete!")
