#!/usr/bin/env python3
"""
CAMSP 05_v9 parity Kaggle inference entrypoint.

Usage (single Kaggle notebook cell):
    !pip install bitsandbytes -q
    %cd /kaggle/working
    !git clone https://github.com/gugOfBoat/SemEval-2026-Task-13-SubtaskA.git
    %cd SemEval-2026-Task-13-SubtaskA
    !python scripts/run_inference.py
"""

import logging
import subprocess
import sys

# Ensure bitsandbytes is available for NF4 quantization
try:
    import bitsandbytes
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],
        check=False, capture_output=True,
    )

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# Add src/ to path for package imports
sys.path.insert(0, ".")

from src.orchestrator import CAMSPipeline


def main():
    pipeline = CAMSPipeline()
    submission = pipeline.run()
    print(f"\n{'='*60}")
    print(f"  CAMSP 05_v9 Parity Pipeline Complete")
    print(f"  Predictions: {len(submission):,} rows")
    print(f"  Machine ratio: {submission['label'].mean():.2%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
