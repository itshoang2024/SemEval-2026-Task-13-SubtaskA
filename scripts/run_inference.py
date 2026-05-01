#!/usr/bin/env python3
"""
CAMSP v10 — Kaggle Inference Entrypoint.

Usage (single Kaggle notebook cell):
    !pip install bitsandbytes -q
    %cd /kaggle/working
    !git clone https://github.com/gugOfBoat/SemEval-2026-Task-13-SubtaskA.git
    %cd SemEval-2026-Task-13-SubtaskA
    !python scripts/run_inference.py
"""

import logging
import os
import subprocess
import sys

# Ensure bitsandbytes is available only when NF4 quantization is requested.
if os.getenv("CAMSP_PPL_LOAD_MODE", "4bit").lower() == "4bit":
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
    print(f"  CAMSP v10 Pipeline Complete")
    print(f"  Predictions: {len(submission):,} rows")
    print(f"  Machine ratio: {submission['label'].mean():.2%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
