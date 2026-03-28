"""
Data Ingestion Script for SemEval-2026 Task 13 (GenAI Code Detection & Attribution).

This script:
1. Sets up the standard directory structure (`data/raw/`, `data/processed/`).
2. Configures logging to track progress and handle errors.
3. Downloads the official SemEval-2026 Task 13 dataset via the Kaggle API.
4. Extracts downloaded archives directly into `data/raw/`.

Usage:
    python src/data/download_data.py
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv

# ── LOGGING SETUP ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load credentials from .env to environment variables
# KaggleApi automatically picks up KAGGLE_USERNAME and KAGGLE_KEY
load_dotenv()

# Tell Kaggle API to look for kaggle.json in the current working directory
# as a fallback if .env is missing.
os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError as e:
    logger.error("Kaggle package is not installed. Please run `pip install kaggle`.")
    raise SystemExit(1) from e

# ── PATHS AND CONSTANTS ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_COMPETITION = "sem-eval-2026-task-13-subtask-a"


def create_directories() -> None:
    """Ensure standard data directories exist."""
    try:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory structure initialized at {RAW_DATA_DIR.parent.resolve()}")
    except OSError as e:
        logger.error(f"Failed to create directories: {e}")
        raise


def download_data(slug: str) -> None:
    """Download data from Kaggle Competition straight from CLI into data/raw/."""
    create_directories()
    
    # 1. Provide Config Context for Kaggle
    os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
    
    logger.info(f"Starting CLI download for competition: '{slug}'...")
    
    # 2. Call Kaggle CLI directly (bypasses python-lib bugs)
    cmd = f"kaggle competitions download -c {slug} -p {RAW_DATA_DIR}"
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        logger.error(f"Download failed out with Exit Code {exit_code}.")
        logger.error("Make sure your API Token in kaggle.json is correct.")
        raise SystemExit(1)
        
    logger.info("Download completed via CLI.")

    # 3. Extract automatically
    zip_path = RAW_DATA_DIR / f"{slug}.zip"
    if zip_path.exists():
        logger.info("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        logger.info(f"Files extracted to: {RAW_DATA_DIR}")
        zip_path.unlink() # remove the zip after extraction
        
    logger.info("Data ingestion completed successfully.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the SemEval-2026 dataset from Kaggle.")
    parser.add_argument("--slug", default=DEFAULT_COMPETITION, help="Kaggle competition slug.")
    args = parser.parse_args()
    
    download_data(args.slug)

if __name__ == "__main__":
    main()
