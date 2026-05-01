# Local Data Download Runbook

Use this runbook to download the Kaggle competition data locally for inspection or development.

## Prerequisites

- Conda or compatible environment manager.
- Kaggle account with API credentials.
- Access to the SemEval 2026 Task 13 Subtask A competition data.

Do not commit `.env` or `kaggle.json`; both are ignored by `.gitignore`.

## Environment setup

```powershell
conda env create -f environment.yml
conda activate semeval2026
```

If the environment already exists:

```powershell
conda env update -f environment.yml --prune
conda activate semeval2026
```

## Credential setup

Create a local `.env` from the example and fill in real credentials:

```powershell
Copy-Item .env.example .env
```

Expected variables:

```text
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

`data/download_data.py` loads `.env` with `python-dotenv` and also sets `KAGGLE_CONFIG_DIR` to the current working directory so the Kaggle CLI can find local credentials.

## Download

From the repository root:

```powershell
python data/download_data.py
```

To override the competition slug:

```powershell
python data/download_data.py --slug sem-eval-2026-task-13-subtask-a
```

The script:

1. creates `data/raw/` and `data/processed/`;
2. runs `kaggle competitions download -c <slug> -p data/raw`;
3. extracts the downloaded zip into `data/raw/`;
4. deletes the zip file after extraction.

## Important local inference note

The downloader extracts into `data/raw/`, but `DataIngestion.discover_data_directory()` currently checks for local split files directly under `data/`, for example `data/train.parquet`.

For local inference, either run on Kaggle as documented in `docs/runbooks/kaggle-inference.md` or place the required split parquet files where the loader expects them. Do not rely on `PipelineConfig.data_dir` as an override without changing the loader; it is currently assigned after discovery.

## Validation

After download:

```powershell
Get-ChildItem -Recurse data\raw
```

Confirm the extracted files include the expected SemEval split parquet files or a nested directory containing them.

Before local pipeline experimentation, confirm the effective data directory contains:

- `train.parquet`
- `validation.parquet`
- `test.parquet`
- optional `test_sample.parquet`

See `docs/contracts/data-and-artifacts.md` for required columns.
