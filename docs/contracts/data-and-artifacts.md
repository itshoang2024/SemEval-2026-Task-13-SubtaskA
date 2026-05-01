# Data and Artifact Contract

This is the canonical contract for pipeline inputs, generated artifacts, and compatibility-sensitive files.

## Input discovery

`DataIngestion.discover_data_directory()` searches fixed Kaggle paths first, then recursively scans `/kaggle/input` for a directory containing `train.parquet`. Outside Kaggle, it checks for `data/train.parquet` relative to the repository.

Important local mismatch: `data/download_data.py` extracts competition files into `data/raw/`, but the inference fallback checks for split parquet files directly under `data/`. Local inference may require moving or copying the split files into the expected location.

`PipelineConfig.data_dir` is assigned after discovery; it is not currently a manual discovery override.

## Required split files

- `train.parquet`: required.
- `validation.parquet`: required.
- `test.parquet`: required.
- `test_sample.parquet`: optional, used for ratio tuning and sample F1 logging when present.

## Required columns

`train.parquet` and `validation.parquet`:

- `code`: source code text. Non-string values are converted to empty or string values in downstream feature stages.
- `label`: binary class label. `1` means machine-generated code and `0` means human-written code.
- `generator`: generator identifier used by `GeneratorFamilyEncoder.build_weights()`.

`test.parquet`:

- `code`: source code text.
- `ID` or `id`: row identifier. The output CSV always uses `ID`.
- `language`: optional. If absent, all rows are treated as `Unknown` and test prediction uses global-only ratio tuning.

`test_sample.parquet`, when present:

- `code`: source code text.
- `label`: binary class label for tuning/evaluation.
- `language`: optional. If absent, sample rows are treated as `Unknown`.

## Label semantics

- `1`: machine-generated code.
- `0`: human-written code.

Hard artifacts detected in test/sample code force prediction `1`. Current artifact signals include configured LLM special tokens, markdown code fences, and assistant-style opening phrases.

## Generated checkpoints

Checkpoint directory:

- Kaggle: `/kaggle/working/_ckpt/`
- Non-Kaggle: `/tmp/_ckpt/`

Current checkpoint names:

- `ppl_train.npy`, `ppl_test.npy`, `ppl_sample.npy`: LLM perplexity feature matrices.
- `sty_train.npy`, `sty_test.npy`, `sty_sample.npy`: style feature matrices after PPL columns are appended.
- `oof.npy`: out-of-fold base-model predictions for the combined train+validation set.
- `te_sum.npy`: accumulated test base-model predictions across folds.
- `sa_sum.npy`: accumulated sample base-model predictions across folds.
- `meta_te.npy`, `meta_sa.npy`: meta-learner scores.
- `run_metrics.json`: run metadata, checkpoint usage, PPL coverage, tuning config, sample F1 when available, machine ratio, and runtime. Written to `/kaggle/working/run_metrics.json` on Kaggle unless `CAMSP_METRICS_PATH` overrides it.

Checkpoint resume behavior:

- `ppl_*.npy` and `sty_*.npy` are loaded automatically when present.
- `meta_te.npy` and `meta_sa.npy` are loaded only when `CAMSP_REUSE_META_SCORES=1` or `CAMSP_TUNING_ONLY=1`.
- `oof.npy`, `te_sum.npy`, and `sa_sum.npy` are saved for recovery/debugging but are not currently loaded to skip stacking.

Checkpoint compatibility depends on:

- dataset row order and row count;
- feature order and feature count;
- fold count and base-model count;
- checkpoint file names;
- the current code version's interpretation of zero-filled fallback features.

Delete stale `_ckpt/*.npy` files before rerunning after incompatible changes.

## Final output contract

`submission.csv` is written to `/kaggle/working/submission.csv` on Kaggle and to `./submission.csv` outside Kaggle.

Columns:

- `ID`: copied from `test.parquet` column `ID` when present, otherwise from `id`.
- `label`: integer prediction, with `1` meaning machine-generated and `0` meaning human-written.

The CSV is written without an index.

## Failure modes

- Missing split files cause data loading to fail.
- Missing `generator` in train/validation causes family-weight construction to fail.
- Missing `label` in train/validation or sample tuning data causes model training or tuning to fail.
- Missing both `ID` and `id` in test causes submission creation to fail.
- Missing CUDA, Transformers, or a loadable LLM does not fail the pipeline; it produces zero LLM features.
- Missing `test_sample.parquet` falls back to `PipelineConfig.fallback_global_ratio`.

## Validation checklist

Before a full run:

- Confirm the discovered data directory contains required split files.
- Confirm train/validation have `code`, `label`, and `generator`.
- Confirm test has `code` and either `ID` or `id`.
- If using sample tuning, confirm sample has `code` and `label`.

After a run:

- Confirm `submission.csv` exists.
- Confirm `run_metrics.json` exists.
- Confirm output columns are exactly `ID,label`.
- Confirm output row count equals `len(test.parquet)`.
- Confirm `label` values are only `0` and `1`.
- Confirm logs report the expected machine ratio and, when sample exists, sample F1.
