# CAMSP Architecture

This document describes the current source behavior of the CAMSP inference pipeline. Keep it synchronized with `src/orchestrator.py`, `src/features.py`, `src/tuning.py`, and `src/config.py`.

## System purpose

CAMSP detects AI-generated code for SemEval 2026 Task 13 Subtask A. It is an offline batch inference pipeline, not a service. Its primary runtime target is Kaggle, where it consumes competition parquet files and writes `submission.csv`.

## Runtime boundaries

- Batch runtime: `scripts/run_inference.py` starts one end-to-end process.
- Data boundary: parquet files are loaded from Kaggle inputs or from a local fallback under `data/`.
- Model boundary: the LLM perplexity stage tries local Kaggle model paths first, then the HuggingFace model id in `PipelineConfig.ppl_candidates`.
- Artifact boundary: expensive arrays are checkpointed as `.npy` files under `_ckpt`; final predictions are written as CSV and run metrics are written as JSON.
- No service/API boundary exists. The only public runtime output is `submission.csv`.

## Module boundaries

- `src/config.py` owns default hyperparameters, feature/model experiment flags, and tuning grids. Treat it as the source of truth for current numeric defaults.
- `src/data_utils.py` owns split discovery/loading, hard artifact detection, reproducibility seeding, and generator-family weights.
- `src/features.py` owns feature extraction. `CodeStyleExtractor` produces deterministic style features; `LLMPerplexityEngine` produces optional Qwen-based NLL features or zero matrices on fallback.
- `src/tuning.py` owns score rank-normalization, top-k ratio application, per-language shrinkage, and global-only ratio tuning.
- `src/orchestrator.py` owns the phase order and artifact wiring. It is the highest-risk file for accidental contract changes.
- `scripts/run_inference.py` owns Kaggle process bootstrapping and logging.

## Data flow

```text
train.parquet + validation.parquet
        |
        v
combined training frame --> generator-family weights --> 5-fold base models
        |                                               |
        |                                               v
        |                                      OOF base predictions
        |
        +--> style features ----------------------------+
        |                                               |
        +--> optional LLM PPL features -----------------+--> HGB meta-learner
                                                        |
test.parquet --> style/PPL/base predictions ------------+--> raw test scores
                                                        |
test_sample.parquet, if present --> sample scores ------+--> ratio tuning / optional score blend
                                                        |
artifact detector --------------------------------------+
                                                        v
                                                submission.csv
```

## Pipeline phases

1. Data loading: `DataIngestion.load_splits()` discovers and loads `train.parquet`, `validation.parquet`, `test.parquet`, and optional `test_sample.parquet`.
2. Training merge: `CAMSPipeline.run()` concatenates train and validation, then builds `y_train` and family weights from `generator`.
3. Artifact detection: hard artifact masks are computed for test and sample code; these force final predictions to machine-generated.
4. LLM perplexity: `LLMPerplexityEngine.execute()` processes test first, then sample, then a random train subsample with remaining budget. Missing CUDA/Transformers/model availability returns zero feature matrices.
5. Style features: `CodeStyleExtractor.extract_batch()` computes Style V1 compression, entropy, indentation, line, and repetition features by default. `CAMSP_STYLE_VERSION=v2` appends comment, identifier, operator, bracket, duplicate-line, long-line, and control-keyword statistics. LLM PPL columns are appended to style feature frames before checkpointing.
6. K-fold stacking: four base estimators produce out-of-fold train predictions and averaged test/sample predictions. `CAMSP_ENABLE_STYLE_ET=1` adds an optional fifth ExtraTrees style base model.
7. Meta-learner: an HGB classifier trains on base OOF predictions plus PPL features and scores test/sample rows.
8. Ratio tuning: `OODRatioTuner` tunes on `test_sample.parquet` when available. It chooses among global, artifact-forced global, and language-aware strategies only when both sample and test language labels are reliable. `CAMSP_ENABLE_SCORE_BLEND=1` can tune a convex post-meta blend with averaged base scores; the blend is accepted only when it improves sample F1 and keeps the test machine ratio within 3 percentage points of the non-blended ratio.
9. Submission: `submission.csv` is written with columns `ID,label`.

## Checkpoints

`src/orchestrator.py` writes `.npy` checkpoints for expensive arrays. In Kaggle, the directory is `/kaggle/working/_ckpt`; outside Kaggle, it is `/tmp/_ckpt`.

Checkpoint reuse is positional and assumes compatible row ordering, feature ordering, fold count, and dataset size. If any of those change, delete stale checkpoints before rerunning.

Style checkpoints are versioned: default Style V1 uses `sty_train.npy`, `sty_test.npy`, and `sty_sample.npy`; Style V2 uses `sty_train_v2.npy`, `sty_test_v2.npy`, and `sty_sample_v2.npy`.

`meta_te.npy` and `meta_sa.npy` can be reused for tuning-only experiments by setting `CAMSP_TUNING_ONLY=1` or opportunistically with `CAMSP_REUSE_META_SCORES=1`. `meta_base_models.npy`, when present, must match the currently enabled base models. This skips PPL, style extraction, stacking, and meta-learning after data/artifact loading. Score-blending tuning-only runs additionally require compatible `te_sum.npy` and `sa_sum.npy`.

`run_metrics.json` is written to the output directory and records load mode, style version, enabled base models, checkpoint usage, PPL coverage, tuning/blend config, sample F1 when available, machine ratio, and runtime.

## Current limitations

- There is no test suite or CI config in this repository.
- The hard pipeline deadline constant exists in `src/orchestrator.py`, but `_check_deadline()` is currently disabled.
- `environment.yml` supports local setup, but the Kaggle inference path also depends on runtime packages such as PyTorch, Transformers, and BitsAndBytes.
- `PipelineConfig.data_dir` is populated after discovery and is not currently used as a manual override.
- The local download helper extracts to `data/raw/`, while the inference local fallback checks for split parquet files directly under `data/`.

## Change impact notes

- Data schema changes must update `docs/contracts/data-and-artifacts.md` and all affected loading/tuning logic.
- Any change to feature order or checkpoint names should invalidate or migrate existing `_ckpt/*.npy` files.
- Any change to base-model order or count should invalidate `oof.npy`, `te_sum.npy`, `sa_sum.npy`, and `meta_*` checkpoints.
- Any change to `test.parquet` language handling must preserve the global-only fallback unless competition data is confirmed to always include `language`.
- Any change to LLM model loading should document whether zero-feature fallback remains acceptable.
- Any change to the final CSV shape must be reflected in the contract doc and Kaggle runbook.
