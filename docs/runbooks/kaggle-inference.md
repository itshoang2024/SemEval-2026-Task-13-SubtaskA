# Kaggle Inference Runbook

Use this runbook for the primary competition workflow.

## Prerequisites

- Kaggle notebook or Kaggle script runtime.
- Competition data added as an input source.
- GPU runtime recommended. The LLM perplexity stage requires CUDA; without it, the pipeline falls back to zero LLM features.
- Internet enabled if the repository or HuggingFace model needs to be downloaded.
- Optional Kaggle model input for Qwen2.5-Coder. `PipelineConfig.ppl_candidates` tries local Kaggle paths before `Qwen/Qwen2.5-Coder-0.5B-Instruct`.

## Run

Default 4-bit PPL mode:

```python
!pip install bitsandbytes -q

%cd /kaggle/working
!rm -rf SemEval-2026-Task-13-SubtaskA
!git clone https://github.com/gugOfBoat/SemEval-2026-Task-13-SubtaskA.git

%cd SemEval-2026-Task-13-SubtaskA
!python scripts/run_inference.py
```

`scripts/run_inference.py` also attempts to install `bitsandbytes` if importing it fails and `CAMSP_PPL_LOAD_MODE` is unset or set to `4bit`.

Full FP16 PPL mode:

```python
%cd /kaggle/working
!rm -rf SemEval-2026-Task-13-SubtaskA
!git clone https://github.com/gugOfBoat/SemEval-2026-Task-13-SubtaskA.git

%cd SemEval-2026-Task-13-SubtaskA
%env CAMSP_PPL_LOAD_MODE=fp16
!python scripts/run_inference.py
```

Supported `CAMSP_PPL_LOAD_MODE` values are `4bit`, `fp16`, `bf16`, and `fp32`. Use `fp16` for the full-weight Kaggle benchmark unless you have a specific reason to test `fp32`.

Cheap feature/model experiments should keep `CAMSP_PPL_LOAD_MODE=4bit` and reuse only compatible `ppl_*.npy` from the baseline run:

```python
%env CAMSP_STYLE_VERSION=v2
!python scripts/run_inference.py
```

To add the optional fifth style base model:

```python
%env CAMSP_STYLE_VERSION=v2
%env CAMSP_ENABLE_STYLE_ET=1
!python scripts/run_inference.py
```

## Expected outputs

- `/kaggle/working/submission.csv`
- `/kaggle/working/_ckpt/*.npy`
- `/kaggle/working/run_metrics.json`
- Logs for all major phases: data loading, LLM perplexity, style extraction, 5-fold stacking, meta-learner, ratio tuning, and submission writing.

The logs and metrics JSON should include PPL load mode, checkpoint usage, PPL coverage, sample F1 when available, machine ratio, and total runtime.

## Resume behavior

The orchestrator loads existing checkpoints from `/kaggle/working/_ckpt/` when present. This can save time after an interrupted run.

When reusing checkpoints from a previous Kaggle notebook version or from a different repo commit, follow `docs/runbooks/checkpoint-reuse.md`. Do not copy the whole `_ckpt` folder by default; copy only the checkpoint groups that are compatible with the current code and data.

For tuning-only experiments after a full run has produced `meta_te.npy` and `meta_sa.npy`, use:

```python
%env CAMSP_TUNING_ONLY=1
!python scripts/run_inference.py
```

This requires valid `meta_te.npy` and, when `test_sample.parquet` exists, valid `meta_sa.npy` in `/kaggle/working/_ckpt/`. It skips PPL, style extraction, stacking, and meta-learning, then reruns ratio tuning, writes `submission.csv`, and exports `run_metrics.json`.

For score-blending-only experiments, also copy compatible `te_sum.npy`, `sa_sum.npy`, and `meta_base_models.npy` from the same full run. Keep the same `CAMSP_ENABLE_STYLE_ET` setting as the source run, then use:

```python
%env CAMSP_TUNING_ONLY=1
%env CAMSP_ENABLE_SCORE_BLEND=1
!python scripts/run_inference.py
```

The orchestrator validates `te_sum/sa_sum` shape against the enabled base-model count. It applies a blend only if `test_sample` Macro F1 beats the non-blended meta score and the test machine ratio stays within 3 percentage points of the non-blended ratio.

If you want to reuse meta scores when present but fall back to a full run when they are missing:

```python
%env CAMSP_REUSE_META_SCORES=1
!python scripts/run_inference.py
```

Delete `/kaggle/working/_ckpt/` before rerunning if any of these changed:

- input data or row order;
- feature extraction code;
- LLM feature names or dimensions;
- fold count or base-model count;
- `CAMSP_STYLE_VERSION` or `CAMSP_ENABLE_STYLE_ET`;
- checkpoint names;
- model/tuning code that changes score interpretation.

## Validation after completion

Run these checks in a Kaggle cell:

```python
import pandas as pd

sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.head())
print(sub.shape)
print(sub["label"].value_counts(dropna=False))

assert list(sub.columns) == ["ID", "label"]
assert set(sub["label"].unique()).issubset({0, 1})
```

Also compare `len(sub)` with the competition `test.parquet` row count if the test file is easy to locate in the notebook.

Inspect run metrics:

```python
import json

with open("/kaggle/working/run_metrics.json") as f:
    metrics = json.load(f)

print({
    "ppl_mode": metrics.get("ppl_load_mode"),
    "style_version": metrics.get("style_version"),
    "base_models": metrics.get("base_models"),
    "checkpoint_usage": metrics.get("checkpoint_usage"),
    "blend": metrics.get("tuning", {}).get("blend"),
    "sample_f1": metrics.get("sample_f1"),
    "machine_ratio": metrics.get("machine_ratio"),
    "total_minutes": metrics.get("total_minutes"),
})
```

## Common issues

- Data not found: ensure the competition data is attached as a Kaggle input. The loader searches known paths and then recursively scans `/kaggle/input` for `train.parquet`.
- No CUDA: the LLM stage logs a warning and returns zero PPL features. The rest of the pipeline continues.
- Transformers/model unavailable: the LLM stage logs a warning or model-load failures and returns zero PPL features.
- Missing `test_sample.parquet`: ratio tuning falls back to `PipelineConfig.fallback_global_ratio`; no sample F1 is logged.
- Missing `language` in `test.parquet`: expected for current code. The orchestrator retunes and applies a global ratio.
- Stale checkpoints: delete `_ckpt/` when behavior looks inconsistent after code or data changes.
