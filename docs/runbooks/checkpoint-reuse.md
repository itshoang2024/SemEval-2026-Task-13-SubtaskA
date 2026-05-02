# Checkpoint Reuse Runbook

Use this runbook when a Kaggle notebook version has already produced `_ckpt/*.npy` files and a later repo version wants to reuse only the compatible checkpoints.

Canonical scenario:

```text
Kaggle notebook v6 ran commit <commit-id> and produced /kaggle/working/_ckpt.
The repo changed after <commit-id>.
Kaggle notebook v7 should decide whether to reuse ppl_*.npy or sty_*.npy.
```

## Current resume behavior

`src/orchestrator.py` currently loads existing checkpoints automatically for these groups:

- `ppl_train.npy`, `ppl_test.npy`, `ppl_sample.npy`
- Style v1: `sty_train.npy`, `sty_test.npy`, `sty_sample.npy`
- Style v2: `sty_train_v2.npy`, `sty_test_v2.npy`, `sty_sample_v2.npy`

It can also load these final score checkpoints when `CAMSP_REUSE_META_SCORES=1` or `CAMSP_TUNING_ONLY=1`:

- `meta_te.npy`, `meta_sa.npy`
- `meta_base_models.npy` when present, to validate the base-model names that produced `meta_*`

It loads these fold-accumulator files only when `CAMSP_ENABLE_SCORE_BLEND=1` and validates their shape against the currently enabled base-model count:

- `te_sum.npy`, `sa_sum.npy`

It saves but does not currently reload `oof.npy`. Copying `oof.npy` into a new Kaggle version does not skip any phase.

## Safety rule

A checkpoint is reusable only if all upstream inputs and all code/config that define its values, row order, column order, and shape are compatible with the checkpoint-producing commit.

If uncertain, prefer this default:

```text
Reuse ppl_*.npy only when PPL logic and data are unchanged.
Recompute sty_*.npy unless style feature version and style/PPL merge logic are definitely unchanged.
Never copy all of _ckpt by default.
```

## Compare current code with the checkpoint commit

Set the old commit id from the Kaggle version that produced `_ckpt`:

```powershell
$BASE = "<commit-id>"
git diff --name-only "$BASE..HEAD"
git diff --stat "$BASE..HEAD"
git status --short
```

Include uncommitted changes in the decision. If there are uncommitted edits, inspect them too:

```powershell
git diff -- src
git diff --cached -- src
```

For targeted review:

```powershell
git diff "$BASE..HEAD" -- src/config.py src/features.py src/orchestrator.py src/data_utils.py src/tuning.py
```

## Decision table

| Change since `<commit-id>` | Reuse `ppl_*.npy`? | Reuse `sty_*.npy`? | Reason |
|---|---:|---:|---|
| Docs, comments, logging, README-only changes | Yes | Yes | No checkpoint semantics changed. |
| `scripts/run_inference.py` bootstrap changes only | Yes | Yes | The loaded arrays are interpreted the same way. |
| Dataset source changed, split files changed, or row order changed | No | No | Checkpoints are positional arrays. |
| `src/data_utils.py` loading behavior changed in a way that can reorder/filter rows | No | No | Row alignment may be invalid. |
| `PipelineConfig.max_chars` changed | No | No | It affects both PPL truncation and style compression inputs. |
| `PipelineConfig.ppl_candidates` changed to a different underlying model/tokenizer | No | No | PPL values come from a different LLM distribution. |
| `PipelineConfig.ppl_load_mode` or `CAMSP_PPL_LOAD_MODE` changed | No | No | PPL values may shift between 4-bit and full-weight inference. |
| `ppl_max_tokens`, `ppl_train_subsample`, `seed`, or PPL completion/subsample policy changed | No | No | PPL shape may match, but values/coverage/row positions can differ. |
| `ppl_batch_size` or `ppl_time_budget_sec` changed only for runtime speed | Usually yes | Usually yes | These should not change completed feature values, but verify old coverage is acceptable. |
| `LLMPerplexityEngine.FEATURE_NAMES` changed | No | No | Column count/order changed; `sty_*` also embeds PPL columns. |
| `LLMPerplexityEngine` tokenization, NLL calculation, quantization, fallback-zero behavior, or model loading changed | No | No | PPL feature semantics changed. |
| `CodeStyleExtractor` changed | Yes | No | PPL is independent; style feature values/order changed. |
| `CAMSP_STYLE_VERSION` changed between `v1` and `v2` | Yes | No | Style checkpoint names and column counts differ; recompute style and downstream model scores. |
| Style/PPL merge logic in `src/orchestrator.py` changed | Maybe | No | PPL may still be valid, but `sty_*` column composition changed. |
| `CAMSP_ENABLE_STYLE_ET` or base-model count changed | Yes | Yes | Feature checkpoints are upstream, but `oof`, `te_sum`, `sa_sum`, and `meta_*` shapes are incompatible. |
| Base model, fold, vectorizer, SGD, style HGB, or ExtraTrees config changed | Yes | Yes | Current code reloads `ppl_*` and compatible `sty_*`; later model stages will rerun. |
| Meta-learner changed | Yes | Yes | Feature checkpoints are upstream, but `meta_*` must be recomputed. |
| `OODRatioTuner` changed only | Yes | Yes | Reuse `meta_te.npy` and `meta_sa.npy` with `CAMSP_TUNING_ONLY=1` if meta scores are compatible. |
| Score blending changed only | Yes | Yes | Reuse compatible `meta_*` plus `te_sum.npy` and `sa_sum.npy` when their column count matches enabled base models. |
| Artifact detection changed only | Yes | Yes | It affects final forced labels, not PPL/style arrays. |

When a row says "Maybe", inspect the exact diff. If the diff changes how a loaded checkpoint is interpreted, recompute it.

## Dependency graph

```text
split data + row order
    + src/config.py: max_chars, ppl_*, seed
    + src/features.py: LLMPerplexityEngine
    + src/orchestrator.py: PPL checkpoint wiring
        -> ppl_train.npy, ppl_test.npy, ppl_sample.npy

split data + row order
    + src/config.py: max_chars
    + src/config.py: CAMSP_STYLE_VERSION / style_version
    + src/features.py: CodeStyleExtractor
    + matching ppl_*.npy and PPL feature order
    + src/orchestrator.py: style/PPL merge wiring
        -> sty_train.npy, sty_test.npy, sty_sample.npy        (v1)
        -> sty_train_v2.npy, sty_test_v2.npy, sty_sample_v2.npy (v2)
```

Important: `sty_*.npy` already contains style features plus appended PPL columns. Reusing `sty_*` with newly computed or different `ppl_*` can create inconsistent training signals.

For tuning-only experiments:

```text
compatible meta_te.npy + compatible meta_sa.npy
    + compatible meta_base_models.npy when present
    + src/tuning.py changes only
    + unchanged test/sample row order
        -> rerun with CAMSP_TUNING_ONLY=1

compatible meta_te.npy + compatible meta_sa.npy
    + compatible meta_base_models.npy when present
    + compatible te_sum.npy + compatible sa_sum.npy
    + unchanged base-model count/order
    + src/tuning.py or score-blend changes only
        -> rerun with CAMSP_TUNING_ONLY=1 and CAMSP_ENABLE_SCORE_BLEND=1
```

## Validate copied checkpoints

Before running full inference in a new Kaggle version, inspect shapes:

```python
import os
import numpy as np

ckpt = "/kaggle/working/_ckpt"
for name in [
    "ppl_train.npy", "ppl_test.npy", "ppl_sample.npy",
    "sty_train.npy", "sty_test.npy", "sty_sample.npy",
    "sty_train_v2.npy", "sty_test_v2.npy", "sty_sample_v2.npy",
    "meta_te.npy", "meta_sa.npy", "meta_base_models.npy",
    "te_sum.npy", "sa_sum.npy",
]:
    path = os.path.join(ckpt, name)
    if os.path.exists(path):
        arr = np.load(path)
        print(name, arr.shape, arr.dtype)
```

Expected compatibility checks:

- `ppl_*` second dimension must equal `len(LLMPerplexityEngine.FEATURE_NAMES)`; currently this is `5`.
- Each checkpoint row count must match its split: `train + validation`, `test`, or `test_sample`.
- If `sty_*` is reused, its file names and column count must match the current `CAMSP_STYLE_VERSION` after PPL columns are appended.
- If `meta_base_models.npy` exists, its names must match the currently enabled base models exactly.
- If `CAMSP_ENABLE_SCORE_BLEND=1`, `te_sum.npy` and `sa_sum.npy` must have columns equal to the currently enabled base model count: `4` by default, `5` with `CAMSP_ENABLE_STYLE_ET=1`.

The current orchestrator validates `meta_*` row counts and `te_sum/sa_sum` shapes for score blending. It does not fully validate PPL/style feature semantics, so the commit-diff review remains required.

## Copy only approved checkpoint groups

Kaggle notebook outputs mounted through "Add Data -> Notebook Output Files" are read-only under `/kaggle/input`. Copy approved checkpoint files into `/kaggle/working/_ckpt` before running `scripts/run_inference.py`.

Copy PPL only:

```python
import glob
import os
import shutil

src_ckpt = glob.glob("/kaggle/input/**/_ckpt", recursive=True)[0]
dst_ckpt = "/kaggle/working/_ckpt"
os.makedirs(dst_ckpt, exist_ok=True)

for name in ["ppl_train.npy", "ppl_test.npy", "ppl_sample.npy"]:
    src = os.path.join(src_ckpt, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_ckpt, name))
        print("copied", name)
    else:
        print("missing", name)
```

Copy PPL and style only when both groups are compatible. Choose the style files that match the current `CAMSP_STYLE_VERSION`:

```python
import glob
import os
import shutil

src_ckpt = glob.glob("/kaggle/input/**/_ckpt", recursive=True)[0]
dst_ckpt = "/kaggle/working/_ckpt"
os.makedirs(dst_ckpt, exist_ok=True)

approved = [
    "ppl_train.npy", "ppl_test.npy", "ppl_sample.npy",
    "sty_train.npy", "sty_test.npy", "sty_sample.npy",
]

for name in approved:
    src = os.path.join(src_ckpt, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_ckpt, name))
        print("copied", name)
    else:
        print("missing", name)
```

For Style V2, replace the style names with:

```python
["sty_train_v2.npy", "sty_test_v2.npy", "sty_sample_v2.npy"]
```

Copy meta scores only for tuning-only experiments:

```python
import glob
import os
import shutil

src_ckpt = glob.glob("/kaggle/input/**/_ckpt", recursive=True)[0]
dst_ckpt = "/kaggle/working/_ckpt"
os.makedirs(dst_ckpt, exist_ok=True)

for name in ["meta_te.npy", "meta_sa.npy", "meta_base_models.npy"]:
    src = os.path.join(src_ckpt, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_ckpt, name))
        print("copied", name)
    else:
        print("missing", name)
```

For score-blending-only experiments, also copy compatible base-score accumulators:

```python
for name in ["te_sum.npy", "sa_sum.npy"]:
    src = os.path.join(src_ckpt, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_ckpt, name))
        print("copied", name)
    else:
        print("missing", name)
```

Do not copy `oof.npy` for resume purposes.

## Agent checklist

Before recommending checkpoint reuse, a coding agent must answer:

- What commit produced the Kaggle checkpoint? Record it as `<commit-id>`.
- Did the Kaggle data version and row order stay unchanged?
- Did any PPL-defining code/config change since `<commit-id>`?
- Did any style-defining code/config or style/PPL merge code change since `<commit-id>`?
- Did `CAMSP_STYLE_VERSION`, `CAMSP_ENABLE_STYLE_ET`, or the base-model order/count change?
- If score blending is enabled, do `te_sum.npy` and `sa_sum.npy` match the enabled base-model count?
- Are uncommitted local changes included in the diff review?
- Is the plan copying only approved checkpoint groups, not the whole `_ckpt` folder?

If any answer is unknown, do not claim reuse is safe. Use the conservative default: copy only compatible `ppl_*` files, or recompute all checkpoints when PPL compatibility is uncertain.
