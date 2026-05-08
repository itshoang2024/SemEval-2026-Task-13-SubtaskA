# Checkpoint Reuse Runbook

Use this runbook when a Kaggle notebook version has already produced `_ckpt/*.npy` files and a later repo version wants to reuse only the compatible checkpoints.

Canonical scenario:

```text
Kaggle notebook v6 ran commit <commit-id> and produced /kaggle/working/_ckpt.
The repo changed after <commit-id>.
Kaggle notebook v7 should decide whether to reuse versioned `ppl_*_v9p.npy` or `sty_*_v9p.npy` files.
```

## Current resume behavior

`src/orchestrator.py` currently loads existing checkpoints only for these groups:

- `ppl_train_v9p.npy`, `ppl_test_v9p.npy`, `ppl_sample_v9p.npy`
- `sty_train_v9p.npy`, `sty_test_v9p.npy`, `sty_sample_v9p.npy`

During a fresh run, `sty_*` checkpoints are also reloaded after style extraction. This is an intentional memory boundary before the char vocabulary fit; it does not make stale style checkpoints safer to reuse across data or code changes.

It saves but does not currently reload these later files:

- `oof_v9p.npy`, `te_sum_v9p.npy`, `sa_sum_v9p.npy`
- `meta_te_v9p.npy`, `meta_sa_v9p.npy`

Copying saved-but-not-reloaded files into a new Kaggle version does not skip those phases unless the orchestrator is changed to load them.

## Safety rule

A checkpoint is reusable only if all upstream inputs and all code/config that define its values, row order, column order, and shape are compatible with the checkpoint-producing commit.

If uncertain, prefer this default:

```text
Reuse `ppl_*_v9p.npy` only when PPL logic and data are unchanged.
Recompute `sty_*_v9p.npy` unless style feature logic is definitely unchanged.
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

| Change since `<commit-id>` | Reuse `ppl_*_v9p.npy`? | Reuse `sty_*_v9p.npy`? | Reason |
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
| `LLMPerplexityEngine` tokenization, NLL calculation, quantization, compression fallback behavior, or model loading changed | No | No | PPL feature semantics changed. |
| `CodeStyleExtractor` changed | Yes | No | PPL is independent; style feature values/order changed. |
| Style/PPL merge logic in `src/orchestrator.py` changed | Maybe | No | PPL may still be valid, but `sty_*` column composition changed. |
| Base model, fold, vectorizer, SGD, or style HGB config changed | Yes | Yes | Current code reloads `ppl_*` and `sty_*`; later model stages will rerun. |
| Meta-learner or `OODRatioTuner` changed | Yes | Yes | Feature checkpoints are upstream and still valid. |
| Artifact detection changed only | Yes | Yes | It affects final forced labels, not PPL/style arrays. |

When a row says "Maybe", inspect the exact diff. If the diff changes how a loaded checkpoint is interpreted, recompute it.

## Dependency graph

```text
split data + row order
    + src/config.py: max_chars, ppl_*, seed
    + src/features.py: LLMPerplexityEngine
    + src/orchestrator.py: PPL checkpoint wiring
        -> ppl_train_v9p.npy, ppl_test_v9p.npy, ppl_sample_v9p.npy

split data + row order
    + src/config.py: max_chars
    + src/features.py: CodeStyleExtractor
    + matching ppl_*_v9p.npy and PPL feature order
    + src/orchestrator.py: style/PPL merge wiring
        -> sty_train_v9p.npy, sty_test_v9p.npy, sty_sample_v9p.npy
```

Important: `sty_*_v9p.npy` already contains style features plus appended PPL and language one-hot columns. Reusing `sty_*_v9p.npy` with newly computed or different `ppl_*_v9p.npy` can create inconsistent training signals.

## Validate copied checkpoints

Before running full inference in a new Kaggle version, inspect shapes:

```python
import os
import numpy as np

ckpt = "/kaggle/working/_ckpt"
for name in [
    "ppl_train_v9p.npy", "ppl_test_v9p.npy", "ppl_sample_v9p.npy",
    "sty_train_v9p.npy", "sty_test_v9p.npy", "sty_sample_v9p.npy",
]:
    path = os.path.join(ckpt, name)
    if os.path.exists(path):
        arr = np.load(path)
        print(name, arr.shape, arr.dtype)
```

Expected compatibility checks:

- `ppl_*_v9p.npy` second dimension must equal `len(LLMPerplexityEngine.FEATURE_NAMES)`; currently this is `8`.
- Each checkpoint row count must match its split: `train + validation`, `test`, or `test_sample`.
- If `sty_*` is reused, its column count must match the current style feature output after PPL columns are appended.

The current orchestrator does not enforce these checks automatically.

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

for name in ["ppl_train_v9p.npy", "ppl_test_v9p.npy", "ppl_sample_v9p.npy"]:
    src = os.path.join(src_ckpt, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_ckpt, name))
        print("copied", name)
    else:
        print("missing", name)
```

Copy PPL and style only when both groups are compatible:

```python
import glob
import os
import shutil

src_ckpt = glob.glob("/kaggle/input/**/_ckpt", recursive=True)[0]
dst_ckpt = "/kaggle/working/_ckpt"
os.makedirs(dst_ckpt, exist_ok=True)

approved = [
    "ppl_train_v9p.npy", "ppl_test_v9p.npy", "ppl_sample_v9p.npy",
    "sty_train_v9p.npy", "sty_test_v9p.npy", "sty_sample_v9p.npy",
]

for name in approved:
    src = os.path.join(src_ckpt, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(dst_ckpt, name))
        print("copied", name)
    else:
        print("missing", name)
```

Do not copy `oof_v9p.npy`, `te_sum_v9p.npy`, `sa_sum_v9p.npy`, `meta_te_v9p.npy`, or `meta_sa_v9p.npy` for resume purposes unless the orchestrator has been changed to validate and load them.

## Agent checklist

Before recommending checkpoint reuse, a coding agent must answer:

- What commit produced the Kaggle checkpoint? Record it as `<commit-id>`.
- Did the Kaggle data version and row order stay unchanged?
- Did any PPL-defining code/config change since `<commit-id>`?
- Did any style-defining code/config or style/PPL merge code change since `<commit-id>`?
- Are uncommitted local changes included in the diff review?
- Is the plan copying only approved checkpoint groups, not the whole `_ckpt` folder?

If any answer is unknown, do not claim reuse is safe. Use the conservative default: copy only compatible `ppl_*_v9p.npy` files, or recompute all checkpoints when PPL compatibility is uncertain.
