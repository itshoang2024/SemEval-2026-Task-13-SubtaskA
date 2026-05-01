# CAMSP v10.2 - Compression-Aware Meta-Stacking Pipeline

> **SemEval 2026 Task 13 Subtask A**: AI-Generated Code Detection
> *Detecting machine-generated code across 8+ programming languages with out-of-distribution resilience*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-Academic-green)

---

## Executive Summary

**CAMSP** treats AI code detection as a **code forensics** problem. Instead of relying only on vocabulary-based features that can break on unseen programming languages, CAMSP uses language-agnostic regularity signals: compression ratios, byte entropy, indentation dynamics, and LLM perplexity.

### Out-of-Distribution (OOD) Resilience

The test set contains unseen languages not present in training. CAMSP's defense strategy:

| Signal | Why It Works on OOD |
|--------|---------------------|
| Compression ratios (zlib, bz2) | Language-agnostic: measures byte-level regularity |
| Shannon byte entropy | Captures information density regardless of syntax |
| Indent delta entropy | AI models often produce mechanically consistent spacing |
| LLM perplexity (Qwen-0.5B) | Neural fingerprint of how expected the code is |
| Adaptive ratio shrinkage | Prevents collapse on low-confidence OOD subsets |

---

## Methodology - Four Pillars

### 1. Stacking Ensemble (4 Base Estimators)

```
char_full    ─┐
char_family  ─┤── 5-Fold OOF ──► HGB Meta-Learner ──► Final Score
word_hash    ─┤
style_hgb    ─┘
```


- **char_full**: Char-level `(3,6)`-gram TF-IDF + SGD logistic regression
- **char_family**: Same architecture, trained with inverse-sqrt family weights to balance generator diversity
- **word_hash**: Word `(1,3)`-gram hashing vectorizer (`2^20` features) + SGD
- **style_hgb**: Compression/entropy/style features fed into HistGradientBoosting

### 2. LLM Perplexity Engine (Sequential Completion Strategy)

Uses **Qwen2.5-Coder-0.5B** quantized to **NF4 4-bit** with BitsAndBytes when CUDA, Transformers, and a loadable model are available.

Current v10.1/v10.2 behavior:

- **Priority 1**: Process the test set first, targeting full completion before any lower-priority split.
- **Priority 2**: Process `test_sample.parquet` next when it exists.
- **Priority 3**: Use the remaining LLM time budget on a train subsample.
- **Fallback**: If CUDA, Transformers, or model loading is unavailable, the pipeline continues with zero-filled LLM features.

Current LLM defaults from `PipelineConfig`:

| Setting | Value |
|---------|-------|
| Max tokens | `128` |
| Batch size | `128` |
| Train subsample | `50,000` |
| LLM time budget | `25,200s` (`7h`) |

### 3. Adaptive Constraint Engine (OODRatioTuner)

Prevents ratio collapse, where the model labels too many OOD samples as human-written code:

- **Global ratio grid**: `0.05` to `0.50` inclusive, step `0.01`
- **Per-language ratio grid**: `0.02` to `0.50` inclusive, step `0.01`
- **Shrinkage interpolation**: `ratio = (1-s) * global + s * per_lang`
- **Shrinkage grid**: `{0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}`
- **Fallback global ratio**: `0.22`

v10.2 also handles the case where `test.parquet` has no `language` column: it re-tunes a global-only ratio on `test_sample.parquet` and applies that global ratio to the full test set.

### 4. Extended Compression Features

Beyond standard zlib ratio, CAMSP adds:

- `bz2_ratio` - Burrows-Wheeler block-sorting compression
- `byte_entropy` - Shannon entropy over raw byte distribution
- `indent_delta_entropy` - Entropy of indentation changes between lines
- `line_len_cv` - Coefficient of variation of line lengths
- `trigram_rep_ratio` - Character trigram repetition rate

---

## Repository Structure

```
SemEval-2026-Task-13-SubtaskA/
├── src/
│   ├── __init__.py           # Package marker
│   ├── config.py             # PipelineConfig dataclass (all hyperparams)
│   ├── data_utils.py         # DataIngestion, ArtifactDetector, GeneratorFamilyEncoder
│   ├── features.py           # CodeStyleExtractor, LLMPerplexityEngine
│   ├── tuning.py             # OODRatioTuner (adaptive shrinkage)
│   └── orchestrator.py       # CAMSPipeline (end-to-end runner)
├── scripts/
│   └── run_inference.py      # Kaggle entrypoint
├── data/
│   └── download_data.py      # Local Kaggle data download helper
├── environment.yml
└── README.md
```

---

## Kaggle Setup Guide

### Prerequisites

| Setting | Value |
|---------|-------|
| **GPU** | T4 x2 recommended |
| **Internet** | ON if cloning/downloading from remote sources |
| **Persistence** | Files only |

### Input Data

1. **Competition data**: `semeval-2026-task13-subtask-a`, auto-discovered by probing known Kaggle input paths and recursively walking `/kaggle/input/`.
2. **Model** (optional): Add `Qwen/Qwen2.5-Coder` version `0.5b-instruct` from Kaggle Models. If not available locally, the configured HuggingFace model id is tried.

### Run (Single Cell)

```python
!pip install bitsandbytes -q

%cd /kaggle/working
!rm -rf SemEval-2026-Task-13-SubtaskA
!git clone https://github.com/gugOfBoat/SemEval-2026-Task-13-SubtaskA.git

%cd SemEval-2026-Task-13-SubtaskA
!python scripts/run_inference.py
```

### Runtime Notes

| Phase | Behavior |
|-------|----------|
| LLM Perplexity | Test first, then sample, then train subsample until the LLM budget is exhausted |
| Style Features | Compression, entropy, indentation, line, and repetition metrics |
| 5-Fold Stacking | 4 base models across 5 stratified folds |
| Meta + Tuning | HGB meta-learner plus ratio search on `test_sample.parquet` when available |

### VRAM & Speed Notes

- Qwen-0.5B at NF4 4-bit is intended to fit comfortably on Kaggle GPU runtimes.
- Batch size defaults to `128`; the LLM engine halves batch size on CUDA OOM down to a minimum of `8`.
- All sparse matrices use CSR-compatible scikit-learn vectorizers.
- Expensive intermediate arrays are checkpointed to `/kaggle/working/_ckpt/` on Kaggle.

---

## Performance & Metrics

The historical CAMSP v10 README reported an internal `test_sample` baseline:

- **Sample Macro F1**: `0.7135`
- **Global OOD Ratio**: `0.10`
- **Adaptive Shrinkage**: `1.00`
- **Test Set Machine Predictions**: `10.06%` (`50,291 / 500,000`)
- **Total Execution Time**: `245.5m`

These numbers are retained as historical reference only. Current v10.2 defaults use a wider ratio grid (`0.05` to `0.50`), 128-token LLM windows, 128 batch size, checkpointed expensive arrays, and global-only retuning when the full test set has no `language` column.

### Historical Tuned Language Ratios

> Derived via 5-Fold OOF stack tuning in the earlier v10 run.

| Language | Ratio | Language | Ratio |
|----------|-------|----------|-------|
| **C#** | `25%` | **Java** | `17%` |
| **Go** | `26%` | **PHP** | `18%` |
| **Python** | `22%` | **JavaScript** | `39%` |
| **C++** | `8%` | **C** | `17%` |

---

## License

Academic use only - SemEval 2026 competition submission.
