# %% [markdown]
# # Phase 4: XGBoost GPU Training — SemEval-2026 Task 13 SubtaskA
# ## AI-Generated Code Detection (Full 500k Pipeline)
#
# **Designed for**: Google Colab with T4 GPU
#
# Pipeline:
# 1. Feature extraction (500k train + 100k val + 500k test)
# 2. XGBoost GPU training (5-fold CV)
# 3. Inference → submission.csv
#
# Usage on Colab:
# ```python
# !pip install xgboost tree-sitter tree-sitter-python tree-sitter-java tree-sitter-cpp
# # Upload data/raw/Task_A/*.parquet to /content/data/
# %run src/03_train_xgboost_gpu.py
# ```

# %%
import sys
import os
import re
import zlib
import math
import hashlib
import pickle
import warnings
import time
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

# -- Detect environment (Colab vs local) --
IN_COLAB = "google.colab" in sys.modules if "google.colab" in sys.modules else os.path.exists("/content")
if IN_COLAB:
    DATA_DIR = Path("/content/data")
    OUT_DIR = Path("/content/outputs")
else:
    DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "Task_A"
    OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
N_WORKERS = min(cpu_count(), 8)  # For parallel feature extraction

# -- 13 Selected Features (see FEATURE_SELECTION.md) --
SELECTED_FEATURES = [
    # Group A: Stylometric (8)
    "indent_consistency",
    "avg_line_length",
    "comment_to_code_ratio",
    "snake_ratio",
    "trailing_ws_ratio",
    "avg_identifier_length",
    "camel_ratio",
    "long_id_ratio",
    # Group B: Statistical (3)
    "shannon_entropy",
    "zlib_compression_ratio",
    "token_entropy",
    # Group C: Structural (2)
    "avg_ast_depth",
    "branch_ratio",
]

print(f"Selected features: {len(SELECTED_FEATURES)}")
print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUT_DIR}")
print(f"Workers: {N_WORKERS}")


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

# %% [markdown]
# ## 1. Feature Extraction Engine

# %%
# -- Compiled regex patterns (module-level for multiprocessing) --
SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
CAMEL_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b")
PASCAL_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
IDENTIFIER_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
SINGLE_COMMENT_RE = re.compile(r"(//.*|#(?!!).*)\s*$", re.MULTILINE)
BLOCK_COMMENT_RE = re.compile(r'/\*[\s\S]*?\*/|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
KEYWORD_SET = frozenset([
    "if", "else", "elif", "for", "while", "return", "def", "class", "import",
    "from", "try", "except", "catch", "finally", "with", "as", "yield",
    "break", "continue", "pass", "raise", "lambda", "async", "await",
    "public", "private", "protected", "static", "void", "int", "float",
    "double", "string", "bool", "boolean", "new", "throw", "throws",
    "switch", "case", "default", "struct", "enum", "interface", "extends",
    "implements", "package", "namespace", "using", "include", "const",
    "var", "let", "auto", "virtual", "override", "abstract", "final",
])

# -- Language detection heuristics (for test set which has no language col) --
LANG_PATTERNS = {
    "Python": re.compile(r"\bdef\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import|print\s*\(|if\s+__name__"),
    "Java": re.compile(r"\bpublic\s+(?:static\s+)?(?:void|class|int|String)|System\.out\.|import\s+java\."),
    "C++": re.compile(r"#include\s*<|cout\s*<<|cin\s*>>|std::|using\s+namespace\s+std|int\s+main\s*\("),
}


def detect_language(code: str) -> str:
    """Simple heuristic language detection."""
    scores = {}
    for lang, pattern in LANG_PATTERNS.items():
        scores[lang] = len(pattern.findall(code))
    if max(scores.values()) == 0:
        return "Python"  # Default (91.5% of training data)
    return max(scores, key=scores.get)


def extract_features(code: str, language: str = None) -> dict:
    """Extract all 13 selected features from a code snippet.

    This is the single entry point called per row.
    If language is None, auto-detect it.
    """
    if language is None:
        language = detect_language(code)

    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    n_lines = max(len(non_empty), 1)
    total_lines = max(len(lines), 1)

    # ── Stylometric ──────────────────────────────────────────────────
    # indent_consistency
    indent_lines = [l for l in non_empty if len(l) > 0 and l[0] in (" ", "\t")]
    if indent_lines:
        tab_count = sum(1 for l in indent_lines if l[0] == "\t")
        space_count = len(indent_lines) - tab_count
        indent_consistency = max(tab_count, space_count) / len(indent_lines)
    else:
        indent_consistency = 1.0

    # avg_line_length
    lengths = [len(l) for l in non_empty]
    avg_line_length = float(np.mean(lengths)) if lengths else 0.0

    # comment_to_code_ratio
    single_comments = len(SINGLE_COMMENT_RE.findall(code))
    block_comments = len(BLOCK_COMMENT_RE.findall(code))
    comment_to_code_ratio = (single_comments + block_comments) / n_lines

    # snake_ratio, camel_ratio
    snakes = len(SNAKE_RE.findall(code))
    camels = len(CAMEL_RE.findall(code))
    pascals = len(PASCAL_RE.findall(code))
    total_named = max(snakes + camels + pascals, 1)
    snake_ratio = snakes / total_named
    camel_ratio = camels / total_named

    # trailing_ws_ratio
    trailing_ws_ratio = sum(1 for l in lines if l != l.rstrip()) / total_lines

    # avg_identifier_length, long_id_ratio
    identifiers = IDENTIFIER_RE.findall(code)
    non_kw = [i for i in identifiers if i.lower() not in KEYWORD_SET]
    if non_kw:
        id_lens = [len(i) for i in non_kw]
        avg_identifier_length = float(np.mean(id_lens))
        long_id_ratio = sum(1 for l in id_lens if l >= 10) / len(non_kw)
    else:
        avg_identifier_length = 0.0
        long_id_ratio = 0.0

    # ── Statistical ──────────────────────────────────────────────────
    # shannon_entropy (character-level)
    if code:
        freq = Counter(code)
        n_chars = len(code)
        shannon_entropy = -sum((c / n_chars) * math.log2(c / n_chars)
                               for c in freq.values())
    else:
        shannon_entropy = 0.0

    # zlib_compression_ratio
    if code:
        encoded = code.encode("utf-8", errors="replace")
        compressed = zlib.compress(encoded, level=6)
        zlib_compression_ratio = len(compressed) / max(len(encoded), 1)
    else:
        zlib_compression_ratio = 0.0

    # token_entropy (word-level)
    tokens = re.findall(r"\b\w+\b", code)
    if tokens:
        freq_t = Counter(tokens)
        n_tok = len(tokens)
        token_entropy = -sum((c / n_tok) * math.log2(c / n_tok)
                             for c in freq_t.values())
    else:
        token_entropy = 0.0

    # ── Structural (AST) ─────────────────────────────────────────────
    avg_ast_depth, branch_ratio = _extract_ast_features(code, language)

    return {
        "indent_consistency": indent_consistency,
        "avg_line_length": avg_line_length,
        "comment_to_code_ratio": comment_to_code_ratio,
        "snake_ratio": snake_ratio,
        "trailing_ws_ratio": trailing_ws_ratio,
        "avg_identifier_length": avg_identifier_length,
        "camel_ratio": camel_ratio,
        "long_id_ratio": long_id_ratio,
        "shannon_entropy": shannon_entropy,
        "zlib_compression_ratio": zlib_compression_ratio,
        "token_entropy": token_entropy,
        "avg_ast_depth": avg_ast_depth,
        "branch_ratio": branch_ratio,
    }


# ── AST feature extraction ──────────────────────────────────────────
# We try tree-sitter first; if unavailable, fall back to regex.

_PARSERS = {}
_TS_OK = False

BRANCH_TYPES = frozenset([
    "if_statement", "elif_clause", "else_clause",
    "for_statement", "for_in_clause",
    "while_statement", "do_statement",
    "try_statement", "except_clause", "catch_clause",
    "switch_statement", "case_statement",
    "conditional_expression", "ternary_expression",
    "match_statement", "case_clause",
])


def _init_tree_sitter():
    """Initialize tree-sitter parsers (called once per process)."""
    global _PARSERS, _TS_OK
    if _TS_OK:
        return
    try:
        import tree_sitter_python as ts_python
        import tree_sitter_java as ts_java
        import tree_sitter_cpp as ts_cpp
        import tree_sitter as ts

        _PARSERS["Python"] = ts.Parser(ts.Language(ts_python.language()))
        _PARSERS["Java"] = ts.Parser(ts.Language(ts_java.language()))
        _PARSERS["C++"] = ts.Parser(ts.Language(ts_cpp.language()))
        _TS_OK = True
    except Exception:
        _TS_OK = False


def _extract_ast_features(code: str, language: str) -> tuple:
    """Return (avg_ast_depth, branch_ratio) using tree-sitter or regex."""
    _init_tree_sitter()

    if _TS_OK and language in _PARSERS:
        try:
            tree = _PARSERS[language].parse(code.encode("utf-8", errors="replace"))
            root = tree.root_node
            depths = []
            branch_count = 0
            total_nodes = 0
            stack = [(root, 0)]
            while stack:
                node, depth = stack.pop()
                total_nodes += 1
                depths.append(depth)
                if node.type in BRANCH_TYPES:
                    branch_count += 1
                for child in node.children:
                    stack.append((child, depth + 1))
            avg_depth = float(np.mean(depths)) if depths else 0.0
            br = branch_count / max(total_nodes, 1)
            return avg_depth, br
        except Exception:
            pass

    # Regex fallback
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return 0.0, 0.0
    indent_depths = []
    for l in non_empty:
        stripped = l.lstrip()
        indent = len(l) - len(stripped)
        indent_depths.append(indent / 4.0)
    branches = len(re.findall(
        r"\b(if|elif|else|for|while|try|except|catch|case|switch)\b", code
    ))
    n = max(len(non_empty), 1)
    return float(np.mean(indent_depths)), branches / n


# ── Parallel feature extraction wrapper ──────────────────────────────

def _extract_row(args):
    """Worker function for multiprocessing."""
    code, lang = args
    return extract_features(code, lang)


def extract_features_parallel(df, code_col="code", lang_col=None, n_workers=4):
    """Extract features from a DataFrame in parallel.

    Args:
        df: DataFrame with at least `code_col`
        code_col: column name for code
        lang_col: column name for language (None = auto-detect)
        n_workers: number of parallel workers
    Returns:
        DataFrame with 13 feature columns
    """
    if lang_col and lang_col in df.columns:
        args = list(zip(df[code_col].values, df[lang_col].values))
    else:
        # Auto-detect language
        args = list(zip(df[code_col].values, [None] * len(df)))

    print(f"  Extracting features from {len(args):,} samples ({n_workers} workers)...")
    t0 = time.time()

    if n_workers <= 1:
        results = [_extract_row(a) for a in tqdm(args, desc="  Features")]
    else:
        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(_extract_row, args, chunksize=500),
                total=len(args),
                desc="  Features",
            ))

    feat_df = pd.DataFrame(results)
    elapsed = time.time() - t0
    speed = len(args) / elapsed
    print(f"  Done in {elapsed:.1f}s ({speed:.0f} samples/s)")
    return feat_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

# %% [markdown]
# ## 2. Load Data & Extract Features

# %%
if __name__ == "__main__":
    print("=" * 70)
    print("  PHASE 4: XGBoost GPU Training Pipeline")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/5] Loading data...")
    df_train = pd.read_parquet(DATA_DIR / "train.parquet")
    df_val = pd.read_parquet(DATA_DIR / "validation.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test.parquet")

    print(f"  Train: {df_train.shape[0]:>10,} rows")
    print(f"  Val:   {df_val.shape[0]:>10,} rows")
    print(f"  Test:  {df_test.shape[0]:>10,} rows")

    # ── Extract features ──
    print("\n[2/5] Feature extraction...")

    # Check for cached features
    cache_train = OUT_DIR / "features_train_500k.parquet"
    cache_val = OUT_DIR / "features_val_100k.parquet"
    cache_test = OUT_DIR / "features_test_500k.parquet"

    if cache_train.exists() and cache_val.exists() and cache_test.exists():
        print("  Loading cached features...")
        feat_train = pd.read_parquet(cache_train)
        feat_val = pd.read_parquet(cache_val)
        feat_test = pd.read_parquet(cache_test)
    else:
        print("\n  --- Train set ---")
        feat_train = extract_features_parallel(
            df_train, code_col="code", lang_col="language", n_workers=N_WORKERS
        )
        feat_train.to_parquet(cache_train, index=False)
        print(f"  Cached: {cache_train}")

        print("\n  --- Validation set ---")
        feat_val = extract_features_parallel(
            df_val, code_col="code", lang_col="language", n_workers=N_WORKERS
        )
        feat_val.to_parquet(cache_val, index=False)
        print(f"  Cached: {cache_val}")

        print("\n  --- Test set (auto-detect language) ---")
        feat_test = extract_features_parallel(
            df_test, code_col="code", lang_col=None, n_workers=N_WORKERS
        )
        feat_test.to_parquet(cache_test, index=False)
        print(f"  Cached: {cache_test}")

    print(f"\n  Feature shapes: train={feat_train.shape}, val={feat_val.shape}, test={feat_test.shape}")

    # ── Prepare X, y matrices ──
    X_train = feat_train[SELECTED_FEATURES].values
    y_train = df_train["label"].values
    X_val = feat_val[SELECTED_FEATURES].values
    y_val = df_val["label"].values
    X_test = feat_test[SELECTED_FEATURES].values
    test_ids = df_test["ID"].values

    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # %% [markdown]
    # ## 3. XGBoost GPU Training

    # %%
    print("\n[3/5] XGBoost Training (GPU)...")
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, classification_report

    # -- Detect GPU --
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        HAS_GPU = result.returncode == 0
    except FileNotFoundError:
        HAS_GPU = False

    if HAS_GPU:
        print("  GPU detected! Using device='cuda'")
        DEVICE = "cuda"
    else:
        print("  No GPU detected. Using CPU (device='cpu')")
        DEVICE = "cpu"

    # -- XGBoost parameters (tuned for binary classification) --
    XGB_PARAMS = {
        "n_estimators": 1000,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": DEVICE,
        "random_state": RANDOM_STATE,
        "n_jobs": -1 if DEVICE == "cpu" else 1,
        "verbosity": 0,
    }

    print(f"  Params: depth={XGB_PARAMS['max_depth']}, lr={XGB_PARAMS['learning_rate']}, "
          f"n_est={XGB_PARAMS['n_estimators']}, device={DEVICE}")

    # -- 5-Fold CV --
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X_train))
    fold_models = []
    fold_aucs = []
    fold_f1s = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
        t0 = time.time()
        print(f"\n  --- Fold {fold}/{N_FOLDS} ---")

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            verbose=False,
        )

        # Best iteration (early stopping via eval)
        preds = model.predict_proba(X_train[va_idx])[:, 1]
        oof_preds[va_idx] = preds

        auc = roc_auc_score(y_train[va_idx], preds)
        f1 = f1_score(y_train[va_idx], (preds > 0.5).astype(int))
        fold_aucs.append(auc)
        fold_f1s.append(f1)
        fold_models.append(model)

        elapsed = time.time() - t0
        print(f"    AUC={auc:.4f}, F1={f1:.4f}, Time={elapsed:.1f}s, "
              f"Best iter={model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")

    # -- CV Summary --
    print("\n" + "=" * 50)
    print(f"  5-Fold CV Results:")
    print(f"    AUC:  {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"    F1:   {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    print("=" * 50)

    # -- OOF Classification Report --
    oof_labels = (oof_preds > 0.5).astype(int)
    print("\n  OOF Classification Report:")
    print(classification_report(y_train, oof_labels, target_names=["Human", "AI"]))

    # -- Validate on held-out validation set --
    print("\n[4/5] Validation set evaluation...")
    val_preds_all = np.zeros(len(X_val))
    for i, model in enumerate(fold_models):
        val_preds_all += model.predict_proba(X_val)[:, 1] / N_FOLDS

    val_auc = roc_auc_score(y_val, val_preds_all)
    val_f1 = f1_score(y_val, (val_preds_all > 0.5).astype(int))
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Validation F1:  {val_f1:.4f}")
    print("\n  Validation Classification Report:")
    print(classification_report(y_val, (val_preds_all > 0.5).astype(int),
                                target_names=["Human", "AI"]))

    # -- Feature importance (averaged across folds) --
    avg_importance = np.zeros(len(SELECTED_FEATURES))
    for model in fold_models:
        avg_importance += model.feature_importances_ / N_FOLDS

    feat_imp = pd.Series(avg_importance, index=SELECTED_FEATURES).sort_values(ascending=False)
    print("\n  Feature Importance (5-fold avg):")
    for rank, (feat, imp) in enumerate(feat_imp.items(), 1):
        bar = "#" * int(imp * 80)
        print(f"    {rank:>2}. {feat:<28s}  {imp:.4f}  {bar}")

    # %% [markdown]
    # ## 4. Inference & Submission

    # %%
    print("\n[5/5] Test set inference...")
    test_preds_all = np.zeros(len(X_test))
    for i, model in enumerate(fold_models):
        test_preds_all += model.predict_proba(X_test)[:, 1] / N_FOLDS

    test_labels = (test_preds_all > 0.5).astype(int)

    # -- Create submission --
    submission = pd.DataFrame({
        "ID": test_ids,
        "label": test_labels,
    })
    sub_path = OUT_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)

    print(f"  Submission saved: {sub_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Label distribution: {submission['label'].value_counts().to_dict()}")
    print(f"  Predicted AI ratio: {test_labels.mean():.4f}")

    # -- Also save probabilities --
    prob_df = pd.DataFrame({
        "ID": test_ids,
        "prob_ai": test_preds_all,
        "label": test_labels,
    })
    prob_df.to_parquet(OUT_DIR / "test_predictions.parquet", index=False)

    # -- Save models --
    model_path = OUT_DIR / "xgb_5fold_models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "models": fold_models,
            "features": SELECTED_FEATURES,
            "cv_aucs": fold_aucs,
            "cv_f1s": fold_f1s,
            "val_auc": val_auc,
            "val_f1": val_f1,
        }, f)
    print(f"  Models saved: {model_path}")

    # -- Final summary --
    print("\n" + "=" * 70)
    print("  PHASE 4 COMPLETE")
    print("=" * 70)
    print(f"  CV AUC:   {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"  Val AUC:  {val_auc:.4f}")
    print(f"  Val F1:   {val_f1:.4f}")
    print(f"  Features: {len(SELECTED_FEATURES)}")
    print(f"  Output:   {sub_path}")
    print("=" * 70)
