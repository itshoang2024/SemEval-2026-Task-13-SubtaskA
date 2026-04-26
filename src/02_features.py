#!/usr/bin/env python3
"""
02_features.py — Feature Engineering Pipeline
================================================
Chạy trên Kaggle:
    Cell 1:
        !pip install lightgbm -q
        %cd /kaggle/working
        !git clone https://github.com/gugOfBoat/SemEval-2026-Task-13-SubtaskA.git
    Cell 2:
        %matplotlib inline
        %run SemEval-2026-Task-13-SubtaskA/src/02_features.py

Steps:
    1. Extract ~45 raw features
    2. Correlation filter (>0.90 drop)
    3. Adversarial validation (drop OOD-leaky)
    4. Target gain LOLO (keep top 15)
    4.5 Gold check on test_sample (8 langs)
    5. Per-language normalize + save .npy
"""

import os, re, math, time, json, warnings, zlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore")
T0 = time.time()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/kaggle/input/competitions/sem-eval-2026-task-13-subtask-a/Task_A")
OUT_DIR  = Path("/kaggle/working")

def log(msg):
    elapsed = (time.time() - T0) / 60
    print(f"[{elapsed:6.1f}m] {msg}", flush=True)

def divider(title):
    print(f"\n{'━'*70}")
    print(f"  {title}")
    print(f"{'━'*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
divider("Loading data")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df   = pd.read_parquet(DATA_DIR / "validation.parquet")
ts_df    = pd.read_parquet(DATA_DIR / "test_sample.parquet")
test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

# Merge train + val
tv_df = pd.concat([train_df, val_df], ignore_index=True)

# Normalise labels
for df in [tv_df, ts_df]:
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"human": 0, "ai": 1})

tv_codes = tv_df["code"].fillna("").values
ts_codes = ts_df["code"].fillna("").values
test_codes = test_df["code"].fillna("").values

y_train = tv_df["label"].values.astype(int)
y_ts    = ts_df["label"].values.astype(int)
train_langs = tv_df["language"].values
ts_langs    = ts_df["language"].values

log(f"Train+Val: {len(tv_df):,} rows | langs={sorted(tv_df['language'].unique())}")
log(f"Test Sample: {len(ts_df):,} rows | langs={sorted(ts_df['language'].unique())}")
log(f"Test: {len(test_df):,} rows | cols={test_df.columns.tolist()}")

def infer_family_from_code(code: str) -> str:
    """
    Phân loại code thành 4 họ ngôn ngữ chính.
    C_CPP, PYTHON, JVM_ISH (Java/C#/Go), SCRIPTING (JS/PHP)
    """
    if not isinstance(code, str) or not code:
        return "PYTHON"
    
    if "<?php" in code or "$_" in code or "echo " in code: return "SCRIPTING"
    if "console.log" in code or "function(" in code or "const " in code or "let " in code or "=>" in code: return "SCRIPTING"
    if "package main" in code or "func " in code or "fmt." in code or "import (" in code: return "JVM_ISH"
    if "public static void main" in code or "System.out.println" in code or "using System" in code or "namespace " in code: return "JVM_ISH"
    if "#include" in code or "std::" in code or "int main" in code or "printf" in code: return "C_CPP"
    if "def " in code or "import " in code or "print(" in code or "class " in code: return "PYTHON"
    
    return "PYTHON"

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: EXTRACT ~45 RAW FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
divider("Step 1 · Extract ~45 raw features")

KEYWORDS = {
    'if','else','for','while','return','int','float','void','class',
    'def','import','from','print','const','let','var','function',
    'true','false','null','self','this','new','try','catch','func',
    'package','struct','interface','echo','foreach','namespace','using',
    'async','await','lambda','with','yield','switch','case','break',
    'public','private','protected','static','final','except','throw',
}
MARKER_RE = re.compile(r"\b(TODO|FIXME|HACK|XXX|DEBUG|OPTIMIZE|WORKAROUND)\b", re.I)
SNAKE_RE  = re.compile(r"[a-z]+_[a-z]+")
CAMEL_RE  = re.compile(r"[a-z]+[A-Z][a-z]+")
IDENT_RE  = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def extract_raw(code: str) -> dict:
    """Extract ~45 language-agnostic features from a code snippet."""
    # Silence numpy warnings within joblib worker threads
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore")
    
    f = {}
    if not code or not isinstance(code, str):
        code = ""

    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    n_lines = max(len(lines), 1)
    n_chars = max(len(code), 1)
    tokens = IDENT_RE.findall(code)
    n_tokens = max(len(tokens), 1)

    # ── Comment axis (8) ─────────────────────────────────────────────────
    comment_lines = [l for l in lines
                     if l.strip().startswith(("//", "#", "/*", "*", "<!--"))]
    f["comment_ratio"] = len(comment_lines) / n_lines

    comment_words = [len(re.findall(r"\b\w+\b", l)) for l in comment_lines]
    f["comment_word_avg"] = float(np.mean(comment_words)) if comment_words else 0.0
    f["cmt_avg_len"] = float(np.mean([len(l) for l in comment_lines])) if comment_lines else 0.0

    cmt_total = sum(len(l) for l in comment_lines)
    code_total = max(n_chars - cmt_total, 1)
    f["cmt_to_code_ratio"] = cmt_total / code_total
    f["has_debug_marker"] = 1.0 if MARKER_RE.search(code) else 0.0
    f["trailing_ws_ratio"] = sum(1 for l in lines if l != l.rstrip()) / n_lines

    # Burstiness: gaps between blank lines
    gaps = []
    gap = 0
    for l in lines:
        if not l.strip():
            gap += 1
        else:
            if gap > 0:
                gaps.append(gap)
            gap = 0
    f["blank_line_gap_std"]  = float(np.std(gaps)) if len(gaps) > 1 else 0.0
    f["blank_line_gap_mean"] = float(np.mean(gaps)) if gaps else 0.0

    # ── Compression/Entropy axis (7) ─────────────────────────────────────
    code_bytes = code.encode("utf-8", errors="replace")
    compressed = zlib.compress(code_bytes, level=6)
    f["zlib_ratio"] = len(compressed) / max(len(code_bytes), 1)

    byte_arr = np.frombuffer(code_bytes, dtype=np.uint8)
    if byte_arr.size > 0:
        cnts = np.bincount(byte_arr, minlength=256)
        probs = cnts[cnts > 0] / byte_arr.size
        f["byte_entropy"] = float(-(probs * np.log2(probs)).sum())
    else:
        f["byte_entropy"] = 0.0

    char_counts = Counter(code)
    total_chars = sum(char_counts.values())
    if total_chars > 0:
        probs_c = np.array(list(char_counts.values()), dtype=float) / total_chars
        f["char_entropy"] = float(-(probs_c * np.log2(probs_c)).sum())
    else:
        f["char_entropy"] = 0.0

    token_counts = Counter(tokens)
    if token_counts:
        probs_t = np.array(list(token_counts.values()), dtype=float) / n_tokens
        f["token_entropy"] = float(-(probs_t * np.log2(probs_t)).sum())
    else:
        f["token_entropy"] = 0.0

    # Trigram repetition
    if len(code) >= 3:
        trigrams = [code[i:i+3] for i in range(len(code) - 2)]
        tri_counts = Counter(trigrams)
        repeated = sum(1 for c in tri_counts.values() if c > 1)
        f["trigram_rep_ratio"] = repeated / max(len(tri_counts), 1)
    else:
        f["trigram_rep_ratio"] = 0.0

    unique_tokens = len(set(tokens))
    f["hapax_ratio"] = sum(1 for v in token_counts.values() if v == 1) / max(unique_tokens, 1)
    f["ttr"] = unique_tokens / n_tokens

    # ── Structural axis (18) ─────────────────────────────────────────────
    f["n_lines"]  = len(lines)
    f["code_len"] = len(code)

    line_lengths = [len(l) for l in lines]
    f["avg_line_len"] = float(np.mean(line_lengths))
    f["std_line_len"] = float(np.std(line_lengths)) if len(line_lengths) > 1 else 0.0
    f["max_line_len"] = max(line_lengths) if line_lengths else 0
    f["line_len_cv"]  = f["std_line_len"] / max(f["avg_line_len"], 1e-6)

    indents = [len(l) - len(l.lstrip()) for l in non_empty] if non_empty else [0]
    f["avg_indent"] = float(np.mean(indents))
    f["indent_std"] = float(np.std(indents)) if len(indents) > 1 else 0.0

    # Nesting depth via {/}
    depth, mx, depths = 0, 0, []
    for ch in code:
        if ch in "{(":
            depth += 1
            mx = max(mx, depth)
        elif ch in "})":
            depth = max(0, depth - 1)
        if ch == "\n":
            depths.append(depth)
    f["nest_max"]  = mx
    f["nest_mean"] = float(np.mean(depths)) if depths else 0.0

    stripped = [l.strip() for l in non_empty] if non_empty else [""]
    f["unique_line_ratio"]    = len(set(stripped)) / max(len(stripped), 1)
    f["duplicate_line_ratio"] = 1.0 - f["unique_line_ratio"]
    f["empty_ratio"]          = sum(1 for l in lines if not l.strip()) / n_lines

    all_indents = [len(l) - len(l.lstrip()) for l in lines]
    f["exact_indent_ratio"] = sum(1 for i in all_indents if i % 4 == 0) / n_lines

    # Line length autocorrelation lag-1
    if len(line_lengths) > 2:
        ll = np.array(line_lengths, dtype=float)
        v1, v2 = ll[:-1], ll[1:]
        if v1.std() > 0 and v2.std() > 0:
            autocorr_val = np.corrcoef(v1, v2)[0, 1]
            f["lag1_autocorr"] = float(autocorr_val) if not np.isnan(autocorr_val) else 0.0
        else:
            f["lag1_autocorr"] = 0.0
    else:
        f["lag1_autocorr"] = 0.0

    # Indent delta entropy
    indent_deltas = [abs(all_indents[i+1] - all_indents[i])
                     for i in range(len(all_indents) - 1)]
    if indent_deltas:
        dc = Counter(indent_deltas)
        dt = sum(dc.values())
        probs_d = np.array(list(dc.values()), dtype=float) / dt
        f["indent_delta_entropy"] = float(-(probs_d * np.log2(probs_d)).sum())
    else:
        f["indent_delta_entropy"] = 0.0

    # Max consecutive empty
    max_ce, cur_ce = 0, 0
    for l in lines:
        if not l.strip():
            cur_ce += 1
            max_ce = max(max_ce, cur_ce)
        else:
            cur_ce = 0
    f["max_consec_empty"] = max_ce

    f["line_len_iqr"] = float(np.percentile(line_lengths, 75) - np.percentile(line_lengths, 25)) if line_lengths else 0.0

    # ── Compression/Entropy Boosts (Language Agnostic) ───────────────────
    if len(code_bytes) > 0:
        import bz2, lzma
        compressed_bz2 = bz2.compress(code_bytes, compresslevel=9)
        f["bz2_ratio"] = len(compressed_bz2) / len(code_bytes)
        
        try:
            compressed_lzma = lzma.compress(code_bytes)
            f["lzma_ratio"] = len(compressed_lzma) / len(code_bytes)
        except Exception:
            f["lzma_ratio"] = f["bz2_ratio"] # Fallback
    else:
        f["bz2_ratio"] = 0.0
        f["lzma_ratio"] = 0.0

    f["num_density"]   = len(re.findall(r"\b\d+\b", code)) / n_tokens
    
    # Generic string literals density
    f["str_lit_density"] = (len(re.findall(r'"(?:[^"\\]|\\.)*"', code)) +
                            len(re.findall(r"'(?:[^'\\]|\\.)*'", code))) / n_lines
                            
    f["op_density"]    = len(re.findall(r"[+\-*/%=<>!&|^~]", code)) / n_chars
    punc = set('!@#$%^&*(),.?":{}\|<>[]\\;\'\`~-+=/')
    f["punc_ratio"] = sum(1 for c in code if c in punc) / n_chars

    return f


# ── Batch extraction ─────────────────────────────────────────────────────────
log("Extracting features from train+val (parallel)...")
t1 = time.time()
n_jobs = 4 # Kaggle CPU typically has 4 cores
features_tr = Parallel(n_jobs=n_jobs)(delayed(extract_raw)(c) for c in tqdm(tv_codes, desc="train"))
df_feat_train = pd.DataFrame(features_tr)
log(f"  Train: {df_feat_train.shape} in {(time.time()-t1)/60:.1f} min")

log("Extracting features from test_sample (parallel)...")
features_ts = Parallel(n_jobs=n_jobs)(delayed(extract_raw)(c) for c in tqdm(ts_codes, desc="test_sample"))
df_feat_ts = pd.DataFrame(features_ts)
log(f"  Test Sample: {df_feat_ts.shape}")

log("Extracting features from test (parallel)...")
t2 = time.time()
features_te = Parallel(n_jobs=n_jobs)(delayed(extract_raw)(c) for c in tqdm(test_codes, desc="test"))
df_feat_test = pd.DataFrame(features_te)
log(f"  Test: {df_feat_test.shape} in {(time.time()-t2)/60:.1f} min")

CANDIDATE_FEATURES = list(df_feat_train.columns)
log(f"  Total candidate features: {len(CANDIDATE_FEATURES)}")
log(f"  NaN counts:\n{df_feat_train.isna().sum()[df_feat_train.isna().sum() > 0].to_string()}")

# Replace NaN/inf
for df in [df_feat_train, df_feat_ts, df_feat_test]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: CORRELATION FILTER (>0.90 → drop one)
# ═══════════════════════════════════════════════════════════════════════════════
divider("Step 2 · Correlation filter (>0.90)")

feat_cols = list(CANDIDATE_FEATURES)
corr_mat = df_feat_train[feat_cols].corr().abs()

# For each highly correlated pair, keep the one with higher AUC vs label
to_drop_corr = set()
checked = set()
log("  Highly correlated pairs (|r| > 0.90):")
for i in range(len(feat_cols)):
    for j in range(i + 1, len(feat_cols)):
        if corr_mat.iloc[i, j] > 0.90:
            fi, fj = feat_cols[i], feat_cols[j]
            if fi in to_drop_corr or fj in to_drop_corr:
                continue
            # Keep the one with higher AUC
            try:
                auc_i = roc_auc_score(y_train, df_feat_train[fi].values)
                auc_j = roc_auc_score(y_train, df_feat_train[fj].values)
            except:
                auc_i, auc_j = 0.5, 0.5
            if auc_i < 0.5:
                auc_i = 1 - auc_i
            if auc_j < 0.5:
                auc_j = 1 - auc_j
            drop = fj if auc_i >= auc_j else fi
            keep = fi if auc_i >= auc_j else fj
            to_drop_corr.add(drop)
            log(f"    {fi} ↔ {fj} (r={corr_mat.iloc[i,j]:.3f}) "
                f"→ DROP {drop} (AUC={min(auc_i,auc_j):.4f}), "
                f"KEEP {keep} (AUC={max(auc_i,auc_j):.4f})")

feat_cols = [f for f in feat_cols if f not in to_drop_corr]
log(f"\n  Correlation filter: {len(CANDIDATE_FEATURES)} → {len(feat_cols)} features")
log(f"  Dropped ({len(to_drop_corr)}): {sorted(to_drop_corr)}")

# Plot correlation heatmap of surviving features
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_feat_train[feat_cols].corr(), annot=False, cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title(f"Feature Correlation Matrix ({len(feat_cols)} features after filter)")
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: ADVERSARIAL VALIDATION (OOD detector)
# ═══════════════════════════════════════════════════════════════════════════════
divider("Step 3 · Adversarial validation")
import lightgbm as lgb

# Combine train + test features, label: 0=train, 1=test
X_adv = np.vstack([
    df_feat_train[feat_cols].values,
    df_feat_test[feat_cols].values,
])
y_adv = np.array([0]*len(df_feat_train) + [1]*len(df_feat_test))

log(f"  Adversarial dataset: {X_adv.shape[0]:,} rows (train={len(df_feat_train):,} + test={len(df_feat_test):,})")

adv_model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=5, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1, n_jobs=-1,
)
adv_model.fit(X_adv, y_adv)
adv_probs = adv_model.predict_proba(X_adv)[:, 1]
adv_auc = roc_auc_score(y_adv, adv_probs)

log(f"  Adversarial AUC: {adv_auc:.4f}")
if adv_auc > 0.80:
    log(f"  ⚠ AUC > 0.80 → SEVERE distribution shift detected!")
elif adv_auc > 0.65:
    log(f"  ⚠ AUC > 0.65 → Moderate distribution shift")
else:
    log(f"  ✓ AUC ≤ 0.65 → Mild shift")

# Feature importance from adversarial model
adv_importance = adv_model.feature_importances_
adv_ranking = sorted(zip(feat_cols, adv_importance), key=lambda x: -x[1])

log(f"\n  Adversarial feature importance (top = most OOD-leaky):")
for rank, (fname, imp) in enumerate(adv_ranking):
    tag = " ← DROP" if rank < 10 else ""
    log(f"    {rank+1:2d}. {fname:25s} gain={imp:8.1f}{tag}")

# Prevent dropping! We will use Feature Neutralization later in 03_train.py
to_drop_adv = set()
log("\n  Adversarial filter: SKIPPING FEATURE DROP - WE WILL NEUTRALIZE FEATURES IN 03_TRAIN.PY INSTEAD.")

# Plot adversarial importance
fig, ax = plt.subplots(figsize=(10, 6))
names_adv = [n for n, _ in adv_ranking]
imps_adv  = [i for _, i in adv_ranking]
colors = ["#E8634C" if n in to_drop_adv else "#4C9BE8" for n in names_adv]
ax.barh(range(len(names_adv)), imps_adv, color=colors)
ax.set_yticks(range(len(names_adv)))
ax.set_yticklabels(names_adv, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Adversarial Importance (gain)")
ax.set_title(f"Adversarial Validation: Red = OOD-leaky (dropped), AUC={adv_auc:.4f}")
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: TARGET GAIN (keep top 15)
# ═══════════════════════════════════════════════════════════════════════════════
divider("Step 4 · Target gain (LOLO CV)")

X_train_filtered = df_feat_train[feat_cols].values.astype(np.float32)
languages_unique = sorted(tv_df["language"].unique())

log(f"  Features remaining: {len(feat_cols)}")
log(f"  LOLO folds: {languages_unique}")

# LOLO CV
fold_importances = np.zeros(len(feat_cols))
lolo_results = {}

for fold_lang in languages_unique:
    val_mask = (train_langs == fold_lang)
    tr_mask  = ~val_mask

    X_tr  = X_train_filtered[tr_mask]
    X_val = X_train_filtered[val_mask]
    y_tr  = y_train[tr_mask]
    y_val = y_train[val_mask]

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs > 0.5).astype(int)
    fold_f1 = f1_score(y_val, preds, average="macro")
    fold_auc = roc_auc_score(y_val, probs)
    lolo_results[fold_lang] = {"f1": fold_f1, "auc": fold_auc,
                                "n_train": tr_mask.sum(), "n_val": val_mask.sum(),
                                "best_iter": model.best_iteration_}

    fold_importances += model.feature_importances_

    log(f"  Fold hold-out={fold_lang}: "
        f"train={tr_mask.sum():>8,} | val={val_mask.sum():>8,} | "
        f"F1={fold_f1:.4f} | AUC={fold_auc:.4f} | best_iter={model.best_iteration_}")

# Average importance across folds
avg_importance = fold_importances / len(languages_unique)
target_ranking = sorted(zip(feat_cols, avg_importance), key=lambda x: -x[1])

log(f"\n  Feature ranking by avg gain (LOLO):")
cum_gain = 0
total_gain = sum(g for _, g in target_ranking)
for rank, (fname, gain) in enumerate(target_ranking):
    cum_gain += gain
    pct = cum_gain / total_gain * 100
    tag = " ✓ SELECTED" if rank < 15 else ""
    log(f"    {rank+1:2d}. {fname:25s} gain={gain:10.1f}  cum={pct:5.1f}%{tag}")

# Keep top 30
TOP_N = min(30, len(target_ranking))
selected_features = [f for f, _ in target_ranking[:TOP_N]]
log(f"\n  Selected top {TOP_N}: {selected_features}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
names_tgt = [n for n, _ in target_ranking]
gains_tgt = [g for _, g in target_ranking]
colors = ["#56B17B" if n in selected_features else "#cccccc" for n in names_tgt]
ax.barh(range(len(names_tgt)), gains_tgt, color=colors)
ax.set_yticks(range(len(names_tgt)))
ax.set_yticklabels(names_tgt, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Average Gain (LOLO)")
ax.set_title(f"Target Gain: Green = Selected Top {TOP_N}")
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: SAVE FINAL RAW RATIO FEATURES (No Scaling)
# ═══════════════════════════════════════════════════════════════════════════════
divider(f"Step 5 · Save Selected {len(selected_features)} Ratio Features")

X_train_final = df_feat_train[selected_features].values.astype(np.float32)
X_test_final  = df_feat_test[selected_features].values.astype(np.float32)
X_ts_final    = df_feat_ts[selected_features].values.astype(np.float32)

np.save(OUT_DIR / "train_handcraft.npy", X_train_final)
np.save(OUT_DIR / "test_handcraft.npy", X_test_final)
np.save(OUT_DIR / "test_sample_handcraft.npy", X_ts_final)

with open(OUT_DIR / "feature_meta.json", "w") as f:
    json.dump({
        "n_features": len(selected_features),
        "features": selected_features,
        "date_created": "2026-04-25"
    }, f, indent=2)

log(f"\n  Saved:")
log(f"    train_handcraft.npy        {X_train_final.shape}")
log(f"    test_handcraft.npy         {X_test_final.shape}")
log(f"    test_sample_handcraft.npy  {X_ts_final.shape}")
log(f"    feature_meta.json")

t1 = time.time()
divider("02_features.py — Final Summary")
log(f"  PIPELINE SUMMARY")
log(f"  ════════════════")
log(f"  Step 1: Extracted {len(df_feat_train.columns)-1} features from {len(df_feat_train)+len(df_feat_test)+len(df_feat_ts):,} samples")
log(f"  Step 2: Correlation filter (>0.90) → dropped {len(df_feat_train.columns)-1 - len(feat_cols)}")
log(f"  Step 3: Adversarial filter → dropped {len(to_drop_adv)} (AUC={adv_auc:.4f})")
log(f"  Step 4: Target gain LOLO → kept top {len(selected_features)}")
log(f"")
log(f"  FINAL {len(selected_features)} FEATURES:")
log(f"  {selected_features}")
log(f"")
log(f"  Total time: {(t1-T0)/60:.1f} min\n")
