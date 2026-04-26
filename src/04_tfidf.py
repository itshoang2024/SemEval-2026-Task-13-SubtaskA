#!/usr/bin/env python3
"""
04_tfidf.py — Linear TF-IDF Semantic Pipeline
================================================
This script treats the AI Code Detection problem as an Authorship Attribution task.
Instead of handcrafted structural counts, we use highly dimensional character n-grams 
to capture semantic fingerprints (spacing habits, naming habits, bracketing).

Outputs: Pushes out a high-quality probability feature `*_tfidf.npy` array 
for Train (via OOF), Test, and Test_Sample to be consumed by the final ensemble.
"""

import os, gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
# 1. LOAD RAW TEXT
# ═══════════════════════════════════════════════════════════════════════════════
divider("Loading Text Data")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df   = pd.read_parquet(DATA_DIR / "validation.parquet")
ts_df    = pd.read_parquet(DATA_DIR / "test_sample.parquet")
test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

tv_df = pd.concat([train_df, val_df], ignore_index=True)

for df in [tv_df, ts_df]:
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().map({"human": 0, "ai": 1})

# Clean text gracefully to prevent NaNs
X_tv_text = tv_df["code"].fillna("").astype(str).values
X_ts_text = ts_df["code"].fillna("").astype(str).values
X_te_text = test_df["code"].fillna("").astype(str).values

y_tv = tv_df["label"].values.astype(int)
y_ts = ts_df["label"].values.astype(int)
tv_langs = tv_df["language"].astype(str).values
ts_langs = ts_df["language"].astype(str).values

log(f"Train/Val texts:  {X_tv_text.shape[0]:,}")
log(f"Test_Sample:      {X_ts_text.shape[0]:,}")
log(f"Test texts:       {X_te_text.shape[0]:,}")
del train_df, val_df, ts_df, test_df
gc.collect()

import re
from sklearn.decomposition import TruncatedSVD

log("Skeletonizing texts... (Stripping alphanumerics)")
# Keep only punctuation, whitespace, brackets, parentheses.
def skeletonize(code_array):
    out = []
    for c in code_array:
        # sub out a-z, A-Z, 0-9
        s = re.sub(r'[a-zA-Z0-9_]+', '', c)
        out.append(s)
    return out

X_tv_skel = skeletonize(X_tv_text)
X_ts_skel = skeletonize(X_ts_text)
X_te_skel = skeletonize(X_te_text)

del X_tv_text, X_ts_text, X_te_text
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FIT TF-IDF
# ═══════════════════════════════════════════════════════════════════════════════
divider("Extracting Sparse Semantic Features (TF-IDF)")

tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 6),   # 2 to 6 character sweeps on punctuation only
    max_features=20000,
    min_df=5,
    max_df=0.7,
    lowercase=False,
    dtype=np.float32,
)

log("Fitting Vectorizer on Train/Val... ")
X_tv_sparse = tfidf.fit_transform(X_tv_skel)
log(f"  Train/Val Sparse Matrix: {X_tv_sparse.shape}")

log("Transforming Test sets...")
X_ts_sparse = tfidf.transform(X_ts_skel)
X_te_sparse = tfidf.transform(X_te_skel)

del X_tv_skel, X_ts_skel, X_te_skel
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SVD COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════
divider("Dimensionality Reduction (TruncatedSVD)")

# Extract top 20 structural/semantic patterns into dense vectors
svd = TruncatedSVD(n_components=20, random_state=42)

log("Fitting SVD on Train/Val matrix...")
X_tv_svd = svd.fit_transform(X_tv_sparse)
log(f"  Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")

log("Transforming Test matrices...")
X_ts_svd = svd.transform(X_ts_sparse)
X_te_svd = svd.transform(X_te_sparse)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXPORT SVD FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
divider("Exporting SVD Features")

np.save(OUT_DIR / "train_tfidf.npy", X_tv_svd.astype(np.float32))
np.save(OUT_DIR / "test_sample_tfidf.npy", X_ts_svd.astype(np.float32))
np.save(OUT_DIR / "test_tfidf.npy", X_te_svd.astype(np.float32))

log(f"  Exported 20-dimensional Structural Trajectory Features.")
log(f"  Pipeline total time: {(time.time() - T0)/60:.1f}m")
