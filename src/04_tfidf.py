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

# ═══════════════════════════════════════════════════════════════════════════════
# 2. FIT TF-IDF
# ═══════════════════════════════════════════════════════════════════════════════
divider("Extracting Sparse Semantic Features (TF-IDF)")

# Character n-grams implicitly track code structure, boilerplate padding, 
# python def vs java public static, space behaviors, brackets logic.
tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),   # 3 to 5 character sweeping
    max_features=30000,   # Keep top 30k most robust tokens to fit in RAM
    min_df=5,             # Must appear in 5 files minimum
    max_df=0.6,           # Skip extremely generic tokens present in 60% docs
    lowercase=False,      # Case sensitive is VITAL for CamelCase vs snake_case tracking
    dtype=np.float32,
)

log("Fitting Vectorizer on Train/Val... (May take a few minutes)")
# Fit purely on train to prevent any test leakage
X_tv_sparse = tfidf.fit_transform(X_tv_text)
log(f"  Train/Val Sparse Matrix bounds: {X_tv_sparse.shape}")

log("Transforming Test & Test_Sample sets...")
X_ts_sparse = tfidf.transform(X_ts_text)
X_te_sparse = tfidf.transform(X_te_text)

log(f"  Test Sample Sparse bounds: {X_ts_sparse.shape}")
log(f"  Test Sparse bounds:        {X_te_sparse.shape}")

# Free huge string ram
del X_tv_text, X_ts_text, X_te_text
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN OOF (OUT-OF-FOLD) CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════
divider("Training OOF K-Fold Models (LogisticRegression)")

# Out-of-fold array for the train set
oof_probs = np.zeros(len(y_tv), dtype=np.float32)
# Final test predictions arrays
test_preds = np.zeros(X_te_sparse.shape[0], dtype=np.float32)
ts_preds   = np.zeros(X_ts_sparse.shape[0], dtype=np.float32)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_tv_sparse, y_tv)):
    t_f0 = time.time()
    
    # Train test split for fold
    X_f_tr, y_f_tr = X_tv_sparse[trn_idx], y_tv[trn_idx]
    X_f_val, y_f_val = X_tv_sparse[val_idx], y_tv[val_idx]
    
    # Use Saga solver for ultra-fast sparse matrix convergence
    clf = LogisticRegression(
        C=0.5,                  # Strong L2 regularization to prevent overfitting 30k vectors
        solver='saga', 
        max_iter=100, 
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_f_tr, y_f_tr)
    
    # Validation
    val_probs = clf.predict_proba(X_f_val)[:, 1]
    oof_probs[val_idx] = val_probs
    
    # Collect predictions for testing bounds
    ts_preds += clf.predict_proba(X_ts_sparse)[:, 1] / 5.0
    test_preds += clf.predict_proba(X_te_sparse)[:, 1] / 5.0
    
    fold_auc = roc_auc_score(y_f_val, val_probs)
    fold_f1  = f1_score(y_f_val, (val_probs > 0.5).astype(int), average='macro')
    
    log(f"  Fold {fold+1} | AUC: {fold_auc:.4f} | F1: {fold_f1:.4f} | Time: {(time.time()-t_f0)/60:.1f}m")

# Global OOF Review
cv_auc = roc_auc_score(y_tv, oof_probs)
cv_f1  = f1_score(y_tv, (oof_probs > 0.5).astype(int), average='macro')
log(f"\n  Final OOF AUC: {cv_auc:.4f}")
log(f"  Final OOF F1:  {cv_f1:.4f}")

# Eval on unseen test_sample
gold_auc = roc_auc_score(y_ts, ts_preds)
gold_f1  = f1_score(y_ts, (ts_preds > 0.5).astype(int), average='macro')
log(f"\n  Test_Sample GOLD AUC: {gold_auc:.4f}")
log(f"  Test_Sample GOLD F1:  {gold_f1:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXPORT PROBABILITIES AS FEATURE
# ═══════════════════════════════════════════════════════════════════════════════
divider("Exporting TF-IDF Features")

# Save as (N, 1) column vectors mapping to probability of being AI
np.save(OUT_DIR / "train_tfidf.npy", oof_probs.reshape(-1, 1))
np.save(OUT_DIR / "test_sample_tfidf.npy", ts_preds.reshape(-1, 1))
np.save(OUT_DIR / "test_tfidf.npy", test_preds.reshape(-1, 1))

log(f"  Exported 1-dimensional Semantic Trajectory Features.")
log(f"  Pipeline total time: {(time.time() - T0)/60:.1f}m")
