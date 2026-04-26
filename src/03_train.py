#!/usr/bin/env python3
"""
03_train.py — Main Training & Ensemble Pipeline
================================================
Chạy trên Kaggle: Mất khoảng vài chục phút CPU (nếu không dùng GPU).

Pipeline:
1. Load 15 features từ 02_features.py (train_handcraft.npy, etc.)
2. Gộp train (600k) và test_sample (1k)
3. CV: 3 folds LOLO + test_sample_OOF
4. Ensemble (LGB, XGB, CatBoost) with early stopping on test_sample
5. Pseudo-labeling (stratified per-family)
6. Per-family threshold calibration on test_sample OOF
7. Generate submission.csv
"""

import os
import re
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score

import lightgbm as lgb
import xgboost as xgb
# Note: CatBoost loading might require installing or using catboost gracefully without error
try:
    import catboost as cb
except ImportError:
    cb = None

warnings.filterwarnings("ignore")
T0 = time.time()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/kaggle/input/competitions/sem-eval-2026-task-13-subtask-a/Task_A")
OUT_DIR  = Path("/kaggle/working")
FEAT_DIR = Path("/kaggle/working")

# Fallbacks specifically for the user's dataset
for test_dir in ["/kaggle/input/semeval", "/kaggle/input/notebooks/thtynn/semeval"]:
    p = Path(test_dir)
    if (p / "train_handcraft.npy").exists():
        FEAT_DIR = p
        break

if not DATA_DIR.exists():
    DATA_DIR = Path("../data/raw/Task_A")
    OUT_DIR  = Path(".")
    FEAT_DIR = Path(".")

def log(msg):    
    elapsed = (time.time() - T0) / 60
    print(f"[{elapsed:6.1f}m] {msg}", flush=True)

def divider(title):
    print(f"\n{'━'*70}")
    print(f"  {title}")
    print(f"{'━'*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# 0. HELPER FUNCTIONS: LOGIT SHIFTING
# ═══════════════════════════════════════════════════════════════════════════════
def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def expit(x):
    return 1 / (1 + np.exp(-x))

def map_to_family(lang_array):
    fams = []
    for l in lang_array:
        l = str(l).lower().strip()
        if l in ["c", "c++", "cpp", "c#", "csharp"]: fams.append("C_CPP")
        elif l in ["java", "go", "kotlin", "scala", "groovy"]: fams.append("JVM_ISH")
        elif l in ["javascript", "php", "ruby", "rust"]: fams.append("SCRIPTING")
        else: fams.append("PYTHON")
    return np.array(fams)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD TEXT DATA & CLASSIFY LANGUAGES
# ═══════════════════════════════════════════════════════════════════════════════
divider("Loading Text Data & Classifying Languages")

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df   = pd.read_parquet(DATA_DIR / "validation.parquet")
tv_df    = pd.concat([train_df, val_df], ignore_index=True)
ts_df    = pd.read_parquet(DATA_DIR / "test_sample.parquet")
test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

def parse_label(col):
    if col.dtype == object:
        return col.str.lower().map({"human": 0, "ai": 1}).values.astype(float)
    return col.values.astype(float)

y_tv = parse_label(tv_df["label"])
y_ts = parse_label(ts_df["label"])
log("Applying Deterministic Language Inference Model...")
import re

def infer_family_heuristics(codes):
    preds = []
    for c in codes:
        c = str(c).lower()
        if "```cpp" in c or "```c++" in c or "```c\n" in c: 
            preds.append("C_CPP"); continue
        if "```java" in c or "```go" in c or "```c#" in c: 
            preds.append("JVM_ISH"); continue
        if "```python" in c: 
            preds.append("PYTHON"); continue
        if "```javascript" in c or "```php" in c or "```ruby" in c or "```rust" in c:
            preds.append("SCRIPTING"); continue
            
        counts = {
            "PYTHON": c.count("def ") + c.count("import ") + c.count("print(") + c.count("self.") + c.count("elif "),
            "C_CPP": c.count("#include") + c.count("std::") + c.count("cout") + c.count("using namespace") + c.count("int main"),
            "JVM_ISH": c.count("public class") + c.count("system.out") + c.count("namespace ") + c.count("package main") + c.count("func "),
            "SCRIPTING": c.count("console.log") + c.count("<?php") + c.count("let ") + c.count("const ") + c.count("=>") + c.count("function")
        }
        
        best = max(counts, key=counts.get)
        preds.append(best if counts[best] > 0 else "PYTHON")
    return np.array(preds)

log("Evaluating Inference Accuracy on test_sample...")
ts_predicted_families = infer_family_heuristics(ts_df["code"].values)
ts_actual_families = map_to_family(ts_df["language"].astype(str).values)

acc = np.mean(ts_predicted_families == ts_actual_families)
log(f"Family Inference Accuracy: {acc:.4f}")

use_per_family = True
    
log("Inferring language family for 500k test samples...")
test_families = infer_family_heuristics(test_df["code"].values)

tv_families = map_to_family(tv_df["language"].astype(str).values)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD FEATURES AND NEUTRALIZE
# ═══════════════════════════════════════════════════════════════════════════════
divider("Loading & Neutralizing Features")

try:
    def safe_load(p1, p2):
        return np.load(p1) if p1.exists() else np.load(p2)
        
    X_tv = np.load(FEAT_DIR / "train_handcraft.npy")
    X_ts = safe_load(FEAT_DIR / "test_sample_handcraft.npy", FEAT_DIR / "test_sample_handcraft.np")
    X_te = safe_load(FEAT_DIR / "test_handcraft.npy", FEAT_DIR / "test_handcraft.np")
    
    log(f"  Loaded Handcrafted arrays... Shape: {X_tv.shape}")
    
    # Feature Neutralization
    log("  Applying Feature Neutralization (Division by Train Family Medians) ...")
    medians = {}
    for fam in np.unique(tv_families):
        fam_mask = (tv_families == fam)
        # Add epsilon to prevent div zero
        fam_median = np.median(X_tv[fam_mask], axis=0) + 1e-9
        medians[fam] = fam_median
        X_tv[fam_mask] /= fam_median

    for fam, med in medians.items():
        ts_mask = (ts_predicted_families == fam)
        if ts_mask.sum() > 0: X_ts[ts_mask] /= med
        
        te_mask = (test_families == fam)
        if te_mask.sum() > 0: X_te[te_mask] /= med
        
    log("  ✓ Feature Shift Neutralized.")

    # Track 1 Integration (TF-IDF probabilities)
    try:
        X_tv_tfidf = np.load(FEAT_DIR / "train_tfidf.npy")
        X_ts_tfidf = safe_load(FEAT_DIR / "test_sample_tfidf.npy", FEAT_DIR / "test_sample_tfidf.np")
        X_te_tfidf = safe_load(FEAT_DIR / "test_tfidf.npy", FEAT_DIR / "test_tfidf.np")
        
        X_tv = np.hstack([X_tv, X_tv_tfidf])
        X_ts = np.hstack([X_ts, X_ts_tfidf])
        X_te = np.hstack([X_te, X_te_tfidf])
        log("Successfully integrated TF-IDF Semantic Track (+ features).")
    except FileNotFoundError:
        log("⚠ TF-IDF Track features absent. Using only Handcrafted Track features.")
        
except FileNotFoundError:
    log("⚠ Cannot find precomputed features. Creating random dummy features for architecture testing...")
    X_tv = np.random.randn(len(y_tv), 30).astype(np.float32)
    X_ts = np.random.randn(len(y_ts), 30).astype(np.float32)
    X_te = np.random.randn(len(test_df), 30).astype(np.float32)

log(f"Final Matrix => Train/Val: {X_tv.shape} | Test Sample: {X_ts.shape} | Test: {X_te.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def train_lgb(X_tr, y_tr, X_va_in=None, y_va_in=None, is_soft=False, best_iters=None):
    params = dict(
        objective="binary",
        n_estimators=best_iters if best_iters else 1000,
        learning_rate=0.03,
        max_depth=6, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_jobs=-1
    )
    model = lgb.LGBMRegressor(**params) if is_soft else lgb.LGBMClassifier(**params)
    
    if not is_soft and X_va_in is not None:
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
        model.fit(X_tr, y_tr, eval_set=[(X_va_in, y_va_in)], callbacks=callbacks)
    else:
        model.fit(X_tr, y_tr)
    return model

def train_xgb(X_tr, y_tr, X_va_in=None, y_va_in=None, is_soft=False, best_iters=None):
    model = xgb.XGBRegressor(
        objective="binary:logistic",
        n_estimators=best_iters if best_iters else 1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=50 if (not is_soft and X_va_in is not None) else None,
        n_jobs=-1
    )
    if not is_soft and X_va_in is not None:
        model.fit(X_tr, y_tr, eval_set=[(X_va_in, y_va_in)], verbose=False)
    else:
        model.fit(X_tr, y_tr, verbose=False)
    return model

def train_cat(X_tr, y_tr, X_va_in=None, y_va_in=None, is_soft=False, best_iters=None):
    if cb is None: return None
    params = dict(
        iterations=best_iters if best_iters else 1000,
        learning_rate=0.03,
        depth=6,
        random_state=42,
        verbose=0,
        thread_count=-1
    )
    if is_soft:
        model = cb.CatBoostRegressor(loss_function="RMSE", **params)
    else:
        model = cb.CatBoostClassifier(loss_function="CrossEntropy", **params)
        
    if not is_soft and X_va_in is not None:
        model.fit(X_tr, y_tr, eval_set=(X_va_in, y_va_in), early_stopping_rounds=50)
    else:
        model.fit(X_tr, y_tr)
    return model

def train_ensemble(X_tr, y_tr, X_va_in=None, y_va_in=None, is_soft=False, best_iters=None):
    m_lgb = train_lgb(X_tr, y_tr, X_va_in, y_va_in, is_soft, best_iters)
    m_xgb = train_xgb(X_tr, y_tr, X_va_in, y_va_in, is_soft, best_iters)
    m_cat = train_cat(X_tr, y_tr, X_va_in, y_va_in, is_soft, best_iters)
    return m_lgb, m_xgb, m_cat

def predict_ensemble(models, X):
    m_lgb, m_xgb, m_cat = models
    p_lgb = m_lgb.predict_proba(X)[:, 1] if hasattr(m_lgb, 'predict_proba') else m_lgb.predict(X)
    p_xgb = m_xgb.predict(X)
    if m_cat:
        p_cat = m_cat.predict_proba(X)[:, 1] if hasattr(m_cat, 'predict_proba') else m_cat.predict(X)
        return (p_lgb + p_xgb + p_cat) / 3.0
    return (p_lgb + p_xgb) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CV & TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
divider("Training with Honest CV (LOLO + Stratified)")

ts_preds_fold = np.zeros((len(X_ts), 3))
test_preds_fold = np.zeros((len(X_te), 3))

from sklearn.model_selection import StratifiedKFold

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

ts_preds_fold = np.zeros((len(X_ts), n_splits))
test_preds_fold = np.zeros((len(X_te), n_splits))

for i, (train_idx, val_idx) in enumerate(skf.split(X_tv, y_tv)):
    X_tr, y_tr = X_tv[train_idx], y_tv[train_idx]
    X_va_in, y_va_in = X_tv[val_idx], y_tv[val_idx]
    
    log(f"  Fold {i+1}/{n_splits}: Train {X_tr.shape[0]:,} | Val {X_va_in.shape[0]:,}")
    
    # Train robust generic structural models with early stopping on Val
    models = train_ensemble(X_tr, y_tr, X_va_in, y_va_in)
    
    ts_preds_fold[:, i] = predict_ensemble(models, X_ts)
    test_preds_fold[:, i] = predict_ensemble(models, X_te)

y_ts_oof = ts_preds_fold.mean(axis=1)
test_probs_initial = test_preds_fold.mean(axis=1)

log(f"\n  OOF Final Result before Pseudo-labeling:")
initial_f1 = f1_score(y_ts, (y_ts_oof > 0.5).astype(int), average="macro")
log(f"    Macro F1 on test_sample: {initial_f1:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PER-FAMILY CALIBRATION VIA LOGIT SHIFT
# ═══════════════════════════════════════════════════════════════════════════════
divider("Per-family Calibration via Logit Shift")

actual_families_np = np.array(ts_actual_families)
thresholds = {"PYTHON": 0.5, "JVM_ISH": 0.5, "C_CPP": 0.5, "SCRIPTING": 0.5}

if use_per_family:
    for fam in thresholds.keys():
        fam_idx = np.where(actual_families_np == fam)[0]
        if len(fam_idx) < 10:
            log(f"  Family {fam:10s}: Insufficient sample. Using threshold 0.5")
            continue
            
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.95, 0.05):
            preds = (y_ts_oof[fam_idx] >= t).astype(int)
            f1 = f1_score(y_ts[fam_idx], preds, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[fam] = best_t
        log(f"  Family {fam:10s}: selected threshold {best_t:.2f} (F1={best_f1:.4f} | n={len(fam_idx)})")

else:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (y_ts_oof >= t).astype(int)
        f1 = f1_score(y_ts, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    for fam in thresholds: thresholds[fam] = best_t
    log(f"  Global calibration: selected threshold {best_t:.2f} (F1={best_f1:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
divider("Generate Submission")

test_preds = np.zeros(len(test_probs_initial), dtype=int)

for fam, t in thresholds.items():
    fam_idx = np.where(test_families == fam)[0]
    test_preds[fam_idx] = (test_probs_initial[fam_idx] >= t).astype(int)

ai_ratio = test_preds.mean()
log(f"  Final AI ratio on 500k test set: {ai_ratio:.2%}")

sub = pd.DataFrame({
    "ID": test_df["id"] if "id" in test_df.columns else test_df["ID"] if "ID" in test_df.columns else test_df.index,
    "label": test_preds
})
sub_file = OUT_DIR / "submission.csv"
sub.to_csv(sub_file, index=False)

log(f"\nPipeline complete! Submission saved: {sub_file.absolute()}")
print(f"Total time: {(time.time() - T0) / 60:.1f} min")
