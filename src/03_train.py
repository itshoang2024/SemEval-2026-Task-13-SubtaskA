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

# Thông minh tìm kiếm các dataset bên ngoài user đính kèm (nếu có .npy)
FEAT_DIR = Path("/kaggle/working")
if Path("/kaggle/input").exists():
    for d in Path("/kaggle/input").rglob("*"):
        if d.is_dir() and (d / "train_handcraft.npy").exists():
            FEAT_DIR = d
            break

if not DATA_DIR.exists():
    # Local dev fallback
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
        l = str(l).lower()
        if l in ["c", "c++"]: fams.append("C_CPP")
        elif l in ["java", "c#", "go"]: fams.append("JVM_ISH")
        elif l in ["javascript", "php"]: fams.append("SCRIPTING")
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

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

log("Training ML Language Inference Model...")
lang_vect = HashingVectorizer(n_features=10000, analyzer='word', ngram_range=(1,2))
X_lang_tv = lang_vect.fit_transform(tv_df["text"].fillna("").astype(str).values)

lang_clf = SGDClassifier(loss='log_loss', max_iter=20, n_jobs=-1, random_state=42)
lang_clf.fit(X_lang_tv, tv_df["language"].astype(str).values)

log("Evaluating Inference Accuracy on test_sample...")
X_lang_ts = lang_vect.transform(ts_df["text"].fillna("").astype(str).values)
predicted_ts_langs = lang_clf.predict(X_lang_ts)

ts_predicted_families = map_to_family(predicted_ts_langs)
ts_actual_families = map_to_family(ts_df["language"].astype(str).values)

acc = np.mean(ts_predicted_families == ts_actual_families)
log(f"Family Inference Accuracy: {acc:.4f} -> {'PASS' if acc >= 0.85 else 'FAIL'}")

use_per_family = acc >= 0.85
if use_per_family: log("✓ Accuracy >= 0.85 → Enabling Per-family tracking.")
    
log("Inferring language family for 500k test samples...")
X_lang_te = lang_vect.transform(test_df["text"].fillna("").astype(str).values)
test_families = map_to_family(lang_clf.predict(X_lang_te))

tv_families = map_to_family(tv_df["language"].astype(str).values)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD FEATURES AND NEUTRALIZE
# ═══════════════════════════════════════════════════════════════════════════════
divider("Loading & Neutralizing Features")

try:
    X_tv = np.load(FEAT_DIR / "train_handcraft.npy")
    X_ts = np.load(FEAT_DIR / "test_sample_handcraft.npy")
    X_te = np.load(FEAT_DIR / "test_handcraft.npy")
    
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
        X_ts_tfidf = np.load(FEAT_DIR / "test_sample_tfidf.npy")
        X_te_tfidf = np.load(FEAT_DIR / "test_tfidf.npy")
        
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
def train_lgb(X_tr, y_tr, X_va_in=None, y_va_in=None, X_va_out=None, y_va_out=None, is_soft=False, best_iters=None):
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
        model.fit(X_tr, y_tr, eval_set=[(X_va_in, y_va_in), (X_va_out, y_va_out)], callbacks=callbacks)
    else:
        model.fit(X_tr, y_tr)
    return model

def train_xgb(X_tr, y_tr, X_va_in=None, y_va_in=None, X_va_out=None, y_va_out=None, is_soft=False, best_iters=None):
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
        model.fit(X_tr, y_tr, eval_set=[(X_va_in, y_va_in), (X_va_out, y_va_out)], verbose=False)
    else:
        model.fit(X_tr, y_tr, verbose=False)
    return model

def train_cat(X_tr, y_tr, X_va_in=None, y_va_in=None, X_va_out=None, y_va_out=None, is_soft=False, best_iters=None):
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

def train_ensemble(X_tr, y_tr, X_va_in=None, y_va_in=None, X_va_out=None, y_va_out=None, is_soft=False, best_iters=None):
    m_lgb = train_lgb(X_tr, y_tr, X_va_in, y_va_in, X_va_out, y_va_out, is_soft, best_iters)
    m_xgb = train_xgb(X_tr, y_tr, X_va_in, y_va_in, X_va_out, y_va_out, is_soft, best_iters)
    m_cat = train_cat(X_tr, y_tr, X_va_in, y_va_in, X_va_out, y_va_out, is_soft, best_iters)
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

y_ts_oof = np.zeros(len(X_ts))
test_preds_fold = np.zeros((len(X_te), 3))

tv_langs = tv_df["language"].astype(str).str.lower().values

# Generate 3 folds
# Fold 0: Validate on Python (from tv) + 1/3 of test_sample
# Fold 1: Validate on Java (from tv) + 1/3 of test_sample
# Fold 2: Validate on C++ (from tv)  + 1/3 of test_sample
ts_indices = np.arange(len(X_ts))
np.random.seed(42)
np.random.shuffle(ts_indices)
ts_folds = np.array_split(ts_indices, 3)

fold_langs = ["python", "java", "c++"]

for i in range(3):
    f_lang = fold_langs[i]
    val_tv_mask = (tv_langs == f_lang)
    tr_tv_mask  = ~val_tv_mask
    
    val_ts_idx = ts_folds[i]
    
    # SỬA LẠI THEO LỜI CHỈ DẪN CỦA CHUYÊN GIA: 
    # Tách TR khỏi test_sample, chỉ train trên tập X_tv [600k] để giữ độ OOD tính trong OOF "sạch sẽ" tuyệt đối
    X_tr = X_tv[tr_tv_mask]
    y_tr = y_tv[tr_tv_mask]
    
    # Validation In (Nội bộ dùng cho Early Stopping chi phối tree split)
    X_va_in = X_tv[val_tv_mask]
    y_va_in = y_tv[val_tv_mask]
    
    # Validation Out (test_sample dùng để track và tạo OOF Predictions)
    X_va_out = X_ts[val_ts_idx]
    y_va_out = y_ts[val_ts_idx]
    
    log(f"  Fold {f_lang}: Train {X_tr.shape[0]:,} | Val-In {X_va_in.shape[0]:,} | Val-Out {X_va_out.shape[0]:,}")
    
    models = train_ensemble(X_tr, y_tr, X_va_in, y_va_in, X_va_out, y_va_out)
    
    # OOF
    y_ts_oof[val_ts_idx] = predict_ensemble(models, X_va_out)
    
    # Test predict
    test_preds_fold[:, i] = predict_ensemble(models, X_te)

test_probs_initial = test_preds_fold.mean(axis=1)

log(f"\n  OOF Final Result before Pseudo-labeling:")
initial_f1 = f1_score(y_ts, (y_ts_oof > 0.5).astype(int), average="macro")
log(f"    Macro F1 on test_sample: {initial_f1:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PER-FAMILY CALIBRATION VIA LOGIT SHIFT
# ═══════════════════════════════════════════════════════════════════════════════
divider("Per-family Calibration via Logit Shift")

actual_families_np = np.array(ts_actual_families)
shifts = {"PYTHON": 0.0, "JVM_ISH": 0.0, "C_CPP": 0.0, "SCRIPTING": 0.0}

if use_per_family:
    for fam in shifts.keys():
        fam_idx = np.where(actual_families_np == fam)[0]
        if len(fam_idx) < 10:
            log(f"  Family {fam:10s}: Insufficient sample. Using shift 0.0")
            continue
            
        best_s, best_f1 = 0.0, 0.0
        # Search shift s over [-3.0, 3.0] space
        y_fam_logits = logit(y_ts_oof[fam_idx])
        for s in np.arange(-3.0, 3.1, 0.1):
            preds = (expit(y_fam_logits + s) > 0.5).astype(int)
            f1 = f1_score(y_ts[fam_idx], preds, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_s = float(s)
        shifts[fam] = best_s
        log(f"  Family {fam:10s}: selected shift {best_s:+.2f} (F1={best_f1:.4f} | n={len(fam_idx)})")

else:
    best_s, best_f1 = 0.0, 0.0
    y_logits = logit(y_ts_oof)
    for s in np.arange(-3.0, 3.1, 0.1):
        preds = (expit(y_logits + s) > 0.5).astype(int)
        f1 = f1_score(y_ts, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_s = float(s)
    for fam in shifts: shifts[fam] = best_s
    log(f"  Global calibration: selected shift {best_s:+.2f} (F1={best_f1:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
divider("Generate Submission")

test_preds = np.zeros(len(test_probs_initial), dtype=int)
test_logits = logit(test_probs_initial)

for fam, s in shifts.items():
    fam_idx = np.where(test_families == fam)[0]
    test_preds[fam_idx] = (expit(test_logits[fam_idx] + s) > 0.5).astype(int)

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
