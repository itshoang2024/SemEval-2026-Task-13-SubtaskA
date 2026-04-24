#!/usr/bin/env python3
"""
SemEval-2026 Task 13A — GBDT Ensemble Pipeline v3
==================================================

Key insight: Test set has 8 languages but train only has 3.
Solution: Language-agnostic features + Gradient Boosting + test_sample calibration.

Data files used:
  - train.parquet + validation.parquet → merged training set (~600k)
  - test.parquet → hidden test set (~500k)
  - test_sample.parquet → 1000 labeled test samples for calibration

Architecture:
  Features: 48 handcrafted + 5 compression + TF-IDF/SVD (200 dims) = ~253 features
  Models: LightGBM + XGBoost + CatBoost (StratifiedKFold 5)
  Calibration: Grid search weights + threshold on test_sample.parquet
  Output: submission.csv

Runtime: ~1 hour on Kaggle (no GPU needed)
"""

import os, sys, re, math, zlib, gzip, warnings, pickle, time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# ============================================================================
# 0. CONFIG & DATA LOADING
# ============================================================================

IN_KAGGLE = os.path.exists("/kaggle/working")

if IN_KAGGLE:
    import glob
    search = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
    DATA_DIR = Path(search[0]).parent if search else Path("/kaggle/input/competitions/sem-eval-2026-task-13-subtask-a/Task_A")
    OUT_DIR = Path("/kaggle/working")
else:
    _root = Path(__file__).resolve().parent.parent
    DATA_DIR = _root / "data" / "raw" / "Task_A"
    OUT_DIR = _root / "data" / "processed" / "v3"

os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
N_FOLDS = 5

print(f"Data:   {DATA_DIR}")
print(f"Output: {OUT_DIR}")

# Load all data
print("\n[1/6] Loading data...")
train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df = pd.read_parquet(DATA_DIR / "validation.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")

test_sample_path = DATA_DIR / "test_sample.parquet"
test_sample_df = pd.read_parquet(test_sample_path) if test_sample_path.exists() else None

# Merge train + val for maximum training data
train_df = pd.concat([train_df, val_df], ignore_index=True)

print(f"  Train:       {len(train_df):,} rows | Languages: {sorted(train_df['language'].unique())}")
print(f"  Train labels: {train_df['label'].value_counts().sort_index().to_dict()}")
print(f"  Test:        {len(test_df):,} rows")
if test_sample_df is not None:
    print(f"  Test sample: {len(test_sample_df):,} rows | Languages: {sorted(test_sample_df['language'].unique())}")
    print(f"  Test sample labels: {test_sample_df['label'].value_counts().sort_index().to_dict()}")

y = train_df["label"].values


# ============================================================================
# 1. FEATURE EXTRACTION: 48 Handcrafted + 5 Compression
# ============================================================================

COMMON_KW = frozenset({
    'if', 'else', 'for', 'while', 'return', 'def', 'class',
    'function', 'var', 'let', 'const', 'int', 'string',
    'import', 'from', 'include', 'using', 'new', 'public',
    'private', 'static', 'void', 'true', 'false', 'null',
    'try', 'catch', 'throw', 'switch', 'case', 'break'
})
FUNC_KW = frozenset({'def', 'function', 'void', 'func', 'fn', 'sub', 'proc'})
IMPORT_KW = frozenset({'import', 'from', 'include', 'require', 'using', 'use', 'extern'})


def extract_features(code: str) -> list:
    """53 language-agnostic features: 48 handcrafted + 5 compression."""
    lines = code.split('\n')
    n_lines = max(len(lines), 1)
    n_chars = max(len(code), 1)

    # Line lengths
    line_lengths = [len(l) for l in lines]
    avg_ll = np.mean(line_lengths)
    std_ll = np.std(line_lengths) if len(line_lengths) > 1 else 0
    max_ll = max(line_lengths)

    # Indentation
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    avg_indent = np.mean(indents) if indents else 0
    std_indent = np.std(indents) if len(indents) > 1 else 0
    max_indent = max(indents) if indents else 0
    max_nest = max_indent // 4 if max_indent > 0 else 0

    # Whitespace
    space_ratio = code.count(' ') / n_chars
    tab_ratio = code.count('\t') / n_chars

    # Brackets
    n_op = code.count('('); n_cp = code.count(')')
    n_ob = code.count('{'); n_cb = code.count('}')

    # Tokens
    tokens = re.findall(r'\b\w+\b', code.lower())
    n_tokens = max(len(tokens), 1)
    kw_count = sum(1 for t in tokens if t in COMMON_KW)
    kw_ratio = kw_count / n_tokens

    # Comments
    comment_lines = sum(1 for l in lines if l.strip().startswith(('//', '#', '/*', '*')))
    comment_ratio = comment_lines / n_lines

    # Empty lines
    empty_lines = sum(1 for l in lines if not l.strip())
    empty_ratio = empty_lines / n_lines

    # Character entropy
    char_counts = Counter(code)
    total_c = sum(char_counts.values())
    entropy = -sum((c/total_c) * math.log2(c/total_c) for c in char_counts.values() if c > 0) if total_c > 0 else 0

    # Token uniqueness
    unique_tokens = len(set(tokens))
    token_uniq_ratio = unique_tokens / n_tokens

    # Identifiers
    identifiers = [t for t in tokens if t not in COMMON_KW and not t.isdigit()]
    avg_id_len = np.mean([len(t) for t in identifiers]) if identifiers else 0
    n_ids = max(len(identifiers), 1)

    # Semicolons, colons per line
    semi_pl = code.count(';') / n_lines
    colon_pl = code.count(':') / n_lines

    code_len = len(code)

    # Single char identifiers
    single_char = sum(1 for t in identifiers if len(t) == 1)
    single_ratio = single_char / n_ids

    # Snake case
    snake_count = sum(1 for t in identifiers if '_' in t)
    snake_ratio = snake_count / n_ids

    # Operator spacing
    operators = re.findall(r'[+\-*/%=<>!&|^~]', code)
    spaced_ops = len(re.findall(r'\s[+\-*/%=<>!&|^~]\s', code))
    op_space_ratio = spaced_ops / (len(operators) + 1) if operators else 0

    # Line variance
    ll_var = np.var(line_lengths) if len(line_lengths) > 1 else 0

    # Comment verbosity
    cmt_texts = [l.strip() for l in lines if l.strip().startswith(('//', '#', '/*', '*'))]
    if cmt_texts:
        cmt_wc = [len(re.findall(r'\b\w+\b', c)) for c in cmt_texts]
        avg_cmt_w = np.mean(cmt_wc)
        max_cmt_w = max(cmt_wc)
    else:
        avg_cmt_w = max_cmt_w = 0.0

    # Func density
    func_count = sum(1 for t in tokens if t in FUNC_KW)
    func_dens = func_count / n_lines

    # Long identifiers
    long_ids = sum(1 for t in identifiers if len(t) > 15)
    long_id_ratio = long_ids / n_ids

    # Duplicate lines
    non_empty = [l.rstrip() for l in lines if l.strip()]
    n_ne = max(len(non_empty), 1)
    dup_lines = n_ne - len(set(non_empty))
    dup_ratio = dup_lines / n_ne

    # Max consecutive empty
    max_consec = cur_consec = 0
    for l in lines:
        if not l.strip():
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    # Bigram repetition
    if len(tokens) > 1:
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        bg_counts = Counter(bigrams)
        bg_rep = sum(1 for c in bg_counts.values() if c > 1) / max(len(bg_counts), 1)
    else:
        bg_rep = 0.0

    # Import density
    imp_count = sum(1 for t in tokens if t in IMPORT_KW)
    imp_dens = imp_count / n_lines

    # Bracket imbalance
    total_imb = abs(n_op - n_cp) + abs(n_ob - n_cb) + abs(code.count('[') - code.count(']'))

    # Line length percentiles
    ll_p25 = float(np.percentile(line_lengths, 25))
    ll_iqr = float(np.percentile(line_lengths, 75)) - ll_p25

    # Indent delta entropy
    all_indents = [len(l) - len(l.lstrip()) for l in lines]
    deltas = [abs(all_indents[i+1] - all_indents[i]) for i in range(len(all_indents) - 1)]
    if deltas:
        dc = Counter(deltas); dt = sum(dc.values())
        indent_d_ent = -sum((c/dt)*math.log2(c/dt) for c in dc.values() if c > 0)
        mcd_ratio = dc.most_common(1)[0][1] / dt
    else:
        indent_d_ent = mcd_ratio = 0.0

    # Empty line gaps
    gaps = []; gap = 0
    for l in lines:
        if not l.strip():
            gap += 1
        else:
            if gap > 0: gaps.append(gap)
            gap = 0
    gap_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
    gap_mean = float(np.mean(gaps)) if gaps else 0.0

    # ID length std
    id_len_std = float(np.std([len(t) for t in identifiers])) if len(identifiers) > 1 else 0.0

    # Line length autocorrelation
    if len(line_lengths) > 2:
        ll_arr = np.array(line_lengths, dtype=float)
        if ll_arr.std() > 0:
            lag1 = float(np.corrcoef(ll_arr[:-1], ll_arr[1:])[0, 1])
            if np.isnan(lag1): lag1 = 0.0
        else:
            lag1 = 0.0
    else:
        lag1 = 0.0

    # Exact indent ratio
    exact_ind = sum(1 for i in all_indents if i % 4 == 0) / n_lines

    # Type hints
    th_count = len(re.findall(r':\s*\w+|<\w+>|->\s*\w+', code))
    th_dens = th_count / n_lines

    # === 5 COMPRESSION FEATURES ===
    code_bytes = code.encode('utf-8', errors='replace')
    nb = max(len(code_bytes), 1)
    zlib_r = len(zlib.compress(code_bytes, level=6)) / nb
    gzip_r = len(gzip.compress(code_bytes, compresslevel=6)) / nb
    byte_freq = Counter(code_bytes)
    byte_ent = -sum((c/nb)*math.log2(c/nb) for c in byte_freq.values()) if nb > 0 else 0.0
    uniq_byte = len(byte_freq) / 256.0
    if len(code) >= 3:
        tri = [code[i:i+3] for i in range(len(code)-2)]
        tf = Counter(tri)
        tri_rep = sum(1 for t in tri if tf[t] > 1) / max(len(tri), 1)
    else:
        tri_rep = 0.0

    return [
        # 48 handcrafted
        n_lines, avg_ll, std_ll, max_ll,
        avg_indent, std_indent, max_indent,
        space_ratio, tab_ratio,
        n_op, n_cp, n_ob, n_cb,
        kw_ratio, comment_ratio, empty_ratio,
        entropy, token_uniq_ratio, avg_id_len,
        semi_pl, colon_pl,
        code_len, n_tokens, unique_tokens,
        max_nest, single_ratio, snake_ratio,
        op_space_ratio, ll_var,
        avg_cmt_w, max_cmt_w,
        func_dens, long_id_ratio,
        dup_ratio, max_consec,
        bg_rep,
        imp_dens, total_imb,
        ll_p25, ll_iqr,
        indent_d_ent, mcd_ratio,
        gap_std, gap_mean,
        id_len_std, lag1,
        exact_ind, th_dens,
        # 5 compression
        zlib_r, gzip_r, byte_ent, uniq_byte, tri_rep,
    ]


FEATURE_NAMES = [
    'n_lines', 'avg_line_len', 'std_line_len', 'max_line_len',
    'avg_indent', 'std_indent', 'max_indent',
    'space_ratio', 'tab_ratio',
    'n_open_paren', 'n_close_paren', 'n_open_brace', 'n_close_brace',
    'keyword_ratio', 'comment_ratio', 'empty_ratio',
    'entropy', 'token_unique_ratio', 'avg_id_len',
    'semicolons_per_line', 'colons_per_line',
    'code_len', 'n_tokens', 'unique_tokens',
    'max_nesting_depth', 'single_char_ratio', 'snake_case_ratio',
    'operator_spacing_ratio', 'line_len_variance',
    'avg_comment_words', 'max_comment_words',
    'func_def_density', 'long_id_ratio',
    'duplicate_line_ratio', 'max_consec_empty',
    'bigram_repetition_ratio',
    'import_density', 'total_bracket_imbalance',
    'line_len_p25', 'line_len_iqr',
    'indent_delta_entropy', 'most_common_delta_ratio',
    'empty_gap_std', 'empty_gap_mean',
    'id_len_std', 'lag1_autocorr',
    'exact_indent_ratio', 'type_hint_density',
    # Compression features
    'zlib_ratio', 'gzip_ratio', 'byte_entropy', 'unique_byte_ratio', 'trigram_rep',
]


# ============================================================================
# 2. CODE NORMALIZATION + TF-IDF + SVD
# ============================================================================

def normalize_code(code: str) -> str:
    """
    Strip language-specific identifiers to make TF-IDF language-agnostic.
    Replace variable/function names with placeholders, keep structure.
    """
    # Remove string literals
    code = re.sub(r'"[^"]*"', '"STR"', code)
    code = re.sub(r"'[^']*'", "'STR'", code)
    # Remove comments
    code = re.sub(r'//.*', '// CMT', code)
    code = re.sub(r'#(?!include|define|ifdef|ifndef|endif|pragma).*', '# CMT', code)
    code = re.sub(r'/\*[\s\S]*?\*/', '/* CMT */', code)
    # Normalize numbers
    code = re.sub(r'\b\d+\.?\d*\b', 'NUM', code)
    return code


# ============================================================================
# 3. EXTRACT ALL FEATURES
# ============================================================================

print(f"\n[2/6] Extracting {len(FEATURE_NAMES)} handcrafted+compression features...")
t0 = time.time()
X_hand_train = np.array([extract_features(c) for c in tqdm(train_df['code'].values, desc="Train feats")])
X_hand_test = np.array([extract_features(c) for c in tqdm(test_df['code'].values, desc="Test feats")])
X_hand_sample = np.array([extract_features(c) for c in tqdm(test_sample_df['code'].values, desc="Sample feats")]) if test_sample_df is not None else None
print(f"  Done in {time.time()-t0:.0f}s | Train: {X_hand_train.shape}, Test: {X_hand_test.shape}")

# Replace NaN/Inf
X_hand_train = np.nan_to_num(X_hand_train, nan=0.0, posinf=0.0, neginf=0.0)
X_hand_test = np.nan_to_num(X_hand_test, nan=0.0, posinf=0.0, neginf=0.0)
if X_hand_sample is not None:
    X_hand_sample = np.nan_to_num(X_hand_sample, nan=0.0, posinf=0.0, neginf=0.0)

# TF-IDF on normalized code
print("\n[3/6] Building TF-IDF (normalized code) + SVD...")
t0 = time.time()
norm_train = [normalize_code(c) for c in tqdm(train_df['code'].values, desc="Norm train")]
norm_test = [normalize_code(c) for c in tqdm(test_df['code'].values, desc="Norm test")]
norm_sample = [normalize_code(c) for c in test_sample_df['code'].values] if test_sample_df is not None else None

tfidf = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 5), max_features=50_000,
    sublinear_tf=True, min_df=5, dtype=np.float32,
)
X_tfidf_train = tfidf.fit_transform(norm_train)
X_tfidf_test = tfidf.transform(norm_test)
X_tfidf_sample = tfidf.transform(norm_sample) if norm_sample is not None else None

# SVD to reduce to 200 dims
svd = TruncatedSVD(n_components=200, random_state=SEED)
X_svd_train = svd.fit_transform(X_tfidf_train).astype(np.float32)
X_svd_test = svd.transform(X_tfidf_test).astype(np.float32)
X_svd_sample = svd.transform(X_tfidf_sample).astype(np.float32) if X_tfidf_sample is not None else None
print(f"  SVD variance explained: {svd.explained_variance_ratio_.sum():.4f}")
print(f"  Done in {time.time()-t0:.0f}s")

# Combine all features
svd_names = [f'svd_{i}' for i in range(200)]
ALL_FEATURE_NAMES = FEATURE_NAMES + svd_names

X_train = np.hstack([X_hand_train, X_svd_train])
X_test = np.hstack([X_hand_test, X_svd_test])
X_sample = np.hstack([X_hand_sample, X_svd_sample]) if X_hand_sample is not None else None

print(f"  Combined features: {X_train.shape[1]} (53 hand+compress + 200 SVD)")
del X_hand_train, X_hand_test, X_svd_train, X_svd_test, norm_train, norm_test


# ============================================================================
# 4. TRAIN GBDT MODELS (StratifiedKFold 5)
# ============================================================================

print(f"\n[4/6] Training LightGBM + XGBoost + CatBoost ({N_FOLDS}-fold)...")

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# --- LightGBM ---
lgb_params = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'learning_rate': 0.05, 'num_leaves': 127, 'max_depth': -1,
    'min_child_samples': 50, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'is_unbalance': True, 'verbose': -1, 'n_jobs': -1,
}

oof_lgb = np.zeros(len(train_df))
lgb_models = []
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print("\n  === LightGBM ===")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtrain = lgb.Dataset(X_tr, y_tr, feature_name=ALL_FEATURE_NAMES)
    dval = lgb.Dataset(X_va, y_va, feature_name=ALL_FEATURE_NAMES, reference=dtrain)

    model = lgb.train(
        lgb_params, dtrain, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )

    probs = model.predict(X_va)
    oof_lgb[va_idx] = probs
    f1 = f1_score(y_va, (probs > 0.5).astype(int), average='macro')
    print(f"    Fold {fold+1}: F1={f1:.4f}")
    lgb_models.append(model)

print(f"  LGB OOF Macro F1: {f1_score(y, (oof_lgb > 0.5).astype(int), average='macro'):.4f}")

# --- XGBoost ---
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 10,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
    'tree_method': 'hist', 'verbosity': 0, 'n_jobs': -1,
}

oof_xgb = np.zeros(len(train_df))
xgb_models = []

print("\n  === XGBoost ===")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=ALL_FEATURE_NAMES)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=ALL_FEATURE_NAMES)

    model = xgb.train(
        xgb_params, dtrain, num_boost_round=2000,
        evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=200,
    )

    probs = model.predict(dval)
    oof_xgb[va_idx] = probs
    f1 = f1_score(y_va, (probs > 0.5).astype(int), average='macro')
    print(f"    Fold {fold+1}: F1={f1:.4f}")
    xgb_models.append(model)

print(f"  XGB OOF Macro F1: {f1_score(y, (oof_xgb > 0.5).astype(int), average='macro'):.4f}")

# --- CatBoost ---
cat_params = {
    'iterations': 2000, 'learning_rate': 0.05, 'depth': 6,
    'min_data_in_leaf': 10, 'subsample': 0.8, 'colsample_bylevel': 0.8,
    'auto_class_weights': 'Balanced', 'eval_metric': 'Logloss',
    'early_stopping_rounds': 50, 'verbose': 200,
    'random_seed': SEED, 'task_type': 'CPU',
}

oof_cat = np.zeros(len(train_df))
cat_models = []

print("\n  === CatBoost ===")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_pool = Pool(X_tr, y_tr, feature_names=ALL_FEATURE_NAMES)
    val_pool = Pool(X_va, y_va, feature_names=ALL_FEATURE_NAMES)

    model = CatBoostClassifier(**cat_params)
    model.fit(train_pool, eval_set=val_pool)

    probs = model.predict_proba(X_va)[:, 1]
    oof_cat[va_idx] = probs
    f1 = f1_score(y_va, (probs > 0.5).astype(int), average='macro')
    print(f"    Fold {fold+1}: F1={f1:.4f}")
    cat_models.append(model)

print(f"  CAT OOF Macro F1: {f1_score(y, (oof_cat > 0.5).astype(int), average='macro'):.4f}")


# ============================================================================
# 5. ENSEMBLE CALIBRATION ON test_sample.parquet
# ============================================================================

print(f"\n[5/6] Calibrating ensemble on test_sample ({len(test_sample_df) if test_sample_df is not None else 0} samples)...")

# Predict test_sample with all models
if X_sample is not None and test_sample_df is not None:
    y_sample = test_sample_df['label'].values

    sample_lgb = np.mean([m.predict(X_sample) for m in lgb_models], axis=0)
    dtest_s = xgb.DMatrix(X_sample, feature_names=ALL_FEATURE_NAMES)
    sample_xgb = np.mean([m.predict(dtest_s) for m in xgb_models], axis=0)
    sample_cat = np.mean([m.predict_proba(X_sample)[:, 1] for m in cat_models], axis=0)

    # Individual scores
    for name, probs in [('LGB', sample_lgb), ('XGB', sample_xgb), ('CAT', sample_cat)]:
        f1 = f1_score(y_sample, (probs > 0.5).astype(int), average='macro')
        print(f"  {name} solo @0.5: F1={f1:.4f}")

    # Grid search: weights + threshold
    best_f1 = 0; best_w = (1/3, 1/3, 1/3); best_thr = 0.5

    for w1 in np.arange(0.1, 0.8, 0.05):
        for w2 in np.arange(0.1, 0.8, 0.05):
            w3 = 1.0 - w1 - w2
            if w3 < 0.05:
                continue
            blend = w1 * sample_lgb + w2 * sample_xgb + w3 * sample_cat
            for thr in np.arange(0.2, 0.9, 0.01):
                f1 = f1_score(y_sample, (blend > thr).astype(int), average='macro')
                if f1 > best_f1:
                    best_f1 = f1; best_w = (w1, w2, w3); best_thr = thr

    print(f"\n  🎯 Best ensemble: LGB={best_w[0]:.2f}, XGB={best_w[1]:.2f}, CAT={best_w[2]:.2f}")
    print(f"     Best threshold: {best_thr:.2f}")
    print(f"     Best test_sample F1: {best_f1:.4f}")

    # Per-language breakdown
    best_blend = best_w[0]*sample_lgb + best_w[1]*sample_xgb + best_w[2]*sample_cat
    best_preds = (best_blend > best_thr).astype(int)
    print(f"\n  Classification Report (test_sample):")
    print(classification_report(y_sample, best_preds, target_names=['Human', 'AI'], digits=4))

    print("  Per-language F1:")
    for lang in sorted(test_sample_df['language'].unique()):
        mask = (test_sample_df['language'] == lang).values
        lf1 = f1_score(y_sample[mask], best_preds[mask], average='macro')
        print(f"    {lang:15s}: F1={lf1:.4f} (n={mask.sum()})")
else:
    best_w = (1/3, 1/3, 1/3)
    best_thr = 0.5
    print("  ⚠️ No test_sample.parquet found — using default weights")


# ============================================================================
# 6. FINAL PREDICTIONS + SUBMISSION
# ============================================================================

print(f"\n[6/6] Generating submission...")

# Test predictions from all models
test_lgb = np.mean([m.predict(X_test) for m in lgb_models], axis=0)
dtest = xgb.DMatrix(X_test, feature_names=ALL_FEATURE_NAMES)
test_xgb = np.mean([m.predict(dtest) for m in xgb_models], axis=0)
test_cat = np.mean([m.predict_proba(X_test)[:, 1] for m in cat_models], axis=0)

# Ensemble
test_probs = best_w[0]*test_lgb + best_w[1]*test_xgb + best_w[2]*test_cat
test_preds = (test_probs > best_thr).astype(int)

submission = pd.DataFrame({'ID': test_df['ID'], 'label': test_preds})
sub_path = OUT_DIR / "submission.csv"
submission.to_csv(sub_path, index=False)

print(f"  📄 Submission: {sub_path}")
print(f"     Threshold: {best_thr:.2f}")
print(f"     AI: {test_preds.sum():,}, Human: {(test_preds == 0).sum():,}")
print(f"     AI ratio: {test_preds.mean():.4f}")
print(f"     Prob stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

# Feature importance (from best LGB model)
if lgb_models:
    imp = lgb_models[0].feature_importance(importance_type='gain')
    feat_imp = sorted(zip(ALL_FEATURE_NAMES, imp), key=lambda x: -x[1])
    print(f"\n  Top 15 features (LGB gain):")
    for name, score in feat_imp[:15]:
        print(f"    {name:30s}: {score:.1f}")

print(f"\n{'='*60}")
print(f"  ✅ PIPELINE v3 COMPLETE")
print(f"{'='*60}")
