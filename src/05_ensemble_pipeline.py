#!/usr/bin/env python3
"""
SemEval-2026 Task 13 Subtask A — 3-Track Ensemble Pipeline
==========================================================

Architecture:
  Track 1: TF-IDF (char n-gram) + Logistic Regression  (~30 min)
  Track 2: CodeBERT fine-tune (SequenceClassification)  (~4-5h)
  Track 3: Compression features + XGBoost               (~20 min)
  Ensemble: 0.25 * T1 + 0.50 * T2 + 0.25 * T3 → submission.csv

Target: Macro F1 > 0.70 on Kaggle hidden test set.
"""

import os, sys, gc, time, math, zlib, gzip, pickle, warnings
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# ============================================================================
# 0. CONFIG
# ============================================================================

@dataclass
class Config:
    data_dir: str = ""
    output_dir: str = ""

    # Track 2: CodeBERT
    backbone: str = "microsoft/codebert-base"
    max_length: int = 512
    train_sample: int = 100_000      # balanced training subset
    epochs: int = 3
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    seed: int = 42

    # Ensemble weights
    w_tfidf: float = 0.25
    w_codebert: float = 0.50
    w_compress: float = 0.25
    threshold: float = 0.5


# -- Auto-detect environment --
IN_KAGGLE = "kaggle_secrets" in sys.modules or os.path.exists("/kaggle/working")
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

if IN_KAGGLE:
    import glob
    search = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
    d_dir = str(Path(search[0]).parent) if search else "/kaggle/input/sem-eval-2026-task-13-subtask-a/Task_A"
    cfg = Config(data_dir=d_dir, output_dir="/kaggle/working/outputs_ensemble")
elif IN_COLAB:
    cfg = Config(data_dir="/content/data", output_dir="/content/outputs_ensemble")
else:
    _root = Path(__file__).resolve().parent.parent
    cfg = Config(
        data_dir=str(_root / "data" / "raw" / "Task_A"),
        output_dir=str(_root / "data" / "processed" / "ensemble"),
    )


def set_seed(seed):
    import random; random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_data(cfg):
    """Load train/val/test parquet files."""
    data_dir = Path(cfg.data_dir)
    print(f"\n{'='*60}")
    print(f"  LOADING DATA from {data_dir}")
    print(f"{'='*60}")

    df_train = pd.read_parquet(data_dir / "train.parquet")
    print(f"  Train: {len(df_train):,} rows, columns: {list(df_train.columns)}")

    val_path = data_dir / "validation.parquet"
    df_val = pd.read_parquet(val_path) if val_path.exists() else None
    if df_val is not None:
        print(f"  Val:   {len(df_val):,} rows")

    test_path = data_dir / "test.parquet"
    df_test = pd.read_parquet(test_path) if test_path.exists() else None
    if df_test is not None:
        print(f"  Test:  {len(df_test):,} rows")

    # Label distribution
    label_counts = df_train["label"].value_counts().sort_index()
    print(f"  Labels: {label_counts.to_dict()}")

    return df_train, df_val, df_test


def make_balanced_subset(df, n_total, seed=42):
    """Create a balanced subset with n_total/2 per class."""
    n_each = n_total // 2
    df0 = df[df["label"] == 0]
    df1 = df[df["label"] == 1]
    n_each = min(n_each, len(df0), len(df1))
    subset = pd.concat([
        df0.sample(n=n_each, random_state=seed),
        df1.sample(n=n_each, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  Balanced subset: {len(subset):,} ({n_each:,} per class)")
    return subset


# ============================================================================
# 2. TRACK 1: TF-IDF + Logistic Regression
# ============================================================================

def run_track1(train_codes, train_labels, val_codes, val_labels, test_codes, cache_dir):
    """
    Character-level TF-IDF + Calibrated Logistic Regression.

    Why char n-gram?
    - AI code uses common tokens uniformly; human code has more variety
    - char_wb catches indent patterns, spacing, naming styles at byte level
    - Fast (30 min), no GPU needed, strong baseline
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score, classification_report

    print(f"\n{'='*60}")
    print(f"  TRACK 1: TF-IDF + Logistic Regression")
    print(f"{'='*60}")

    cache_path = cache_dir / "track1_test_probs.npy"
    if cache_path.exists():
        print("  [Cache] Loading cached Track 1 predictions...")
        test_probs = np.load(cache_path)
        val_probs = np.load(cache_dir / "track1_val_probs.npy") if (cache_dir / "track1_val_probs.npy").exists() else None
        return test_probs, val_probs

    t0 = time.time()

    # TF-IDF: Character-level n-grams with word boundaries
    print("  Fitting TF-IDF (char_wb, n-gram 3-6, 100K features)...")
    tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 6),
        max_features=100_000,
        sublinear_tf=True,
        min_df=3,
        strip_accents='unicode',
        dtype=np.float32,
    )

    X_train = tfidf.fit_transform(train_codes)
    print(f"  TF-IDF matrix: {X_train.shape}")

    # Logistic Regression with calibration
    print("  Training CalibratedLR...")
    lr = LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1,
    )
    model = CalibratedClassifierCV(lr, cv=3, method='isotonic', n_jobs=-1)
    model.fit(X_train, train_labels)

    # Validate
    val_probs = None
    if val_codes is not None:
        X_val = tfidf.transform(val_codes)
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"  Val Macro F1: {val_f1:.4f}")
        print(f"  Val AI ratio: {val_preds.mean():.4f} (actual: {val_labels.mean():.4f})")
        print(classification_report(val_labels, val_preds,
                                    target_names=["Human", "AI"], digits=4))
        np.save(cache_dir / "track1_val_probs.npy", val_probs)

    # Test inference
    test_probs = None
    if test_codes is not None:
        print("  Predicting on test set...")
        X_test = tfidf.transform(test_codes)
        test_probs = model.predict_proba(X_test)[:, 1]
        np.save(cache_path, test_probs)
        print(f"  Test prob stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    # Save model
    with open(cache_dir / "track1_model.pkl", "wb") as f:
        pickle.dump({"tfidf": tfidf, "model": model}, f)

    elapsed = time.time() - t0
    print(f"  Track 1 done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return test_probs, val_probs


# ============================================================================
# 3. TRACK 3: Compression Features + XGBoost
# ============================================================================

def extract_compression_features(code: str) -> np.ndarray:
    """
    5 compression-based features that capture generation fingerprint.

    Why compression?
    - AI code is more compressible (repetitive patterns, predictable structure)
    - Human code has more variety (creative naming, diverse comments)
    - Model-agnostic: doesn't depend on any specific LLM
    - Extremely fast to compute
    """
    code_bytes = code.encode('utf-8', errors='replace')
    n = max(len(code_bytes), 1)

    # 1. zlib compression ratio (lower = more compressible = more AI-like)
    zlib_ratio = len(zlib.compress(code_bytes, level=6)) / n

    # 2. gzip compression ratio (different algorithm = complementary signal)
    gzip_ratio = len(gzip.compress(code_bytes, compresslevel=6)) / n

    # 3. Byte Shannon entropy
    freq = Counter(code_bytes)
    entropy = -sum((c / n) * math.log2(c / n) for c in freq.values()) if n > 0 else 0.0

    # 4. Unique byte ratio (AI code tends to use fewer distinct bytes)
    unique_ratio = len(freq) / 256.0

    # 5. Trigram repetition score (AI code has more repeated patterns)
    if len(code) >= 3:
        trigrams = [code[i:i+3] for i in range(len(code) - 2)]
        tri_freq = Counter(trigrams)
        repeated = sum(1 for t in trigrams if tri_freq[t] > 1)
        rep_score = repeated / max(len(trigrams), 1)
    else:
        rep_score = 0.0

    return np.array([zlib_ratio, gzip_ratio, entropy, unique_ratio, rep_score],
                    dtype=np.float32)


def run_track3(train_codes, train_labels, val_codes, val_labels, test_codes, cache_dir):
    """Compression features + XGBoost classifier."""
    import xgboost as xgb
    from sklearn.metrics import f1_score, classification_report

    print(f"\n{'='*60}")
    print(f"  TRACK 3: Compression Features + XGBoost")
    print(f"{'='*60}")

    cache_path = cache_dir / "track3_test_probs.npy"
    if cache_path.exists():
        print("  [Cache] Loading cached Track 3 predictions...")
        test_probs = np.load(cache_path)
        val_probs = np.load(cache_dir / "track3_val_probs.npy") if (cache_dir / "track3_val_probs.npy").exists() else None
        return test_probs, val_probs

    t0 = time.time()
    feat_names = ["zlib_ratio", "gzip_ratio", "entropy", "unique_byte_ratio", "trigram_rep"]

    # Extract features
    print(f"  Extracting {len(feat_names)} compression features for train...")
    X_train = np.stack([extract_compression_features(c) for c in tqdm(train_codes, desc="  Train feats")])

    X_val = None
    if val_codes is not None:
        X_val = np.stack([extract_compression_features(c) for c in tqdm(val_codes, desc="  Val feats")])

    X_test = None
    if test_codes is not None:
        X_test = np.stack([extract_compression_features(c) for c in tqdm(test_codes, desc="  Test feats")])

    # Train XGBoost
    print("  Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        early_stopping_rounds=20,
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1,
    )

    eval_set = [(X_val, val_labels)] if X_val is not None else None
    model.fit(X_train, train_labels, eval_set=eval_set, verbose=50)

    # Validate
    val_probs = None
    if X_val is not None:
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"  Val Macro F1: {val_f1:.4f}")
        print(f"  Val AI ratio: {val_preds.mean():.4f}")
        print(classification_report(val_labels, val_preds,
                                    target_names=["Human", "AI"], digits=4))
        np.save(cache_dir / "track3_val_probs.npy", val_probs)

    # Feature importance
    print("  Feature importance:")
    for name, imp in sorted(zip(feat_names, model.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"    {name:25s}: {imp:.4f}")

    # Test inference
    test_probs = None
    if X_test is not None:
        test_probs = model.predict_proba(X_test)[:, 1]
        np.save(cache_path, test_probs)
        print(f"  Test prob stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    with open(cache_dir / "track3_model.pkl", "wb") as f:
        pickle.dump(model, f)

    elapsed = time.time() - t0
    print(f"  Track 3 done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return test_probs, val_probs


# ============================================================================
# 4. TRACK 2: CodeBERT Fine-tune (Simple Sequence Classification)
# ============================================================================

def run_track2(train_codes, train_labels, val_codes, val_labels, test_codes, cache_dir, cfg):
    """
    Fine-tune CodeBERT for binary sequence classification.

    Why simple fine-tune instead of our old Hybrid approach?
    - AutoModelForSequenceClassification = 1 linear layer on CLS token
    - No Expert Branch, no SupCon, no Gated Fusion = fewer failure modes
    - HuggingFace Trainer handles mixed precision, grad accumulation, eval
    - CodeBERT pretrained on 6.4M code functions → understands code semantics
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import f1_score, classification_report

    print(f"\n{'='*60}")
    print(f"  TRACK 2: CodeBERT Fine-tune")
    print(f"{'='*60}")

    cache_path = cache_dir / "track2_test_probs.npy"
    if cache_path.exists():
        print("  [Cache] Loading cached Track 2 predictions...")
        test_probs = np.load(cache_path)
        val_probs = np.load(cache_dir / "track2_val_probs.npy") if (cache_dir / "track2_val_probs.npy").exists() else None
        return test_probs, val_probs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")

    # Dataset
    class CodeDataset(Dataset):
        def __init__(self, codes, labels, tokenizer, max_len):
            self.codes = codes
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.codes)
        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.codes[idx], max_length=self.max_len,
                padding="max_length", truncation=True, return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.backbone, num_labels=2,
        problem_type="single_label_classification",
    ).to(device)

    # Enable gradient checkpointing to save VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    ds_train = CodeDataset(train_codes, train_labels, tokenizer, cfg.max_length)
    ds_val = CodeDataset(val_codes, val_labels, tokenizer, cfg.max_length) if val_codes is not None else None

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size * 2, shuffle=False,
                        num_workers=2, pin_memory=True) if ds_val else None

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(dl_train) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(enabled=cfg.fp16)

    model_path = cache_dir / "track2_best.pt"
    best_f1 = 0.0

    for epoch in range(1, cfg.epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0; steps = 0
        t0 = time.time()
        pbar = tqdm(dl_train, desc=f"  Train E{epoch}", leave=False, mininterval=10)

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=cfg.fp16):
                out = model(**batch)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1
            if steps % 100 == 0:
                pbar.set_postfix(loss=f"{total_loss/steps:.4f}")

        avg_loss = total_loss / max(steps, 1)

        # --- Evaluate ---
        val_f1 = 0.0
        if dl_val:
            model.eval()
            all_probs = []; all_labels = []
            with torch.no_grad():
                for batch in tqdm(dl_val, desc="  Eval", leave=False, mininterval=10):
                    labels_cpu = batch.pop("labels").numpy()
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.amp.autocast("cuda", enabled=cfg.fp16):
                        out = model(**batch)
                    probs = F.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
                    all_probs.append(probs)
                    all_labels.append(labels_cpu)

            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            val_preds = (all_probs > 0.5).astype(int)
            val_f1 = f1_score(all_labels, val_preds, average='macro')
            ai_ratio = val_preds.mean()

            elapsed = time.time() - t0
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}, val_MacroF1={val_f1:.4f}, "
                  f"AI_ratio={ai_ratio:.4f}, time={elapsed:.0f}s")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), model_path)
                print(f"    → Saved best model (F1={best_f1:.4f})")
        else:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}, time={elapsed:.0f}s")
            torch.save(model.state_dict(), model_path)

    # Load best model for inference
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Val probs
    val_probs = None
    if dl_val:
        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in tqdm(dl_val, desc="  Val inference", leave=False, mininterval=10):
                batch.pop("labels", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=cfg.fp16):
                    out = model(**batch)
                all_probs.append(F.softmax(out.logits, dim=-1)[:, 1].cpu().numpy())
        val_probs = np.concatenate(all_probs)
        np.save(cache_dir / "track2_val_probs.npy", val_probs)

        val_preds = (val_probs > 0.5).astype(int)
        print(f"\n  Best Val Macro F1: {f1_score(val_labels, val_preds, average='macro'):.4f}")
        print(classification_report(val_labels, val_preds,
                                    target_names=["Human", "AI"], digits=4))

    # Test inference
    test_probs = None
    if test_codes is not None:
        print(f"\n  Predicting on {len(test_codes):,} test samples...")
        ds_test = CodeDataset(test_codes, None, tokenizer, cfg.max_length)
        dl_test = DataLoader(ds_test, batch_size=cfg.batch_size * 2, shuffle=False,
                             num_workers=2, pin_memory=True)
        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in tqdm(dl_test, desc="  Test inference", leave=False, mininterval=10):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=cfg.fp16):
                    out = model(**batch)
                all_probs.append(F.softmax(out.logits, dim=-1)[:, 1].cpu().numpy())
        test_probs = np.concatenate(all_probs)
        np.save(cache_path, test_probs)
        print(f"  Test prob stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Track 2 done (best val F1={best_f1:.4f})")
    return test_probs, val_probs


# ============================================================================
# 5. ENSEMBLE + SUBMISSION
# ============================================================================

def ensemble_and_submit(test_probs_list, val_probs_list, weights,
                        val_labels, df_test, cfg, cache_dir):
    """Weighted ensemble of all tracks."""
    from sklearn.metrics import f1_score, classification_report

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE (weights: {weights})")
    print(f"{'='*60}")

    # Filter out None tracks
    active = [(p, vp, w) for p, vp, w in zip(test_probs_list, val_probs_list, weights)
              if p is not None]

    if not active:
        print("  [ERROR] No tracks produced predictions!")
        return

    # Renormalize weights
    total_w = sum(w for _, _, w in active)
    active = [(p, vp, w / total_w) for p, vp, w in active]

    # Ensemble val probs
    if all(vp is not None for _, vp, _ in active):
        val_ens = sum(w * vp for _, vp, w in active)
        val_preds = (val_ens > cfg.threshold).astype(int)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"  Ensemble Val Macro F1: {val_f1:.4f}")
        print(f"  Ensemble Val AI ratio: {val_preds.mean():.4f} (actual: {val_labels.mean():.4f})")
        print(f"\n  Ensemble Classification Report (validation):")
        print(classification_report(val_labels, val_preds,
                                    target_names=["Human", "AI"], digits=4))

    # Ensemble test probs
    test_ens = sum(w * p for p, _, w in active)

    # Submission
    final_labels = (test_ens > cfg.threshold).astype(int)
    submission = pd.DataFrame({
        "ID": df_test["ID"].values,
        "label": final_labels,
    })

    if IN_KAGGLE:
        sub_path = Path("/kaggle/working/submission.csv")
    else:
        sub_path = cache_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)

    print(f"\n  Submission saved: {sub_path}")
    print(f"  Threshold: {cfg.threshold}")
    print(f"  Predicted AI ratio: {final_labels.mean():.4f}")
    print(f"  Label distribution: {dict(zip(*np.unique(final_labels, return_counts=True)))}")
    print(f"  Prob stats: mean={test_ens.mean():.4f}, std={test_ens.std():.4f}, "
          f"min={test_ens.min():.4f}, max={test_ens.max():.4f}")

    return sub_path


# ============================================================================
# 6. MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SemEval-2026 Task 13 — 3-Track Ensemble")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_sample", type=int, default=None)
    parser.add_argument("--skip_track2", action="store_true", help="Skip CodeBERT (Track 2)")
    args, _ = parser.parse_known_args()

    if args.data_dir: cfg.data_dir = args.data_dir
    if args.output_dir: cfg.output_dir = args.output_dir
    if args.train_sample: cfg.train_sample = args.train_sample

    cache_dir = Path(cfg.output_dir)
    os.makedirs(cache_dir, exist_ok=True)

    set_seed(cfg.seed)

    print(f"\n{'='*60}")
    print(f"  SemEval-2026 Task 13 — 3-Track Ensemble Pipeline")
    print(f"  Data:   {cfg.data_dir}")
    print(f"  Output: {cfg.output_dir}")
    print(f"  Sample: {cfg.train_sample:,}")
    print(f"{'='*60}")

    # -- Load Data --
    df_train, df_val, df_test = load_data(cfg)

    # -- Balanced subset --
    df_sub = make_balanced_subset(df_train, cfg.train_sample, seed=cfg.seed)

    train_codes = df_sub["code"].values
    train_labels = df_sub["label"].values
    val_codes = df_val["code"].values if df_val is not None else None
    val_labels = df_val["label"].values if df_val is not None else None
    test_codes = df_test["code"].values if df_test is not None else None

    # -- Track 1: TF-IDF + LR --
    t1_test, t1_val = run_track1(train_codes, train_labels,
                                  val_codes, val_labels,
                                  test_codes, cache_dir)

    # -- Track 3: Compression + XGBoost --
    t3_test, t3_val = run_track3(train_codes, train_labels,
                                  val_codes, val_labels,
                                  test_codes, cache_dir)

    # -- Track 2: CodeBERT Fine-tune --
    t2_test, t2_val = None, None
    if not args.skip_track2:
        t2_test, t2_val = run_track2(train_codes, train_labels,
                                      val_codes, val_labels,
                                      test_codes, cache_dir, cfg)
    else:
        print("\n  [SKIP] Track 2 (CodeBERT) skipped by --skip_track2 flag")

    # -- Ensemble --
    if df_test is not None:
        sub_path = ensemble_and_submit(
            test_probs_list=[t1_test, t2_test, t3_test],
            val_probs_list=[t1_val, t2_val, t3_val],
            weights=[cfg.w_tfidf, cfg.w_codebert, cfg.w_compress],
            val_labels=val_labels,
            df_test=df_test,
            cfg=cfg,
            cache_dir=cache_dir,
        )

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
