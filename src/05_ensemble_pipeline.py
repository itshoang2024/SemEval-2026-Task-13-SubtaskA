#!/usr/bin/env python3
"""
SemEval-2026 Task 13 Subtask A — 3-Track Ensemble Pipeline (Professional)
==========================================================================

Features:
  - Checkpoint/Resume: Train 100k samples per session, continue next session
  - Monitoring: Per-step CSV log, epoch summaries, GPU memory tracking
  - Incremental: Track which samples are already seen, pick new ones
  - Caching: Track 1/3 predictions cached, skip if already computed

Architecture:
  Track 1: TF-IDF (char n-gram) + Logistic Regression    [CPU, ~30 min, full data]
  Track 2: CodeBERT fine-tune (SequenceClassification)    [GPU, ~4h/100k, incremental]
  Track 3: Compression features + XGBoost                 [CPU, ~20 min, full data]
  Ensemble: w1*T1 + w2*T2 + w3*T3 → submission.csv

Usage:
  # Session 1: Train on first 100k → get submission
  python 05_ensemble_pipeline.py

  # Session 2: Resume from checkpoint, train on NEXT 100k
  python 05_ensemble_pipeline.py

  # Skip CodeBERT if you only want Track 1 + Track 3
  python 05_ensemble_pipeline.py --skip_track2
"""

import os, sys, gc, json, time, math, zlib, gzip, pickle, warnings, csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import Counter
from datetime import datetime

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
    codebert_batch_size: int = 100_000  # samples PER SESSION (incremental)
    epochs_per_session: int = 3         # epochs to train per session
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
    d_dir = str(Path(search[0]).parent) if search else \
            "/kaggle/input/sem-eval-2026-task-13-subtask-a/Task_A"
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
# 1. MONITORING — CSV Logger + Console
# ============================================================================

class TrainingMonitor:
    """Log metrics to CSV + console with GPU memory tracking."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.start_time = time.time()
        self._csv_file = None
        self._writer = None

        # Initialize CSV
        is_new = not log_path.exists()
        self._csv_file = open(log_path, "a", newline="")
        self._writer = csv.writer(self._csv_file)
        if is_new:
            self._writer.writerow([
                "timestamp", "session", "epoch", "step", "phase",
                "loss", "lr", "val_macro_f1", "val_ai_ratio",
                "gpu_mem_gb", "elapsed_sec"
            ])
            self._csv_file.flush()

    def log_step(self, session, epoch, step, loss, lr, phase="train"):
        gpu_mem = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
        except:
            pass

        elapsed = time.time() - self.start_time
        self._writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            session, epoch, step, phase,
            f"{loss:.6f}", f"{lr:.8f}", "", "",
            f"{gpu_mem:.2f}", f"{elapsed:.0f}"
        ])
        if step % 200 == 0:
            self._csv_file.flush()

    def log_eval(self, session, epoch, val_f1, val_ai_ratio):
        elapsed = time.time() - self.start_time
        self._writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            session, epoch, 0, "eval",
            "", "", f"{val_f1:.6f}", f"{val_ai_ratio:.4f}",
            "", f"{elapsed:.0f}"
        ])
        self._csv_file.flush()

    def close(self):
        if self._csv_file:
            self._csv_file.close()

    def print_summary(self, msg):
        elapsed = time.time() - self.start_time
        h, m = divmod(int(elapsed), 3600)
        m, s = divmod(m, 60)
        print(f"  [{h:02d}:{m:02d}:{s:02d}] {msg}")


# ============================================================================
# 2. TRAINING STATE — Checkpoint Management
# ============================================================================

@dataclass
class TrainingState:
    """Persistent state for incremental CodeBERT training."""
    session: int = 0                    # Current session number
    total_steps: int = 0                # Cumulative steps across sessions
    total_samples_seen: int = 0         # Cumulative training samples
    best_val_f1: float = 0.0            # Best validation Macro F1
    seen_indices: list = field(default_factory=list)  # Train indices already used
    completed: bool = False             # True if all data has been trained on

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingState":
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def save_checkpoint(model, optimizer, scheduler, scaler, state, ckpt_dir: Path):
    """Save full training state for resume."""
    import torch
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), ckpt_dir / "scaler.pt")
    state.save(ckpt_dir / "training_state.json")
    print(f"    📁 Checkpoint saved: session={state.session}, "
          f"samples_seen={state.total_samples_seen:,}, best_f1={state.best_val_f1:.4f}")


def load_checkpoint(model, optimizer, scheduler, scaler, ckpt_dir: Path, device):
    """Load full training state for resume."""
    import torch
    state = TrainingState.load(ckpt_dir / "training_state.json")

    model_path = ckpt_dir / "model.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"    ✅ Model weights loaded")

    opt_path = ckpt_dir / "optimizer.pt"
    if opt_path.exists() and optimizer is not None:
        optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))
        print(f"    ✅ Optimizer state loaded")

    sch_path = ckpt_dir / "scheduler.pt"
    if sch_path.exists() and scheduler is not None:
        scheduler.load_state_dict(torch.load(sch_path, weights_only=True))
        print(f"    ✅ Scheduler state loaded")

    scaler_path = ckpt_dir / "scaler.pt"
    if scaler_path.exists() and scaler is not None:
        scaler.load_state_dict(torch.load(scaler_path, weights_only=True))

    print(f"    📋 Resuming: session={state.session}, "
          f"samples_seen={state.total_samples_seen:,}, best_f1={state.best_val_f1:.4f}")
    return state


# ============================================================================
# 3. DATA LOADING
# ============================================================================

def load_data(cfg):
    data_dir = Path(cfg.data_dir)
    print(f"\n{'='*60}")
    print(f"  DATA LOADING — {data_dir}")
    print(f"{'='*60}")

    df_train = pd.read_parquet(data_dir / "train.parquet")
    print(f"  Train: {len(df_train):,} rows | Labels: {df_train['label'].value_counts().sort_index().to_dict()}")

    val_path = data_dir / "validation.parquet"
    df_val = pd.read_parquet(val_path) if val_path.exists() else None
    if df_val is not None:
        print(f"  Val:   {len(df_val):,} rows | Labels: {df_val['label'].value_counts().sort_index().to_dict()}")

    test_path = data_dir / "test.parquet"
    df_test = pd.read_parquet(test_path) if test_path.exists() else None
    if df_test is not None:
        print(f"  Test:  {len(df_test):,} rows")

    return df_train, df_val, df_test


def pick_incremental_batch(df_train, batch_size, seen_indices, seed=42):
    """Pick next balanced batch of unseen samples."""
    seen_set = set(seen_indices)
    remaining = df_train[~df_train.index.isin(seen_set)]

    if len(remaining) == 0:
        print("  ⚠️  All training samples have been seen!")
        return None, seen_indices

    # Balanced sampling from remaining
    n_each = batch_size // 2
    df0 = remaining[remaining["label"] == 0]
    df1 = remaining[remaining["label"] == 1]
    n_each = min(n_each, len(df0), len(df1))

    if n_each == 0:
        print("  ⚠️  Not enough remaining samples for balanced batch!")
        return None, seen_indices

    batch = pd.concat([
        df0.sample(n=n_each, random_state=seed),
        df1.sample(n=n_each, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=False)

    new_indices = batch["index"].tolist() if "index" in batch.columns else batch.index.tolist()
    seen_indices = seen_indices + new_indices
    batch = batch.drop(columns=["index"], errors="ignore")

    print(f"  📊 Incremental batch: {len(batch):,} samples ({n_each:,}/class)")
    print(f"     Total seen so far: {len(seen_indices):,} / {len(df_train):,}")
    return batch, seen_indices


# ============================================================================
# 4. TRACK 1: TF-IDF + Logistic Regression (full data, fast)
# ============================================================================

def run_track1(train_codes, train_labels, val_codes, val_labels, test_codes, cache_dir):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import f1_score, classification_report

    print(f"\n{'='*60}")
    print(f"  TRACK 1: TF-IDF + Logistic Regression (full data)")
    print(f"{'='*60}")

    cache_test = cache_dir / "track1_test_probs.npy"
    cache_val = cache_dir / "track1_val_probs.npy"
    if cache_test.exists() and cache_val.exists():
        print("  ♻️  Loading cached Track 1 predictions...")
        return np.load(cache_test), np.load(cache_val)

    t0 = time.time()

    print(f"  Fitting TF-IDF on {len(train_codes):,} samples...")
    tfidf = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 6), max_features=100_000,
        sublinear_tf=True, min_df=3, strip_accents='unicode', dtype=np.float32,
    )
    X_train = tfidf.fit_transform(train_codes)
    print(f"  TF-IDF shape: {X_train.shape}")

    print("  Training CalibratedLR (3-fold)...")
    lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                            class_weight='balanced', n_jobs=-1)
    model = CalibratedClassifierCV(lr, cv=3, method='isotonic', n_jobs=-1)
    model.fit(X_train, train_labels)

    # Validate
    val_probs = None
    if val_codes is not None:
        X_val = tfidf.transform(val_codes)
        val_probs = model.predict_proba(X_val)[:, 1]
        vp = (val_probs > 0.5).astype(int)
        print(f"  ✅ Val Macro F1: {f1_score(val_labels, vp, average='macro'):.4f}")
        print(f"     Val AI ratio: {vp.mean():.4f} (actual: {val_labels.mean():.4f})")
        print(classification_report(val_labels, vp, target_names=["Human","AI"], digits=4))
        np.save(cache_val, val_probs)

    # Test
    test_probs = None
    if test_codes is not None:
        print(f"  Predicting {len(test_codes):,} test samples...")
        test_probs = model.predict_proba(tfidf.transform(test_codes))[:, 1]
        np.save(cache_test, test_probs)
        print(f"  Test stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    with open(cache_dir / "track1_model.pkl", "wb") as f:
        pickle.dump({"tfidf": tfidf, "model": model}, f)

    print(f"  ⏱️  Track 1 done in {time.time()-t0:.0f}s")
    return test_probs, val_probs


# ============================================================================
# 5. TRACK 3: Compression Features + XGBoost (full data, fast)
# ============================================================================

def extract_compression_features(code: str) -> np.ndarray:
    code_bytes = code.encode('utf-8', errors='replace')
    n = max(len(code_bytes), 1)
    zlib_r = len(zlib.compress(code_bytes, level=6)) / n
    gzip_r = len(gzip.compress(code_bytes, compresslevel=6)) / n
    freq = Counter(code_bytes)
    entropy = -sum((c/n)*math.log2(c/n) for c in freq.values()) if n > 0 else 0.0
    uniq = len(freq) / 256.0
    if len(code) >= 3:
        tri = [code[i:i+3] for i in range(len(code)-2)]
        tf = Counter(tri)
        rep = sum(1 for t in tri if tf[t] > 1) / max(len(tri), 1)
    else:
        rep = 0.0
    return np.array([zlib_r, gzip_r, entropy, uniq, rep], dtype=np.float32)


def run_track3(train_codes, train_labels, val_codes, val_labels, test_codes, cache_dir):
    import xgboost as xgb
    from sklearn.metrics import f1_score, classification_report

    print(f"\n{'='*60}")
    print(f"  TRACK 3: Compression Features + XGBoost (full data)")
    print(f"{'='*60}")

    cache_test = cache_dir / "track3_test_probs.npy"
    cache_val = cache_dir / "track3_val_probs.npy"
    if cache_test.exists() and cache_val.exists():
        print("  ♻️  Loading cached Track 3 predictions...")
        return np.load(cache_test), np.load(cache_val)

    t0 = time.time()
    feat_names = ["zlib_ratio","gzip_ratio","entropy","unique_bytes","trigram_rep"]

    print(f"  Extracting features for {len(train_codes):,} train samples...")
    X_tr = np.stack([extract_compression_features(c) for c in tqdm(train_codes, desc="  Train", leave=False)])
    X_va = np.stack([extract_compression_features(c) for c in tqdm(val_codes, desc="  Val", leave=False)]) if val_codes is not None else None
    X_te = np.stack([extract_compression_features(c) for c in tqdm(test_codes, desc="  Test", leave=False)]) if test_codes is not None else None

    print("  Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        early_stopping_rounds=20, random_state=42, n_jobs=-1,
    )
    eval_set = [(X_va, val_labels)] if X_va is not None else None
    model.fit(X_tr, train_labels, eval_set=eval_set, verbose=50)

    # Validate
    val_probs = None
    if X_va is not None:
        val_probs = model.predict_proba(X_va)[:, 1]
        vp = (val_probs > 0.5).astype(int)
        print(f"  ✅ Val Macro F1: {f1_score(val_labels, vp, average='macro'):.4f}")
        print(classification_report(val_labels, vp, target_names=["Human","AI"], digits=4))
        np.save(cache_val, val_probs)

    # Importances
    print("  Feature importance:")
    for nm, imp in sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1]):
        print(f"    {nm:20s}: {imp:.4f}")

    # Test
    test_probs = None
    if X_te is not None:
        test_probs = model.predict_proba(X_te)[:, 1]
        np.save(cache_test, test_probs)
        print(f"  Test stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    with open(cache_dir / "track3_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"  ⏱️  Track 3 done in {time.time()-t0:.0f}s")
    return test_probs, val_probs


# ============================================================================
# 6. TRACK 2: CodeBERT Fine-tune (INCREMENTAL with checkpointing)
# ============================================================================

def run_track2(df_train, val_codes, val_labels, test_codes, cache_dir, cfg):
    """
    Incremental CodeBERT fine-tuning with full checkpoint/resume.

    Each session trains on `cfg.codebert_batch_size` NEW samples.
    State is saved so next session continues from where we left off.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import get_cosine_schedule_with_warmup
    from sklearn.metrics import f1_score, classification_report

    print(f"\n{'='*60}")
    print(f"  TRACK 2: CodeBERT Fine-tune (INCREMENTAL)")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU: {torch.cuda.get_device_name()} ({gpu.total_memory / 1e9:.1f} GB)")

    ckpt_dir = cache_dir / "track2_checkpoint"
    log_path = cache_dir / "track2_train_log.csv"
    monitor = TrainingMonitor(log_path)

    # -- Load or initialize state --
    state = TrainingState.load(ckpt_dir / "training_state.json")

    if state.completed:
        print("  ✅ All training data has been seen. Loading best model for inference...")
        best_path = cache_dir / "track2_best.pt"
        if not best_path.exists():
            print("  ❌ No best model found!")
            return None, None
        # Skip to inference
        return _track2_inference(best_path, val_codes, val_labels, test_codes,
                                cache_dir, cfg, device)

    state.session += 1
    print(f"\n  === SESSION {state.session} ===")
    print(f"  Samples trained so far: {state.total_samples_seen:,}")

    # -- Pick next incremental batch --
    batch_df, state.seen_indices = pick_incremental_batch(
        df_train, cfg.codebert_batch_size, state.seen_indices, seed=cfg.seed + state.session
    )
    if batch_df is None:
        state.completed = True
        state.save(ckpt_dir / "training_state.json")
        return _track2_inference(cache_dir / "track2_best.pt", val_codes, val_labels,
                                test_codes, cache_dir, cfg, device)

    train_codes = batch_df["code"].values
    train_labels_arr = batch_df["label"].values

    # -- Dataset --
    class CodeDataset(Dataset):
        def __init__(self, codes, labels, tokenizer, max_len):
            self.codes = codes; self.labels = labels
            self.tokenizer = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.codes)
        def __getitem__(self, idx):
            enc = self.tokenizer(self.codes[idx], max_length=self.max_len,
                                 padding="max_length", truncation=True, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.backbone, num_labels=2, problem_type="single_label_classification"
    ).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    ds_train = CodeDataset(train_codes, train_labels_arr, tokenizer, cfg.max_length)
    ds_val = CodeDataset(val_codes, val_labels, tokenizer, cfg.max_length) if val_codes is not None else None

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size*2, shuffle=False,
                        num_workers=2, pin_memory=True) if ds_val else None

    # -- Optimizer, Scheduler, Scaler --
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(dl_train) * cfg.epochs_per_session
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler(enabled=cfg.fp16)

    # -- Resume checkpoint if exists --
    if (ckpt_dir / "model.pt").exists() and state.session > 1:
        print("  🔄 Resuming from checkpoint...")
        state_loaded = load_checkpoint(model, optimizer, scheduler, scaler, ckpt_dir, device)
        # Keep the new session number but restore best_f1 and seen_indices
        state.best_val_f1 = state_loaded.best_val_f1

    best_model_path = cache_dir / "track2_best.pt"

    # ---- TRAINING LOOP ----
    for epoch in range(1, cfg.epochs_per_session + 1):
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
            state.total_steps += 1

            # Monitor log
            current_lr = scheduler.get_last_lr()[0]
            monitor.log_step(state.session, epoch, state.total_steps, loss.item(), current_lr)

            if steps % 100 == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                pbar.set_postfix(loss=f"{total_loss/steps:.4f}", lr=f"{current_lr:.2e}",
                                 gpu=f"{gpu_mem:.1f}G")

        avg_loss = total_loss / max(steps, 1)

        # ---- EVALUATE ----
        val_f1 = 0.0
        ai_ratio = 0.0
        if dl_val:
            model.eval()
            all_probs = []; all_labels_list = []
            with torch.no_grad():
                for batch in tqdm(dl_val, desc="  Eval", leave=False, mininterval=10):
                    labels_cpu = batch.pop("labels").numpy()
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.amp.autocast("cuda", enabled=cfg.fp16):
                        out = model(**batch)
                    all_probs.append(F.softmax(out.logits, dim=-1)[:, 1].cpu().numpy())
                    all_labels_list.append(labels_cpu)

            all_probs_arr = np.concatenate(all_probs)
            all_labels_arr = np.concatenate(all_labels_list)
            val_preds = (all_probs_arr > 0.5).astype(int)
            val_f1 = f1_score(all_labels_arr, val_preds, average='macro')
            ai_ratio = val_preds.mean()

            monitor.log_eval(state.session, epoch, val_f1, ai_ratio)

        elapsed = time.time() - t0
        monitor.print_summary(
            f"Session {state.session} | Epoch {epoch}/{cfg.epochs_per_session} | "
            f"loss={avg_loss:.4f} | val_F1={val_f1:.4f} | AI_ratio={ai_ratio:.4f} | "
            f"time={elapsed:.0f}s"
        )

        # ---- SAVE BEST ----
        if val_f1 > state.best_val_f1:
            state.best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            monitor.print_summary(f"🏆 New best model! F1={val_f1:.4f}")

    # ---- SAVE CHECKPOINT ----
    state.total_samples_seen += len(train_codes)

    # Check if all data seen
    remaining = len(df_train) - len(set(state.seen_indices))
    if remaining < cfg.codebert_batch_size // 4:
        state.completed = True
        monitor.print_summary(f"✅ Training data exhausted ({state.total_samples_seen:,} seen)")

    save_checkpoint(model, optimizer, scheduler, scaler, state, ckpt_dir)
    monitor.close()

    # ---- INFERENCE ----
    del model, optimizer, scheduler, scaler, ds_train, dl_train
    gc.collect(); torch.cuda.empty_cache()

    return _track2_inference(best_model_path, val_codes, val_labels,
                             test_codes, cache_dir, cfg, device)


def _track2_inference(model_path, val_codes, val_labels, test_codes, cache_dir, cfg, device):
    """Run inference with best saved model."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import f1_score, classification_report

    if not model_path.exists():
        print("  ❌ No model found for inference!")
        return None, None

    print(f"\n  Running Track 2 inference with {model_path.name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.backbone, num_labels=2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    class CodeDS(Dataset):
        def __init__(self, codes, labels, tok, ml):
            self.codes=codes; self.labels=labels; self.tok=tok; self.ml=ml
        def __len__(self): return len(self.codes)
        def __getitem__(self, i):
            enc = self.tok(self.codes[i], max_length=self.ml,
                           padding="max_length", truncation=True, return_tensors="pt")
            item = {k:v.squeeze(0) for k,v in enc.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item

    # Val
    val_probs = None
    if val_codes is not None:
        dl = DataLoader(CodeDS(val_codes, val_labels, tokenizer, cfg.max_length),
                        batch_size=cfg.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
        probs_list = []
        with torch.no_grad():
            for batch in tqdm(dl, desc="  Val inference", leave=False, mininterval=10):
                batch.pop("labels", None)
                batch = {k:v.to(device) for k,v in batch.items()}
                with torch.amp.autocast("cuda", enabled=cfg.fp16):
                    out = model(**batch)
                probs_list.append(F.softmax(out.logits, dim=-1)[:,1].cpu().numpy())
        val_probs = np.concatenate(probs_list)
        np.save(cache_dir / "track2_val_probs.npy", val_probs)
        vp = (val_probs > 0.5).astype(int)
        print(f"  ✅ Track 2 Val Macro F1: {f1_score(val_labels, vp, average='macro'):.4f}")
        print(classification_report(val_labels, vp, target_names=["Human","AI"], digits=4))

    # Test
    test_probs = None
    if test_codes is not None:
        print(f"  Predicting {len(test_codes):,} test samples...")
        dl = DataLoader(CodeDS(test_codes, None, tokenizer, cfg.max_length),
                        batch_size=cfg.batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
        probs_list = []
        with torch.no_grad():
            for batch in tqdm(dl, desc="  Test inference", leave=False, mininterval=10):
                batch = {k:v.to(device) for k,v in batch.items()}
                with torch.amp.autocast("cuda", enabled=cfg.fp16):
                    out = model(**batch)
                probs_list.append(F.softmax(out.logits, dim=-1)[:,1].cpu().numpy())
        test_probs = np.concatenate(probs_list)
        np.save(cache_dir / "track2_test_probs.npy", test_probs)
        print(f"  Test stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return test_probs, val_probs


# ============================================================================
# 7. ENSEMBLE + SUBMISSION
# ============================================================================

def ensemble_and_submit(test_probs_list, val_probs_list, weights,
                        val_labels, df_test, cfg, cache_dir):
    from sklearn.metrics import f1_score, classification_report

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE — weights: TF-IDF={weights[0]}, CodeBERT={weights[1]}, Compress={weights[2]}")
    print(f"{'='*60}")

    names = ["TF-IDF", "CodeBERT", "Compression"]

    # Filter active tracks
    active = [(p, vp, w, n) for p, vp, w, n in zip(test_probs_list, val_probs_list, weights, names)
              if p is not None]

    if not active:
        print("  ❌ No tracks produced predictions!")
        return None

    # Print individual track stats
    for p, vp, w, n in active:
        if vp is not None and val_labels is not None:
            f1 = f1_score(val_labels, (vp > 0.5).astype(int), average='macro')
            print(f"  {n:12s}: val_F1={f1:.4f}, test_mean={p.mean():.4f}, weight={w:.2f}")

    # Renormalize weights
    total_w = sum(w for _, _, w, _ in active)
    active_norm = [(p, vp, w/total_w, n) for p, vp, w, n in active]

    # Ensemble
    if all(vp is not None for _, vp, _, _ in active_norm):
        val_ens = sum(w * vp for _, vp, w, _ in active_norm)
        vp = (val_ens > cfg.threshold).astype(int)
        val_f1 = f1_score(val_labels, vp, average='macro')
        print(f"\n  🎯 Ensemble Val Macro F1: {val_f1:.4f}")
        print(f"     Ensemble Val AI ratio: {vp.mean():.4f} (actual: {val_labels.mean():.4f})")
        print(classification_report(val_labels, vp, target_names=["Human","AI"], digits=4))

    test_ens = sum(w * p for p, _, w, _ in active_norm)
    final = (test_ens > cfg.threshold).astype(int)

    submission = pd.DataFrame({"ID": df_test["ID"].values, "label": final})
    sub_path = Path("/kaggle/working/submission.csv") if IN_KAGGLE else cache_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)

    print(f"\n  📄 Submission: {sub_path}")
    print(f"     Threshold: {cfg.threshold}")
    print(f"     AI ratio: {final.mean():.4f}")
    print(f"     Distribution: {dict(zip(*np.unique(final, return_counts=True)))}")
    print(f"     Prob stats: mean={test_ens.mean():.4f}, std={test_ens.std():.4f}")
    return sub_path


# ============================================================================
# 8. MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SemEval-2026 Task 13 — 3-Track Ensemble")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--codebert_batch_size", type=int, default=None,
                        help="Samples per CodeBERT session (default: 100000)")
    parser.add_argument("--skip_track2", action="store_true")
    args, _ = parser.parse_known_args()

    if args.data_dir: cfg.data_dir = args.data_dir
    if args.output_dir: cfg.output_dir = args.output_dir
    if args.codebert_batch_size: cfg.codebert_batch_size = args.codebert_batch_size

    cache_dir = Path(cfg.output_dir)
    os.makedirs(cache_dir, exist_ok=True)
    set_seed(cfg.seed)

    print(f"\n{'='*60}")
    print(f"  SemEval-2026 Task 13 — 3-Track Ensemble Pipeline")
    print(f"  Data:     {cfg.data_dir}")
    print(f"  Output:   {cfg.output_dir}")
    print(f"  CodeBERT: {cfg.codebert_batch_size:,} samples/session, {cfg.epochs_per_session} epochs")
    print(f"  Backbone: {cfg.backbone}")
    print(f"{'='*60}")

    # -- Load --
    df_train, df_val, df_test = load_data(cfg)

    all_train_codes = df_train["code"].values
    all_train_labels = df_train["label"].values
    val_codes = df_val["code"].values if df_val is not None else None
    val_labels = df_val["label"].values if df_val is not None else None
    test_codes = df_test["code"].values if df_test is not None else None

    # -- Track 1: TF-IDF (FULL data) --
    t1_test, t1_val = run_track1(all_train_codes, all_train_labels,
                                  val_codes, val_labels, test_codes, cache_dir)

    # -- Track 3: Compression (FULL data) --
    t3_test, t3_val = run_track3(all_train_codes, all_train_labels,
                                  val_codes, val_labels, test_codes, cache_dir)

    # -- Track 2: CodeBERT (INCREMENTAL) --
    t2_test, t2_val = None, None
    if not args.skip_track2:
        t2_test, t2_val = run_track2(df_train, val_codes, val_labels,
                                      test_codes, cache_dir, cfg)
    else:
        # Try loading cached
        t2_test_path = cache_dir / "track2_test_probs.npy"
        t2_val_path = cache_dir / "track2_val_probs.npy"
        if t2_test_path.exists():
            t2_test = np.load(t2_test_path)
            t2_val = np.load(t2_val_path) if t2_val_path.exists() else None
            print("  ♻️  Loaded cached Track 2 predictions")
        else:
            print("  ⚠️  Track 2 skipped and no cache found")

    # -- Ensemble --
    if df_test is not None:
        ensemble_and_submit(
            [t1_test, t2_test, t3_test],
            [t1_val, t2_val, t3_val],
            [cfg.w_tfidf, cfg.w_codebert, cfg.w_compress],
            val_labels, df_test, cfg, cache_dir,
        )

    print(f"\n{'='*60}")
    print(f"  ✅ PIPELINE COMPLETE")
    print(f"  📁 All outputs in: {cache_dir}")
    print(f"  📋 Training log: {cache_dir / 'track2_train_log.csv'}")
    state = TrainingState.load(cache_dir / "track2_checkpoint" / "training_state.json")
    print(f"  📊 CodeBERT: {state.total_samples_seen:,} samples trained, "
          f"best_F1={state.best_val_f1:.4f}, session={state.session}")
    print(f"{'='*60}")
