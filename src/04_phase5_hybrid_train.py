"""
Phase 5: Hybrid Deep Feature Fusion + Contrastive Learning
===========================================================
SemEval-2026 Task 13 SubtaskA — AI-Generated Code Detection

Architecture:
    Semantic Branch (CodeBERT/DeBERTa) ──> CLS [768]
                                                  ├──> Concat [768+64] ──> Classifier
    Expert Branch (13 feats + XGB prob) ──> MLP [64]

Loss: L = alpha * SupCon + (1 - alpha) * CrossEntropy

Designed for: Google Colab T4 GPU (16 GB VRAM)

Usage on Colab:
    !pip install transformers accelerate xgboost tree-sitter \
         tree-sitter-python tree-sitter-java tree-sitter-cpp -q
    %run src/04_phase5_hybrid_train.py
"""

import os, sys, gc, math, time, pickle, re, zlib, warnings
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================


@dataclass
class Config:
    """All hyperparameters in one place. Edit here to experiment."""

    # -- Paths --
    data_dir: str = ""                         # Set below
    output_dir: str = ""                       # Set below
    backbone_name: str = "microsoft/codebert-base"  # or "microsoft/deberta-v3-base"

    # -- Data --
    max_length: int = 512                      # Token limit (covers ~95% samples)
    train_sample: int = 0                      # 0 = use all; set e.g. 100000 for quick test

    # -- Model --
    expert_input_dim: int = 14                 # 13 features + 1 XGB probability
    expert_hidden_dim: int = 64                # Expert branch MLP output
    projection_dim: int = 128                  # SupCon projection head dim
    dropout: float = 0.1

    # -- Training --
    n_folds: int = 5
    epochs: int = 1
    batch_size: int = 16                       # Per-GPU batch
    grad_accum_steps: int = 4                  # Effective batch = 16*4 = 64
    lr_backbone: float = 1e-5                  # Differential LR
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True                          # Mixed precision
    supcon_alpha: float = 0.3                  # Loss weight: alpha*SupCon + (1-a)*CE
    supcon_temperature: float = 0.07
    seed: int = 42

    # -- 13 Selected Features (from FEATURE_SELECTION.md) --
    selected_features: List[str] = field(default_factory=lambda: [
        "indent_consistency", "avg_line_length", "shannon_entropy",
        "comment_to_code_ratio", "snake_ratio", "trailing_ws_ratio",
        "avg_identifier_length", "camel_ratio", "avg_ast_depth",
        "token_entropy", "long_id_ratio", "branch_ratio",
        "zlib_compression_ratio",
    ])


# -- Auto-detect environment --
IN_KAGGLE = "kaggle_secrets" in sys.modules or os.path.exists("/kaggle/working")
IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

if IN_KAGGLE:
    base_k = Path("/kaggle/input/sem-eval-2026-task-13-subtask-a")
    if (base_k / "Task_A").exists():
        d_dir = str(base_k / "Task_A")
    else:
        d_dir = str(base_k)
    cfg = Config(data_dir=d_dir, output_dir="/kaggle/working/outputs_p5")
elif IN_COLAB:
    cfg = Config(data_dir="/content/data", output_dir="/content/outputs_p5")
else:
    _root = Path(__file__).resolve().parent.parent
    cfg = Config(
        data_dir=str(_root / "data" / "raw" / "Task_A"),
        output_dir=str(_root / "data" / "processed" / "phase5"),
    )

os.makedirs(cfg.output_dir, exist_ok=True)
print(f"Config: backbone={cfg.backbone_name}, batch={cfg.batch_size}, "
      f"accum={cfg.grad_accum_steps}, epochs={cfg.epochs}, folds={cfg.n_folds}")
print(f"Data:   {cfg.data_dir}")
print(f"Output: {cfg.output_dir}")


# ============================================================================
# 2. FEATURE EXTRACTION (reuse from Phase 4)
# ============================================================================

# -- Regex patterns --
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CAMEL = re.compile(r"\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b")
_PASCAL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
_IDENT = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_SINGLE_CMT = re.compile(r"(//.*|#(?!!).*)\s*$", re.MULTILINE)
_BLOCK_CMT = re.compile(r'/\*[\s\S]*?\*/|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
_KW = frozenset([
    "if","else","elif","for","while","return","def","class","import","from",
    "try","except","catch","finally","with","as","yield","break","continue",
    "pass","raise","lambda","async","await","public","private","protected",
    "static","void","int","float","double","string","bool","boolean","new",
    "throw","throws","switch","case","default","struct","enum","interface",
    "extends","implements","package","namespace","using","include","const",
    "var","let","auto","virtual","override","abstract","final",
])
_LANG = {
    "Python": re.compile(r"\bdef\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import"),
    "Java":   re.compile(r"\bpublic\s+(?:static\s+)?(?:void|class|int|String)|System\.out"),
    "C++":    re.compile(r"#include\s*<|cout\s*<<|std::|using\s+namespace"),
}
_BRANCH_KW = re.compile(r"\b(if|elif|else|for|while|try|except|catch|case|switch)\b")


def detect_language(code: str) -> str:
    scores = {l: len(p.findall(code)) for l, p in _LANG.items()}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "Python"


def extract_13_features(code: str) -> np.ndarray:
    """Extract 13 features as a numpy array (same order as cfg.selected_features)."""
    lines = code.split("\n")
    ne = [l for l in lines if l.strip()]
    n_ne = max(len(ne), 1)
    n_all = max(len(lines), 1)

    # indent_consistency
    il = [l for l in ne if len(l) > 0 and l[0] in (" ", "\t")]
    if il:
        tc = sum(1 for l in il if l[0] == "\t")
        indent_consistency = max(tc, len(il) - tc) / len(il)
    else:
        indent_consistency = 1.0

    # avg_line_length
    lens = [len(l) for l in ne]
    avg_ll = float(np.mean(lens)) if lens else 0.0

    # shannon_entropy
    if code:
        freq = Counter(code); n = len(code)
        shannon = -sum((c/n)*math.log2(c/n) for c in freq.values())
    else:
        shannon = 0.0

    # comment_to_code_ratio
    sc = len(_SINGLE_CMT.findall(code)); bc = len(_BLOCK_CMT.findall(code))
    ccr = (sc + bc) / n_ne

    # snake_ratio, camel_ratio
    sn = len(_SNAKE.findall(code)); ca = len(_CAMEL.findall(code))
    pa = len(_PASCAL.findall(code)); tn = max(sn + ca + pa, 1)
    snake_r = sn / tn; camel_r = ca / tn

    # trailing_ws_ratio
    tws = sum(1 for l in lines if l != l.rstrip()) / n_all

    # avg_identifier_length, long_id_ratio
    ids = _IDENT.findall(code)
    nkw = [i for i in ids if i.lower() not in _KW]
    if nkw:
        id_lens = [len(i) for i in nkw]
        avg_id = float(np.mean(id_lens))
        long_r = sum(1 for l in id_lens if l >= 10) / len(nkw)
    else:
        avg_id = 0.0; long_r = 0.0

    # avg_ast_depth (regex fallback — fast)
    depths = []
    for l in ne:
        s = l.lstrip()
        depths.append((len(l) - len(s)) / 4.0)
    avg_ast = float(np.mean(depths)) if depths else 0.0

    # token_entropy
    toks = re.findall(r"\b\w+\b", code)
    if toks:
        ft = Counter(toks); nt = len(toks)
        tok_ent = -sum((c/nt)*math.log2(c/nt) for c in ft.values())
    else:
        tok_ent = 0.0

    # branch_ratio
    branches = len(_BRANCH_KW.findall(code))
    branch_r = branches / n_ne

    # zlib_compression_ratio
    if code:
        enc = code.encode("utf-8", errors="replace")
        zr = len(zlib.compress(enc, 6)) / max(len(enc), 1)
    else:
        zr = 0.0

    return np.array([
        indent_consistency, avg_ll, shannon, ccr, snake_r, tws,
        avg_id, camel_r, avg_ast, tok_ent, long_r, branch_r, zr,
    ], dtype=np.float32)


def extract_features_batch(codes: list) -> np.ndarray:
    """Vectorized batch extraction. Returns (N, 13) array."""
    return np.stack([extract_13_features(c) for c in codes])


# ============================================================================
# 3. DATASET
# ============================================================================

class HybridCodeDataset(Dataset):
    """Dataset yielding (input_ids, attention_mask, expert_features, label)."""

    def __init__(self, codes, expert_feats, labels=None, tokenizer=None,
                 max_length=512):
        self.codes = codes
        self.expert_feats = expert_feats        # (N, 14) numpy — already z-scored
        self.labels = labels                     # None for test
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        enc = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "expert_feats": torch.tensor(self.expert_feats[idx], dtype=torch.float32),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================================
# 4. MODEL — Hybrid Feature Fusion
# ============================================================================

class ExpertBranch(nn.Module):
    """MLP to compress 14-dim expert vector → hidden_dim."""
    def __init__(self, input_dim=14, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class HybridFusionModel(nn.Module):
    """
    Two-branch architecture:
        Semantic: Backbone → CLS token [hidden_size]
        Expert:   14-dim → MLP → [expert_hidden]
        Fusion:   Concat → Projection Head → Classifier

    Returns:
        logits     (N, 2) for CrossEntropy
        embeddings (N, projection_dim) for SupCon
    """
    def __init__(self, backbone_name, expert_input_dim=14,
                 expert_hidden_dim=64, projection_dim=128, dropout=0.1):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for base

        self.expert_branch = ExpertBranch(expert_input_dim, expert_hidden_dim, dropout)

        fused_dim = hidden_size + expert_hidden_dim  # 768 + 64 = 832

        # Projection head (for SupCon)
        self.projector = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, projection_dim),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 2),
        )

    def forward(self, input_ids, attention_mask, expert_feats):
        # Semantic branch: CLS token (index 0)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]    # (B, hidden_size)

        # Expert branch
        expert_out = self.expert_branch(expert_feats)      # (B, expert_hidden)

        # Fusion
        fused = torch.cat([cls_token, expert_out], dim=-1) # (B, fused_dim)

        # Outputs
        logits = self.classifier(fused)                    # (B, 2)
        embeddings = F.normalize(self.projector(fused), dim=-1)  # (B, proj_dim)

        return logits, embeddings


# ============================================================================
# 5. LOSSES
# ============================================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Pulls same-label samples together, pushes different-label apart
    in embedding space. Key for OOD robustness.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, dim) L2-normalized embeddings
            labels: (B,) integer labels
        Returns:
            scalar loss
        """
        device = embeddings.device
        B = embeddings.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)

        # Cosine similarity matrix
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # (B, B)

        # Mask: same label = 1, different = 0 (exclude self)
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float().to(device)       # (B, B)
        self_mask = torch.eye(B, device=device)
        mask = mask - self_mask                               # Remove diagonal

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        # Log-softmax over all non-self entries
        exp_sim = torch.exp(sim) * (1 - self_mask)            # Zero out self
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs
        pos_count = mask.sum(dim=1).clamp(min=1)
        loss = -(mask * log_prob).sum(dim=1) / pos_count

        return loss.mean()


class HybridLoss(nn.Module):
    """L = alpha * SupCon + (1 - alpha) * CrossEntropy"""

    def __init__(self, alpha=0.3, temperature=0.07, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.supcon = SupConLoss(temperature)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, embeddings, labels):
        loss_ce = self.ce(logits, labels)
        loss_sc = self.supcon(embeddings, labels)
        return self.alpha * loss_sc + (1 - self.alpha) * loss_ce, loss_ce, loss_sc


# ============================================================================
# 6. TRAINING ENGINE
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model, cfg):
    """Differential learning rates: backbone slower, head faster."""
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay},
        {"params": head_params,     "lr": cfg.lr_head,     "weight_decay": cfg.weight_decay},
    ])


def get_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    from transformers import get_cosine_schedule_with_warmup
    warmup_steps = int(num_training_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, cfg, epoch):
    model.train()
    total_loss = 0; total_ce = 0; total_sc = 0
    steps = 0
    pbar = tqdm(loader, desc=f"  Train E{epoch}", leave=False, mininterval=10.0)

    optimizer.zero_grad()
    for i, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        expert    = batch["expert_feats"].to(device)
        labels    = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            logits, embeds = model(input_ids, attn_mask, expert)
            loss, lce, lsc = criterion(logits, embeds, labels)
            loss = loss / cfg.grad_accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * cfg.grad_accum_steps
        total_ce += lce.item()
        total_sc += lsc.item()
        steps += 1
        pbar.set_postfix(loss=f"{total_loss/steps:.4f}",
                         ce=f"{total_ce/steps:.4f}", sc=f"{total_sc/steps:.4f}")

    return total_loss / steps


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    all_probs = []; all_labels = []

    for batch in tqdm(loader, desc="  Eval", leave=False, mininterval=10.0):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        expert    = batch["expert_feats"].to(device)

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            logits, _ = model(input_ids, attn_mask, expert)

        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        if "labels" in batch:
            all_labels.append(batch["labels"].numpy())

    all_probs = np.concatenate(all_probs)
    if all_labels:
        all_labels = np.concatenate(all_labels)
        auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, (all_probs > 0.5).astype(int))
        return all_probs, auc, f1
    return all_probs, None, None


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":
    from transformers import AutoTokenizer

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 7.1 Load data ──
    print("\n[1/6] Loading data...")
    train_path = data_dir / "task_a_training_set_1.parquet" if (data_dir / "task_a_training_set_1.parquet").exists() else data_dir / "train.parquet"
    val_path   = data_dir / "task_a_validation_set.parquet" if (data_dir / "task_a_validation_set.parquet").exists() else data_dir / "validation.parquet"
    test_path  = data_dir / "test.parquet"

    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)
    df_test  = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame(columns=["ID", "code", "label", "language"])

    if cfg.train_sample > 0:
        df_train = df_train.sample(cfg.train_sample, random_state=cfg.seed).reset_index(drop=True)
        print(f"  Subsampled train to {len(df_train):,}")

    print(f"  Train={len(df_train):,}, Val={len(df_val):,}, Test={len(df_test):,}")

    # ── 7.2 Extract expert features ──
    print("\n[2/6] Extracting expert features (13-dim)...")

    cache_dir = Path(cfg.output_dir)

    def get_or_cache_features(df, name, code_col="code"):
        cache = cache_dir / f"expert_feats_{name}.npy"
        if cache.exists():
            print(f"  Loaded cached: {cache.name}")
            return np.load(cache)
        feats = np.stack([extract_13_features(c) for c in
                          tqdm(df[code_col], desc=f"  {name}", leave=False, mininterval=10.0)])
        np.save(cache, feats)
        print(f"  Cached: {cache.name} {feats.shape}")
        return feats

    feats_train = get_or_cache_features(df_train, "train")
    feats_val   = get_or_cache_features(df_val,   "val")
    feats_test  = get_or_cache_features(df_test,  "test")

    # ── 7.3 XGBoost probability (14th feature) ──
    print("\n[3/6] Generating XGBoost probabilities...")

    # Try to load Phase 4 models; if not available, train a quick one
    xgb_model_path = cache_dir / "xgb_for_p5.pkl"
    if xgb_model_path.exists():
        with open(xgb_model_path, "rb") as f:
            xgb_data = pickle.load(f)
        print("  Loaded cached XGB model")
    else:
        import xgboost as xgb_lib
        print("  Training quick XGBoost (for probability feature)...")
        xgb_model = xgb_lib.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", device="cuda" if device.type == "cuda" else "cpu",
            random_state=cfg.seed, verbosity=0,
        )
        xgb_model.fit(feats_train, df_train["label"].values,
                      eval_set=[(feats_val, df_val["label"].values)],
                      verbose=False)
        xgb_data = {"model": xgb_model}
        with open(xgb_model_path, "wb") as f:
            pickle.dump(xgb_data, f)

    xgb_m = xgb_data if isinstance(xgb_data, dict) and "model" in xgb_data else {"model": xgb_data}
    xgb_model = xgb_m.get("model", xgb_m)
    if isinstance(xgb_model, dict):
        xgb_model = xgb_model.get("models", [xgb_model])[0]

    xgb_prob_train = xgb_model.predict_proba(feats_train)[:, 1].reshape(-1, 1)
    xgb_prob_val   = xgb_model.predict_proba(feats_val)[:, 1].reshape(-1, 1)
    xgb_prob_test  = xgb_model.predict_proba(feats_test)[:, 1].reshape(-1, 1) if len(feats_test) > 0 else np.empty((0, 1))

    # Concat: 13 features + 1 XGB prob = 14-dim
    expert_train = np.hstack([feats_train, xgb_prob_train])  # (N, 14)
    expert_val   = np.hstack([feats_val,   xgb_prob_val])
    expert_test  = np.hstack([feats_test,  xgb_prob_test]) if len(feats_test) > 0 else np.empty((0, 14))

    # Z-score normalization (mu=0, sigma=1)
    scaler = StandardScaler()
    expert_train = scaler.fit_transform(expert_train).astype(np.float32)
    expert_val   = scaler.transform(expert_val).astype(np.float32)
    expert_test  = scaler.transform(expert_test).astype(np.float32) if len(expert_test) > 0 else expert_test

    with open(cache_dir / "zscore_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Expert features: train={expert_train.shape}, val={expert_val.shape}")

    # ── 7.4 Tokenizer ──
    print(f"\n[4/6] Loading tokenizer: {cfg.backbone_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_name)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── 7.5 K-Fold Training ──
    print(f"\n[5/6] Training {cfg.n_folds}-Fold CV...")

    codes_train = df_train["code"].values
    labels_train = df_train["label"].values
    codes_val = df_val["code"].values
    labels_val = df_val["label"].values

    skf = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_aucs = []; fold_f1s = []
    oof_probs = np.zeros(len(codes_train))
    fold_models_dir = cache_dir / "fold_models"
    fold_models_dir.mkdir(exist_ok=True)

    criterion = HybridLoss(alpha=cfg.supcon_alpha,
                           temperature=cfg.supcon_temperature)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(codes_train, labels_train), 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{cfg.n_folds}")
        print(f"{'='*60}")
        print(f"  Train: {len(tr_idx):,}, Val: {len(va_idx):,}")

        # Checkpoint resume
        fold_model_path = fold_models_dir / f"fold{fold}_best.pt"
        if fold_model_path.exists():
            print(f"  [Resume] Tìm thấy model của fold {fold}. Bỏ qua train và tiến hành load...")
            model = HybridFusionModel(
                cfg.backbone_name, cfg.expert_input_dim,
                cfg.expert_hidden_dim, cfg.projection_dim, cfg.dropout,
            ).to(device)
            model.load_state_dict(torch.load(fold_model_path, weights_only=True))

            ds_va = HybridCodeDataset(
                codes_train[va_idx], expert_train[va_idx], labels_train[va_idx],
                tokenizer, cfg.max_length)
            dl_va = DataLoader(ds_va, batch_size=cfg.batch_size * 2, shuffle=False,
                               num_workers=2, pin_memory=True)
                               
            val_probs, val_auc, val_f1 = evaluate(model, dl_va, device, cfg)
            oof_probs[va_idx] = val_probs
            fold_aucs.append(val_auc)
            fold_f1s.append(val_f1)
            print(f"  Fold {fold} resumed: AUC={val_auc:.4f}, F1={val_f1:.4f}")
            del model, ds_va, dl_va
            gc.collect(); torch.cuda.empty_cache()
            continue

        # Datasets
        ds_tr = HybridCodeDataset(
            codes_train[tr_idx], expert_train[tr_idx], labels_train[tr_idx],
            tokenizer, cfg.max_length)
        ds_va = HybridCodeDataset(
            codes_train[va_idx], expert_train[va_idx], labels_train[va_idx],
            tokenizer, cfg.max_length)

        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)
        dl_va = DataLoader(ds_va, batch_size=cfg.batch_size * 2, shuffle=False,
                           num_workers=2, pin_memory=True)

        # Model
        model = HybridFusionModel(
            cfg.backbone_name, cfg.expert_input_dim,
            cfg.expert_hidden_dim, cfg.projection_dim, cfg.dropout,
        ).to(device)

        # Enable gradient checkpointing (saves VRAM)
        if hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable()

        optimizer = get_optimizer(model, cfg)
        total_steps = (len(dl_tr) // cfg.grad_accum_steps) * cfg.epochs
        scheduler = get_scheduler(optimizer, total_steps, cfg.warmup_ratio)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

        best_auc = 0; patience = 0
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, dl_tr, criterion, optimizer, scheduler, scaler,
                device, cfg, epoch)

            val_probs, val_auc, val_f1 = evaluate(model, dl_va, device, cfg)
            elapsed = time.time() - t0

            print(f"  Epoch {epoch}: loss={train_loss:.4f}, "
                  f"val_AUC={val_auc:.4f}, val_F1={val_f1:.4f}, "
                  f"time={elapsed:.0f}s")

            if val_auc > best_auc:
                best_auc = val_auc
                patience = 0
                torch.save(model.state_dict(),
                           fold_models_dir / f"fold{fold}_best.pt")
            else:
                patience += 1
                if patience >= 2:
                    print(f"  Early stop at epoch {epoch}")
                    break

        # Reload best and get OOF predictions
        model.load_state_dict(torch.load(fold_models_dir / f"fold{fold}_best.pt",
                                         weights_only=True))
        val_probs, val_auc, val_f1 = evaluate(model, dl_va, device, cfg)
        oof_probs[va_idx] = val_probs
        fold_aucs.append(val_auc)
        fold_f1s.append(val_f1)
        print(f"  Fold {fold} best: AUC={val_auc:.4f}, F1={val_f1:.4f}")

        # Cleanup
        del model, optimizer, scheduler, scaler, ds_tr, ds_va, dl_tr, dl_va
        gc.collect(); torch.cuda.empty_cache()

    # -- CV Summary --
    print(f"\n{'='*60}")
    print(f"  {cfg.n_folds}-Fold CV Summary:")
    print(f"    AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"    F1:  {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    print(f"{'='*60}")

    oof_labels = (oof_probs > 0.5).astype(int)
    print("\n  OOF Classification Report:")
    print(classification_report(labels_train, oof_labels,
                                target_names=["Human", "AI"]))

    # ── 7.6 Inference on test set ──
    print("\n[6/6] Test inference (ensemble of best folds)...")

    if len(df_test) > 0:
        test_ds = HybridCodeDataset(
            df_test["code"].values, expert_test, None, tokenizer, cfg.max_length)
        test_dl = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                             num_workers=2, pin_memory=True)

        test_probs = np.zeros(len(df_test))
        for fold in range(1, cfg.n_folds + 1):
            model = HybridFusionModel(
                cfg.backbone_name, cfg.expert_input_dim,
                cfg.expert_hidden_dim, cfg.projection_dim, cfg.dropout,
            ).to(device)
            model.load_state_dict(torch.load(
                fold_models_dir / f"fold{fold}_best.pt", weights_only=True))

            probs, _, _ = evaluate(model, test_dl, device, cfg)
            test_probs += probs / cfg.n_folds
            del model; gc.collect(); torch.cuda.empty_cache()

        # Submission
        submission = pd.DataFrame({
            "ID": df_test["ID"].values,
            "label": (test_probs > 0.5).astype(int),
        })
        sub_path = cache_dir / "submission_p5.csv"
        submission.to_csv(sub_path, index=False)

        # Probabilities
        pd.DataFrame({
            "ID": df_test["ID"].values,
            "prob_ai": test_probs,
            "label": (test_probs > 0.5).astype(int),
        }).to_parquet(cache_dir / "test_preds_p5.parquet", index=False)

        print(f"\n  Submission: {sub_path}")
        print(f"  Predicted AI ratio: {(test_probs > 0.5).mean():.4f}")
        print(f"  Label dist: {submission['label'].value_counts().to_dict()}")
    else:
        print("  Bỏ qua Inference trên test.parquet do không tìm thấy file.")

    # -- Validate on val set --
    print("\n  Validation set (ensemble)...")
    val_ds = HybridCodeDataset(
        codes_val, expert_val, labels_val, tokenizer, cfg.max_length)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

    val_probs_ens = np.zeros(len(codes_val))
    for fold in range(1, cfg.n_folds + 1):
        model = HybridFusionModel(
            cfg.backbone_name, cfg.expert_input_dim,
            cfg.expert_hidden_dim, cfg.projection_dim, cfg.dropout,
        ).to(device)
        model.load_state_dict(torch.load(
            fold_models_dir / f"fold{fold}_best.pt", weights_only=True))
        probs, _, _ = evaluate(model, val_dl, device, cfg)
        val_probs_ens += probs / cfg.n_folds
        del model; gc.collect(); torch.cuda.empty_cache()

    val_auc_ens = roc_auc_score(labels_val, val_probs_ens)
    val_f1_ens = f1_score(labels_val, (val_probs_ens > 0.5).astype(int))
    print(f"  Val AUC (ensemble): {val_auc_ens:.4f}")
    print(f"  Val F1  (ensemble): {val_f1_ens:.4f}")

    print(f"\n{'='*60}")
    print(f"  PHASE 5 COMPLETE")
    print(f"  CV AUC:  {np.mean(fold_aucs):.4f}")
    print(f"  Val AUC: {val_auc_ens:.4f}")
    print(f"  Output:  {sub_path}")
    print(f"{'='*60}")

    # ── 7.7 TẠO FILE QUÉT NỘP THỬ THEO FORMAT test_sample.parquet ──
    print("\n[7/7] Tạo output dựa trên format của test_sample.parquet...")
    sample_path = data_dir / "task_a_test_set_sample.parquet" if (data_dir / "task_a_test_set_sample.parquet").exists() else data_dir / "test_sample.parquet"
    if sample_path.exists():
        df_sample = pd.read_parquet(sample_path)
        feats_sample = np.stack([extract_13_features(c) for c in df_sample["code"]])
        xgb_prob_sample = xgb_model.predict_proba(feats_sample)[:, 1].reshape(-1, 1)
        expert_sample = np.hstack([feats_sample, xgb_prob_sample])
        expert_sample = scaler.transform(expert_sample).astype(np.float32)

        sample_ds = HybridCodeDataset(
            df_sample["code"].values, expert_sample, None, tokenizer, cfg.max_length)
        sample_dl = DataLoader(sample_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                               num_workers=2, pin_memory=True)

        sample_probs = np.zeros(len(df_sample))
        for fold in range(1, cfg.n_folds + 1):
            model = HybridFusionModel(
                cfg.backbone_name, cfg.expert_input_dim,
                cfg.expert_hidden_dim, cfg.projection_dim, cfg.dropout,
            ).to(device)
            model.load_state_dict(torch.load(
                fold_models_dir / f"fold{fold}_best.pt", weights_only=True))
            probs, _, _ = evaluate(model, sample_dl, device, cfg)
            sample_probs += probs / cfg.n_folds
            del model; gc.collect(); torch.cuda.empty_cache()

        df_sample["label"] = (sample_probs > 0.5).astype(int)
        df_sample = df_sample[['code', 'generator', 'label', 'language']]
        
        out_sample_path = cache_dir / "test_sample_predictions.parquet"
        df_sample.to_parquet(out_sample_path, index=False)
        print(f"  Đã lưu output quét thử thành công tại: {out_sample_path}")
    else:
        print(f"  Không tìm thấy file {sample_path}.")
