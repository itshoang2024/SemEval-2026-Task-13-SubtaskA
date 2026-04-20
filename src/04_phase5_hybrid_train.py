"""
Phase 5: Hybrid Deep Feature Fusion + Contrastive Learning (v2)
================================================================
SemEval-2026 Task 13 SubtaskA — AI-Generated Code Detection

Architecture:
    Semantic Branch (CodeBERT/DeBERTa) --> CLS [768]
                                                  ├--> Concat [768+64] --> Classifier
    Expert Branch (13 feats + XGB prob) --> MLP [64]

Loss: L = alpha * SupCon + (1 - alpha) * CrossEntropy

Chiến lược v2 (Kaggle 12h T4):
    - Bỏ K-Fold → train 1 lần trên train set, validate trên validation set
    - Balanced Ensemble: train N models trên các balanced subsets khác nhau
    - Progressive Unfreezing: freeze backbone epoch 1-2, unfreeze epoch 3+
    - Early Stopping theo val_AUC (patience=2)
    - Subsample 50K mỗi subset × 3 models = tổng 150K samples

Usage on Kaggle:
    !pip install transformers accelerate xgboost tree-sitter \
         tree-sitter-python tree-sitter-java tree-sitter-cpp -q
    !python src/04_phase5_hybrid_train.py

    # Inference only (load weights đã train):
    !python src/04_phase5_hybrid_train.py --inference_only --model_dir /kaggle/working
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
    backbone_name: str = "microsoft/unixcoder-base"  # UniXcoder: unified cross-modal code repr

    # -- Data --
    max_length: int = 384                      # ↓ Giảm từ 512 → 384 tiết kiệm thời gian (vẫn cover ~90%)
    train_sample: int = 100_000                # 100K balanced (50K/class) — vừa đủ 12h Kaggle

    # -- Ensemble --
    n_ensemble: int = 2                        # ↓ Giảm từ 3 → 2 models (tiết kiệm ~33% thời gian)
    # Mỗi model train trên 1 balanced subset khác nhau (seed khác nhau)

    # -- Model --
    expert_input_dim: int = 14                 # 13 features + 1 XGB probability
    expert_hidden_dim: int = 64                # Expert branch MLP output
    projection_dim: int = 128                  # SupCon projection head dim
    dropout: float = 0.3                       # ↑ Tăng mạnh từ 0.1 → 0.3 chống overfit
    expert_feat_mask_prob: float = 0.5         # [NEW] Xác suất mask expert features khi train
    expert_gate_lambda: float = 0.01           # [NEW] L2 penalty trên expert gate
    code_augment: bool = True                  # [NEW] Augment code formatting khi train

    # -- Training --
    epochs: int = 3                            # ↓ Giảm lại 3 cho vừa 12h Kaggle
    freeze_backbone_epochs: int = 1            # ↓ Giảm xuống 1 để CodeBERT học sớm hơn
    batch_size: int = 32                       # Per-GPU batch
    grad_accum_steps: int = 2                  # Effective batch = 32*2 = 64
    lr_backbone: float = 2e-5                  # ↑ Tăng nhẹ backbone LR
    lr_head: float = 5e-4                      # ↓ Giảm head LR
    weight_decay: float = 0.05                 # ↑ Tăng mạnh từ 0.01 → 0.05
    warmup_ratio: float = 0.1
    fp16: bool = True                          # Mixed precision
    label_smoothing: float = 0.1               # ↑ Tăng từ 0.05 → 0.1
    supcon_alpha: float = 0.1                  # ↓ Giảm mạnh: ít SupCon, nhiều CE hơn
    supcon_temperature: float = 0.07
    early_stop_patience: int = 2               # Patience cho early stopping
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
    import glob
    search_paths = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
    if not search_paths:
        search_paths = glob.glob("/kaggle/input/**/*.parquet", recursive=True)
    if search_paths:
        d_dir = str(Path(search_paths[0]).parent)
    else:
        d_dir = "/kaggle/input/sem-eval-2026-task-13-subtask-a/Task_A"
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
      f"accum={cfg.grad_accum_steps}, epochs={cfg.epochs}, "
      f"ensemble={cfg.n_ensemble}, sample={cfg.train_sample}")
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

# -- Tree-sitter parsers (real AST) --
try:
    from tree_sitter import Language, Parser as TSParser
    import tree_sitter_python as _tspy
    import tree_sitter_java as _tsja
    import tree_sitter_cpp as _tscpp
    _TS_LANGS = {
        "Python": Language(_tspy.language()),
        "Java":   Language(_tsja.language()),
        "C++":    Language(_tscpp.language()),
    }
    _TS_AVAILABLE = True
    print("[AST] Tree-sitter loaded (Python, Java, C++)")
except (ImportError, Exception):
    _TS_AVAILABLE = False
    print("[AST] Tree-sitter NOT available — using indentation fallback")

_AST_BRANCH_TYPES = frozenset({
    "if_statement", "elif_clause", "else_clause",
    "for_statement", "while_statement", "for_in_clause",
    "try_statement", "except_clause", "catch_clause",
    "switch_statement", "switch_expression", "case_statement",
    "conditional_expression", "ternary_expression", "do_statement",
})


def _compute_ast_metrics(code: str, lang: str) -> Tuple[float, float]:
    """Compute avg_ast_depth and branch_ratio using real tree-sitter AST.
    Returns (avg_ast_depth, branch_ratio) or (None, None) on failure.
    """
    if not _TS_AVAILABLE:
        return None, None
    try:
        ts_lang = _TS_LANGS.get(lang, _TS_LANGS["Python"])
        parser = TSParser(ts_lang)
        tree = parser.parse(code.encode("utf-8", errors="replace"))
        root = tree.root_node
        if root.child_count == 0:
            return 0.0, 0.0
        depths = []
        branch_count = 0
        total_nodes = 0
        stack = [(root, 0)]
        while stack:
            node, depth = stack.pop()
            total_nodes += 1
            depths.append(depth)
            if node.type in _AST_BRANCH_TYPES:
                branch_count += 1
            for child in node.children:
                stack.append((child, depth + 1))
        avg_d = float(np.mean(depths)) if depths else 0.0
        branch_r = branch_count / max(total_nodes, 1)
        return avg_d, branch_r
    except Exception:
        return None, None


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

    # avg_ast_depth & branch_ratio — real tree-sitter AST with fallback
    lang = detect_language(code)
    ast_depth, ast_branch_r = _compute_ast_metrics(code, lang)
    if ast_depth is not None:
        avg_ast = ast_depth
    else:
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

    if ast_branch_r is not None:
        branch_r = ast_branch_r
    else:
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
# 3. BALANCED SAMPLING
# ============================================================================

def create_balanced_subset(df, sample_size, seed):
    """Tạo balanced subset với 50/50 human/AI.
    
    Args:
        df: DataFrame có cột 'label' (0=human, 1=AI), index phải là 0..N-1
        sample_size: tổng số samples mong muốn
        seed: random seed (mỗi ensemble member dùng seed khác nhau)
    
    Returns:
        (DataFrame balanced subset với reset index, 
         np.array original_indices để map expert features)
    """
    per_class = sample_size // 2
    
    df_human = df[df["label"] == 0]
    df_ai = df[df["label"] == 1]
    
    n_human = min(per_class, len(df_human))
    n_ai = min(per_class, len(df_ai))
    
    # Nếu 1 class ít hơn, giảm class kia cho cân bằng
    n_each = min(n_human, n_ai)
    
    subset_human = df_human.sample(n_each, random_state=seed)
    subset_ai = df_ai.sample(n_each, random_state=seed)
    
    balanced = pd.concat([subset_human, subset_ai]).sample(
        frac=1.0, random_state=seed  # Shuffle
    )
    
    # Lưu original indices TRƯỚC khi reset
    original_indices = balanced.index.values.copy()
    balanced = balanced.reset_index(drop=True)
    
    print(f"  Balanced subset (seed={seed}): {len(balanced):,} samples "
          f"({n_each:,} human + {n_each:,} AI)")
    return balanced, original_indices


# ============================================================================
# 4. CODE AUGMENTATION (chống overfit vào formatting shortcuts)
# ============================================================================

import random as _random

def normalize_code(code: str, augment: bool = True) -> str:
    """Normalize code formatting to prevent model from learning superficial
    formatting shortcuts (indent style, trailing whitespace, comment presence).
    
    During TRAINING (augment=True): randomly apply transformations.
    During INFERENCE (augment=False): apply deterministic normalization only.
    """
    lines = code.split("\n")
    result = []
    
    for line in lines:
        # Always: strip trailing whitespace (phá shortcut trailing_ws_ratio)
        line = line.rstrip()
        
        if augment:
            # Random: skip empty lines (50% chance) → phá shortcut avg_line_length
            if not line.strip() and _random.random() < 0.5:
                continue
            
            # Random: remove single-line comments (30% chance) 
            # → phá shortcut comment_to_code_ratio
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                if _random.random() < 0.3:
                    continue
            
            # Random: normalize tabs↔spaces (20% chance)
            # → phá shortcut indent_consistency
            if _random.random() < 0.2:
                if "\t" in line:
                    line = line.replace("\t", "    ")
                else:
                    # Convert leading groups of 4 spaces to tab
                    leading = len(line) - len(line.lstrip(" "))
                    if leading >= 4:
                        tabs = leading // 4
                        line = "\t" * tabs + line[tabs * 4:]
        
        result.append(line)
    
    return "\n".join(result)


# ============================================================================
# 4b. DATASET
# ============================================================================

class HybridCodeDataset(Dataset):
    """Dataset yielding (input_ids, attention_mask, expert_features, label)."""

    def __init__(self, codes, expert_feats, labels=None, tokenizer=None,
                 max_length=512, augment=False):
        self.codes = codes
        self.expert_feats = expert_feats        # (N, 14) numpy — already z-scored
        self.labels = labels                     # None for test
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment                   # [NEW] Code augment khi train

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        
        # [NEW] Code augmentation: normalize formatting during training
        if self.augment:
            code = normalize_code(code, augment=True)
        
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
# 5. MODEL — Hybrid Feature Fusion
# ============================================================================

class ExpertBranch(nn.Module):
    """MLP to compress 14-dim expert vector → hidden_dim.
    
    [ANTI-OVERFIT] Feature Masking: randomly zero-out entire feature columns
    during training to prevent reliance on any single heuristic shortcut.
    """
    def __init__(self, input_dim=14, hidden_dim=64, dropout=0.3,
                 feat_mask_prob=0.5):
        super().__init__()
        self.feat_mask_prob = feat_mask_prob  # Probability of masking each feature
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
        # [ANTI-OVERFIT] Feature-level dropout during training
        # Randomly zero-out entire columns so model can't rely on any one feature
        if self.training and self.feat_mask_prob > 0:
            mask = torch.bernoulli(
                torch.full(x.shape, 1.0 - self.feat_mask_prob, device=x.device)
            )
            x = x * mask  # Zero out random features
        return self.net(x)


class HybridFusionModel(nn.Module):
    """
    Two-branch architecture with GATED FUSION:
        Semantic: Backbone → CLS token [hidden_size]
        Expert:   14-dim → MLP (with feature masking) → [expert_hidden]
        Fusion:   Gated: cls + gate * expert → Classifier
    
    [ANTI-OVERFIT] Learnable gate controls expert branch influence.
    L2 penalty on gate encourages model to rely on CodeBERT semantics.

    Returns:
        logits     (N, 2) for CrossEntropy
        embeddings (N, projection_dim) for SupCon
        gate_val   scalar for L2 penalty in loss
    """
    def __init__(self, backbone_name, expert_input_dim=14,
                 expert_hidden_dim=64, projection_dim=128, dropout=0.3,
                 feat_mask_prob=0.5):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for base

        self.expert_branch = ExpertBranch(
            expert_input_dim, expert_hidden_dim, dropout, feat_mask_prob
        )

        fused_dim = hidden_size + expert_hidden_dim  # 768 + 64 = 832

        # [ANTI-OVERFIT] Learnable gate: controls how much expert influences output
        # Initialized to 0.3 so expert starts with low influence
        self.expert_gate = nn.Parameter(torch.tensor(0.3))

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

    def freeze_backbone(self):
        """Freeze tất cả params của backbone (CodeBERT)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  [Freeze] Backbone frozen — chỉ train head")

    def unfreeze_backbone(self):
        """Unfreeze backbone để fine-tune."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("  [Unfreeze] Backbone unfrozen — fine-tune toàn bộ")

    def forward(self, input_ids, attention_mask, expert_feats):
        # Semantic branch: CLS token (index 0)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]    # (B, hidden_size)

        # Expert branch
        expert_out = self.expert_branch(expert_feats)      # (B, expert_hidden)

        # [ANTI-OVERFIT] Gated Fusion: sigmoid gate limits expert influence
        gate = torch.sigmoid(self.expert_gate)             # scalar in (0, 1)
        gated_expert = gate * expert_out                   # Scale expert contribution
        fused = torch.cat([cls_token, gated_expert], dim=-1)  # (B, fused_dim)

        # Outputs
        logits = self.classifier(fused)                    # (B, 2)
        
        # Calculate projection and normalize in float32 to prevent FP16 NaN underflows
        proj = self.projector(fused).float()
        embeddings = F.normalize(proj, p=2, dim=-1, eps=1e-8)  # (B, proj_dim)

        return logits, embeddings, gate


# ============================================================================
# 6. LOSSES
# ============================================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        B = embeddings.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)

        # Assure embeddings are fp32 for stable dot products
        embeddings = embeddings.float()
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float().to(device)
        self_mask = torch.eye(B, device=device)
        mask = mask - self_mask

        logits_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - logits_max.detach()

        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        pos_count = mask.sum(dim=1).clamp(min=1)
        loss = -(mask * log_prob).sum(dim=1) / pos_count

        return loss.mean()


class HybridLoss(nn.Module):
    """L = alpha * SupCon + (1 - alpha) * CrossEntropy + lambda * gate_penalty
    
    [ANTI-OVERFIT] Added gate_lambda penalty to discourage heavy expert reliance.
    """

    def __init__(self, alpha=0.1, temperature=0.07, label_smoothing=0.1,
                 gate_lambda=0.01):
        super().__init__()
        self.alpha = alpha
        self.gate_lambda = gate_lambda
        self.supcon = SupConLoss(temperature)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, embeddings, labels, gate_val=None):
        loss_ce = self.ce(logits, labels)
        loss_sc = self.supcon(embeddings, labels)
        total = self.alpha * loss_sc + (1 - self.alpha) * loss_ce
        
        # [ANTI-OVERFIT] L2 penalty on expert gate — push gate toward 0
        # This encourages model to rely on CodeBERT semantics over expert heuristics
        if gate_val is not None:
            gate_penalty = self.gate_lambda * (gate_val ** 2)
            total = total + gate_penalty
        
        return total, loss_ce, loss_sc


# ============================================================================
# 7. TRAINING ENGINE
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

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.lr_backbone,
                       "weight_decay": cfg.weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": cfg.lr_head,
                       "weight_decay": cfg.weight_decay})

    return torch.optim.AdamW(groups)


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
            logits, embeds, gate_val = model(input_ids, attn_mask, expert)
            loss, lce, lsc = criterion(logits, embeds, labels, gate_val)
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
        if steps % 50 == 0:
            pbar.set_postfix(loss=f"{total_loss/steps:.4f}",
                             ce=f"{total_ce/steps:.4f}", sc=f"{total_sc/steps:.4f}")

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    all_probs = []; all_labels = []

    for batch in tqdm(loader, desc="  Eval", leave=False, mininterval=10.0):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        expert    = batch["expert_feats"].to(device)

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            logits, _, _ = model(input_ids, attn_mask, expert)

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


def train_single_model(model_idx, df_train_subset, expert_train_subset,
                       codes_val, expert_val, labels_val,
                       tokenizer, device, cfg, save_dir):
    """Train 1 model trên 1 balanced subset, trả về best val_AUC."""

    print(f"\n{'='*60}")
    print(f"  ENSEMBLE MODEL {model_idx}/{cfg.n_ensemble}")
    print(f"  Train: {len(df_train_subset):,}, Val: {len(codes_val):,}")
    print(f"{'='*60}")

    model_path = save_dir / f"model{model_idx}_best.pt"

    # Kiểm tra checkpoint
    if model_path.exists():
        print(f"  [Resume] Model {model_idx} đã train. Bỏ qua...")
        model = HybridFusionModel(
            cfg.backbone_name, cfg.expert_input_dim,
            cfg.expert_hidden_dim, cfg.projection_dim, cfg.dropout,
            cfg.expert_feat_mask_prob,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        
        val_ds = HybridCodeDataset(codes_val, expert_val, labels_val, tokenizer, cfg.max_length)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                            num_workers=2, pin_memory=True)
        _, val_auc, val_f1 = evaluate(model, val_dl, device, cfg)
        print(f"  Resumed model {model_idx}: val_AUC={val_auc:.4f}, val_F1={val_f1:.4f}")
        del model, val_ds, val_dl
        gc.collect(); torch.cuda.empty_cache()
        return val_auc

    # Datasets
    codes_tr = df_train_subset["code"].values
    labels_tr = df_train_subset["label"].values

    ds_tr = HybridCodeDataset(codes_tr, expert_train_subset, labels_tr,
                              tokenizer, cfg.max_length, augment=cfg.code_augment)
    ds_va = HybridCodeDataset(codes_val, expert_val, labels_val,
                              tokenizer, cfg.max_length, augment=False)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size * 2, shuffle=False,
                       num_workers=2, pin_memory=True)

    # Model
    model = HybridFusionModel(
        cfg.backbone_name, cfg.expert_input_dim,
        cfg.expert_hidden_dim, cfg.projection_dim, cfg.dropout,
        cfg.expert_feat_mask_prob,
    ).to(device)

    # Gradient checkpointing (saves VRAM)
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    # Criterion (with gate penalty and increased label smoothing)
    criterion = HybridLoss(alpha=cfg.supcon_alpha,
                           temperature=cfg.supcon_temperature,
                           label_smoothing=cfg.label_smoothing,
                           gate_lambda=cfg.expert_gate_lambda)

    # Initialize Optimizer and Scaler ONCE to preserve momentum and state
    model.freeze_backbone()  # Start with frozen backbone
    optimizer = get_optimizer(model, cfg)
    total_steps = (len(dl_tr) // cfg.grad_accum_steps) * cfg.epochs
    scheduler = get_scheduler(optimizer, total_steps, cfg.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    best_auc = 0; patience = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # === Progressive Unfreezing ===
        if epoch <= cfg.freeze_backbone_epochs:
            backbone_frozen = True
        else:
            backbone_frozen = False
            if epoch == cfg.freeze_backbone_epochs + 1:
                model.unfreeze_backbone()

        # Train
        train_loss = train_one_epoch(
            model, dl_tr, criterion, optimizer, scheduler, scaler,
            device, cfg, epoch)

        # Validate
        val_probs, val_auc, val_f1 = evaluate(model, dl_va, device, cfg)
        elapsed = time.time() - t0

        frozen_str = " [FROZEN]" if backbone_frozen else " [FULL]"
        print(f"  Epoch {epoch}{frozen_str}: loss={train_loss:.4f}, "
              f"val_AUC={val_auc:.4f}, val_F1={val_f1:.4f}, "
              f"time={elapsed:.0f}s")

        # Check AI prediction ratio + gate value
        pred_labels = (val_probs > 0.5).astype(int)
        ai_ratio = pred_labels.mean()
        gate_v = torch.sigmoid(model.expert_gate).item()
        print(f"    → Predicted AI ratio on val: {ai_ratio:.4f} "
              f"(actual: {labels_val.mean():.4f}), gate={gate_v:.4f}")

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
            torch.save(model.state_dict(), model_path)
            print(f"    → Saved best model (AUC={best_auc:.4f})")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"  Early stop at epoch {epoch} (patience={patience})")
                break

    # Cleanup
    del model, optimizer, scheduler, scaler, ds_tr, ds_va, dl_tr, dl_va
    gc.collect(); torch.cuda.empty_cache()

    print(f"  Model {model_idx} done: best_AUC={best_auc:.4f}")
    return best_auc


# ============================================================================
# 8. INFERENCE ENGINE
# ============================================================================

class HybridInferencer:
    """Load trained ensemble weights và chạy inference."""
    def __init__(self, model_dir, backbone_name, device, max_length=512, n_ensemble=3):
        from transformers import AutoTokenizer
        self.model_dir = Path(model_dir)
        self.device = device
        self.max_length = max_length
        self.n_ensemble = n_ensemble

        # 1. Load XGBoost model
        xgb_path = self.model_dir / "xgb_for_p5_v2.pkl"
        if not xgb_path.exists():
            xgb_path = self.model_dir / "xgb_for_p5.pkl"  # fallback v1
        if not xgb_path.exists():
            raise FileNotFoundError(f"Không tìm thấy XGBoost model tại {self.model_dir}")
        with open(xgb_path, "rb") as f:
            xgb_m = pickle.load(f)
            self.xgb_model = xgb_m.get("model", xgb_m) if isinstance(xgb_m, dict) else xgb_m
            if isinstance(self.xgb_model, dict):
                self.xgb_model = self.xgb_model.get("models", [self.xgb_model])[0]

        # 2. Load Scaler
        scaler_path = self.model_dir / "zscore_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Không tìm thấy scaler tại {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # 3. Load Tokenizer
        print(f"Loading tokenizer: {backbone_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        # 4. Load PyTorch models (ensemble)
        models_dir = self.model_dir / "ensemble_models"
        # Fallback: check old fold_models too
        if not models_dir.exists():
            models_dir = self.model_dir / "fold_models"
        
        self.models = []
        for i in range(1, n_ensemble + 1):
            model = HybridFusionModel(
                backbone_name, expert_input_dim=14, expert_hidden_dim=64,
                projection_dim=128, dropout=0.3, feat_mask_prob=0.0
            ).to(self.device)
            
            # Try ensemble path first, then fold path
            model_pt = self.model_dir / "ensemble_models" / f"model{i}_best.pt"
            if not model_pt.exists():
                model_pt = self.model_dir / "fold_models" / f"fold{i}_best.pt"
            if not model_pt.exists():
                print(f"  [WARN] Không tìm thấy model {i}, bỏ qua...")
                del model
                continue
                
            print(f"  Loading {model_pt.name}...")
            model.load_state_dict(torch.load(model_pt, map_location=self.device, weights_only=True))
            model.eval()
            self.models.append(model)
        
        if not self.models:
            raise FileNotFoundError("Không tìm thấy bất kỳ model nào!")
        print(f"Inference engine ready ({len(self.models)} models).")

        # 5. Load optimal threshold
        thresh_path = self.model_dir / "optimal_threshold.txt"
        if thresh_path.exists():
            self.threshold = float(thresh_path.read_text().strip())
            print(f"  Optimal threshold loaded: {self.threshold:.4f}")
        else:
            self.threshold = 0.5
            print(f"  Using default threshold: {self.threshold}")

    @torch.no_grad()
    def predict(self, codes: list, batch_size=32):
        print(f"Extracting features for {len(codes)} samples...")
        from tqdm.auto import tqdm
        feats = np.stack([extract_13_features(c) for c in tqdm(codes, desc="13 Features", leave=False)])
        xgb_prob = self.xgb_model.predict_proba(feats)[:, 1].reshape(-1, 1)
        expert = np.hstack([feats, xgb_prob])
        expert = self.scaler.transform(expert).astype(np.float32)

        ds = HybridCodeDataset(codes, expert, None, self.tokenizer, self.max_length)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        test_probs = np.zeros(len(codes))
        for i, model in enumerate(self.models, 1):
            print(f"Predicting with model {i}/{len(self.models)}...")
            all_probs = []
            for batch in tqdm(dl, leave=False, desc=f"Model {i}"):
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attention_mask"].to(self.device)
                exp = batch["expert_feats"].to(self.device)

                with torch.cuda.amp.autocast(enabled=True):
                    logits, _, _ = model(input_ids, attn_mask, exp)
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.append(probs)

            test_probs += np.concatenate(all_probs) / len(self.models)

        return test_probs


# ============================================================================
# 9. MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="SemEval-2026 Task 13 Hybrid Training v2")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Đường dẫn chứa model weights (cho inference)")
    parser.add_argument("--inference_only", action="store_true",
                        help="Chỉ chạy inference, không train")
    parser.add_argument("--train_sample", type=int, default=None,
                        help="Số samples mỗi balanced subset (default: 50000)")
    parser.add_argument("--n_ensemble", type=int, default=None,
                        help="Số models trong ensemble (default: 3)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Số epochs per model (default: 3)")
    args, _ = parser.parse_known_args()

    if args.data_dir: cfg.data_dir = args.data_dir
    if args.output_dir: cfg.output_dir = args.output_dir
    if args.train_sample: cfg.train_sample = args.train_sample
    if args.n_ensemble: cfg.n_ensemble = args.n_ensemble
    if args.epochs: cfg.epochs = args.epochs

    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"\n[CLI Overrides] Data: {cfg.data_dir} | Output: {cfg.output_dir}")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- INFERENCE ONLY MODE ---
    if args.inference_only:
        print("\n=== RUNNING INFERENCE ONLY ===")
        model_dir = args.model_dir if args.model_dir else cfg.output_dir
        inferencer = HybridInferencer(
            model_dir, cfg.backbone_name, device, cfg.max_length, cfg.n_ensemble)

        data_dir = Path(cfg.data_dir)
        test_path = data_dir / "test.parquet"

        if test_path.exists():
            print(f"Reading {test_path}...")
            df_test = pd.read_parquet(test_path)
            probs = inferencer.predict(df_test["code"].values, batch_size=cfg.batch_size * 2)

            _thresh = inferencer.threshold
            submission = pd.DataFrame({
                "ID": df_test["ID"].values,
                "label": (probs > _thresh).astype(int),
            })

            sub_path = Path("/kaggle/working/submission.csv") if IN_KAGGLE else Path(model_dir) / "submission.csv"
            submission.to_csv(sub_path, index=False)

            pd.DataFrame({
                "ID": df_test["ID"].values,
                "prob_ai": probs,
                "label": (probs > _thresh).astype(int),
            }).to_parquet(Path(model_dir) / "test_preds_p5.parquet", index=False)

            print(f"\n[DONE] Submission: {sub_path}")
            print(f"  Threshold: {_thresh:.4f}")
            print(f"  Predicted AI ratio: {(probs > _thresh).mean():.4f}")
            print(f"  Label distribution: {submission['label'].value_counts().to_dict()}")
            print(f"  Prob stats: mean={probs.mean():.4f}, std={probs.std():.4f}, "
                  f"min={probs.min():.4f}, max={probs.max():.4f}")
        else:
            print(f"[WARN] Không tìm thấy {test_path}")

        sys.exit(0)

    # ── 9.1 Load data ──
    print("\n[1/5] Loading data...")
    data_dir = Path(cfg.data_dir)
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "validation.parquet"
    test_path = data_dir / "test.parquet"

    df_train_full = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame(columns=["ID", "code", "label", "language"])

    print(f"  Full train={len(df_train_full):,}, Val={len(df_val):,}, Test={len(df_test):,}")
    print(f"  Train label distribution: {df_train_full['label'].value_counts().to_dict()}")

    # ── 9.2 Extract expert features (trên toàn bộ data, cache) ──
    print("\n[2/5] Extracting expert features (13-dim)...")

    cache_dir = Path(cfg.output_dir)

    def get_or_cache_features(df, name, code_col="code"):
        if len(df) == 0:
            return np.empty((0, 13), dtype=np.float32)
        cache = cache_dir / f"expert_feats_{name}_v2ast.npy"
        if cache.exists():
            feats = np.load(cache)
            if feats.shape[0] == len(df):
                print(f"  Loaded cached: {cache.name} {feats.shape}")
                return feats
            else:
                print(f"  Cache size mismatch ({feats.shape[0]} vs {len(df)}), re-extracting...")
        feats = np.stack([extract_13_features(c) for c in
                          tqdm(df[code_col], desc=f"  {name}", leave=False, mininterval=10.0)])
        np.save(cache, feats)
        print(f"  Cached: {cache.name} {feats.shape}")
        return feats

    feats_train_full = get_or_cache_features(df_train_full, "train")
    feats_val = get_or_cache_features(df_val, "val")
    feats_test = get_or_cache_features(df_test, "test")

    # ── 9.3 XGBoost probability (14th feature) — OOF to prevent leakage ──
    print("\n[3/5] Generating XGBoost probabilities (OOF for train)...")

    xgb_model_path = cache_dir / "xgb_for_p5_v2.pkl"
    xgb_oof_path = cache_dir / "xgb_oof_probs_v2.npy"

    from sklearn.model_selection import StratifiedKFold

    if xgb_model_path.exists() and xgb_oof_path.exists():
        with open(xgb_model_path, "rb") as f:
            xgb_data = pickle.load(f)
        xgb_prob_train_full = np.load(xgb_oof_path).reshape(-1, 1)
        print(f"  Loaded cached XGB model + OOF probs {xgb_prob_train_full.shape}")
    else:
        import xgboost as xgb_lib
        print("  Training XGBoost with 5-fold OOF (prevents data leakage)...")
        train_labels = df_train_full["label"].values
        xgb_oof = np.zeros(len(feats_train_full))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(feats_train_full, train_labels), 1):
            xgb_fold = xgb_lib.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", device="cuda" if device.type == "cuda" else "cpu",
                random_state=cfg.seed, verbosity=0,
            )
            xgb_fold.fit(feats_train_full[tr_idx], train_labels[tr_idx],
                         eval_set=[(feats_train_full[va_idx], train_labels[va_idx])],
                         verbose=False)
            xgb_oof[va_idx] = xgb_fold.predict_proba(feats_train_full[va_idx])[:, 1]
            fold_auc = roc_auc_score(train_labels[va_idx], xgb_oof[va_idx])
            print(f"    Fold {fold}: OOF AUC = {fold_auc:.4f}")
            del xgb_fold; gc.collect()

        xgb_prob_train_full = xgb_oof.reshape(-1, 1)
        np.save(xgb_oof_path, xgb_oof)
        print(f"  OOF AUC (full): {roc_auc_score(train_labels, xgb_oof):.4f}")

        # Final XGB on all train data (for val/test inference)
        print("  Training final XGBoost on all training data...")
        xgb_model = xgb_lib.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", device="cuda" if device.type == "cuda" else "cpu",
            random_state=cfg.seed, verbosity=0,
        )
        xgb_model.fit(feats_train_full, train_labels,
                      eval_set=[(feats_val, df_val["label"].values)],
                      verbose=False)
        xgb_data = {"model": xgb_model}
        with open(xgb_model_path, "wb") as f:
            pickle.dump(xgb_data, f)

    xgb_m = xgb_data if isinstance(xgb_data, dict) and "model" in xgb_data else {"model": xgb_data}
    xgb_model = xgb_m.get("model", xgb_m)
    if isinstance(xgb_model, dict):
        xgb_model = xgb_model.get("models", [xgb_model])[0]

    # Val/Test: use final model (only train uses OOF)
    xgb_prob_val = xgb_model.predict_proba(feats_val)[:, 1].reshape(-1, 1)
    xgb_prob_test = xgb_model.predict_proba(feats_test)[:, 1].reshape(-1, 1) if len(feats_test) > 0 else np.empty((0, 1))
    print(f"  OOF train probs: mean={xgb_prob_train_full.mean():.4f}, std={xgb_prob_train_full.std():.4f}")
    print(f"  Val probs:       mean={xgb_prob_val.mean():.4f}, std={xgb_prob_val.std():.4f}")

    # 14-dim expert features
    expert_train_full = np.hstack([feats_train_full, xgb_prob_train_full])
    expert_val = np.hstack([feats_val, xgb_prob_val])
    expert_test = np.hstack([feats_test, xgb_prob_test]) if len(feats_test) > 0 else np.empty((0, 14))

    # Z-score (fit trên toàn bộ train)
    scaler = StandardScaler()
    expert_train_full = scaler.fit_transform(expert_train_full).astype(np.float32)
    expert_val = scaler.transform(expert_val).astype(np.float32)
    expert_test = scaler.transform(expert_test).astype(np.float32) if len(expert_test) > 0 else expert_test

    with open(cache_dir / "zscore_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Expert features: train={expert_train_full.shape}, val={expert_val.shape}")

    # Lưu index mapping cho balanced subsets
    # (vì expert features đã tính trên full train, ta cần map index)
    df_train_full = df_train_full.reset_index(drop=True)

    # ── 9.4 Tokenizer ──
    print(f"\n[4/5] Loading tokenizer: {cfg.backbone_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_name)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── 9.5 Balanced Ensemble Training ──
    print(f"\n[5/5] Training Balanced Ensemble ({cfg.n_ensemble} models)...")
    print(f"  Mỗi model: {cfg.train_sample:,} samples (balanced 50/50)")
    print(f"  Epochs per model: {cfg.epochs}")
    print(f"  Progressive unfreezing: freeze {cfg.freeze_backbone_epochs} epochs đầu")

    codes_val = df_val["code"].values
    labels_val = df_val["label"].values

    ensemble_dir = cache_dir / "ensemble_models"
    ensemble_dir.mkdir(exist_ok=True)

    model_aucs = []

    for model_idx in range(1, cfg.n_ensemble + 1):
        model_seed = cfg.seed + model_idx * 100

        # Tạo balanced subset cho model này
        df_subset, original_indices = create_balanced_subset(df_train_full, cfg.train_sample, model_seed)

        # Lấy expert features tương ứng (dùng original index từ df_train_full)
        expert_subset = expert_train_full[original_indices]

        # Train
        auc = train_single_model(
            model_idx, df_subset, expert_subset,
            codes_val, expert_val, labels_val,
            tokenizer, device, cfg, ensemble_dir
        )
        model_aucs.append(auc)

    # -- Ensemble Summary --
    print(f"\n{'='*60}")
    print(f"  BALANCED ENSEMBLE SUMMARY ({cfg.n_ensemble} models)")
    print(f"  Individual AUCs: {[f'{a:.4f}' for a in model_aucs]}")
    print(f"  Mean AUC: {np.mean(model_aucs):.4f}")
    print(f"{'='*60}")

    # -- Ensemble validation --
    print("\n  Ensemble validation...")
    val_ds = HybridCodeDataset(codes_val, expert_val, labels_val, tokenizer, cfg.max_length)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

    val_probs_ens = np.zeros(len(codes_val))
    n_loaded = 0
    for model_idx in range(1, cfg.n_ensemble + 1):
        model_pt = ensemble_dir / f"model{model_idx}_best.pt"
        if not model_pt.exists():
            continue
        model = HybridFusionModel(
            cfg.backbone_name, cfg.expert_input_dim,
            cfg.expert_hidden_dim, cfg.projection_dim, cfg.dropout,
            cfg.expert_feat_mask_prob,
        ).to(device)
        model.load_state_dict(torch.load(model_pt, weights_only=True))
        probs, _, _ = evaluate(model, val_dl, device, cfg)
        val_probs_ens += probs
        n_loaded += 1
        del model; gc.collect(); torch.cuda.empty_cache()

    val_probs_ens /= max(n_loaded, 1)
    val_auc_ens = roc_auc_score(labels_val, val_probs_ens)
    val_f1_ens = f1_score(labels_val, (val_probs_ens > 0.5).astype(int))
    print(f"  Ensemble Val AUC: {val_auc_ens:.4f}")
    print(f"  Ensemble Val F1:  {val_f1_ens:.4f}")
    print(f"  Ensemble Predicted AI ratio: {(val_probs_ens > 0.5).mean():.4f}")

    # Classification report
    print("\n  Ensemble Classification Report (validation):")
    print(classification_report(labels_val, (val_probs_ens > 0.5).astype(int),
                                target_names=["Human (0)", "AI (1)"]))

    # -- [ANTI-OVERFIT] Fixed threshold = 0.5 --
    # Lý do: auto-search threshold trên val set quá dễ sẽ chọn threshold quá thấp
    # gây ra false positive cực đoan trên test set (86% predicted AI).
    optimal_threshold = 0.5
    opt_preds = (val_probs_ens > optimal_threshold).astype(int)
    mf1_fixed = f1_score(labels_val, opt_preds, average="macro")
    print(f"  [ANTI-OVERFIT] Fixed threshold: {optimal_threshold:.3f} (Macro F1={mf1_fixed:.4f})")
    print(f"  Val pred ratio (fixed): AI={opt_preds.mean():.4f} "
          f"(actual={labels_val.mean():.4f})")
    print(f"  Val label dist (fixed): "
          f"{{0: {(opt_preds==0).sum()}, 1: {(opt_preds==1).sum()}}}")

    # Save threshold for inference-only mode
    with open(cache_dir / "optimal_threshold.txt", "w") as f:
        f.write(f"{optimal_threshold:.6f}\n")

    del val_ds, val_dl
    gc.collect(); torch.cuda.empty_cache()

    # ── 9.6 Inference on test set ──
    print("\n[6/6] Test inference (ensemble)...")

    if len(df_test) > 0:
        test_ds = HybridCodeDataset(
            df_test["code"].values, expert_test, None, tokenizer, cfg.max_length)
        test_dl = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                             num_workers=2, pin_memory=True)

        test_probs = np.zeros(len(df_test))
        n_loaded = 0
        for model_idx in range(1, cfg.n_ensemble + 1):
            model_pt = ensemble_dir / f"model{model_idx}_best.pt"
            if not model_pt.exists():
                continue
            model = HybridFusionModel(
                cfg.backbone_name, cfg.expert_input_dim,
                cfg.expert_hidden_dim, cfg.projection_dim, cfg.dropout,
                cfg.expert_feat_mask_prob,
            ).to(device)
            model.load_state_dict(torch.load(model_pt, weights_only=True))
            probs, _, _ = evaluate(model, test_dl, device, cfg)
            test_probs += probs
            n_loaded += 1
            del model; gc.collect(); torch.cuda.empty_cache()

        test_probs /= max(n_loaded, 1)

        # Submission
        submission = pd.DataFrame({
            "ID": df_test["ID"].values,
            "label": (test_probs > optimal_threshold).astype(int),
        })
        if IN_KAGGLE:
            sub_path = Path("/kaggle/working/submission.csv")
        else:
            sub_path = cache_dir / "submission.csv"
        submission.to_csv(sub_path, index=False)

        # Probabilities
        pd.DataFrame({
            "ID": df_test["ID"].values,
            "prob_ai": test_probs,
            "label": (test_probs > optimal_threshold).astype(int),
        }).to_parquet(cache_dir / "test_preds_p5.parquet", index=False)

        print(f"\n  Submission: {sub_path}")
        print(f"  Threshold: {optimal_threshold:.3f}")
        print(f"  Predicted AI ratio: {(test_probs > optimal_threshold).mean():.4f}")
        print(f"  Label distribution: {submission['label'].value_counts().to_dict()}")
        print(f"  Prob stats: mean={test_probs.mean():.4f}, std={test_probs.std():.4f}")
    else:
        print("  Bỏ qua test inference (không tìm thấy test.parquet)")

    print(f"\n{'='*60}")
    print(f"  PHASE 5 v2 COMPLETE")
    print(f"  Ensemble Val AUC: {val_auc_ens:.4f}")
    print(f"  Ensemble Val F1:  {val_f1_ens:.4f}")
    print(f"  Models saved: {ensemble_dir}")
    print(f"{'='*60}")
