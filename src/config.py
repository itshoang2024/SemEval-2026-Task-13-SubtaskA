"""
CAMSP 05_v9 parity pipeline configuration module.

Centralizes all hyperparameters, model paths, and tuning grids
using Python dataclasses for type safety and documentation.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PipelineConfig:
    """Immutable configuration container for the CAMSP pipeline.

    Groups parameters into logical sections:
        - Reproducibility: seed
        - Text Vectorization: char n-gram, word hashing
        - Style Feature Engineering: compression, entropy
        - LLM Perplexity: model candidates, token budget
        - Stacking Meta-Learner: HGB hyperparameters
        - OOD Ratio Tuning: grid search bounds

    Attributes:
        seed: Global random seed for reproducibility.
        data_dir: Auto-discovered path to parquet data files.
        char_max_features: Vocabulary cap for char TF-IDF vectorizer.
        char_ngram_range: Character n-gram range (min, max).
        word_hash_features: Dimensionality for hashing vectorizer.
        max_chars: Truncation limit per code sample (bytes).
        text_alpha: L2 regularization strength for SGD classifiers.
        text_max_iter: Maximum SGD training epochs.
        style_subsample: Row limit for HGB style model training.
        ppl_candidates: Ordered list of LLM checkpoint paths to try.
        ppl_max_tokens: Maximum token length per LLM forward pass.
        ppl_batch_size: Batch size for LLM inference.
        ppl_train_subsample: Number of training samples for LLM perplexity.
        ppl_time_budget_sec: Total seconds allocated for LLM computation.
        n_folds: Number of stratified folds for stacking.
        meta_lr: Learning rate for the HGB meta-learner.
        meta_max_iter: Maximum boosting iterations for meta-learner.
        meta_max_leaf_nodes: Tree complexity cap for meta-learner.
        global_ratio_grid: Search grid for global prediction ratio.
        lang_ratio_grid: Search grid for per-language prediction ratio.
        shrink_grid: Interpolation weights between global and language ratios.
        lang_priors: Default per-language ratios used when tuning data is absent.
        fallback_global_ratio: Default ratio when tuning data is unavailable.
        special_tokens: LLM control tokens indicating AI-generated artifacts.
    """

    # --- Reproducibility ---
    seed: int = 42
    data_dir: Optional[str] = None

    # --- Text Vectorization ---
    char_max_features: int = 80_000
    char_ngram_range: Tuple[int, int] = (3, 6)
    word_hash_features: int = 2**20
    max_chars: int = 4_500
    text_alpha: float = 2e-6
    text_max_iter: int = 20

    # --- Style Feature Engineering ---
    style_subsample: int = 350_000

    # --- LLM Perplexity ---
    ppl_candidates: List[str] = field(default_factory=lambda: [
        "/kaggle/input/qwen2.5-coder/transformers/0.5b-instruct/1",
        "/kaggle/input/qwen2.5-coder/transformers/1.5b-instruct/1",
        "/kaggle/input/qwen2.5-coder/transformers/0.5b/1",
        "/kaggle/input/qwen2.5-coder/transformers/1.5b/1",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    ])
    ppl_load_mode: str = field(
        default_factory=lambda: os.getenv("CAMSP_PPL_LOAD_MODE", "4bit").lower()
    )
    ppl_max_tokens: int = 64
    ppl_batch_size: int = 64
    ppl_train_subsample: int = 50_000
    ppl_time_budget_sec: int = 7_200  # 120 minutes, matching 05_v9

    # --- Stacking Meta-Learner ---
    n_folds: int = 5
    meta_lr: float = 0.05
    meta_max_iter: int = 300
    meta_max_leaf_nodes: int = 31

    # --- OOD Ratio Tuning ---
    ratio_floor: float = 0.10
    ratio_ceil: float = 0.40
    global_ratio_grid: np.ndarray = field(
        default_factory=lambda: np.arange(0.10, 0.41, 0.01)
    )
    lang_ratio_grid: np.ndarray = field(
        default_factory=lambda: np.arange(0.05, 0.41, 0.01)
    )
    shrink_grid: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    lang_priors: Dict[str, float] = field(
        default_factory=lambda: {
            "C": 0.20,
            "C#": 0.23,
            "C++": 0.17,
            "Go": 0.27,
            "Java": 0.16,
            "JavaScript": 0.38,
            "PHP": 0.06,
            "Python": 0.26,
        }
    )
    fallback_global_ratio: float = 0.22

    # --- Internal Lexicons ---
    special_tokens: List[str] = field(default_factory=lambda: [
        "\x3c|endoftext|\x3e",
        "\x3c|im_end|\x3e",
        "\x3c|assistant|\x3e",
        "\x3c|user|\x3e",
        "\x3c|system|\x3e",
        "\x3c|pad|\x3e",
        "\x3c|begin|\x3e",
        "\x3c|end|\x3e",
        "\x3c|start_header_id|\x3e",
        "\x3c|im_start|\x3e",
        "\x3c|eot_id|\x3e",
    ])
