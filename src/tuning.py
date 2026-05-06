"""
CAMSP 05_v9 parity OOD adaptive ratio tuning.

Prevents catastrophic ratio collapse on unseen languages by using
constrained grid search with language-aware shrinkage interpolation.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class OODRatioTuner:
    """Tunes machine-generation ratio thresholds per language.

    The core problem: on OOD test data with unseen languages, a global
    ratio threshold can collapse to extreme values (e.g., 5%), causing
    the model to label everything as human. This tuner solves it via:

    1. **Global grid search** over ``[ratio_floor, ratio_ceil]``.
    2. **Per-language ratio** optimized independently on the test sample.
    3. **Shrinkage interpolation** that blends global and language ratios
       to regularize languages with too few samples.

    Args:
        config: Pipeline configuration with ratio grids and constraints.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    @staticmethod
    def rank_normalize(scores: np.ndarray) -> np.ndarray:
        """Converts raw scores to uniform ranks in [0, 1).

        Args:
            scores: Raw prediction scores.

        Returns:
            Rank-normalized scores preserving relative ordering.
        """
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float32)
        if len(scores) <= 1:
            ranks[:] = 0.5
            return ranks
        ranks[order] = np.linspace(0.0, 1.0, len(scores), endpoint=False, dtype=np.float32)
        return ranks

    @staticmethod
    def apply_ratio(scores: np.ndarray, ratio: float) -> np.ndarray:
        """Applies a fixed ratio threshold via top-k selection.

        Args:
            scores: Rank-normalized scores.
            ratio: Fraction of samples to label as machine-generated.

        Returns:
            Binary prediction array (1=machine, 0=human).
        """
        preds = np.zeros(len(scores), dtype=np.int8)
        n = int(round(len(scores) * float(np.clip(ratio, 0.0, 1.0))))
        if n > 0:
            preds[np.argsort(scores)[::-1][:n]] = 1
        return preds

    def language_aware_predict(
        self,
        scores: np.ndarray,
        languages: np.ndarray,
        global_ratio: float,
        lang_map: Dict[str, float],
        shrink: float,
    ) -> np.ndarray:
        """Generates predictions with per-language ratio interpolation.

        Args:
            scores: Rank-normalized prediction scores.
            languages: Language label for each sample.
            global_ratio: The system-wide base ratio.
            lang_map: Per-language optimal ratios.
            shrink: Interpolation weight (0 = global only, 1 = per-lang only).

        Returns:
            Binary prediction array.
        """
        preds = np.zeros(len(scores), dtype=np.int8)
        lang_series = pd.Series(languages).fillna("Unknown").astype(str)
        for lang in lang_series.unique():
            idx = np.where(lang_series.values == lang)[0]
            lang_ratio = lang_map.get(lang, global_ratio)
            adj = float(np.clip(
                (1.0 - shrink) * global_ratio + shrink * lang_ratio, 0.0, 1.0
            ))
            preds[idx] = self.apply_ratio(scores[idx], adj)
        return preds

    def tune(
        self,
        sample_labels: np.ndarray,
        sample_scores: np.ndarray,
        lang_series: pd.Series,
        forced_artifacts: np.ndarray,
    ) -> dict:
        """Runs constrained grid search to find optimal ratio configuration.

        Args:
            sample_labels: Ground-truth binary labels from test_sample.
            sample_scores: Raw model scores from the meta-learner.
            lang_series: Language labels for each sample.
            forced_artifacts: Boolean mask of detected hard artifacts.

        Returns:
            Dict with keys: score, global, l_map, shrink.
        """
        logger.info("Starting OOD Adaptive Shrinkage Tuning")
        scores = self.rank_normalize(sample_scores)
        best = {
            "score": -1.0,
            "global": self.cfg.fallback_global_ratio,
            "l_map": dict(self.cfg.lang_priors),
            "shrink": 0.0,
        }

        for gr in self.cfg.global_ratio_grid:
            lang_map = {}
            for lang in lang_series.unique():
                idx = np.where(lang_series.values == lang)[0]
                if len(idx) < 8:
                    lang_map[lang] = float(gr)
                    continue
                best_sub = -1.0
                for r in self.cfg.lang_ratio_grid:
                    s = f1_score(
                        sample_labels[idx],
                        self.apply_ratio(scores[idx], r),
                        average="macro",
                    )
                    if s > best_sub:
                        best_sub = s
                        lang_map[lang] = float(r)

            for shrink in self.cfg.shrink_grid:
                preds = self.language_aware_predict(
                    scores, lang_series.values, float(gr), lang_map, shrink
                )
                preds[forced_artifacts] = 1
                f1 = f1_score(sample_labels, preds, average="macro")
                if f1 > best["score"]:
                    best = {
                        "score": float(f1),
                        "global": float(gr),
                        "l_map": lang_map.copy(),
                        "shrink": float(shrink),
                    }

        logger.info(
            "Tuned -> F1: %.4f | Ratio: %.2f | Shrink: %.2f",
            best["score"], best["global"], best["shrink"],
        )
        return best
