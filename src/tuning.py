"""
CAMSP v10 — OOD Adaptive Ratio Tuning.

Prevents catastrophic ratio collapse on unseen languages by using
constrained grid search with language-aware shrinkage interpolation.
"""

import logging
from typing import Dict, Optional

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
    def has_reliable_languages(languages, min_count: int = 8) -> bool:
        """Returns True when language labels are useful for per-language tuning."""
        if languages is None:
            return False
        lang_series = pd.Series(languages).fillna("Unknown").astype(str)
        counts = lang_series[lang_series != "Unknown"].value_counts()
        return int((counts >= min_count).sum()) >= 2

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

    def predict_from_config(
        self,
        scores: np.ndarray,
        tune_cfg: dict,
        languages: Optional[np.ndarray] = None,
        artifacts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Applies a tuned ratio configuration to raw scores."""
        norm = self.rank_normalize(scores)
        strategy = tune_cfg.get("strategy", "global")
        use_language = (
            strategy.startswith("language")
            and self.has_reliable_languages(languages)
        )
        if use_language:
            preds = self.language_aware_predict(
                norm,
                languages,
                tune_cfg["global"],
                tune_cfg.get("l_map", {}),
                tune_cfg.get("shrink", 0.0),
            )
        else:
            preds = self.apply_ratio(norm, tune_cfg.get("global", self.cfg.fallback_global_ratio))

        if tune_cfg.get("force_artifacts", False) and artifacts is not None:
            preds[np.asarray(artifacts, dtype=bool)] = 1
        return preds

    @staticmethod
    def _maybe_force_artifacts(preds: np.ndarray, artifacts: Optional[np.ndarray]) -> np.ndarray:
        forced = preds.copy()
        if artifacts is not None:
            forced[np.asarray(artifacts, dtype=bool)] = 1
        return forced

    def _consider(
        self,
        best: dict,
        labels: np.ndarray,
        preds: np.ndarray,
        strategy: str,
        global_ratio: float,
        lang_map: Optional[Dict[str, float]] = None,
        shrink: float = 0.0,
        force_artifacts: bool = False,
    ) -> dict:
        f1 = f1_score(labels, preds, average="macro")
        if f1 <= best["score"]:
            return best
        return {
            "score": float(f1),
            "strategy": strategy,
            "global": float(global_ratio),
            "l_map": (lang_map or {}).copy(),
            "shrink": float(shrink),
            "force_artifacts": bool(force_artifacts),
        }

    def tune(
        self,
        sample_labels: np.ndarray,
        sample_scores: np.ndarray,
        lang_series: pd.Series,
        forced_artifacts: np.ndarray,
        allow_language: bool = True,
    ) -> dict:
        """Runs constrained grid search to find optimal ratio configuration.

        Args:
            sample_labels: Ground-truth binary labels from test_sample.
            sample_scores: Raw model scores from the meta-learner.
            lang_series: Language labels for each sample.
            forced_artifacts: Boolean mask of detected hard artifacts.
            allow_language: If False, only global ratio strategies are tested.

        Returns:
            Dict with score, strategy, global, l_map, shrink, force_artifacts.
        """
        logger.info("Starting OOD ratio tuning")
        scores = self.rank_normalize(sample_scores)
        artifacts = (
            np.asarray(forced_artifacts, dtype=bool)
            if forced_artifacts is not None else None
        )
        best = {
            "score": -1.0,
            "strategy": "global",
            "global": self.cfg.fallback_global_ratio,
            "l_map": {},
            "shrink": 0.0,
            "force_artifacts": False,
        }

        # Always test global ratio strategies. This is also the only valid
        # mode when the full test set has no reliable language labels.
        for gr in self.cfg.global_ratio_grid:
            base_preds = self.apply_ratio(scores, float(gr))
            best = self._consider(
                best, sample_labels, base_preds, "global", float(gr)
            )
            if artifacts is not None and artifacts.any():
                best = self._consider(
                    best,
                    sample_labels,
                    self._maybe_force_artifacts(base_preds, artifacts),
                    "global_artifact",
                    float(gr),
                    force_artifacts=True,
                )

        use_language = allow_language and self.has_reliable_languages(lang_series)
        if not use_language:
            logger.info("Language-aware tuning skipped; using best global strategy")
            logger.info(
                "Tuned -> F1: %.4f | Strategy: %s | Ratio: %.2f | Force artifacts: %s",
                best["score"], best["strategy"], best["global"], best["force_artifacts"],
            )
            return best

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
                best = self._consider(
                    best,
                    sample_labels,
                    preds,
                    "language",
                    float(gr),
                    lang_map,
                    float(shrink),
                )
                if artifacts is not None and artifacts.any():
                    best = self._consider(
                        best,
                        sample_labels,
                        self._maybe_force_artifacts(preds, artifacts),
                        "language_artifact",
                        float(gr),
                        lang_map,
                        float(shrink),
                        force_artifacts=True,
                    )

        logger.info(
            "Tuned -> F1: %.4f | Strategy: %s | Ratio: %.2f | Shrink: %.2f | Force artifacts: %s",
            best["score"], best["strategy"], best["global"], best["shrink"],
            best["force_artifacts"],
        )
        return best

    def tune_global_only(
        self,
        sample_labels: np.ndarray,
        sample_scores: np.ndarray,
        forced_artifacts: np.ndarray,
    ) -> dict:
        """Finds the best global ratio treating ALL samples as one group.

        Used when the test set has no 'language' column, making per-language
        ratios useless. This re-tunes the global ratio by evaluating on the
        sample WITHOUT any language-based splitting.

        Args:
            sample_labels: Ground-truth binary labels from test_sample.
            sample_scores: Raw model scores from the meta-learner.
            forced_artifacts: Boolean mask of detected hard artifacts.

        Returns:
            Global-only tune config dict.
        """
        logger.info("Re-tuning global ratio for language-free test set")
        cfg = self.tune(
            sample_labels,
            sample_scores,
            pd.Series(np.full(len(sample_labels), "Unknown", dtype=object)),
            forced_artifacts,
            allow_language=False,
        )

        logger.info(
            "Global-only tuning -> strategy: %s | ratio: %.2f | F1=%.4f",
            cfg["strategy"], cfg["global"], cfg["score"],
        )
        return cfg
