"""
CAMSP v10.2 — Pipeline Orchestrator.

Wires together data ingestion, feature extraction, stacking,
meta-learning, and adaptive ratio tuning into a single callable pipeline.

v10.2 Critical Fixes:
- Safe language column handling (test.parquet has no 'language' column)
- NPY checkpointing for all expensive operations (PPL, style, OOF, meta)
- 8-hour total budget guard with graceful early-exit
"""

import gc
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .config import PipelineConfig
from .data_utils import ArtifactDetector, DataIngestion, GeneratorFamilyEncoder, set_seed
from .features import CodeStyleExtractor, LLMPerplexityEngine
from .tuning import OODRatioTuner

logger = logging.getLogger(__name__)

# Maximum wall-clock for the entire pipeline (safety margin for Kaggle 12h)
PIPELINE_DEADLINE_SEC = 8 * 3600  # 8 hours hard cap


def _safe_lang_col(df: pd.DataFrame) -> np.ndarray:
    """Extracts language column safely; returns 'Unknown' if missing."""
    if df is None:
        return None
    if "language" in df.columns:
        return df["language"].fillna("Unknown").astype(str).values
    logger.warning("No 'language' column found — using 'Unknown' for all rows")
    return np.full(len(df), "Unknown", dtype=object)


def _ckpt_dir() -> str:
    """Returns checkpoint directory (creates if needed)."""
    d = "/kaggle/working/_ckpt" if os.path.isdir("/kaggle/working") else "/tmp/_ckpt"
    os.makedirs(d, exist_ok=True)
    return d


def _out_dir() -> str:
    """Returns output directory for Kaggle or local runs."""
    return "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def _save_ckpt(name: str, arr: np.ndarray) -> None:
    """Saves a numpy array checkpoint."""
    path = os.path.join(_ckpt_dir(), f"{name}.npy")
    np.save(path, arr)
    logger.info("Checkpoint saved: %s (%s)", path, arr.shape)


def _load_ckpt(name: str) -> np.ndarray:
    """Loads a checkpoint if it exists, returns None otherwise."""
    path = os.path.join(_ckpt_dir(), f"{name}.npy")
    if os.path.exists(path):
        arr = np.load(path)
        logger.info("Checkpoint loaded: %s (%s)", path, arr.shape)
        return arr
    return None


def _load_score_ckpt(name: str, expected_rows: int) -> np.ndarray:
    """Loads a 1D score checkpoint and validates row count."""
    arr = _load_ckpt(name)
    if arr is None:
        return None
    if arr.ndim != 1 or len(arr) != expected_rows:
        logger.warning(
            "Ignoring checkpoint %s: expected shape (%d,), got %s",
            name, expected_rows, arr.shape,
        )
        return None
    return arr.astype(np.float32, copy=False)


def _load_matrix_ckpt(name: str, expected_rows: int, expected_cols: int) -> np.ndarray:
    """Loads a 2D checkpoint and validates shape."""
    arr = _load_ckpt(name)
    if arr is None:
        return None
    expected = (expected_rows, expected_cols)
    if arr.ndim != 2 or arr.shape != expected:
        logger.warning(
            "Ignoring checkpoint %s: expected shape %s, got %s",
            name, expected, arr.shape,
        )
        return None
    return arr.astype(np.float32, copy=False)


def _ppl_coverage(arr: np.ndarray) -> dict:
    """Summarizes how many PPL rows contain non-zero token counts."""
    if arr is None or len(arr) == 0:
        return {"rows": 0, "covered": 0, "coverage": 0.0}
    covered = int(np.count_nonzero(arr[:, -1] > 0))
    return {
        "rows": int(len(arr)),
        "covered": covered,
        "coverage": covered / max(int(len(arr)), 1),
    }


def _jsonable(obj):
    """Converts numpy-heavy metric objects to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _save_metrics(metrics: dict, path: str) -> None:
    """Writes run metrics as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(metrics), fh, indent=2, sort_keys=True)
    logger.info("Run metrics saved: %s", path)


class CAMSPipeline:
    """Compression-Aware Meta-Stacking Pipeline orchestrator.

    Execution flow:
        1. Auto-discover and load parquet splits.
        2. Extract LLM perplexity features (sequential completion).
        3. Extract compression-aware style features.
        4. K-fold stacking with 4 base estimators.
        5. HGB meta-learner on OOF predictions + perplexity.
        6. Adaptive ratio tuning on test_sample.
        7. Generate submission.csv.

    All expensive operations save .npy checkpoints so that a crash
    late in the pipeline does not lose earlier computation.

    Args:
        config: Pipeline configuration. Uses defaults if None.
    """

    def __init__(self, config: PipelineConfig = None) -> None:
        self.cfg = config or PipelineConfig()
        self.data_mgr = DataIngestion(self.cfg)
        self.style_eng = CodeStyleExtractor(self.cfg)
        self.ppl_eng = LLMPerplexityEngine(self.cfg)
        self.tuner = OODRatioTuner(self.cfg)

    def _truncate(self, codes) -> list:
        """Truncates code samples to max_chars for vectorization."""
        return [str(x)[: self.cfg.max_chars] if x else "" for x in codes]

    def _base_model_names(self) -> list:
        """Returns enabled base model names in score-column order."""
        names = ["char_full", "char_family", "word_hash", "style_hgb"]
        if self.cfg.enable_style_et:
            names.append("style_et")
        return names

    def _style_ckpt_name(self, split: str) -> str:
        """Returns style checkpoint name for the configured style version."""
        if self.style_eng.style_version == "v1":
            return f"sty_{split}"
        return f"sty_{split}_{self.style_eng.style_version}"

    def _meta_base_signature_ok(self, base_names: list) -> bool:
        """Checks whether reusable meta scores match the enabled base models."""
        saved = _load_ckpt("meta_base_models")
        if saved is None:
            default_base = ["char_full", "char_family", "word_hash", "style_hgb"]
            if list(base_names) != default_base:
                logger.warning(
                    "Ignoring meta checkpoints: missing meta_base_models.npy for non-default base models"
                )
                return False
            return True

        saved_names = [str(name) for name in saved.tolist()]
        if saved_names != list(base_names):
            logger.warning(
                "Ignoring meta checkpoints: expected base models %s, got %s",
                base_names, saved_names,
            )
            return False
        return True

    def _check_deadline(self, t_start: float, phase: str) -> None:
        """Deadline guard disabled to allow full 12h Kaggle execution."""
        pass

    def _blend_component(self, base_scores: np.ndarray, component: str, base_names: list) -> np.ndarray:
        """Returns a rank-normalized base-score component for blending."""
        if component == "base_mean":
            cols = [
                self.tuner.rank_normalize(base_scores[:, j])
                for j in range(base_scores.shape[1])
            ]
            return np.column_stack(cols).mean(axis=1).astype(np.float32)
        if component not in base_names:
            raise ValueError(f"Unknown blend component: {component}")
        idx = base_names.index(component)
        return self.tuner.rank_normalize(base_scores[:, idx])

    def _apply_score_blend(
        self,
        meta_scores: np.ndarray,
        base_scores: np.ndarray,
        blend_cfg: dict,
        base_names: list,
    ) -> np.ndarray:
        """Applies a tuned convex blend to raw meta and base scores."""
        meta_norm = self.tuner.rank_normalize(meta_scores)
        component_scores = self._blend_component(
            base_scores, blend_cfg["component"], base_names
        )
        meta_weight = float(blend_cfg["meta_weight"])
        return (
            meta_weight * meta_norm + (1.0 - meta_weight) * component_scores
        ).astype(np.float32)

    def _tune_score_blend(
        self,
        labels: np.ndarray,
        meta_sa: np.ndarray,
        base_sa: np.ndarray,
        lang_series: pd.Series,
        artifacts: np.ndarray,
        allow_language: bool,
        base_names: list,
    ) -> tuple:
        """Tunes a small convex blend between meta scores and base scores."""
        best_tune = None
        best_blend = None
        candidates = ["base_mean"] + list(base_names)
        meta_norm = self.tuner.rank_normalize(meta_sa)

        for component in candidates:
            component_scores = self._blend_component(base_sa, component, base_names)
            for meta_weight in self.cfg.score_blend_grid:
                blended_sa = (
                    float(meta_weight) * meta_norm
                    + (1.0 - float(meta_weight)) * component_scores
                ).astype(np.float32)
                tune_cfg = self.tuner.tune(
                    labels,
                    blended_sa,
                    lang_series,
                    artifacts,
                    allow_language=allow_language,
                )
                if best_tune is None or tune_cfg["score"] > best_tune["score"]:
                    best_tune = tune_cfg
                    best_blend = {
                        "enabled": True,
                        "component": component,
                        "meta_weight": float(meta_weight),
                    }

        logger.info(
            "Score blend tuned -> F1: %.4f | component=%s | meta_weight=%.2f",
            best_tune["score"], best_blend["component"], best_blend["meta_weight"],
        )
        return best_tune, best_blend

    def _tune_save_submission(
        self,
        te_df: pd.DataFrame,
        sa_df: pd.DataFrame,
        te_langs: np.ndarray,
        sa_langs: np.ndarray,
        te_artifacts: np.ndarray,
        sa_artifacts: np.ndarray,
        meta_te: np.ndarray,
        meta_sa: np.ndarray,
        t_start: float,
        metrics: dict,
        base_te: np.ndarray = None,
        base_sa: np.ndarray = None,
        base_names: list = None,
    ) -> pd.DataFrame:
        """Tunes ratios, writes submission.csv, exports metrics, and returns it."""
        # -- 6. Adaptive Ratio Tuning --
        logger.info("=" * 60)
        logger.info("PHASE 6/7: Adaptive Ratio Tuning")
        logger.info("=" * 60)

        test_has_reliable_language = self.tuner.has_reliable_languages(te_langs)
        sample_has_reliable_language = self.tuner.has_reliable_languages(sa_langs)
        allow_language = test_has_reliable_language and sample_has_reliable_language

        if sa_df is not None and meta_sa is not None:
            sa_lang_series = pd.Series(sa_langs).fillna("Unknown").astype(str)
            baseline_tune = self.tuner.tune(
                sa_df["label"].values,
                meta_sa,
                sa_lang_series,
                sa_artifacts,
                allow_language=allow_language,
            )
            if self.cfg.enable_score_blend and base_te is not None and base_sa is not None:
                candidate_tune, blend_cfg = self._tune_score_blend(
                    sa_df["label"].values,
                    meta_sa,
                    base_sa,
                    sa_lang_series,
                    sa_artifacts,
                    allow_language,
                    base_names or self._base_model_names(),
                )
                blended_te = self._apply_score_blend(
                    meta_te, base_te, blend_cfg, base_names or self._base_model_names()
                )
                blended_sa = self._apply_score_blend(
                    meta_sa, base_sa, blend_cfg, base_names or self._base_model_names()
                )
                baseline_preds = self.tuner.predict_from_config(
                    meta_te, baseline_tune, te_langs, te_artifacts
                )
                candidate_preds = self.tuner.predict_from_config(
                    blended_te, candidate_tune, te_langs, te_artifacts
                )
                baseline_ratio = float(baseline_preds.mean())
                candidate_ratio = float(candidate_preds.mean())
                score_gain = float(candidate_tune["score"] - baseline_tune["score"])
                ratio_shift = abs(candidate_ratio - baseline_ratio)

                blend_cfg.update({
                    "baseline_score": float(baseline_tune["score"]),
                    "score_gain": score_gain,
                    "baseline_machine_ratio": baseline_ratio,
                    "candidate_machine_ratio": candidate_ratio,
                    "ratio_shift": ratio_shift,
                    "ratio_shift_limit": 0.03,
                })
                if score_gain > 0.0 and ratio_shift <= 0.03:
                    meta_te = blended_te
                    meta_sa = blended_sa
                    tune_cfg = candidate_tune
                    tune_cfg["blend"] = blend_cfg
                else:
                    reason = "machine_ratio_shift" if ratio_shift > 0.03 else "no_sample_f1_gain"
                    logger.info(
                        "Score blend disabled: %s | score_gain=%.6f | ratio_shift=%.4f",
                        reason, score_gain, ratio_shift,
                    )
                    tune_cfg = baseline_tune
                    tune_cfg["blend"] = {
                        **blend_cfg,
                        "enabled": False,
                        "requested": True,
                        "reason": reason,
                    }
            else:
                if self.cfg.enable_score_blend:
                    logger.warning("Score blend requested but base score matrices are unavailable")
                tune_cfg = baseline_tune
                tune_cfg["blend"] = {"enabled": False}
        else:
            tune_cfg = {
                "score": None,
                "strategy": "global",
                "global": self.cfg.fallback_global_ratio,
                "l_map": {},
                "shrink": 0.0,
                "force_artifacts": True,
                "blend": {"enabled": False},
            }
            logger.warning("No sample scores available; using fallback tune config")

        if tune_cfg["strategy"].startswith("language") and not test_has_reliable_language:
            logger.warning("Test languages are not reliable; falling back to global prediction")
            tune_cfg["strategy"] = "global_artifact" if tune_cfg.get("force_artifacts") else "global"
            tune_cfg["l_map"] = {}
            tune_cfg["shrink"] = 0.0

        # -- 7. Save Submission --
        logger.info("=" * 60)
        logger.info("PHASE 7/7: Saving submission")
        logger.info("=" * 60)

        preds = self.tuner.predict_from_config(meta_te, tune_cfg, te_langs, te_artifacts)

        id_col = "ID" if "ID" in te_df.columns else "id"
        sub = pd.DataFrame({"ID": te_df[id_col].values, "label": preds.astype(int)})

        out_dir = _out_dir()
        out_path = os.path.join(out_dir, "submission.csv")
        sub.to_csv(out_path, index=False)

        machine_ratio = float(sub["label"].mean())
        logger.info("Submission saved: %s", out_path)
        logger.info(
            "Machine ratio: %.2f%% (%d / %d)",
            machine_ratio * 100, int(sub["label"].sum()), len(sub),
        )

        sample_f1 = None
        if sa_df is not None and meta_sa is not None:
            sa_preds = self.tuner.predict_from_config(meta_sa, tune_cfg, sa_langs, sa_artifacts)
            sample_f1 = float(f1_score(sa_df["label"].values, sa_preds, average="macro"))
            logger.info("Sample F1: %.4f", sample_f1)

        total_min = (time.time() - t_start) / 60
        logger.info("Total elapsed: %.1f minutes (%.1f hours)", total_min, total_min / 60)

        metrics.update({
            "tuning": tune_cfg,
            "sample_f1": sample_f1,
            "machine_ratio": machine_ratio,
            "submission_path": out_path,
            "total_minutes": float(total_min),
        })
        metrics_path = self.cfg.metrics_path or os.path.join(out_dir, "run_metrics.json")
        _save_metrics(metrics, metrics_path)
        return sub

    def run(self) -> pd.DataFrame:
        """Executes the full CAMSP pipeline end-to-end.

        Returns:
            Submission DataFrame with columns [ID, label].
        """
        set_seed(self.cfg.seed)
        t_start = time.time()
        metrics = {
            "ppl_load_mode": self.cfg.ppl_load_mode,
            "style_version": self.style_eng.style_version,
            "enable_style_et": bool(self.cfg.enable_style_et),
            "enable_score_blend": bool(self.cfg.enable_score_blend),
            "reuse_meta_scores": bool(self.cfg.reuse_meta_scores),
            "tuning_only": bool(self.cfg.tuning_only),
            "base_models": self._base_model_names(),
            "checkpoint_usage": {
                "ppl": False,
                "style": False,
                "meta": False,
                "base_scores": False,
            },
        }

        # ── 1. Data Loading ──
        logger.info("=" * 60)
        logger.info("PHASE 1/7: Loading data")
        logger.info("=" * 60)
        tr_df, va_df, te_df, sa_df = self.data_mgr.load_splits()
        tr_full = pd.concat([tr_df, va_df], ignore_index=True)
        del tr_df, va_df
        gc.collect()
        logger.info("Train+Val combined: %d samples", len(tr_full))
        metrics["row_counts"] = {
            "train_val": int(len(tr_full)),
            "test": int(len(te_df)),
            "sample": int(len(sa_df)) if sa_df is not None else 0,
        }

        y_train = tr_full["label"].astype(int).values
        fw_train = GeneratorFamilyEncoder.build_weights(tr_full["generator"])

        te_artifacts = ArtifactDetector.detect(te_df["code"].values, self.cfg.special_tokens)
        sa_artifacts = (
            ArtifactDetector.detect(sa_df["code"].values, self.cfg.special_tokens)
            if sa_df is not None else None
        )

        # Safe language extraction (test.parquet does NOT have 'language')
        te_langs = _safe_lang_col(te_df)
        sa_langs = _safe_lang_col(sa_df)

        meta_te = None
        meta_sa = None
        if self.cfg.reuse_meta_scores or self.cfg.tuning_only:
            logger.info("Trying meta score checkpoints for tuning-only workflow")
            base_names = self._base_model_names()
            meta_signature_ok = self._meta_base_signature_ok(base_names)
            meta_te = _load_score_ckpt("meta_te", len(te_df))
            if sa_df is not None:
                meta_sa = _load_score_ckpt("meta_sa", len(sa_df))
            base_te = None
            base_sa = None
            base_scores_ready = True
            if self.cfg.enable_score_blend:
                te_sum_ckpt = _load_matrix_ckpt("te_sum", len(te_df), len(base_names))
                if sa_df is not None:
                    sa_sum_ckpt = _load_matrix_ckpt("sa_sum", len(sa_df), len(base_names))
                else:
                    sa_sum_ckpt = None
                base_scores_ready = te_sum_ckpt is not None and (sa_df is None or sa_sum_ckpt is not None)
                if base_scores_ready:
                    metrics["checkpoint_usage"]["base_scores"] = True
                    base_te = te_sum_ckpt / self.cfg.n_folds
                    base_sa = sa_sum_ckpt / self.cfg.n_folds if sa_sum_ckpt is not None else None

            if (
                meta_signature_ok and
                meta_te is not None
                and (sa_df is None or meta_sa is not None)
                and (not self.cfg.enable_score_blend or base_scores_ready)
            ):
                metrics["checkpoint_usage"]["meta"] = True
                logger.info("Meta score checkpoints found; skipping PPL/style/stacking/meta phases")
                return self._tune_save_submission(
                    te_df, sa_df, te_langs, sa_langs, te_artifacts, sa_artifacts,
                    meta_te, meta_sa, t_start, metrics, base_te, base_sa, base_names,
                )
            if self.cfg.tuning_only:
                raise FileNotFoundError(
                    "CAMSP_TUNING_ONLY=1 requires valid meta_te.npy/meta_sa.npy"
                    " and, when score blending is enabled, compatible te_sum.npy/sa_sum.npy"
                )
            logger.warning("Meta score checkpoints incomplete; continuing full pipeline")

        # ── 2. LLM Perplexity (Sequential Completion) ──
        logger.info("=" * 60)
        logger.info("PHASE 2/7: LLM Perplexity")
        logger.info("=" * 60)

        ppl_tr = _load_ckpt("ppl_train")
        ppl_te = _load_ckpt("ppl_test")
        ppl_sa = _load_ckpt("ppl_sample")

        if ppl_te is None:
            ppl_tr, ppl_te, ppl_sa = self.ppl_eng.execute(
                tr_full["code"].values,
                te_df["code"].values,
                sa_df["code"].values if sa_df is not None else None,
            )
            _save_ckpt("ppl_train", ppl_tr)
            _save_ckpt("ppl_test", ppl_te)
            if ppl_sa is not None:
                _save_ckpt("ppl_sample", ppl_sa)
        else:
            logger.info("PPL checkpoints found — skipping LLM inference")
            metrics["checkpoint_usage"]["ppl"] = True
            if ppl_tr is None:
                ppl_tr = np.zeros((len(tr_full), len(LLMPerplexityEngine.FEATURE_NAMES)), dtype=np.float32)
            if ppl_sa is None and sa_df is not None:
                ppl_sa = np.zeros((len(sa_df), len(LLMPerplexityEngine.FEATURE_NAMES)), dtype=np.float32)

        metrics["ppl_coverage"] = {
            "train": _ppl_coverage(ppl_tr),
            "test": _ppl_coverage(ppl_te),
            "sample": _ppl_coverage(ppl_sa) if ppl_sa is not None else None,
        }
        logger.info(
            "PPL coverage -> train: %.1f%% | test: %.1f%% | sample: %s",
            metrics["ppl_coverage"]["train"]["coverage"] * 100,
            metrics["ppl_coverage"]["test"]["coverage"] * 100,
            (
                f"{metrics['ppl_coverage']['sample']['coverage'] * 100:.1f}%"
                if metrics["ppl_coverage"]["sample"] is not None else "N/A"
            ),
        )

        self._check_deadline(t_start, "LLM Perplexity")

        # ── 3. Style Features ──
        logger.info("=" * 60)
        logger.info("PHASE 3/7: Style feature extraction")
        logger.info("=" * 60)

        X_sty_all_ckpt = _load_ckpt(self._style_ckpt_name("train"))
        X_sty_te_ckpt = _load_ckpt(self._style_ckpt_name("test"))
        X_sty_sa_ckpt = (
            _load_ckpt(self._style_ckpt_name("sample"))
            if sa_df is not None else None
        )

        if (
            X_sty_all_ckpt is not None
            and X_sty_te_ckpt is not None
            and (sa_df is None or X_sty_sa_ckpt is not None)
        ):
            logger.info("Style checkpoints found — skipping extraction")
            metrics["checkpoint_usage"]["style"] = True
            X_sty_all = X_sty_all_ckpt
            X_sty_te = X_sty_te_ckpt
            X_sty_sa = X_sty_sa_ckpt
        else:
            sty_tr = self.style_eng.extract_batch(tr_full["code"].values, "Train")
            sty_te = self.style_eng.extract_batch(te_df["code"].values, "Test")
            sty_sa = (
                self.style_eng.extract_batch(sa_df["code"].values, "Sample")
                if sa_df is not None else None
            )

            # Merge perplexity into style
            for k, col in enumerate(LLMPerplexityEngine.FEATURE_NAMES):
                sty_tr[f"ppl_{col}"] = ppl_tr[:, k]
                sty_te[f"ppl_{col}"] = ppl_te[:, k]
                if sty_sa is not None and ppl_sa is not None:
                    sty_sa[f"ppl_{col}"] = ppl_sa[:, k]

            X_sty_all = sty_tr.astype(np.float32).values
            X_sty_te = sty_te.astype(np.float32).values
            X_sty_sa = sty_sa.astype(np.float32).values if sty_sa is not None else None

            _save_ckpt(self._style_ckpt_name("train"), X_sty_all)
            _save_ckpt(self._style_ckpt_name("test"), X_sty_te)
            if X_sty_sa is not None:
                _save_ckpt(self._style_ckpt_name("sample"), X_sty_sa)

            del sty_tr, sty_te, sty_sa
            gc.collect()

        self._check_deadline(t_start, "Style Features")

        # ── 4. K-Fold Stacking ──
        logger.info("=" * 60)
        logger.info("PHASE 4/7: K-Fold Stacking (%d folds)", self.cfg.n_folds)
        logger.info("=" * 60)

        n_train, n_test = len(tr_full), len(te_df)
        n_sample = len(sa_df) if sa_df is not None else 0
        base_names = self._base_model_names()
        n_base = len(base_names)

        oof = np.zeros((n_train, n_base), dtype=np.float32)
        te_sum = np.zeros((n_test, n_base), dtype=np.float32)
        sa_sum = np.zeros((n_sample, n_base), dtype=np.float32) if n_sample > 0 else None

        # Pre-fit char vocabulary
        logger.info("Pre-computing char vocabulary")
        cv_master = TfidfVectorizer(
            analyzer="char", ngram_range=self.cfg.char_ngram_range,
            max_features=self.cfg.char_max_features, min_df=3,
            sublinear_tf=True, lowercase=False, dtype=np.float32,
        )
        cv_master.fit(self._truncate(tr_full["code"].values))
        char_vocab = cv_master.vocabulary_
        del cv_master
        gc.collect()

        wv = HashingVectorizer(
            analyzer="word", token_pattern=r"\b\w+\b", ngram_range=(1, 3),
            n_features=self.cfg.word_hash_features, alternate_sign=False,
            lowercase=False, norm="l2", dtype=np.float32,
        )

        skf = StratifiedKFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=self.cfg.seed)
        for fi, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y_train)):
            t_fold = time.time()
            logger.info("--- Fold %d/%d (train=%d, val=%d) ---", fi + 1, self.cfg.n_folds, len(tr_idx), len(va_idx))

            y_tr = y_train[tr_idx]
            fw_tr = fw_train[tr_idx]
            fold_codes = tr_full.iloc[tr_idx]["code"].values

            # Base 1+2: Char TF-IDF
            cv = TfidfVectorizer(
                analyzer="char", ngram_range=self.cfg.char_ngram_range,
                vocabulary=char_vocab, sublinear_tf=True, lowercase=False, dtype=np.float32,
            )
            Xct = cv.fit_transform(self._truncate(fold_codes))
            Xcv = cv.transform(self._truncate(tr_full.iloc[va_idx]["code"].values))
            Xce = cv.transform(self._truncate(te_df["code"].values))
            Xcs = cv.transform(self._truncate(sa_df["code"].values)) if sa_df is not None else None

            c1 = SGDClassifier(loss="log_loss", alpha=self.cfg.text_alpha, max_iter=self.cfg.text_max_iter, tol=1e-3, random_state=self.cfg.seed)
            c1.fit(Xct, y_tr)
            oof[va_idx, 0] = c1.decision_function(Xcv).astype(np.float32)
            te_sum[:, 0] += c1.decision_function(Xce).astype(np.float32)
            if Xcs is not None:
                sa_sum[:, 0] += c1.decision_function(Xcs).astype(np.float32)

            c2 = SGDClassifier(loss="log_loss", alpha=self.cfg.text_alpha * 1.5, max_iter=self.cfg.text_max_iter, tol=1e-3, random_state=self.cfg.seed)
            c2.fit(Xct, y_tr, sample_weight=fw_tr)
            oof[va_idx, 1] = c2.decision_function(Xcv).astype(np.float32)
            te_sum[:, 1] += c2.decision_function(Xce).astype(np.float32)
            if Xcs is not None:
                sa_sum[:, 1] += c2.decision_function(Xcs).astype(np.float32)
            del Xct, Xcv, Xce, Xcs, c1, c2, cv
            gc.collect()

            # Base 3: Word hash
            Xwt = wv.transform(self._truncate(fold_codes))
            Xwv = wv.transform(self._truncate(tr_full.iloc[va_idx]["code"].values))
            Xwe = wv.transform(self._truncate(te_df["code"].values))
            Xws = wv.transform(self._truncate(sa_df["code"].values)) if sa_df is not None else None

            c3 = SGDClassifier(loss="log_loss", alpha=self.cfg.text_alpha, max_iter=self.cfg.text_max_iter, tol=1e-3, random_state=self.cfg.seed)
            c3.fit(Xwt, y_tr)
            oof[va_idx, 2] = c3.decision_function(Xwv).astype(np.float32)
            te_sum[:, 2] += c3.decision_function(Xwe).astype(np.float32)
            if Xws is not None:
                sa_sum[:, 2] += c3.decision_function(Xws).astype(np.float32)
            del Xwt, Xwv, Xwe, Xws, c3
            gc.collect()

            # Base 4: Style HGB
            Xs_tr, ys_tr = X_sty_all[tr_idx], y_tr
            if len(Xs_tr) > self.cfg.style_subsample:
                si = np.random.choice(len(Xs_tr), self.cfg.style_subsample, replace=False)
                Xs_tr, ys_tr = Xs_tr[si], y_tr[si]

            c4 = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=250, max_leaf_nodes=63,
                min_samples_leaf=40, l2_regularization=0.1, random_state=self.cfg.seed,
            )
            c4.fit(Xs_tr, ys_tr)
            oof[va_idx, 3] = c4.predict_proba(X_sty_all[va_idx])[:, 1].astype(np.float32)
            te_sum[:, 3] += c4.predict_proba(X_sty_te)[:, 1].astype(np.float32)
            if X_sty_sa is not None:
                sa_sum[:, 3] += c4.predict_proba(X_sty_sa)[:, 1].astype(np.float32)
            del c4
            gc.collect()

            if self.cfg.enable_style_et:
                c5 = ExtraTreesClassifier(
                    n_estimators=240,
                    max_features="sqrt",
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=self.cfg.seed + fi,
                )
                c5.fit(Xs_tr, ys_tr)
                style_et_idx = base_names.index("style_et")
                oof[va_idx, style_et_idx] = (
                    c5.predict_proba(X_sty_all[va_idx])[:, 1].astype(np.float32)
                )
                te_sum[:, style_et_idx] += (
                    c5.predict_proba(X_sty_te)[:, 1].astype(np.float32)
                )
                if X_sty_sa is not None:
                    sa_sum[:, style_et_idx] += (
                        c5.predict_proba(X_sty_sa)[:, 1].astype(np.float32)
                    )
                del c5
                gc.collect()

            logger.info("Fold %d done in %.1fs", fi + 1, time.time() - t_fold)

            # Checkpoint after each fold
            _save_ckpt("oof", oof)
            _save_ckpt("te_sum", te_sum)
            if sa_sum is not None:
                _save_ckpt("sa_sum", sa_sum)

        te_avg = te_sum / self.cfg.n_folds
        sa_avg = sa_sum / self.cfg.n_folds if sa_sum is not None else None
        del X_sty_all, X_sty_te, X_sty_sa
        gc.collect()

        self._check_deadline(t_start, "K-Fold Stacking")

        # ── 5. Meta-Learner ──
        logger.info("=" * 60)
        logger.info("PHASE 5/7: HGB Meta-Learner")
        logger.info("=" * 60)

        Xm_tr = np.column_stack([oof, ppl_tr])
        Xm_te = np.column_stack([te_avg, ppl_te])
        Xm_sa = np.column_stack([sa_avg, ppl_sa]) if sa_avg is not None and ppl_sa is not None else None

        meta = HistGradientBoostingClassifier(
            learning_rate=self.cfg.meta_lr, max_iter=self.cfg.meta_max_iter,
            max_leaf_nodes=self.cfg.meta_max_leaf_nodes, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=self.cfg.seed,
        )
        meta.fit(Xm_tr, y_train)
        meta_te = meta.predict_proba(Xm_te)[:, 1].astype(np.float32)
        meta_sa = meta.predict_proba(Xm_sa)[:, 1].astype(np.float32) if Xm_sa is not None else None

        _save_ckpt("meta_te", meta_te)
        if meta_sa is not None:
            _save_ckpt("meta_sa", meta_sa)
        _save_ckpt("meta_base_models", np.array(base_names, dtype="U32"))

        del meta, Xm_tr, Xm_te, Xm_sa, oof
        gc.collect()

        return self._tune_save_submission(
            te_df, sa_df, te_langs, sa_langs, te_artifacts, sa_artifacts,
            meta_te, meta_sa, t_start, metrics, te_avg, sa_avg, base_names,
        )
