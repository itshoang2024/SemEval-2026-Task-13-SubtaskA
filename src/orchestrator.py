"""
CAMSP parity orchestrator.

This keeps the refactored package boundary while restoring the prediction
behavior of `semeval-2026-peak/src/05_v9_stacking.py`.
"""

import gc
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .config import PipelineConfig
from .data_utils import ArtifactDetector, DataIngestion, GeneratorFamilyEncoder, set_seed
from .features import CodeStyleExtractor, LLMPerplexityEngine
from .tuning import OODRatioTuner

logger = logging.getLogger(__name__)

PIPELINE_DEADLINE_SEC = 8 * 3600
CHECKPOINT_VERSION = "v9p"


def _safe_lang_col(df: pd.DataFrame) -> np.ndarray:
    """Returns a string language array, using Unknown when the column is absent."""
    if df is None:
        return None
    if "language" in df.columns:
        return df["language"].fillna("Unknown").astype(str).values
    logger.warning("No 'language' column found; using Unknown for all rows")
    return np.full(len(df), "Unknown", dtype=object)


def _ckpt_dir() -> str:
    """Returns the checkpoint directory, creating it if needed."""
    directory = "/kaggle/working/_ckpt" if os.path.isdir("/kaggle/working") else "/tmp/_ckpt"
    os.makedirs(directory, exist_ok=True)
    return directory


def _ckpt_name(name: str) -> str:
    """Returns the versioned checkpoint filename for 05_v9 parity arrays."""
    return f"{name}_{CHECKPOINT_VERSION}.npy"


def _save_ckpt(name: str, arr: np.ndarray) -> None:
    """Saves a numpy checkpoint under the current parity version."""
    path = os.path.join(_ckpt_dir(), _ckpt_name(name))
    np.save(path, arr)
    logger.info("Checkpoint saved: %s (%s)", path, arr.shape)


def _load_ckpt(name: str) -> np.ndarray:
    """Loads a versioned checkpoint if present."""
    path = os.path.join(_ckpt_dir(), _ckpt_name(name))
    if os.path.exists(path):
        arr = np.load(path)
        logger.info("Checkpoint loaded: %s (%s)", path, arr.shape)
        return arr
    return None


def _collect_language_classes(*dfs: pd.DataFrame) -> list:
    """Collects stable language one-hot classes from available split columns."""
    classes = set()
    for df in dfs:
        if df is not None and "language" in df.columns:
            classes.update(df["language"].fillna("Unknown").astype(str).unique().tolist())
    return sorted(classes) if classes else ["Unknown"]


def _language_onehot(languages: np.ndarray, n_rows: int, classes: list) -> np.ndarray:
    """Builds the 05_v9 language one-hot matrix."""
    onehot = np.zeros((n_rows, len(classes)), dtype=np.float32)
    if languages is None:
        return onehot
    lang_series = pd.Series(languages).fillna("Unknown").astype(str)
    for idx, label in enumerate(classes):
        onehot[:, idx] = (lang_series.values == label).astype(np.float32)
    return onehot


def _add_language_features(
    style_df: pd.DataFrame,
    languages: np.ndarray,
    classes: list,
) -> pd.DataFrame:
    """Appends 05_v9 language one-hot features to style features."""
    if languages is None:
        return style_df
    lang_matrix = _language_onehot(languages, len(style_df), classes)
    lang_df = pd.DataFrame(
        lang_matrix,
        columns=[f"lang_is_{label}" for label in classes],
        index=style_df.index,
    )
    return pd.concat(
        [style_df.reset_index(drop=True), lang_df.reset_index(drop=True)],
        axis=1,
    )


class CAMSPipeline:
    """End-to-end CAMSP inference pipeline."""

    def __init__(self, config: PipelineConfig = None) -> None:
        self.cfg = config or PipelineConfig()
        self.data_mgr = DataIngestion(self.cfg)
        self.style_eng = CodeStyleExtractor(self.cfg)
        self.ppl_eng = LLMPerplexityEngine(self.cfg)
        self.tuner = OODRatioTuner(self.cfg)

    def _truncate(self, codes) -> list:
        """Truncates code samples to the configured max_chars."""
        return [
            str(code)[: self.cfg.max_chars] if code is not None else ""
            for code in codes
        ]

    def _check_deadline(self, t_start: float, phase: str) -> None:
        """Deadline guard is intentionally disabled for full Kaggle execution."""
        return None

    def run(self) -> pd.DataFrame:
        """Executes the full pipeline and returns the submission DataFrame."""
        set_seed(self.cfg.seed)
        t_start = time.time()

        logger.info("=" * 60)
        logger.info("PHASE 1/7: Loading data")
        logger.info("=" * 60)
        tr_df, va_df, te_df, sa_df = self.data_mgr.load_splits()
        tr_full = pd.concat([tr_df, va_df], ignore_index=True)
        del tr_df, va_df
        gc.collect()
        logger.info("Train+Val combined: %d samples", len(tr_full))

        y_train = tr_full["label"].astype(int).values
        fw_train = GeneratorFamilyEncoder.build_weights(tr_full["generator"])

        te_artifacts = ArtifactDetector.detect(te_df["code"].values, self.cfg.special_tokens)
        sa_artifacts = (
            ArtifactDetector.detect(sa_df["code"].values, self.cfg.special_tokens)
            if sa_df is not None
            else None
        )

        train_langs = _safe_lang_col(tr_full)
        te_langs = _safe_lang_col(te_df)
        sa_langs = _safe_lang_col(sa_df)
        lang_classes = _collect_language_classes(tr_full, sa_df, te_df)
        logger.info("Language one-hot classes: %s", lang_classes)

        logger.info("=" * 60)
        logger.info("PHASE 2/7: Perplexity features")
        logger.info("=" * 60)

        ppl_tr = _load_ckpt("ppl_train")
        ppl_te = _load_ckpt("ppl_test")
        ppl_sa = _load_ckpt("ppl_sample")
        if ppl_tr is None or ppl_te is None or (sa_df is not None and ppl_sa is None):
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
            logger.info("PPL checkpoints found - skipping inference")

        self._check_deadline(t_start, "PPL")

        logger.info("=" * 60)
        logger.info("PHASE 3/7: Style feature extraction")
        logger.info("=" * 60)

        X_sty_all_ckpt = _load_ckpt("sty_train")
        X_sty_te_ckpt = _load_ckpt("sty_test")
        X_sty_sa_ckpt = _load_ckpt("sty_sample") if sa_df is not None else None

        if (
            X_sty_all_ckpt is not None
            and X_sty_te_ckpt is not None
            and (sa_df is None or X_sty_sa_ckpt is not None)
        ):
            logger.info("Style checkpoints found - skipping extraction")
            X_sty_all = X_sty_all_ckpt
            X_sty_te = X_sty_te_ckpt
            X_sty_sa = X_sty_sa_ckpt
        else:
            sty_tr = self.style_eng.extract_batch(tr_full["code"].values, "Train")
            sty_te = self.style_eng.extract_batch(te_df["code"].values, "Test")
            sty_sa = (
                self.style_eng.extract_batch(sa_df["code"].values, "Sample")
                if sa_df is not None
                else None
            )

            for idx, col in enumerate(LLMPerplexityEngine.FEATURE_NAMES):
                sty_tr[f"ppl_{col}"] = ppl_tr[:, idx]
                sty_te[f"ppl_{col}"] = ppl_te[:, idx]
                if sty_sa is not None and ppl_sa is not None:
                    sty_sa[f"ppl_{col}"] = ppl_sa[:, idx]

            sty_tr = _add_language_features(sty_tr, train_langs, lang_classes)
            sty_te = _add_language_features(sty_te, te_langs, lang_classes)
            if sty_sa is not None:
                sty_sa = _add_language_features(sty_sa, sa_langs, lang_classes)

            X_sty_all = sty_tr.astype(np.float32).values
            X_sty_te = sty_te.astype(np.float32).values
            X_sty_sa = sty_sa.astype(np.float32).values if sty_sa is not None else None

            _save_ckpt("sty_train", X_sty_all)
            _save_ckpt("sty_test", X_sty_te)
            if X_sty_sa is not None:
                _save_ckpt("sty_sample", X_sty_sa)

            del sty_tr, sty_te, sty_sa
            gc.collect()

        self._check_deadline(t_start, "Style")

        logger.info("=" * 60)
        logger.info("PHASE 4/7: K-Fold Stacking (%d folds)", self.cfg.n_folds)
        logger.info("=" * 60)

        n_train, n_test = len(tr_full), len(te_df)
        n_sample = len(sa_df) if sa_df is not None else 0
        oof = np.zeros((n_train, 4), dtype=np.float32)
        te_sum = np.zeros((n_test, 4), dtype=np.float32)
        sa_sum = np.zeros((n_sample, 4), dtype=np.float32) if n_sample > 0 else None

        logger.info("Pre-computing char vocabulary")
        cv_master = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.cfg.char_ngram_range,
            max_features=self.cfg.char_max_features,
            min_df=3,
            sublinear_tf=True,
            lowercase=False,
            dtype=np.float32,
        )
        cv_master.fit(self._truncate(tr_full["code"].values))
        char_vocab = cv_master.vocabulary_
        del cv_master
        gc.collect()

        wv = HashingVectorizer(
            analyzer="word",
            token_pattern=r"\b\w+\b",
            ngram_range=(1, 3),
            n_features=self.cfg.word_hash_features,
            alternate_sign=False,
            lowercase=False,
            norm="l2",
            dtype=np.float32,
        )

        skf = StratifiedKFold(
            n_splits=self.cfg.n_folds,
            shuffle=True,
            random_state=self.cfg.seed,
        )
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y_train)):
            fold_t0 = time.time()
            logger.info(
                "--- Fold %d/%d (train=%d, val=%d) ---",
                fold_idx + 1,
                self.cfg.n_folds,
                len(tr_idx),
                len(va_idx),
            )

            y_tr = y_train[tr_idx]
            fw_tr = fw_train[tr_idx]
            fold_codes = tr_full.iloc[tr_idx]["code"].values

            cv = TfidfVectorizer(
                analyzer="char",
                ngram_range=self.cfg.char_ngram_range,
                vocabulary=char_vocab,
                sublinear_tf=True,
                lowercase=False,
                dtype=np.float32,
            )
            Xct = cv.fit_transform(self._truncate(fold_codes))
            Xcv = cv.transform(self._truncate(tr_full.iloc[va_idx]["code"].values))
            Xce = cv.transform(self._truncate(te_df["code"].values))
            Xcs = cv.transform(self._truncate(sa_df["code"].values)) if sa_df is not None else None

            c1 = SGDClassifier(
                loss="log_loss",
                alpha=self.cfg.text_alpha,
                max_iter=self.cfg.text_max_iter,
                tol=1e-3,
                random_state=self.cfg.seed,
            )
            c1.fit(Xct, y_tr)
            oof[va_idx, 0] = c1.decision_function(Xcv).astype(np.float32)
            te_sum[:, 0] += c1.decision_function(Xce).astype(np.float32)
            if Xcs is not None:
                sa_sum[:, 0] += c1.decision_function(Xcs).astype(np.float32)

            c2 = SGDClassifier(
                loss="log_loss",
                alpha=self.cfg.text_alpha * 1.5,
                max_iter=self.cfg.text_max_iter,
                tol=1e-3,
                random_state=self.cfg.seed,
            )
            c2.fit(Xct, y_tr, sample_weight=fw_tr)
            oof[va_idx, 1] = c2.decision_function(Xcv).astype(np.float32)
            te_sum[:, 1] += c2.decision_function(Xce).astype(np.float32)
            if Xcs is not None:
                sa_sum[:, 1] += c2.decision_function(Xcs).astype(np.float32)
            del Xct, Xcv, Xce, Xcs, c1, c2, cv
            gc.collect()

            Xwt = wv.transform(self._truncate(fold_codes))
            Xwv = wv.transform(self._truncate(tr_full.iloc[va_idx]["code"].values))
            Xwe = wv.transform(self._truncate(te_df["code"].values))
            Xws = wv.transform(self._truncate(sa_df["code"].values)) if sa_df is not None else None

            c3 = SGDClassifier(
                loss="log_loss",
                alpha=self.cfg.text_alpha,
                max_iter=self.cfg.text_max_iter,
                tol=1e-3,
                random_state=self.cfg.seed,
            )
            c3.fit(Xwt, y_tr)
            oof[va_idx, 2] = c3.decision_function(Xwv).astype(np.float32)
            te_sum[:, 2] += c3.decision_function(Xwe).astype(np.float32)
            if Xws is not None:
                sa_sum[:, 2] += c3.decision_function(Xws).astype(np.float32)
            del Xwt, Xwv, Xwe, Xws, c3
            gc.collect()

            Xs_tr = X_sty_all[tr_idx]
            ys_tr = y_tr
            if len(Xs_tr) > self.cfg.style_subsample:
                sample_idx = np.random.choice(
                    len(Xs_tr),
                    self.cfg.style_subsample,
                    replace=False,
                )
                Xs_tr = Xs_tr[sample_idx]
                ys_tr = y_tr[sample_idx]

            c4 = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=63,
                min_samples_leaf=40,
                l2_regularization=0.1,
                random_state=self.cfg.seed,
            )
            c4.fit(Xs_tr, ys_tr)
            oof[va_idx, 3] = c4.predict_proba(X_sty_all[va_idx])[:, 1].astype(np.float32)
            te_sum[:, 3] += c4.predict_proba(X_sty_te)[:, 1].astype(np.float32)
            if X_sty_sa is not None:
                sa_sum[:, 3] += c4.predict_proba(X_sty_sa)[:, 1].astype(np.float32)
            del c4
            gc.collect()

            logger.info("Fold %d done in %.1fs", fold_idx + 1, time.time() - fold_t0)
            _save_ckpt("oof", oof)
            _save_ckpt("te_sum", te_sum)
            if sa_sum is not None:
                _save_ckpt("sa_sum", sa_sum)

        te_avg = te_sum / self.cfg.n_folds
        sa_avg = sa_sum / self.cfg.n_folds if sa_sum is not None else None
        del X_sty_all, X_sty_te, X_sty_sa
        gc.collect()

        self._check_deadline(t_start, "Stacking")

        logger.info("=" * 60)
        logger.info("PHASE 5/7: HGB Meta-Learner")
        logger.info("=" * 60)

        Xm_tr = np.column_stack([
            oof,
            ppl_tr,
            _language_onehot(train_langs, n_train, lang_classes),
        ])
        Xm_te = np.column_stack([
            te_avg,
            ppl_te,
            _language_onehot(te_langs, n_test, lang_classes),
        ])
        Xm_sa = (
            np.column_stack([
                sa_avg,
                ppl_sa,
                _language_onehot(sa_langs, n_sample, lang_classes),
            ])
            if sa_avg is not None and ppl_sa is not None
            else None
        )
        logger.info("Meta features: %d", Xm_tr.shape[1])

        meta = HistGradientBoostingClassifier(
            learning_rate=self.cfg.meta_lr,
            max_iter=self.cfg.meta_max_iter,
            max_leaf_nodes=self.cfg.meta_max_leaf_nodes,
            min_samples_leaf=50,
            l2_regularization=1.0,
            random_state=self.cfg.seed,
        )
        meta.fit(Xm_tr, y_train)
        meta_te = meta.predict_proba(Xm_te)[:, 1].astype(np.float32)
        meta_sa = meta.predict_proba(Xm_sa)[:, 1].astype(np.float32) if Xm_sa is not None else None

        _save_ckpt("meta_te", meta_te)
        if meta_sa is not None:
            _save_ckpt("meta_sa", meta_sa)
        del meta, Xm_tr, Xm_te, Xm_sa, oof
        gc.collect()

        logger.info("=" * 60)
        logger.info("PHASE 6/7: Adaptive Ratio Tuning")
        logger.info("=" * 60)

        if sa_df is not None and meta_sa is not None and sa_langs is not None:
            sa_lang_series = pd.Series(sa_langs).fillna("Unknown").astype(str)
            tune_cfg = self.tuner.tune(
                sa_df["label"].values,
                meta_sa,
                sa_lang_series,
                sa_artifacts,
            )
        else:
            tune_cfg = {
                "global": self.cfg.fallback_global_ratio,
                "l_map": dict(self.cfg.lang_priors),
                "shrink": 0.5,
            }

        norm_scores = self.tuner.rank_normalize(meta_te)
        preds = self.tuner.language_aware_predict(
            norm_scores,
            te_langs,
            tune_cfg["global"],
            tune_cfg["l_map"],
            tune_cfg["shrink"],
        )
        preds[te_artifacts] = 1

        logger.info("=" * 60)
        logger.info("PHASE 7/7: Saving submission")
        logger.info("=" * 60)

        id_col = "ID" if "ID" in te_df.columns else "id"
        sub = pd.DataFrame({"ID": te_df[id_col].values, "label": preds.astype(int)})
        out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
        out_path = os.path.join(out_dir, "submission.csv")
        sub.to_csv(out_path, index=False)

        logger.info("Submission saved: %s", out_path)
        logger.info(
            "Machine ratio: %.2f%% (%d / %d)",
            sub["label"].mean() * 100,
            sub["label"].sum(),
            len(sub),
        )

        if sa_df is not None and meta_sa is not None:
            sa_norm = self.tuner.rank_normalize(meta_sa)
            sa_preds = self.tuner.language_aware_predict(
                sa_norm,
                sa_langs,
                tune_cfg["global"],
                tune_cfg["l_map"],
                tune_cfg["shrink"],
            )
            sa_preds[sa_artifacts] = 1
            logger.info(
                "Sample F1: %.4f",
                f1_score(sa_df["label"].values, sa_preds, average="macro"),
            )

        total_min = (time.time() - t_start) / 60
        logger.info("Total elapsed: %.1f minutes (%.1f hours)", total_min, total_min / 60)
        return sub
