"""
CAMSP v10.1 — Feature Engineering Engines.

Two complementary feature extraction systems:
1. CodeStyleExtractor: Language-agnostic compression and structural metrics.
2. LLMPerplexityEngine: Sequential Completion strategy with adaptive batch sizing.

LLM Engine v10.1 Changes:
- Sequential Completion: Test 100% → Sample 100% → Train (remaining budget)
- Adaptive OOM recovery: auto-halves batch size on CUDA OOM
- Expanded context window: 128 tokens for richer perplexity discrimination
"""

import bz2
import gc
import logging
import os
import subprocess
import sys
import time
import zlib
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class CodeStyleExtractor:
    """Extracts compression-aware stylometric features from source code.

    Produces language-agnostic signals that capture the structural
    regularity of AI-generated code vs human-written code:
    - Compression ratios (zlib, bz2)
    - Byte-level Shannon entropy
    - Indentation delta entropy
    - Line statistics

    Args:
        config: Pipeline configuration with ``max_chars`` attribute.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    def _extract_single(self, code: str) -> dict:
        """Computes style features for a single code sample.

        Args:
            code: Raw source code string.

        Returns:
            Dictionary of feature name -> float value.
        """
        if not isinstance(code, str) or len(code) == 0:
            return {}

        lines = code.split("\n")
        non_empty = [line for line in lines if line.strip()]
        f = {
            "line_count": max(len(lines), 1),
            "char_count": max(len(code), 1),
        }
        f["empty_line_ratio"] = 1.0 - (len(non_empty) / f["line_count"])

        # --- Line length statistics ---
        ll = np.array([len(line) for line in lines], dtype=np.float32)
        f["avg_line_length"] = float(ll.mean())
        f["std_line_length"] = float(ll.std())
        f["max_line_length"] = float(ll.max())
        f["line_len_cv"] = float(ll.std()) / max(float(ll.mean()), 1e-6)

        # --- Compression ratios ---
        tb = code[: self.cfg.max_chars].encode("utf-8", errors="replace")
        blen = max(len(tb), 1)
        f["zlib_ratio"] = len(zlib.compress(tb, level=1)) / blen if tb else 0.0

        if tb:
            f["bz2_ratio"] = len(bz2.compress(tb, compresslevel=9)) / blen
            byte_arr = np.frombuffer(tb, dtype=np.uint8)
            cnts = np.bincount(byte_arr, minlength=256)
            probs = cnts[cnts > 0] / byte_arr.size
            f["byte_entropy"] = float(-(probs * np.log2(probs)).sum())
        else:
            f["bz2_ratio"] = 0.0
            f["byte_entropy"] = 0.0

        # --- Indentation dynamics ---
        indents = [len(line) - len(line.lstrip()) for line in lines]
        if non_empty:
            ne_ind = np.array(
                [len(line) - len(line.lstrip()) for line in non_empty],
                dtype=np.float32,
            )
            f["indent_std"] = float(ne_ind.std())
            f["indent_unique"] = float(len(set(ne_ind.tolist())))
        else:
            f["indent_std"] = 0.0
            f["indent_unique"] = 0.0

        deltas = [abs(indents[i + 1] - indents[i]) for i in range(len(indents) - 1)]
        if deltas:
            dc = Counter(deltas)
            dt = sum(dc.values())
            pd_ = np.array(list(dc.values()), dtype=np.float64) / dt
            f["indent_delta_entropy"] = float(-(pd_ * np.log2(pd_ + 1e-12)).sum())
        else:
            f["indent_delta_entropy"] = 0.0

        # --- Character distribution ---
        cc = max(len(code), 1)
        char_counter = Counter(code)
        cp = np.array(list(char_counter.values()), dtype=np.float64) / cc
        f["char_entropy"] = float(-np.sum(cp * np.log2(cp + 1e-12)))
        f["unique_char_ratio"] = len(char_counter) / cc
        f["space_ratio"] = code.count(" ") / cc
        f["newline_ratio"] = code.count("\n") / cc

        # --- Code smell indicators ---
        f["has_markdown_fence"] = int("```" in code)
        f["has_special_token"] = int("\x3c|" in code)

        # --- Trigram repetition ---
        if len(code) >= 3:
            trigrams = [code[i : i + 3] for i in range(len(code) - 2)]
            tc = Counter(trigrams)
            f["trigram_rep_ratio"] = sum(1 for c in tc.values() if c > 1) / max(
                len(tc), 1
            )
        else:
            f["trigram_rep_ratio"] = 0.0

        return f

    def extract_batch(self, codes: np.ndarray, desc: str) -> pd.DataFrame:
        """Extracts style features for an entire array of code samples.

        Args:
            codes: NumPy array of raw code strings.
            desc: Human-readable label for progress logging.

        Returns:
            DataFrame with one row per sample, columns = feature names.
        """
        logger.info("Extracting CAMSP style features for %s (%d samples)", desc, len(codes))
        t0 = time.time()
        rows = []
        for i, code in enumerate(codes, 1):
            try:
                rows.append(self._extract_single(code))
            except Exception:
                rows.append({})
            if i % 100_000 == 0:
                logger.info("  %d / %d | %.0f it/s", i, len(codes), i / max(time.time() - t0, 1))
        df = pd.DataFrame(rows).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        logger.info("%s: done in %.1fs | shape=%s", desc, time.time() - t0, df.shape)
        return df


class LLMPerplexityEngine:
    """Computes token-level NLL features using a causal LM.

    **Sequential Completion Strategy (v10.1)**:
    Instead of fixed percentage splits, this engine runs each dataset
    to 100% completion in strict priority order:

        Priority 1: Test set (500k) — run until FULLY COMPLETE
        Priority 2: Sample set (1k)  — run until FULLY COMPLETE
        Priority 3: Train subsample  — consume ALL remaining budget

    This guarantees zero dead-zero rows in the test PPL features,
    eliminating the train/test distribution mismatch that crippled
    the meta-learner in v10.0.

    **Adaptive OOM Recovery**: If a batch triggers CUDA OOM, the engine
    automatically halves the batch size and retries, down to bs=8.

    Args:
        config: Pipeline configuration with LLM parameters.
    """

    FEATURE_NAMES = ["nll_mean", "nll_std", "nll_q25", "nll_q75", "token_count"]

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self._effective_bs = config.ppl_batch_size  # may shrink on OOM

    def execute(
        self,
        train_codes: np.ndarray,
        test_codes: np.ndarray,
        sample_codes: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Runs the full perplexity pipeline with Sequential Completion.

        Args:
            train_codes: Training code array (600k).
            test_codes: Test code array (500k).
            sample_codes: Optional test_sample code array (1k).

        Returns:
            Tuple of (ppl_train, ppl_test, ppl_sample) feature matrices.
        """
        logger.info(
            "LLM Perplexity Engine v10.1 — Sequential Completion | "
            "budget=%ds, tokens=%d, batch=%d, load_mode=%s",
            self.cfg.ppl_time_budget_sec,
            self.cfg.ppl_max_tokens,
            self.cfg.ppl_batch_size,
            self.cfg.ppl_load_mode,
        )

        model, tokenizer = self._load_model()
        if model is None:
            logger.warning("No LLM available — returning zero features for all sets")
            return (
                self._zeros(len(train_codes)),
                self._zeros(len(test_codes)),
                self._zeros(len(sample_codes)) if sample_codes is not None else None,
            )

        t_global = time.time()
        deadline = t_global + self.cfg.ppl_time_budget_sec

        # ── Priority 1: TEST SET (run to 100% completion) ──
        logger.info("="*60)
        logger.info("PRIORITY 1/3: Test set — %d samples (target: 100%%)", len(test_codes))
        logger.info("="*60)
        ppl_test, n_test = self._infer_until_done(
            test_codes, model, tokenizer, deadline
        )
        test_pct = n_test / len(test_codes) * 100
        logger.info(
            "Test LLM coverage: %d / %d (%.1f%%) in %.1f min",
            n_test, len(test_codes), test_pct, (time.time() - t_global) / 60,
        )

        # ── Priority 2: SAMPLE SET (run to 100% completion) ──
        ppl_sample = None
        if sample_codes is not None and time.time() < deadline:
            logger.info("="*60)
            logger.info("PRIORITY 2/3: Sample set — %d samples", len(sample_codes))
            logger.info("="*60)
            ppl_sample, n_sample = self._infer_until_done(
                sample_codes, model, tokenizer, deadline
            )
            logger.info("Sample LLM coverage: %d / %d (%.1f%%)", n_sample, len(sample_codes), n_sample / len(sample_codes) * 100)

        # ── Priority 3: TRAIN SUBSAMPLE (use ALL remaining budget) ──
        ppl_train = self._zeros(len(train_codes))
        remaining = deadline - time.time()
        if remaining > 60:
            n_sub = min(self.cfg.ppl_train_subsample, len(train_codes))
            logger.info("="*60)
            logger.info(
                "PRIORITY 3/3: Train subsample — %d / %d samples (%.0f min remaining)",
                n_sub, len(train_codes), remaining / 60,
            )
            logger.info("="*60)
            sub_idx = np.sort(np.random.choice(len(train_codes), n_sub, replace=False))
            ppl_sub, n_done = self._infer_until_done(
                train_codes[sub_idx], model, tokenizer, deadline
            )
            ppl_train[sub_idx[:n_done]] = ppl_sub[:n_done]
            logger.info("Train LLM coverage: %d / %d target", n_done, n_sub)
        else:
            logger.warning("No time remaining for train set PPL")

        # Cleanup GPU memory
        del model, tokenizer
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        total_min = (time.time() - t_global) / 60
        logger.info("LLM Perplexity Engine completed in %.1f min (budget was %.1f min)", total_min, self.cfg.ppl_time_budget_sec / 60)
        return ppl_train, ppl_test, ppl_sample

    def _load_model(self):
        """Attempts to load a causal LM from Kaggle model inputs.

        Tries each candidate path in order. The default loading mode is NF4
        4-bit quantization, but fp16/bf16/fp32 can be selected with
        ``PipelineConfig.ppl_load_mode`` or ``CAMSP_PPL_LOAD_MODE``.

        Returns:
            Tuple of (model, tokenizer) or (None, None) on failure.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("No CUDA available")
                return None, None
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.warning("transformers not installed")
            return None, None

        load_mode = (self.cfg.ppl_load_mode or "4bit").lower()
        if load_mode not in {"4bit", "fp16", "bf16", "fp32"}:
            logger.warning("Unsupported PPL load mode '%s' — falling back to 4bit", load_mode)
            load_mode = "4bit"

        for path in self.cfg.ppl_candidates:
            if path.startswith("/") and not os.path.isdir(path):
                continue
            try:
                logger.info("Trying LLM: %s (load_mode=%s)", path, load_mode)
                tokenizer = AutoTokenizer.from_pretrained(
                    path, trust_remote_code=True, padding_side="right"
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model_kwargs = {
                    "device_map": "auto",
                    "trust_remote_code": True,
                }
                if load_mode == "4bit":
                    try:
                        from transformers import BitsAndBytesConfig
                    except ImportError as exc:
                        raise ImportError(
                            "bitsandbytes/4-bit loading requires transformers BitsAndBytesConfig"
                        ) from exc
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                    loaded_desc = "BnB NF4 4-bit"
                else:
                    dtype_map = {
                        "fp16": torch.float16,
                        "bf16": torch.bfloat16,
                        "fp32": torch.float32,
                    }
                    model_kwargs["torch_dtype"] = dtype_map[load_mode]
                    loaded_desc = load_mode

                model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
                model.eval()
                logger.info("Loaded %s (%s)", path, loaded_desc)
                return model, tokenizer
            except Exception as exc:
                logger.warning("Failed %s: %s", path, exc)

        return None, None

    def _infer_until_done(
        self,
        codes: np.ndarray,
        model,
        tokenizer,
        deadline: float,
    ) -> Tuple[np.ndarray, int]:
        """Runs LLM inference until dataset is fully completed OR deadline hit.

        Implements adaptive OOM recovery: on CUDA OOM, batch size is
        halved and the failed batch is retried. Minimum batch size = 8.

        Args:
            codes: Array of code strings to process.
            model: Loaded causal LM.
            tokenizer: Corresponding tokenizer.
            deadline: Unix timestamp after which processing must stop.

        Returns:
            Tuple of (feature_matrix, n_samples_completed).
        """
        import torch

        n = len(codes)
        features = self._zeros(n)
        bs = self._effective_bs
        t0 = time.time()
        last_end = 0
        log_interval = max(1, 50_000 // max(bs, 1))  # log every ~50k samples

        start = 0
        while start < n:
            if time.time() >= deadline:
                logger.info("Deadline reached at %d / %d (%.1f%%)", start, n, start / n * 100)
                break

            end = min(start + bs, n)
            batch = [
                c[: self.cfg.max_chars] if isinstance(c, str) else ""
                for c in codes[start:end]
            ]

            try:
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.cfg.ppl_max_tokens,
                    padding=True,
                )
                ids = enc.input_ids.to(model.device)
                mask = enc.attention_mask.to(model.device)

                with torch.inference_mode():
                    logits = model(input_ids=ids, attention_mask=mask).logits

                sl = logits[:, :-1, :].contiguous()
                st = ids[:, 1:].contiguous()
                sm = mask[:, 1:].contiguous().float()
                nll = (
                    torch.nn.CrossEntropyLoss(reduction="none")(
                        sl.view(-1, sl.size(-1)), st.view(-1)
                    ).view(st.size())
                    * sm
                )

                for j in range(end - start):
                    vals = nll[j][sm[j].bool()].float().cpu().numpy()
                    if len(vals) == 0:
                        continue
                    q25, q75 = np.percentile(vals, [25, 75])
                    features[start + j] = [
                        np.mean(vals),
                        np.std(vals),
                        q25,
                        q75,
                        float(len(vals)),
                    ]

                del ids, mask, logits, sl, st, sm, nll
                torch.cuda.empty_cache()

                last_end = end
                batch_idx = start // bs
                if batch_idx % log_interval == 0 and start > 0:
                    elapsed = time.time() - t0
                    speed = end / elapsed
                    eta = (n - end) / max(speed, 1) / 60
                    logger.info(
                        "  %d / %d (%.1f%%) | %.0f samp/s | ETA %.1f min",
                        end, n, end / n * 100, speed, eta,
                    )

                start = end  # advance to next batch

            except torch.cuda.OutOfMemoryError:
                # Adaptive OOM recovery: halve batch size and retry
                torch.cuda.empty_cache()
                gc.collect()
                old_bs = bs
                bs = max(bs // 2, 8)
                self._effective_bs = bs
                logger.warning(
                    "CUDA OOM at batch [%d:%d] — reducing batch size %d → %d",
                    start, end, old_bs, bs,
                )
                # Don't advance start — retry the same batch with smaller bs

            except Exception as exc:
                logger.error("Inference error at [%d:%d]: %s", start, end, exc)
                last_end = end
                start = end  # skip this batch

        return features, last_end

    def _zeros(self, n: int) -> np.ndarray:
        """Creates a zero-filled feature matrix."""
        return np.zeros((n, len(self.FEATURE_NAMES)), dtype=np.float32)
