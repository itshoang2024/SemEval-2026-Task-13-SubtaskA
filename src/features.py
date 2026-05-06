"""
Feature engineering engines for the CAMSP parity pipeline.

This module intentionally preserves the feature semantics of
`semeval-2026-peak/src/05_v9_stacking.py` while keeping the refactored
package layout.
"""

import bz2
import gc
import logging
import os
import re
import time
import zlib
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig

logger = logging.getLogger(__name__)


class CodeStyleExtractor:
    """Extracts the 05_v9 handcrafted style feature set."""

    FEATURE_NAMES = [
        "char_count",
        "line_count",
        "empty_line_ratio",
        "avg_line_length",
        "std_line_length",
        "max_line_length",
        "median_ne_line_length",
        "indent_std",
        "indent_unique",
        "space_ratio",
        "tab_ratio",
        "newline_ratio",
        "digit_ratio",
        "uppercase_ratio",
        "punct_ratio",
        "operator_ratio",
        "char_entropy",
        "unique_char_ratio",
        "token_entropy",
        "unique_word_ratio",
        "avg_word_length",
        "hapax_ratio",
        "keyword_ratio",
        "avg_identifier_length",
        "single_char_id_ratio",
        "long_id_ratio",
        "snake_case_ratio",
        "camel_case_ratio",
        "identifier_diversity",
        "line_dup_ratio",
        "comment_ratio",
        "has_block_comment",
        "brace_balance_abs",
        "paren_balance_abs",
        "has_markdown_fence",
        "has_special_token",
        "has_llm_preamble",
        "compression_ratio",
        "bz2_ratio",
        "byte_entropy",
        "line_len_cv",
        "indent_delta_entropy",
        "trigram_rep_ratio",
    ]

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

    def _extract_single(self, code: str) -> dict:
        """Computes one row of 05_v9 style features."""
        if not isinstance(code, str) or len(code) == 0:
            return {}

        lines = code.split("\n")
        non_empty = [line for line in lines if line.strip()]
        words = re.findall(r"\b\w+\b", code)
        identifiers = re.findall(r"\b[a-zA-Z_]\w*\b", code)
        cc = max(len(code), 1)
        lc = max(len(lines), 1)
        wc = max(len(words), 1)
        f = {"char_count": cc, "line_count": lc}

        ll = np.array([len(line) for line in lines], dtype=np.float32)
        nel = (
            np.array([len(line) for line in non_empty], dtype=np.float32)
            if non_empty
            else np.array([0.0], dtype=np.float32)
        )
        indents = (
            np.array(
                [len(line) - len(line.lstrip()) for line in non_empty],
                dtype=np.float32,
            )
            if non_empty
            else np.array([0.0], dtype=np.float32)
        )

        f["empty_line_ratio"] = 1.0 - (len(non_empty) / lc)
        f["avg_line_length"] = float(ll.mean())
        f["std_line_length"] = float(ll.std())
        f["max_line_length"] = float(ll.max())
        f["median_ne_line_length"] = float(np.median(nel))
        f["indent_std"] = float(indents.std())
        f["indent_unique"] = float(len(set(indents.tolist())))
        f["space_ratio"] = code.count(" ") / cc
        f["tab_ratio"] = code.count("\t") / cc
        f["newline_ratio"] = code.count("\n") / cc
        f["digit_ratio"] = sum(ch.isdigit() for ch in code) / cc
        f["uppercase_ratio"] = sum(ch.isupper() for ch in code) / max(
            sum(ch.isalpha() for ch in code), 1
        )
        f["punct_ratio"] = sum(ch in "{}[]();,.:" for ch in code) / cc
        f["operator_ratio"] = len(re.findall(r"[+\-*/=<>!&|^~%]", code)) / cc

        char_counter = Counter(code)
        char_probs = np.array(list(char_counter.values()), dtype=np.float64) / cc
        f["char_entropy"] = float(-np.sum(char_probs * np.log2(char_probs + 1e-12)))
        f["unique_char_ratio"] = len(char_counter) / cc

        if words:
            word_counter = Counter(words)
            word_probs = np.array(list(word_counter.values()), dtype=np.float64) / len(words)
            f["token_entropy"] = float(
                -np.sum(word_probs * np.log2(word_probs + 1e-12))
            )
            f["unique_word_ratio"] = len(word_counter) / wc
            f["avg_word_length"] = float(np.mean([len(word) for word in words]))
            f["hapax_ratio"] = (
                sum(1 for count in word_counter.values() if count == 1)
                / len(word_counter)
            )
        else:
            f["token_entropy"] = 0.0
            f["unique_word_ratio"] = 0.0
            f["avg_word_length"] = 0.0
            f["hapax_ratio"] = 0.0

        keywords = {
            "def",
            "class",
            "if",
            "else",
            "for",
            "while",
            "return",
            "import",
            "from",
            "int",
            "void",
            "public",
            "private",
            "static",
            "new",
            "try",
            "catch",
            "except",
            "finally",
            "with",
            "as",
            "in",
            "not",
            "and",
            "or",
            "true",
            "false",
            "null",
            "none",
            "let",
            "const",
            "function",
            "func",
            "self",
            "this",
            "switch",
            "case",
            "break",
            "continue",
            "package",
            "namespace",
        }
        f["keyword_ratio"] = sum(1 for word in words if word.lower() in keywords) / wc

        if identifiers:
            ident_lengths = np.array(
                [len(token) for token in identifiers], dtype=np.float32
            )
            f["avg_identifier_length"] = float(ident_lengths.mean())
            f["single_char_id_ratio"] = (
                sum(1 for token in identifiers if len(token) == 1) / len(identifiers)
            )
            f["long_id_ratio"] = (
                sum(1 for token in identifiers if len(token) > 10) / len(identifiers)
            )
            f["snake_case_ratio"] = (
                sum(1 for token in identifiers if "_" in token and token != "_")
                / len(identifiers)
            )
            f["camel_case_ratio"] = (
                sum(1 for token in identifiers if re.search(r"[a-z][A-Z]", token))
                / len(identifiers)
            )
            f["identifier_diversity"] = len(set(identifiers)) / len(identifiers)
        else:
            f["avg_identifier_length"] = 0.0
            f["single_char_id_ratio"] = 0.0
            f["long_id_ratio"] = 0.0
            f["snake_case_ratio"] = 0.0
            f["camel_case_ratio"] = 0.0
            f["identifier_diversity"] = 0.0

        stripped_lines = [line.strip() for line in non_empty]
        f["line_dup_ratio"] = 1.0 - (
            len(set(stripped_lines)) / max(len(stripped_lines), 1)
        )
        f["comment_ratio"] = sum(
            1
            for line in non_empty
            if line.strip().startswith(("//", "#", "/*", "*", "--"))
        ) / max(len(non_empty), 1)
        f["has_block_comment"] = int("/*" in code or "'''" in code or '"""' in code)
        f["brace_balance_abs"] = abs(code.count("{") - code.count("}"))
        f["paren_balance_abs"] = abs(code.count("(") - code.count(")"))
        f["has_markdown_fence"] = int("```" in code)
        f["has_special_token"] = int("\x3c|" in code)
        f["has_llm_preamble"] = int(
            bool(re.match(r"^(Here is|Here's|Sure,|Certainly|Below is|The following)", code))
        )

        tb = code[: self.cfg.max_chars].encode("utf-8", errors="replace")
        blen = max(len(tb), 1)
        f["compression_ratio"] = len(zlib.compress(tb, level=1)) / blen if tb else 0.0
        if tb:
            f["bz2_ratio"] = len(bz2.compress(tb, compresslevel=9)) / blen
            byte_arr = np.frombuffer(tb, dtype=np.uint8)
            cnts = np.bincount(byte_arr, minlength=256)
            probs = cnts[cnts > 0] / byte_arr.size
            f["byte_entropy"] = float(-(probs * np.log2(probs)).sum())
        else:
            f["bz2_ratio"] = 0.0
            f["byte_entropy"] = 0.0

        f["line_len_cv"] = float(ll.std()) / max(float(ll.mean()), 1e-6)
        all_indents = [len(line) - len(line.lstrip()) for line in lines]
        deltas = [
            abs(all_indents[i + 1] - all_indents[i])
            for i in range(len(all_indents) - 1)
        ]
        if deltas:
            delta_counter = Counter(deltas)
            delta_probs = (
                np.array(list(delta_counter.values()), dtype=np.float64)
                / sum(delta_counter.values())
            )
            f["indent_delta_entropy"] = float(
                -(delta_probs * np.log2(delta_probs + 1e-12)).sum()
            )
        else:
            f["indent_delta_entropy"] = 0.0

        if len(code) >= 3:
            trigrams = [code[i : i + 3] for i in range(len(code) - 2)]
            trigram_counter = Counter(trigrams)
            f["trigram_rep_ratio"] = sum(
                1 for count in trigram_counter.values() if count > 1
            ) / max(len(trigram_counter), 1)
        else:
            f["trigram_rep_ratio"] = 0.0

        return f

    def extract_batch(self, codes: np.ndarray, desc: str) -> pd.DataFrame:
        """Extracts style features for an array of code samples."""
        logger.info("Extracting style features for %s (%d samples)", desc, len(codes))
        t0 = time.time()
        rows = []
        for i, code in enumerate(codes, 1):
            try:
                rows.append(self._extract_single(code))
            except Exception:
                rows.append({})
            if i % 100_000 == 0:
                logger.info(
                    "  %d / %d | %.0f it/s",
                    i,
                    len(codes),
                    i / max(time.time() - t0, 1),
                )
        df = (
            pd.DataFrame(rows)
            .reindex(columns=self.FEATURE_NAMES, fill_value=0.0)
            .fillna(0.0)
            .replace([np.inf, -np.inf], 0.0)
        )
        logger.info("%s: done in %.1fs | shape=%s", desc, time.time() - t0, df.shape)
        return df


class LLMPerplexityEngine:
    """Computes 05_v9 PPL features with compression-backed fallback."""

    FEATURE_NAMES = [
        "nll_mean",
        "nll_std",
        "nll_max",
        "nll_q25",
        "nll_q75",
        "nll_low_frac",
        "nll_iqr",
        "token_count",
    ]

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self._effective_bs = config.ppl_batch_size

    def execute(
        self,
        train_codes: np.ndarray,
        test_codes: np.ndarray,
        sample_codes: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Runs compression baseline first, then overwrites completed LLM rows."""
        logger.info(
            "LLM PPL 05_v9 parity | budget=%ds, tokens=%d, batch=%d, load_mode=%s",
            self.cfg.ppl_time_budget_sec,
            self.cfg.ppl_max_tokens,
            self.cfg.ppl_batch_size,
            self.cfg.ppl_load_mode,
        )

        ppl_train = self._compression_features(train_codes, "Train")
        ppl_test = self._compression_features(test_codes, "Test")
        ppl_sample = (
            self._compression_features(sample_codes, "Sample")
            if sample_codes is not None
            else None
        )

        model, tokenizer = self._load_model()
        if model is None:
            logger.warning("No LLM available; using compression-backed PPL features")
            return ppl_train, ppl_test, ppl_sample

        import torch

        t_global = time.time()
        budget = self.cfg.ppl_time_budget_sec
        logger.info("PPL budget: %ds (%.0f min)", budget, budget / 60)

        remaining = budget - (time.time() - t_global)
        if remaining > 120:
            ppl_test_llm, n_test = self._infer_with_budget(
                test_codes,
                model,
                tokenizer,
                time_budget_sec=remaining * 0.55,
                desc="Test-LLM",
            )
            ppl_test[:n_test] = ppl_test_llm[:n_test]
            logger.info(
                "Test LLM coverage: %d / %d (%.1f%%)",
                n_test,
                len(test_codes),
                n_test / len(test_codes) * 100,
            )

        remaining = budget - (time.time() - t_global)
        if sample_codes is not None and remaining > 30:
            ppl_sample_llm, n_sample = self._infer_with_budget(
                sample_codes,
                model,
                tokenizer,
                time_budget_sec=min(remaining, 120),
                desc="Sample-LLM",
            )
            ppl_sample[:n_sample] = ppl_sample_llm[:n_sample]

        remaining = budget - (time.time() - t_global)
        if remaining > 120:
            n_sub = min(self.cfg.ppl_train_subsample, len(train_codes))
            sub_idx = np.sort(np.random.choice(len(train_codes), n_sub, replace=False))
            ppl_sub, n_done = self._infer_with_budget(
                train_codes[sub_idx],
                model,
                tokenizer,
                time_budget_sec=remaining,
                desc="Train-LLM",
            )
            ppl_train[sub_idx[:n_done]] = ppl_sub[:n_done]
            logger.info("Train LLM coverage: %d / %d target", n_done, n_sub)

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("LLM PPL completed in %.1f min", (time.time() - t_global) / 60)
        return ppl_train, ppl_test, ppl_sample

    def _load_model(self):
        """Attempts to load a causal LM from Kaggle model inputs."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not torch.cuda.is_available():
                logger.warning("No CUDA available")
                return None, None
        except ImportError:
            logger.warning("transformers not installed")
            return None, None

        load_mode = (self.cfg.ppl_load_mode or "4bit").lower()
        if load_mode not in {"4bit", "fp16", "bf16", "fp32"}:
            logger.warning("Unsupported PPL load mode '%s'; falling back to 4bit", load_mode)
            load_mode = "4bit"

        for path in self.cfg.ppl_candidates:
            if path.startswith("/") and not os.path.isdir(path):
                continue
            try:
                logger.info("Trying LLM: %s", path)
                tokenizer = AutoTokenizer.from_pretrained(
                    path, trust_remote_code=True, padding_side="right"
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model_kwargs = {"device_map": "auto", "trust_remote_code": True}
                if load_mode == "4bit":
                    from transformers import BitsAndBytesConfig

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    ),
                    device_map="auto",
                    trust_remote_code=True,
                )
                model.eval()
                logger.info("Loaded %s (BnB NF4 4-bit)", path)
                return model, tokenizer
            except Exception as exc:
                logger.warning("Failed %s: %s", path, exc)

        return None, None

    def _infer_with_budget(
        self,
        codes: np.ndarray,
        model,
        tokenizer,
        time_budget_sec: float,
        desc: str,
    ) -> Tuple[np.ndarray, int]:
        """Runs LLM inference until complete or predicted budget exhaustion."""
        import torch

        n = len(codes)
        features = self._zeros(n)
        bs = self._effective_bs
        t0 = time.time()
        last_end = 0
        start = 0

        while start < n:
            end = min(start + bs, n)
            batch = [
                code[: self.cfg.max_chars] if isinstance(code, str) else ""
                for code in codes[start:end]
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

                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_targets = ids[:, 1:].contiguous()
                shifted_mask = mask[:, 1:].contiguous().float()
                nll = (
                    torch.nn.CrossEntropyLoss(reduction="none")(
                        shifted_logits.view(-1, shifted_logits.size(-1)),
                        shifted_targets.view(-1),
                    ).view(shifted_targets.size())
                    * shifted_mask
                )

                for j in range(end - start):
                    vals = nll[j][shifted_mask[j].bool()].float().cpu().numpy()
                    if len(vals) == 0:
                        continue
                    q25, q75 = np.percentile(vals, [25, 75])
                    features[start + j] = [
                        np.mean(vals),
                        np.std(vals),
                        np.max(vals),
                        q25,
                        q75,
                        np.mean(vals < 1.0),
                        q75 - q25,
                        float(len(vals)),
                    ]

                del ids, mask, logits, shifted_logits, shifted_targets, shifted_mask, nll
                torch.cuda.empty_cache()
                last_end = end

                elapsed = time.time() - t0
                if (start // max(bs, 1)) % 100 == 0 and start > 0:
                    speed = end / max(elapsed, 1e-9)
                    eta_min = (n - end) / max(speed, 1) / 60
                    logger.info(
                        "  %s: %d / %d (%.0f samp/s, ETA %.1fm)",
                        desc,
                        end,
                        n,
                        speed,
                        eta_min,
                    )

                if elapsed > 60 and end < n:
                    speed = end / max(elapsed, 1e-9)
                    if elapsed + (n - end) / max(speed, 1) > time_budget_sec:
                        logger.info("  %s: abort at %d / %d", desc, end, n)
                        return features, last_end

                start = end

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                old_bs = bs
                bs = max(bs // 2, 8)
                self._effective_bs = bs
                logger.warning(
                    "CUDA OOM at batch [%d:%d]; reducing batch size %d -> %d",
                    start,
                    end,
                    old_bs,
                    bs,
                )

            except Exception as exc:
                logger.error("Inference error at [%d:%d]: %s", start, end, exc)
                last_end = end
                start = end

        logger.info("  %s: done %d in %.1fs", desc, n, time.time() - t0)
        return features, last_end

    def _compression_features(self, codes: np.ndarray, desc: str) -> np.ndarray:
        """Computes the 8-column compression fallback used by 05_v9."""
        n = len(codes)
        features = self._zeros(n)
        t0 = time.time()
        for i, code in enumerate(codes):
            if not isinstance(code, str) or len(code) == 0:
                continue
            text = code[: self.cfg.max_chars].encode("utf-8", errors="replace")
            blen = max(len(text), 1)
            ratio = len(zlib.compress(text, level=1)) / blen
            mid = blen // 2
            if mid > 10:
                r1 = len(zlib.compress(text[:mid], level=1)) / mid
                r2 = len(zlib.compress(text[mid:], level=1)) / max(blen - mid, 1)
                q25, q75 = min(r1, r2), max(r1, r2)
            else:
                q25 = ratio
                q75 = ratio
            features[i] = [
                ratio,
                abs(q75 - q25) / 2,
                q75,
                q25,
                q75,
                ratio,
                q75 - q25,
                float(blen),
            ]
        logger.info("  %s: compression %d in %.1fs", desc, n, time.time() - t0)
        return features

    def _zeros(self, n: int) -> np.ndarray:
        """Creates a zero-filled feature matrix."""
        return np.zeros((n, len(self.FEATURE_NAMES)), dtype=np.float32)
