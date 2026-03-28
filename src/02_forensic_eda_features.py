# %% [markdown]
# # [EDA] Phase 2 & 3: Forensic EDA + Feature Engineering
# ## SemEval-2026 Task 13 -- AI-Generated Code Detection
#
# **Pipeline**: Integrity Check -> Forensic EDA -> Feature Engineering -> Correlation
#
# This script implements 3 major tasks:
# 1. **Integrity & Leakage Check**: duplicates, cross-set leakage, class imbalance
# 2. **Forensic EDA**: Stylometric + Statistical + Structural features on 20k sample
# 3. **Correlation Analysis**: Heatmap + Top-5 feature importance via XGBoost

# %%
import sys
import os
import re
import zlib
import math
import hashlib
import warnings
from pathlib import Path
from collections import Counter

# Windows console fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script mode

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
tqdm.pandas()

# -- Paths ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Task_A"
IMG_DIR = PROJECT_ROOT / "img" / "phase2"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -- Constants ------------------------------------------------------------------
CODE_COL = "code"
LABEL_COL = "label"
LANG_COL = "language"
GEN_COL = "generator"
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42

# -- Aesthetics -----------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.05, rc={
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.titleweight": "bold",
    "font.family": "sans-serif",
})
PALETTE = {"Human": "#27ae60", "AI": "#e74c3c"}
LABEL_MAP = {0: "Human", 1: "AI"}


def save_fig(fig, name: str) -> None:
    """Save figure to img/phase2/."""
    path = IMG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"  [CHART] Saved: {path.relative_to(PROJECT_ROOT)}")


# %% [markdown]
# ---
# ## 1. Load Data

# %%
print("=" * 70)
print("  PHASE 2: FORENSIC EDA + FEATURE ENGINEERING")
print("=" * 70)

df_train = pd.read_parquet(RAW_DIR / "train.parquet")
df_val = pd.read_parquet(RAW_DIR / "validation.parquet")

print(f"\n  Train : {df_train.shape[0]:>10,} rows x {df_train.shape[1]} cols")
print(f"  Val   : {df_val.shape[0]:>10,} rows x {df_val.shape[1]} cols")
print(f"  Cols  : {list(df_train.columns)}")


# ==============================================================================
# ##  TASK 1: INTEGRITY & LEAKAGE CHECK
# ==============================================================================

# %% [markdown]
# ---
# ## Task 1: Integrity & Leakage Check
# ### 1.1 Exact Duplicates (within sets)

# %%
print("\n" + "=" * 70)
print("  TASK 1 > Integrity & Leakage Check")
print("=" * 70)

# -- 1.1 Exact Duplicates --------------------------------------------------
print("\n  1.1  Exact Duplicate Detection")
print("  " + "-" * 50)

# Hash each code snippet for fast O(n) comparison
df_train["_hash"] = df_train[CODE_COL].apply(
    lambda x: hashlib.md5(x.encode("utf-8", errors="replace")).hexdigest()
)
df_val["_hash"] = df_val[CODE_COL].apply(
    lambda x: hashlib.md5(x.encode("utf-8", errors="replace")).hexdigest()
)

train_dup = df_train.duplicated(subset="_hash", keep=False).sum()
val_dup = df_val.duplicated(subset="_hash", keep=False).sum()
train_dup_unique = df_train.duplicated(subset="_hash", keep="first").sum()
val_dup_unique = df_val.duplicated(subset="_hash", keep="first").sum()

print(f"  Train duplicated rows     : {train_dup:>8,}  ({train_dup/len(df_train)*100:.2f}%)")
print(f"    |--- Removable duplicates : {train_dup_unique:>8,}")
print(f"  Val duplicated rows       : {val_dup:>8,}  ({val_dup/len(df_val)*100:.2f}%)")
print(f"    |--- Removable duplicates : {val_dup_unique:>8,}")

# -- Check for label inconsistency among duplicates --
dup_hashes = df_train[df_train.duplicated(subset="_hash", keep=False)]["_hash"].unique()
if len(dup_hashes) > 0:
    label_conflict = 0
    for h in dup_hashes[:5000]:  # Check sample for speed
        labels = df_train.loc[df_train["_hash"] == h, LABEL_COL].unique()
        if len(labels) > 1:
            label_conflict += 1
    print(f"  [!]  Label conflicts in dup groups (sample 5k): {label_conflict}")

# %% [markdown]
# ### 1.2 Cross-Set Leakage (Train n Validation)

# %%
print("\n  1.2  Cross-Set Leakage (Train n Validation)")
print("  " + "-" * 50)

train_hashes = set(df_train["_hash"])
val_hashes = set(df_val["_hash"])
overlap = train_hashes & val_hashes

print(f"  Unique hashes in Train : {len(train_hashes):>10,}")
print(f"  Unique hashes in Val   : {len(val_hashes):>10,}")
print(f"  [!]  Cross-set overlap  : {len(overlap):>10,} snippets")

if len(overlap) > 0:
    overlap_train = df_train[df_train["_hash"].isin(overlap)]
    overlap_val = df_val[df_val["_hash"].isin(overlap)]
    print(f"  Train rows in overlap  : {len(overlap_train):>10,}")
    print(f"  Val rows in overlap    : {len(overlap_val):>10,}")

    # Check if leaked samples have same labels
    merged = overlap_train.groupby("_hash")[LABEL_COL].first().reset_index()
    merged.columns = ["_hash", "train_label"]
    val_labels = overlap_val.groupby("_hash")[LABEL_COL].first().reset_index()
    val_labels.columns = ["_hash", "val_label"]
    check = merged.merge(val_labels, on="_hash")
    label_match = (check["train_label"] == check["val_label"]).sum()
    print(f"  Same label in both sets: {label_match}/{len(check)} ({label_match/max(len(check),1)*100:.1f}%)")
else:
    print("  [OK] No leakage detected -- datasets are clean.")

# %% [markdown]
# ### 1.3 Class Imbalance Analysis

# %%
print("\n  1.3  Class Imbalance Analysis")
print("  " + "-" * 50)

# -- Language x Label breakdown --
print("\n  [INFO] Language x Label Distribution (Train):")
lang_label = pd.crosstab(df_train[LANG_COL], df_train[LABEL_COL].map(LABEL_MAP),
                          margins=True, margins_name="Total")
lang_label["AI_ratio"] = (lang_label.get("AI", 0) /
                          lang_label["Total"] * 100).round(1)
print(lang_label.to_string())

# -- Generator distribution --
print("\n  [INFO] Generator Distribution (Train):")
gen_counts = df_train[GEN_COL].value_counts()
gen_stats = pd.DataFrame({
    "count": gen_counts,
    "pct": (gen_counts / len(df_train) * 100).round(2),
})
print(gen_stats.to_string())

# -- Visualization: Language imbalance --
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Language x label
ct = pd.crosstab(df_train[LANG_COL], df_train[LABEL_COL].map(LABEL_MAP))
ct.plot(kind="bar", ax=axes[0], color=[PALETTE["Human"], PALETTE["AI"]],
        edgecolor="white", width=0.7)
axes[0].set_title("Language x Label Distribution")
axes[0].set_xlabel("")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(title="Label")
for c in axes[0].containers:
    axes[0].bar_label(c, fmt="{:,.0f}", fontsize=8)

# Generator family (top-15)
top_gen = gen_counts.head(15)
colors = ["#27ae60" if g == "human" else "#3498db" for g in top_gen.index]
axes[1].barh(range(len(top_gen)), top_gen.values, color=colors)
axes[1].set_yticks(range(len(top_gen)))
axes[1].set_yticklabels([g.split("/")[-1][:25] for g in top_gen.index], fontsize=8)
axes[1].set_title("Top 15 Generators")
axes[1].set_xlabel("Count")
axes[1].invert_yaxis()
for i, v in enumerate(top_gen.values):
    axes[1].text(v + 500, i, f"{v:,}", va="center", fontsize=7)

plt.tight_layout()
save_fig(fig, "01_integrity_distributions")
plt.close()


# ==============================================================================
# ##  TASK 2: FORENSIC EDA -- Feature Extraction
# ==============================================================================

# %% [markdown]
# ---
# ## Task 2: Forensic EDA -- 3 Feature Groups on 20k Sample
# ### 2.0 Stratified Sampling

# %%
print("\n" + "=" * 70)
print("  TASK 2 > Forensic EDA -- Feature Extraction (20k sample)")
print("=" * 70)

# Stratified sample: maintain label + language proportion
df_sample = df_train.groupby([LABEL_COL, LANG_COL], group_keys=False).apply(
    lambda x: x.sample(
        n=min(len(x), max(1, int(SAMPLE_SIZE * len(x) / len(df_train)))),
        random_state=RANDOM_STATE,
    )
).reset_index(drop=True)

print(f"\n  Sampled {len(df_sample):,} rows (stratified by label x language)")
print(f"  Label dist: {df_sample[LABEL_COL].value_counts().to_dict()}")
print(f"  Lang  dist: {df_sample[LANG_COL].value_counts().to_dict()}")

df_sample["label_name"] = df_sample[LABEL_COL].map(LABEL_MAP)


# ------------------------------------------------------------------------------
# 2.1  GROUP A: STYLOMETRIC FEATURES
# ------------------------------------------------------------------------------

# %% [markdown]
# ### 2.1 Stylometric Features
# - `naming_consistency` (snake_case vs camelCase ratio)
# - `trailing_ws_ratio`
# - `avg_line_length`, `line_length_variance`
# - `indent_consistency` (spaces vs tabs)
# - `comment_to_code_ratio`
# - `avg_identifier_length`

# %%
print("\n  2.1  GROUP A: Stylometric Features")
print("  " + "-" * 50)

# -- Regex patterns --
SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
CAMEL_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b")
PASCAL_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
IDENTIFIER_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
SINGLE_COMMENT_RE = re.compile(r"(//.*|#(?!!).*)\s*$", re.MULTILINE)
BLOCK_COMMENT_RE = re.compile(r'/\*[\s\S]*?\*/|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
KEYWORD_SET = frozenset([
    "if", "else", "elif", "for", "while", "return", "def", "class", "import",
    "from", "try", "except", "catch", "finally", "with", "as", "yield",
    "break", "continue", "pass", "raise", "lambda", "async", "await",
    "public", "private", "protected", "static", "void", "int", "float",
    "double", "string", "bool", "boolean", "new", "throw", "throws",
    "switch", "case", "default", "struct", "enum", "interface", "extends",
    "implements", "package", "namespace", "using", "include", "const",
    "var", "let", "auto", "virtual", "override", "abstract", "final",
])


def extract_stylometric(code: str) -> dict:
    """Extract stylometric features from a code snippet."""
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    n_lines = max(len(non_empty), 1)

    # -- Naming conventions --
    snakes = len(SNAKE_RE.findall(code))
    camels = len(CAMEL_RE.findall(code))
    pascals = len(PASCAL_RE.findall(code))
    total_named = max(snakes + camels + pascals, 1)
    naming_consistency = max(snakes, camels, pascals) / total_named

    # -- Trailing whitespace --
    trailing_ws = sum(1 for l in lines if l != l.rstrip())
    trailing_ws_ratio = trailing_ws / max(len(lines), 1)

    # -- Line lengths --
    lengths = [len(l) for l in non_empty]
    avg_line_len = np.mean(lengths) if lengths else 0
    line_len_var = np.var(lengths) if lengths else 0

    # -- Indentation --
    indent_lines = [l for l in non_empty if l[0] in (" ", "\t")]
    tab_lines = sum(1 for l in indent_lines if l[0] == "\t")
    space_lines = len(indent_lines) - tab_lines
    indent_total = max(len(indent_lines), 1)
    indent_consistency = max(tab_lines, space_lines) / indent_total

    # -- Comments --
    single_comments = len(SINGLE_COMMENT_RE.findall(code))
    block_comments = len(BLOCK_COMMENT_RE.findall(code))
    total_comments = single_comments + block_comments
    comment_to_code_ratio = total_comments / n_lines

    # -- Identifiers --
    identifiers = IDENTIFIER_RE.findall(code)
    non_kw = [i for i in identifiers if i.lower() not in KEYWORD_SET]
    avg_id_len = np.mean([len(i) for i in non_kw]) if non_kw else 0
    short_id_ratio = sum(1 for i in non_kw if len(i) <= 2) / max(len(non_kw), 1)
    long_id_ratio = sum(1 for i in non_kw if len(i) >= 10) / max(len(non_kw), 1)

    # -- Keyword density --
    all_tokens = identifiers
    kw_count = sum(1 for t in all_tokens if t.lower() in KEYWORD_SET)
    keyword_density = kw_count / max(len(all_tokens), 1)

    return {
        "naming_consistency": naming_consistency,
        "snake_ratio": snakes / total_named,
        "camel_ratio": camels / total_named,
        "trailing_ws_ratio": trailing_ws_ratio,
        "avg_line_length": avg_line_len,
        "line_length_variance": line_len_var,
        "indent_consistency": indent_consistency,
        "comment_to_code_ratio": comment_to_code_ratio,
        "avg_identifier_length": avg_id_len,
        "short_id_ratio": short_id_ratio,
        "long_id_ratio": long_id_ratio,
        "keyword_density": keyword_density,
    }


print("  Extracting stylometric features...")
stylo_df = df_sample[CODE_COL].progress_apply(extract_stylometric).apply(pd.Series)
df_sample = pd.concat([df_sample, stylo_df], axis=1)
print(f"  [OK] {len(stylo_df.columns)} stylometric features extracted")

# -- Violin plots: naming_consistency & trailing_ws_ratio --
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.violinplot(data=df_sample, x="label_name", y="naming_consistency",
               palette=PALETTE, inner="quartile", ax=axes[0], cut=0)
axes[0].set_title("Naming Consistency\n(1.0 = fully consistent style)")
axes[0].set_xlabel("")

sns.violinplot(data=df_sample, x="label_name", y="trailing_ws_ratio",
               palette=PALETTE, inner="quartile", ax=axes[1], cut=0)
axes[1].set_title("Trailing Whitespace Ratio\n(Higher = more trailing spaces)")
axes[1].set_xlabel("")

plt.suptitle("Stylometric: Human vs AI", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "02_stylometric_violin")
plt.close()

# -- Additional boxplots for identifier & comment features --
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.boxplot(data=df_sample, x="label_name", y="avg_identifier_length",
            palette=PALETTE, ax=axes[0], flierprops={"alpha": 0.2})
axes[0].set_title("Avg Identifier Length")
axes[0].set_xlabel("")

sns.boxplot(data=df_sample, x="label_name", y="comment_to_code_ratio",
            palette=PALETTE, ax=axes[1], flierprops={"alpha": 0.2})
axes[1].set_title("Comment-to-Code Ratio")
axes[1].set_xlabel("")

sns.boxplot(data=df_sample, x="label_name", y="keyword_density",
            palette=PALETTE, ax=axes[2], flierprops={"alpha": 0.2})
axes[2].set_title("Keyword Density")
axes[2].set_xlabel("")

plt.suptitle("Stylometric Features: Boxplots", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "03_stylometric_boxplots")
plt.close()

# Print mean comparison
print("\n  [INFO] Stylometric Mean by Label:")
stylo_cols = list(stylo_df.columns)
print(df_sample.groupby("label_name")[stylo_cols].mean().T.round(4).to_string())


# ------------------------------------------------------------------------------
# 2.2  GROUP B: STATISTICAL FEATURES (Information Theory)
# ------------------------------------------------------------------------------

# %% [markdown]
# ### 2.2 Statistical Features
# - `shannon_entropy`: character-level information entropy
# - `zlib_compression_ratio`: compressibility (AI code is more compressible)
# - `token_entropy`: word-level entropy
# - `burstiness`: standard deviation of inter-token distances

# %%
print("\n  2.2  GROUP B: Statistical Features (Information Theory)")
print("  " + "-" * 50)


def shannon_entropy(text: str) -> float:
    """Compute character-level Shannon entropy."""
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def zlib_compression_ratio(text: str) -> float:
    """Ratio of compressed size to original. Lower = more compressible."""
    if not text:
        return 0.0
    encoded = text.encode("utf-8", errors="replace")
    compressed = zlib.compress(encoded, level=6)
    return len(compressed) / max(len(encoded), 1)


def token_entropy(text: str) -> float:
    """Word-level Shannon entropy."""
    tokens = re.findall(r"\b\w+\b", text)
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    n = len(tokens)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def burstiness(text: str) -> float:
    """Measure burstiness: std/mean of inter-token distances.
    High burstiness -> certain patterns cluster together."""
    tokens = re.findall(r"\b\w+\b", text)
    if len(tokens) < 3:
        return 0.0
    # Track unique token positions
    positions: dict[str, list[int]] = {}
    for i, t in enumerate(tokens):
        positions.setdefault(t, []).append(i)
    # Compute inter-arrival times for repeating tokens
    gaps = []
    for tok, pos_list in positions.items():
        if len(pos_list) >= 2:
            gaps.extend(np.diff(pos_list))
    if not gaps:
        return 0.0
    mu = np.mean(gaps)
    sigma = np.std(gaps)
    return (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0.0


def extract_statistical(code: str) -> dict:
    """Extract information-theoretic features."""
    return {
        "shannon_entropy": shannon_entropy(code),
        "zlib_compression_ratio": zlib_compression_ratio(code),
        "token_entropy": token_entropy(code),
        "burstiness": burstiness(code),
    }


print("  Extracting statistical features...")
stat_df = df_sample[CODE_COL].progress_apply(extract_statistical).apply(pd.Series)
df_sample = pd.concat([df_sample, stat_df], axis=1)
print(f"  [OK] {len(stat_df.columns)} statistical features extracted")

# -- Hypothesis Test: "AI code is more compressible than Human code" --
print("\n  [TEST] Hypothesis Test: AI code is more compressible")
print("  " + "-" * 50)

human_comp = df_sample.loc[df_sample[LABEL_COL] == 0, "zlib_compression_ratio"]
ai_comp = df_sample.loc[df_sample[LABEL_COL] == 1, "zlib_compression_ratio"]

print(f"  Human zlib_ratio: mean={human_comp.mean():.4f}, std={human_comp.std():.4f}")
print(f"  AI    zlib_ratio: mean={ai_comp.mean():.4f}, std={ai_comp.std():.4f}")

# Two-sample t-test (one-sided: AI < Human)
t_stat, p_value = stats.ttest_ind(ai_comp, human_comp, alternative="less")
print(f"  t-statistic = {t_stat:.4f}")
print(f"  p-value     = {p_value:.2e}")
if p_value < 0.05:
    print("  [OK] REJECT H0: AI code IS significantly more compressible (p < 0.05)")
else:
    print("  [X] FAIL to reject H0: No significant difference at alpha=0.05")

# -- Mann-Whitney U test (non-parametric alternative) --
u_stat, p_mw = stats.mannwhitneyu(ai_comp, human_comp, alternative="less")
print(f"\n  Mann-Whitney U = {u_stat:.0f}")
print(f"  p-value (MW)   = {p_mw:.2e}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((human_comp.std()**2 + ai_comp.std()**2) / 2)
cohens_d = (human_comp.mean() - ai_comp.mean()) / pooled_std
print(f"  Cohen's d       = {cohens_d:.4f} ({'small' if abs(cohens_d)<0.2 else 'medium' if abs(cohens_d)<0.8 else 'large'})")

# -- Shannon entropy test --
print("\n  [TEST] Shannon Entropy: Human vs AI")
human_ent = df_sample.loc[df_sample[LABEL_COL] == 0, "shannon_entropy"]
ai_ent = df_sample.loc[df_sample[LABEL_COL] == 1, "shannon_entropy"]
print(f"  Human entropy: mean={human_ent.mean():.4f}")
print(f"  AI    entropy: mean={ai_ent.mean():.4f}")
t_ent, p_ent = stats.ttest_ind(human_ent, ai_ent)
print(f"  t={t_ent:.4f}, p={p_ent:.2e}")

# -- Visualization --
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

sns.histplot(data=df_sample, x="zlib_compression_ratio", hue="label_name",
             palette=PALETTE, kde=True, stat="density", common_norm=False,
             alpha=0.4, ax=axes[0], bins=50)
axes[0].set_title("Zlib Compression Ratio\n(Lower = more compressible)")
axes[0].axvline(human_comp.mean(), color=PALETTE["Human"], ls="--", lw=1.5, label=f"Human mu={human_comp.mean():.3f}")
axes[0].axvline(ai_comp.mean(), color=PALETTE["AI"], ls="--", lw=1.5, label=f"AI mu={ai_comp.mean():.3f}")
axes[0].legend(fontsize=8)

sns.histplot(data=df_sample, x="shannon_entropy", hue="label_name",
             palette=PALETTE, kde=True, stat="density", common_norm=False,
             alpha=0.4, ax=axes[1], bins=50)
axes[1].set_title("Shannon Entropy\n(Higher = more diverse characters)")

sns.violinplot(data=df_sample, x="label_name", y="token_entropy",
               palette=PALETTE, inner="quartile", ax=axes[2], cut=0)
axes[2].set_title("Token Entropy\n(Word-level diversity)")
axes[2].set_xlabel("")

plt.suptitle("Statistical Features: Information Theory", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "04_statistical_distributions")
plt.close()


# ------------------------------------------------------------------------------
# 2.3  GROUP C: STRUCTURAL FEATURES (AST-based)
# ------------------------------------------------------------------------------

# %% [markdown]
# ### 2.3 Structural Features (AST via Tree-sitter)
# - `max_ast_depth`: deepest nesting in AST
# - `avg_ast_depth`: average depth across all nodes
# - `ast_node_count`: total nodes
# - `branch_ratio`: ratio of branching nodes (if/for/while) to total
# - `cyclomatic_complexity_approx`: approximation via branch counting

# %%
print("\n  2.3  GROUP C: Structural Features (AST via Tree-sitter)")
print("  " + "-" * 50)

# -- Tree-sitter setup --
try:
    import tree_sitter_python as ts_python
    import tree_sitter_java as ts_java
    import tree_sitter_cpp as ts_cpp
    import tree_sitter as ts

    PARSERS = {}

    # Python
    py_lang = ts.Language(ts_python.language())
    py_parser = ts.Parser(py_lang)
    PARSERS["Python"] = py_parser

    # Java
    java_lang = ts.Language(ts_java.language())
    java_parser = ts.Parser(java_lang)
    PARSERS["Java"] = java_parser

    # C++
    cpp_lang = ts.Language(ts_cpp.language())
    cpp_parser = ts.Parser(cpp_lang)
    PARSERS["C++"] = cpp_parser

    TREE_SITTER_OK = True
    print("  [OK] Tree-sitter parsers loaded: Python, Java, C++")
except Exception as e:
    TREE_SITTER_OK = False
    print(f"  [!]  Tree-sitter unavailable: {e}")
    print("  Falling back to regex-based structural analysis.")

# -- Branch node types per language --
BRANCH_TYPES = {
    "if_statement", "elif_clause", "else_clause",
    "for_statement", "for_in_clause",
    "while_statement", "do_statement",
    "try_statement", "except_clause", "catch_clause",
    "switch_statement", "case_statement",
    "conditional_expression", "ternary_expression",
    "match_statement", "case_clause",
}


def ast_features(code: str, language: str) -> dict:
    """Extract AST-based structural features using tree-sitter."""
    result = {
        "max_ast_depth": 0,
        "avg_ast_depth": 0.0,
        "ast_node_count": 0,
        "branch_ratio": 0.0,
        "cyclomatic_approx": 1,
    }

    if not TREE_SITTER_OK or language not in PARSERS:
        return _regex_structural(code)

    parser = PARSERS[language]
    try:
        tree = parser.parse(code.encode("utf-8", errors="replace"))
    except Exception:
        return _regex_structural(code)

    root = tree.root_node
    depths = []
    branch_count = 0
    total_nodes = 0

    # BFS traversal
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        total_nodes += 1
        depths.append(depth)
        if node.type in BRANCH_TYPES:
            branch_count += 1
        for child in node.children:
            stack.append((child, depth + 1))

    if depths:
        result["max_ast_depth"] = max(depths)
        result["avg_ast_depth"] = np.mean(depths)
    result["ast_node_count"] = total_nodes
    result["branch_ratio"] = branch_count / max(total_nodes, 1)
    result["cyclomatic_approx"] = branch_count + 1

    return result


def _regex_structural(code: str) -> dict:
    """Fallback: regex-based structural complexity."""
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]
    n = max(len(non_empty), 1)

    # Indent depth as proxy for nesting
    indent_depths = []
    for l in non_empty:
        stripped = l.lstrip()
        indent = len(l) - len(stripped)
        indent_depths.append(indent // 4)  # Assume 4-space indent

    branches = len(re.findall(
        r"\b(if|elif|else|for|while|try|except|catch|case|switch)\b", code
    ))

    return {
        "max_ast_depth": max(indent_depths) if indent_depths else 0,
        "avg_ast_depth": np.mean(indent_depths) if indent_depths else 0.0,
        "ast_node_count": len(non_empty),
        "branch_ratio": branches / n,
        "cyclomatic_approx": branches + 1,
    }


print("  Extracting AST features...")
ast_df = df_sample.apply(
    lambda row: ast_features(row[CODE_COL], row[LANG_COL]), axis=1
).progress_apply(pd.Series)
df_sample = pd.concat([df_sample, ast_df], axis=1)
print(f"  [OK] {len(ast_df.columns)} structural features extracted")

# -- Test: Does AI write "shallower" code? --
print("\n  [TEST] Hypothesis: AI writes shallower (flatter) code")
print("  " + "-" * 50)

human_depth = df_sample.loc[df_sample[LABEL_COL] == 0, "max_ast_depth"]
ai_depth = df_sample.loc[df_sample[LABEL_COL] == 1, "max_ast_depth"]

print(f"  Human max_ast_depth: mean={human_depth.mean():.2f}, median={human_depth.median():.0f}")
print(f"  AI    max_ast_depth: mean={ai_depth.mean():.2f}, median={ai_depth.median():.0f}")

t_depth, p_depth = stats.ttest_ind(ai_depth, human_depth, alternative="less")
print(f"  t={t_depth:.4f}, p={p_depth:.2e}")

d_depth = (human_depth.mean() - ai_depth.mean()) / np.sqrt((human_depth.std()**2 + ai_depth.std()**2) / 2)
print(f"  Cohen's d = {d_depth:.4f}")

if p_depth < 0.05:
    print("  [OK] AI code IS significantly shallower than Human code")
else:
    print("  [X] No significant difference in AST depth")

# -- Visualization --
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

sns.violinplot(data=df_sample, x="label_name", y="max_ast_depth",
               palette=PALETTE, inner="quartile", ax=axes[0], cut=0)
axes[0].set_title("Max AST Depth\n(Deeper = more nested)")
axes[0].set_xlabel("")

sns.boxplot(data=df_sample, x="label_name", y="cyclomatic_approx",
            palette=PALETTE, ax=axes[1], flierprops={"alpha": 0.15})
axes[1].set_title("Cyclomatic Complexity (Approx.)\n(More = more branching)")
axes[1].set_xlabel("")

sns.histplot(data=df_sample, x="branch_ratio", hue="label_name",
             palette=PALETTE, kde=True, stat="density", common_norm=False,
             alpha=0.4, ax=axes[2], bins=40)
axes[2].set_title("Branch Ratio\n(Branching nodes / total nodes)")

plt.suptitle("Structural Features: AST Analysis", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "05_structural_ast")
plt.close()

# -- AST depth distribution per language --
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for i, lang in enumerate(["Python", "Java", "C++"]):
    lang_data = df_sample[df_sample[LANG_COL] == lang]
    if len(lang_data) > 0:
        sns.violinplot(data=lang_data, x="label_name", y="max_ast_depth",
                       palette=PALETTE, inner="quartile", ax=axes[i], cut=0)
    axes[i].set_title(f"{lang}: Max AST Depth")
    axes[i].set_xlabel("")
plt.suptitle("AST Depth by Language", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "06_ast_depth_per_lang")
plt.close()


# ==============================================================================
# ##  TASK 3: CORRELATION ANALYSIS + FEATURE IMPORTANCE
# ==============================================================================

# %% [markdown]
# ---
# ## Task 3: Correlation Analysis & Feature Importance

# %%
print("\n" + "=" * 70)
print("  TASK 3 > Correlation Analysis & Feature Importance")
print("=" * 70)

# -- Collect all engineered features --
ALL_FEATURES = stylo_cols + list(stat_df.columns) + list(ast_df.columns)
print(f"\n  Total engineered features: {len(ALL_FEATURES)}")
print(f"  Features: {ALL_FEATURES}")

# -- 3.1 Correlation Heatmap --
print("\n  3.1  Feature Correlation Heatmap")
print("  " + "-" * 50)

corr_cols = ALL_FEATURES + [LABEL_COL]
corr_matrix = df_sample[corr_cols].corr()

# -- Correlation with label --
label_corr = corr_matrix[LABEL_COL].drop(LABEL_COL).sort_values(key=abs, ascending=False)
print("\n  [INFO] Feature Correlation with Label:")
for feat, val in label_corr.items():
    bar = "#" * int(abs(val) * 40)
    sign = "+" if val > 0 else "-"
    print(f"    {feat:<28s}  {sign}{abs(val):.4f}  {bar}")

# -- Heatmap --
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True, fmt=".2f",
    center=0, cmap="RdBu_r",
    vmin=-1, vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    ax=ax,
    annot_kws={"size": 7},
)
ax.set_title("Feature Correlation Heatmap (incl. Label)", fontsize=14, fontweight="bold")
ax.tick_params(labelsize=8)
plt.tight_layout()
save_fig(fig, "07_correlation_heatmap")
plt.close()

# -- Bar chart: correlation with label --
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#e74c3c" if v > 0 else "#3498db" for v in label_corr.values]
ax.barh(range(len(label_corr)), label_corr.values, color=colors, edgecolor="white")
ax.set_yticks(range(len(label_corr)))
ax.set_yticklabels(label_corr.index, fontsize=9)
ax.set_xlabel("Pearson Correlation with Label", fontsize=11)
ax.set_title("Feature-Label Correlation (Sorted by |r|)", fontsize=13, fontweight="bold")
ax.axvline(0, color="black", lw=0.8)
ax.invert_yaxis()
for i, v in enumerate(label_corr.values):
    ax.text(v + (0.005 if v > 0 else -0.005), i,
            f"{v:.3f}", va="center", ha="left" if v > 0 else "right", fontsize=8)
plt.tight_layout()
save_fig(fig, "08_label_correlation_bar")
plt.close()


# -- 3.2 Feature Importance via XGBoost --
print("\n  3.2  Feature Importance (XGBoost)")
print("  " + "-" * 50)

try:
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score

    X = df_sample[ALL_FEATURES].fillna(0).values
    y = df_sample[LABEL_COL].values

    # Quick 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    importances = np.zeros(len(ALL_FEATURES))
    aucs = []
    f1s = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[va_idx], y[va_idx])],
                  verbose=False)

        preds = model.predict_proba(X[va_idx])[:, 1]
        auc = roc_auc_score(y[va_idx], preds)
        f1 = f1_score(y[va_idx], (preds > 0.5).astype(int))
        aucs.append(auc)
        f1s.append(f1)
        importances += model.feature_importances_

    importances /= 5  # Average across folds

    print(f"\n  [CHART] XGBoost 5-Fold CV Results (on {len(ALL_FEATURES)} features):")
    print(f"    AUC-ROC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"    F1-Score: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # -- Top-5 features --
    feat_imp = pd.Series(importances, index=ALL_FEATURES).sort_values(ascending=False)
    print("\n  [TOP] Top-5 Features by XGBoost Importance:")
    for rank, (feat, imp) in enumerate(feat_imp.head(5).items(), 1):
        bar = "#" * int(imp * 100)
        print(f"    {rank}. {feat:<28s}  {imp:.4f}  {bar}")

    # -- Full importance bar chart --
    fig, ax = plt.subplots(figsize=(10, 7))
    colors_imp = ["#e74c3c" if i < 5 else "#95a5a6" for i in range(len(feat_imp))]
    ax.barh(range(len(feat_imp)), feat_imp.values, color=colors_imp, edgecolor="white")
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels(feat_imp.index, fontsize=9)
    ax.set_xlabel("Mean XGBoost Feature Importance (Gain)", fontsize=11)
    ax.set_title("Feature Importance: XGBoost 5-Fold CV\n"
                 f"AUC={np.mean(aucs):.4f} | F1={np.mean(f1s):.4f}",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for i, v in enumerate(feat_imp.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    save_fig(fig, "09_xgboost_importance")
    plt.close()

except ImportError:
    print("  [!]  XGBoost not installed. Skipping feature importance.")


# ==============================================================================
# ##  SAVE PROCESSED DATA
# ==============================================================================

# %%
print("\n" + "=" * 70)
print("  SAVING PROCESSED DATA")
print("=" * 70)

# Save feature-enriched sample
out_path = PROCESSED_DIR / "sample_20k_features.parquet"
save_cols = [CODE_COL, GEN_COL, LABEL_COL, LANG_COL, "label_name"] + ALL_FEATURES
df_sample[save_cols].to_parquet(out_path, index=False)
print(f"  [SAVE] Saved: {out_path.relative_to(PROJECT_ROOT)}  ({len(df_sample):,} rows x {len(save_cols)} cols)")

# Save feature list
feat_path = PROCESSED_DIR / "feature_list.txt"
with open(feat_path, "w") as f:
    for feat in ALL_FEATURES:
        f.write(f"{feat}\n")
print(f"  [SAVE] Feature list: {feat_path.relative_to(PROJECT_ROOT)}")

# Clean up temp columns
df_train.drop(columns=["_hash"], inplace=True, errors="ignore")
df_val.drop(columns=["_hash"], inplace=True, errors="ignore")

print("\n" + "=" * 70)
print("  [OK] PHASE 2 & 3 COMPLETE")
print("=" * 70)
print(f"\n  Outputs:")
print(f"    [CHART] Charts  -> img/phase2/")
print(f"    [PKG] Data    -> data/processed/sample_20k_features.parquet")
print(f"    [INFO] Features-> data/processed/feature_list.txt")
print(f"\n  Next: Use features in XGBoost/LightGBM model training (Phase 4)")
