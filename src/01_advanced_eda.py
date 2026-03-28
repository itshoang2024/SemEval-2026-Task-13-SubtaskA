# %% [markdown]
# # 🔬 Advanced EDA – SemEval-2026 Task 13 SubtaskA
# ## AI-Generated Code Detection: Feature Engineering-Driven Analysis
#
# **Goal**: Binary classification – Human (`0`) vs AI-generated (`1`) code.
#
# This notebook analyzes the dataset across 4 dimensions:
# 1. **Distribution & Imbalance** – class balance, language × label
# 2. **Verbosity & Token-Level** – code length, line count, avg line length
# 3. **Stylometric & Lexical** – naming conventions, comments, whitespace (CRITICAL)
# 4. **Structural Complexity** – indentation depth, nested loops, operators
#
# Each section explains *WHY* the feature separates Human vs AI, and *HOW* to
# convert it into a numerical feature vector for ML.

# %%
import re
import warnings
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
IMG_DIR = Path(__file__).resolve().parent.parent / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

CODE_COL = "code"
LABEL_COL = "label"
LANG_COL = "language"
GEN_COL = "generator"

# Aesthetic defaults
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE_BINARY = {"Human": "#2ecc71", "AI-Generated": "#e74c3c"}
LABEL_NAMES = {0: "Human", 1: "AI-Generated"}

def save_fig(fig, name: str):
    fig.savefig(IMG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: img/{name}.png")


# %% [markdown]
# ---
# ## 0. Load Data

# %%
task_dir = RAW_DATA_DIR / "Task_A" if (RAW_DATA_DIR / "Task_A").exists() else RAW_DATA_DIR

train = pd.read_parquet(task_dir / "train.parquet")
val = pd.read_parquet(task_dir / "validation.parquet")
test_sample = pd.read_parquet(task_dir / "test_sample.parquet")

print(f"Train:       {train.shape}")
print(f"Validation:  {val.shape}")
print(f"Test Sample: {test_sample.shape}")
print(f"\nColumns: {train.columns.tolist()}")
print(f"\nDtypes:\n{train.dtypes}")
print(f"\nFirst 3 rows:\n{train.head(3)}")
print(f"\nMissing values (train):\n{train.isnull().sum()}")

# We'll do most of the EDA on train, then verify patterns hold on validation.
df = train.copy()
df["label_name"] = df[LABEL_COL].map(LABEL_NAMES)


# %% [markdown]
# ---
# ## 1. 📊 Basic Distribution & Imbalance
#
# **WHY**: Severe class imbalance affects model training (need oversampling, class
# weights, or focal loss). Knowing which languages/generators are over/under-represented
# tells us where the model might struggle.
#
# **FEATURE**: `is_minority_lang` (bool), `generator_family` (categorical).

# %%
print("=" * 60)
print(" 1.1  Class Balance")
print("=" * 60)

vc = df[LABEL_COL].value_counts().sort_index()
for label, count in vc.items():
    pct = count / len(df) * 100
    print(f"  {label} ({LABEL_NAMES[label]}): {count:>8,}  ({pct:.1f}%)")

imbalance_ratio = vc.min() / vc.max()
print(f"\n  Imbalance ratio (minority/majority): {imbalance_ratio:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
bars = axes[0].bar(
    [f"{LABEL_NAMES[k]}\n(label={k})" for k in vc.index],
    vc.values,
    color=[PALETTE_BINARY[k] for k in vc.index],
    edgecolor="white", linewidth=1.5,
)
for bar, val in zip(bars, vc.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                 f"{val:,}", ha="center", fontweight="bold", fontsize=12)
axes[0].set_title("Class Distribution", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Count")

# Pie chart
axes[1].pie(vc.values, labels=[f"{LABEL_NAMES[k]}" for k in vc.index],
            colors=[PALETTE_BINARY[k] for k in vc.index],
            autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12})
axes[1].set_title("Class Proportions", fontsize=14, fontweight="bold")

plt.tight_layout()
save_fig(fig, "01_class_distribution")
plt.show()

# %%
print("=" * 60)
print(" 1.2  Programming Language Distribution")
print("=" * 60)

lang_counts = df[LANG_COL].value_counts()
print(f"\n  Unique languages: {df[LANG_COL].nunique()}")
print(f"\n  Top 15:\n{lang_counts.head(15).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Overall language distribution
top_langs = lang_counts.head(15)
sns.barplot(x=top_langs.values, y=top_langs.index, ax=axes[0], palette="viridis")
axes[0].set_title("Top 15 Programming Languages", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Count")

# Language × Label stacked
lang_label = df.groupby([LANG_COL, LABEL_COL]).size().unstack(fill_value=0)
lang_label = lang_label.loc[top_langs.index]
lang_label.plot(kind="barh", stacked=True, ax=axes[1],
                color=[PALETTE_BINARY[0], PALETTE_BINARY[1]])
axes[1].set_title("Language × Label Distribution", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Count")
axes[1].legend(["Human", "AI"], loc="lower right")

plt.tight_layout()
save_fig(fig, "02_language_distribution")
plt.show()

# %%
print("=" * 60)
print(" 1.3  Generator (Model) Distribution")
print("=" * 60)

gen_counts = df[GEN_COL].value_counts()
print(f"\n  Unique generators: {df[GEN_COL].nunique()}")
print(f"\n  All generators:\n{gen_counts.to_string()}")

fig, ax = plt.subplots(figsize=(12, max(6, len(gen_counts) * 0.4)))
colors = ["#2ecc71" if g == "human" else "#e74c3c" for g in gen_counts.index]
sns.barplot(x=gen_counts.values, y=gen_counts.index, ax=ax, palette=colors)
for i, (v, g) in enumerate(zip(gen_counts.values, gen_counts.index)):
    ax.text(v + gen_counts.max() * 0.01, i, f"{v:,}", va="center", fontsize=10)
ax.set_title("Generator Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Count")
plt.tight_layout()
save_fig(fig, "03_generator_distribution")
plt.show()


# %% [markdown]
# ---
# ## 2. 📏 Verbosity & Token-Level Analysis
#
# **WHY**: AI models tend to produce code of more uniform length (trained on
# similar prompts → similar output sizes). Humans show higher variance – from
# one-liners to 500-line scripts.
#
# **FEATURES**: `char_count`, `line_count`, `token_count`, `avg_line_length`,
# `max_line_length`, `empty_line_ratio`.

# %%
print("=" * 60)
print(" 2.  Computing Verbosity Features")
print("=" * 60)

df["char_count"] = df[CODE_COL].str.len()
df["line_count"] = df[CODE_COL].str.count("\n") + 1
df["token_count"] = df[CODE_COL].str.split().str.len()
df["avg_line_length"] = df[CODE_COL].apply(
    lambda c: np.mean([len(l) for l in c.splitlines()]) if c.strip() else 0
)
df["max_line_length"] = df[CODE_COL].apply(
    lambda c: max((len(l) for l in c.splitlines()), default=0)
)
df["empty_line_ratio"] = df[CODE_COL].apply(
    lambda c: sum(1 for l in c.splitlines() if l.strip() == "") / max(len(c.splitlines()), 1)
)

verbosity_cols = ["char_count", "line_count", "token_count",
                  "avg_line_length", "max_line_length", "empty_line_ratio"]

summary = df.groupby("label_name")[verbosity_cols].agg(["mean", "median", "std"])
print("\n  Verbosity Summary (Human vs AI):\n")
print(summary.round(2).to_string())

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(verbosity_cols):
    for label, name in LABEL_NAMES.items():
        subset = df[df[LABEL_COL] == label][col]
        # Clip outliers for visualization
        clip_val = subset.quantile(0.99)
        axes[i].hist(subset.clip(upper=clip_val), bins=50, alpha=0.6,
                     label=name, color=PALETTE_BINARY[label], edgecolor="white")
    axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    axes[i].legend()
    axes[i].set_ylabel("Frequency")

plt.suptitle("Verbosity Distributions: Human vs AI", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "04_verbosity_distributions")
plt.show()

# %%
# KDE plots for smoother comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(["char_count", "line_count", "avg_line_length"]):
    for label, name in LABEL_NAMES.items():
        subset = df[df[LABEL_COL] == label][col].clip(upper=df[col].quantile(0.99))
        subset.plot.kde(ax=axes[i], label=name, color=PALETTE_BINARY[label], linewidth=2)
    axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
    axes[i].legend()
    axes[i].set_xlim(left=0)

plt.suptitle("KDE: Human vs AI Code Length", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "05_verbosity_kde")
plt.show()

# %%
# Boxplots by label
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(["char_count", "line_count", "token_count"]):
    clip_val = df[col].quantile(0.95)
    sns.boxplot(data=df[df[col] <= clip_val], x="label_name", y=col, ax=axes[i],
                palette=PALETTE_BINARY, hue="label_name", legend=False)
    axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")

plt.suptitle("Boxplots: Code Length by Class (95th pctl clip)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "06_verbosity_boxplots")
plt.show()


# %% [markdown]
# ---
# ## 3. 🔍 Stylometric & Lexical Features (CRITICAL for Feature Engineering)
#
# These features capture **coding style patterns** that fundamentally differ
# between humans and AI. LLMs are trained to produce consistent, "clean" code,
# while human code is messy, inconsistent, and full of shortcuts.

# %% [markdown]
# ### 3.1 Naming Convention Consistency
#
# **WHY**: AI is perfectly consistent with naming conventions (all snake_case
# or all camelCase). Humans MIX conventions within the same file – e.g.
# `my_var` next to `myVar`. The **ratio of snake_case to camelCase** is a
# strong discriminative feature.
#
# **FEATURE**: `snake_case_ratio`, `camel_case_ratio`, `naming_consistency_score`.

# %%
print("=" * 60)
print(" 3.1  Naming Convention Analysis")
print("=" * 60)

# Regex patterns for identifier extraction and classification
IDENTIFIER_RE = re.compile(r'\b([a-zA-Z_]\w{1,})\b')              # identifiers (len >= 2)
SNAKE_RE = re.compile(r'^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$')         # snake_case
CAMEL_RE = re.compile(r'^[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*$')    # camelCase
PASCAL_RE = re.compile(r'^[A-Z][a-z][a-zA-Z0-9]*$')               # PascalCase
UPPER_RE = re.compile(r'^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$')         # UPPER_CASE constants

# Language keywords to exclude (common across many languages)
KEYWORDS = {
    "if", "else", "elif", "for", "while", "return", "def", "class", "import",
    "from", "in", "not", "and", "or", "is", "None", "True", "False", "try",
    "except", "finally", "with", "as", "lambda", "yield", "pass", "break",
    "continue", "raise", "del", "global", "nonlocal", "assert", "async",
    "await", "print", "self", "this", "var", "let", "const", "function",
    "new", "null", "undefined", "void", "typeof", "instanceof", "static",
    "public", "private", "protected", "int", "float", "double", "string",
    "bool", "boolean", "char", "long", "short", "byte", "unsigned", "signed",
    "struct", "enum", "interface", "extends", "implements", "abstract",
    "virtual", "override", "final", "package", "include", "using",
    "namespace", "template", "typename", "auto", "register", "extern",
    "volatile", "sizeof", "switch", "case", "default", "goto", "do",
    "throw", "throws", "catch", "super", "main", "args", "System", "out",
    "println", "String", "Integer", "List", "Map", "Set", "ArrayList",
}


def analyze_naming(code: str) -> dict:
    """Extract naming convention metrics from a code snippet."""
    identifiers = IDENTIFIER_RE.findall(code)
    # Filter out keywords and single-char
    ids = [i for i in identifiers if i not in KEYWORDS and len(i) > 1]

    if not ids:
        return {"snake_count": 0, "camel_count": 0, "pascal_count": 0,
                "upper_count": 0, "other_count": 0, "total_ids": 0,
                "naming_consistency": 1.0}

    unique_ids = set(ids)
    snake = sum(1 for i in unique_ids if SNAKE_RE.match(i))
    camel = sum(1 for i in unique_ids if CAMEL_RE.match(i))
    pascal = sum(1 for i in unique_ids if PASCAL_RE.match(i))
    upper = sum(1 for i in unique_ids if UPPER_RE.match(i))
    other = len(unique_ids) - snake - camel - pascal - upper

    total = len(unique_ids)
    # Consistency = how dominant the most common convention is (1.0 = perfectly consistent)
    counts = [snake, camel, pascal, upper]
    consistency = max(counts) / sum(counts) if sum(counts) > 0 else 1.0

    return {
        "snake_count": snake, "camel_count": camel, "pascal_count": pascal,
        "upper_count": upper, "other_count": other, "total_ids": total,
        "naming_consistency": round(consistency, 4),
    }

naming_df = df[CODE_COL].apply(analyze_naming).apply(pd.Series)
df = pd.concat([df, naming_df], axis=1)

df["snake_ratio"] = df["snake_count"] / df["total_ids"].clip(lower=1)
df["camel_ratio"] = df["camel_count"] / df["total_ids"].clip(lower=1)

summary = df.groupby("label_name")[["snake_ratio", "camel_ratio",
                                     "naming_consistency", "total_ids"]].mean()
print("\n  Naming Convention Summary:\n")
print(summary.round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(data=df, x="label_name", y="naming_consistency", ax=axes[0],
            palette=PALETTE_BINARY, hue="label_name", legend=False)
axes[0].set_title("Naming Consistency Score", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Consistency (1.0 = perfectly uniform)")

sns.boxplot(data=df, x="label_name", y="snake_ratio", ax=axes[1],
            palette=PALETTE_BINARY, hue="label_name", legend=False)
axes[1].set_title("snake_case Ratio", fontsize=12, fontweight="bold")

sns.boxplot(data=df, x="label_name", y="camel_ratio", ax=axes[2],
            palette=PALETTE_BINARY, hue="label_name", legend=False)
axes[2].set_title("camelCase Ratio", fontsize=12, fontweight="bold")

plt.suptitle("Naming Convention Analysis: Human vs AI", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "07_naming_conventions")
plt.show()


# %% [markdown]
# ### 3.2 Identifier Length Analysis
#
# **WHY**: Humans use ultra-short names (`i`, `j`, `x`, `tmp`, `n`) for quick
# scripts. AI tends to use descriptive names (`input_array`, `result_list`,
# `current_index`). The distribution of identifier lengths is a strong signal.
#
# **FEATURE**: `avg_identifier_length`, `short_id_ratio` (len <= 2),
# `long_id_ratio` (len >= 10).

# %%
print("=" * 60)
print(" 3.2  Identifier Length Analysis")
print("=" * 60)


def identifier_length_stats(code: str) -> dict:
    ids = IDENTIFIER_RE.findall(code)
    ids = [i for i in ids if i not in KEYWORDS]
    if not ids:
        return {"avg_id_len": 0, "short_id_ratio": 0, "long_id_ratio": 0}
    lengths = [len(i) for i in ids]
    return {
        "avg_id_len": np.mean(lengths),
        "short_id_ratio": sum(1 for l in lengths if l <= 2) / len(lengths),
        "long_id_ratio": sum(1 for l in lengths if l >= 10) / len(lengths),
    }

id_len_df = df[CODE_COL].apply(identifier_length_stats).apply(pd.Series)
df = pd.concat([df, id_len_df], axis=1)

summary = df.groupby("label_name")[["avg_id_len", "short_id_ratio", "long_id_ratio"]].mean()
print("\n  Identifier Length Summary:\n")
print(summary.round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, col in enumerate(["avg_id_len", "short_id_ratio", "long_id_ratio"]):
    sns.violinplot(data=df, x="label_name", y=col, ax=axes[i],
                   palette=PALETTE_BINARY, hue="label_name", legend=False, inner="box")
    axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")

plt.suptitle("Identifier Length: Human vs AI", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "08_identifier_lengths")
plt.show()


# %% [markdown]
# ### 3.3 Comment Density & Patterns
#
# **WHY**: AI produces generic, instruction-like comments ("This function takes
# an input and returns..."). Humans leave TODO/FIXME/HACK/BUG markers, personal
# notes, swear words, and sometimes no comments at all.
#
# **FEATURES**: `comment_ratio`, `has_todo`, `has_fixme`, `has_hack`,
# `comment_avg_length`, `inline_comment_ratio`.

# %%
print("=" * 60)
print(" 3.3  Comment Density & Patterns")
print("=" * 60)

# Multi-language comment detection
SINGLE_COMMENT_RE = re.compile(r'(//.*|#(?!!).*)$', re.MULTILINE)      # // or # comments
BLOCK_COMMENT_RE = re.compile(r'/\*[\s\S]*?\*/|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')
HUMAN_MARKERS_RE = re.compile(r'\b(TODO|FIXME|HACK|BUG|XXX|NOTE|WORKAROUND|TEMP|DIRTY)\b',
                               re.IGNORECASE)
AI_COMMENT_PATTERNS = re.compile(
    r'(# This (function|method|class)|// This (function|method|class)|'
    r'# Returns|# Parameters|# Args|# Example|# Input|# Output)',
    re.IGNORECASE
)


def analyze_comments(code: str) -> dict:
    lines = code.splitlines()
    total_lines = max(len(lines), 1)

    # Count comment lines
    single_comments = SINGLE_COMMENT_RE.findall(code)
    block_comments = BLOCK_COMMENT_RE.findall(code)
    comment_lines = len(single_comments) + sum(c.count("\n") + 1 for c in block_comments)

    # Extract all comment text for analysis
    all_comment_text = " ".join(single_comments) + " ".join(block_comments)

    human_markers = len(HUMAN_MARKERS_RE.findall(all_comment_text))
    ai_patterns = len(AI_COMMENT_PATTERNS.findall(all_comment_text))

    # Inline comments (comments at end of code lines, not standalone)
    inline = sum(1 for l in lines
                 if l.strip() and not l.strip().startswith(("#", "//"))
                 and ("#" in l or "//" in l))

    return {
        "comment_line_count": comment_lines,
        "comment_ratio": comment_lines / total_lines,
        "has_human_markers": int(human_markers > 0),
        "human_marker_count": human_markers,
        "ai_pattern_count": ai_patterns,
        "inline_comment_count": inline,
        "inline_comment_ratio": inline / total_lines,
    }

comment_df = df[CODE_COL].apply(analyze_comments).apply(pd.Series)
df = pd.concat([df, comment_df], axis=1)

summary = df.groupby("label_name")[
    ["comment_ratio", "has_human_markers", "human_marker_count",
     "ai_pattern_count", "inline_comment_ratio"]
].mean()
print("\n  Comment Analysis Summary:\n")
print(summary.round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(data=df, x="label_name", y="comment_ratio", ax=axes[0],
            palette=PALETTE_BINARY, hue="label_name", legend=False)
axes[0].set_title("Comment Line Ratio", fontsize=12, fontweight="bold")

# Human markers presence
marker_pct = df.groupby("label_name")["has_human_markers"].mean() * 100
axes[1].bar(marker_pct.index, marker_pct.values,
            color=[PALETTE_BINARY[0], PALETTE_BINARY[1]], edgecolor="white")
for j, v in enumerate(marker_pct.values):
    axes[1].text(j, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
axes[1].set_title("% with TODO/FIXME/HACK Markers", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Percentage")

sns.violinplot(data=df, x="label_name", y="inline_comment_ratio", ax=axes[2],
               palette=PALETTE_BINARY, hue="label_name", legend=False, inner="box")
axes[2].set_title("Inline Comment Ratio", fontsize=12, fontweight="bold")

plt.suptitle("Comment Patterns: Human vs AI", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "09_comment_patterns")
plt.show()


# %% [markdown]
# ### 3.4 Whitespace & Formatting
#
# **WHY**: AI uses perfectly consistent formatting (spaces around operators,
# consistent indentation). Humans are inconsistent – mixing `a=b` with `a = b`,
# tabs with spaces, varying blank line frequencies.
#
# **FEATURES**: `tight_operator_ratio` (a=b style), `spaced_operator_ratio`,
# `formatting_consistency`, `trailing_whitespace_ratio`.

# %%
print("=" * 60)
print(" 3.4  Whitespace & Formatting Analysis")
print("=" * 60)

# Operator spacing patterns
TIGHT_OP_RE = re.compile(r'(?<=[a-zA-Z0-9_])(?:==|!=|<=|>=|[+\-*/=<>])(?=[a-zA-Z0-9_])')
SPACED_OP_RE = re.compile(r'(?<=[a-zA-Z0-9_])\s+(?:==|!=|<=|>=|[+\-*/=<>])\s+(?=[a-zA-Z0-9_])')


def analyze_whitespace(code: str) -> dict:
    lines = code.splitlines()
    if not lines:
        return {"tight_op_count": 0, "spaced_op_count": 0, "op_spacing_consistency": 1.0,
                "trailing_ws_ratio": 0, "tab_indent": 0, "space_indent": 0,
                "mixed_indent": 0}

    # Operator spacing
    tight = len(TIGHT_OP_RE.findall(code))
    spaced = len(SPACED_OP_RE.findall(code))
    total_ops = tight + spaced
    op_consistency = max(tight, spaced) / total_ops if total_ops > 0 else 1.0

    # Trailing whitespace (humans leave this, AI/formatters don't)
    trailing = sum(1 for l in lines if l != l.rstrip())
    trailing_ratio = trailing / len(lines)

    # Indentation consistency (tabs vs spaces)
    indented_lines = [l for l in lines if l and l[0] in (" ", "\t")]
    tab_indent = sum(1 for l in indented_lines if l[0] == "\t")
    space_indent = sum(1 for l in indented_lines if l[0] == " ")
    mixed = int(tab_indent > 0 and space_indent > 0)

    return {
        "tight_op_count": tight,
        "spaced_op_count": spaced,
        "op_spacing_consistency": round(op_consistency, 4),
        "trailing_ws_ratio": round(trailing_ratio, 4),
        "tab_indent": tab_indent,
        "space_indent": space_indent,
        "mixed_indent": mixed,
    }

ws_df = df[CODE_COL].apply(analyze_whitespace).apply(pd.Series)
df = pd.concat([df, ws_df], axis=1)

summary = df.groupby("label_name")[
    ["op_spacing_consistency", "trailing_ws_ratio", "mixed_indent"]
].mean()
print("\n  Whitespace/Formatting Summary:\n")
print(summary.round(4).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(data=df, x="label_name", y="op_spacing_consistency", ax=axes[0],
            palette=PALETTE_BINARY, hue="label_name", legend=False)
axes[0].set_title("Operator Spacing Consistency", fontsize=12, fontweight="bold")

sns.boxplot(data=df, x="label_name", y="trailing_ws_ratio", ax=axes[1],
            palette=PALETTE_BINARY, hue="label_name", legend=False)
axes[1].set_title("Trailing Whitespace Ratio", fontsize=12, fontweight="bold")

mixed_pct = df.groupby("label_name")["mixed_indent"].mean() * 100
axes[2].bar(mixed_pct.index, mixed_pct.values,
            color=[PALETTE_BINARY[0], PALETTE_BINARY[1]], edgecolor="white")
for j, v in enumerate(mixed_pct.values):
    axes[2].text(j, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
axes[2].set_title("% with Mixed Tab/Space Indentation", fontsize=12, fontweight="bold")
axes[2].set_ylabel("Percentage")

plt.suptitle("Whitespace & Formatting: Human vs AI", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "10_whitespace_formatting")
plt.show()


# %% [markdown]
# ---
# ## 4. 🏗️ Structural Complexity
#
# **WHY**: AI generates well-structured code with moderate complexity.
# Human code shows wider variance – from trivially simple to deeply nested
# spaghetti code. Extreme indentation depths and nested loops are more common
# in human code.
#
# **FEATURES**: `max_indent_depth`, `avg_indent_depth`, `nested_loop_count`,
# `complex_logic_count`, `function_count`, `class_count`.

# %%
print("=" * 60)
print(" 4.  Structural Complexity Analysis")
print("=" * 60)

# Pattern detections
LOOP_KW_RE = re.compile(r'\b(for|while|do)\b')
CONDITIONAL_RE = re.compile(r'\b(if|elif|else if|switch|case)\b')
FUNC_DEF_RE = re.compile(r'\b(def |function |func |fn |void |int |float |double |string |public |private |static )\w+\s*\(')
CLASS_DEF_RE = re.compile(r'\b(class |struct |interface )\w+')
COMPLEX_LOGIC_RE = re.compile(r'(\band\b|\bor\b|&&|\|\||[?:])')  # complex boolean/ternary


def analyze_structure(code: str) -> dict:
    lines = code.splitlines()
    if not lines:
        return {"max_indent_depth": 0, "avg_indent_depth": 0,
                "loop_count": 0, "conditional_count": 0,
                "func_count": 0, "class_count": 0,
                "complex_logic_count": 0, "nesting_score": 0}

    # Indentation depth (counting leading spaces/tabs)
    indent_depths = []
    for line in lines:
        stripped = line.lstrip()
        if stripped:  # skip empty lines
            spaces = len(line) - len(stripped)
            # Normalize: 1 tab = 4 spaces
            depth = spaces // 4 if "\t" not in line[:spaces] else line[:spaces].count("\t")
            indent_depths.append(depth)

    max_depth = max(indent_depths) if indent_depths else 0
    avg_depth = np.mean(indent_depths) if indent_depths else 0

    # Count structural elements
    loops = len(LOOP_KW_RE.findall(code))
    conditionals = len(CONDITIONAL_RE.findall(code))
    funcs = len(FUNC_DEF_RE.findall(code))
    classes = len(CLASS_DEF_RE.findall(code))
    complex_logic = len(COMPLEX_LOGIC_RE.findall(code))

    # Nesting score = loops × max_depth (proxy for deeply nested loops)
    nesting_score = loops * max_depth

    return {
        "max_indent_depth": max_depth,
        "avg_indent_depth": round(avg_depth, 2),
        "loop_count": loops,
        "conditional_count": conditionals,
        "func_count": funcs,
        "class_count": classes,
        "complex_logic_count": complex_logic,
        "nesting_score": nesting_score,
    }

struct_df = df[CODE_COL].apply(analyze_structure).apply(pd.Series)
df = pd.concat([df, struct_df], axis=1)

struct_cols = ["max_indent_depth", "avg_indent_depth", "loop_count",
               "conditional_count", "func_count", "complex_logic_count", "nesting_score"]

summary = df.groupby("label_name")[struct_cols].mean()
print("\n  Structural Complexity Summary:\n")
print(summary.round(3).to_string())

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

plot_cols = ["max_indent_depth", "avg_indent_depth", "loop_count",
             "conditional_count", "complex_logic_count", "nesting_score"]

for i, col in enumerate(plot_cols):
    clip_val = df[col].quantile(0.97)
    sns.violinplot(data=df[df[col] <= clip_val], x="label_name", y=col, ax=axes[i],
                   palette=PALETTE_BINARY, hue="label_name", legend=False, inner="box")
    axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")

plt.suptitle("Structural Complexity: Human vs AI", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "11_structural_complexity")
plt.show()


# %% [markdown]
# ---
# ## 5. 📊 Feature Correlation Heatmap
#
# Visualize correlations between all engineered features to identify
# redundant vs complementary signals for ML.

# %%
print("=" * 60)
print(" 5.  Feature Correlation Heatmap")
print("=" * 60)

feature_cols = (verbosity_cols +
                ["naming_consistency", "snake_ratio", "camel_ratio",
                 "avg_id_len", "short_id_ratio", "long_id_ratio",
                 "comment_ratio", "has_human_markers", "inline_comment_ratio",
                 "op_spacing_consistency", "trailing_ws_ratio", "mixed_indent",
                 "max_indent_depth", "avg_indent_depth", "loop_count",
                 "conditional_count", "complex_logic_count", "nesting_score"])

corr_with_label = df[feature_cols + [LABEL_COL]].corr()[LABEL_COL].drop(LABEL_COL)
corr_with_label = corr_with_label.sort_values(ascending=False)

print("\n  Feature correlation with label (sorted):\n")
for feat, corr in corr_with_label.items():
    direction = "→ AI" if corr > 0 else "→ Human"
    bar = "█" * int(abs(corr) * 50)
    print(f"  {feat:35s} {corr:+.4f}  {bar}  {direction}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Correlation with label
colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in corr_with_label.values]
axes[0].barh(corr_with_label.index, corr_with_label.values, color=colors, edgecolor="white")
axes[0].set_title("Feature Correlation with Label", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Pearson Correlation")
axes[0].axvline(0, color="black", linewidth=0.8)

# Full feature correlation matrix
corr_matrix = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, ax=axes[1], cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, fmt=".1f",
            xticklabels=True, yticklabels=True,
            cbar_kws={"shrink": 0.8})
axes[1].set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
axes[1].tick_params(axis="x", rotation=45)
axes[1].tick_params(axis="y", rotation=0)

plt.tight_layout()
save_fig(fig, "12_feature_correlations")
plt.show()


# %% [markdown]
# ---
# ## 6. 📊 Cross-Validation on Validation Set
#
# Quick sanity check: do the patterns we found in train hold on validation?

# %%
print("=" * 60)
print(" 6.  Validation Set Sanity Check")
print("=" * 60)

val_df = val.copy()
val_df["label_name"] = val_df[LABEL_COL].map(LABEL_NAMES)

# Compute a few key features on validation
val_df["char_count"] = val_df[CODE_COL].str.len()
val_df["line_count"] = val_df[CODE_COL].str.count("\n") + 1
val_naming = val_df[CODE_COL].apply(analyze_naming).apply(pd.Series)
val_comments = val_df[CODE_COL].apply(analyze_comments).apply(pd.Series)

val_df = pd.concat([val_df, val_naming, val_comments], axis=1)

print("\nValidation – Key features by label:\n")
key_feats = ["char_count", "line_count", "naming_consistency",
             "comment_ratio", "has_human_markers"]
available = [f for f in key_feats if f in val_df.columns]
print(val_df.groupby("label_name")[available].mean().round(4).to_string())

print("\n✓ If patterns match train → features are robust and generalizable.")


# %% [markdown]
# ---
# ## 7. 🎯 Summary: Top Features for ML
#
# Based on correlation with label and separability in plots:
#
# | Feature | Direction | Rationale |
# |---------|-----------|-----------|
# | `comment_ratio` | ↕ varies | AI tends to have more structured comments |
# | `has_human_markers` | → Human | TODO/FIXME/HACK almost exclusively human |
# | `naming_consistency` | → AI | AI is perfectly consistent |
# | `avg_id_len` | → AI | AI uses longer, descriptive names |
# | `short_id_ratio` | → Human | `i`, `j`, `x`, `n` are human shortcuts |
# | `trailing_ws_ratio` | → Human | Human editors leave trailing whitespace |
# | `mixed_indent` | → Human | Mixing tabs/spaces is human messiness |
# | `op_spacing_consistency` | → AI | AI formatters are consistent |
# | `max_indent_depth` | ↕ varies | Extreme nesting more common in human code |
# | `nesting_score` | ↕ varies | Deep nested loops = complex human code |
# | `empty_line_ratio` | ↕ varies | AI uses more structural blank lines |
#
# **Next Steps**:
# 1. Extract these features for all splits
# 2. Train a gradient-boosted classifier (XGBoost/LightGBM) as baseline
# 3. Fine-tune CodeBERT/UniXcoder with these as auxiliary features

# %%
print("\n" + "=" * 60)
print(" ✓ EDA Complete!")
print("=" * 60)
print(f"\n  Total features engineered: {len(feature_cols)}")
print(f"  Plots saved to: {IMG_DIR}/")
print(f"\n  Feature columns:\n  {feature_cols}")
