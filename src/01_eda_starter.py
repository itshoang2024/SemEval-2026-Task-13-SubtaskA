# %% [markdown]
# # SemEval-2026 Task 13: EDA Starter
#
# Exploratory Data Analysis for GenAI Code Detection & Attribution (Subtasks A, B, C).
# This notebook loads the raw data, checks the schema, missing values, and visualizes
# the distribution of human vs LLM-generated code.
#
# **Directory assumption**: Assumes data has been downloaded to `data/raw/` via
# `src/data/download_data.py`.

# %%
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics for plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

# %% [markdown]
# ## 1. Load Data
# We will locate the 'train.parquet' file inside `data/raw/`. 

# %%
# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# The dataset zip extracts into a 'Task_A' directory normally
train_path = RAW_DATA_DIR / "Task_A" / "train.parquet"
if not train_path.exists():
    train_path = RAW_DATA_DIR / "train.parquet"
    
if not train_path.exists():
    raise FileNotFoundError(f"Data not found at {train_path}. Run download_data.py first.")

print(f"Loading training data from {train_path}...")
df_train = pd.read_parquet(train_path)
print(f"Dataset shape: {df_train.shape}")

# %% [markdown]
# ## 2. Data Schema & Missing Values
# Understanding the columns available (Subtasks A, B, C features) and data quality.

# %%
print("--- Data Schema ---")
print(df_train.dtypes)

print("\n--- First 3 Rows ---")
display(df_train.head(3))

print("\n--- Missing Values ---")
missing_vals = df_train.isnull().sum()
print(missing_vals[missing_vals > 0] if missing_vals.any() else "No missing values found.")

# Describe numerical/categorical basics
print("\n--- Summary Statistics ---")
display(df_train.describe(include='all').T)

# %% [markdown]
# ## 3. Feature Engineering: Basic Code Metrics
# Calculate the length of the code snippets to see if verbosity separates human from AI.

# %%
# Assuming 'code' or 'code_snippet' is the column containing the source code.
# We'll dynamically find the code column:
code_col = 'code' if 'code' in df_train.columns else 'code_snippet'
label_col = 'label' if 'label' in df_train.columns else 'target'
generator_col = 'generator' if 'generator' in df_train.columns else 'model'
lang_col = 'language' if 'language' in df_train.columns else 'lang'

# Calculate text lengths
df_train['char_length'] = df_train[code_col].str.len()
df_train['line_count'] = df_train[code_col].str.count('\n') + 1

print("\n--- Length Statistics ---")
display(df_train[['char_length', 'line_count']].describe())

# %% [markdown]
# ## 4. Visualizations: Labels and Languages
# Explore the balance of the dataset.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Binary Label Distribution (Subtask A: Human vs AI)
# Label 0 is usually Human, Label 1 is AI-Generated
if label_col in df_train.columns:
    vc_label = df_train[label_col].value_counts().sort_index()
    sns.barplot(x=vc_label.index, y=vc_label.values, ax=axes[0], palette=['#2ecc71', '#e74c3c'])
    axes[0].set_title("Label Distribution (Subtask A)", fontweight='bold')
    axes[0].set_xlabel("Label (0=Human, 1=AI)")
    axes[0].set_ylabel("Count")

    # Add counts on top of bars
    for i, v in enumerate(vc_label.values):
        axes[0].text(i, v + (v * 0.02), f"{v:,}", ha='center', va='bottom')

# Plot 2: Programming Language Distribution
if lang_col in df_train.columns:
    top_langs = df_train[lang_col].value_counts().head(10)
    sns.barplot(x=top_langs.values, y=top_langs.index, ax=axes[1], palette='viridis')
    axes[1].set_title("Top 10 Programming Languages", fontweight='bold')
    axes[1].set_xlabel("Count")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Subtask B/C: Generator Family Attribution
# Visualizing the distribution of specific generators (e.g., GPT-4, Claude, Human).

# %%
if generator_col in df_train.columns:
    plt.figure(figsize=(10, 6))
    gen_counts = df_train[generator_col].value_counts().head(15)
    
    # Highlight 'human' differently if present
    colors = ['#2ecc71' if g.lower() == 'human' else '#3498db' for g in gen_counts.index]
    
    ax = sns.barplot(x=gen_counts.values, y=gen_counts.index, palette=colors)
    ax.set_title("Generator Distribution (Subtasks B & C)", fontweight='bold')
    ax.set_xlabel("Count")
    ax.set_ylabel("Generator Model")
    
    # Add count text
    for i, v in enumerate(gen_counts.values):
        ax.text(v + (v * 0.01), i, f"{v:,}", va='center')
        
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Next Steps
# - Run `src/data/download_data.py` periodically if the data updates.
# - Investigate stylometric and structural complexity features in `notebooks/01_advanced_eda.py`
# - Prepare tokenization and model ingestion scripts using artifacts generated in `data/processed/`.
