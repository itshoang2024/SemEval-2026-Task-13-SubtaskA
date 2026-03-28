# SemEval-2026 Task 13: GenAI Code Detection & Attribution

> **Shared Task on Multilingual AI-Generated Code Detection** | SemEval 2026 · Task 13 (Subtasks A, B, C)
> 
> Distinguishing between human-written and LLM-generated code across multiple programming languages and attributing specific generator models.

## Repository Structure

We implement an organized Data Engineering pipeline to ingest, store, and process the dataset. 

```
.
├── 📁 data/                          # Dataset directory (gitignored)
│   ├── 📁 raw/                       # Immutable, raw downloaded datasets
│   └── 📁 processed/                 # Cleaned datasets and feature sets
│
├── 📁 notebooks/                     # Exploratory Data Analysis
│   ├── 01_eda_starter.py             # Basic EDA, schemas, imbalances
│   └── 01_advanced_eda.py            # Advanced stylometric feature distribution
│
├── 📁 src/                           # Source Code
│   └── 📁 data/                      # Data engineering modules
│       └── 🐍 download_data.py       # Kaggle download & directory scaffolding script
│
├── ⚙️ environment.yml                # Conda dependencies
├── .gitignore                        
└── README.md
```

## Quick Start

### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate semeval2026
```

### 2. Configure Credentials
Option A: Place `kaggle.json` in `~/.kaggle/`
Option B: Create a `.env` file at the root:
```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### 3. Data Ingestion
Run the data pipeline to automatically download and extract data into `data/raw/`:
```bash
python src/data/download_data.py
```

### 4. Exploratory Data Analysis
Start interacting with the data to investigate features:
```bash
# Basic Dataset Inspection (Starter)
python notebooks/01_eda_starter.py

# Advanced Feature Engineering (Advanced)
python notebooks/01_advanced_eda.py 
```

## Task Sub-Objectives
- **Subtask A**: Binary classification (Human vs. AI-Generated).
- **Subtask B & C**: Granular generator attribution (e.g., GPT-4 vs Claude-3 vs Gemini).
