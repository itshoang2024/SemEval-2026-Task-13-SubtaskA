# Phase 4: XGBoost GPU Training Pipeline

## Tong quan

Script `src/03_train_xgboost_gpu.py` thuc hien toan bo pipeline:

```
[500k train.parquet] ──┐
[100k val.parquet]   ──┼──> Feature Extraction ──> XGBoost GPU ──> submission.csv
[500k test.parquet]  ──┘     (13 features)         (5-fold CV)
```

---

## Cach chay tren Google Colab (T4 GPU)

### Buoc 1: Tao notebook moi tren Colab

Vao [colab.research.google.com](https://colab.research.google.com), tao notebook moi.

**Bat GPU**: Runtime > Change runtime type > T4 GPU

### Buoc 2: Cai dat dependencies

```python
!pip install xgboost tree-sitter tree-sitter-python tree-sitter-java tree-sitter-cpp -q
```

### Buoc 3: Upload du lieu

```python
import os
os.makedirs('/content/data', exist_ok=True)

# Option A: Upload tu may
from google.colab import files
# uploaded = files.upload()  # Upload train.parquet, validation.parquet, test.parquet

# Option B: Tu Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/SemEval/Task_A/*.parquet /content/data/

# Option C: Tu Kaggle
!pip install kaggle -q
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c semeval-2026-task13 -p /content/data/ && cd /content/data && unzip -o *.zip
```

### Buoc 4: Upload va chay script

```python
# Upload file src/03_train_xgboost_gpu.py
# Hoac paste noi dung vao cell

%run src/03_train_xgboost_gpu.py
```

### Buoc 5: Download ket qua

```python
from google.colab import files
files.download('/content/outputs/submission.csv')
files.download('/content/outputs/xgb_5fold_models.pkl')
```

---

## Outputs

| File | Mo ta |
|---|---|
| `outputs/submission.csv` | File nop: `ID, label` |
| `outputs/test_predictions.parquet` | Probabilities + labels |
| `outputs/xgb_5fold_models.pkl` | 5 models + metadata |
| `outputs/features_train_500k.parquet` | Cached train features |
| `outputs/features_val_100k.parquet` | Cached val features |
| `outputs/features_test_500k.parquet` | Cached test features |

---

## Thoi gian du kien (T4 GPU)

| Buoc | Thoi gian |
|---|---|
| Feature extraction (500k train) | ~10-15 phut |
| Feature extraction (100k val) | ~2-3 phut |
| Feature extraction (500k test) | ~10-15 phut |
| XGBoost 5-fold training (GPU) | ~3-5 phut |
| Inference | ~1 phut |
| **Tong** | **~25-40 phut** |

> Lan chay thu 2 se nhanh hon vi features duoc cache vao parquet.

---

## 13 Features su dung

Xem chi tiet tai `FEATURE_SELECTION.md`.

| # | Feature | Nhom |
|---|---|---|
| 1 | `indent_consistency` | Stylometric |
| 2 | `avg_line_length` | Stylometric |
| 3 | `shannon_entropy` | Statistical |
| 4 | `comment_to_code_ratio` | Stylometric |
| 5 | `snake_ratio` | Stylometric |
| 6 | `trailing_ws_ratio` | Stylometric |
| 7 | `avg_identifier_length` | Stylometric |
| 8 | `camel_ratio` | Stylometric |
| 9 | `avg_ast_depth` | Structural |
| 10 | `token_entropy` | Statistical |
| 11 | `long_id_ratio` | Stylometric |
| 12 | `branch_ratio` | Structural |
| 13 | `zlib_compression_ratio` | Statistical |

---

## XGBoost Hyperparameters

```python
{
    "n_estimators": 1000,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "device": "cuda",       # T4 GPU
}
```

---

## Format file nop (Submission)

```csv
ID,label
0,1
1,0
2,1
...
```

- `ID`: ID tu test.parquet
- `label`: 0 = Human, 1 = AI-Generated

---

## Luu y quan trong

1. **Test set KHONG co cot `language`** — script tu detect ngon ngu bang regex heuristics
2. **Features duoc cache** — neu da chay 1 lan, lan sau chi can train model
3. **GPU mode**: XGBoost dung `tree_method="hist"` + `device="cuda"` (nhanh hon CPU 3-5x)
4. **5-fold ensemble**: Ket qua cuoi cung la trung binh xac suat cua 5 models
