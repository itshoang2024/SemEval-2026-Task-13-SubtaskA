# SemEval-2026 Task 13 — SubtaskA: AI-Generated Code Detection

## Ban do du an (Project Map)

> **Nhiem vu**: Cho 1 doan code, phan loai no la do **Human viet (0)** hay **AI sinh (1)**.

---

## 1. Cau truc thu muc

```
SemEval-2026-Task-13-SubtaskA/
├── .env                          # Kaggle credentials (KAGGLE_USERNAME, KAGGLE_KEY)
├── .env.example                  # Template cho .env
├── .gitignore                    # Ignore data files, logs, etc.
├── environment.yml               # Conda environment (semeval2026)
├── kaggle.json                   # Kaggle API key
├── kaggle_workflow.md            # Huong dan download data tu Kaggle
├── README.md                     # README goc
│
├── data/
│   ├── download_data.py          # Script download data tu Kaggle API
│   ├── raw/
│   │   └── Task_A/
│   │       ├── train.parquet     # 500,000 samples (train set)
│   │       ├── validation.parquet # 100,000 samples (validation set)
│   │       ├── test.parquet      # Test set (KHONG co label - dung de nop)
│   │       └── test_sample.parquet # Sample nho cua test set
│   └── processed/
│       ├── sample_20k_features.parquet  # 20k samples + 21 features (output Phase 2)
│       └── feature_list.txt             # Danh sach 21 features
│
├── src/
│   ├── 01_eda_starter.py         # [Phase 1] EDA co ban: schema, distribution
│   ├── 01_advanced_eda.py        # [Phase 1] EDA nang cao: stylometric, whitespace
│   └── 02_forensic_eda_features.py # [Phase 2-3] *** FILE CHINH ***
│                                    # Integrity check + 3 nhom features + XGBoost
│
├── notebooks/
│   ├── eda.ipynb                 # Notebook EDA cu (Phase 1)
│   └── 02_forensic_eda_report.ipynb  # *** NOTEBOOK TONG HOP ***
│                                      # Hien thi tat ca ket qua Phase 2-3
│
└── img/
    └── phase2/                   # 9 bieu do duoc tao tu Phase 2-3
        ├── 01_integrity_distributions.png
        ├── 02_stylometric_violin.png
        ├── 03_stylometric_boxplots.png
        ├── 04_statistical_distributions.png
        ├── 05_structural_ast.png
        ├── 06_ast_depth_per_lang.png
        ├── 07_correlation_heatmap.png
        ├── 08_label_correlation_bar.png
        └── 09_xgboost_importance.png
```

---

## 2. Y nghia tung file

### 2.1 Files Cau hinh (Config)

| File | Y nghia |
|---|---|
| `.env` | Chua `KAGGLE_USERNAME` va `KAGGLE_KEY` de download data |
| `.env.example` | Template — copy thanh `.env` va dien thong tin |
| `environment.yml` | Cai dat moi truong conda voi tat ca dependencies |
| `kaggle.json` | API key cua Kaggle (alternative cho .env) |
| `.gitignore` | Ignore `data/`, `*.parquet`, `__pycache__`, etc. |

### 2.2 Scripts (src/)

| File | Phase | Chuc nang |
|---|---|---|
| `01_eda_starter.py` | Phase 1 | Load data, xem schema, label distribution, language distribution |
| `01_advanced_eda.py` | Phase 1 | Phan tich naming conventions, comment patterns, whitespace, structural complexity |
| **`02_forensic_eda_features.py`** | **Phase 2-3** | **Script chinh**: Integrity check, 3 nhom features, hypothesis tests, correlation, XGBoost importance |

### 2.3 Notebooks (notebooks/)

| File | Chuc nang |
|---|---|
| `eda.ipynb` | Notebook EDA Phase 1 (cu) |
| **`02_forensic_eda_report.ipynb`** | **Notebook tong hop Phase 2-3** — Mo file nay de xem tat ca charts va ket qua |

### 2.4 Du lieu (data/)

| File | Mo ta | Kich thuoc |
|---|---|---|
| `data/raw/Task_A/train.parquet` | Training set | 500,000 rows x 4 cols |
| `data/raw/Task_A/validation.parquet` | Validation set | 100,000 rows x 4 cols |
| `data/raw/Task_A/test.parquet` | **Test set (KHONG co label)** — dung de nop | ? rows x 3 cols |
| `data/processed/sample_20k_features.parquet` | 20k samples + 21 features | 9.2 MB |
| `data/processed/feature_list.txt` | Danh sach 21 feature names | 365 bytes |

### 2.5 Charts (img/phase2/)

| File | Noi dung |
|---|---|
| `01_integrity_distributions.png` | Language x Label + Top 15 Generators |
| `02_stylometric_violin.png` | Naming Consistency & Trailing WS (Human vs AI) |
| `03_stylometric_boxplots.png` | Identifier Length, Comment Ratio, Keyword Density |
| `04_statistical_distributions.png` | Zlib Compression, Shannon Entropy, Token Entropy |
| `05_structural_ast.png` | Max AST Depth, Cyclomatic Complexity, Branch Ratio |
| `06_ast_depth_per_lang.png` | AST Depth theo tung ngon ngu (Python, Java, C++) |
| `07_correlation_heatmap.png` | Ma tran tuong quan 21 features + label |
| `08_label_correlation_bar.png` | Bieu do correlation voi label (sorted by |r|) |
| `09_xgboost_importance.png` | XGBoost Feature Importance (Top features) |

---

## 3. Cot du lieu

| Column | Type | Mo ta |
|---|---|---|
| `code` | string | Doan ma nguon can phan loai |
| `generator` | string | Ten mo hinh AI da sinh ra code (hoac `human`) |
| `label` | int | **0** = Human, **1** = AI-Generated |
| `language` | string | Ngon ngu lap trinh: Python, C++, Java |

> **Luu y**: File `test.parquet` **KHONG co cot `label`** — day la cot ban can du doan.

---

## 4. 21 Features da trich xuat

### Group A: Stylometric (Phong cach viet code)

| # | Feature | Mo ta | Tai sao quan trong? |
|---|---|---|---|
| 1 | `naming_consistency` | Ty le naming style chiem da so (snake/camel/pascal) | AI rat nhat quan, nguoi hay pha tron |
| 2 | `snake_ratio` | Ty le snake_case identifiers | AI thich snake_case |
| 3 | `camel_ratio` | Ty le camelCase identifiers | Language-dependent |
| 4 | `trailing_ws_ratio` | Ty le dong co trailing whitespace | LLM hay them space thua |
| 5 | `avg_line_length` | Chieu dai dong trung binh | AI viet dong dai hon, deu hon |
| 6 | `line_length_variance` | Phuong sai chieu dai dong | Nguoi viet dong ngan dai tuy hung |
| 7 | `indent_consistency` | Nhat quan thut le (space vs tab) | **Feature #1** theo XGBoost |
| 8 | `comment_to_code_ratio` | Ty le comment / code lines | AI viet comment nhieu, trang trong |
| 9 | `avg_identifier_length` | Do dai trung binh ten bien/ham | AI thich ten mo ta dai |
| 10 | `short_id_ratio` | Ty le identifiers ngan (len<=2) | Nguoi hay viet `i`, `j`, `x` |
| 11 | `long_id_ratio` | Ty le identifiers dai (len>=10) | AI thich `input_array`, `result_list` |
| 12 | `keyword_density` | Ty le keyword / tong token | Nguoi dung nhieu keyword hon |

### Group B: Statistical (Ly thuyet thong tin)

| # | Feature | Mo ta | Tai sao quan trong? |
|---|---|---|---|
| 13 | `shannon_entropy` | Entropy ky tu (do hon loan) | Nguoi co entropy cao hon |
| 14 | `zlib_compression_ratio` | Ty le nen zlib | **AI de nen hon** — tin hieu manh |
| 15 | `token_entropy` | Entropy cap tu (word-level) | AI da dang tu vung hon do sinh code dai |
| 16 | `burstiness` | Do cum tu — std/mean cua khoang cach giua cac token lap lai | AI co burstiness cao hon |

### Group C: Structural (Cau truc AST)

| # | Feature | Mo ta | Tai sao quan trong? |
|---|---|---|---|
| 17 | `max_ast_depth` | Do sau lon nhat cua cay cu phap | AI hoi phang hon nhung khac biet nho |
| 18 | `avg_ast_depth` | Do sau trung binh | Tuong quan am voi AI |
| 19 | `ast_node_count` | Tong so node trong AST | Gan nhu khong phan biet |
| 20 | `branch_ratio` | Ty le node branching (if/for/while) | AI it branching hon |
| 21 | `cyclomatic_approx` | Do phuc tap thuong (so branch + 1) | AI code don gian hon |

---

## 5. Ket qua chinh

### 5.1 Kiem tra tinh toan ven
- **0 duplicates** trong ca Train va Validation
- **0 leakage** — khong co sample nao xuat hien o ca 2 tap
- Label **gan can bang**: 52.3% AI, 47.7% Human

### 5.2 Gia thuyet thong ke

| Gia thuyet | Ket qua | p-value | Effect Size |
|---|---|---|---|
| AI code de nen hon | **XAC NHAN** | ~0.00 | large (Cohen's d) |
| AI viet code phang hon | **XAC NHAN** (nhung nho) | < 0.05 | small |

### 5.3 XGBoost Baseline (chi 21 hand-crafted features)

| Metric | Value |
|---|---|
| **AUC-ROC** | **0.9721 +/- 0.0011** |
| **F1-Score** | **0.9214 +/- 0.0032** |

> Chi voi 21 features don gian, XGBoost da dat AUC = 0.97 tren 20k samples!

### 5.4 Top-5 Features (XGBoost Importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `indent_consistency` | 0.1278 |
| 2 | `avg_line_length` | 0.1247 |
| 3 | `shannon_entropy` | 0.0832 |
| 4 | `comment_to_code_ratio` | 0.0740 |
| 5 | `snake_ratio` | 0.0727 |

---

## 6. Ban can lam gi tiep? (Roadmap de nop Challenge)

### Phase 4: Train model tren toan bo du lieu (QUAN TRONG NHAT)

```
[BAN DANG O DAY] ────────────────────────────→ [NOP BAI]
     Phase 2-3 done         Phase 4-7 can lam
```

| Buoc | Mo ta | Do uu tien |
|---|---|---|
| **4a** | Trich xuat 21 features tren **TOAN BO** 500k train samples (khong chi 20k) | CAO |
| **4b** | Train XGBoost/LightGBM tren toan bo features | CAO |
| **4c** | Validation: danh gia tren validation set | CAO |

### Phase 5: Nang cap model (Transformer-based)

| Buoc | Mo ta | Do uu tien |
|---|---|---|
| **5a** | Fine-tune `microsoft/codebert-base` hoac `microsoft/graphcodebert-base` | CAO |
| **5b** | Dung CodeBERT embedding lam **them features** cho XGBoost | TRUNG BINH |
| **5c** | Hoac train head classifier truc tiep tren CodeBERT | CAO |

### Phase 6: Ensemble

| Buoc | Mo ta | Do uu tien |
|---|---|---|
| **6a** | Ket hop XGBoost (21 features) + CodeBERT (transformer) | TRUNG BINH |
| **6b** | Weighted average hoac stacking | TRUNG BINH |

### Phase 7: Tao file nop (Submission)

| Buoc | Mo ta | Do uu tien |
|---|---|---|
| **7a** | Load `test.parquet` (khong co label) | CAO |
| **7b** | Trich xuat features + predict voi model tot nhat | CAO |
| **7c** | Tao file CSV submission: `id, label` | CAO |
| **7d** | Upload len Kaggle (hoac SemEval portal) | CAO |

### Code mau cho Phase 7 (Submission):

```python
import pandas as pd

# Load test data
df_test = pd.read_parquet('data/raw/Task_A/test.parquet')

# Extract features (dung cung ham nhu Phase 2)
# ... extract_stylometric, extract_statistical, ast_features ...

# Predict
predictions = model.predict(X_test)

# Tao submission file
submission = pd.DataFrame({
    'id': range(len(predictions)),
    'label': predictions
})
submission.to_csv('submission.csv', index=False)
```

---

## 7. Cach chay lai tu dau

### Buoc 1: Cai dat moi truong
```bash
conda env create -f environment.yml
conda activate semeval2026
```

### Buoc 2: Download data
```bash
# Setup .env voi KAGGLE_USERNAME va KAGGLE_KEY
python data/download_data.py
```

### Buoc 3: Chay Forensic EDA (Phase 2-3)
```bash
python src/02_forensic_eda_features.py
```

### Buoc 4: Xem ket qua
Mo notebook:
```bash
jupyter notebook notebooks/02_forensic_eda_report.ipynb
```

---

## 8. Cac lenh huu ich

```bash
# Xem data nhanh
python -c "import pandas as pd; df=pd.read_parquet('data/raw/Task_A/train.parquet'); print(df.shape); print(df.head())"

# Xem test set
python -c "import pandas as pd; df=pd.read_parquet('data/raw/Task_A/test.parquet'); print(df.shape); print(df.columns.tolist()); print(df.head(2))"

# Xem features da trich xuat
python -c "import pandas as pd; df=pd.read_parquet('data/processed/sample_20k_features.parquet'); print(df.shape); print(df.columns.tolist())"
```

---

## 9. Links & References

- **Competition**: [SemEval-2026 Task 13](https://semeval.github.io/SemEval2026/tasks)
- **Tree-sitter**: Parser da ngon ngu dung de trich AST features
- **XGBoost**: Gradient boosting — baseline model
- **CodeBERT**: Pre-trained model cho code understanding
- **DetectGPT**: Paper ve phat hien text do AI tao

---

> **Trang thai hien tai**: Phase 2-3 **HOAN THANH**. Du lieu sach, 21 features sẵn sang, XGBoost baseline AUC = 0.97.  
> **Viec tiep theo**: Phase 4 — trich xuat features tren toan bo 500k samples va train model chinh thuc.
