# Feature Selection Report — SemEval-2026 Task 13 SubtaskA

## Tong quan

Tu 21 dac trung ban dau (Phase 2-3), chon **13 dac trung** dua tren 2 tieu chi:
1. **XGBoost Feature Importance (Gain)** — tren 20k sample, 5-Fold CV, AUC=0.9721
2. **Pearson Correlation Heatmap** — loai bo cac cap co |r| > 0.6 (giu feature manh hon)

---

## Cac cap tuong quan cao (|r| > 0.6)

| Cap Feature | |r| | Giu lai | Ly do |
|---|---|---|---|
| `naming_consistency` <-> `snake_ratio` | 0.71 | `snake_ratio` | Importance 0.073 > 0.019 |
| `short_id_ratio` <-> `avg_identifier_length` | 0.80 | `avg_identifier_length` | Importance 0.048 > 0.015 |
| `max_ast_depth` <-> `avg_ast_depth` | 0.81 | `avg_ast_depth` | Importance 0.041 > 0.021 |
| `ast_node_count` <-> `cyclomatic_approx` | 0.78 | *bo ca 2* | Ca 2 deu yeu (0.023, 0.025) |
| `avg_line_length` <-> `line_length_variance` | 0.62 | `avg_line_length` | #2 importance (0.125 >> 0.036) |
| `burstiness` <-> `token_entropy` | 0.61 | `token_entropy` | Importance 0.040 > 0.021 |
| `burstiness` <-> `zlib_compression_ratio` | 0.62 | `zlib_compression_ratio` | Da xac nhan bang hypothesis test |

---

## Bang quyet dinh chon dac trung

### 13 Features DUOC GIU

| # | Feature | Nhom | Importance | Corr w/ label | Ly do giu |
|---|---|---|---|---|---|
| 1 | `indent_consistency` | Stylometric | **0.128** | -0.10 | #1 importance; non-linear signal manh |
| 2 | `avg_line_length` | Stylometric | **0.125** | +0.23 | #2 importance |
| 3 | `shannon_entropy` | Statistical | **0.083** | -0.21 | #3 importance; thong tin entropy doc lap |
| 4 | `comment_to_code_ratio` | Stylometric | **0.074** | +0.32 | #4 importance; AI viet comment nhieu |
| 5 | `snake_ratio` | Stylometric | **0.073** | +0.35 | #5 importance; dai dien naming style |
| 6 | `trailing_ws_ratio` | Stylometric | 0.065 | +0.26 | Tin hieu doc lap (LLM de trailing space) |
| 7 | `avg_identifier_length` | Stylometric | 0.048 | +0.39 | Corr cao nhat voi label; AI dung ten dai |
| 8 | `camel_ratio` | Stylometric | 0.042 | +0.02 | Doc lap voi snake_ratio (r=-0.29) |
| 9 | `avg_ast_depth` | Structural | 0.041 | -0.17 | Dai dien cau truc AST tot hon max_depth |
| 10 | `token_entropy` | Statistical | 0.040 | +0.37 | Word-level info; dai dien burstiness |
| 11 | `long_id_ratio` | Stylometric | 0.037 | +0.29 | Bo sung avg_identifier_length |
| 12 | `branch_ratio` | Structural | 0.033 | -0.25 | Tin hieu cau truc duy nhat (branching) |
| 13 | `zlib_compression_ratio` | Statistical | 0.029 | -0.31 | Xac nhan bang t-test va Mann-Whitney |

### 8 Features BI LOAI

| Feature | Importance | Ly do loai |
|---|---|---|
| `naming_consistency` | 0.019 | r=0.71 voi `snake_ratio` (manh hon) |
| `short_id_ratio` | 0.015 | r=-0.80 voi `avg_identifier_length` (manh hon) |
| `max_ast_depth` | 0.021 | r=0.81 voi `avg_ast_depth` (manh hon) |
| `burstiness` | 0.021 | r=0.61 voi `token_entropy` (manh hon) |
| `line_length_variance` | 0.036 | r=0.62 voi `avg_line_length` (manh hon) |
| `ast_node_count` | 0.023 | r=0.78 voi `cyclomatic_approx`; ca 2 yeu |
| `cyclomatic_approx` | 0.025 | r=0.78 voi `ast_node_count`; ca 2 yeu |
| `keyword_density` | 0.023 | Importance thap, tin hieu yeu |

---

## Cong thuc tinh tung feature

### Group A: Stylometric

**1. indent_consistency**
```
indent_lines = cac dong bat dau bang space hoac tab
tab_count = so dong bat dau bang tab
space_count = so dong bat dau bang space
indent_consistency = max(tab_count, space_count) / total_indent_lines
```
Y nghia: 1.0 = hoan toan nhat quan (chi dung space HOAC tab). AI luon = 1.0.

**2. avg_line_length**
```
avg_line_length = mean(len(line) for line in non_empty_lines)
```

**3. comment_to_code_ratio**
```
single_comments = count(// hoac # comments)
block_comments = count(/* */ hoac """ """ comments)
comment_to_code_ratio = (single + block) / total_non_empty_lines
```

**4. snake_ratio**
```
snake_count = count(regex: \b[a-z][a-z0-9]*(_[a-z0-9]+)+\b)
total_named = snake_count + camel_count + pascal_count
snake_ratio = snake_count / max(total_named, 1)
```

**5. trailing_ws_ratio**
```
trailing_ws_ratio = count(lines co space/tab o cuoi) / total_lines
```

**6. avg_identifier_length**
```
identifiers = tat ca tokens match [a-zA-Z_]\w* KHONG phai keyword
avg_identifier_length = mean(len(id) for id in identifiers)
```

**7. camel_ratio**
```
camel_count = count(regex: \b[a-z][a-z0-9]*([A-Z][a-z0-9]*)+\b)
camel_ratio = camel_count / max(total_named, 1)
```

**8. long_id_ratio**
```
long_id_ratio = count(identifiers co len >= 10) / total_identifiers
```

### Group B: Statistical

**9. shannon_entropy**
```
H(X) = -sum(P(xi) * log2(P(xi)))
voi P(xi) = tan suat xuat hien cua ky tu xi trong code
```

**10. zlib_compression_ratio**
```
compressed = zlib.compress(code.encode('utf-8'), level=6)
zlib_compression_ratio = len(compressed) / len(original_bytes)
```
Thap hon = de nen hon = nhieu pattern lap lai hon (dac trung cua AI).

**11. token_entropy**
```
tokens = tat ca cac tu (word-level)
H_token = -sum(P(ti) * log2(P(ti)))
```

### Group C: Structural

**12. avg_ast_depth**
```
Dung tree-sitter parse code -> AST
Duyet BFS tat ca nodes, ghi nhan depth cua moi node
avg_ast_depth = mean(all_node_depths)
```

**13. branch_ratio**
```
branch_nodes = {if_statement, for_statement, while_statement, 
                try_statement, switch_statement, ...}
branch_ratio = count(branch_nodes) / total_nodes
```

---

## Bieu do tham khao

Xem chi tiet tai:
- `img/phase2/09_xgboost_importance.png` — Feature Importance
- `img/phase2/07_correlation_heatmap.png` — Correlation Heatmap
- `img/phase2/08_label_correlation_bar.png` — Correlation with Label
