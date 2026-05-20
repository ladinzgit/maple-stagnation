# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Applied Data Analysis term project (응용데이터분석 텀프로젝트) that detects MapleStory "parking users" (주차 유저) — players who freeze their character's growth at a specific level to farm boss drops repeatedly — using Nexon OpenAPI data and unsupervised clustering.

Full project plan and hypothesis details are in `docs/메이플스토리 주차 유저 클러스터링.md`.

**Three research hypotheses:**
1. Parking users form a distinct cluster separable by K-Means/DBSCAN on growth-change features
2. Parking users are unevenly distributed across level brackets and job classes (Chi-Square test)
3. A rule derived from Feature Importance can identify parking users with an acceptable False Positive Rate

## Running the Scripts

```powershell
# Install dependencies
pip install requests pandas python-dotenv scikit-learn scipy xgboost matplotlib seaborn numpy statsmodels

# Collect main characters (stratified random sampling, writes data/main_characters.csv)
python scripts/collect_main_characters.py

# Collect 24-month monthly snapshots and compute delta features (writes data/features_monthly.csv, ~13 min)
python scripts/collect_features.py

# Open analysis notebook (data collection already complete — start here for analysis)
jupyter notebook eda/eda.ipynb
```

**Data collection is already complete.** Both CSV files exist locally in `data/` (gitignored). Do not re-run the collection scripts unless the data needs refreshing.

## API Configuration

The Nexon OpenAPI key is stored in `.env` as `MAPLE_API_KEY`. All scripts load it via `python-dotenv`.

**Critical API constraints:**
- Rate limit: **500 req/s**, **20,000,000 req/day** — both scripts use a `RateLimiter` capped at 400 req/s (80% of limit) with 30 concurrent threads
- Data availability: last 2 years only; snapshots refresh daily around 08:00 KST
- `date` parameter format: `YYYY-MM-DD` (use yesterday or earlier — today's data may not be ready)
- History APIs (cube, starforce, potential) require account-owner authentication → **excluded from this project**

## Data Collection Architecture

### Pipeline overview

Both scripts share the same `RateLimiter` + `ThreadPoolExecutor(30)` pattern and a persistent `requests.Session` with a 60-connection pool.

**`collect_main_characters.py`** — stratified random sampling of main characters

1. Samples union-ranking pages across 5 tiers (pages 1–6000) using `PAGES_PER_TIER=5` random pages per tier
2. For each candidate: `id` → `character/basic` → `user/union-raider` (3 sequential calls)
3. Main-character filter: `max(union_block.block_level) <= character_level` (if a higher-level block exists, the character is an alt)
4. Saves to `main_characters.csv` incrementally (UTF-8-BOM, deduped by OCID); stops at `TARGET_COUNT=1300`

**`collect_features.py`** — 24-month monthly snapshot collection

1. Reads `main_characters.csv`, skips OCIDs already in `features_monthly.csv`
2. For each character × each of 24 months (2024-06 → 2026-05): calls `character/basic`, `character/stat` × 7 days (→ max combat power), `user/union`, `character/symbol-equipment`; if `basic` returns None the month is skipped
3. `avg_monthly_delta()` computes per-feature monthly average change between first and last valid month (requires ≥2 valid months)
4. Saves to `features_monthly.csv` incrementally every 100 characters

Total API calls: ~1,300 × 24 × 10 ≈ 312,000 → ~780 s at 400 req/s

### Output files

| File | Description |
|---|---|
| `data/main_characters.csv` | 1,497 main characters with name, OCID, level, class, world, union level |
| `data/features_monthly.csv` | 1,497 rows; 24-month delta features per character (see below) |

### Feature columns in `features_monthly.csv`

Snapshot values are taken from the **last valid month**. Delta values are monthly averages over the observed window.

| Column | Description |
|---|---|
| `level`, `union_level` | Latest snapshot value |
| `arcane_symbol_score`, `authentic_symbol_score` | Sum of symbol levels (latest) |
| `exp`, `log_exp` | Latest exp; `log1p` transform |
| `avg_monthly_delta_level` | Key parking signal: near-zero for parked users |
| `avg_monthly_delta_combat_power` | Key parking signal |
| `avg_monthly_delta_union_level` | Key parking signal |
| `avg_monthly_delta_arcane_symbol` | Symbol growth rate |
| `avg_monthly_delta_authentic_symbol` | Symbol growth rate |
| `first_valid_month`, `last_valid_month`, `num_valid_months` | Valid data window |

## Analysis Notebook (`eda.ipynb`)

EDA is complete. The notebook is structured in sections:

| Section | Content |
|---|---|
| Sec 0 | Environment setup, load `features_monthly.csv`, define `DELTA_COLS`, `BAND_PALETTE`, level bands |
| Sec 1 | Data quality: 34 rows NaN in `delta_level/cp/union` → listwise deletion → `df_clean` (1,463 rows); `num_valid_months` is always 24 |
| Sec 2–3 | Univariate distributions; bivariate correlations; VIF (all ≤ 5 — no multicollinearity) |
| Sec 4 | Job class grouping via `CLASS_GROUP_MAP` (5 계열: 전사/마법사/궁수/도적/해적) |
| Sec 5 | Level band Chi-Square pre-check: H2 feasible for 4 level bands and 5 class groups |
| Sec 6–7 | Exploratory: `exp_rank_within_level` as parking proxy; arcane/authentic symbol saturation analysis |
| Sec 8 | `stagnation_score` (0–5): 54 characters score 5 (all signals stagnant) |
| Sec 9 | H1/H2/H3 feasibility checklists |
| Sec 10 | **Preprocessing decisions** — defines `df_final` and two candidate feature sets for clustering |

### Key EDA Findings Affecting Clustering (Sec 10)

- **`delta_cp` Winsorize P5–P95**: 22% of values are negative; apply `winsorize(limits=[0.05, 0.05])`
- **`delta_union` and `delta_arcane` negative values → clamp to 0** (3 and 1 rows respectively)
- **Arcane symbol binarization**: 84% of characters have `arcane_symbol_score == 120` (max), making `delta_arcane` 0 for 67% → replace with `arcane_stagnant` binary flag (`arcane < 120 AND delta == 0`)
- **Two candidate feature sets** to compare by Silhouette Score:
  - `CLUSTER_FEATURES_A`: `[delta_level, delta_cp, delta_union, delta_authentic, arcane_stagnant]`
  - `CLUSTER_FEATURES_B`: original 5 delta columns (arcane as-is)

### Remaining Analysis Phases

| Phase | Input | Method | Output |
|---|---|---|---|
| Clustering (H1) | `df_final`, feature sets A & B | StandardScaler → K-Means (Elbow/Silhouette) + DBSCAN | Cluster labels |
| Distribution test (H2) | Cluster labels × level band/class group | Chi-Square (α=0.05) | p-values |
| Rule evaluation (H3) | Cluster labels as pseudo-labels | Random Forest / XGBoost → threshold rules | Precision/Recall/FPR/ROC-AUC |

### Other Files

- `docs/PLAN.md`: Historical design document for the `collect_features.py` monthly-snapshot redesign. Superseded by the implemented code — reference only.
