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

# Collect main characters (ranking/overall, level 260–285, 5계열×400=2,000명, writes data/main_characters.csv)
python scripts/collect_main_characters.py

# Collect 12-month monthly snapshots and compute delta features (writes data/features_monthly.csv, ~11 min)
python scripts/collect_features.py

# EDA (read-only — do NOT add experiment code here)
jupyter notebook eda/eda.ipynb

# H1 clustering experiment
jupyter notebook h1_clustering/h1_clustering.ipynb
```

**Data collection is complete.** Both CSV files exist in `data/` (gitignored).

## API Configuration

The Nexon OpenAPI key is stored in `.env` as `MAPLE_API_KEY`. All scripts load it via `python-dotenv`.

**Critical API constraints:**
- Rate limit: **500 req/s**, **20,000,000 req/day** — both scripts use a `RateLimiter` capped at 400 req/s with 30 concurrent threads
- Data availability: last 2 years only; snapshots refresh daily around 08:00 KST
- `date` parameter format: `YYYY-MM-DD` (use yesterday or earlier)
- History APIs (cube, starforce, potential) require account-owner authentication → excluded

## Data Collection Architecture

Both scripts share the same `RateLimiter` + `ThreadPoolExecutor(30)` pattern and a persistent `requests.Session` with a 60-connection pool.

**`collect_main_characters.py` (v2)** — `ranking/overall`-based main character collection, level 260–285

1. Binary-search page ranges per job class to locate the 260–285 band in `ranking/overall`
2. 5계열 × 400명 = 2,000명 target; per-계열: 3 level bins (260–269/270–279/280–285) each ~133명
3. Phase 1 (diversity): collect up to `max(MIN_PER_CLASS=10, 400÷job_count)` per job; Phase 2 (fill): round-robin up to `MAX_PER_CLASS=100`
4. For each candidate: `ranking/overall` → `id` → `character/basic` (create-date filter) → `user/union-raider` (main-char check)
5. Main-char filter: `max(union_block.block_level) <= character_level`
6. Create-date filter: `character_date_create > CREATE_CUTOFF=2025-06-30` → skip (new chars lack 12-month window)
7. Saves to `main_characters.csv` incrementally (UTF-8-BOM, deduped by OCID)

Design rationale: level 285+ active users also have `delta_level ≈ 0` → noise; 260–285 is the parking-signal window.

**`collect_features.py` (v2.2)** — 12-month monthly snapshot collection

1. Reads `main_characters.csv`, skips OCIDs already in `features_monthly.csv`
2. 12-month window (not 24): current parking behavior matters; 24mo risks classifying "parked-then-returned" users
3. Per character × 12 months: `character/basic` + `character/stat` × 7 days (→ max combat power) + `user/union` + `character/symbol-equipment` + `character/hexamatrix`
4. Computes `avg_monthly_delta_*` (12mo) and `recent{3,6}_delta_*` (short-window slope)
5. Saves to `features_monthly.csv` incrementally every 100 characters

Total API calls: ~2,000 × 12 × 11 ≈ 264,000 → ~660 s at 400 req/s

### Output files

| File | Description |
|---|---|
| `data/main_characters.csv` | ~2,000 main characters, level 260–285, 5계열 균등 |
| `data/features_monthly.csv` | ~2,000 rows; 12-month delta features per character |
| `data/cluster_labels.csv` | `cluster_km`, `is_parking` per character (written by H1 notebook) |

### Feature columns in `features_monthly.csv`

| Column | Description |
|---|---|
| `level`, `union_level` | Latest snapshot value |
| `arcane_symbol_score`, `authentic_symbol_score` | Sum of symbol levels (latest) |
| `hexa_level_sum` | Sum of HEXA core levels (latest) — clean monotonic activity signal at 260+ |
| `avg_monthly_delta_level` | Key parking signal: near-zero for parked users |
| `avg_monthly_delta_combat_power` | Key parking signal |
| `avg_monthly_delta_union_level` | Key parking signal |
| `avg_monthly_delta_arcane_symbol` | Symbol growth rate |
| `avg_monthly_delta_authentic_symbol` | Symbol growth rate |
| `avg_monthly_delta_hexa` | HEXA growth rate |
| `recent3_delta_*`, `recent6_delta_*` | Short-window (3/6 mo) slopes — age-debiased parking signal |
| `access_active_months`, `access_ratio`, `access_recent` | Recent login activity (from `access_flag` in `character/basic`) |
| `character_age_months`, `created_in_window` | Age diagnostics; `created_in_window=1` = new class (렌 cohort) |
| `first_valid_month`, `last_valid_month`, `num_valid_months` | Valid data window |

## Notebook Structure

**Rule: experiment code goes in hypothesis folders, never in `eda/eda.ipynb`.**

| Notebook | Role |
|---|---|
| `eda/eda.ipynb` | EDA only — distributions, correlations, preprocessing decisions. Do not add H1/H2/H3 code. |
| `h1_clustering/h1_clustering.ipynb` | H1: K-Means + DBSCAN clustering |
| `h2_distribution/` | H2: Chi-Square distribution tests (notebook not yet created) |
| `h3_rule/` | H3: Random Forest / XGBoost rule extraction (notebook not yet created) |

Each hypothesis notebook is self-contained: loads `data/features_monthly.csv` and reproduces preprocessing inline (no intermediate CSV hand-off from eda.ipynb).

## Clustering Decisions (H1 — Complete)

### Feature sets

- **Feature Set A (12mo avg) — primary**: `[avg_monthly_delta_level, delta_cp(winsorized P5–P95), delta_union(clamped ≥ 0), avg_monthly_delta_authentic_symbol, arcane_stagnant]`
- **Feature Set A' (recent6) — age-debiased**: same but using `recent6_delta_*`; used to check if 렌 cohort (2025-06 launch, high initial delta) causes age bias in A

`normalized_delta_level` is **excluded** — empirical P75 normalization is biased by parking-user concentration in the 260–270 range (Spearman r=−0.429, wiki vs empirical mismatch). Only propose it if the user explicitly asks.

### Preprocessing

- `delta_cp`: `winsorize(limits=[0.05, 0.05])` — 22% negative values
- `delta_union`, `delta_arcane`: clamp to 0 (tiny negative artifact)
- `arcane_stagnant` binary: `arcane_symbol_score < 120 AND avg_monthly_delta_arcane == 0`

### H1 Results (on v1 data — re-run needed on v2 data)

Best k=5, Silhouette=0.4517 → "분리 가능 (H1 지지)". Parking cluster (ID=1): 52명 (3.8%), 78.8% parked-proxy, stagnation_score=5 for 100% of stag5 chars.

## Remaining Analysis Phases

| Phase | Notebook | Status |
|---|---|---|
| H1 Clustering | `h1_clustering/h1_clustering.ipynb` | Complete (v1 data); re-run on v2 data |
| H2 Distribution test | `h2_distribution/` (create notebook) | Not started |
| H3 Rule evaluation | `h3_rule/` (create notebook) | Not started |

H2 uses `cluster_labels.csv` (from H1) × level_band/class_group → Chi-Square (α=0.05).  
H3 uses `is_parking` as pseudo-labels → Random Forest/XGBoost → threshold rules → Precision/Recall/FPR/ROC-AUC.

## Other Files

- `docs/PLAN.md`: Historical design document. Superseded by implemented code — reference only.
- `h1_clustering/h1_clustering.py`: Script version of H1 notebook.
- `assets/NanumSquareNeo-bRg.ttf`: Korean font used in all notebooks.
