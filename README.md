# Maple Parking Detect

메이플스토리 Nexon OpenAPI 데이터로 270~290레벨 본캐 표본에서 성장 정체 기반 주차 후보군을 탐색하는 분석 프로젝트다.

공개 API에서는 주간 보스 수행 기록이나 메소 생산량을 직접 볼 수 없으므로, 본 프로젝트의 결과는 확정 주차 유저 라벨이 아니라 성장/접속 행동 기반의 후보 라벨이다.

## 현재 기준

- 기준일: 2026-06-03 실행 결과
- 수집 표본: active checkpoint 조건을 통과한 본캐 2,000명
- checkpoint months: `2025-06`, `2025-12`, `2026-05`
- checkpoint 이유:
  - 6월, 12월은 메이플스토리 대형 이벤트가 주로 진행되는 시점
  - `2026-05`는 현재 실험 기준에서 실제 활동 중인 유저를 대상으로 하기 위한 시점

현재 로컬 데이터:

| 파일 | 행 수 | 설명 |
|---|---:|---|
| `data/main_characters.csv` | 2,000 | active 본캐 표본 |
| `data/features_monthly.csv` | 2,000 | 12개월 성장/접속 피처 |
| `data/monthly_snapshots_raw.csv` | 24,000 | 월별 raw snapshot |
| `data/hexa_fragments.csv` | 2,000 | HEXA 조각 소비 이력 |
| `data/cluster_labels.csv` | 1,979 | H1 cluster label |
| `data/h1_current_candidates.csv` | 1,884 | 현재성 후보/H2 보조 분석 라벨 |

## H1 결과

H1은 성장 정체 특성이 강한 cluster를 찾는 비지도 분석이다.

최종 선택 피처:

| Alias | 실제 값 |
|---|---|
| `cumexp_avg` | `log1p_avg_monthly_delta_cumexp` |
| `union_avg` | `avg_monthly_delta_union_level`, `[0, p99]` clipping |
| `access_ratio` | 월별 접속 비율 |

결과:

| 항목 | 값 |
|---|---:|
| K-Means k | 3 |
| Silhouette | 0.5446 |
| 주차 후보 cluster id | 1 |
| H1 주차 후보 수 | 412명 |

클러스터별 인원:

| Cluster | 인원 | 해석 |
|---|---:|---|
| 0 | 1,419명 | 일반 성장 군집 |
| 1 | 412명 | 성장 정체/주차 후보 군집 |
| 2 | 148명 | 유니온 성장 특화 군집 |

상세 결과: `h1_clustering/RESULT.md`

## H2 결과

H2는 H1 클러스터 후보 412명이 레벨 구간 또는 직업 계열에 불균일하게 분포하는지 검정한다.

- 기본 분석 표본: 1,979명
- H1 클러스터 후보: 412명, 20.82%
- 보조 현재성 후보: 63명, 3.34%
- 보조 고신뢰 후보: 35명, 1.86%

| 검정 | p | Holm 보정 p | Cramer's V | 판정 |
|---|---:|---:|---:|---|
| 레벨 구간 x H1 후보 | 1.486e-61 | 2.972e-61 | 0.376 | 유의 |
| 직업 계열 x H1 후보 | 0.7476 | 0.7476 | 0.031 | 유의하지 않음 |

H1 후보는 `270-279` 구간에서 40.14%로 강하게 집중되고, `286-290` 구간에서는 2.82%로 낮다. 직업 계열 집중은 확인되지 않는다. 63명 현재성 후보는 보조 분석으로 유지한다.

상세 결과: `h2_distribution/RESULTS.md`

## H3 상태

H3는 아직 구현 예정 단계다.

- 1단계 target: `is_stagnant_cluster` 412명
- 최종 rule 평가 target: `is_current_parking_candidate` 63명
- 민감도 target: `is_high_confidence_candidate` 35명
- 목표: H1 후보 cluster를 다른 관측 피처로 근사하고 현재성 접속 게이트를 결합한 rule 도출

## 실행 순서

```bash
python scripts/collect_main_characters.py
python scripts/collect_features.py --refresh-raw
python scripts/collect_hexa_fragments.py
python eda/profile_active_features.py
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/temporal_external_validation.ipynb
python h2_distribution/run_analysis.py
jupyter nbconvert --to notebook --execute --inplace h2_distribution/h2_distribution.ipynb
```

## 디렉터리 구조

```text
maple_parking_detect/
├── scripts/                 # API-backed collectors
├── eda/                     # feature profile and figures
├── h1_clustering/           # feature selection, clustering, validation
├── h2_distribution/         # chi-square distribution tests
├── h3_rule/                 # planned rule/feature-importance analysis
├── docs/                    # project notes and plans
├── assets/                  # report-ready assets
└── data/                    # generated CSV files, gitignored
```

## 환경

```bash
pip install requests pandas python-dotenv scikit-learn scipy xgboost matplotlib seaborn statsmodels numpy openpyxl jupyter
```

`.env`에 `MAPLE_API_KEY=...`를 설정한다. API key와 `data/` 산출물은 커밋하지 않는다.
