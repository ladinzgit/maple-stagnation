# Maple Parking Detect

메이플스토리 Nexon OpenAPI 데이터로 270~290레벨 본캐 표본에서 성장 정체 기반 주차 후보군을 탐색하는 분석 프로젝트다.

공개 API에서는 주간 보스 수행 기록이나 메소 생산량을 직접 볼 수 없으므로, 본 프로젝트의 결과는 확정 주차 유저 라벨이 아니라 성장/접속 행동 기반의 후보 라벨이다.

## 현재 기준 (2026-06-05)

- 수집 표본: 12개월(2025-06~2026-05) 스냅샷 중 **≥10개월 접속**(`MIN_CHECKPOINT_MONTHS=10`)을 통과한 270~290 본캐 **2,000명** (5계열×400)
- 접속(access) = 표본 통제변인 → 전원 `access_active_months ≥ 10`. 클러스터링 피처 아님.
- H1 분석 윈도우: **최근 6개월(2025-12~2026-05)** delta (주차 = 현재 행동)

## 연구 가설

| 구분 | 가설 문장 | 분석 방법 |
|---|---|---|
| H1 | 최근 6개월간 주요 성장 지표가 낮은 활성 캐릭터들은 일반 성장 캐릭터들과 구분되는 성장 정체 군집으로 식별될 것이다. | K-Means 클러스터링, DBSCAN 보조 검증 |
| H2 | H1에서 도출한 성장 정체 후보군의 비율은 캐릭터 속성인 레벨 구간 및 직업 계열에 따라 불균일하게 분포할 것이다. | 카이제곱 독립성 검정 |
| H3 | H1 후보군은 H1에서 사용하지 않은 관측 피처를 활용해 설명 가능한 단순 rule로 근사할 수 있을 것이다. | RandomForest 중요도, 얕은 Decision Tree, threshold sweep |

현재 로컬 데이터:

| 파일 | 행 수 | 설명 |
|---|---:|---|
| `data/main_characters.csv` | 2,000 | active 본캐 표본 |
| `data/features_monthly.csv` | 2,000 | 12개월 성장/접속 피처 |
| `data/monthly_snapshots_raw.csv` | 24,000 | 월별 raw snapshot |
| `data/hexa_fragments.csv` | 2,000 | HEXA 조각 소비 이력 |
| `data/cluster_labels.csv` | 2,000 | H1 cluster label (6mo·4피처·k=4, 주차 후보 380명) — H2/H3 단일 입력 라벨 |

## H1 결과 (6mo 전환 2026-06-05)

H1은 성장 정체 특성이 강한 cluster가 일반 성장 캐릭터와 구분되는지 확인하는 비지도 분석이다. **최근 6개월(2025-12~2026-05)** 성장 delta(주차=현재 행동)로 클러스터링한다.

최종 선택 피처 (4-시스템 다축):

| Alias | 실제 값 | 시스템 |
|---|---|---|
| `cp_slog` | 전투력 월 Δ signed-log | 전투력 |
| `hexa_avg` | `avg_monthly_delta_hexa` clip≥0 | 헥사 |
| `union_log` | `log1p(Δunion_level)` | 유니온 |
| `auth_log` | `log1p(Δauthentic_symbol)` | 어센틱 심볼 |

결과:

| 항목 | 값 |
|---|---:|
| K-Means k | 4 (elbow; best-sil k=6) |
| Silhouette @k4 | 0.357 |
| 주차 후보 cluster id | 1 |
| H1 주차 후보 수 | **380명 (19.0%)** |
| park(stag≥4) enrich / recall | 4.37x / 83.0% |

클러스터별 인원:

| Cluster | 인원 | 해석 |
|---|---:|---|
| 0 | 882명 | 일반 성장 군집 |
| 1 | 380명 | 성장 정체/주차 후보 (전투력 감소 + 4-시스템 저재투자) |
| 2 | 551명 | 고성장 군집 (헥사 특화) |
| 3 | 187명 | 전투력만 증가, 유니온·심볼 정체 |

DBSCAN(3군집)이 정체 380 ↔ cluster1(355) 강한 정합. 상세 결과: `h1_clustering/RESULT.md`

## H2 결과 (380, 2026-06-06)

H2는 H1 성장 정체 후보군의 비율이 레벨 구간 및 직업 계열에 따라 불균일하게 분포하는지(Chi-Square) 검정한다.

- **레벨 구간 유의**(χ²=38.54, p=4.27e-9, Cramér's V 0.139): 표준화 잔차 기준 `280-285` 집중(+3.18, OR 1.96), `286-290` 희박(-4.47). 270-279는 비율 높으나 n=107로 비유의.
- **직업 계열 무유의**(χ²=6.89, p=0.142, V 0.059): 실질적 집중 없음.
- **결론**: **H2 부분 지지**(레벨 구간 측면 지지, 직업 계열 측면 미지지; 조작적 정의 = H1 정체 후보 라벨 기준). 검정 대상이 후보 라벨이므로 실제 주차 분포로의 일반화는 라벨 타당성 전제.
- **일반화 주의**: 표본은 270-290 본캐 + ≥10/12 접속 + 전투력 ~50M↑ active/capable 캐릭터 → 결론은 표본 설계 내에서만 일반화(저레벨 구간 보수 해석, 레벨 표본 107/1176/717 불균형). 상세: `h2_distribution/RESULTS.md`.

## H3 결과 (380, 2026-06-06)

H3는 H1 후보군을 H1에서 사용하지 않은 관측 피처로 근사하고 접속 게이트를 결합한 설명 가능한 단순 rule을 도출한다(RandomForest 단독, 13피처).

- 학습/평가 target: `is_stagnant_cluster` = **380명(6mo)** 단일 (현재성 라벨 폐기 — `temporal_external_validation` 삭제)
- **누수 가드(H1 네 축) 확장**: union·auth·cp·hexa delta 제외, |corr|>0.85 점검 → drop 0. RF 5-fold ROC-AUC **0.850**.
- 운용 rule = `log1p_recent3_delta_cumexp≤30.84 AND combat_power_latest≤70M AND access_active_months≥2` → Precision 0.500 / Recall 0.289 / **FPR 0.068**(≤0.10) → **조작적 정의 기준 제한적 지지**(H1 정체 후보 라벨 기준).
- **해석 범위**: RF ROC-AUC만 5-fold OOF; sweep·rule 평가는 in-sample(held-out 없음) → "H1 후보 라벨 근사도"(실제 주차 탐지력은 후속 외부검증 대상). 상세: `h3_rule/RESULT.md`.

## 실행 순서

```bash
python scripts/collect_main_characters.py
python scripts/collect_features.py --refresh-raw
python scripts/collect_hexa_fragments.py
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
python h2_distribution/run_analysis.py
jupyter nbconvert --to notebook --execute --inplace h2_distribution/h2_distribution.ipynb
jupyter nbconvert --to notebook --execute --inplace h3_rule/h3_rule.ipynb
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
