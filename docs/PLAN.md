# 수집 및 분석 계획 (현행 2026-06-05)

> 현행 상태 요약. 상세는 `CLAUDE.md` 및 각 가설 폴더 `README.md`/`RESULT.md` 참조.

## 수집

`scripts/collect_main_characters.py` → `collect_features.py --refresh-raw` → `collect_hexa_fragments.py`.

- 표본: 12개월(2025-06~2026-05) 스냅샷 중 **≥10개월 접속**(`MIN_CHECKPOINT_MONTHS=10`)을 통과한 270~290 본캐 **2,000명** (5계열×400)
- 접속(access) = 표본 통제변인 (클러스터링 피처 아님)
- 산출물: `main_characters.csv`(2,000) · `features_monthly.csv`(2,000, 12mo 집계) · `monthly_snapshots_raw.csv`(24,000) · `hexa_fragments.csv`(2,000)

## H1 — Done (6mo 전환 2026-06-05)

```bash
jupyter nbconvert --to notebook --execute --inplace h1_clustering/feature_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace h1_clustering/h1_clustering.ipynb
```

- 가설: 최근 6개월간 주요 성장 지표가 낮은 활성 캐릭터들은 일반 성장 캐릭터들과 구분되는 성장 정체 군집으로 식별될 것이다.
- 분석 윈도우: **최근 6개월(2025-12~2026-05)** delta = features_monthly의 `recent6_delta_*` 컬럼 (주차=현재 행동; raw 6mo 재산출과 동등·features_monthly 단독 재현)
- 채택 4피처: `cp_slog`, `hexa_avg`, `union_log`, `auth_log` (전투력·헥사·유니온·어센틱 심볼)
- K-Means **k=4**(elbow; best-sil k=6), silhouette@k4 0.357, 주차 후보 **380명(19.0%)**, recall 83.0%
- DBSCAN 정합(380↔355)으로 같은 피처 공간의 보조 구조 확인. (시간분할 외부검증 노트북은 폐기·삭제.)
- 산출물: `optimal_feature_set.json`, `data/cluster_labels.csv`(380명, H2/H3 단일 입력 라벨)

## H2 — Done (380, 2026-06-06)

```bash
python h2_distribution/run_analysis.py
```

- 가설: H1에서 도출한 성장 정체 후보군의 비율은 캐릭터 속성인 레벨 구간 및 직업 계열에 따라 불균일하게 분포할 것이다.
- 입력: `is_stagnant_cluster`(380) **단일 라벨** × 레벨구간/직업계열 → Chi-Square (현재성 보조 라벨 폐기)
- 결과: **H2 부분 지지**(조작적 정의 = H1 정체 후보 라벨). 레벨 구간은 유의(p=4.27e-9, Cramer's V 0.139; 280-285 집중 잔차 +3.18, 286-290 희박 -4.47), 직업계열은 **무유의**(p=0.142, V 0.059). 후보 라벨 기준이므로 실제 주차 분포 일반화는 라벨 타당성 전제.
- 일반화 주의: 표본 = active/capable 본캐(270-290·≥10 접속·전투력 ~50M↑) → 표본 설계 내 일반화, 저레벨(270-279, n=107) 보수 해석.
- 과거 결과(412/471/277 라벨)는 폐기.

## H3 — Done (380, 2026-06-06)

- target: `is_stagnant_cluster`(380) **단일** (현재성 라벨 폐기)
- 가설: H1 후보군은 H1에서 사용하지 않은 관측 피처를 활용해 설명 가능한 단순 rule로 근사할 수 있을 것이다.
- RandomForest 단독(13피처) → permutation importance → depth-2 DecisionTree rule → 접속 게이트 결합
- 누수 가드: union·auth가 H1 채택축 → **cp·hexa·union·auth delta + 파생 전부 제외**, H1 네 축 |corr|>0.85 점검 → drop 0
- 결과: ROC-AUC 0.850(5-fold OOF), rule `cum3≤30.84 ∧ cpl≤70M ∧ access≥2` → Precision 0.500 / Recall 0.289 / **FPR 0.068**(≤0.10) → **조작적 정의 기준 제한적 지지**(H1 정체 후보 라벨 기준).
- 해석 범위: sweep·rule 평가는 in-sample(held-out 없음) → "H1 후보 라벨 근사도"(실제 주차 탐지력은 후속 외부검증 대상).

## 해석 원칙

- 확정 주차 유저 ground truth 없음 → 모든 라벨은 후보.
- H1 = 성장 정체 후보군 탐색, H2 = 분포 검정, H3 = 운영 rule 근사(실제 주차 검증 아님).
